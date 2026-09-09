"""The model call, and nothing else — where a window of audio and a list of words become times.

Everything that decides *what* to align and *where* is in `track.py`; this file is the part that
needs weights, and it is deliberately thin so that the interesting half stays testable without a GPU.

**Forced alignment, not transcription, and the difference is the whole design.** Whisper is asked
"what was sung" and answers with words that may be wrong — on a vocal under a full mix it often is,
and its word timings are a DTW over attention, which is an estimate of an estimate. Here the words
are already known: they are the same lyric that went to Siren. So the only question left is *where*,
and `forced_align` answers it with an exact Viterbi path through a CTC lattice. It cannot invent a
word, it cannot drop one, and it hands back a per-token probability that says how much to believe it.

**Why this does not repeat the memory failure the author already hit.** `wav2vec2` is a transformer
over 20 ms frames, so encoding a three-minute song in one go is ~9 000 frames of quadratic
attention. It is never asked to. `track.py` cuts the song into blocks bounded by the plan, and each
block is encoded on its own — 15-45 s, i.e. 750-2 250 frames. Peak memory is set by the longest
SECTION and stops depending on the song's length entirely. A block that somehow exceeds the cap is
split rather than allowed to grow.

Two consequences of using a per-block window that are worth stating, because they are the reason it
beats one pass over the whole song:

* the Viterbi is better conditioned — a chorus sung four times cannot have its second occurrence
  matched to its fourth, because the fourth is not inside the window;
* the `*` star token only has to absorb a few seconds of neighbouring music instead of the whole
  arrangement. That token is what makes an instrumental lead-in free: the model is told "there is
  audio here the transcript does not cover" rather than being forced to hear a word in a guitar.

The price is that windows overlap by their slack, so ~20% of the song is encoded twice. That is
seconds of GPU time and it buys both of the above.
"""
import os

#: Under `ComfyUI/models/`. Its own folder rather than `stt/whisper/`, which belongs to another
#: pack and holds a different kind of model — a shared folder is how two packs start deleting each
#: other's downloads.
MODEL_SUBDIR = os.path.join("stt", "echo")

#: `MMS_FA` — wav2vec2-large fine-tuned for alignment on 1130 languages. One file, from Meta's own
#: bucket, via `torch.hub`. Named here so the node can say what it is about to fetch BEFORE it
#: fetches 1.2 GB on somebody's metered connection.
MODEL_NAME = "MMS_FA (ctc_alignment_mling_uroman)"
MODEL_BYTES = 1_270_000_000

#: The longest window handed to the encoder in one piece, in seconds. 45 s is ~2 250 frames, whose
#: attention is a few hundred MB and comfortable on any card that can run a diffusion model. Raise
#: it and the quadratic term starts to bite; lower it and long verses get split for no reason.
MAX_WINDOW = 45.0

#: One emission frame. Fixed by the convolutional stack (strides 5·2·2·2·2·2·2 = 320 samples at
#: 16 kHz), so a word can never be located more precisely than this and there is no point pretending
#: otherwise in a report.
FRAME_SECONDS = 0.02

_CACHE = {}


def download_root(create=False):
    """Where the weights live. Under ComfyUI's own `models/` so they are visible and deletable,
    rather than in `~/.cache/torch`, where nobody finds 1.2 GB they no longer want.

    Creates nothing unless asked. A getter that makes a directory leaves one behind every time
    anything merely *asks* the question — including a test suite that never intended to download.
    """
    try:
        import folder_paths
        base = folder_paths.models_dir
    except Exception:                                  # outside ComfyUI: only ever reached by tests
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    path = os.path.join(base, MODEL_SUBDIR)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def is_downloaded():
    """Whether the first run will cost a download. `torch.hub` names the file after the URL."""
    root = download_root()
    return any(n.endswith(".pt") for n in os.listdir(root)) if os.path.isdir(root) else False


def _device(prefer=None):
    if prefer not in (None, "", "auto"):
        return prefer
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


def load(device=None, half=True):
    """`(model, tokenizer, aligner)`, cached for the process.

    Kept alive between runs on purpose: 1.2 GB of weights takes several seconds to read off disk and
    a music video is aligned many times while its subtitles are being styled. `unload()` is the
    other half of that bargain and the node offers it as a switch.
    """
    import torch
    from torchaudio.pipelines import MMS_FA as bundle

    device = _device(device)
    dtype = torch.float16 if (half and str(device).startswith("cuda")) else torch.float32
    key = (str(device), str(dtype))
    if key in _CACHE:
        return _CACHE[key]

    model = bundle.get_model(with_star=True,
                             dl_kwargs={"model_dir": download_root(create=True), "progress": True})
    model = model.to(device=device, dtype=dtype).eval()
    _CACHE.clear()                      # one device at a time; two copies is 2.4 GB for no reason
    _CACHE[key] = (model, bundle.get_tokenizer(), bundle.get_aligner())
    return _CACHE[key]


def unload():
    """Drop the weights and hand the VRAM back."""
    _CACHE.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def mono_16k(audio):
    """A ComfyUI `AUDIO` -> `(1, n)` float32 at 16 kHz, and the length in seconds.

    Channels are summed to mono rather than one being picked: a lead vocal is almost always centred,
    so the sum reinforces it while the wide-panned guitars partly cancel. Picking the left channel
    would instead take whatever happens to be panned there.
    """
    import torch
    import torchaudio.functional as AF

    wave = audio["waveform"]
    wave = wave[0] if wave.dim() == 3 else wave
    wave = wave.detach().to(torch.float32).cpu()
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    sr = int(audio["sample_rate"])
    total = wave.shape[-1] / float(sr)
    if sr != 16000:
        wave = AF.resample(wave, sr, 16000)
    return wave, total


def _split_window(block, max_window):
    """A block too long for one encode -> several, split by how much text is in each.

    Only reached by a section longer than three quarters of a minute, which in practice means a plan
    row that swallowed several sections. Splitting by character count rather than by line keeps the
    sub-windows proportional to how long the words take to sing, which is the best guess available
    without having aligned anything yet.
    """
    import math

    lo, hi = float(block["window"][0]), float(block["window"][1])
    span = hi - lo
    parts = max(1, int(math.ceil(span / max(1.0, float(max_window)))))
    if parts == 1 or len(block["lines"]) <= 1:
        return [(block["lines"], (lo, hi))]

    sizes = [max(1, sum(len(w["text"]) for w in ln["words"])) for ln in block["lines"]]
    total = float(sum(sizes))

    groups, edges, carried, used = [], [], [], 0.0
    for ln, size in zip(block["lines"], sizes):
        carried.append(ln)
        used += size
        # Close a part once its share of the text is inside it. The LAST part is never closed here
        # and simply takes what is left, so a rounding error cannot strand a line outside every
        # window — a line nobody searches for is a line with no timing at all.
        if len(groups) < parts - 1 and used / total >= (len(groups) + 1) / parts:
            groups.append(carried)
            edges.append(lo + span * (used / total))
            carried = []
    if carried:
        groups.append(carried)

    # A second of overlap on each seam: the split point is a guess from character counts, and a word
    # sitting right on it must be inside SOME window or it is never found.
    out, prev = [], lo
    for i, group in enumerate(groups):
        end = hi if i >= len(edges) else edges[i]
        out.append((group, (max(lo, prev - 1.0), min(hi, end + 1.0))))
        prev = end
    return out


def _score(spans):
    """One word's confidence: its tokens' probabilities, weighted by how long each was held.

    Length-weighted rather than a plain mean, because a long vowel the model is sure about should
    not be outvoted by a consonant that lasted one frame.
    """
    held = sum(len(s) for s in spans)
    return (sum(s.score * len(s) for s in spans) / held) if held else 0.0


#: How close a found span may sit to its window's edge before the WINDOW is suspected of being the
#: thing that put it there. A section that really is at the edge and one that was crammed against it
#: look identical from inside; the difference only shows when the window is opened and the words
#: move somewhere better.
PIN_EDGE = 0.35

#: How much wider the second attempt is, on each side. Proportional, because the error in a plan is
#: proportional to the section's own length.
PIN_WIDEN = 0.6
PIN_WIDEN_MIN = 5.0


def _align_window(model, tokenizer, aligner, wave, lo, hi, lines, star, label, notes):
    """One (lines, window) -> `[(word, start, end, score)]`, or `None`.

    Nothing is written onto the words here, on purpose: a placement has to be comparable against a
    second attempt before either of them is kept.
    """
    import torch

    n = wave.shape[-1]
    a, b = max(0, int(round(lo * 16000))), min(n, int(round(hi * 16000)))
    if b - a < 1600:                                       # under a tenth of a second
        notes.append(f"{label}: the search window is empty — the plan puts this section outside "
                     f"the audio. Its words were not aligned.")
        return None

    # `key` is the romanized spelling; a word without one is never sent, and its slot is kept so the
    # spans that come back can be handed to the right words afterwards.
    targets, slots = [], []
    for ln in lines:
        for w in ln["words"]:
            if w["key"]:
                targets.append(w["key"])
                slots.append(w)
    if not targets:
        return None

    words = (["*"] + targets + ["*"]) if star else list(targets)
    try:
        tokens = tokenizer(words)
    except KeyError as e:
        notes.append(f"{label}: {e} is not a character the aligner knows — the romanizer let "
                     f"something through. Section skipped.")
        return None

    chunk = wave[:, a:b].to(device=next(model.parameters()).device,
                            dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        emission, _ = model(chunk)
    emission = emission[0].float().cpu()                   # (frames, tokens), already log-domain
    del chunk

    try:
        spans = aligner(emission, tokens)
    except Exception as e:
        notes.append(f"{label}: the aligner could not place these words ({e}). Its lines keep the "
                     f"plan's timing instead.")
        del emission
        return None
    if star:
        spans = spans[1:-1]

    # Frames back to seconds. The ratio is measured rather than assumed: the convolutional stack
    # drops a few samples at the end, so `n / 320` and the real frame count differ by one or two and
    # a fixed 0.02 would drift by that much over a long window.
    ratio = (b - a) / max(1, emission.shape[0]) / 16000.0
    del emission

    placed = []
    for w, word_spans in zip(slots, spans):
        if word_spans:
            placed.append((w, lo + word_spans[0].start * ratio, lo + word_spans[-1].end * ratio,
                           float(_score(word_spans))))
    return placed or None


def _confidence(placed):
    return sum(p[3] for p in placed) / len(placed) if placed else 0.0


def _pinned(placed, lo, hi):
    """Whether the words ended up flat against an edge of the window they were searched in.

    The failure this exists for: the plan says a section is at 1:20, the singer came in at 1:35, and
    the window stopped at 1:28. Forced alignment must place every word it is given, so it crams them
    into the last seconds it is allowed to use — the subtitles then appear EARLY, on top of whoever
    is really singing, and nothing at all appears where the words belong. From inside one window
    that is indistinguishable from a correct answer, which is why it is caught by geometry.
    """
    if not placed:
        return False
    return (min(p[1] for p in placed) - lo) < PIN_EDGE or (hi - max(p[2] for p in placed)) < PIN_EDGE


def align_job(job, audio, device=None, half=True, max_window=MAX_WINDOW, star=True,
              retry_pinned=True, anchor=True, slack=None, progress=None):
    """Fill `start` / `end` / `score` on every findable word of `job`, in place. Returns notes.

    Words with no romanized form are left untouched — `track.finish` carries them onto their
    neighbours' timing, which is a decision about subtitles rather than about audio and belongs
    there.

    **Blocks are aligned in order, and each one teaches the next where to look.** With `anchor` on,
    a block's window is its planned span shifted by the drift the last trusted block measured, and
    floored at where that block's words actually stopped — see `track.anchor` for why the floor is
    the half that matters. A block whose own confidence comes back under `track.ANCHOR_TRUST`
    teaches nothing: it neither shifts nor floors the next one, so one bad placement cannot walk
    the whole rest of the song off the music.
    """
    from . import track as TR

    notes = []
    wave, total = mono_16k(audio)
    model, tokenizer, aligner = load(device, half)
    slack = TR.SLACK if slack is None else float(slack)

    blocks = [b for b in (job.get("blocks") or []) if b["lines"]]
    if not blocks:
        return notes

    prev_end, drift, floors = None, 0.0, 0
    for done, block in enumerate(blocks, 1):
        window = (TR.anchor(block, prev_end, drift, slack, total) if anchor
                  else tuple(block["window"]))
        floor = window[0] if (anchor and prev_end is not None) else None
        placed_all, conf, first, last = [], [], None, None

        # Still split for the encoder's sake — the cap is about attention memory and applies to an
        # anchored window exactly as it did to a planned one.
        parts = _split_window({**block, "window": window}, float(max_window))
        if len(parts) > 1 and done == 1:
            notes.append(f"a section was longer than the {max_window:.0f} s encode window and was "
                         f"split into several. That is a plan row covering more than one section — "
                         f"the words are still placed, just searched for in smaller neighbourhoods.")

        for lines, (lo, hi) in parts:
            label = block["label"]
            placed = _align_window(model, tokenizer, aligner, wave, lo, hi, lines, star, label,
                                   notes)

            if placed and retry_pinned and _pinned(placed, lo, hi):
                # Open the window and ask again. The wider answer is kept only if it is MORE
                # confident: a section that genuinely starts at its window's edge comes back the
                # same and loses nothing, while a crammed one was scoring badly precisely because
                # the words are not there. One retry, never a search.
                grow = max(PIN_WIDEN_MIN, (hi - lo) * PIN_WIDEN)
                # The floor is NOT widened past. Letting the retry dig below where the previous
                # section stopped singing is exactly the overlap anchoring exists to prevent, and a
                # section pinned against it is telling you something rather than needing rescue.
                lo2 = max(0.0, lo - grow) if floor is None else max(floor, lo - grow)
                hi2 = min(total, hi + grow)
                if (lo2, hi2) != (lo, hi):
                    again = _align_window(model, tokenizer, aligner, wave, lo2, hi2, lines, star,
                                          label, notes)
                    if again and _confidence(again) > _confidence(placed):
                        moved = min(p[1] for p in again) - min(p[1] for p in placed)
                        notes.append(
                            f"{label}: its words sat flat against the edge of its window, so it "
                            f"was searched again {grow:.0f} s wider and found {moved:+.1f} s "
                            f"away, confidence {_confidence(placed):.2f} -> "
                            f"{_confidence(again):.2f}.")
                        placed, lo = again, lo2
                    elif _confidence(placed) < TR.ANCHOR_TRUST:
                        if floor is not None and (min(p[1] for p in placed) - floor) < PIN_EDGE:
                            floors += 1
                            notes.append(
                                f"{label}: its words start exactly where the previous section "
                                f"stopped singing, at only {_confidence(placed):.2f} confidence. "
                                f"That normally means this section was NOT sung — the aligner has "
                                f"to put the words somewhere and this is the earliest it is "
                                f"allowed. Do not trust its subtitles.")
                        else:
                            notes.append(
                                f"{label}: its words sit against the edge of the search window at "
                                f"only {_confidence(placed):.2f} confidence, and a wider search "
                                f"did no better. They are probably not sung where the plan says.")

            for entry in placed or []:
                placed_all.append(entry)
                conf.append(entry[3])
                first = entry[1] if first is None else min(first, entry[1])
                last = entry[2] if last is None else max(last, entry[2])

        for w, start, end, score in placed_all:
            w["start"], w["end"], w["score"] = start, end, score

        # What this block teaches the next one — only if it is worth believing.
        if anchor and conf and (sum(conf) / len(conf)) >= TR.ANCHOR_TRUST:
            prev_end, drift = last, first - float(block["start"])

        if progress is not None:
            try:
                progress(done, len(blocks))
            except Exception:
                pass

    if floors:
        notes.append(f"{floors} section(s) were pinned against the previous one's last word. Those "
                     f"are the sections to check first: it is what an unsung section looks like.")
    return notes
