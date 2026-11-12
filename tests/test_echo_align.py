"""The parts of Echo's aligner that do not need weights: the encode-window cap, the word score and
the loop that decides where each section is searched for.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "context", "echo", "siren", "timer", "util")
load_module("kn.util.anytype", "util/anytype.py")
load_module("kn.context.character_card", "context/character_card.py")
load_module("kn.timer.timer_nodes", "timer/timer_nodes.py")
load_module("kn.siren.cast", "siren/cast.py")
load_module("kn.siren.score", "siren/score.py")
load_module("kn.echo.translit", "echo/translit.py")
TR = load_module("kn.echo.track", "echo/track.py")
A = load_module("kn.echo.align", "echo/align.py")

check = Checker()


# ------------------------------------------------- a long window is encoded in pieces, searched whole
# The words are never divided between pieces of a window — only the audio is, for the encoder's
# memory. So what has to hold is that the pieces come back as ONE gapless run of frames.
import torch  # noqa: E402


class Ramp(torch.nn.Module):
    """Emits, for every frame, the first sample of that frame — on a wave that counts its frames."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(1))
        self.seen = []

    def forward(self, chunk):
        self.seen.append(chunk.shape[-1] / 16000.0)
        return chunk[:, ::A.FRAME_SAMPLES].unsqueeze(-1), None


ramp = Ramp()
frames = 7000                                                       # 140 s
wave = torch.arange(frames).repeat_interleave(A.FRAME_SAMPLES).float().unsqueeze(0)
lo_s, hi_s = 100 * A.FRAME_SAMPLES, 6900 * A.FRAME_SAMPLES
em = A._emission(ramp, wave, lo_s, hi_s)
check("a 136 s window is encoded in several pieces", len(ramp.seen) == 4, ramp.seen)
check("no piece exceeds the cap by more than its context",
      max(ramp.seen) <= A.MAX_WINDOW + 2 * A.ENCODE_CONTEXT / 16000.0, ramp.seen)
check("the pieces come back as one gapless run of frames",
      em.shape[0] == 6800 and torch.equal(em[:, 0], torch.arange(100, 6900).float()),
      (em.shape, em[:3, 0], em[-3:, 0]))
ramp.seen.clear()
check("a window inside the cap is one piece, untouched",
      A._emission(ramp, wave, 0, 1000 * A.FRAME_SAMPLES).shape[0] == 1000 and len(ramp.seen) == 1,
      ramp.seen)


# -------------------------------------------------------------------------- the word confidence
class Span:
    """A stand-in for `torchaudio.functional.TokenSpan`."""

    def __init__(self, start, end, score):
        self.start, self.end, self.score = start, end, score

    def __len__(self):
        return self.end - self.start


check("one token's score is its own", abs(A._score([Span(0, 4, 0.8)]) - 0.8) < 1e-9)
# Length-weighted, so a held vowel the model is sure of is not outvoted by a one-frame consonant.
mixed = A._score([Span(0, 9, 0.9), Span(9, 10, 0.1)])
check("the score is weighted by how long each token was held", abs(mixed - 0.82) < 1e-9, mixed)
check("a plain mean would have said 0.50 — it does not", abs(mixed - 0.5) > 0.3)
check("no tokens is 0.0 rather than a division by zero", A._score([]) == 0.0)


# ------------------------------------------------------ the window is caught placing the words
# The failure the author hit: the plan says a section is at 1:20, the singer came in at 1:35, the
# window stopped at 1:28. Forced alignment MUST place every word it is given, so it crams them
# against the last second it is allowed — early, over whoever is really singing, with silence where
# the words belong. From inside the window that is indistinguishable from a correct answer, so it is
# caught by geometry: words flat against an edge are words the window chose the position of.
def placed(pairs):
    return [({}, a, b, 0.7) for a, b in pairs]


check("words hard against the low edge are pinned",
      A._pinned(placed([(20.05, 21.0), (21.0, 22.0)]), 20.0, 40.0))
check("words hard against the high edge are pinned",
      A._pinned(placed([(38.0, 39.0), (39.0, 39.95)]), 20.0, 40.0))
check("words sitting comfortably inside are not",
      not A._pinned(placed([(25.0, 26.0), (30.0, 31.0)]), 20.0, 40.0))
check("the edge tolerance is a fraction of a second, not zero",
      A._pinned(placed([(20.3, 21.0)]), 20.0, 40.0) and
      not A._pinned(placed([(20.5, 21.0)]), 20.0, 40.0), A.PIN_EDGE)
check("nothing placed cannot be pinned", not A._pinned([], 0.0, 10.0))
check("the retry widens by a real amount",
      max(A.PIN_WIDEN_MIN, 30.0 * A.PIN_WIDEN) >= 18.0,
      max(A.PIN_WIDEN_MIN, 30.0 * A.PIN_WIDEN))

check("confidence is the mean over the placed words",
      abs(A._confidence(placed([(0, 1), (1, 2)])) - 0.7) < 1e-9)
check("no words is 0.0 confidence, not an error", A._confidence([]) == 0.0)
# The acceptance rule for a retry: a wider window is kept only when it is MORE sure. A section that
# genuinely starts at its window's edge comes back identical and loses nothing.
better = [({}, 0, 1, 0.9)]
check("a more confident retry wins", A._confidence(better) > A._confidence(placed([(0, 1)])))


# ------------------------------------------------------------------- where the weights will live
root = A.download_root()
check("the model root is under ComfyUI's own models folder, not a hidden cache",
      root.replace("\\", "/").endswith("stt/echo"), root)
check("...and it is its own folder, not shared with another pack's whisper",
      not root.replace("\\", "/").endswith("stt/whisper"))
check("is_downloaded answers without raising on an empty folder",
      A.is_downloaded() in (True, False), A.is_downloaded())
check("the download is named and sized before it is fetched",
      "MMS_FA" in A.MODEL_NAME and 1e9 < A.MODEL_BYTES < 2e9)

# =============================================== the loop: each section teaches the next one where
# The whole orchestration over a STUB aligner that behaves the way the real one has to: forced
# alignment must place every word it is given INSIDE the window it is given, so a window that does
# not contain the words crams them against an edge and the confidence collapses. That single
# property is what the anchoring is built against, and it is enough to test the loop without a GPU.
TRUTH = {                                       # the author's real take, section -> where it is
    "Verse 1": (14.32, 34.46), "Pre-Chorus": (38.61, 53.96), "Chorus 1": (55.95, 80.34),
    "Verse 2": (81.66, 100.75), "Bridge": (103.43, 116.66), "Chorus 2": (117.21, 146.10),
    # Never sung. Without a floor its words land confidently on top of Chorus 2, which is exactly
    # what happened on the real render: subtitles over a chorus that was singing something else.
    "Outro": (118.02, 125.77),
}
PLANNED = {"Verse 1": (12.8, 32.0), "Pre-Chorus": (32.0, 44.8), "Chorus 1": (44.8, 64.0),
           "Verse 2": (64.0, 89.6), "Bridge": (89.6, 102.4), "Chorus 2": (108.8, 128.0),
           "Outro": (128.0, 134.4)}
SONG = 179.2

A.mono_16k = lambda audio: (None, SONG)
A.load = lambda device=None, half=True: (None, None, None)
SEEN = []


def stub_window(model, tokenizer, aligner, wave, lo, hi, lines, star, label, notes):
    SEEN.append((label, round(lo, 2), round(hi, 2)))
    a, b = TRUTH[label]
    # Clamped into the window, because that is the one thing the real aligner cannot refuse to do.
    a2 = max(lo, min(a, hi))
    b2 = min(hi, max(b, a2 + 0.5))
    conf = max(0.0, 1.0 - (abs(a2 - a) + abs(b2 - b)) / 12.0)
    words = [w for ln in lines for w in ln["words"] if w["key"]]
    step = (b2 - a2) / max(1, len(words))
    return [(w, a2 + i * step, a2 + (i + 1) * step, conf) for i, w in enumerate(words)] or None


A._align_window = stub_window


def a_job():
    blocks = []
    for label, (p0, p1) in PLANNED.items():
        pad = max(TR.SLACK_MIN, (p1 - p0) * TR.SLACK)
        blocks.append({"label": label, "voice": "Nina", "start": p0, "end": p1,
                       "window": (max(0.0, p0 - pad), min(SONG, p1 + pad)),
                       "lines": [{"text": "x y z", "voice": "Nina", "label": label,
                                  "words": [{"text": w, "key": w} for w in ("aa", "bb", "cc")]}]})
    return {"total": SONG, "blocks": blocks, "language": "uk"}


def outcome(**kw):
    SEEN.clear()
    job = a_job()
    notes = A.align_job(job, {"waveform": None, "sample_rate": 16000}, **kw)
    got = {}
    for b in job["blocks"]:
        ws = [w for ln in b["lines"] for w in ln["words"] if w.get("start") is not None]
        if ws:
            got[b["label"]] = (ws[0]["start"], ws[-1]["end"],
                               sum(w["score"] for w in ws) / len(ws))
    return got, notes


anchored, anchored_notes = outcome(anchor=True, retry_pinned=False)
plain, _ = outcome(anchor=False, retry_pinned=False)

sung = [k for k in TRUTH if k != "Outro"]


def whole(got, k):
    """BOTH ends, not just the start. Every un-anchored window here contains its section's first
    word — what it does not contain is the last one, so the rest is crammed against the edge and
    the confidence collapses. Measuring the start alone says everything is fine."""
    return abs(got[k][0] - TRUTH[k][0]) < 1.0 and abs(got[k][1] - TRUTH[k][1]) < 1.0


good_a = [k for k in sung if whole(anchored, k)]
good_p = [k for k in sung if whole(plain, k)]
check("anchored, every sung section lands where it really is", len(good_a) == 6, good_a)
check("...where reading the plan alone gets one of six", len(good_p) == 1, good_p)
check("...and that one is Verse 1, exactly as on the real take", good_p == ["Verse 1"], good_p)
check("anchored confidence is high across the board",
      min(anchored[k][2] for k in sung) > 0.9, {k: round(anchored[k][2], 2) for k in sung})
check("...where un-anchored it collapses, which is what the real report showed",
      max(plain[k][2] for k in sung if k != "Verse 1") < 0.8,
      {k: round(plain[k][2], 2) for k in sung})

# The headline: the Outro was never sung, and the floor turns a confident wrong answer into a
# reported non-answer. Without the floor its words sit on top of a chorus that IS singing.
check("un-anchored, the Outro is laid on top of the chorus that is really singing",
      plain["Outro"][0] < plain["Chorus 2"][1], (plain["Outro"], plain["Chorus 2"]))
check("anchored, it is searched AFTER the chorus stopped instead",
      anchored["Outro"][0] >= TRUTH["Chorus 2"][1] - TR.ANCHOR_OVERLAP - 0.01, anchored["Outro"])
check("...so it no longer collides with anything",
      anchored["Outro"][0] >= anchored["Chorus 2"][1] - TR.ANCHOR_OVERLAP - 0.01,
      (anchored["Outro"], anchored["Chorus 2"]))
check("...and its confidence collapses, which is the signal that it was not sung",
      anchored["Outro"][2] < 0.4, round(anchored["Outro"][2], 2))
check("the collapse is SAID, not left to be noticed",
      any("previous section" in n and "Outro" in n for n in
          A.align_job(a_job(), {"waveform": None, "sample_rate": 16000}, anchor=True)),
      [n for n in anchored_notes])

# A section nobody could place must not poison the rest: it teaches the next one nothing.
BAD = dict(TRUTH)
BAD["Chorus 1"] = (5.0, 9.0)                   # nowhere near its window; comes back clamped and weak
TRUTH_BACKUP = dict(TRUTH)
TRUTH.update(BAD)
after_bad, _ = outcome(anchor=True, retry_pinned=False)
TRUTH.clear()
TRUTH.update(TRUTH_BACKUP)
check("a section that failed does not drag the ones after it off the music",
      abs(after_bad["Verse 2"][0] - TRUTH["Verse 2"][0]) < 1.5, after_bad["Verse 2"])
check("...which is what the confidence gate is for", TR.ANCHOR_TRUST > 0.0, TR.ANCHOR_TRUST)

# Windows must still be bounded — anchoring may not quietly grow them into the whole song.
widest = max(hi - lo for _, lo, hi in SEEN)
check("no anchored window approaches the length of the song", widest < SONG * 0.35,
      f"{widest:.1f} s of {SONG:.0f}")

# ================================================ a line cannot start after the next section did
# The same track's LAST chorus: its lines 4-5 are growled and the model cannot hear them (0.04 where
# they really are), and the clean outro at 240 s is inside the same window — so they were laid on
# the outro at 0.27, two voices' words in one shot and nothing over the growl itself.
GROWL = {"g4": 226.4, "g5": 232.8}


def stub_growl(model, tokenizer, aligner, wave, lo, hi, lines, star, label, notes):
    SEEN.append((label, round(lo, 2), round(hi, 2)))
    out = []
    for ln in lines:
        for w in ln["words"]:
            k = w["key"]
            if k in GROWL:
                # A clean voice inside the window beats the growl; without one, the growl is found.
                out.append((w, 247.3, 249.8, 0.27) if hi > 247.0 else (w, GROWL[k], GROWL[k] + 2.5, 0.05))
            else:
                at = {"c1": 217.0, "c2": 220.2, "c3": 223.1, "o1": 240.1, "o2": 243.4}[k]
                out.append((w, at, at + 2.5, 0.8))
    return out


def one_line(key, label):
    return {"text": "x" * 33, "voice": "", "label": label, "words": [{"text": "x" * 33, "key": key}]}


A._align_window = stub_growl
last_chorus = {"label": "Chorus", "voice": "", "start": 200.4, "end": 237.4, "window": (190.0, 256.0),
               "lines": [one_line(k, "Chorus") for k in ("c1", "c2", "c3", "g4", "g5")]}
outro = {"label": "Outro", "voice": "", "start": 237.4, "end": 266.4, "window": (236.0, 267.0),
         "lines": [one_line(k, "Outro") for k in ("o1", "o2")]}
SEEN.clear()
ceiling_notes = A.align_job({"total": 267.0, "blocks": [last_chorus, outro], "language": "uk"},
                            {"waveform": None, "sample_rate": 16000}, anchor=False,
                            retry_pinned=False)
at = {ln["words"][0]["key"]: ln["words"][0]["start"] for ln in last_chorus["lines"] + outro["lines"]}
check("lines laid over the next section are searched again before it",
      all(abs(at[k] - GROWL[k]) < 0.5 for k in GROWL), at)
check("...in a window that stops where that section starts", SEEN[-1][2] <= 240.1 + TR.ANCHOR_OVERLAP,
      SEEN[-1])
check("...and starts at the last line that stayed", SEEN[-1][1] == 223.1, SEEN[-1])
check("the lines that were fine are not touched", (at["c1"], at["c3"], at["o1"]) == (217.0, 223.1, 240.1),
      at)
check("it is said, with the confidence that explains why",
      any("Outro" in n and "0.05" in n for n in ceiling_notes), ceiling_notes)

# ======================================== a section sung early in an interlude nobody gave a length
# Same track, 2:21-2:41: with no plan the `[Interlude]` is a row of no length, so the bridge was
# planned at 162 s and searched from 155 s. It is sung from 145 s — the subtitles stayed dark while
# it was sung and then showed its words over what came next.
A._align_window = stub_window
TRUTH_BACKUP = dict(TRUTH)
TRUTH.clear()
TRUTH.update({"Chorus": (121.0, 141.3), "Bridge": (145.0, 152.0)})
early = {"total": 267.0, "language": "uk", "blocks": [
    {"label": label, "voice": "", "start": p0, "end": p1, "window": (p0 - 3.0, p1 + 3.0),
     "lines": [one_line("aa", label)]}
    for label, (p0, p1) in (("Chorus", (124.6, 161.5)), ("Bridge", (162.2, 175.4)))]}
SEEN.clear()
A.align_job(early, {"waveform": None, "sample_rate": 16000}, anchor=True, retry_pinned=False)
bridge_at = early["blocks"][1]["lines"][0]["words"][0]["start"]
TRUTH.clear()
TRUTH.update(TRUTH_BACKUP)
check("a section sung inside the interlude before it is found there", abs(bridge_at - 145.0) < 0.5,
      (bridge_at, SEEN))
check("...because its window reaches back to the previous section's last word",
      abs(SEEN[-1][1] - (141.3 - TR.ANCHOR_OVERLAP)) < 0.01, SEEN[-1])
check("...and is never wider than one encode", SEEN[-1][2] - SEEN[-1][1] <= A.MAX_WINDOW, SEEN[-1])

# ================================================= a section after a solo longer than one encode
# Same track, isolated vocal: the bridge ends at 172 s, the solo runs to 216 s and the last chorus
# is 217-237 s. Its window was 61 s, so its first lines were looked for in the first half of it —
# all solo — and their subtitles came up 24 s early over silence.
A._align_window = stub_window
TRUTH_BACKUP = dict(TRUTH)
TRUTH.clear()
TRUTH.update({"Bridge": (161.0, 172.0), "Chorus": (217.0, 237.0)})
solo = {"total": 267.0, "language": "uk", "blocks": [
    {"label": label, "voice": "", "start": p0, "end": p1, "window": (p0 - 3.0, p1 + 3.0),
     "lines": [one_line(k, label) for k in keys]}
    for label, (p0, p1), keys in (("Bridge", (162.0, 175.0), ("aa",)),
                                  ("Chorus", (200.4, 237.4), ("aa", "bb", "cc", "dd", "ee")))]}
SEEN.clear()
A.mono_16k = lambda audio: (None, 267.0)
A.align_job(solo, {"waveform": None, "sample_rate": 16000}, anchor=True, retry_pinned=False)
chorus_at = solo["blocks"][1]["lines"][0]["words"][0]["start"]
TRUTH.clear()
TRUTH.update(TRUTH_BACKUP)
check("a window longer than one encode is still searched ONCE, with every line in it",
      [label for label, _, _ in SEEN] == ["Bridge", "Chorus"], SEEN)
check("...and wide enough to hold the solo AND the chorus after it",
      SEEN[-1][1] < 173.0 and SEEN[-1][2] > 237.0 and SEEN[-1][2] - SEEN[-1][1] > A.MAX_WINDOW,
      SEEN[-1])
check("...so the chorus is found after the solo, not inside it", abs(chorus_at - 217.0) < 0.5,
      chorus_at)

check.done()
