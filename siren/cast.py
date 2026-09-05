"""Siren Cast — one AceStep 1.5 text encode that says WHO sings WHERE.

Replaces `TextEncodeAceStepAudio1.5`. Two problems it exists to fix, both of them things the core
node cannot express rather than things it does badly.

**1. Tags have no time axis, and the plan does.** With `generate_audio_codes` on, the text encoder
runs a Qwen LM that emits `ceil(duration) * 5` tokens — one every 200 ms, i.e. a 5 Hz plan of the
whole song (`comfy/text_encoders/ace15.py:200`). Each token is a single FSQ index
(`fsq_levels=[8,8,8,5,5,5]` = 64000 values, `fsq_input_num_quantizers=1`), so the tokens are
independent of one another. The detokenizer expands each one into 5 latent frames with attention
running ONLY inside its own 5-frame window — `x.view(B * T, P, D)`, `ace_step15.py:998` — so there
is no mixing between neighbouring codes at all. What comes out is concatenated onto the latent
channel-wise, frame for frame: `torch.cat([context_latents, hidden_states], dim=-1)`
(`ace_step15.py:681`).

That makes the plan the strongest and the only *time-aligned* conditioning in the model. `tags` and
`lyrics` reach the DiT globally, through cross-attention — which is exactly why "male vocal,
female vocal" in the tags reads as a wish about the average track and not as an instruction about
the second verse. So this node builds the plan **per section**, each with its own voice.

Not by generating each section separately and gluing: before decoding section k the codes already
produced for sections 1..k-1 are put back into the prompt as the assistant's output so far, and
decoding continues from there — a real continuation, with the section's own caption in front of it
and the whole song's metas in the `<think>` block. The sampling loop itself is comfy's own
(`ace15.sample_manual_loop_no_classes`, handed ready-made `ids`), so nothing about how a token is
drawn is reimplemented here. Total decode cost is the same as one full-length pass; the extra cost
is one prefill per section.

**2. `cfg_scale` on the core node barely touches the caption.** The LM's negative prompt is built
with the SAME caption and the SAME lyrics — only the `<think>` metas block is emptied
(`ace15.py:212-222`). So `cfg_scale 2.0` amplifies "with bpm/duration/key" against "without them",
and the caption gets no guidance whatsoever. The tokenizer already accepts `caption_negative`,
`lyrics_negative` and `*_negative` metas; the core node just never passes them. This node does, and
mirrors the metas into the negative so the two prompts differ in the caption ALONE. In the default
`voice delta` mode the negative is that section's caption *minus the voice line*, which points
`cfg_scale` at precisely the difference between "someone sings this" and "SHE sings this".

Everything else follows from the 5 Hz grid: a section's length is rounded to whole codes
(0.2 s), `seconds` comes out as the exact total to drive `EmptyAceStep1.5LatentAudio`, and the
timeline is printed in m:ss so a section can be typed straight into Siren Section for a retake.
"""
import json
import math
import re
import time

from ..context.character_card import VOICE_TYPE
from ..timer.timer_nodes import _format_elapsed
from ..categories import CAT_SIREN

# One audio code per 200 ms. Not a guess: `tokens_duration = duration * 5` in the tokenizer, and
# `audio_codes.shape[1] * 5` is compared against the 25 fps latent length in the model.
CODES_HZ = 5.0

# Copied from comfy_extras/nodes_ace.py rather than imported, so this module stays importable
# (and testable) without ComfyUI on the path. Both are static lists there.
LANGUAGES = ['ar', 'az', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr',
             'he', 'hi', 'hr', 'ht', 'hu', 'id', 'is', 'it', 'ja', 'ko', 'la', 'lt', 'ms', 'ne',
             'nl', 'no', 'pa', 'pl', 'pt', 'ro', 'ru', 'sa', 'sk', 'sr', 'sv', 'sw', 'ta', 'te',
             'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yue', 'zh', 'unknown']
KEYSCALES = [f"{root} {quality}" for quality in ["major", "minor"]
             for root in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab",
                          "A", "A#", "Bb", "B"]]
TIMESIGS = ['2', '3', '4', '6']

GUID_DELTA = "voice delta"
GUID_TAGS = "negative tags"

# What a row naming several singers becomes. The plan carries one timbre per 200 ms code, so all
# three are ways of saying one thing where two were asked for — they differ in WHICH one thing.
DUET_LEAD = "lead + backing note"
DUET_UNISON = "one unison duet"
DUET_BLEND = "both descriptions (old behaviour)"
DUET_MODES = [DUET_LEAD, DUET_UNISON, DUET_BLEND]
GUID_CORE = "metas only (core behaviour)"
GUIDANCE = [GUID_DELTA, GUID_TAGS, GUID_CORE]

# The value comfy pads a short code sequence with (`ace_step15.py:1097`) — used here only if the LM
# somehow returns fewer tokens than were asked for, which the loop's own bookkeeping prevents.
PAD_CODE = 35847

# What a plan row's voice cell may say to mean "nobody sings here".
_NO_VOCAL = {"", "-", "--", "---", "—", "–", "none", "no vocal", "no vocals", "instrumental",
             "instr", "n/a", "na", "tacet"}

# Words that make a row a pasted table header rather than a section.
_HEADERISH = {"section", "part", "voice", "voices", "singer", "who", "length", "seconds", "secs",
              "sec", "bars", "time", "duration", "notes", "extra", "vocal", "vocals", "name"}

# An explicit end-of-table marker, and everything after it is ignored. This exists for the LLM that
# writes the plan: a grammar-constrained model has to *choose* the EOS token to stop, and EOS is
# usually a low-probability option that top_p / min_p prune away — leaving "another row" as the only
# legal continuation, i.e. a runaway. A terminator the grammar can require turns stopping into a
# word the model is happy to write, after which EOS is the only token the grammar still permits.
_END_WORDS = {"end", "end.", "[end]", "(end)", "конец"}

_SEP_ROW = re.compile(r"^[\s|:+=-]+$")
_MMSS = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")
_BARS = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:b|bar|bars)$")
_SECS = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?$")


# ------------------------------------------------------------------------------------- formatting
def _mmss(sec):
    """m:ss.s — the unit you point at a spot in a track with."""
    sec = max(0.0, float(sec))
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def _num(v):
    """A number for a prompt: 2.0 → "2", 1.7142857 → "1.714". Trailing zeros in a budget line read
    like precision that isn't there."""
    return f"{float(v):.3f}".rstrip("0").rstrip(".") or "0"


# Sloppy spellings a config pass produces for a language code. AceStep's list has 'uk' for
# Ukrainian and 'zh' for Chinese, and a model asked for two letters cheerfully writes 'ua' or 'cn'.
_LANG_FIX = {"ua": "uk", "ukr": "uk", "cn": "zh", "chi": "zh", "zho": "zh", "jp": "ja", "jpn": "ja",
             "kr": "ko", "kor": "ko", "gr": "el", "ell": "el", "cz": "cs", "se": "sv", "dk": "da",
             "rus": "ru", "eng": "en", "ger": "de", "deu": "de", "fra": "fr", "spa": "es",
             "por": "pt", "ita": "it", "pol": "pl", "tur": "tr", "heb": "he", "ara": "ar"}

_KEY_RE = re.compile(r"^\s*([A-G])\s*(#|b|♯|♭|sharp|flat)?\s*"
                     r"(major|minor|maj|min|m)?\s*$", re.I)


def _beats(text, default=4):
    """Beats per bar out of whatever the config pass produced: 4, "4", " 4/4 ".

    A plain widget rather than a dropdown for one practical reason: a combo input cannot take the
    STRING a text parser hands it, and this value always arrives from the song config."""
    m = re.search(r"[1-9][0-9]?", str(text if text is not None else ""))
    return int(m.group()) if m else int(default)


def _language(text, default="en"):
    """(code, note). The tokenizer only prepends this to the lyrics prompt, so a wrong code is not
    fatal — it is just a lie told to the model, which is worth a line in the report."""
    raw = str(text or "").strip().lower()
    if raw in LANGUAGES:
        return raw, ""
    if raw in _LANG_FIX:
        return _LANG_FIX[raw], f"language {raw!r} is not one of AceStep's codes — read as "                                f"{_LANG_FIX[raw]!r}"
    return default, (f"language {raw!r} is not a code AceStep knows — using {default!r}. Its list is "
                     f"two-letter ISO plus 'yue' and 'unknown'." if raw else "")


def _keyscale(text, default="C major"):
    """(value, note) for a key that arrives as text. Accepts 'C major', 'c# minor', 'C sharp minor',
    'Am' — anything AceStep's own list can express. The list carries both spellings of every black
    key (C# and Db), so whichever was written is kept."""
    raw = " ".join(str(text or "").split())
    if raw in KEYSCALES:
        return raw, ""
    m = _KEY_RE.match(raw)
    if m:
        root, acc, qual = m.group(1).upper(), (m.group(2) or "").lower(), (m.group(3) or "").lower()
        acc = {"sharp": "#", "♯": "#", "flat": "b", "♭": "b"}.get(acc, acc)
        # An explicit set, NOT a startswith: "major" starts with an 'm' too, and a prefix test
        # here quietly turned every major key minor.
        quality = "minor" if qual in ("minor", "min", "m") else "major"
        val = f"{root}{acc} {quality}"
        if val in KEYSCALES:
            return val, ("" if val == raw else f"key {raw!r} read as {val!r}")
    return default, (f"AceStep's key list cannot express {raw!r} — using {default!r}. It is a root "
                     f"(C, C#, Db, …) plus 'major' or 'minor'." if raw else "")


def _bar_seconds(bpm, beats_per_bar):
    """Length of one bar in seconds. **The one place this arithmetic lives.** `Siren Tempo` hands it
    to the LLM that writes the plan and `_parse_length` measures the plan's `N bars` with it, so the
    budget the model aims at and the seconds the node computes cannot drift apart. None when there is
    no tempo to measure against."""
    if float(bpm) <= 0 or int(beats_per_bar) <= 0:
        return None
    return (60.0 / float(bpm)) * int(beats_per_bar)


def _join_caption(base, *adds):
    """Glue the section's own lines onto the shared caption as prose.

    AceStep's caption is a paragraph, not a tag list (every shipped template writes it that way),
    so additions are appended as sentences. An empty addition changes nothing at all, which is what
    makes `voice delta` guidance exact: the negative is this function called without the voice."""
    out = (base or "").strip()
    for add in adds:
        add = (add or "").strip().strip(",;")
        if not add:
            continue
        if out and out[-1] not in ".!?":
            out += "."
        out = (out + " " + add).strip()
        if out[-1] not in ".!?":
            out += "."
    return out


# ------------------------------------------------------------------------------------------- plan
def _parse_length(text, bpm, beats_per_bar):
    """`24`, `24s`, `0:24`, `8 bars` → seconds. Returns (seconds, error) with one of them None/"".

    Bars are the unit a section is actually written in, and at a known bpm they land on whole
    codes: 8 bars of 4/4 at 120 bpm is 16.0 s = exactly 80 codes."""
    t = str(text or "").strip().lower().replace(",", ".")
    if not t:
        return None, "no length given"
    m = _MMSS.match(t)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)), ""
    m = _BARS.match(t)
    if m:
        bar = _bar_seconds(bpm, beats_per_bar)
        if bar is None:
            return None, "a length in bars needs a bpm"
        return float(m.group(1)) * bar, ""
    m = _SECS.match(t)
    if m:
        return float(m.group(1)), ""
    return None, f"can't read the length {text!r} (try 24, 24s, 0:24 or 8 bars)"


def _is_header(cells):
    """A pasted `section | voice | seconds` header, or a markdown table rule."""
    words = [re.sub(r"[^a-z]", "", c.lower()) for c in cells if c.strip()]
    return bool(words) and all(w in _HEADERISH for w in words)


def _parse_plan(plan, bpm, beats_per_bar):
    """The plan block → a list of section rows, plus notes about the lines that were dropped.

    One row per line: ``label | voice | length`` and an optional 4th cell appended to that
    section's caption. Blank lines, `#` comments, table rules and a pasted header are skipped in
    silence — the block is meant to be paste-able from whatever wrote it. A bare ``END`` line ends
    the table and everything below it is dropped, also in silence: that is how an LLM says it is
    finished, and whatever it rambles afterwards is not a section."""
    rows, notes = [], []
    for lineno, raw in enumerate(str(plan or "").splitlines(), 1):
        line = raw.strip()
        if line.lower() in _END_WORDS:
            break
        if not line or line.startswith("#") or line.startswith("//") or _SEP_ROW.match(line):
            continue
        if "|" not in line:
            notes.append(f"plan line {lineno} has no '|' separator and was skipped: {line[:60]!r}")
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if _is_header(cells):
            continue
        while len(cells) < 3:
            cells.append("")
        secs, err = _parse_length(cells[2], bpm, beats_per_bar)
        label = cells[0] or f"section {len(rows) + 1}"
        if secs is None or secs <= 0:
            notes.append(f"plan line {lineno} ({label}): {err or 'the length is 0'} — row skipped")
            continue
        codes = max(1, int(round(secs * CODES_HZ)))
        rows.append({"label": label, "voice_raw": cells[1], "codes": codes,
                     "seconds": codes / CODES_HZ, "asked_seconds": secs,
                     "extra": ", ".join(c for c in cells[3:] if c), "line": lineno})
    return rows, notes


# 'female' contains 'male', so male is matched with a lookbehind and female is tested first. These
# live here rather than in `score.py` because both nodes need them and score imports from cast, not
# the other way round — a second copy is how the two would start disagreeing about who is a man.
_FEMALE_RE = re.compile(r"female|woman|women|girl|жен|жіно|жино", re.I)
_MALE_RE = re.compile(r"(?<!fe)male|\bman\b|\bmen\b|\bboy\b|муж|чолов", re.I)
_WORD_RE = re.compile(r"[0-9a-zA-Zа-яёА-ЯЁіїєґІЇЄҐ]+")

#: Head nouns that carry no timbre of their own. Dropped when a voice is being shortened, because
#: "gritty growls male backing harmonies" is a worse sentence than "gritty male backing harmonies"
#: and the budget is only a few words.
_NOT_TIMBRE = {"vocal", "vocals", "voice", "voices", "singing", "sung", "singer", "sings",
               "backing", "harmony", "harmonies", "lead", "and", "with", "a", "an", "the",
               "in", "of", "on"}


def _gender_of(text):
    """'female' / 'male' / None. Female first — 'female' contains 'male'."""
    t = str(text or "")
    if _FEMALE_RE.search(t):
        return "female"
    if _MALE_RE.search(t):
        return "male"
    return None


def _timbre_words(tags, limit=2):
    """Up to `limit` describing words from a voice's tags, in the order they were written.

    Only a couple of words, because length is what breaks this: a full voice description standing
    next to another one is what made a duet come back as a single indistinct voice. Two adjectives
    tell one singer from another and are short enough not to compete with the lead.

    The whole string is scanned, not just the first clause. A card written the natural way opens
    with what the singer *is* — "male lead vocal, raspy baritone, close-mic" — so the first clause is
    often nothing but the words this deliberately skips, and stopping at the comma would hand back
    an empty list for a card that describes its singer perfectly well.
    """
    out = []
    for w in _WORD_RE.findall(str(tags or "")):
        lw = w.lower()
        if lw in _NOT_TIMBRE or _gender_of(lw):
            continue
        out.append(lw)
        if len(out) >= max(0, int(limit)):
            break
    return out


def _short_voice(tags, limit=2):
    """A voice in three or four words: what it sounds like, then who it is — 'gritty male'.

    The alternative, and what this replaces, is naming only the gender. That is short but it is not
    an identity: a pack that learned in Phantas that a character survives only when re-stated in the
    same literal words should not reduce its second singer to 'male'."""
    words = _timbre_words(tags, limit)
    g = _gender_of(tags)
    return " ".join(words + ([g] if g else []))


def _resolve_voice(voice_raw, voices):
    """A plan row's voice cell → the caption fragment for that section.

    `Nina` or `Alex + Nina` look the roster up by name (a wired Character Card's `voice` output);
    anything that isn't in the roster is taken as free text VERBATIM, whole cell at a time. That
    last rule is why the cell is only ever split on `+` and `&`: splitting on commas too would chop
    `female, airy, close-mic` into three names that resolve to nothing."""
    raw = str(voice_raw or "").strip()
    out = {"add": "", "names": [], "unknown": [], "silent": [], "verbatim": False}
    if raw.lower() in _NO_VOCAL:
        return out
    tokens = [t.strip() for t in re.split(r"[+&]", raw) if t.strip()]
    found, missing = [], []
    for t in tokens:
        v = voices.get(t.lower()) or voices.get(t.lower().split()[0] if t.split() else "")
        (found if v else missing).append(v or t)
    if missing or not found:
        # Free text. Any names that DID resolve are still reported, because a half-matching row
        # ("Nina + backing choir") is nearly always a typo in the half that didn't.
        out.update(add=raw, names=[v["name"] for v in found], unknown=missing, verbatim=True)
        return out
    out["names"] = [v["name"] or "(unnamed)" for v in found]
    out["silent"] = [v["name"] or "(unnamed)" for v in found if not v.get("tags")]
    out["add"] = ", ".join(v["tags"] for v in found if v.get("tags"))
    return out


def _collapse_duet(names, voices, mode):
    """A row naming several singers → ONE caption fragment, or None when there is nothing to do.

    This lives in **Cast** rather than only in `Siren Score` because the reason for it is a fact
    about the model, not about who filled the table in. The plan is one audio code per 200 ms and a
    section's caption is one description, so two full voice descriptions side by side ask for one
    caption to hold two contradictory timbres — measured once as "two female vocals" where a man and
    a woman were asked for. Score protects a plan it wrote; a plan typed by hand into the widget went
    straight to that failure with no warning at all, which is the case this closes.

    Returns `others` as well: the voices that lost their place in the section caption. They still
    belong in the GLOBAL one, or a singer who only ever appears in duets never reaches the model at
    all — and the global list is, per `cast_in_caption`, the only route by which a second timbre can
    colour those frames.
    """
    if mode == DUET_BLEND:
        return None            # the old behaviour IS the absence of a collapse
    pairs = []
    for n in names:
        key = str(n).strip().lower()
        v = voices.get(key) or voices.get(key.split()[0] if key.split() else "")
        tags = ((v or {}).get("tags") or "").strip()
        if tags:
            pairs.append((n, tags))
    if len(pairs) < 2:
        return None
    lead_name, lead_tags = pairs[0]
    others = pairs[1:]
    both = " + ".join(n for n, _ in pairs)

    if mode == DUET_UNISON:
        shorts = [s for s in (_short_voice(t) for _, t in pairs) if s]
        if len(shorts) < 2:
            return None
        add = "two voices singing together in close unison harmony, " + " and ".join(shorts)
        why = (f"{both} were written as ONE unison duet — '{add}' — instead of a lead with backing. "
               f"This asks the model for a duet as a single sound rather than for two singers at "
               f"once; it is the mode to A/B, not the safe one.")
    else:
        genders = []
        for _, t in others:
            g = _gender_of(t)
            if g and g not in genders:
                genders.append(g)
        short = (_short_voice(others[0][1]) if len(others) == 1
                 else (genders[0] if len(genders) == 1 else ""))
        back = f"with {short} backing harmonies" if short else "with backing harmonies"
        add = f"{lead_tags}, {back}" if lead_tags else back
        why = (f"{both} sing together — {lead_name} was made the lead and the rest became "
               f"'{back}'.")
    return {"add": add, "note": why, "others": [t for _, t in others]}


def _voices_in_order(kwargs):
    """The wired `voice_N` inputs, in slot order. Sorted NUMERICALLY — a lexical sort would put
    `voice_10` between `voice_1` and `voice_2`, silently reordering the roster on the tenth member.
    Order only decides who wins a name clash, but a clash you can't predict is worse than one you
    can."""
    found = []
    for key, val in kwargs.items():
        m = re.fullmatch(r"voice_(\d+)", key)
        if m and isinstance(val, dict):
            found.append((int(m.group(1)), val))
    return [v for _, v in sorted(found, key=lambda p: p[0])]


def _roster(voices):
    """The wired `voice` inputs → {lowercased name: voice}, plus notes. Later wins on a clash, and
    a member is also reachable by their first name so a plan can say `Alex` for `Alex Kin`."""
    table, notes = {}, []
    for v in voices:
        if not isinstance(v, dict):
            continue
        name = (v.get("name") or "").strip()
        if not name:
            notes.append("a wired voice has no name — a plan row cannot refer to it; "
                         "fill in the Character Card's 'name'")
            continue
        key = name.lower()
        if key in table:
            notes.append(f"two wired voices are both called {name!r} — the later one wins")
        table[key] = v
        first = key.split()[0]
        table.setdefault(first, v)
    return table, notes


# --------------------------------------------------------------------------------------- LM plumbing
def _audio_start_id():
    """The token id audio codes start at. Read off comfy's own sampling loop so a change there
    can't silently shift every code by a constant; the literal is the fallback."""
    try:
        import inspect

        import comfy.text_encoders.ace15 as ace15
        return int(inspect.signature(ace15.sample_manual_loop_no_classes)
                   .parameters["audio_start_id"].default)
    except Exception:
        return 151669


def _ids_of(tokens, key):
    """Plain token ids for one of the tokenizer's prompts. The Qwen3 tokenizer is built with
    `max_length=99999999, pad_to_max_length=False`, so a prompt is always one chunk — the same
    `[0]` comfy's own `generate_audio_codes` takes."""
    chunks = tokens.get(key) or []
    if not chunks:
        return []
    return [t for t, *_ in chunks[0]]


def _pair_ids(pos, neg, prefix, pad, audio_start, use_cfg):
    """cond (and uncond) id sequences, front-padded to equal length and continued from `prefix`.

    Front-padding is comfy's rule and it matters here for a second reason: it leaves the appended
    codes at the SAME positions in both sequences, so cond and uncond are looking at the same point
    of the song when the next token is drawn."""
    pos, neg = list(pos), list(neg)
    if use_cfg and neg:
        if len(neg) < len(pos):
            neg = [pad] * (len(pos) - len(neg)) + neg
        elif len(pos) < len(neg):
            pos = [pad] * (len(neg) - len(pos)) + pos
        seqs = [pos, neg]
    else:
        seqs = [pos]
    tail = [int(c) + audio_start for c in prefix]
    return [s + tail for s in seqs]


def _graph_dropper():
    """comfy's own end-of-node graph cleanup, to be run at every SECTION boundary instead.

    Since comfy's `Prs/ace graphs` the ACE 1.5 audio-code LM runs its decode steps through a captured
    CUDA graph (`ace15.py` turns on `graph_dynamic_vbar_blocks`), and the graph has the addresses of
    that call's `FixedKV` key/value/position/seqlen baked into it. Comfy drops those graphs exactly
    once, in `execution.py`'s `finally` — because comfy's own sampler is called exactly once per node
    execution.

    This node calls it once per section, and every call allocates a fresh KV cache. The graph cache
    is keyed on weight residency alone and knows nothing about the cache it was captured against, so
    from the second section on comfy replayed the first section's graph against buffers that had
    already been freed and handed back out. `position` then held whatever the allocator had put
    there, and the scatter into the cache (`llama.py`, `fixed_cache.key.scatter_(1, position, xk)`)
    tripped a device-side assert — which takes the whole ComfyUI process down with it, not just the
    prompt.

    So the graphs go at every section boundary. The cost is a re-warm and a re-capture per section —
    two decode steps — against a section that is dozens to hundreds of steps long.

    Returns None on a comfy that has none of this, where there is nothing to drop."""
    try:
        import comfy.model_prefetch
    except Exception:
        return None
    return getattr(comfy.model_prefetch, "cleanup_prefetch_queues", None)


def _decode_is_graphed(lm):
    """Whether comfy will capture a decode-step CUDA graph for this LM at all. Only used to decide
    whether a missing `_graph_dropper` is worth warning about."""
    model = getattr(getattr(lm, "transformer", None), "model", None)
    return bool(getattr(model, "graph_dynamic_vbar_blocks", False))


class _OnePlanBar:
    """One progress bar for the whole plan instead of one per section.

    `sample_manual_loop_no_classes` builds its own `comfy.utils.ProgressBar` and its own tqdm on every
    call, so a twelve-section plan produced twelve bars each restarting at zero — which reads as if
    the node were stuck in a loop. Both are looked up off `comfy.utils` by attribute at call time, so
    they can be swapped for the duration of the decode and put back in a `finally`.

    Entered once per section — the patch goes on and comes off around each call — but the two bars
    themselves are made once and live across all of them, which is what makes the progress continuous.
    `close()` ends them after the last section.

    Nothing else runs while this is patched (ComfyUI executes one node at a time), and if either name
    ever moves the swap is skipped and the per-section bars come back — a cosmetic regression rather
    than a crash."""

    def __init__(self, total):
        import comfy.utils
        self.utils = comfy.utils
        self.total = max(1, int(total))
        self.done = 0
        self._real_bar = getattr(comfy.utils, "ProgressBar", None)
        self._real_trange = getattr(comfy.utils, "model_trange", None)
        self.ok = self._real_bar is not None and self._real_trange is not None
        self.bar = self._real_bar(self.total) if self.ok else None
        self.tq = None

    def __enter__(self):
        if not self.ok:
            return self
        outer = self

        class _Shim:                       # what the inner loop thinks is a fresh ProgressBar
            def __init__(self, total=0, node_id=None):
                pass

            def update_absolute(self, value, total=None, preview=None):
                outer.bar.update_absolute(min(outer.total, outer.done + int(value) + 1), outer.total)

            def update(self, n=1):
                self.update_absolute(int(n))

        def _trange(n, *args, **kwargs):    # one tqdm for every section's tokens together
            for i in range(int(n)):
                yield i
                if outer.tq is not None:
                    outer.tq.update(1)

        if self.tq is None:                 # made once, not once per section
            try:
                from tqdm.auto import tqdm
                self.tq = tqdm(total=self.total, desc="LM plan", smoothing=1.0)
            except Exception:
                self.tq = None
        self.utils.ProgressBar = _Shim
        self.utils.model_trange = _trange
        return self

    def section_done(self, count):
        self.done = min(self.total, self.done + int(count))

    def __exit__(self, *exc):
        if self.ok:
            self.utils.ProgressBar = self._real_bar
            self.utils.model_trange = self._real_trange
        return False

    def close(self):
        """After the last section. Separate from `__exit__` so the bars survive between them."""
        if self.tq is not None:
            self.tq.close()
            self.tq = None
        if self.bar is not None:
            self.bar.update_absolute(self.total, self.total)


def _lm_of(clip):
    """The audio-codes LM inside the ACE 1.5 text encoder — `getattr(self, self.lm_model)`, the
    same lookup `ACE15TEModel.encode_token_weights` does, with its `qwen3_06b` fallback."""
    te = getattr(clip, "cond_stage_model", None)
    if te is None:
        raise RuntimeError("[Siren Cast] this CLIP has no text encoder model.")
    name = getattr(te, "lm_model", None)
    lm = getattr(te, name, None) if name else None
    if lm is None:
        lm = getattr(te, "qwen3_06b", None)
    if lm is None:
        raise RuntimeError(
            "[Siren Cast] this CLIP is not an AceStep 1.5 text encoder — no audio-codes LM on it. "
            "Load both encoders with DualCLIPLoader (qwen_0.6b_ace15 + qwen_4b_ace15, type 'ace').")
    return te, lm


# --------------------------------------------------------------------------------------- the node
class KinburgSirenCast:
    """AceStep 1.5 text encode with a per-section voice plan and working caption guidance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The AceStep 1.5 text encoders — DualCLIPLoader with BOTH qwen_0.6b_ace15 and qwen_4b_ace15 (type 'ace'). Not either/or: the small one embeds the caption and the lyrics, the big one writes the audio-code plan."}),
                "tags": ("STRING", {"forceInput": True, "tooltip": "The caption for the WHOLE song — genre, instrumentation, era, mix, energy. Write it as prose, the way the shipped AceStep templates do, not as a comma-separated tag list.\n\nLeave the vocals out of it: each section gets its own voice line from the plan, and repeating 'male vocal' here fights the plan for the sections a woman sings. What DOES belong here is anything true of the whole track ('two lead vocalists trading verses' is fine)."}),
                "lyrics": ("STRING", {"forceInput": True, "tooltip": "The full lyrics, with the usual '[Verse 1 - ...]' / '[Chorus - ...]' markers. Always the WHOLE song, for every section's pass — the LM needs to see where in the text it is, and the marker names are how it lines the plan up with the words.\n\nKeep the section markers spelled the same way as the plan's labels. Nothing enforces it, but you will be reading both."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Seed for the audio-code plan — THE seed that decides the musical idea. The sampler's seed only draws noise; change this one for a different piece.\n\nEach section draws from seed + its index, so editing one section's voice leaves every earlier section bit-identical."}),
                "bpm": ("INT", {"default": 120, "min": 10, "max": 300, "tooltip": "Tempo, into the metas the plan is written against. Also what 'N bars' in the plan is measured with."}),
                "duration": ("FLOAT", {"default": 120.0, "min": 0.0, "max": 2000.0, "step": 0.1, "advanced": True, "tooltip": "Song length in seconds — used ONLY when the plan is empty. With a plan the length is the sum of its rows and this widget is ignored (the report says so).\n\nEither way the true value comes out on 'seconds': wire it into Empty Ace Step 1.5 Latent Audio instead of typing the number twice. Duration is baked into the plan's tokens, so a latent that disagrees with it is the classic AceStep mistake."}),
                "timesignature": ("STRING", {"default": "4", "tooltip": "Beats per bar, into the metas — and what 'N bars' in the plan is measured with. A plain field rather than a dropdown because a combo input cannot accept the STRING a text parser hands it; anything with a digit in it is read (4, '4', '4/4')."}),
                "language": ("STRING", {"default": "en", "tooltip": "Language of the lyrics, prepended to the lyrics embedding. A plain field so the song config can wire straight in, with the usual slips corrected: Ukrainian is 'uk' (not 'ua'), Chinese 'zh' (not 'cn'), Japanese 'ja'. An unknown code is reported and falls back to 'en'."}),
                "keyscale": ("STRING", {"default": "C major", "tooltip": "Key and mode, into the metas. A plain field so the song config can wire straight in: 'C major', 'c# minor', 'C sharp minor' and 'Am' are all read. AceStep's list carries both spellings of every black key (C# and Db), so whichever was written is kept; anything it cannot express is reported and falls back to 'C major'."}),
                "guidance": (GUIDANCE, {"default": GUID_DELTA, "advanced": True, "tooltip": "What the plan LM's 'cfg_scale' is actually pushing against. This is the fix for 'the model ignores my tags'.\n\n• voice delta (recommended) — the negative is the SHARED caption, so cfg_scale amplifies whatever this section ADDS to it: the difference between 'someone sings this' and 'SHE sings this'. Note that a section's 4th column rides the same delta, which is right for a duet's 'with male backing harmonies' and is worth knowing for an arrangement note like 'drums drop out'. Sections that add nothing fall back to core behaviour.\n\n• negative tags — the negative is the 'negative_tags' text below. General prompt adherence rather than per-section vocals.\n\n• metas only (core behaviour) — what TextEncodeAceStepAudio1.5 does: the negative repeats the same caption and the same lyrics, with only the metas block emptied. So cfg_scale guides bpm/duration/key and NOTHING about the caption. Here for A/B only.\n\nIn the first two modes the metas are copied into the negative as well, so the two prompts differ in the caption alone."}),
                "negative_tags": ("STRING", {"multiline": True, "default": "", "advanced": True, "tooltip": "The caption to guide AWAY from ('spoken word, off-key, muddy mix, drum machine'). Used by the 'negative tags' mode; in 'voice delta' mode it is appended to that section's negative, so it stacks.\n\nNOTE this is not the sampler's negative — that one must stay a ConditioningZeroOut. This text never reaches the DiT; it only shapes the plan."}),
                "cast_in_caption": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Append the distinct voices used by the plan to the GLOBAL caption — the one that reaches the DiT through cross-attention for the whole track.\n\nWhat it actually governs, from listening: whether a SECOND voice can appear inside a section — the backing lines in round brackets. A section's own caption names one singer, so the only route by which another timbre can reach those frames is this global list. Off, and the brackets tend to be sung by the section's own voice.\n\nIt is not the accuracy dial (that is 'lyrics_in_negative'), and neither setting is reliable enough to call correct — it depends on the song, so it is worth trying both ways on a new one."}),
                "lyrics_in_negative": ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Keep the lyrics in the LM's negative prompt. True is what the core node does; OFF is the default here, and it is the single most important setting on this node.\n\nWith the lyrics dropped from the negative, cfg_scale guides the LYRICS as well as the caption — including the '[Verse 1 - Nina]' markers inside them, which is why the voices then land where the text says. Measured across many takes of one song on a fixed seed and sampler: near-perfect assignment with this off, unreliable with it on.\n\nTurn it back on only to reproduce the core node's behaviour, or if diction comes out over-articulated."}),
                "generate_audio_codes": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Run the plan LM at all. Off is for when you are giving the model a reference audio instead (Set Reference Audio), which replaces the plan with the reference's own tokens.\n\nOff makes the plan meaningless — there are no codes to assemble — and the node says so."}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1, "advanced": True, "tooltip": "Guidance for the plan LM. Left at the core default on purpose: with 'guidance' set to voice delta it now means something it did not mean before, and that is one change to judge on its own before this number moves.\n\nAt exactly 1.0 the negative pass is skipped entirely and every guidance mode becomes a no-op (it also halves the LM's cost)."}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01, "advanced": True, "tooltip": "Randomness of the plan. 0.85 is the core default; 0.6-0.7 makes the plan follow the caption and the lyrics more closely.\n\nDo NOT go to 0: that is greedy decoding, and an autoregressive audio LM decoded greedily tends to fall into a repeating loop — a section that keeps restarting the same bar."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2000.0, "step": 0.01, "advanced": True, "tooltip": "Nucleus sampling for the plan. Core default 0.9."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "advanced": True, "tooltip": "Keep only the k most likely codes per step. 0 = off, which is the core default."}),
                "min_p": ("FLOAT", {"default": 0.000, "min": 0.0, "max": 1.0, "step": 0.001, "advanced": True, "tooltip": "Drop codes less likely than min_p x the top code. 0 = off. 0.02-0.05 is a gentler way to tighten the plan than lowering temperature, because it only cuts the tail."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Print the timeline and the warnings to the console. The same text is always on the outputs."}),
                # APPENDED, and it has to stay last: ComfyUI maps a saved workflow's widget values by
                # POSITION, so a new widget anywhere earlier would shift every value after it.
                "duet_mode": (DUET_MODES, {"default": DUET_LEAD, "advanced": True, "tooltip": "What a plan row naming SEVERAL singers ('Nina + Alex') becomes.\n\nThe plan carries one audio code per 200 ms and a section's caption is one description, so two timbres over the same frames come back as their average — measured once as 'two female vocals' where a man and a woman were asked for. All three modes therefore say ONE thing where two were asked for; they differ in which one.\n\n• lead + backing note (default) — the first name leads and the rest become a short phrase in words ('with gritty male backing harmonies'). What Siren Score already writes into the plan, applied here too so a hand-typed row behaves the same.\n\n• one unison duet — names the pair as a SINGLE sound ('two voices singing together in close unison harmony, airy female and gritty male') instead of a lead with backing. Worth A/B-ing on a fixed seed: the measurement above was taken on two full descriptions pasted side by side, which is a contradictory caption rather than a description of a duet, and AceStep has certainly heard duets. Untested — that is what it is here for.\n\n• both descriptions — the old behaviour, both voices' full tags joined. Here to A/B against.\n\nEither way the singers who lose their place in the SECTION caption still reach the global one, so a member who only ever appears in duets is not invisible to the model.\n\nRows naming one singer, or free text, are untouched. Per-line markers in the lyrics remain the only way to make voices genuinely ALTERNATE."}),
            },
            "optional": {
                "plan": ("STRING", {"forceInput": True, "tooltip": "WHO sings WHERE — one section per line:\n\n  Intro    | -           | 8\n  Verse 1  | Alex        | 24\n  Chorus   | Nina        | 8 bars\n  Verse 2  | Mike        | 0:24\n  Chorus   | Nina + Alex | 8 bars\n  Outro    | -           | 8\n\n• column 1 — the label, only for the report\n• column 2 — a wired voice's name (several joined by '+'), or free text used as-is, or '-' for no vocal\n• column 3 — the length: seconds, '24s', 'm:ss', or 'N bars' (needs bpm)\n• column 4 — OPTIONAL, appended to this section's caption ('drums drop out')\n\nBlank lines, '#' comments and a pasted table header are ignored, so this can come straight out of an LLM. The lengths add up to the 'seconds' output — wire that into Empty Ace Step 1.5 Latent Audio and the two can never disagree.\n\nEMPTY = one caption for the whole song, i.e. the core node's behaviour plus the guidance fix below. Start there."}),
                "voice_1": (VOICE_TYPE, {"tooltip": "A band member — the 'voice' output of a Character Card (or Card Presets). Its 'voice_tags' is what a plan row referring to that name pastes onto the caption; nothing else from the card is used here.\n\nConnect one and another slot appears. Plan rows can also just say the description in words, so wiring these is optional — it is how you stop retyping the same voice in every song."}),
                "voice_2": (VOICE_TYPE,),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING", "STRING", "GEN_INFO")
    RETURN_NAMES = ("conditioning", "seconds", "timeline", "report", "gen_extra_info")
    FUNCTION = "run"
    CATEGORY = CAT_SIREN
    DESCRIPTION = ("AceStep 1.5 text encode that says who sings where. The audio-code plan the text "
                   "encoder's LM writes is a 5 Hz, frame-aligned track — the only conditioning in "
                   "the model with a time axis — so this node builds it section by section, each "
                   "with its own voice, as one continuous decode rather than separate clips glued "
                   "together. It also passes the negative caption the core node never passes, "
                   "without which the LM's cfg_scale guides the bpm and the key and nothing about "
                   "the prompt. Voices come from Character Card's 'voice' output, so a band member "
                   "is described once and feeds the lyrics LLM, the cover art and the song. "
                   "'seconds' drives Empty Ace Step 1.5 Latent Audio so the two can't disagree.")

    # --------------------------------------------------------------------------- prompt assembly
    @staticmethod
    def _metas(bpm, duration, timesignature, keyscale):
        return {"bpm": int(bpm), "duration": float(duration),
                "timesignature": int(timesignature), "keyscale": keyscale}

    @staticmethod
    def _negatives(neg_caption, metas, lyrics_in_negative):
        """The `*_negative` kwargs. Mirroring the metas is the whole point: without it the negative
        prompt loses its `<think>` block, and cfg_scale spends itself on bpm/duration/key instead of
        on the caption.

        `duration` is ceiled here to match what the positive prompt will contain: the tokenizer
        rounds the positive duration up to a whole second before building the think block, but a
        `duration_negative` is popped raw (`ace15.py:198-207`). Left alone, the two blocks would read
        `112` and `112.0` — a difference cfg_scale would then dutifully guide on."""
        out = {}
        if neg_caption is not None:
            out["caption_negative"] = neg_caption
            for k, v in metas.items():
                out[f"{k}_negative"] = int(math.ceil(float(v))) if k == "duration" else v
        if not lyrics_in_negative:
            out["lyrics_negative"] = ""
        return out

    def _tokenize(self, clip, caption, lyrics, metas, language, seed, gen_codes, lm_params,
                  negatives):
        return clip.tokenize(caption, lyrics=lyrics, seed=int(seed),
                             generate_audio_codes=bool(gen_codes), language=language,
                             bpm=metas["bpm"], duration=metas["duration"],
                             timesignature=metas["timesignature"], keyscale=metas["keyscale"],
                             **lm_params, **negatives)

    # ------------------------------------------------------------------------------------- codes
    def _sectional_codes(self, clip, rows, base_caption, lyrics, metas, language, seed, guidance,
                         negative_tags, lyrics_in_negative, lm_params, lines, notes):
        """Decode the plan one section at a time, each continuing the codes already written."""
        import comfy.model_management
        import comfy.text_encoders.ace15 as ace15

        te, lm = _lm_of(clip)
        audio_start = _audio_start_id()
        pad = int((getattr(lm, "special_tokens", None) or {}).get("pad", 151643))
        use_cfg = float(lm_params["cfg_scale"]) != 1.0
        if not use_cfg:
            notes.append("cfg_scale is 1.0, so the negative pass is skipped entirely and "
                         "'guidance' does nothing — the plan follows the captions unguided")
        te.reset_clip_options()
        device = clip.patcher.load_device
        te.set_clip_options({"execution_device": device})

        codes = []
        bar = _OnePlanBar(sum(int(r["codes"]) for r in rows))
        if not bar.ok:
            notes.append("comfy.utils has moved ProgressBar or model_trange, so the plan's progress "
                         "is reported one section at a time again")
        drop_graphs = _graph_dropper()
        if drop_graphs is None and _decode_is_graphed(lm):
            notes.append("comfy captures a CUDA graph for the LM's decode step, but "
                         "comfy.model_prefetch.cleanup_prefetch_queues has moved — so the graph "
                         "cannot be dropped between sections and it will be replayed against a KV "
                         "cache that no longer exists. If this run dies on a 'scatter gather kernel "
                         "index out of bounds' assert, that is why: start ComfyUI with "
                         "--disable-cuda-graphs until this node catches up.")
        for i, row in enumerate(rows):
            caption = _join_caption(base_caption, row["add"], row["extra"])
            if guidance == GUID_DELTA:
                # The negative is the SHARED caption, so what cfg_scale amplifies is everything this
                # section adds to it: its voice line and its 4th column both. Worth being exact
                # about, because it cuts two ways — a duet's 'with gritty male backing harmonies'
                # gets guided, which is what you want, and so does an arrangement note like 'drums
                # drop out', which rides a dial named after the voice. A section that adds nothing
                # has no delta and falls back to core behaviour unless negative_tags gives it
                # something to push on.
                neg = (_join_caption(base_caption, negative_tags)
                       if (row["add"] or negative_tags) else None)
            elif guidance == GUID_TAGS:
                neg = negative_tags or None
            else:
                neg = None
            tk = self._tokenize(clip, caption, lyrics, metas, language, int(seed) + i, False,
                                lm_params, self._negatives(neg, metas, lyrics_in_negative))
            ids = _pair_ids(_ids_of(tk, "lm_prompt"), _ids_of(tk, "lm_prompt_negative"),
                            codes, pad, audio_start, use_cfg)
            # The loop reads the uncond row as `logits[1:2]`, so a cfg above 1.0 with only one
            # sequence would be doing arithmetic on an empty tensor. Can't happen from the
            # tokenizer, which always builds both prompts — but it must not fail silently.
            eff_cfg = float(lm_params["cfg_scale"]) if len(ids) == 2 else 1.0
            if len(ids) == 1 and use_cfg:
                notes.append(f"{row['label']}: the tokenizer returned no negative prompt, so this "
                             f"section ran unguided at cfg_scale 1.0")
            n = int(row["codes"])
            clip.load_model(tk)
            t0 = time.perf_counter()
            with comfy.model_management.cuda_device_context(device), bar:
                got = ace15.sample_manual_loop_no_classes(
                    lm, ids, cfg_scale=eff_cfg,
                    temperature=float(lm_params["temperature"]), top_p=float(lm_params["top_p"]),
                    top_k=(int(lm_params["top_k"]) or None), min_p=float(lm_params["min_p"]),
                    seed=int(seed) + i, min_tokens=n, max_new_tokens=n)
            secs = time.perf_counter() - t0
            # The decode graph comfy just captured points at the KV cache THIS call allocated; the
            # next section allocates its own, so the graph has to go with it. See `_graph_dropper`.
            if drop_graphs is not None:
                drop_graphs()
            if len(got) < n:
                notes.append(f"{row['label']}: the LM stopped after {len(got)} of {n} codes — the "
                             f"rest was padded, so the tail of that section has no plan")
                got = list(got) + [PAD_CODE] * (n - len(got))
            codes.extend(got[:n])
            bar.section_done(n)
            lines.append(f"  {row['label']}: {n} codes ({row['seconds']:.1f} s) in "
                         f"{_format_elapsed(secs, 'auto')} · continued from code {len(codes) - n}"
                         + ("" if neg is None else " · guided"))
        bar.close()
        return codes

    # --------------------------------------------------------------------------------------- run
    def run(self, clip, tags, lyrics, seed, bpm, duration, timesignature, language, keyscale,
            guidance, negative_tags, cast_in_caption, lyrics_in_negative, generate_audio_codes,
            cfg_scale, temperature, top_p, top_k, min_p, verbose=True, duet_mode=DUET_LEAD,
            plan=None, **kwargs):
        import node_helpers

        notes, lines, timeline = [], [], []
        tags = (tags or "").strip()
        lyrics = (lyrics or "").strip()
        negative_tags = (negative_tags or "").strip()
        lm_params = {"cfg_scale": float(cfg_scale), "temperature": float(temperature),
                     "top_p": float(top_p), "top_k": int(top_k), "min_p": float(min_p)}
        beats_per_bar = _beats(timesignature)
        language, lang_note = _language(language)
        keyscale, key_note = _keyscale(keyscale)
        notes.extend(n for n in (lang_note, key_note) if n)
        voices = _voices_in_order(kwargs)
        roster, roster_notes = _roster(voices)
        notes.extend(roster_notes)

        rows, plan_notes = _parse_plan(plan, bpm, beats_per_bar)
        notes.extend(plan_notes)
        for row in rows:
            row.update(_resolve_voice(row["voice_raw"], roster))
            row["cast_extra"] = []
            # A row naming several singers cannot be sung by several singers. Free text is left
            # alone — somebody who typed a whole description into the cell meant it literally.
            if len(row["names"]) > 1 and not row["verbatim"] and duet_mode != DUET_BLEND:
                col = _collapse_duet(row["names"], roster, duet_mode)
                if col:
                    row["add"] = col["add"]
                    row["cast_extra"] = col["others"]
                    notes.append(f"{row['label']}: {col['note']} Per-line markers inside the "
                                 f"section in the LYRICS are the only way to have them alternate "
                                 f"instead — see 'Siren Score'.")
            if row["unknown"] and roster:
                notes.append(f"{row['label']}: {', '.join(repr(u) for u in row['unknown'])} is not "
                             f"a wired voice — the whole cell was used as plain text instead. "
                             f"Roster: {', '.join(sorted({v['name'] for v in voices if v.get('name')})) or '(empty)'}")
            for s in row["silent"]:
                notes.append(f"{row['label']}: {s} has no 'voice_tags' on their Character Card, so "
                             f"this section gets no vocal description at all")
            if abs(row["seconds"] - row["asked_seconds"]) > 1e-6:
                lines.append(f"  {row['label']}: {row['asked_seconds']:.3f} s snapped to "
                             f"{row['seconds']:.1f} s (the plan runs at {CODES_HZ:.0f} Hz)")

        if rows and not generate_audio_codes:
            notes.append(f"'generate_audio_codes' is OFF, so there is no plan to build — the "
                         f"{len(rows)} section(s) were ignored and the whole song gets one caption. "
                         f"Turn it on, or clear the plan.")
            rows = []

        # The cast summary goes only into the GLOBAL caption (cross-attention), never into a
        # section's — a section's own line is what makes its part of the plan different.
        #
        # `cast_extra` is why the voices a duet collapsed are listed here too. Built from the section
        # captions alone, this list only ever held singers who lead somewhere, so a member who
        # appears exclusively in duets reached the model nowhere at all — and per `cast_in_caption`
        # this list is the only route by which a second timbre can colour a section's frames.
        cast = []
        for row in rows:
            for frag in [row["add"]] + list(row.get("cast_extra") or []):
                if frag and frag not in cast:
                    cast.append(frag)
        global_caption = _join_caption(tags, *(cast if (cast_in_caption and rows) else []))

        if rows:
            total_codes = sum(int(r["codes"]) for r in rows)
            seconds = total_codes / CODES_HZ
        else:
            seconds = float(duration)
            total_codes = None
        metas = self._metas(bpm, seconds, timesignature, keyscale)
        if not rows and float(duration) <= 0:
            raise RuntimeError("[Siren Cast] with an empty plan, 'duration' is the song length — "
                               "it cannot be 0. Type a duration, or write a plan.")

        t_all = time.perf_counter()
        if rows:
            codes = self._sectional_codes(clip, rows, tags, lyrics, metas, language, seed,
                                          guidance, negative_tags, lyrics_in_negative, lm_params,
                                          lines, notes)
            tk = self._tokenize(clip, global_caption, lyrics, metas, language, seed, False,
                                lm_params, {})
            cond = clip.encode_from_tokens_scheduled(tk)
            cond = node_helpers.conditioning_set_values(cond, {"audio_codes": [codes]})
            src = f"{len(rows)} section(s), {len(codes)} codes assembled"
        else:
            # No plan: comfy's own path end to end, with the negatives it never passes. The one
            # honest way to A/B the guidance fix on its own.
            if guidance == GUID_DELTA:
                if negative_tags.strip():
                    neg = _join_caption(tags, negative_tags)
                    notes.append("'voice delta' guidance needs a plan to have a delta — with no "
                                 "plan the negative is the caption plus 'negative_tags'")
                else:
                    neg = None
                    notes.append("'voice delta' guidance needs a plan (or 'negative_tags') to have "
                                 "anything to guide against — cfg_scale is back to guiding the "
                                 "metas only, exactly like the core node")
            elif guidance == GUID_TAGS:
                neg = negative_tags if negative_tags.strip() else None
                if neg is None:
                    notes.append("'negative tags' guidance is selected but 'negative_tags' is "
                                 "empty — cfg_scale is guiding the metas only, like the core node")
            else:
                neg = None
            tk = self._tokenize(clip, tags, lyrics, metas, language, seed, generate_audio_codes,
                                lm_params, self._negatives(neg, metas, lyrics_in_negative))
            cond = clip.encode_from_tokens_scheduled(tk)
            codes = (cond[0][1].get("audio_codes") or [[]])[0] if cond else []
            src = (f"one caption, {len(codes)} codes from the LM" if generate_audio_codes
                   else "one caption, no plan (audio codes off)")
        elapsed = _format_elapsed(time.perf_counter() - t_all, "auto")

        head = (f"Siren Cast — {src} · {_mmss(seconds)} ({seconds:.1f} s) · "
                f"{len(voices)} voice(s) wired · seed {int(seed)} · {guidance} · "
                f"cfg_scale {lm_params['cfg_scale']} · {elapsed}")
        at = 0.0
        for row in rows:
            timeline.append(f"{_mmss(at)} → {_mmss(at + row['seconds'])}  {row['label']:<14} "
                            f"{'+ ' + row['add'] if row['add'] else '(no vocal)'}")
            at += row["seconds"]
        timeline_txt = "\n".join(timeline) or f"0:00.0 → {_mmss(seconds)}  whole song, one caption"

        report = "\n".join([head] + lines + ["  " + t for t in timeline_txt.splitlines()]
                           + [f"  ⚠ {w}" for w in notes])
        if verbose:
            print("[Siren Cast] " + report.replace("\n", "\n[Siren Cast] "))

        params = {"length": f"{seconds:.2f} s ({len(codes)} codes @ {CODES_HZ:.0f} Hz)",
                  "seed": int(seed), "guidance": guidance, "cfg_scale": lm_params["cfg_scale"],
                  "temperature": lm_params["temperature"], "top_p": lm_params["top_p"],
                  "top_k": lm_params["top_k"], "min_p": lm_params["min_p"],
                  "bpm": int(bpm), "timesignature": int(timesignature), "keyscale": keyscale,
                  "cast": "; ".join(f"{r['label']}: {', '.join(r['names']) or r['add'] or '-'}"
                                     for r in rows) or "one caption, no plan",
                  "time": elapsed}
        gen_extra = json.dumps([{"class_type": "Siren Cast", "ord": 1, "params": params}],
                               ensure_ascii=False)
        return (cond, float(seconds), timeline_txt, report, gen_extra)



NODE_CLASS_MAPPINGS = {"KinburgSirenCast": KinburgSirenCast}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSirenCast": "Siren Cast (Voice Plan) 🧜"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
