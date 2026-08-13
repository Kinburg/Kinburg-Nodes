"""Siren Cast: the plan block, the roster, and the two prompts that must differ in ONE thing.

None of this needs torch or a model — the arithmetic (5 Hz plan against a 25 fps latent), the
parsing and the negative-prompt assembly are the parts that can be wrong silently, so they are the
parts that get pinned here. What is NOT covered, and cannot be without the model: whether a guided
plan actually sings the right verse in the right voice.
"""
import contextlib
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, comfy_on_path, fake_package, load_module  # noqa: E402

fake_package("kn", "context", "timer", "util", "siren")
load_module("kn.util.anytype", "util/anytype.py")
load_module("kn.context.character_card", "context/character_card.py")
load_module("kn.timer.timer_nodes", "timer/timer_nodes.py")
cast = load_module("kn.siren.cast", "siren/cast.py")

check = Checker()
Cast = cast.KinburgSirenCast

# ------------------------------------------------------------------------------ length parsing
for text, want in (("24", 24.0), ("24s", 24.0), (" 24 seconds ", 24.0), ("1:04", 64.0),
                   ("0:07.5", 7.5), ("8 bars", 16.0), ("8b", 16.0), ("2,5", 2.5)):
    got, err = cast._parse_length(text, 120, 4)
    check(f"length {text!r} → {want}", got == want, f"got {got} {err}")

check("bars without a bpm are refused, not guessed",
      cast._parse_length("8 bars", 0, 4)[0] is None)
check("bars follow the time signature",
      cast._parse_length("8 bars", 120, 3)[0] == 12.0)
check("nonsense is refused with a hint",
      cast._parse_length("soon", 120, 4)[0] is None
      and "8 bars" in cast._parse_length("soon", 120, 4)[1])

# ----------------------------------------------------------------------------------- plan block
PLAN = """
# state of the band, written by the LLM
section | voice       | length
--------|-------------|-------
Intro   | -           | 8
Verse 1 | Alex        | 24
Chorus  | Nina        | 8 bars
Bridge  | Nina + Alex | 0:12 | drums drop out
Verse 2 | Mike        | 24
Outro   | female, airy, wordless | 4
oops    | Alex        | soon
no pipes here
"""
rows, notes = cast._parse_plan(PLAN, 120, 4)
check("six good rows out of the block, two dropped", len(rows) == 6, [r["label"] for r in rows])
check("the comment, the header and the table rule went quietly",
      not any(w in " ".join(notes).lower() for w in ("section", "state of the band", "---")), notes)
check("the unreadable length is reported once, by label",
      len([n for n in notes if "oops" in n]) == 1, notes)
check("a line with no '|' is reported too",
      any("no pipes here" in n for n in notes), notes)
check("'8 bars' became 16.0 s", rows[2]["seconds"] == 16.0)
check("'0:12' became 12.0 s", rows[3]["seconds"] == 12.0)
check("the 4th cell is kept for the caption", rows[3]["extra"] == "drums drop out")
check("labels survive", [r["label"] for r in rows][:3] == ["Intro", "Verse 1", "Chorus"])
check("an empty plan is empty, not an error", cast._parse_plan("", 120, 4) == ([], []))

# The END terminator: what the plan grammar requires the LLM to finish with, and the thing that
# stops a runaway from reaching the node as forty bogus sections.
ended, enotes = cast._parse_plan(
    "Intro | - | 4\nVerse 1 | Alex | 8\nChorus | Nina | 8\nEND\n"
    "Outro | - | 4\nOutro | - | 4\nи ещё немного болтовни\n", 120, 4)
check("END ends the table", len(ended) == 3, [r["label"] for r in ended])
check("...and everything after it is dropped in silence, not warned about", enotes == [], enotes)
for word in ("END", "end", " End. ", "[END]", "конец"):
    check(f"{word!r} is recognised as the terminator",
          cast._parse_plan(f"A | - | 4\n{word}\nB | - | 4\n", 120, 4)[0].__len__() == 1)
check("a section actually called 'End' is not lost — a row has pipes, the terminator does not",
      len(cast._parse_plan("End | Alex | 4\n", 120, 4)[0]) == 1)

# --------------------------------------------------- the three values that arrive as text, not enums
# `timesignature`, `language` and `keyscale` used to be dropdowns, and a combo input cannot accept the
# STRING a text parser hands it — which is the whole reason the song config could not be wired
# straight in. Plain fields now, with the slips a config-writing LLM actually makes corrected.
check("beats per bar out of anything with a digit in it",
      [cast._beats(x) for x in ("4", 4, " 4/4 ", "3", "6", "", None)] == [4, 4, 4, 3, 6, 4, 4],
      [cast._beats(x) for x in ("4", 4, " 4/4 ", "3", "6", "", None)])
check("a good language code passes through untouched, with nothing to report",
      cast._language("uk") == ("uk", "") and cast._language("yue") == ("yue", ""))
check("...and the classic slips are corrected, out loud — Ukrainian is 'uk', not 'ua'",
      cast._language("ua")[0] == "uk" and "read as 'uk'" in cast._language("ua")[1]
      and cast._language("cn")[0] == "zh" and cast._language("jp")[0] == "ja")
check("...and a code AceStep has never heard of falls back rather than lying quietly",
      cast._language("xx") == ("en", cast._language("xx")[1]) and cast._language("xx")[1])
check("every corrected language actually exists in AceStep's list",
      all(v in cast.LANGUAGES for v in cast._LANG_FIX.values()),
      [v for v in cast._LANG_FIX.values() if v not in cast.LANGUAGES])
check("a key in the list passes through", cast._keyscale("A minor") == ("A minor", ""))
for text, want in (("c# minor", "C# minor"), ("C sharp minor", "C# minor"),
                   ("Db major", "Db major"), ("d flat major", "Db major"),
                   ("Am", "A minor"), ("C", "C major"), ("g maj", "G major")):
    got, note = cast._keyscale(text)
    check(f"key {text!r} reads as {want!r}", got == want and (note or text == want), (got, note))
check("a key the list cannot express falls back and says so, rather than poisoning the metas",
      cast._keyscale("H moll") == ("C major", cast._keyscale("H moll")[1])
      and "cannot express" in cast._keyscale("H moll")[1])
check("every key the parser can emit is a value the tokenizer will accept",
      all(cast._keyscale(f"{r} {q}")[0] in cast.KEYSCALES
          for q in ("major", "minor") for r in ("C", "C#", "Db", "F#", "Bb", "A")))


# A row is stored in whole codes, and the codes have to tile the latent EXACTLY: the model
# compares `audio_codes.shape[1] * 5` against the 25 fps latent length and pads if it is short.
odd, _ = cast._parse_plan("Verse | Alex | 7.13", 120, 4)
check("an odd length snaps to whole codes", odd[0]["codes"] == 36 and odd[0]["seconds"] == 7.2,
      odd[0])
total = sum(r["codes"] for r in rows)
seconds = total / cast.CODES_HZ
check("the assembled plan tiles the latent with nothing left over",
      total * 5 == round(seconds * 25), (total, seconds, round(seconds * 25)))
check("...and the seconds output lands on the 0.2 s grid",
      abs(seconds * 5 - round(seconds * 5)) < 1e-9, seconds)

# --------------------------------------------------------------------------------------- roster
ALEX = {"name": "Alex Kin", "tags": "male lead vocal, raspy baritone, close-mic", "notes": ""}
NINA = {"name": "Nina", "tags": "female lead vocal, airy alto", "notes": "avoids anything high"}
BOB = {"name": "Bob", "tags": "", "notes": "plays bass, never sings"}
roster, rnotes = cast._roster([ALEX, NINA, BOB])
check("a member is reachable by full name", roster["alex kin"] is ALEX)
check("...and by first name, which is what a plan row types", roster["alex"] is ALEX)
check("an unnamed voice is refused with a reason",
      any("no name" in n for n in cast._roster([{"tags": "x"}])[1]))
check("a name clash is reported, later wins",
      cast._roster([ALEX, {"name": "Alex Kin", "tags": "other"}])[0]["alex kin"]["tags"] == "other"
      and any("two wired voices" in n for n in cast._roster([ALEX, {"name": "Alex Kin"}])[1]))

r = cast._resolve_voice("Nina + Alex", roster)
check("'+' resolves both, in order", r["names"] == ["Nina", "Alex Kin"], r)
check("their tags are joined, not the names",
      "airy alto" in r["add"] and "raspy baritone" in r["add"] and "Nina" not in r["add"])
check("a resolved row is not verbatim", not r["verbatim"] and not r["unknown"])

r = cast._resolve_voice("female, airy, wordless", roster)
check("free text is used whole — NOT split on its commas",
      r["add"] == "female, airy, wordless" and r["verbatim"], r)

r = cast._resolve_voice("Nina + backing choir", roster)
check("a half-matching row falls back to text and names the miss",
      r["verbatim"] and r["unknown"] == ["backing choir"] and r["names"] == ["Nina"], r)

for silent in ("-", "", "  ", "instrumental", "None"):
    r = cast._resolve_voice(silent, roster)
    check(f"{silent!r} means no vocal", r["add"] == "" and not r["verbatim"], r)

r = cast._resolve_voice("Bob", roster)
check("a member with no voice_tags contributes nothing, and is reported",
      r["add"] == "" and r["silent"] == ["Bob"], r)

# -------------------------------------------------------------------------------- caption delta
BASE = "Warm 90s alt-rock, live drums, tape saturation"
pos = cast._join_caption(BASE, ALEX["tags"], "")
neg = cast._join_caption(BASE, "")
check("no addition leaves the caption untouched", cast._join_caption(BASE) == BASE)
check("the negative is a strict prefix of the positive", pos.startswith(neg), (neg, pos))
check("the ONLY difference is the voice line",
      pos == neg + ". " + ALEX["tags"] + ".", pos)
check("a 4th-cell note stacks on after the voice",
      cast._join_caption(BASE, NINA["tags"], "drums drop out").endswith("drums drop out."))
check("an addition is never glued onto a sentence without a stop",
      ". " in cast._join_caption("Ends without a stop", "x"))
check("...and a stop that is already there is not doubled",
      ".." not in cast._join_caption("Ends with a stop.", "x"))

# --------------------------------------------------------------------- the two prompts, negatives
METAS = {"bpm": 120, "duration": 111.6, "timesignature": 4, "keyscale": "A minor"}
neg_kw = Cast._negatives("some caption", METAS, True)
check("every meta is mirrored into the negative",
      set(neg_kw) == {"caption_negative", "bpm_negative", "duration_negative",
                      "timesignature_negative", "keyscale_negative"}, sorted(neg_kw))
check("the negative duration is the ceiled INT the positive think-block will carry",
      neg_kw["duration_negative"] == 112 and isinstance(neg_kw["duration_negative"], int),
      neg_kw["duration_negative"])
check("lyrics stay in the negative by default", "lyrics_negative" not in neg_kw)
check("...and can be dropped to guide them too",
      Cast._negatives("c", METAS, False)["lyrics_negative"] == "")
check("core mode passes no negatives at all — comfy's own behaviour",
      Cast._negatives(None, METAS, True) == {})
check("dropping the lyrics works without a negative caption too",
      Cast._negatives(None, METAS, False) == {"lyrics_negative": ""})

# ------------------------------------------------------------------------------- id continuation
ids = cast._pair_ids([1, 2, 3, 4], [9, 9], prefix=[7, 8], pad=0, audio_start=1000, use_cfg=True)
check("cond and uncond both come back", len(ids) == 2)
check("the short one is front-padded to equal length", len(ids[0]) == len(ids[1]) == 6, ids)
check("the padding went in FRONT, so the codes line up in both",
      ids[1][:2] == [0, 0] and ids[0][-2:] == ids[1][-2:] == [1007, 1008], ids)
check("the prompt is untouched apart from the padding", ids[0][:4] == [1, 2, 3, 4])
check("cfg 1.0 asks for one sequence and no padding",
      cast._pair_ids([1, 2], [9], [], 0, 1000, False) == [[1, 2]])
check("an empty prefix is the first section — nothing appended",
      cast._pair_ids([1, 2], [3, 4], [], 0, 1000, True) == [[1, 2], [3, 4]])
check("the audio-code offset is read off comfy's own loop signature (or falls back)",
      cast._audio_start_id() == 151669, cast._audio_start_id())

# --------------------------------------------------------------------------------- wiring sanity
check("voice slots sort numerically, so the 10th member doesn't jump the queue",
      [v["name"] for v in cast._voices_in_order(
          {"voice_10": {"name": "ten"}, "voice_2": {"name": "two"}, "voice_1": {"name": "one"},
           "seed": 3, "voice_3": None})] == ["one", "two", "ten"])

req = set(Cast.INPUT_TYPES()["required"])
opt = set(Cast.INPUT_TYPES()["optional"])
params = set(inspect.signature(Cast.run).parameters)
check("every required input is a run() parameter", req <= params, sorted(req - params))
check("the optional voice slots arrive through **kwargs",
      "kwargs" in params and not ({k for k in opt if k.startswith("voice_")} & params),
      sorted({k for k in opt if k.startswith("voice_")} & params))
check("...while the optional plan is a named parameter defaulting to nothing",
      inspect.signature(Cast.run).parameters["plan"].default is None)
check("the node declares as many outputs as it names",
      len(Cast.RETURN_TYPES) == len(Cast.RETURN_NAMES) == 5)
check("'seconds' comes out as a FLOAT for Empty Ace Step 1.5 Latent Audio",
      Cast.RETURN_TYPES[Cast.RETURN_NAMES.index("seconds")] == "FLOAT")

# The card is the one place a voice is described; Siren Cast must read exactly that shape.
CardMod = sys.modules["kn.context.character_card"]
card, voice = CardMod.CharacterCard().run(
    name="Nina", gender="female", voice_tags="female lead vocal, airy alto",
    voice_notes="never goes above E5")
check("the card carries the voice for the LLM",
      "- Voice (music tags): female lead vocal, airy alto" in card and "- Voice: never" in card,
      card)
check("...and hands the music side the voice plus the gender Siren Score needs — and no looks",
      voice == {"name": "Nina", "tags": "female lead vocal, airy alto",
                "notes": "never goes above E5", "gender": "female"}, voice)
check("a card resolves against the roster it just filled",
      cast._resolve_voice("Nina", cast._roster([voice])[0])["add"] == voice["tags"])
check("a card with no voice fields still emits a usable (empty) voice",
      CardMod.CharacterCard().run(name="Bob", gender="male")[1] ==
      {"name": "Bob", "tags": "", "notes": "", "gender": "male"})

# ---------------------------------------------------- the sectional decode, against a stubbed LM
# The novel part of this node is that section k is decoded as a CONTINUATION: the codes already
# written are appended to the prompt as the assistant's output so far. Nothing about that is
# visible in the audio until a model runs, but the ids handed to the sampling loop are exactly
# checkable — so they are checked here, with comfy's loop swapped for a recorder.
comfy_on_path()
import comfy.model_management as mm            # noqa: E402
import comfy.text_encoders.ace15 as ace15      # noqa: E402

CALLS = []
AUDIO_START = cast._audio_start_id()


def _expect_error(fn):
    try:
        fn()
    except Exception as e:
        return e
    return None


def _recorder(model, ids=None, cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=None, min_p=0.0,
              seed=1, min_tokens=1, max_new_tokens=2048, **kw):
    CALLS.append({"ids": [list(s) for s in ids], "seed": seed, "n": max_new_tokens,
                  "min_tokens": min_tokens, "cfg_scale": cfg_scale, "top_k": top_k})
    # Codes 100, 101, … so a later call's prefix is recognisable on sight.
    return list(range(100, 100 + max_new_tokens))


ace15.sample_manual_loop_no_classes = _recorder
mm.cuda_device_context = contextlib.nullcontext


class _FakeLM:
    special_tokens = {"pad": 7}


class _FakeTE:
    lm_model = "qwen3_2b"

    def __init__(self):
        self.qwen3_2b = _FakeLM()
        self.opts = []

    def reset_clip_options(self):
        self.opts.append("reset")

    def set_clip_options(self, o):
        self.opts.append(o)


class _FakeClip:
    """Just enough CLIP. Prompt ids encode the length of the text they came from, so a check can
    tell one section's caption from another's — and the positive from the negative."""

    def __init__(self):
        self.cond_stage_model = _FakeTE()
        self.patcher = type("P", (), {"load_device": "cpu"})()
        self.seen = []

    def tokenize(self, text, **kw):
        self.seen.append(dict(kw, text=text))
        return {"lm_prompt": [[(1000 + len(text), 1.0)]],
                "lm_prompt_negative": [[(2000 + len(kw.get("caption_negative") or ""), 1.0)]],
                "lm_metadata": {"min_tokens": 10}}

    def load_model(self, tokens=None):
        return None

    def encode_from_tokens_scheduled(self, tokens):
        return [["<cond tensor>", {"pooled_output": None}]]


SMALL = "Intro | -    | 4\nVerse 1 | Alex | 8\nChorus | Nina | 4\n"
ARGS = dict(tags=BASE, lyrics="[Verse 1 - Alex]\nla la", seed=5, bpm=120, timesignature="4",
            language="en", keyscale="A minor", negative_tags="", cast_in_caption=True,
            lyrics_in_negative=True, generate_audio_codes=True, cfg_scale=2.0, temperature=0.85,
            top_p=0.9, top_k=0, min_p=0.0, verbose=False)

clip = _FakeClip()
cond, seconds, timeline, report, gen = Cast().run(
    clip=clip, plan=SMALL, duration=0.0, guidance=cast.GUID_DELTA,
    voice_1=ALEX, voice_2=NINA, **ARGS)

check("one LM call per section", len(CALLS) == 3, len(CALLS))
check("each call asks for exactly that section's codes", [c["n"] for c in CALLS] == [20, 40, 20],
      [c["n"] for c in CALLS])
check("min_tokens == max_new_tokens, or comfy's KV cache is preallocated too small",
      all(c["min_tokens"] == c["n"] for c in CALLS), [(c["min_tokens"], c["n"]) for c in CALLS])
check("section 1 starts from the prompt alone", len(CALLS[0]["ids"][0]) == 1, CALLS[0]["ids"])
check("section 2 continues from 20 codes", len(CALLS[1]["ids"][0]) == 1 + 20)
check("section 3 continues from all 60", len(CALLS[2]["ids"][0]) == 1 + 60)
check("the appended codes carry the audio-token offset",
      CALLS[1]["ids"][0][1:4] == [100 + AUDIO_START, 101 + AUDIO_START, 102 + AUDIO_START],
      CALLS[1]["ids"][0][1:4])
check("cond and uncond continue from the SAME codes",
      CALLS[2]["ids"][0][1:] == CALLS[2]["ids"][1][1:])
check("each section draws from seed + its index, so editing one leaves the others alone",
      [c["seed"] for c in CALLS] == [5, 6, 7], [c["seed"] for c in CALLS])
check("top_k 0 reaches the loop as None (its own 'off' value)",
      all(c["top_k"] is None for c in CALLS))

pos_lens = [c["ids"][0][0] - 1000 for c in CALLS]
neg_lens = [c["ids"][1][0] - 2000 for c in CALLS]
check("a voiced section's caption is longer than the bare one",
      pos_lens[0] == len(BASE) and pos_lens[1] > pos_lens[0], pos_lens)
check("voice delta: the negative IS the bare caption for a voiced section",
      neg_lens[1] == len(BASE) and neg_lens[2] == len(BASE), neg_lens)
check("...and a section with no voice has no delta, so it falls back to core behaviour",
      neg_lens[0] == 0, neg_lens)
check("every section's LM pass has audio codes switched off — this node assembles them itself",
      all(s["generate_audio_codes"] is False for s in clip.seen))

final = clip.seen[-1]
check("the final encode uses the GLOBAL caption, with the cast listed on it",
      ALEX["tags"] in final["text"] and NINA["tags"] in final["text"], final["text"])
check("the final encode carries the plan's own total duration, not the widget",
      final["duration"] == 16.0 and seconds == 16.0, (final["duration"], seconds))
check("the whole plan came out as audio_codes on the conditioning",
      len(cond[0][1]["audio_codes"][0]) == 80, len(cond[0][1]["audio_codes"][0]))
check("...and it is the concatenation of the three sections, in order",
      cond[0][1]["audio_codes"][0][:3] == [100, 101, 102]
      and cond[0][1]["audio_codes"][0][20:23] == [100, 101, 102])
check("the timeline reads in m:ss so a section can be typed into Siren Section",
      timeline.splitlines()[1].startswith("0:04.0 → 0:12.0"), timeline.splitlines())
check("gen_extra_info records who sang what", "Verse 1: Alex Kin" in gen, gen)

# cast_in_caption off: the global caption must be the user's tags, untouched.
clip2 = _FakeClip()
Cast().run(clip=clip2, plan=SMALL, duration=0.0, guidance=cast.GUID_DELTA, voice_1=ALEX,
           voice_2=NINA, **{**ARGS, "cast_in_caption": False})
check("cast_in_caption off leaves the global caption alone", clip2.seen[-1]["text"] == BASE,
      clip2.seen[-1]["text"])

# No plan: comfy's own path, one encode, no custom loop at all.
before = len(CALLS)
clip3 = _FakeClip()
_, secs3, tl3, rep3, _ = Cast().run(clip=clip3, plan="", duration=30.0,
                                    guidance=cast.GUID_TAGS,
                                    **{**ARGS, "negative_tags": "spoken word, off-key"})
check("with no plan the sectional decoder never runs", len(CALLS) == before)
check("...it is one encode, with the codes left to comfy",
      len(clip3.seen) == 1 and clip3.seen[0]["generate_audio_codes"] is True)
check("...the duration widget is what sets the length then", secs3 == 30.0)
check("...and the negative caption is still passed, which the core node never does",
      clip3.seen[0]["caption_negative"] == "spoken word, off-key"
      and clip3.seen[0]["duration_negative"] == 30, clip3.seen[0].get("caption_negative"))
check("an empty plan and a zero duration is refused, not silently 0 s",
      isinstance(_expect_error(lambda: Cast().run(clip=_FakeClip(), plan="", duration=0.0,
                                                  guidance=cast.GUID_CORE, **ARGS)), RuntimeError))
# ------------------------------------------------- the built-in grammar and the parser must agree
# The "Siren Voice Plan (table)" GBNF preset exists so an LLM can write the plan. Whether llama.cpp
# accepts the grammar can only be checked with the runtime loaded; what IS checkable here is that a
# table the grammar is able to emit parses cleanly — the two ends drifting apart is the real risk.
gstore = load_module("kn.grammar_presets.store", "grammar_presets/store.py")
check("the plan grammar is in the preset list",
      "Siren Voice Plan (table)" in gstore.list_names(), gstore.list_names())
GRAMMAR_OUTPUT = ("Intro | - | 4 bars\nVerse 1 | Alex | 8 bars\nPre-Chorus | Nina | 4 bars\n"
                  "Chorus | Nina + Alex | 8 bars\nBridge | Мария | 4 bars\nOutro | - | 2 bars\n"
                  "END\n")
grows, gnotes = cast._parse_plan(GRAMMAR_OUTPUT, 120, 4)
check("everything the grammar can emit parses, with nothing dropped",
      len(grows) == 6 and gnotes == [], (len(grows), gnotes))
check("the grammar bounds the row count, so a model that won't stop still stops",
      "row{3,16}" in gstore.SIREN_PLAN, gstore.SIREN_PLAN.splitlines()[0])
check("...and it requires the END line the parser knows about",
      '"END"' in gstore.SIREN_PLAN
      and "end" in [ln.split("::=")[0].strip() for ln in gstore.SIREN_PLAN.splitlines()])
check("a hyphenated label is not mistaken for a table rule",
      grows[2]["label"] == "Pre-Chorus")
check("a two-word member name is expressible — the rule allows spaces inside a name",
      "name ::= word" in gstore.SIREN_PLAN and '(" " word)' in gstore.SIREN_PLAN,
      [ln for ln in gstore.SIREN_PLAN.splitlines() if ln.startswith("name ")])
TWO_WORD = "Intro | Gru BNik | 8 bars\nVerse 1 | Keen Burg | 8 bars\nChorus | Keen Burg + Gru BNik | 8 bars\nEND\n"
trows, tnotes = cast._parse_plan(TWO_WORD, 145, 4)
check("...and a two-word name parses and resolves, solo and in a duet",
      len(trows) == 3 and tnotes == []
      and [cast._resolve_voice(r["voice_raw"], cast._roster(
          [{"name": "Gru BNik", "tags": "male, aggressive"},
           {"name": "Keen Burg", "tags": "female, airy"}])[0])["names"] for r in trows]
      == [["Gru BNik"], ["Keen Burg"], ["Keen Burg", "Gru BNik"]], (len(trows), tnotes))
check("a Cyrillic name survives the roster lookup",
      cast._resolve_voice("Мария", cast._roster([{"name": "Мария", "tags": "alto"}])[0])["add"]
      == "alto")
check("bar counts land on whole codes at the grammar's own units",
      all(float(r["codes"]).is_integer() for r in grows)
      and sum(r["codes"] for r in grows) == 300, sum(r["codes"] for r in grows))


# The config grammar pins keyscale / language / timesignature to what Siren Cast's COMBOs accept.
# That agreement is the point of pinning them, so it is checked against the node's own lists rather
# than against a copy: a value the grammar can emit but the combo rejects is a dead workflow.
def _alts(grammar, rule):
    """The quoted literals of a one-line `rule ::= "a" | "b"` production."""
    for line in grammar.splitlines():
        if line.split("::=")[0].strip() == rule:
            return re.findall(r'"([^"]*)"', line.split("::=", 1)[1])
    return []


CONF = gstore.SIREN_CONFIG
roots = _alts(CONF, "keyroot")
langs = _alts(CONF, "language")
check("the config grammar is in the preset list too",
      "Siren Song Config (text)" in gstore.list_names(), gstore.list_names())
check("every key the grammar can write is a value the keyscale combo accepts",
      roots and all(f"{r} {q}" in cast.KEYSCALES for r in roots for q in ("major", "minor")),
      [f"{r} major" for r in roots if f"{r} major" not in cast.KEYSCALES])
check("...and it can write every one of them, all 34",
      len(roots) * 2 == len(cast.KEYSCALES), (len(roots) * 2, len(cast.KEYSCALES)))
check("every language it can write is in the language combo",
      langs and all(ln in cast.LANGUAGES for ln in langs),
      [ln for ln in langs if ln not in cast.LANGUAGES])
check("the 'ua' trap is unreachable, and 'uk' is there",
      "ua" not in langs and "uk" in langs)
check("the time signature stays inside the combo",
      set(_alts(CONF, "timesignature-line")) - {"timesignature: ", "\\n"} <= set(cast.TIMESIGS),
      _alts(CONF, "timesignature-line"))
check("no vocals line — per-section voices are the plan's job, not the caption's",
      "Vocals" not in CONF)
_inst = [ln for ln in CONF.splitlines() if ln.startswith("instruments-section ")][0]
check("the caption block is still separated from the metas by a blank line — the dropped vocals "
      "line is where that blank line used to live",
      _inst.rstrip().endswith('"\\n\\n"'), _inst)
check("a genre can contain digits, so '90s alt-rock' is writable",
      "0-9" in CONF.split("textline ::=", 1)[1].splitlines()[0],
      CONF.split("textline ::=", 1)[1].splitlines()[0])

check("the plan input is OPTIONAL — an unwired one means one caption for the whole song",
      "plan" in Cast.INPUT_TYPES()["optional"] and "plan" not in Cast.INPUT_TYPES()["required"])
clip4 = _FakeClip()
_, secs4, _, _, _ = Cast().run(clip=clip4, duration=42.0, guidance=cast.GUID_CORE, **ARGS)
check("...and the node runs with the argument absent entirely, not just empty",
      len(clip4.seen) == 1 and secs4 == 42.0, (len(clip4.seen), secs4))

# One progress bar for the whole plan. comfy's loop makes its own per call, so twelve sections used
# to mean twelve bars each restarting at zero, which reads as a stuck node.
import comfy.utils as _U                                                          # noqa: E402
_REAL_BAR, _REAL_TRANGE = _U.ProgressBar, _U.model_trange
_seen = []


class _RecBar(_REAL_BAR):
    def update_absolute(self, value, total=None, preview=None):
        _seen.append((value, total or self.total))


_U.ProgressBar = _RecBar
_bar = cast._OnePlanBar(30)              # three sections of ten codes
_tqs, _restored, _shimmed = [], [], []
for _ in range(3):
    with _bar:
        _shimmed.append(_U.ProgressBar is not _RecBar)
        _inner = _U.ProgressBar(10)      # what comfy's own loop constructs
        for _step in _U.model_trange(10, desc="LM sampling"):
            _inner.update_absolute(_step)
    _restored.append(_U.ProgressBar is _RecBar and _U.model_trange is _REAL_TRANGE)
    _tqs.append(id(_bar.tq))
    _bar.section_done(10)
_bar.close()
_U.ProgressBar = _REAL_BAR

check("the inner loop is handed a shim while a section decodes", all(_shimmed))
check("...and comfy.utils is put back after every one of them", all(_restored))
check("one console bar spans all three sections", len(set(_tqs)) == 1, _tqs)
check("the web bar counts 1..30 straight through, never restarting",
      [v for v, _ in _seen[:-1]] == list(range(1, 31)), _seen[:5])
check("...against one total, not a per-section one", {t for _, t in _seen} == {30})
check("and it is finished off at the end", _seen[-1] == (30, 30) and _bar.tq is None)
_saved_trange = _U.model_trange
del _U.model_trange                      # pretend comfy renamed it
_degraded = cast._OnePlanBar(10)
_U.model_trange = _saved_trange
with _degraded:
    _still_real = _U.ProgressBar is _REAL_BAR
_degraded.section_done(10)
_degraded.close()
check("if comfy ever moves one of those names, the swap is skipped rather than fatal",
      _degraded.ok is False and _still_real, (_degraded.ok, _still_real))

check("a plan with audio codes switched off is reported, not silently obeyed",
      "generate_audio_codes' is OFF" in Cast().run(
          clip=_FakeClip(), plan=SMALL, duration=12.0, guidance=cast.GUID_CORE, voice_1=ALEX,
          **{**ARGS, "generate_audio_codes": False})[3])

check.done()
