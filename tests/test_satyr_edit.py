"""Satyr Edit: a plan read into notes and written back without breaking it.

Three promises are pinned here, in falling order of how silently they could be broken.

**Nothing you did not touch changes.** Load a plan and save it unedited and the text must come back
byte for byte; edit one note and only the line that note lives on may differ. Anything looser and an
edit in the chorus quietly respells the verse.

**What is re-rendered is the exporter's own text.** The fixtures that matter most are written by
ComfyUI's SheetSage2 exporter itself — made-up songs pushed through `events_to_abc`, so the format is
the real one and the content is ours. Forcing every bar through our renderer must reproduce that text
byte for byte: same grouping, same chord restating, same `^^F`, same tie splits, same `Z` folds.

**A syllable stays a syllable.** One Vocal note is one sung syllable, so every read and every edit
that does not mean to change the words must keep the count of attacks exactly.
"""
import copy
import difflib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, comfy_on_path, fake_package, load_module  # noqa: E402

comfy_on_path()
fake_package("kn", "satyr", "siren", "context", "timer", "util")
load_module("kn.util.anytype", "util/anytype.py")
load_module("kn.context.character_card", "context/character_card.py")
load_module("kn.timer.timer_nodes", "timer/timer_nodes.py")
load_module("kn.siren.cast", "siren/cast.py")
load_module("kn.siren.score", "siren/score.py")
N = load_module("kn.satyr.notation", "satyr/notation.py")
B = load_module("kn.satyr.bands", "satyr/bands.py")
load_module("kn.satyr.rewrite", "satyr/rewrite.py")
load_module("kn.satyr.layout", "satyr/layout.py")
E = load_module("kn.satyr.edit", "satyr/edit.py")
load_module("kn.satyr.midi", "satyr/midi.py")
load_module("kn.satyr.trim", "satyr/trim.py")
Nodes = load_module("kn.satyr.nodes", "satyr/nodes.py")
from comfy.audio_encoders.sheetsage2_abc import events_to_abc  # noqa: E402

check = Checker()


# ------------------------------------------------------------------------------------ fixtures
def exported(chords=True, key_change=None, seed=0):
    """A made-up song in D# minor at 96 bpm, written out by ComfyUI's own SheetSage2 exporter.

    It carries what real plans carry: an instrumental intro, a low verse and a high chorus (two voice
    bands), a sung note held across a barline (a tie), an interlude where both voices rest (folded to
    `Z`), and optionally a key change — at a group boundary (`"group"`) or in the middle of a bar
    (`"inline"`, which the exporter writes as `[K:]`)."""
    rnd = random.Random(seed)
    beat = 60.0 / 96
    form = [("intro", 4, None), ("verse", 4, (60, 67)), ("interlude", 2, "rest"),
            ("chorus", 4, (69, 77)), ("outro", 2, None)]
    prog = ["D#:min", "B:maj", "F#:maj", "C#:maj"]
    events, bar = [], 0
    for label, bars, band in form:
        for b in range(bars):
            for q in range(4):
                time = (bar * 4 + q) * beat
                values = {"rhythm": {"meter": (4, 4), "eighth_position": q * 2}, "melody": []}
                if bar == 0 and q == 0:
                    values["key"] = "D#:minor"
                if b == 0 and q == 0:
                    values["structure"] = label
                if key_change == "group" and label == "chorus" and b == 0 and q == 0:
                    values["key"] = "E:minor"
                if key_change == "inline" and label == "chorus" and b == 1 and q == 2:
                    values["key"] = "E:minor"
                if chords and q in (0, 2) and band != "rest":
                    values["chord"] = prog[(bar + q // 2) % 4]
                if band is None and q % 2 == 0:
                    values["melody"].append({"pitch": rnd.choice([51, 54, 58, 63]),
                                             "end_time": time + 2 * beat, "track": 1})
                elif isinstance(band, tuple):
                    held = b % 2 == 1 and q == 3          # held into the next bar: a tie
                    after_hold = b % 2 == 0 and b > 0 and q == 0
                    if not after_hold:
                        values["melody"].append({"pitch": rnd.randint(*band),
                                                 "end_time": time + (2 if held else 1) * beat, "track": 0})
                    if q == 0:
                        values["melody"].append({"pitch": rnd.choice([39, 42, 46]),
                                                 "end_time": time + 4 * beat, "track": 1})
                events.append({"time": time, "values": values})
            bar += 1
    return events_to_abc(events, bar * 4 * beat, melody_only=not chords)


HAND = """X:1
T:
M:2/4
L:1/16
Q:1/4=77
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:D#m
% intro
V: Vocal
"D#m"z8|"D#m"z4A2A2|
V: Ins
Z|D,2D2F3E-|
% verse
V: Vocal
"B"a2g2f2e2-|"B"e2f4z2|
V: Ins
Z2|
"""

# A melisma (a tie onto a different pitch), a body that opens without a label, two labels in a row,
# no trailing newline, and Vocal/Ins bar counts that disagree.
ODD = """X:1
T:odd
M:4/4
L:1/8
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
V: Vocal
"C"C2D2E2-D2|"G"G4z4|
V: Ins
C,8|G,,8|
% intro
% verse
V: Vocal
"Am"A2B2c4|
V: Ins
Z2|"""

FIXTURES = {
    "exported": exported(),
    "exported, melody only": exported(chords=False),
    "exported, key change at a group": exported(key_change="group", seed=1),
    "exported, key change inside a bar": exported(key_change="inline", seed=2),
    "hand-written": HAND,
    "odd": ODD,
    "Windows line ends": HAND.replace("\n", "\r\n"),
}
EXPORTED = [name for name in FIXTURES if name.startswith("exported")]


def forced(model):
    """The same model with every origin wiped — so `save` must render every bar itself."""
    m = copy.deepcopy(model)
    for b in m["bars"]:
        b["o"] = None
    for s in m["sections"]:
        s["o"] = None
    return m


def notes_of(model):
    return [(n["v"], n["t"], n["d"], n["p"], n["j"]) for n in model["notes"]]


def shape(model):
    return [(b["w"], tuple(b["m"])) for b in model["bars"]], model["keys"]


def attacks(text):
    """Syllables by `notation`'s own count, over the whole Vocal voice at once: a note held from one
    group into the next is one syllable, and counting line by line would make it two."""
    score = N.parse(text)
    return N.attacks("".join(score.lines[g.vocal] for s in score.sections for g in s.groups
                             if g.vocal is not None), score.key)


def syllables(model):
    return sum(1 for n in model["notes"] if n["v"] == 0 and not n["j"])


def changed_lines(a, b):
    return [line for line in difflib.ndiff(a.splitlines(), b.splitlines()) if line[:2] in ("- ", "+ ")]


# ------------------------------------------------------------------------------- the fixtures are real
song = FIXTURES["exported"]
check("the exporter fixture has a tie across a barline", "-|" in song)
check("...an empty stretch folded to Z", "Z2|" in FIXTURES["exported, melody only"])
check("...a key change at a group", "\nK:Em\n" in FIXTURES["exported, key change at a group"])
check("...and one inside a bar", "[K:Em]" in FIXTURES["exported, key change inside a bar"])
check("...and the exporter's own key-relative spelling", "^^" in song or "=E" in song)

# --------------------------------------------------------------------------------- round trips
for name, text in FIXTURES.items():
    model = E.load(text)
    same, _ = E.save(text, model)
    check(f"[{name}] saved unedited, the text comes back byte for byte", same == text,
          changed_lines(text, same)[:4])
    again, _ = E.save(text, forced(model))
    back = E.load(again)
    check(f"[{name}] re-rendering every bar keeps every note", notes_of(back) == notes_of(model),
          [x for x in zip(notes_of(back), notes_of(model)) if x[0] != x[1]][:3])
    check(f"[{name}] ...every chord", back["chords"] == model["chords"])
    check(f"[{name}] ...every bar's width and meter, and every key change", shape(back) == shape(model),
          (shape(back)[1], shape(model)[1]))
    check(f"[{name}] the model sings as many syllables as the plan has attacks",
          syllables(model) == attacks(text), (syllables(model), attacks(text)))
    check(f"[{name}] ...and as many as Satyr Read's phrases add up to",
          sum(p.notes for p in B.read(N.parse(text))) == syllables(model),
          (sum(p.notes for p in B.read(N.parse(text))), syllables(model)))
    check(f"[{name}] a re-rendered plan is well-formed", N.problems(N.parse(again)) == [],
          N.problems(N.parse(again)))
    if name in EXPORTED:
        check(f"[{name}] re-rendering every bar reproduces the exporter's text byte for byte",
              again == text, changed_lines(text, again)[:6])

# ------------------------------------------------------------------------------------ reading
model = E.load(song)
check("bars carry their original index", [b["o"] for b in model["bars"]] == list(range(len(model["bars"]))))
check("16 bars of 4/4 at L:1/16, 16 units each",
      len(model["bars"]) == 16 and all(b["w"] == 16 and b["m"] == [4, 4] for b in model["bars"]))
check("five sections, in order", [s["label"] for s in model["sections"]]
      == ["intro", "verse", "interlude", "chorus", "outro"])
check("a held note is ONE note, not two", any(n["t"] % 16 + n["d"] > 16 for n in model["notes"] if n["v"] == 0))
check("the length is the bars' length at the plan's tempo", abs(model["seconds"] - 16 * 2.5) < 1e-6,
      model["seconds"])
check("two voice bands are found in a low verse and a high chorus", model["bands"] is not None
      and model["bands"]["low"][1] < model["bands"]["split"] < model["bands"]["high"][0], model["bands"])
check("the section vocabulary goes to the editor", "chorus" in model["labels"] and "pre-chorus" in model["labels"])

hand = E.load(HAND)
check("chords are kept as changes, the barline restating dropped",
      [(c["t"], c["c"]) for c in hand["chords"]] == [(0, "D#m"), (16, "B")], hand["chords"])
vocal = [(n["t"], n["d"], n["p"]) for n in hand["notes"] if n["v"] == 0]
check("an A in D# minor is A# — the signature is applied", vocal[0][2] == 70, vocal)
check("e2- tied over the barline is one note of 4 units — E# in this key, so MIDI 77", (22, 4, 77) in vocal,
      vocal)
odd = E.load(ODD)
check("a tie onto a new pitch is a joined note: one syllable on two pitches",
      [(n["p"], n["j"]) for n in odd["notes"] if n["v"] == 0][2:4] == [(64, 0), (62, 1)], odd["notes"][:5])
check("a body with no label opens an unlabelled section", odd["sections"][0]["label"] == "")
check("two labels in a row: the first holds no bars", [(s["label"], s["n"]) for s in odd["sections"]]
      == [("", 2), ("intro", 0), ("verse", 2)], odd["sections"])
check("disagreeing bar counts are reported, not hidden", any("Ins 2" in w for w in odd["warnings"]),
      odd["warnings"])


# -------------------------------------------------------------------------------------- edits
def where(model, section):
    """(first bar, bar count, first unit, units) of the first section with that label."""
    at = 0
    for s in model["sections"]:
        if s["label"] == section:
            start = sum(b["w"] for b in model["bars"][:at])
            return at, s["n"], start, sum(b["w"] for b in model["bars"][at:at + s["n"]])
        at += s["n"]
    raise KeyError(section)


def delete_section(model, label):
    """What the editor does for 'delete section': its bars, notes and chords go, a note held into it
    stops at its edge, and everything after moves up."""
    m = copy.deepcopy(model)
    first, count, t0, width = where(m, label)
    del m["bars"][first:first + count]
    m["sections"] = [s for s in m["sections"] if not (s["label"] == label and s["n"] == count)]
    kept = []
    for n in m["notes"]:
        if t0 <= n["t"] < t0 + width:
            continue
        if n["t"] < t0 < n["t"] + n["d"]:
            n["d"] = t0 - n["t"]
        if n["t"] >= t0 + width:
            n["t"] -= width
        kept.append(n)
    m["notes"] = kept
    for field in ("chords", "keys"):
        m[field] = [dict(c, t=c["t"] - width if c["t"] >= t0 + width else c["t"]) for c in m[field]
                    if not t0 <= c["t"] < t0 + width or c["t"] == 0]
    return m


# One note in the chorus, a semitone up.
m = copy.deepcopy(model)
_, _, t0, _ = where(m, "chorus")
target = next(n for n in m["notes"] if n["v"] == 0 and n["t"] >= t0 and n["t"] % 16 + n["d"] <= 16)
target["p"] += 1
out, report = E.save(song, m)
diff = changed_lines(song, out)
check("one note moved: exactly one line changes, the Vocal line it lives on",
      len(diff) == 2 and diff[0].startswith("- ") and "|" in diff[0], diff)
check("...and reading it back finds the note where it was put",
      any(n["t"] == target["t"] and n["p"] == target["p"] for n in E.load(out)["notes"]))
check("the report counts what was kept and what was re-rendered",
      any("re-rendered" in line for line in report) and "4 bar(s) re-rendered" in report[1], report)

# The intro, deleted: nothing is held across its edges, so every other line survives as it was.
m = delete_section(model, "intro")
out, report = E.save(song, m)
after = E.load(out)
check("deleting a section drops its bars", len(after["bars"]) == len(model["bars"]) - 4)
check("...and not one sung syllable", syllables(after) == syllables(model), (syllables(after), syllables(model)))
check("...and leaves both voices the same length everywhere", N.problems(N.parse(out)) == [],
      N.problems(N.parse(out)))
check("...and every other section's lines are untouched — the diff is deletions only",
      all(line.startswith("- ") for line in changed_lines(song, out)), changed_lines(song, out)[:6])

# The interlude, deleted: the verse's last note was held into it, so that one line loses its tie.
m = delete_section(model, "interlude")
out, report = E.save(song, m)
added = [line for line in changed_lines(song, out) if line.startswith("+ ")]
removed = [line for line in changed_lines(song, out) if line.startswith("- ")]
check("a note held into a deleted section stops at its edge: one line changes, losing only its tie",
      len(added) == 1 and any(r[2:] == added[0][2:-1] + "-|" for r in removed), (added, removed[:2]))
check("...still not one syllable lost", syllables(E.load(out)) == syllables(model))

# Two bars before the chorus, new and empty.
m = copy.deepcopy(model)
first, _, t0, _ = where(m, "chorus")
m["bars"][first:first] = [{"w": 16, "m": [4, 4], "o": None}, {"w": 16, "m": [4, 4], "o": None}]
for s in m["sections"]:
    if s["label"] == "interlude":
        s["n"] += 2
m["notes"] = [dict(n, t=n["t"] + 32 if n["t"] >= t0 else n["t"]) for n in m["notes"]]
m["chords"] = [dict(c, t=c["t"] + 32 if c["t"] >= t0 else c["t"]) for c in m["chords"]]
m["keys"] = [dict(k, t=k["t"] + 32 if k["t"] >= t0 else k["t"]) for k in m["keys"]]
out, _ = E.save(song, m)
grown = E.load(out)
check("two empty bars inserted, well-formed", N.problems(N.parse(out)) == []
      and len(grown["bars"]) == len(model["bars"]) + 2)
check("...silent in both voices", not any(t0 <= n["t"] < t0 + 32 for n in grown["notes"]))
check("...and the sung bars after them did not move a note",
      notes_of(grown)[-10:] == [(v, t + 32, d, p, j) for v, t, d, p, j in notes_of(model)[-10:]])

# The whole song, up a whole tone: D# minor becomes F minor.
m = copy.deepcopy(model)
for n in m["notes"]:
    n["p"] += 2
for k in m["keys"]:
    k["k"] = "Fm"
names = {"D#m": "Fm", "B": "C#", "F#": "G#", "C#": "D#"}
for c in m["chords"]:
    c["c"] = names.get(c["c"], c["c"])
out, _ = E.save(song, m)
up = E.load(out)
check("transposed: every pitch is two semitones higher",
      [n["p"] for n in up["notes"]] == [n["p"] + 2 for n in model["notes"]])
check("...the header names the new key", "\nK:Fm\n" in out)
check("...the chords moved with it", '"Fm"' in out and '"D#m"' not in out)
check("...and it reads back as the same rhythm", [(n["t"], n["d"]) for n in up["notes"]]
      == [(n["t"], n["d"]) for n in model["notes"]])

# Tempo only.
m = copy.deepcopy(model)
m["bpm"] = 120
out, _ = E.save(song, m)
check("a tempo change touches the Q: line and nothing else",
      changed_lines(song, out) == ["- Q:1/4=96", "+ Q:1/4=120"], changed_lines(song, out))

# A chord changed in the middle of a bar.
m = copy.deepcopy(model)
_, _, t0, _ = where(m, "verse")
m["chords"].append({"t": t0 + 4, "c": "G#m"})
out, _ = E.save(song, m)
diff = changed_lines(song, out)
check("a chord added mid-bar changes one Vocal line", len(diff) == 2, diff)
check("...and is written where it was put", '"G#m"' in diff[1], diff)

# The held note, moved: both bars it spans are re-rendered and the tie survives.
m = copy.deepcopy(model)
held = next(n for n in m["notes"] if n["v"] == 0 and n["t"] % 16 + n["d"] > 16)
held["p"] -= 1
out, _ = E.save(song, m)
check("a held note moved keeps its tie and its single syllable",
      syllables(E.load(out)) == syllables(model) and "-|" in out)

# The melisma survives a full re-render.
again, _ = E.save(ODD, forced(odd))
check("a joined note is written back as a tie onto the new pitch", "E2-D2" in again,
      [line for line in again.splitlines() if "C2D2" in line])

# -------------------------------------------------------------------------------------- cleaning
m = copy.deepcopy(hand)
vocal = [n for n in m["notes"] if n["v"] == 0]
vocal[0]["d"] += 6                       # now runs over the next note
m["notes"].append({"v": 1, "t": 28, "d": 30, "p": 50, "j": 0})   # past the end
m["notes"].append({"v": 1, "t": 2, "d": 1, "p": 50, "j": 1})     # joined to nothing
out, report = E.save(HAND, m)
check("an overlap left by a drag is cut, and said", any("overlapping" in r for r in report), report)
check("a note past the end is shortened, and said", any("past the end" in r for r in report), report)
check("a join with nothing before it becomes a new syllable, and said", any("nothing to join" in r for r in report),
      report)
check("...and the result is still well-formed", N.problems(N.parse(out)) == [], N.problems(N.parse(out)))

# ------------------------------------------------------------------------------------- the words
# Lyrics written to fit the exporter song exactly: the verse gets as many syllables as it has Vocal
# onsets, the chorus likewise, the intro and the interlude none.
KEEN = {"name": "Keen", "tags": "melodic female vocal"}
BURG = {"name": "Burg", "tags": "gritty male vocal"}
_, n_verse, t_verse, w_verse = where(model, "verse")
_, n_chorus, t_chorus, w_chorus = where(model, "chorus")
sung = [n for n in model["notes"] if n["v"] == 0 and not n["j"]]
in_verse = sum(1 for n in sung if t_verse <= n["t"] < t_verse + w_verse)
in_chorus = sum(1 for n in sung if t_chorus <= n["t"] < t_chorus + w_chorus)


def filled(count, word="сирени"):
    """A lyric line of exactly `count` syllables, in three-syllable words where it can."""
    words = [word] * (count // 3) + ["та"] * (count % 3)
    return " ".join(words)


LYRICS = (f"[Intro]\n\n[Verse - Burg]\n{filled(in_verse)}\n\n[Interlude]\n\n"
          f"[Chorus - Keen]\n{filled(in_chorus, 'серденько')}\n")
laid = E.words(song, model, LYRICS, [KEEN, BURG])
by_label = {s["label"]: s for s in laid["sections"]}
check("the words come back one entry per plan section", len(laid["sections"]) == len(model["sections"]),
      [s["label"] for s in laid["sections"]])
check("the verse carries the verse, as many syllables as it has notes",
      by_label["verse"]["voice"] == "Burg" and len(by_label["verse"]["syllables"]) == in_verse == by_label["verse"]["holds"],
      (by_label["verse"]["voice"], len(by_label["verse"]["syllables"]), in_verse))
check("...and the chorus the chorus", by_label["chorus"]["voice"] == "Keen" and len(by_label["chorus"]["syllables"]) == in_chorus,
      (by_label["chorus"]["voice"], len(by_label["chorus"]["syllables"]), in_chorus))
check("wordless sections take no words", not by_label["intro"]["syllables"] and not by_label["interlude"]["syllables"])
first = by_label["verse"]["syllables"][:3]
check("syllables are cut the way they are counted, and know their word",
      [s["s"] for s in first] == ["си", "ре", "ни"] and all(s["word"] == "сирени" for s in first)
      and first[0]["first"] and first[2]["last"] and not first[1]["first"], first)
check("...and the line they belong to", laid["lines"][first[0]["line"]] == filled(in_verse))
check("...and the block and the run of one voice they are sung in",
      laid["blocks"][first[0]["block"]]["label"] == "Verse"
      and by_label["chorus"]["syllables"][0]["unit"] == first[0]["unit"] + 1,
      (first[0].get("block"), first[0].get("unit"), by_label["chorus"]["syllables"][0].get("unit")))
check("the singers are named for their bands", laid["singers"] == {"high": ["Keen"], "low": ["Burg"]}, laid["singers"])
check("a section knows the band its singer should be in, and the one it is in",
      (by_label["verse"]["band"], by_label["verse"]["was"], by_label["chorus"]["band"], by_label["chorus"]["was"])
      == ("low", "low", "high", "high"),
      [(s["label"], s["band"], s["was"]) for s in laid["sections"]])
swapped = E.words(song, model, LYRICS.replace("Verse - Burg", "Verse - Keen").replace("Chorus - Keen", "Chorus - Burg"),
                  [KEEN, BURG])
check("markers that contradict the plan's registers are visible as such",
      [(s["band"], s["was"]) for s in swapped["sections"] if s["label"] in ("verse", "chorus")] == [("high", "low"), ("low", "high")],
      [(s["label"], s["band"], s["was"]) for s in swapped["sections"]])
check("what Satyr Score would report comes along", isinstance(laid["findings"], list) and laid["findings"],
      laid["findings"])
bare = E.words(song, model, "just some words with no markers", [])
check("lyrics with no markers lay nothing, and say so instead of raising",
      not any(s["syllables"] for s in bare["sections"]) and bare["singers"] == {"high": [], "low": []})
edited_words = E.words(song, delete_section(model, "intro"), LYRICS, [KEEN, BURG])
check("the words are laid onto the edit, not the plan it came from",
      len(edited_words["sections"]) == len(model["sections"]) - 1)
check("the lyric's blocks are listed for the editor to pin, wordless ones too",
      [b["label"] for b in laid["blocks"]] == ["Intro", "Verse", "Break", "Chorus"]
      and laid["blocks"][-1]["voice"] == "Keen" and laid["blocks"][1]["syllables"] == in_verse,
      [(b["label"], b["voice"], b["syllables"]) for b in laid["blocks"]])

# Pins — what the matcher cannot know. A bridge the plan wrote no melody for: by counts its words
# belong to some neighbour; pinned, the interlude carries them and the rest fits around it.
BRIDGED = (f"[Verse - Burg]\n{filled(in_verse)}\n\n[Bridge - Burg]\n{filled(9)}\n\n"
           f"[Chorus - Keen]\n{filled(in_chorus, 'серденько')}\n")
pinned = copy.deepcopy(model)
inter = [s["label"] for s in model["sections"]].index("interlude")
pinned["sections"][inter]["sings"] = 1
bridged = E.words(song, pinned, BRIDGED, [KEEN, BURG])
by_index = bridged["sections"]
check("a section pinned to a block carries exactly that block", by_index[inter]["blocks"] == ["Bridge"]
      and by_index[inter]["pinned"] and by_index[inter]["carries"] == [1], by_index[inter])
check("...and the matcher fits the others around it",
      by_index[inter - 1]["blocks"] == ["Verse"] and by_index[inter + 1]["blocks"] == ["Chorus"],
      [(s["label"], s["blocks"]) for s in by_index])
quiet = copy.deepcopy(model)
verse_at = [s["label"] for s in model["sections"]].index("verse")
quiet["sections"][verse_at]["sings"] = -1
silent = E.words(song, quiet, LYRICS, [KEEN, BURG])["sections"]
check("a section pinned to nothing carries nothing, and no word is lost",
      not silent[verse_at]["syllables"] and sum(len(s["syllables"]) for s in silent) == in_verse + in_chorus,
      [(s["label"], len(s["syllables"])) for s in silent])
crossed = copy.deepcopy(model)
crossed["sections"][verse_at]["sings"] = 1
crossed["sections"][inter + 1]["sings"] = 0          # runs backwards against the pin before it
crossed_words = E.words(song, crossed, BRIDGED, [KEEN, BURG])["sections"]
check("a pin running backwards against an earlier one is ignored",
      crossed_words[inter + 1]["pinned"] is False and crossed_words[verse_at]["pinned"] is True
      and crossed_words[verse_at]["blocks"] == ["Bridge"], [(s["label"], s["blocks"], s["pinned"]) for s in crossed_words])
check("pins live in the editor only: the plan written is the same", E.save(song, pinned)[0] == song)
# A pickup: the hand plan's intro ends on two notes after a beat of rest, running straight into the
# verse's downbeat. They sing the verse's first words, so they count for it — Satyr Score's rule —
# and the editor is told how many notes before its barline a section's words start.
LONE = E.words(HAND, E.load(HAND), "[Verse - Burg]\nти дивишся в стелю б'ється й\n", [KEEN, BURG])["sections"]
shown = [s["s"] for sec in LONE for s in sec["syllables"]]
check("a word with no vowel is shown with the word it is sung with, and counts no syllable of its own",
      shown == ["ти", "ди", "виш", "ся", "в сте", "лю", "б'єт", "ься й"], shown)
UP = E.words(HAND, E.load(HAND), "[Intro]\n\n[Verse - Burg]\n" + filled(7), [KEEN, BURG])["sections"]
check("a pickup counts for the section it opens, and the editor is told how many notes it is",
      [(s["holds"], s["pickup"]) for s in UP] == [(0, 0), (7, 2)] and len(UP[1]["syllables"]) == 7,
      [(s["label"], s["holds"], s["pickup"], len(s["syllables"])) for s in UP])


# -------------------------------------------------------------------------------------- the node
node = Nodes.KinburgSatyrEdit()
kept = json.dumps({"edited": HAND, "upstream": song,
                   "editedReport": "2 group(s) kept as written, 2 bar(s) re-rendered"})
nothing = json.dumps({"edited": "", "upstream": song})
check("not frozen: the upstream is asked for", node.check_lazy_status(use_edited=False, plan_state=kept) == ["abc"])
check("frozen with an edit saved: the upstream is not run at all",
      node.check_lazy_status(use_edited=True, plan_state=kept) == [])
check("frozen with nothing saved: the upstream is still needed",
      node.check_lazy_status(use_edited=True, plan_state=nothing) == ["abc"])
check("a mangled carrier reads as empty rather than raising",
      node.check_lazy_status(use_edited=True, plan_state="{oops") == ["abc"])
out = node.run(abc=song, use_edited=False, plan_state=kept)
check("not frozen: the upstream plan goes out unchanged", out["result"][0] == song)
check("...and is sent back for the editor to open", out["ui"]["satyr_plan"][0]["upstream"] == song)
check("...with its length in seconds", abs(out["result"][2] - 40.0) < 1e-6, out["result"][2])
out = node.run(abc=None, use_edited=True, plan_state=kept)
check("frozen: the edited plan goes out", out["result"][0] == HAND)
check("...the report says the upstream was not run", "not run" in out["result"][1], out["result"][1])
check("...and what the last save kept and re-rendered",
      "last save: 2 group(s) kept as written, 2 bar(s) re-rendered" in out["result"][1], out["result"][1])
check("...and nothing is sent back, so the node keeps the upstream it has",
      out["ui"]["satyr_plan"][0]["upstream"] is None)
out = node.run(abc=song, use_edited=False, plan_state=kept, lyrics=LYRICS, voice_1=KEEN, voice_2=BURG)
check("the lyrics and the voices go to the editor with the run",
      out["ui"]["satyr_plan"][0]["lyrics"] == LYRICS and out["ui"]["satyr_plan"][0]["voices"] == [KEEN, BURG])
check("...and the plan going out is not changed by them", out["result"][0] == song)
out = node.run(abc=song, use_edited=False, plan_state=kept)
check("unwired, they are sent as empty — so the editor forgets them too",
      out["ui"]["satyr_plan"][0]["lyrics"] == "" and out["ui"]["satyr_plan"][0]["voices"] == [])
out = node.run(abc=song, use_edited=True, plan_state=nothing)
check("frozen with nothing saved: the upstream goes out, and the report says why",
      out["result"][0] == song and "nothing has been saved" in out["result"][1], out["result"][1])
for label, kwargs in (("an empty input", {"abc": "", "use_edited": False, "plan_state": ""}),
                      ("a plan that is not ABC", {"abc": "hello", "use_edited": False, "plan_state": ""})):
    try:
        node.run(**kwargs)
        check(f"{label} is refused", False)
    except RuntimeError as e:
        check(f"{label} is refused in a sentence", str(e).startswith("[Satyr Edit]"), str(e))

for label, broken in (("no bars", dict(hand, bars=[], sections=[])),
                      ("sections that do not add up", dict(hand, sections=[{"label": "x", "n": 1, "o": None}])),
                      ("a pitch off the keyboard", dict(hand, notes=[{"v": 0, "t": 0, "d": 1, "p": 300, "j": 0}])),
                      ("a tempo of zero", dict(hand, bpm=0))):
    try:
        E.save(HAND, broken)
        check(f"{label} is refused", False)
    except ValueError as e:
        check(f"{label} is refused with a reason", bool(str(e)), str(e))

check.done()
