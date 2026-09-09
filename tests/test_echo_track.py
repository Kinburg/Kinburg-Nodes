"""Echo's windowing and bookkeeping — the half of alignment that decides the problem.

Two claims. **A window is bounded by the section, not by the song**, which is what stops a
three-minute track from costing three minutes of quadratic attention — the same wall the author hit
from the other side, with a video decoded into a frame batch. And **the plan is a hint, not a
truth**: a block's corrected span is where its words turned out to be, and the disagreement with the
plan is the output rather than an error.

The aligner is not here. Word times are fabricated, because everything in this module is what
happens *around* the model.
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
T = load_module("kn.echo.track", "echo/track.py")

check = Checker()

PLAN = """Intro | - | 8 bars
Verse 1 | Nina | 16 bars
Chorus | Nina + Gru | 16 bars
Solo | - | 8 bars
Verse 2 | Gru | 16 bars"""

LYRICS = """[Intro]
(Distorted bassline, haunting guitar)

[Verse 1 - Nina]
Живий! Я живий, і кожен подих — мій
Лише пусті знаки

[Chorus - Nina]
Ой у лузі червона калина
[Gru]
Похилилася

[Verse 2 - Gru]
Ще не вмерла України"""

ROSTER = [{"name": "Nina"}, {"name": "Gru"}]
TOTAL = 180.0


# --------------------------------------------------------------- the plan carries the VOICE
rows, notes = T.plan_rows(PLAN)
check("every plan row is read", len(rows) == 5, [r["label"] for r in rows])
check("the voice column survives — this is what colouring rests on",
      [r["voice"] for r in rows][:3] == ["-", "Nina", "Nina + Gru"], [r["voice"] for r in rows])
check("bars become a weight", rows[1]["weight"] > rows[0]["weight"], rows[1]["weight"])

dur_rows, dur_notes = T.plan_rows("3.5, 4.0, 8")
check("Orpheus' bare seconds list is read too", len(dur_rows) == 3, dur_rows)
check("...and it is SAID that it carries no voices", any("voices" in n for n in dur_notes),
      dur_notes)
check("an empty plan is empty, not an error", T.plan_rows("") == ([], []))


# --------------------------------------------------------------------- proportions, not seconds
sp = T.spans(rows, TOTAL)
check("the plan is scaled onto the song", abs(sp[-1]["end"] - TOTAL) < 1e-9, sp[-1]["end"])
check("the first span starts at zero", sp[0]["start"] == 0.0)
check("the spans are contiguous",
      all(abs(sp[i]["end"] - sp[i + 1]["start"]) < 1e-9 for i in range(len(sp) - 1)))
# The nominal bpm cancels: a plan written entirely in bars gives the same spans at any tempo.
other = T.spans([dict(r, weight=r["weight"] * 3.7) for r in rows], TOTAL)
check("the nominal tempo cancels out",
      all(abs(a["end"] - b["end"]) < 1e-6 for a, b in zip(sp, other)))


# ------------------------------------------------------- the line survives its own tokenization
for text in ["Живий! Я живий, і кожен подих — мій", "— Ой, у лузі...", "Don't stop me now!",
             "  подвійний   пробіл  ", "(Живий!)"]:
    ws = __import__("sys").modules["kn.echo.translit"].words(text)
    chunks = T.display_chunks(text, ws)
    check(f"the chunks reassemble {text[:22]!r}", "".join(chunks) == text, "".join(chunks))
check("one chunk per word",
      len(T.display_chunks("а б в", __import__("sys").modules["kn.echo.translit"].words("а б в"))) == 3)


# ----------------------------------------------------------------------------- building the job
job, jnotes = T.build(LYRICS, PLAN, TOTAL, voices=ROSTER)
check("one block per plan row", len(job["blocks"]) == 5, len(job["blocks"]))
by_label = {b["label"]: b for b in job["blocks"]}
check("the instrumental rows have no lines",
      by_label["Intro"]["lines"] == [] and by_label["Solo"]["lines"] == [])
check("...but they are KEPT, so the corrected timeline covers the whole song",
      by_label["Solo"]["end"] > by_label["Solo"]["start"])
check("a sung row carries its lines", len(by_label["Verse 1"]["lines"]) == 2,
      by_label["Verse 1"]["lines"])
check("the language is decided once over the whole lyric", job["language"] == "uk", job["language"])

# The alternation `Siren Score` writes: one plan row, two voices, split by an inner `[Gru]` marker.
chorus = by_label["Chorus"]["lines"]
check("consecutive sub-sections stay in ONE block — they are consecutive audio",
      len(chorus) == 2, len(chorus))
check("...and each line keeps its own singer",
      [ln["voice"] for ln in chorus] == ["Nina", "Gru"], [ln["voice"] for ln in chorus])
check("a line with no inner marker inherits the plan row's voice",
      by_label["Verse 1"]["lines"][0]["voice"] == "Nina")

# A production note in round brackets before any sung line belongs to the caption, not the mouth.
check("a bracketed production note is not a sung line", by_label["Intro"]["lines"] == [])


# ------------------------------------------------------------------------------- the window
v1 = by_label["Verse 1"]
width = v1["end"] - v1["start"]
pad = max(T.SLACK_MIN, width * T.SLACK)
check("the window is the span plus slack on each side",
      abs(v1["window"][0] - (v1["start"] - pad)) < 1e-9 and
      abs(v1["window"][1] - (v1["end"] + pad)) < 1e-9, v1["window"])
check("the first block's window cannot start before the song does",
      job["blocks"][0]["window"][0] == 0.0, job["blocks"][0]["window"][0])
check("the last block's window cannot run past the end",
      job["blocks"][-1]["window"][1] == TOTAL, job["blocks"][-1]["window"][1])
# The claim the memory cost rests on: a window is bounded by its section, never by the song.
longest = max(b["window"][1] - b["window"][0] for b in job["blocks"])
check("no window is anywhere near the length of the song", longest < TOTAL * 0.45,
      f"{longest:.1f} s of {TOTAL:.0f}")


# ------------------------------------------------------------------------- what is said out loud
missing, mnotes = T.build(LYRICS, "Intro | - | 8 bars\nVerse 1 | Nina | 16 bars", TOTAL,
                          voices=ROSTER)
check("a lyric section with no plan row is reported, not silently dropped",
      any("Chorus" in n for n in mnotes), mnotes)
none_job, nnotes = T.build(LYRICS, "", TOTAL, voices=ROSTER)
check("with no plan the lyric is still spread over the song", len(none_job["blocks"]) == 4,
      len(none_job["blocks"]))
check("...and the wider windows are said to cost more", any("wide" in n for n in nnotes), nnotes)
check("a no-plan window really is wider",
      (none_job["blocks"][1]["window"][1] - none_job["blocks"][1]["window"][0]) >
      (v1["window"][1] - v1["window"][0]))
_, bad = T.build("no markers at all", PLAN, TOTAL)
check("a lyric with nothing to match is reported", any("nothing to align" in n for n in bad), bad)
_, digits = T.build("[Verse 1 - Nina]\nраз 2 три 4", PLAN, TOTAL)
check("an unspellable lyric is reported before a GPU is spent",
      any("alphabet" in n for n in digits), digits)


# ------------------------------------------------------------------ finishing: the times come back
def align_fake(job, per_word=0.4, skip_keys=()):
    """Lay every findable word end to end from its block's planned start."""
    for b in job["blocks"]:
        t = b["start"] + 1.0
        for ln in b["lines"]:
            for w in ln["words"]:
                if not w["key"] or w["key"] in skip_keys:
                    continue
                w["start"], w["end"], w["score"] = t, t + per_word, 0.8
                t += per_word
            t += 0.5
    return job


track = T.finish(align_fake(T.build(LYRICS, PLAN, TOTAL, voices=ROSTER)[0]))
check("every sung line is on the track", len(track["lines"]) == 5, len(track["lines"]))
check("the lines come out in time order",
      all(track["lines"][i]["start"] <= track["lines"][i + 1]["start"]
          for i in range(len(track["lines"]) - 1)))
check("a line's span covers its own words",
      all(ln["start"] == ln["words"][0]["start"] and ln["end"] == ln["words"][-1]["end"]
          for ln in track["lines"]))
check("every word has a drawable width",
      all(w["end"] > w["start"] for ln in track["lines"] for w in ln["words"]))
check("the mean confidence is reported", abs(track["score"] - 0.8) < 1e-9, track["score"])
check("the display text is still the author's", "Живий" in track["lines"][0]["text"])

corrected = {s["label"]: s for s in track["sections"]}
check("a sung block's span is where the WORDS are, not where the plan said",
      abs(corrected["Verse 1"]["start"] - (corrected["Verse 1"]["planned"][0] + 1.0)) < 1e-6,
      (corrected["Verse 1"]["start"], corrected["Verse 1"]["planned"]))
check("an instrumental block is flagged as one", corrected["Solo"]["sung"] is False)
# It does NOT keep the plan's position. Nothing measured an instrumental section, so once the sung
# ones around it have moved it is simply between them — which is both correct and the only thing
# anyone knows about it. On the author's real song five Break/Solo rows kept their planned places
# while the singing moved +25 s, and the corrected plan's overlap warning filled with their names.
check("...and it is placed BETWEEN the singing, not where the plan put it",
      corrected["Solo"]["start"] >= corrected["Chorus"]["end"] - 1e-6
      and corrected["Solo"]["end"] <= corrected["Verse 2"]["start"] + 1e-6,
      (corrected["Solo"]["start"], corrected["Solo"]["end"],
       corrected["Chorus"]["end"], corrected["Verse 2"]["start"]))
check("...while the plan's own guess is kept for the report to compare against",
      corrected["Solo"]["planned"] == (112.5, 135.0), corrected["Solo"]["planned"])

# A run of consecutive instrumental rows shares its gap in proportion to their planned lengths.
run = T.finish({"total": 100.0, "language": "uk", "blocks": [
    {"label": "V", "voice": "N", "start": 0.0, "end": 20.0, "window": (0.0, 20.0),
     "lines": [{"text": "a", "voice": "N", "label": "V",
                "words": [{"text": "a", "key": "a", "start": 5.0, "end": 10.0, "score": 0.9}]}]},
    {"label": "Break", "voice": "-", "start": 20.0, "end": 30.0, "window": (20.0, 30.0),
     "lines": []},
    {"label": "Solo", "voice": "-", "start": 30.0, "end": 60.0, "window": (30.0, 60.0),
     "lines": []},
    {"label": "V2", "voice": "N", "start": 60.0, "end": 100.0, "window": (60.0, 100.0),
     "lines": [{"text": "b", "voice": "N", "label": "V2",
                "words": [{"text": "b", "key": "b", "start": 70.0, "end": 90.0, "score": 0.9}]}]},
]})
gap = {s["label"]: (s["start"], s["end"]) for s in run["sections"]}
check("a run of instrumental rows fills exactly the gap between the singing",
      abs(gap["Break"][0] - 10.0) < 1e-6 and abs(gap["Solo"][1] - 70.0) < 1e-6, gap)
check("...and they are contiguous", abs(gap["Break"][1] - gap["Solo"][0]) < 1e-6, gap)
check("...shared in proportion to their planned lengths (10 s vs 30 s of a 60 s gap)",
      abs((gap["Break"][1] - gap["Break"][0]) - 15.0) < 1e-6, gap["Break"])

# The edges of the song: a leading instrumental starts at 0, a trailing one runs to the end.
edges = T.finish({"total": 60.0, "language": "uk", "blocks": [
    {"label": "Intro", "voice": "-", "start": 0.0, "end": 10.0, "window": (0.0, 10.0), "lines": []},
    {"label": "V", "voice": "N", "start": 10.0, "end": 40.0, "window": (10.0, 40.0),
     "lines": [{"text": "a", "voice": "N", "label": "V",
                "words": [{"text": "a", "key": "a", "start": 18.0, "end": 33.0, "score": 0.9}]}]},
    {"label": "Outro", "voice": "-", "start": 40.0, "end": 60.0, "window": (40.0, 60.0),
     "lines": []},
]})
edge = {s["label"]: (s["start"], s["end"]) for s in edges["sections"]}
check("a leading instrumental starts at the top of the song",
      edge["Intro"] == (0.0, 18.0), edge["Intro"])
check("a trailing one runs to the last sample", abs(edge["Outro"][1] - 60.0) < 1e-6, edge["Outro"])
check("a song with no singing at all does not divide by zero",
      T.finish({"total": 30.0, "language": "", "blocks": [
          {"label": "Solo", "voice": "-", "start": 0.0, "end": 30.0, "window": (0.0, 30.0),
           "lines": []}]})["sections"][0]["end"] == 30.0)
check("the quiet stretches are found — this is where an instrumental shot goes",
      len(track["quiet"]) >= 2, track["quiet"])

# A word the aligner was never asked about — a digit — must still be on screen and still lit.
carried = T.finish(align_fake(T.build("[Verse 1 - Nina]\nраз 2 три", PLAN, TOTAL)[0]))
ws = carried["lines"][0]["words"]
check("an unspellable word is carried between its neighbours",
      len(ws) == 3 and ws[0]["end"] <= ws[1]["start"] and ws[1]["end"] <= ws[2]["start"], ws)
check("...and is marked as carried rather than counted as found",
      ws[1].get("carried") is True and carried["lines"][0]["found"] == 2, carried["lines"][0])

# A whole line the model could not find at all still has to be drawable.
lost = T.finish(align_fake(T.build("[Verse 1 - Nina]\nраз два три", PLAN, TOTAL)[0],
                           skip_keys={"raz", "dva", "try"}))
ln = lost["lines"][0]
check("a line nobody found still gets a legal span", ln["end"] > ln["start"], (ln["start"], ln["end"]))
check("...with a confidence of zero, so it is visible in the report", ln["score"] == 0.0)

# ----------------------------------------------------- a repeated label is a repeated SECTION
# The bug this pins, found on the author's first real render: grouping the lyric's sections by
# label across the whole song put every chorus's words into the FIRST chorus's window. On screen
# that is three lines at once, minutes early, over whoever is really singing — and silence where
# the last chorus actually is. Adjacency, not name, is what makes a group.
REPEAT_PLAN = """Intro | - | 8 bars
Verse 1 | Nina | 16 bars
Chorus | Nina | 16 bars
Verse 2 | Gru | 16 bars
Chorus | Nina | 16 bars"""
REPEAT_LYRICS = """[Intro]
[Verse 1 - Nina]
Перший куплет
[Chorus - Nina]
Перший приспів
[Verse 2 - Gru]
Другий куплет
[Chorus - Nina]
Другий приспів"""
rep, rnotes = T.build(REPEAT_LYRICS, REPEAT_PLAN, TOTAL, voices=ROSTER)
got = [(b["label"], [ln["text"] for ln in b["lines"]]) for b in rep["blocks"]]
check("a chorus sung twice is two blocks, one line each",
      got == [("Intro", []), ("Verse 1", ["Перший куплет"]), ("Chorus", ["Перший приспів"]),
              ("Verse 2", ["Другий куплет"]), ("Chorus", ["Другий приспів"])], got)
check("the SECOND chorus is timed against the second half of the song",
      rep["blocks"][4]["start"] > TOTAL * 0.7, rep["blocks"][4]["start"])
check("nothing was said, because nothing went wrong", rnotes == [], rnotes)

# The alternation must still collapse: an inner `[Gru]` marker is a sub-section of the chorus that
# is open, not a new one — those lines are consecutive audio and align in one pass.
alt, _ = T.build("[Chorus - Nina]\nрядок Ніни\n[Gru]\nрядок Гру", "Chorus | Nina + Gru | 16 bars",
                 60.0, voices=ROSTER)
check("an inner voice marker stays inside its own block", len(alt["blocks"][0]["lines"]) == 2)
check("...with each line keeping its singer",
      [ln["voice"] for ln in alt["blocks"][0]["lines"]] == ["Nina", "Gru"],
      [ln["voice"] for ln in alt["blocks"][0]["lines"]])

# A count that does not match on the two sides is the failure that used to be silent.
short_plan, snotes = T.build(REPEAT_LYRICS, "Intro | - | 8 bars\nVerse 1 | Nina | 16 bars\n"
                                            "Chorus | Nina | 16 bars\nVerse 2 | Gru | 16 bars",
                             TOTAL, voices=ROSTER)
check("two sung choruses against one planned row is named as such",
      any("2 time(s) in the lyrics" in n or "1 time(s) in the plan" in n for n in snotes), snotes)


# ------------------------------------------------------------------ two sections at once = wrong
# What a misplacement looks like from the outside, and the only symptom a confidence score misses.
def line_at(start, end, section, label):
    return {"start": start, "end": end, "section": section, "label": label}


over = T.collisions([line_at(10.0, 14.0, 0, "Verse 1"), line_at(12.0, 16.0, 1, "Chorus")])
check("two sections overlapping is a collision", len(over) == 1 and abs(over[0]["seconds"] - 2.0) < 1e-9,
      over)
check("...named on both sides", over[0]["a"] == "Verse 1" and over[0]["b"] == "Chorus", over)
inside = T.collisions([line_at(10.0, 14.0, 0, "Chorus"), line_at(10.5, 13.0, 0, "Chorus")])
check("a backing echo under its own lead is NOT a collision", inside == [], inside)
apart = T.collisions([line_at(10.0, 14.0, 0, "A"), line_at(14.0, 18.0, 1, "B")])
check("lines that merely touch do not collide", apart == [], apart)
check("an empty track has no collisions", T.collisions([]) == [])
check("the finished track carries them", "collisions" in track, list(track))


# --------------------------------------------- anchoring: each section teaches the next one where
# Built from the author's real song. The plan and the performance ran apart PROGRESSIVELY — +1.5 s
# at the first verse, +17.7 s by the second — so the last section's error is the next one's best
# correction. And the floor is the half that matters: the Outro of that take was never sung, and
# without a floor its words were laid confidently on top of the chorus that WAS being sung there.
REAL = [                  # label, planned start/end, where the words were actually found start/end
    ("Verse 1", 12.80, 32.00, 14.32, 34.46),
    ("Pre-Chorus", 32.00, 44.80, 38.61, 53.96),
    ("Chorus", 44.80, 64.00, 55.95, 80.34),
    ("Verse 2", 64.00, 89.60, 81.66, 100.75),
    ("Bridge", 89.60, 102.40, 103.43, 116.66),
    ("Chorus", 108.80, 128.00, 117.21, 146.10),
]


def anchored_windows():
    """Each section's window, walked in order the way the aligner walks them."""
    out, drift, prev = [], 0.0, None
    for label, p0, p1, f0, f1 in REAL:
        lo, hi = T.anchor({"label": label, "start": p0, "end": p1}, prev, drift, T.SLACK, 179.2)
        out.append((label, lo, hi, f0, f1))
        drift, prev = f0 - p0, f1
    return out


def plain_windows():
    """What `build()` produces with anchoring off — the plan's span plus slack, and nothing else.
    Spelled out rather than taken from `anchor(…, None, 0)`, which is a different formula."""
    out = []
    for label, p0, p1, f0, f1 in REAL:
        pad = max(T.SLACK_MIN, (p1 - p0) * T.SLACK)
        out.append((label, p0 - pad, p1 + pad, f0, f1))
    return out


# The WHOLE span has to fit, not just the start: a window that holds a section's first word and not
# its last is what crams the rest against the edge, and that is what happened to five of these six.
missed = [(l, round(lo, 1), round(hi, 1), f0, f1)
          for l, lo, hi, f0, f1 in anchored_windows() if not (lo <= f0 and f1 <= hi)]
check("every section of the author's real song fits inside its anchored window", missed == [],
      missed)
plain = [l for l, lo, hi, f0, f1 in plain_windows() if not (lo <= f0 and f1 <= hi)]
check("...where the un-anchored windows fit only the first one", len(plain) == 5, plain)
check("...which is exactly the section that did NOT need the retry on that take",
      "Verse 1" not in plain, plain)

# The asymmetry that a first version got wrong: the drift GREW to +17.7 s and then shrank, so a
# window whose start had been pushed forward by the last measurement began after the Bridge it was
# looking for. Drift may only ever make a window bigger.
ahead = T.anchor({"label": "Bridge", "start": 89.6, "end": 102.4}, 100.75, 17.66, T.SLACK, 179.2)
check("a large forward drift never pushes a window's START past its section",
      ahead[0] <= 103.43, ahead)
check("...while it does extend the ceiling", ahead[1] >= 116.66, ahead)

# The floor, on the case it exists for: the Outro planned at 2:08 while the chorus before it sang
# on until 2:26. Its window must start AFTER the singing, not on top of it.
lo, hi = T.anchor({"label": "Outro", "start": 128.0, "end": 134.4}, 146.10, 8.41, T.SLACK, 179.2)
check("a section cannot be searched before the previous one stopped singing",
      lo >= 146.10 - T.ANCHOR_OVERLAP - 1e-9, lo)
check("...and its window is still big enough to hold its own words", hi - lo >= 6.4, (lo, hi))
check("...and stays inside the song", hi <= 179.2 + 1e-9, hi)

# A backing echo starts fractionally before its lead's last word ends, so the floor is soft by a
# fraction of a second rather than absolute.
lo2, _ = T.anchor({"label": "X", "start": 50.0, "end": 60.0}, 50.2, 0.0, T.SLACK, 179.2)
check("the floor allows the overlap a backing echo really has",
      abs(lo2 - (50.2 - T.ANCHOR_OVERLAP)) < 1e-9, lo2)

# Drift pointing backwards must not squeeze a window below what it has to hold.
lo3, hi3 = T.anchor({"label": "X", "start": 100.0, "end": 120.0}, None, -40.0, T.SLACK, 179.2)
check("a backwards drift cannot make a window too small for its own words",
      hi3 - lo3 >= 20.0, (lo3, hi3, hi3 - lo3))
pad0 = max(T.SLACK_MIN, 20.0 * T.SLACK)
bare = T.anchor({"label": "X", "start": 10.0, "end": 30.0}, None, 0.0, T.SLACK, 100.0)
check("with nothing to go on the window starts where the plan's own did",
      abs(bare[0] - (10.0 - pad0)) < 1e-9, bare)
check("...and is never shorter than the plan's own", bare[1] >= 30.0 + pad0 - 1e-9, bare)
check("a window never starts before the song does",
      T.anchor({"label": "X", "start": 1.0, "end": 5.0}, None, -50.0, T.SLACK, 100.0)[0] == 0.0)


# ------------------------------------------ a last word that ran into the silence after the line
# Seen on the author's render: the closing line of a verse was sung by 0:34.5 and stayed on screen
# until 0:39.9, over the top of the next section's first line. A block is aligned with a `*` star
# token at each end, but entering it costs something, and at a section's end it is often cheaper for
# the Viterbi path to hold the last word's final character across the instrumental than to pay for
# the star. So the last word swallows the gap.
def spoken(pairs):
    return [{"text": t, "start": a, "end": b} for t, a, b in pairs]


line_ok = spoken([("Місто ", 30.0, 30.8), ("мовчить, ", 30.8, 32.0), ("розпад.", 32.0, 33.2)])
check("an ordinary line is left alone", T._trim_tail(line_ok) == 0.0, line_ok[-1])

held = spoken([("Місто ", 30.0, 30.8), ("мовчить, ", 30.8, 32.0), ("розпад.", 32.0, 34.4)])
check("a genuinely held final note survives", T._trim_tail(held) == 0.0, held[-1]["end"])

stranded = spoken([("Місто ", 30.0, 30.8), ("мовчить, ", 30.8, 32.0),
                   ("приховуючи ", 32.0, 33.4), ("розпад.", 33.4, 39.46)])
cut = T._trim_tail(stranded)
check("a last word that ran on into the silence is pulled back", cut > 2.0, round(cut, 2))
check("...but not clipped to nothing", stranded[-1]["end"] - stranded[-1]["start"] >= T.TAIL_FLOOR,
      stranded[-1])
check("...and the trim is reported in seconds, so it can be checked by ear",
      isinstance(cut, float) and cut > 0)
check("the words before it are untouched",
      [w["end"] for w in stranded[:-1]] == [30.8, 32.0, 33.4], stranded)

# A one-word line has no pace of its own to measure against, so it is never trimmed: a subtitle
# that flashes is worse than one that lingers.
check("a one-word line is never trimmed", T._trim_tail(spoken([("Серце...", 10.0, 18.0)])) == 0.0)
check("a line whose body has no duration is not divided by",
      T._trim_tail(spoken([("a", 5.0, 5.0), ("b", 5.0, 12.0)])) == 0.0)

# End to end: the trim reaches the track, and the line's own span shrinks with it.
tail_track = T.finish({"total": 60.0, "language": "uk", "blocks": [
    {"label": "V", "voice": "N", "start": 0.0, "end": 60.0, "window": (0.0, 60.0),
     "lines": [{"text": "Місто мовчить, приховуючи розпад.", "voice": "N", "label": "V",
                "words": [{"text": t, "key": "x", "start": a, "end": b, "score": 0.9}
                          for t, a, b in [("Місто ", 30.0, 30.8), ("мовчить, ", 30.8, 32.0),
                                          ("приховуючи ", 32.0, 33.4), ("розпад.", 33.4, 39.46)]]}]},
]})
check("the shortened line ends earlier than it was aligned",
      tail_track["lines"][0]["end"] < 37.0, tail_track["lines"][0]["end"])
check("...and the track says which line it happened to",
      tail_track["stranded"] and tail_track["stranded"][0][0] == "V", tail_track["stranded"])
check("a track with nothing stranded says so with an empty list",
      T.finish({"total": 60.0, "language": "uk", "blocks": [
          {"label": "V", "voice": "N", "start": 0.0, "end": 60.0, "window": (0.0, 60.0),
           "lines": [{"text": "a b", "voice": "N", "label": "V",
                      "words": [{"text": "aa ", "key": "aa", "start": 1.0, "end": 2.0, "score": 0.9},
                                {"text": "bb", "key": "bb", "start": 2.0, "end": 3.0,
                                 "score": 0.9}]}]}]})["stranded"] == [])


# ------------------------------------------------------- what is sung over each shot, for Phantas
SUNG = {"total": 40.0, "lines": [
    {"start": 2.0, "end": 5.0, "voice": "Nina", "label": "Verse 1", "section": 0,
     "text": "Воно не стихне під гуркіт небес!", "words": [], "score": 0.9, "found": 5},
    {"start": 5.2, "end": 6.0, "voice": "Gru", "label": "Verse 1", "section": 0,
     "text": "(Живий!)", "words": [], "score": 0.5, "found": 1},
    {"start": 12.0, "end": 14.0, "voice": "Nina", "label": "Chorus", "section": 1,
     "text": "ЦЕ СТАЛЕВЕ СЕРЦЕ!", "words": [], "score": 0.8, "found": 3},
]}
rows = T.shot_lyrics(SUNG, [(0.0, 8.0), (8.0, 16.0), (16.0, 24.0)])
check("one row per shot", len(rows) == 3, rows)
check("a shot with no words says so", "instrumental" in rows[2], rows[2])
check("both singers of a shot are named",
      "Nina - " in rows[0] and "Gru - " in rows[0], rows[0])
check("the words are quoted verbatim", "Воно не стихне під гуркіт небес!" in rows[0], rows[0])
check("shots are numbered from 1", rows[0].startswith("1. ") and rows[2].startswith("3. "), rows)

# The rule Phantas depends on and that nothing else enforces: the planner emits weights and the
# frame grid is applied afterwards, so a model shown seconds starts reasoning in them.
import re as _re  # noqa: E402
check("NO timestamps reach the planner, in any shape",
      not _re.search(r"\d+:\d\d|\d+\.\d+\s*s|\bsecond", "\n".join(rows)), rows)

# A line straddling a cut is context for BOTH shots — that is right for mood and would be wrong
# for a subtitle, which is why this lives apart from the subtitle path.
straddle = T.shot_lyrics(SUNG, [(0.0, 4.0), (4.0, 8.0)])
check("a line across a cut is listed on both sides",
      "стихне" in straddle[0] and "стихне" in straddle[1], straddle)
# ...but only when it really is on both sides.
check("a line that merely grazes a boundary is not double-counted everywhere",
      "СТАЛЕВЕ" not in straddle[0] and "СТАЛЕВЕ" not in straddle[1], straddle)
check("lines_in is empty outside the song", T.lines_in(SUNG, 30.0, 40.0) == [])
check("shot_lyrics on an empty timing still returns a row per shot",
      len(T.shot_lyrics({"lines": []}, [(0, 5), (5, 10)])) == 2)


# --------------------------------------------------------------- printable on a cp1251 console
# The node prints its notes under `verbose`, and the console on the machine this pack is written on
# is cp1251. One nice-looking arrow turns a report into a UnicodeEncodeError.
unprintable = []
for note in jnotes + mnotes + nnotes + bad + digits + dur_notes + rnotes + snotes:
    try:
        note.encode("cp1251")
    except UnicodeEncodeError as e:
        unprintable.append((note[:40], str(e)[:60]))
check("every note survives a cp1251 console", unprintable == [], unprintable[:2])

check.done()
