"""The corrected plan Echo writes back — the output `Save Clip` and `Orpheus` are meant to trust.

Built from the author's first clean run, which contained one genuine misplacement: an `Outro` whose
words came back at 1:58 while the `Chorus` before it ran to 2:26. Two sections cannot be sung at
once, so walking the plan's own order produced a row of **negative** length — which clamped to
`0.00s` and was then dropped by the parser downstream, deleting a section from the very table whose
job is to be more trustworthy than the plan it corrects.

The node is loaded without torch: `align.py` imports it only inside its functions, so everything
here runs on stubs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "context", "echo", "save_video", "siren", "timer", "util")
load_module("kn.util.anytype", "util/anytype.py")
load_module("kn.context.character_card", "context/character_card.py")
load_module("kn.timer.timer_nodes", "timer/timer_nodes.py")
load_module("kn.categories", "categories.py")
load_module("kn.save_video.timeline", "save_video/timeline.py")
load_module("kn.siren.cast", "siren/cast.py")
load_module("kn.siren.score", "siren/score.py")
load_module("kn.echo.translit", "echo/translit.py")
load_module("kn.echo.subs", "echo/subs.py")
load_module("kn.echo.align", "echo/align.py")
T = load_module("kn.echo.track", "echo/track.py")
N = load_module("kn.echo.nodes", "echo/nodes.py")

check = Checker()


def section(index, label, start, end, voice="Nina", sung=True, planned=None):
    return {"index": index, "label": label, "start": start, "end": end, "voice": voice,
            "sung": sung, "planned": planned or (start, end)}


def lengths(table):
    """The table parsed the way `Save Clip` parses it — through Siren's own reader."""
    rows, notes = T.plan_rows(table)
    return [r["weight"] for r in rows], notes


# ------------------------------------------------------------------------------ the ordinary case
tidy = {"total": 120.0, "sections": [
    section(0, "Intro", 0.0, 10.0, "-", sung=False),
    section(1, "Verse 1", 12.0, 40.0),
    section(2, "Chorus", 44.0, 70.0),
    section(3, "Outro", 80.0, 110.0),
]}
table, notes = N._plan_table(tidy)
check("one row per section", len(table.splitlines()) == 4, table)
check("nothing to say about a tidy timeline", notes == [], notes)
widths, parse_notes = lengths(table)
check("the table parses back through Siren's own reader", len(widths) == 4, parse_notes)
check("the rows are contiguous and cover the whole song",
      abs(sum(widths) - 120.0) < 0.02, sum(widths))
check("a section runs to where the NEXT one's words begin", abs(widths[1] - 32.0) < 0.02, widths)
check("the voice column survives into the corrected plan", "| Nina |" in table, table.splitlines()[1])
check("an instrumental row keeps its dash", "| - |" in table, table.splitlines()[0])


# ------------------------------------------------- the misplacement from the author's real song
# `Outro` found at 118.0 while the `Chorus` before it runs to 146.1. Walking the plan's order gives
# the Chorus a length of 118.0 - 117.2 and the Outro a NEGATIVE one.
tangled = {"total": 179.2, "sections": [
    section(0, "Verse 2", 81.7, 100.8),
    section(1, "Bridge", 103.4, 116.7),
    section(2, "Chorus", 117.2, 146.1),
    section(3, "Outro", 118.0, 125.8),
    section(4, "Break", 146.1, 179.2, "-", sung=False),
]}
table, notes = N._plan_table(tangled)
widths, parse_notes = lengths(table)
check("no row is dropped by the parser", len(widths) == 5, (widths, parse_notes))
check("every row has a positive length", all(w > 0 for w in widths), widths)
check("the rows still add up to the song", abs(sum(widths) - 179.2) < 0.05, sum(widths))
check("the table is emitted in the order the words were FOUND",
      [r.split("|")[0].strip() for r in table.splitlines()][:4]
      == ["Verse 2", "Bridge", "Chorus", "Outro"],
      [r.split("|")[0].strip() for r in table.splitlines()])
check("...and the overlap is said out loud, because it means something is misplaced",
      any("cannot be sung at once" in n for n in notes), notes)
check("the section that jumped is named", any("Chorus" in n or "Outro" in n for n in notes), notes)

# The same shape, one step worse: a section found entirely before the one it follows.
inverted = {"total": 60.0, "sections": [
    section(0, "Verse 1", 40.0, 50.0),
    section(1, "Chorus", 10.0, 20.0),
]}
table, notes = N._plan_table(inverted)
check("an inverted pair comes out in time order",
      [r.split("|")[0].strip() for r in table.splitlines()] == ["Chorus", "Verse 1"], table)
check("...and is reported", any("cannot be sung at once" in n for n in notes), notes)
widths, _ = lengths(table)
check("both rows survive with a real length", len(widths) == 2 and all(w > 0 for w in widths),
      widths)


# ------------------------------------------------------------------------------- degenerate input
empty, empty_notes = N._plan_table({"total": 0.0, "sections": []})
check("no sections is an empty table, not an exception", empty == "" and empty_notes == [])
one, _ = N._plan_table({"total": 30.0, "sections": [section(0, "Solo", 0.0, 30.0, "-", sung=False)]})
check("a single section covers the song", abs(lengths(one)[0][0] - 30.0) < 0.02, one)


# ------------------------------------------------------------- an instrumental plan is not a fault
# Five `Break` rows against one `[Break]` marker with no words under it is an ordinary arrangement.
# Warning about it on every run is how a report teaches people to stop reading its warnings.
PLAN = ("Verse 1 | Nina | 16 bars\nBreak | - | 4 bars\nSolo | - | 4 bars\nBreak | - | 4 bars\n"
        "Solo | - | 4 bars\nBreak | - | 4 bars")
_, quiet = T.build("[Verse 1 - Nina]\nПерший рядок\n[Break]", PLAN, 120.0)
check("a plan with more instrumental rows than markers says nothing",
      not any("Break" in n or "break" in n for n in quiet), quiet)

# But a label SUNG more often than the plan has rows for it is still the real failure.
_, loud = T.build("[Chorus - Nina]\nПерший приспів\n[Verse 1 - Nina]\nКуплет\n[Chorus - Nina]\n"
                  "Другий приспів", "Chorus | Nina | 16 bars\nVerse 1 | Nina | 16 bars", 120.0)
check("a chorus sung twice against one row is still reported",
      any("sung 2 time(s)" in n for n in loud), loud)

check.done()
