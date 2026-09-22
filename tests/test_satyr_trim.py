"""Cutting the silence out of a plan without cutting the song.

The property under test is one sentence long and everything else is detail: **the number of sung
notes must not change**. A trim that removes a note has removed a word, and the failure is not
visible in a plan that still parses — it turns up three minutes into a render as a line the singer
never got to.

The second property is that both voices lose the same bars. A group is one span carried by Vocal and
Ins at once; drop half of one and the two run out of step from there to the end of the song, which
again parses fine and sounds wrong.

The numbers in the fixture are a real plan's shape in miniature: a long instrumental opening, a short
interlude between two sung sections, and an outro. On the plan those came from, 61 of 171 bars — 36%
of the song — carried no sung note, and removing them took 4:33 down to 3:45 with every one of its
326 sung notes intact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "satyr")
N = load_module("kn.satyr.notation", "satyr/notation.py")
T = load_module("kn.satyr.trim", "satyr/trim.py")

check = Checker()


def score_of(*sections):
    """A 4/4 score at 120bpm — one bar is exactly two seconds, so the arithmetic reads at a glance."""
    head = ["X:1", "T:", "M:4/4", "L:1/8", "Q:1/4=120",
            'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
            'V: Ins clef=treble name="Ins Melody" snm="Inst."', "K:C"]
    for label, groups in sections:
        head.append(f"% {label}")
        for vocal, ins in groups:
            head += ["V: Vocal", vocal, "V: Ins", ins]
    return "\n".join(head) + "\n"


QUIET = ("z8|z8|z8|z8|", "CDEFGABc|CDEFGABc|CDEFGABc|CDEFGABc|")     # 4 bars, nobody sings
SUNG = ("CCDDEEFF|", "Z|")                                           # 1 bar, 8 syllables

PLAN = score_of(
    ("intro", [QUIET, QUIET, QUIET]),        # 12 bars = 24s
    ("verse", [SUNG, SUNG]),                 # 2 bars, sung
    ("interlude", [QUIET]),                  # 4 bars = 8s
    ("chorus", [SUNG, SUNG]),                # 2 bars, sung
    ("outro", [QUIET, QUIET]),               # 8 bars = 16s
)
before = N.parse(PLAN)
check("a bar is two seconds", abs(before.bar_seconds - 2.0) < 1e-9, before.bar_seconds)
check("the fixture is 28 bars", before.bar_count() == 28, before.bar_count())
check("and 32 sung notes", T.sung_notes(before) == 32, T.sung_notes(before))


# ------------------------------------------------------------------------------------ classifying
found = T.stretches(before)
check("three silent stretches", [kind for kind, _ in found] == [T.INTRO, T.BETWEEN, T.OUTRO],
      [kind for kind, _ in found])
check("a sung section is never a stretch",
      all(s.label not in ("verse", "chorus") for _, ss in found for s in ss))

# The label does not decide it — what is actually in the section does.
ODD = N.parse(score_of(("verse", [QUIET]), ("chorus", [SUNG])))
check("a section labelled verse with no voice is still an opening",
      [kind for kind, _ in T.stretches(ODD)] == [T.INTRO], T.stretches(ODD))
SILENT = N.parse(score_of(("intro", [QUIET]), ("outro", [QUIET])))
check("a plan nobody sings in has nothing to protect, so nothing is cut",
      T.stretches(SILENT) == [], T.stretches(SILENT))
OPENS = N.parse(score_of(("verse", [SUNG]), ("outro", [QUIET])))
check("a song that opens on a voice has no introduction",
      [kind for kind, _ in T.stretches(OPENS)] == [T.OUTRO])


# ------------------------------------------------------------------------------------ cutting
after, rows = T.trim(N.parse(PLAN), intro=8.0, outro=8.0, between=8.0)
check("sung notes are untouched", T.sung_notes(after) == 32, T.sung_notes(after))
check("the plan got shorter", after.bar_count() < before.bar_count(), after.bar_count())
# Three eight-second groups against an eight-second limit: exactly one survives.
check("the intro was cut to its limit", len(after.sections[0].groups) == 1,
      len(after.sections[0].groups))
check("which is the limit in seconds", after.sections[0].groups[0] and
      N.bars(after.lines[after.sections[0].groups[0].vocal]) * after.bar_seconds == 8.0)
check("the score still parses cleanly", N.problems(after) == [], N.problems(after))
check("both voices kept the same bars",
      all(N.bars(after.lines[g.vocal]) == N.bars(after.lines[g.ins])
          for s in after.sections for g in s.groups if g.vocal is not None))
check("every section still has both voices",
      all(g.vocal is not None and g.ins is not None
          for s in after.sections for g in s.groups))
check("no stray voice header is left behind",
      after.text().count("V: Vocal") == after.text().count("V: Ins"))

# The sung sections have to come through exactly as they were.
kept = [line for line in after.lines if "CCDDEEFF" in line]
check("every sung line survived", len(kept) == 4, len(kept))

# An intro leads into the singing, so its END is what is kept; everything else keeps its start.
long_intro = score_of(("intro", [("z8|", "C8|"), ("z8|", "D8|"), ("z8|", "E8|")]),
                      ("verse", [SUNG]),
                      ("outro", [("z8|", "F8|"), ("z8|", "G8|"), ("z8|", "A8|")]))
ends, _ = T.trim(N.parse(long_intro), intro=2.0, outro=2.0, between=0.0)
text = ends.text()
check("the intro keeps its last bar", "E8|" in text and "C8|" not in text)
check("the outro keeps its first", "F8|" in text and "A8|" not in text)


# ------------------------------------------------------------------------------------ the limits
gone, _ = T.trim(N.parse(PLAN), intro=0.0, outro=0.0, between=0.0)
check("zero removes a stretch outright", gone.bar_count() == 4, gone.bar_count())
check("and takes its label with it", "% intro" not in gone.text())
check("but never a sung note", T.sung_notes(gone) == 32)

same, rows = T.trim(N.parse(PLAN), intro=999.0, outro=999.0, between=999.0)
check("a limit past the end changes nothing", same.text() == PLAN)
check("and the rows still say what is there", [r[2] for r in rows] == [12, 4, 8], rows)

# "At most" taken literally would delete a stretch shorter than one group; one group always stays.
tight, _ = T.trim(N.parse(PLAN), intro=1.0, outro=1.0, between=1.0)
check("a limit under one group still keeps one", tight.bar_count() == 4 + 4 + 4 + 4,
      tight.bar_count())
check("and zero still means none", T.trim(N.parse(PLAN), 0.0, 0.0, 0.0)[0].bar_count() == 4)


# ------------------------------------------------------------------------------------ the report
text = T.report(before, after, rows)
check("the report proves the notes survived", "untouched" in text, text.splitlines()[1])
check("and names each stretch", all(k in text for k in (T.INTRO, T.OUTRO, T.BETWEEN)))
solo = N.parse(score_of(("verse", [SUNG])))
check("a plan with nothing to cut says so", "nothing to cut" in T.report(solo, solo, []))

check.done()
