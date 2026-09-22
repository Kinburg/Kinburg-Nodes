"""Both Satyr nodes end to end: a real plan and real lyrics in, the strings YuE2 takes out.

Runs the actual `run()`. No comfy is needed — the nodes deliberately import nothing from it — and
that is the point of having this at all: the arithmetic is pinned by the other three suites, while
the wiring between them is only ever exercised here. The bug that prompted it was exactly that kind.
`_format_elapsed` takes a format argument, the node called it with one, and nothing in a hundred and
seventy-eight passing checks touched the line, because every one of them tested a module rather than
a node. It would have raised on the author's first run.

What this pins: that the plan comes back parseable and the same length, that the lyrics come back
with every marker gone — YuE2 sings round brackets and has no field for a name, so a marker left in
is not ignored — and that the three ways a person can wire this wrong say so in a sentence instead
of raising something from inside a parser.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

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
Nodes = load_module("kn.satyr.nodes", "satyr/nodes.py")

check = Checker()

PLAN = """X:1
T:
M:4/4
L:1/8
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% intro
V: Vocal
z8|
V: Ins
D2D2D2D2|
% verse
V: Vocal
C2D2E2F2|
V: Ins
Z|
% verse
V: Vocal
D2E2F2G2|
V: Ins
Z|
% chorus
V: Vocal
e2f2g2a2|
V: Ins
Z|
% chorus
V: Vocal
d2e2f2g2|
V: Ins
Z|
"""

LYRICS = """[Intro]
(distorted bass, atmospheric guitar)

[Verse - Gru BNik]
Світло ранку на уламках скла,
Діаманти пилу в нічному вогні.

[Chorus - Keen Burg]
Сирени ритм став частиною дня,
(Наша сила...)
"""

VOICES = {"voice_1": {"name": "Keen Burg", "tags": "melodic female vocal"},
          "voice_2": {"name": "Gru BNik", "tags": "gritty male vocal"}}

score_node = Nodes.KinburgSatyrScore()
read_node = Nodes.KinburgSatyrRead()


# ------------------------------------------------------------------------------------ the rewrite
abc, lyrics, report = score_node.run(PLAN, LYRICS, recast=True, refit=True, verbose=False, **VOICES)

check("three strings come back", all(isinstance(x, str) and x for x in (abc, lyrics, report)))
check("the report is timed", report.strip().splitlines()[-1].endswith(("ms", "s")),
      report.strip().splitlines()[-1])

after = N.parse(abc)
check("the plan still parses", len(after.sections) == len(N.parse(PLAN).sections))
check("and is the same length", after.bar_count() == N.parse(PLAN).bar_count(), after.bar_count())
check("and is still well-formed", N.problems(after) == [], N.problems(after))
check("the line count never moves", len(abc.splitlines()) == len(PLAN.splitlines()))
check("only Vocal lines were touched",
      all(PLAN.splitlines()[g.ins] == abc.splitlines()[g.ins]
          for s in after.sections for g in s.groups if g.ins is not None))
check("something was actually rewritten", abc != PLAN)


# ------------------------------------------------------------------------------------ the lyrics
# The markers used to be stripped, on the reasoning that YuE2 has no field for a singer. Listening
# said otherwise: the model takes the PERFORMANCE from them — strain, growls, belts — and stripping
# them flattens the take. It never took who sings from them; that comes from the style string. So
# passing them through is the default now.
check("the lyrics pass through untouched by default", lyrics == LYRICS)
check("markers and all", "Gru BNik" in lyrics and "Keen Burg" in lyrics)

bare = score_node.run(PLAN, LYRICS, recast=True, refit=True, verbose=False,
                      keep_markers=False, **VOICES)[1]
check("stripping is still available", "[Verse]" in bare and "[Chorus]" in bare)
check("and takes the names with it", "Keen Burg" not in bare and "Gru BNik" not in bare)
# YuE2 reads round brackets as a backing vocal, so a stage direction left in one is sung aloud —
# the only thing stripping is genuinely needed for.
check("a stage direction in round brackets is dropped", "distorted bass" not in bare)
check("but a real backing line is kept", "(Наша сила...)" in bare)
check("and the words are all there", "Світло ранку на уламках скла," in bare)


# ------------------------------------------------------------------------------------ the switches
plain, _, _ = score_node.run(PLAN, LYRICS, recast=False, refit=False, verbose=False, **VOICES)
check("with both switches off the plan is untouched", plain == PLAN)
moved, _, _ = score_node.run(PLAN, LYRICS, recast=True, refit=False, verbose=False, **VOICES)
check("recasting alone keeps every syllable count",
      [p.notes for p in B.read(N.parse(moved))] == [p.notes for p in B.read(N.parse(PLAN))])
cut, _, _ = score_node.run(PLAN, LYRICS, recast=False, refit=True, verbose=False, **VOICES)
check("refitting alone moves no phrase",
      [p.middle for p in B.read(N.parse(cut))] == [p.middle for p in B.read(N.parse(PLAN))])

solo, _, said = score_node.run(PLAN, LYRICS, recast=True, refit=True, verbose=False,
                               voice_1=VOICES["voice_1"])
check("one wired voice recasts nothing", solo == score_node.run(
    PLAN, LYRICS, recast=False, refit=True, verbose=False, voice_1=VOICES["voice_1"])[0])
check("recast is off unless asked for",
      Nodes.KinburgSatyrScore.INPUT_TYPES()["required"]["recast"][1]["default"] is False)
check("and so is refit",
      Nodes.KinburgSatyrScore.INPUT_TYPES()["required"]["refit"][1]["default"] is False)
check("and says why", "nothing to alternate" in said)


# ------------------------------------------------------------------------------------ the wiring
def fails(call, word):
    try:
        call()
    except RuntimeError as e:
        return word in str(e)
    return False


check("an empty plan is explained", fails(
    lambda: score_node.run("", LYRICS, True, True, False), "does not compose"))
check("a plan that is not a score is explained", fails(
    lambda: score_node.run("hello\nworld\n", LYRICS, True, True, False), "reads as a YuE2 score"))
check("lyrics with no markers are explained", fails(
    lambda: score_node.run(PLAN, "just some words\nand some more\n", True, True, False),
    "no section markers"))


# ------------------------------------------------------------------------------------ Satyr Read
map_text, seconds = read_node.run(PLAN, verbose=False)
check("the map names both voices", "two voices" in map_text, map_text.splitlines()[1])
check("a bar is two seconds at 120bpm, so five bars is ten", abs(seconds - 10.0) < 1e-6, seconds)
rows = [l for l in map_text.splitlines() if l.startswith(("intro", "verse", "chorus"))]
check("every phrase is listed", len(rows) == 5, len(rows))
check("reading changes nothing", read_node.run(PLAN, verbose=False)[0] == map_text)
check("a plan that is not a score is explained", fails(
    lambda: read_node.run("nonsense", verbose=False), "reads as a YuE2 score"))

check.done()
