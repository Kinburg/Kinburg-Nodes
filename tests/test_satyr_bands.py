"""Finding the two singers in a score, and moving a phrase from one to the other.

The case that earns its keep here is the banker's-rounding one. `octaves_to` used to divide the
distance to the target band by twelve and round it, which is correct for every phrase except the one
sitting exactly half an octave away — and Python resolves `round(-0.5)` to zero, so that phrase was
told to stay put. It parsed, it rendered, it sounded fine, and one line of the song came out of the
wrong mouth. The real song had such a phrase: the low band's seat was 69 and a chorus phrase sat at
75. So the fixture below is not invented, it is that measurement.

The second thing under test is a distinction the first draft got wrong: a band's NOTES overlap its
neighbour's while its PHRASES do not. A phrase belonging to the man as a whole still reaches up into
the woman's notes for a syllable, so a classifier working from note ranges leaves straddling phrases
unassigned, and a move aimed at the middle of a note range aims at a pitch both singers use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "satyr")
N = load_module("kn.satyr.notation", "satyr/notation.py")
B = load_module("kn.satyr.bands", "satyr/bands.py")

check = Checker()

# 4/4 at 120bpm: a bar is two seconds. Five sung phrases and one silent one; the third reaches up
# into the upper voice's notes without belonging to it.
SCORE = """X:1
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
% verse
V: Vocal
C2D2E2a2|
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

score = N.parse(SCORE)
phrases = B.read(score)
bands = B.find(phrases)

# ------------------------------------------------------------------------------------ reading
check("one phrase per Vocal line", len(phrases) == 6, len(phrases))
check("phrases come in performance order", [p.start_bar for p in phrases] == [0, 1, 2, 3, 4, 5],
      [p.start_bar for p in phrases])
check("a rest-only phrase is silent", phrases[0].silent)
check("and is kept, not dropped", phrases[0].bars == 1)
check("syllable budget is read per phrase", [p.notes for p in phrases] == [0, 4, 4, 4, 4, 4],
      [p.notes for p in phrases])
check("a bar is two seconds at 120bpm", abs(score.bar_seconds - 2.0) < 1e-9, score.bar_seconds)
check("timings follow the bars", phrases[2].seconds(score) == (4.0, 6.0), phrases[2].seconds(score))

# ------------------------------------------------------------------------------------ finding
check("two voices are found", bands.two)
check("the cut falls in the empty middle", 65 < bands.split < 76, bands.split)
check("every sung phrase is assigned", all(p.band for p in phrases if not p.silent))
check("a silent phrase belongs to neither", phrases[0].band is None)
check("phrases land where they sit",
      [p.band for p in phrases] == [None, B.LOW, B.LOW, B.LOW, B.HIGH, B.HIGH],
      [p.band for p in phrases])

# The straddler is the point: it reaches 81, which is inside the upper band's notes.
check("note ranges overlap between bands",
      bands.spans[B.LOW][1] >= bands.spans[B.HIGH][0], (bands.spans[B.LOW], bands.spans[B.HIGH]))
check("but the seats do not", bands.seats[B.LOW][1] < bands.seats[B.HIGH][0],
      (bands.seats[B.LOW], bands.seats[B.HIGH]))
check("separation is measured between seats", bands.separation() > 0, bands.separation())

# One cluster is one singer, and saying so is a real answer rather than a failure.
solo = B.find([p for p in phrases if p.band != B.HIGH])
check("a single cluster is one voice", not solo.two)
check("and nothing gets classified", solo.of(phrases[1]) is None)


# ------------------------------------------------------------------------------------ moving
for p in phrases:
    if p.silent:
        continue
    for target in (B.LOW, B.HIGH):
        moved = p.middle + 12 * B.octaves_to(p, target, bands)
        landed = B.LOW if moved < bands.split else B.HIGH
        check(f"a {p.band} phrase asked for {target} lands in {target}", landed == target,
              f"median {p.middle} -> {moved}")

check("a phrase already in its band does not move",
      all(B.octaves_to(p, p.band, bands) == 0 for p in phrases if not p.silent))
check("a silent phrase never moves", B.octaves_to(phrases[0], B.LOW, bands) == 0)

# The regression, with the real song's numbers: low seat 68-70 (centre 69), phrase sitting at 75.
# `round((69 - 75) / 12)` is `round(-0.5)`, which Python makes 0 — leaving the phrase in the upper
# voice while reporting that it had been moved to the lower one.
real = B.Bands(split=72.5, spans={B.LOW: (63, 75), B.HIGH: (63, 82)},
               seats={B.LOW: (68, 70), B.HIGH: (75, 80)})
halfway = B.Phrase("chorus", 0, 0, 4, 1, [75])
check("centre of a band is the middle of its seat", real.centre(B.LOW) == 69, real.centre(B.LOW))
check("exactly half an octave away still moves", B.octaves_to(halfway, B.LOW, real) == -1,
      B.octaves_to(halfway, B.LOW, real))
check("and lands below the split", 75 + 12 * B.octaves_to(halfway, B.LOW, real) < real.split)


# ------------------------------------------------------------------------------------ held over
# The exporter writes at most four bars to a line, so a note held from one group into the next ends
# one Vocal line `F2-|` and opens the next `F2`. That is the first phrase's syllable; read line by
# line it was counted again at the start of the second, one too many at every such boundary.
HELD = """X:1
T:
M:4/4
L:1/8
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
C2D2E2F2-|
V: Ins
Z|
% chorus
V: Vocal
F2G2A2B2|
V: Ins
Z|
% chorus
V: Vocal
c4z2B2-|
V: Ins
Z|
% outro
V: Vocal
z8|
V: Ins
Z|
% outro
V: Vocal
B8|
V: Ins
Z|
"""
held = B.read(N.parse(HELD))
check("a note held into the next group is the first phrase's syllable, not the second's",
      [p.notes for p in held] == [4, 3, 2, 0, 1], [p.notes for p in held])
check("...and the phrase it runs into knows it opens on it",
      [p.tied for p in held] == [False, True, False, True, False], [p.tied for p in held])
check("...so the phrases add up to what the voice actually sings",
      sum(p.notes for p in held) == N.attacks("".join(N.parse(HELD).lines[p.line] for p in held)))
check("a group of rests in between breaks the tie: the note after it is a new syllable", held[4].notes == 1)


# ------------------------------------------------------------------------------------ applying
edited = N.parse(SCORE)
fresh = B.read(edited)
found = B.find(fresh)
moves = [(p, B.octaves_to(p, B.LOW, found)) for p in fresh if p.band == B.HIGH]
done = B.apply(edited, moves)
check("both upper phrases were moved", done == 2, done)
before, after = SCORE.splitlines(), edited.text().splitlines()
check("exactly two lines changed", sum(1 for a, b in zip(before, after) if a != b) == 2)
check("the line count is unchanged", len(before) == len(after))
check("the Ins voice is untouched",
      all(before[g.ins] == after[g.ins] for s in edited.sections for g in s.groups if g.ins))
check("the song is the same length", N.parse(edited.text()).bar_count() == score.bar_count())
check("nothing is left in the upper voice", not B.find(B.read(N.parse(edited.text()))).two)

check.done()
