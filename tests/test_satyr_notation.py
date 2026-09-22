"""Reading and editing a YuE2 score without breaking it.

Two failures this guards against are silent rather than loud, which is why they get their own cases.

A rest is the first. `z` is a beat of silence and `Z` is a whole bar of it, so an octave shift that
reached for the letter case without checking would turn `z4` into `Z4` — four empty bars where there
was a quarter of one — and every note after it in the song would move. Nothing would raise; the take
would just be wrong three minutes in.

A tie is the second. `d2-d2` is one syllable held across a barline, and counting the glyphs instead
of the onsets gives a phrase one more syllable than a singer can put in it. That error compounds
along a section until a voice change lands mid-word, which is exactly the artefact this suite exists
to remove.

The fixture is deliberately small but carries every construct the real scores use: two voices, a
`%` section label, folded whole-bar rests, chord symbols, an inline key change, ties, accidentals and
both octave marks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "satyr")
N = load_module("kn.satyr.notation", "satyr/notation.py")

check = Checker()

SCORE = """X:1
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


# ------------------------------------------------------------------------------------ parsing
score = N.parse(SCORE)
check("round-trips byte for byte", score.text() == SCORE)
check("reads the meter", score.meter == (2, 4), score.meter)
check("reads the tempo", score.bpm == 77, score.bpm)
check("reads the key", score.key == "D#m", score.key)
check("finds both sections", [s.label for s in score.sections] == ["intro", "verse"],
      [s.label for s in score.sections])
check("header V: lines are not bodies", len(score.sections[0].groups) == 1,
      len(score.sections[0].groups))

# `Z` folds four bars into one glyph; a bar count that missed it would halve the song.
check("expands a folded whole-bar rest", N.bars("Z2|") == 2, N.bars("Z2|"))
check("counts written bars", N.bars('"B"a2g2f2e2-|"B"e2f4z2|') == 2)
check("both voices agree on bars",
      all(N.bars(score.lines[g.vocal]) == N.bars(score.lines[g.ins])
          for s in score.sections for g in s.groups))

# 2/4 at 77 bpm: a bar is two quarters, a quarter is 60/77 s.
check("bar length is arithmetic", abs(score.bar_seconds - 1.5584) < 0.001, score.bar_seconds)
check("duration follows from the bars", abs(score.duration() - 4 * 1.5584) < 0.01, score.duration())


# ------------------------------------------------------------------------------------ pitch
line = '"B"a2g2f2e2-|"B"e2f4z2|'
# D#m carries six sharps (F C G D A E), so every one of these letters is raised.
check("applies the key signature", N.pitches(line, "D#m") == [82, 80, 78, 77, 77, 78],
      N.pitches(line, "D#m"))
check("no signature without a key", N.pitches(line, "") == [81, 79, 77, 76, 76, 77], N.pitches(line, ""))
check("an explicit accidental wins", N.pitches("^c2|", "") == [73], N.pitches("^c2|", ""))
check("and propagates by letter across octaves inside the bar",
      N.pitches("^cC|", "") == [73, 61], N.pitches("^cC|", ""))
check("but resets at the barline", N.pitches("^c|c|", "") == [73, 72], N.pitches("^c|c|", ""))
check("octave marks move by twelve", N.pitches("C,C c c'|", "") == [48, 60, 72, 84],
      N.pitches("C,C c c'|", ""))

# The tie is one held syllable: six notes sound here, but the singer starts only five of them.
check("six sounding notes, five attacks", len(N.pitches(line, "D#m")) == 6
      and N.attacks(line, "D#m") == 5, N.attacks(line, "D#m"))
check("a rest is no attack", N.attacks('"D#m"z8|', "D#m") == 0)
check("a folded rest is no attack", N.attacks("Z4|", "") == 0)


# ------------------------------------------------------------------------------------ shifting
check("an octave down is exactly twelve",
      N.pitches(N.shift_octaves(line, -1), "D#m") == [70, 68, 66, 65, 65, 66])
check("an octave up is exactly twelve",
      N.pitches(N.shift_octaves(line, 1), "D#m") == [94, 92, 90, 89, 89, 90])
check("two octaves compose", N.shift_octaves(line, -2) == N.shift_octaves(N.shift_octaves(line, -1), -1))
check("a shift of nothing changes nothing", N.shift_octaves(line, 0) == line)
check("it is reversible", N.shift_octaves(N.shift_octaves(line, -3), 3) == line)

# The trap: a quarter rest must not become four empty bars.
check("a beat rest keeps its case", N.shift_octaves('"D#m"z4A2A2|', -1) == '"D#m"z4A,2A,2|',
      N.shift_octaves('"D#m"z4A2A2|', -1))
check("a whole-bar rest is untouched", N.shift_octaves("Z4|", -1) == "Z4|")
check("bars survive a shift", N.bars(N.shift_octaves('"D#m"z4A2A2|', -1)) == 1)

# Everything that is not a pitch has to come back unchanged.
check("chord symbols survive", '"B"' in N.shift_octaves(line, -1))
check("an inline key change survives", N.shift_octaves('[K:Am]c2|', -1) == '[K:Am]C2|',
      N.shift_octaves('[K:Am]c2|', -1))
check("durations and ties survive", N.shift_octaves("e2-|", -1) == "E2-|", N.shift_octaves("e2-|", -1))
check("attacks are invariant under shifting", N.attacks(N.shift_octaves(line, -1), "D#m") == 5)

# Editing one line must leave every other byte of the score alone.
edited = N.parse(SCORE)
target = edited.sections[1].groups[0].vocal
edited.replace(target, N.shift_octaves(edited.lines[target], -1))
before, after = SCORE.splitlines(), edited.text().splitlines()
check("a surgical edit touches exactly one line",
      sum(1 for a, b in zip(before, after) if a != b) == 1)
check("and keeps the line count", len(before) == len(after))


# ------------------------------------------------------------------------------------ problems
check("a well-formed score has no problems", N.problems(score) == [], N.problems(score))

bent = SCORE.replace("Z2|", "Z3|")          # Ins now claims three bars against the Vocal's two
check("unequal bar counts are caught", any("bars" in p for p in N.problems(N.parse(bent))),
      N.problems(N.parse(bent)))

misplaced = SCORE.replace("Z2|", '"B"Z2|')  # harmony written where nothing reads it
check("a chord on the Ins voice is caught",
      any("Ins voice" in p for p in N.problems(N.parse(misplaced))),
      N.problems(N.parse(misplaced)))

check.done()
