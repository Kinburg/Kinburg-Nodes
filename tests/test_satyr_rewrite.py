"""Changing how many syllables a phrase holds, without breaking the bar it lives in.

The invariant under every case here is that a measure's units still add up afterwards. A bar that no
longer sums to its meter is not a bar the format can hold, and the damage is not local: every
following bar of the song shifts, so a three-minute take comes back wrong from that point on. Both
operations therefore work inside one measure and conserve its total.

Three cases are here because they were bugs, not because they were foreseen.

The first is a tie across a barline. `d8-|d6f2|` is two syllables, not three — the `d6` is the far
end of the tie — and an onset count that restarts at each barline reads three. That made `refit`
believe a phrase already holding the right number of syllables held one too many, so it merged a note
out of a line that was correct. Asking for the count a phrase already has must return it untouched.

The second is the order of merges. The format has no note of five units or seven, so a bar of 2,1,2,3
that merges its shortest pair first lands on 3,2,3 and jams — both remaining pairs sum to five.
Merging the other pair gives 2,3,3, which collapses all the way to a single eight.

The third is a phrase built of long notes tied across the barlines, which has exactly one onset per
measure and therefore no adjacent pair anywhere. Merging alone cannot shorten it at all; silencing a
note can, and that is why the last resort exists — and why it is counted separately, being the one
operation that removes a sound instead of reshaping one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "satyr")
N = load_module("kn.satyr.notation", "satyr/notation.py")
R = load_module("kn.satyr.rewrite", "satyr/rewrite.py")

check = Checker()


def units(line):
    """Units per measure — the thing that must never change."""
    out = []
    for measure in N.split_measures(line):
        if N.FULL_REST.match(measure.strip()):
            out.append(None)
            continue
        total = 0
        for m in N.ELEMENT.finditer(measure):
            if m.group("chord") is None and m.group("key") is None:
                total += int(m.group("units") or 1)
        out.append(total)
    return out


# ------------------------------------------------------------------------- asking for what is there
TIED = '"D#m"d8-|"D#m"d6f2|"C#"e8-|"C#"e4z2g2|'
check("a tie across a barline is one syllable", N.attacks(TIED, "D#m") == 4, N.attacks(TIED, "D#m"))
same, reached, cost = R.refit(TIED, 4)
check("asking for the count it has changes nothing", same == TIED, same)
check("and reports that count", reached == 4, reached)
check("and costs nothing", cost == {"split": 0, "merged": 0, "dropped": 0}, cost)

PLAIN = '"B"a2g2f2e2-|"B"e2f4z2|'
check("a plain phrase is stable too", R.refit(PLAIN, N.attacks(PLAIN, "D#m"))[0] == PLAIN)


# ------------------------------------------------------------------------------------ conserving
BAR = "g2fg2f3|"                               # 2+1+2+3 = 8 units, four onsets
for want in range(1, 9):
    line, reached, cost = R.refit(BAR, want)
    check(f"{want} syllables: the bar still sums to eight", units(line) == units(BAR),
          f"{line} {units(line)}")
    check(f"{want} syllables: the bar count holds", N.bars(line) == N.bars(BAR))
    check(f"{want} syllables: the report matches the line", N.attacks(line) == reached,
          f"{line} says {reached}, holds {N.attacks(line)}")

check("it reaches every count the bar can hold", all(R.refit(BAR, w)[1] == w for w in range(1, 9)))
check("capacity is the sung units, not the bars", R.capacity(BAR) == 8, R.capacity(BAR))
check("and it is the real ceiling", R.refit(BAR, 9)[1] == R.capacity(BAR))

# A phrase that is mostly silence can hold far less than its bars suggest.
SPARSE = '"G#m"z8|"G#m"z8|"G#m"z8|g2fg2f3|'
check("rests are not room to sing", R.capacity(SPARSE) == 8, R.capacity(SPARSE))
check("a phrase is never filled into its rests", R.refit(SPARSE, 20)[1] == 8)
check("and the rests are still rests", R.refit(SPARSE, 20)[0].count("z8") == 3)


# ------------------------------------------------------------------------------------ the lookahead
# Greedy-shortest merges 2+1 first and jams on 3,2,3 (five is not a length this format has).
line, reached, cost = R.refit(BAR, 1)
check("a jamming bar still collapses to one note", reached == 1, f"{line} -> {reached}")
check("and it is one note of the full bar", units(line) == [8], units(line))
check("collapsing cost only merges", cost["dropped"] == 0, cost)


# ------------------------------------------------------------------------------------ splitting
grown, reached, cost = R.refit(BAR, 6)
check("growing adds onsets", reached == 6, grown)
check("growing costs nothing but splits", cost["merged"] == 0 and cost["dropped"] == 0, cost)
check("a split repeats a pitch it already had",
      set(N.pitches(grown)) <= set(N.pitches(BAR)), (N.pitches(grown), N.pitches(BAR)))


# ------------------------------------------------------------------------------------ the last resort
# One onset per measure and nothing adjacent to merge with: only silencing can shorten this.
short, reached, cost = R.refit(TIED, 3)
check("a merge-proof phrase can still be shortened", reached == 3, f"{short} -> {reached}")
check("and it had to silence a note", cost["dropped"] == 1 and cost["merged"] == 0, cost)
check("silencing conserves the bar", units(short) == units(TIED), units(short))
check("a note that ties onward is never silenced", short.startswith('"D#m"d8-|'), short)
check("dropping is a last resort, not a first",
      R.refit(BAR, 3)[2]["dropped"] == 0, R.refit(BAR, 3)[2])


# ------------------------------------------------------------------------------------ what survives
rich = '"Gmaj7"[K:Am]a2g2f2e2|'
out = R.refit(rich, 2)[0]
check("chord symbols survive surgery", '"Gmaj7"' in out, out)
check("an inline key change survives", "[K:Am]" in out, out)

MIXED = '"B"a2g2f2e2-|"B"e2f4z2|'
touched = R.refit(MIXED, N.attacks(MIXED) + 1)[0]
check("only the measure that changed is rewritten",
      sum(1 for a, b in zip(N.split_measures(MIXED), N.split_measures(touched)) if a != b) == 1,
      touched)
check("a rest is never turned into a note", "z2" in touched, touched)

check.done()
