"""Laying the words onto the plan: syllables, voice runs, and which phrase carries what.

Two of these are regressions for bugs that had already shipped into a report before anyone looked.

The first is the syllable count, which lives in Siren and was wrong there for both nodes. It counted
vowel *groups* — right for English, where `beautiful` has five vowels and three syllables, and wrong
for Ukrainian, where two vowels side by side are two syllables: `знає` is зна-є and `загартує` is
за-гар-ту-є, and it read them as one and three. In Siren the error hid, because every section was
short by the same few per cent and only the proportions between them decide bar counts. Here a
syllable is a note and the count is checkable against the model, which is how it surfaced: the
measured chorus counted 44 against a plan the model had written 47 notes for.

The second is duller and worse. `lay` walked the phrases with a cursor called `at` and then, inside
the same scope, reused `at` to walk the words of a run. The inner loop clobbered the outer one, so
every run after the first was handed the wrong slice of phrases and the last run of a section usually
got none at all — three phrases silently vanished from a 27-phrase song, all of them the bracketed
backing lines that are the whole reason the node exists. Nothing raised. The check that catches it is
simply that every sung phrase gets exactly one take.
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
S = load_module("kn.siren.score", "siren/score.py")
N = load_module("kn.satyr.notation", "satyr/notation.py")
B = load_module("kn.satyr.bands", "satyr/bands.py")
load_module("kn.satyr.rewrite", "satyr/rewrite.py")
L = load_module("kn.satyr.layout", "satyr/layout.py")

check = Checker()

VOICES = [{"name": "Keen Burg", "tags": "melodic female vocal, piercing"},
          {"name": "Gru BNik", "tags": "gritty male vocal, building tension"}]


# ------------------------------------------------------------------------------------ syllables
for word, real in [("знає", 2), ("загартує", 4), ("свою", 2), ("приховуючи", 5),
                   ("Сирени", 3), ("єдина", 3), ("ціль", 1), ("свій", 1)]:
    check(f"{word} is {real} syllables", L.syllables(word) == real, L.syllables(word))
check("the counter is Siren's own, now fixed", L.syllables is S._syllables)
check("adjacent vowels each count", S._syllables("загартує") == 4, S._syllables("загартує"))
check("English keeps its vowel groups", L.syllables("beautiful") == 3, L.syllables("beautiful"))
check("and its silent final e", L.syllables("stone") == 1, L.syllables("stone"))
check("й is not a syllable", L.syllables("край") == 1, L.syllables("край"))

CHORUS = """Сирени ритм став частиною дня,
Ми звикли до звуку, що б'є у вікна.
Нас страх не зламає, він загартує,
В цьому мовчанні — наша єдина ціль."""
check("the measured chorus is 43 syllables", sum(L.syllables(x) for x in CHORUS.splitlines()) == 43,
      sum(L.syllables(x) for x in CHORUS.splitlines()))
check("and its bracketed line is 4", L.syllables("Наша сила...") == 4)


# ------------------------------------------------------------------------------------ blocks
LYRICS = """[Intro]
(distorted bass, atmospheric guitar)

[Verse - Gru BNik]
Світло ранку на уламках скла,
Діаманти пилу в нічному вогні.

[Chorus - Keen Burg]
""" + CHORUS + """
(Наша сила...)
"""
blocks, notes = L.blocks(LYRICS, VOICES)
check("one block per top-level section", [b.label for b in blocks] == ["Intro", "Verse", "Chorus"],
      [b.label for b in blocks])
check("a stage direction is not sung", blocks[0].units == [], blocks[0].units)
check("a name in a section header is found", blocks[1].units[0].voice == "Gru BNik",
      blocks[1].units[0].voice)
check("a lead unit and a backing unit", [u.backing for u in blocks[2].units] == [False, True],
      [u.backing for u in blocks[2].units])
check("the chorus totals 47 syllables", blocks[2].syllables == 47, blocks[2].syllables)

EXCHANGE = """[Chorus - Keen Burg + Gru BNik]
[Gru BNik]
Сирени ритм став частиною дня,
[Keen Burg]
Нас страх не зламає, він загартує,
"""
swap, _ = L.blocks(EXCHANGE, VOICES)
check("an exchange stays one block", len(swap) == 1, [b.label for b in swap])
check("with one unit per voice", [u.voice for u in swap[0].units] == ["Gru BNik", "Keen Burg"],
      [u.voice for u in swap[0].units])


# ------------------------------------------------------------------------------------ bands
table, said = L.voice_bands(VOICES)
check("a female card takes the upper band", table["Keen Burg"] == B.HIGH)
check("a male card takes the lower one", table["Gru BNik"] == B.LOW)
check("and nothing had to be guessed", said == [], said)

same, said = L.voice_bands([{"name": "A", "tags": "soprano"}, {"name": "B", "tags": "mezzo"}])
check("a stated range is read", same == {"A": B.HIGH, "B": B.LOW}, same)
twins, said = L.voice_bands([{"name": "A", "tags": "bright soprano"}, {"name": "B", "tags": "soprano"}])
check("two of one range are still split", set(twins.values()) == {B.HIGH, B.LOW}, twins)
check("but warned about", any("sound like one" in s for s in said), said)
ranges, _ = L.voice_bands([{"name": "T", "tags": "warm tenor"}, {"name": "Bs", "tags": "deep bass"}])
check("range beats gender when both are male", ranges == {"T": B.HIGH, "Bs": B.LOW}, ranges)
mixed, _ = L.voice_bands([{"name": "T", "tags": "warm tenor"}, {"name": "S", "tags": "soprano"}])
check("and the same tenor is the lower one next to a soprano",
      mixed == {"S": B.HIGH, "T": B.LOW}, mixed)
solo, said = L.voice_bands([{"name": "Only", "tags": "female vocal"}])
check("one wired voice is not recast at all", solo == {}, solo)
check("and says why", any("nothing to alternate" in s for s in said), said)
blind, said = L.voice_bands([{"name": "X", "tags": "close-mic"}, {"name": "Y", "tags": "warm"}])
check("a card with no range still gets a band", set(blind.values()) == {B.HIGH, B.LOW}, blind)
check("and the guess is reported", len([s for s in said if "wiring order" in s]) == 2, said)


# ------------------------------------------------------------------------------------ allot
check("allot adds back up", sum(L.allot(47, [43, 4])) == 47)
check("allot honours the floor", L.allot(3, [100, 1], floor=1) == [2, 1], L.allot(3, [100, 1], 1))
check("allot survives zero weights", sum(L.allot(5, [0, 0, 0])) == 5, L.allot(5, [0, 0, 0]))
check("allot with nothing to give", L.allot(2, [1, 1], floor=1) == [1, 1])
check("allot of nothing", L.allot(0, []) == [])


# ------------------------------------------------------------------------------------ aligning
# The shape of a real song, in counts: nine lyric blocks against nine plan sections that do NOT
# correspond. The model merged the verse with the pre-chorus after it (63 + 47 landed on one
# 112-note section), gave an interlude and a drop no words, and left an 8-syllable outro with no
# notes. Pairing these positionally is what put somebody else's lyrics under nearly every section.
BLOCKS = [0, 63, 47, 49, 61, 39, 0, 49, 8]
SECTIONS = [0, 112, 47, 61, 0, 17, 4, 48, 0]
pairs = L.align(BLOCKS, SECTIONS)
check("one entry per section", len(pairs) == len(SECTIONS), len(pairs))
check("every block is placed exactly once",
      sorted(i for p in pairs for i in p) == list(range(len(BLOCKS))), pairs)
check("blocks stay in order", [i for p in pairs for i in p] == sorted(i for p in pairs for i in p))
check("the verse takes the pre-chorus with it", pairs[1] == [1, 2], pairs[1])
check("and the counts agree", sum(BLOCKS[i] for i in pairs[1]) == 110, pairs[1])
check("a wordless section takes nothing", pairs[4] == [], pairs[4])
check("the outro keeps its own words", pairs[8] == [8], pairs[8])
check("an empty block is not swept into its neighbour", pairs[0] == [0], pairs[0])

check("align survives no sections", L.align([3], []) == [])
check("align survives no blocks", L.align([], [4, 4]) == [[], []])
check("one to one when the counts line up",
      L.align([10, 10, 10], [10, 10, 10]) == [[0], [1], [2]])


# ------------------------------------------------------------------------------------ planning
#: Short names, because the fixtures below write markers like `[Verse 1 - Burg]` and Siren matches a
#: marker against a card's full name or its FIRST name — `Burg` is neither of `Gru BNik`.
PAIR = [{"name": "Keen", "tags": "melodic female vocal"},
        {"name": "Burg", "tags": "gritty male vocal"}]


def score_of(*sections):
    """A 4/4 score at 120bpm; each section is one Vocal line and a silent Ins."""
    head = ['X:1', 'T:', 'M:4/4', 'L:1/8', 'Q:1/4=120',
            'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
            'V: Ins clef=treble name="Ins Melody" snm="Inst."', 'K:C']
    for label, vocal in sections:
        head += [f"% {label}", "V: Vocal", vocal, "V: Ins", "Z|"]
    return "\n".join(head) + "\n"


LOW, HIGH = "CCDDEEFF|", "ccddeeff|"        # 8 notes at 60-65, and the same an octave up
PLAN = score_of(("intro", "z8|"), ("verse", LOW), ("verse", LOW), ("chorus", "c2d2e2f2|"))


def words(*blocks):
    out = []
    for label, syl in blocks:
        out.append(f"[{label}]")
        out.append(" ".join(["та"] * syl) if syl else "")
    return "\n".join(out) + "\n"


# Burg sings low, Keen high — which is exactly where this plan already puts them.
AGREES = words(("Intro", 0), ("Verse 1 - Burg", 4), ("Pre-Chorus - Burg", 4),
               ("Verse 2 - Burg", 8), ("Chorus - Keen", 4))
kept, spots, said = L.plan(N.parse(PLAN), AGREES, PAIR)
check("a plan that already agrees is not touched", kept.text() == PLAN)
check("and nothing is reported as moved", all(s.octaves == 0 for s in spots))
check("the merge is still found", spots[1].wants == 8 and spots[1].holds == 8,
      (spots[1].wants, spots[1].holds))
check("the section knows whose it is", spots[1].voice == "Burg", spots[1].voice)
check("and what register it uses", spots[1].was == B.LOW, spots[1].was)

# Now hand the second verse to Keen. Moving it up leaves the first verse holding the low band, so
# the two registers survive and the correction stands.
DISAGREES = words(("Intro", 0), ("Verse 1 - Burg", 4), ("Pre-Chorus - Burg", 4),
                  ("Verse 2 - Keen", 8), ("Chorus - Keen", 4))
fixed, spots, said = L.plan(N.parse(PLAN), DISAGREES, PAIR)
check("a section that contradicts its marker is corrected", spots[2].octaves == 1, spots[2].octaves)
check("and only that section moves", sum(1 for s in spots if s.octaves) == 1)
check("exactly one line changed",
      sum(1 for a, b in zip(PLAN.splitlines(), fixed.text().splitlines()) if a != b) == 1)
moved = B.read(N.parse(fixed.text()))
B.find(moved)
check("it landed in the register asked for", moved[2].band == B.HIGH, moved[2].middle)
check("the song is the same length",
      N.parse(fixed.text()).bar_count() == N.parse(PLAN).bar_count())

# Move the ONLY low section up and there is no low register left. That is how a real plan came back
# sung by one singer, so the correction has to be taken back rather than made.
TWO = score_of(("verse", LOW), ("chorus", "c2d2e2f2|"))
ALLUP = words(("Verse - Keen", 8), ("Chorus - Keen", 4))
same, spots, said = L.plan(N.parse(TWO), ALLUP, PAIR)
check("a correction that would collapse the registers is reverted", same.text() == TWO)
check("and none is reported as kept", all(s.octaves == 0 for s in spots))
check("with the reason named", any("taken back" in s for s in said), said)
check("recast off leaves the plan alone",
      L.plan(N.parse(PLAN), DISAGREES, PAIR, recast=False)[0].text() == PLAN)


# ------------------------------------------------------------------------------------ findings
# The verdict is the point of the node: both real failures were visible here before rendering.
NOWORDS = score_of(("verse", LOW), ("chorus", "c2d2e2f2|"), ("outro", "z8|"))
WITHOUTRO = words(("Verse - Burg", 8), ("Chorus - Keen", 4), ("Outro - Keen", 6))
_, spots, said = L.plan(N.parse(NOWORDS), WITHOUTRO, PAIR)
check("words the plan gives no notes are reported",
      any("will not be sung" in s for s in said), said)

SHORT = score_of(("verse", LOW), ("chorus", "c2|"))
_, _, said = L.plan(N.parse(SHORT), words(("Verse - Burg", 8), ("Chorus - Keen", 20)), PAIR)
check("a section short of notes is reported", any("dropped or slurred" in s for s in said), said)

# Registers this close came back as one voice in a real run.
CLOSE = score_of(("verse", "CCDDEEFF|"), ("chorus", "FFGGAABB|"))
_, _, said = L.plan(N.parse(CLOSE), words(("Verse - Burg", 8), ("Chorus - Keen", 8)), PAIR)
check("registers too close together are reported",
      any("semitones apart" in s for s in said), said)

ONEBAND = score_of(("verse", LOW), ("chorus", "CCDDEEFF|"))
_, _, said = L.plan(N.parse(ONEBAND), words(("Verse - Burg", 8), ("Chorus - Keen", 8)), PAIR)
check("a single-register plan is reported",
      any("one register" in s for s in said), said)
# The report must not predict how many singers there will be. It was claiming a register gap below
# eight semitones meant one voice; a third measurement killed that — two plans five semitones apart,
# one sung by two voices and one by one, differing only in their style string.
check("and nothing predicts the number of singers",
      not any("will be sung by one singer" in s for s in said), said)

check("the report renders", "bars =" in L.report(*L.plan(N.parse(PLAN), AGREES, PAIR)))

check.done()
