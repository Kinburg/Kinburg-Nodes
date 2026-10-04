"""Laying the words onto the plan: which lyric line each phrase carries, and who sings it.

Siren already knows how to read this pack's lyric format — where a section starts, which member a
marker names, which lines are sung and which bracketed line is a stage direction rather than a word.
That reader is imported rather than copied, private names and all, because two parsers for one format
is two parsers to keep in step, and the one that drifts is always the copy.

**Everything here rests on matching the right words to the right section, and the obvious way to do
that is wrong.** YuE2 does not write one section per lyric block: it merges neighbours, inserts
instrumental sections that carry no words, and renames what is left by its own reading. Pairing them
positionally hands nearly every section somebody else's lyrics — and then the register each phrase is
given and the syllables it is cut to are confidently applied to the wrong text. That is not a
hypothetical; it is what this module did, and the songs came back sparse and sung by one voice.
`align` matches them by their counts instead, which is possible because the model writes about one
note per syllable (1.03 measured over a whole song, and exactly 47/47 and 39/39 on single sections).

**Register decides who sings once it agrees with the markers — with a LoRA on the text encoder.**
At one seed, a plan whose sections contradicted their markers in five places came back in a single
voice, and the same plan with those five moved an octave was sung by exactly the singers marked, its
melody, rhythm and arrangement unchanged. That is why `recast` is on by default. On the base model it
is not a control: a song whose style names no voices is sung by one singer however the plan is
written, and the same correction turned one section of four, sitting no higher than the three that
did not turn. That was the September verdict, and on the base model it still holds.

**The most useful thing this produces is not an edit but a verdict.** `_findings` says, before a
render is spent, whether the plan has two separated registers at all, which lyric block landed where,
which block was given too few notes to be sung in full, and which was given none. Every failure the
author hit over a week of runs is visible in it beforehand.

**The band a singer gets is relative, so the rule has to be too.** There is nothing in YuE2 to bind
a name to a timbre; the plan can only say upper or lower, and which timbre each one *is* comes from
the global `style` string. So the wired cards are ranked on one ladder of voice types and the upper
half takes the upper band. An absolute rule does not survive a real roster: "male means lower" is
right beside a soprano and wrong beside a bass, and the same tenor has to come out on either side
depending on who else is singing.
"""
from fractions import Fraction

from ..siren.cast import _gender_of
from ..siren.score import _PARENS, _split_sections
from ..siren.score import _syllables as syllables
from . import bands as B
from . import notation as N
from . import rewrite as R


class Unit:
    """A run of lyric lines one voice sings, or a single bracketed backing line."""

    __slots__ = ("voice", "syllables", "backing", "lines", "label")

    def __init__(self, voice, syllables, lines, label, backing=False):
        self.voice, self.syllables = voice, syllables
        self.lines, self.label, self.backing = lines, label, backing

    def __repr__(self):
        kind = "backing" if self.backing else "lead"
        return f"<{kind} {self.voice or '?'} {self.syllables}syl>"


class Block:
    """One top-level lyric section and the units inside it, in order."""

    __slots__ = ("label", "marker", "units", "notes")

    def __init__(self, label, marker):
        self.label, self.marker = label, marker
        self.units, self.notes = [], []

    @property
    def syllables(self):
        return sum(u.syllables for u in self.units)

    def __repr__(self):
        return f"<Block {self.label!r} {self.units}>"


def _names(section, voices):
    """Members this section names. A sub-section has them parsed already; a top-level header keeps
    them in its free-text marker, where Siren does not look for them because the label won the
    prefix match first."""
    if section["names"]:
        return list(section["names"])
    from ..siren.score import _names_in
    return _names_in(section["marker"], voices)


def blocks(lyrics, voices):
    """Lyrics → `Block` per top-level section, plus the reader's own notes.

    Siren returns one flat list in which a voice change is its own entry, so the top-level sections
    have to be recovered. A new one starts at an entry that is not a sub-section, and also at a
    sub-section that INHERITED — that is the shape left behind when a header named its singers and
    had no lines of its own, so the first exchange replaced the header outright.
    """
    sections, notes = _split_sections(lyrics, voices)
    out = []
    for section in sections:
        starts = (not section["sub"]) or bool(section["inherit"])
        if starts or not out:
            out.append(Block(section["label"], section["marker"]))
        block = out[-1]
        block.notes.extend(section["notes"])
        named = _names(section, voices)
        voice = named[0] if named else ""
        # Walked in written order, so a bracketed echo sitting between two sung lines keeps its
        # place. Collecting the echoes separately and appending them to the end looks equivalent
        # and is not: it would hand the echo a phrase at the close of the section instead of the
        # one it interrupts, and put it after words it is meant to answer.
        #
        # `text` is already the right set to walk — Siren puts a bracketed line there only once the
        # section has sung something, so a stage direction standing under a bare header ("(distorted
        # bass, atmospheric guitar)") never reaches it. That distinction matters more than it looks:
        # YuE2 reads round brackets as a backing vocal, so a stage direction left in the lyrics is
        # not ignored, it is sung.
        lead = None
        for line in section["text"]:
            if _PARENS.match(line):
                block.units.append(Unit(voice, syllables(line), [line], section["label"],
                                        backing=True))
                lead = None
                continue
            if lead is None:
                lead = Unit(voice, 0, [], section["label"])
                block.units.append(lead)
            lead.lines.append(line)
            lead.syllables += syllables(line)
    return out, notes


def strip_markers(blocks):
    """The lyrics cut back to bare `[Verse]` headers and the words. **Usually the wrong thing to do.**

    This was the default until it was listened to. YuE2's own guidance says to keep instructions out
    of the lyrics, and the reasoning here was that a marker naming a singer has nowhere to land since
    the plan carries the voice — so the markers came out. Songs made that way came back flatter: the
    model does read the markers, and what it takes from them is the PERFORMANCE. `[Keen - powerful
    belts]`, `[Burg - deep growl]`, `vocal duel, intense emotional peak` turn into strain and
    intensity in the take, and stripping them removes that while giving nothing back.

    On their own the markers do not decide who sings: how many singers there are comes from the
    style string, and where they sing follows a plan that agrees with the markers, with a LoRA on the
    text encoder. So they cost nothing to leave in.

    Kept for the one case that still wants it: a lyric carrying stage directions in ROUND brackets,
    which YuE2 reads as a backing vocal and will sing aloud. Square-bracket markers are safe.
    """
    out = []
    for block in blocks:
        out.append(f"[{block.label}]")
        for unit in block.units:
            out.extend(unit.lines)
        out.append("")
    return "\n".join(out).strip() + "\n"


#: Voice types on one ladder, highest first. A named range says more here than gender does, and it
#: has to be a ladder rather than two buckets: a tenor is above a bass and below a soprano, so no
#: fixed word-to-band map can be right for both pairings.
LADDER = (
    (0, ("soprano", "сопрано")),
    (1, ("mezzo", "меццо")),
    (2, ("alto", "contralto", "альт")),
    (3, ("countertenor", "контртенор")),
    (4, ("tenor", "тенор")),
    (5, ("baritone", "баритон")),
    (6, ("bass", "basso", "бас")),
)
#: Where a card that only says "female" or "male" sits on that ladder — between the ranges each
#: usually covers, so a named range always wins against a bare gender.
FEMALE_RANK, MALE_RANK = 1.5, 4.5


def _rank(tags):
    """Where a card sits on the ladder, or None when it says nothing about range."""
    text = str(tags or "").lower()
    for rank, words in LADDER:
        if any(w in text for w in words):
            return rank
    gender = _gender_of(text)
    return FEMALE_RANK if gender == "female" else MALE_RANK if gender == "male" else None


def voice_bands(voices):
    """{member name: band}, plus notes about anything it had to guess.

    The bands are relative, so the assignment is too: rank the wired voices, and the upper half takes
    the upper band. Absolute rules do not survive contact with a real roster — "male means lower" is
    right next to a soprano and wrong next to a bass.

    A single wired voice gets no band at all. There is nothing to alternate with, and recasting a
    solo song would move its phrases for no reason.
    """
    named = [( (v.get("name") or "").strip(), _rank(v.get("tags") or "") )
             for v in voices if (v.get("name") or "").strip()]
    notes = []
    if len(named) < 2:
        if named:
            notes.append(f"only {named[0][0]} is wired — with one singer there is nothing to "
                         f"alternate, so no phrase is recast. Wire a second voice to split them")
        return {}, notes

    blind = [n for n, r in named if r is None]
    for name in blind:
        notes.append(f"{name}'s card says nothing about range or gender, so the wiring order "
                     f"decided the band — put a range ('mezzo', 'baritone') or 'female'/'male' in "
                     f"the card's tags to pin it")
    # Unranked voices sort last but keep their wiring order, so a roster is at least stable run to run.
    order = sorted(range(len(named)),
                   key=lambda i: (named[i][1] is None, named[i][1] or 0, i))
    half = len(order) // 2
    table = {}
    for place, i in enumerate(order):
        table[named[i][0]] = B.HIGH if place < half else B.LOW

    above, below = named[order[half - 1]], named[order[half]]
    if above[1] is not None and below[1] is not None and abs(above[1] - below[1]) < 1:
        notes.append(f"{above[0]} and {below[0]} sit in the same range, so they were still split "
                     f"across the two bands — but YuE2 takes its timbres from the style string, not "
                     f"from the plan, and two singers this close will tend to sound like one")
    return table, notes


#: What a section pays for taking a second block on top of its first, in syllables. Small, because a
#: real merge is worth a couple of syllables of error; large enough that an exact tie never merges.
MERGE_COST = 0.5


def align(wants, holds):
    """Which lyric blocks each plan section carries. → [[block index, ...]] per section.

    The obvious pairing — zip the blocks against the sections — is wrong, and wrong in a way that
    looks right until you check the numbers. YuE2 does not lay one section per block. It MERGES
    neighbours (a measured plan gave one `% verse` 112 notes, which is a 63-syllable verse plus the
    47-syllable pre-chorus after it, 110), it INSERTS sections that carry no lyric at all (an
    interlude, a drop), and it renames whatever is left by its own reading. Pairing positionally
    therefore hands almost every section somebody else's words — and then everything downstream,
    the syllable refit and the voice each phrase is given, is confidently applied to the wrong text.
    That is what it did.

    So the blocks are matched to the sections by what the two of them are made of. `wants` is each
    block's syllable count and `holds` is each section's note count, and the model writes about one
    note per syllable — measured at 1.03 over a whole song, and exactly 47/47 and 39/39 on
    individual sections. Sections keep their order, blocks keep theirs, and a section takes a
    consecutive run of blocks, possibly none.

    A plain shortest-path over (blocks consumed, sections consumed): the cost of giving section `j`
    the blocks `k..i` is how far their syllables land from its notes. Greedy matching cannot do this
    — deciding that the first verse takes only the first block is locally perfect and forces every
    later section onto the wrong words.

    Merging costs a little on top, which is what `MERGE_COST` is for. Without it an empty section and
    an empty block tie at zero, and the solver is free to sweep a wordless `[Intro]` into the verse
    below it — the same answer arithmetically, and wrong to read. One block per section unless the
    counts genuinely ask for more.
    """
    n, m = len(wants), len(holds)
    if not m:
        return []
    running = [0]
    for w in wants:
        running.append(running[-1] + w)
    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(m + 1)]
    back = [[0] * (n + 1) for _ in range(m + 1)]
    best[0][0] = 0
    for j in range(1, m + 1):
        for i in range(n + 1):
            for k in range(i + 1):
                if best[j - 1][k] == INF:
                    continue
                cost = (best[j - 1][k] + abs(holds[j - 1] - (running[i] - running[k]))
                        + MERGE_COST * max(0, i - k - 1))
                if cost < best[j][i]:
                    best[j][i], back[j][i] = cost, k
    out, i = [], n
    for j in range(m, 0, -1):
        k = back[j][i]
        out.append(list(range(k, i)))
        i = k
    return list(reversed(out))


def allot(total, weights, floor=0):
    """Split `total` into one integer per weight, proportional, each at least `floor`.

    Largest remainder, so the parts always add back up to `total` — rounding each share on its own
    loses or invents a syllable, and a syllable that goes missing here is a word the singer never
    gets to.
    """
    n = len(weights)
    if n == 0:
        return []
    if total <= floor * n:
        return [floor] * n
    spare = total - floor * n
    mass = sum(weights)
    if mass <= 0:
        # Nothing to weigh by, so share it evenly. Leaving the weights at zero looks harmless and is
        # not: every exact share becomes zero, the whole of `spare` falls to the remainder step, and
        # that step can only hand out one each — so a total larger than the number of parts silently
        # comes back short.
        weights, mass = [1] * n, n
    exact = [spare * (w or 0) / mass for w in weights]
    parts = [int(e) for e in exact]
    for i in sorted(range(n), key=lambda i: exact[i] - parts[i], reverse=True)[:spare - sum(parts)]:
        parts[i] += 1
    return [floor + p for p in parts]


class Pickup:
    """The notes a section ends on that open the NEXT section's words: how many syllables they sing,
    and the phrase whose last bar they close."""

    __slots__ = ("onsets", "phrase")

    def __init__(self, onsets, phrase):
        self.onsets, self.phrase = onsets, phrase

    def __repr__(self):
        return f"<Pickup {self.onsets} on line {self.phrase.line}>"


def _beat(score):
    """`L:` units in one beat of the meter: a quarter in 4/4, an eighth in 6/8."""
    return max(1, round(Fraction(1, score.meter[1]) / score.unit))


def _lead(line, tied):
    """Rest before a line's first note, in `L:` units; 0 when it opens on a held note, None when a whole
    bar of rest comes first or no note comes at all."""
    if tied:
        return 0
    total = 0
    for measure in N.split_measures(line):
        if N.FULL_REST.match(measure.strip()):
            return None
        for m in N.ELEMENT.finditer(measure):
            if m.group("letter") == "z":
                total += int(m.group("units") or 1)
            elif m.group("letter"):
                return total
    return None


def _tail(line, tied, beat):
    """The run of notes a line ends on, when all of it sits in the line's last bar after a breath of at
    least a beat: (syllables it holds, where it starts in the text, rest after it). None otherwise.

    A run reaching back over the barline is the end of a longer phrase, not a pickup — and so is one
    whose breath cannot be seen because the bar before it is on another line.
    """
    cut = line.rstrip().rstrip("|").rfind("|") + 1
    if not cut:
        return None
    tie, els = N.tied_out(line[:cut], tied), []
    for m in N.ELEMENT.finditer(line, cut):
        if m.group("letter") == "z":
            els.append((None, int(m.group("units") or 1), m.start()))
            tie = False
        elif m.group("letter"):
            els.append((tie, int(m.group("units") or 1), m.start()))
            tie = bool(m.group("tie"))
    notes = [i for i, (held, _, _) in enumerate(els) if held is not None]
    if not notes:
        return None
    gap, first = 0, notes[-1]
    for i in range(notes[-1], -1, -1):
        held, units, _ = els[i]
        if held is None:
            gap += units
        elif gap >= beat:
            break
        else:
            gap, first = 0, i
    else:
        # The run opens the bar, so its breath is whatever rest the bar before ends on.
        if els[first][0]:
            return None
        before = N.split_measures(line[:cut])
        if gap < beat and not (before and N.FULL_REST.match(before[-1].strip())):
            trail = 0
            for m in N.ELEMENT.finditer(before[-1] if before else ""):
                if m.group("letter") == "z":
                    trail += int(m.group("units") or 1)
                elif m.group("letter"):
                    trail = 0
            if gap + trail < beat:
                return None
    run = els[first:notes[-1] + 1]
    return (sum(1 for held, _, _ in run if held is False), els[first][2],
            sum(units for _, units, _ in els[notes[-1] + 1:]))


def _pickup(score, here, there, beat):
    """The pickup between two neighbouring sections' Vocal phrases, or None."""
    if not here or not there or here[-1].silent or there[0].silent:
        return None
    tail = _tail(score.lines[here[-1].line], here[-1].tied, beat)
    lead = _lead(score.lines[there[0].line], there[0].tied)
    if tail is None or lead is None or tail[2] + lead >= beat:
        return None
    return Pickup(tail[0], here[-1])


def sections(score, phrases):
    """Each plan section as (label, its Vocal phrases, the pickup it opens on, the pickup it ends on).

    **A section's words start where its first phrase does, and that is often before its barline.**
    YuE2 writes an upbeat the way any song has one: a chorus opening "Ми не-втом-ні!" was planned as
    three quick notes in the last beat of the section before, the fourth on the chorus's downbeat,
    then a breath. Counted by barline that chorus held three notes fewer than its 52 syllables and the
    verse before it three more — and the same song's other chorus and its outro were off the same
    way. Counted from the pickup, all three came out exact. So a run of notes in the last bar of a
    section, after a breath of at least a beat, that carries straight on into the next section's first
    note belongs to the next section's words. The plan's labels stay where they are; YuE2 does not read
    them for timing anyway.
    """
    at, out = 0, []
    for section in score.sections:
        count = sum(1 for group in section.groups if group.vocal is not None)
        out.append((section.label, phrases[at:at + count]))
        at += count
    beat = _beat(score)
    ends = [_pickup(score, a, b, beat) for (_, a), (_, b) in zip(out, out[1:])] + [None]
    return [(label, ph, ends[i - 1] if i else None, ends[i]) for i, (label, ph) in enumerate(out)]


def held(phrases, opens=None, lends=None):
    """Syllables a section's notes carry: its own onsets, plus the pickup it opens on, less the one it
    ends on, which sings the next section's words."""
    return sum(p.notes for p in phrases) + (opens.onsets if opens else 0) - (lends.onsets if lends else 0)


#: Semitones of separation below which a recast is taken back. This is a guard against an edit
#: making things worse, NOT a prediction about the result: an earlier version of this comment claimed
#: 8 was where two singers start to blur, and a third measurement killed that outright. Two plans sat
#: 5 apart — one was sung by two voices and the other by one, and what differed was the style string,
#: not the plan. How many singers a song has is decided in `style` and nowhere else.
SAFE_GAP = 8


class Spot:
    """One plan section, the lyric blocks that landed on it, and who should be singing there."""

    __slots__ = ("label", "phrases", "blocks", "opens", "lends", "holds", "wants", "voice", "band", "was",
                 "octaves")

    def __init__(self, label, phrases, blocks, opens=None, lends=None):
        self.label, self.phrases, self.blocks = label, phrases, blocks
        self.opens, self.lends = opens, lends      # the pickup it opens on, and the one it ends on
        self.holds = held(phrases, opens, lends)
        self.wants = sum(b.syllables for b in blocks)
        self.voice = next((u.voice for b in blocks for u in b.units if u.voice), "")
        self.band = None            # the register the marker asks for
        self.was = None             # the register the plan already uses here
        self.octaves = 0

    @property
    def sung(self):
        return [p for p in self.phrases if not p.silent]

    def seconds(self, score):
        return self.phrases[0].seconds(score)[0] if self.phrases else 0.0

    def __repr__(self):
        return f"<Spot {self.label!r} holds={self.holds} wants={self.wants}>"


def _settled(spot):
    """The register the plan already uses here: the band most of its phrases sit in."""
    seen = [p.band for p in spot.sung if p.band]
    return max(set(seen), key=seen.count) if seen else None


def plan(score, lyrics, voices, refit=False, recast=True):
    """Match the lyrics to the plan, correct what disagrees, and report what it found.

    The order matters, and it used to be wrong. Sections are matched to blocks by their counts FIRST
    (`align`), because everything afterwards is applied to whichever words that match chose — and a
    positional pairing hands nearly every section somebody else's.

    Recasting is a **correction**, not a rewrite. YuE2 places its registers by habit, verses low and
    choruses high: on one measured plan that matched every marker, on two later ones it contradicted
    four and five sections of nine. So only a section whose register contradicts its marker moves,
    and it moves whole. The earlier version pushed every
    phrase toward a band centre, which on a plan whose registers were already close smeared the two
    clusters together — and a plan without two separated clusters is a song in one voice.
    """
    found, notes = blocks(lyrics, voices)
    table, voice_notes = voice_bands(voices)
    notes = list(notes) + voice_notes
    phrases = B.read(score)
    bands = B.find(phrases)

    parts = sections(score, phrases)
    pairing = align([b.syllables for b in found], [held(ph, o, e) for _, ph, o, e in parts])
    spots = [Spot(label, ph, [found[i] for i in idx], o, e) for (label, ph, o, e), idx in zip(parts, pairing)]
    beat = _beat(score)

    for spot in spots:
        spot.was = _settled(spot)
        spot.band = table.get(spot.voice)
        if not (recast and bands.two and spot.band and spot.was) or spot.band == spot.was:
            continue
        # The section moves as one. Deciding phrase by phrase is what scattered a verse across both
        # registers and turned a duet into a melody that leaps an octave mid-line.
        steps = [s for s in (B.octaves_to(p, spot.band, bands) for p in spot.sung) if s]
        if not steps:
            continue
        spot.octaves = max(set(steps), key=steps.count)
        _move(score, spot, spot.octaves, beat)

    # A correction that leaves the registers closer together than it found them cannot have helped,
    # so it is made, measured and taken back. This is hygiene rather than a promise: narrowing the
    # gap is pointless, but widening it has never been shown to change who sings either.
    if any(spot.octaves for spot in spots):
        after = B.find(B.read(N.parse(score.text())))
        kept = after.two and after.separation() >= min(bands.separation(), SAFE_GAP)
        if not kept:
            for spot in spots:
                if spot.octaves:
                    _move(score, spot, -spot.octaves, beat)
            moved =", ".join(f"'{s.label}'" for s in spots if s.octaves)
            spread = "into one register" if not after.two else \
                f"from {bands.separation():.0f} to {after.separation():.0f} semitones apart"
            notes.append(f"{moved} disagreed with the markers, but moving it pushed the two "
                         f"registers {spread}. The move was taken back and the plan is unchanged")
            for spot in spots:
                spot.octaves = 0

    if refit:
        notes.extend(_refit(score, spots))
    return score, spots, notes + _findings(score, spots, bands)


def _move(score, spot, octaves, beat):
    """Shift a section by whole octaves — with the pickup it opens on, which sits at the end of the
    section before, and without the one it ends on, which the next section's singer sings."""
    for phrase in spot.sung:
        line = score.lines[phrase.line]
        cut = _tail(line, phrase.tied, beat)[1] if spot.lends and spot.lends.phrase is phrase else len(line)
        score.replace(phrase.line, N.shift_octaves(line[:cut], octaves) + line[cut:])
    if spot.opens:
        phrase = spot.opens.phrase
        line = score.lines[phrase.line]
        cut = _tail(line, phrase.tied, beat)[1]
        score.replace(phrase.line, line[:cut] + N.shift_octaves(line[cut:], octaves))


def _refit(score, spots):
    """Cut each section's phrases to the syllables its blocks actually hold.

    Pickups are left exactly as written: the one a section opens on already sings its first
    syllables from the bar before, and the last bar of a section ending on one belongs to the next.
    """
    said = []
    for spot in spots:
        sung = spot.sung
        if not sung or not spot.wants:
            continue
        weights = [R.capacity(score.lines[p.line]) for p in sung]
        own = max(0, spot.wants - (spot.opens.onsets if spot.opens else 0))
        for phrase, want in zip(sung, allot(own, weights, floor=0)):
            lent = spot.lends.onsets if spot.lends and spot.lends.phrase is phrase else 0
            line, reached, _ = R.refit(score.lines[phrase.line], want + lent, phrase.tied, keep=1 if lent else 0)
            score.replace(phrase.line, line)
            if reached - lent != want:
                said.append(f"'{spot.label}': a phrase was asked for {want} syllables and holds "
                            f"{reached - lent}")
    return said


def _findings(score, spots, bands):
    """Everything worth knowing BEFORE spending four minutes on a render.

    This is the output the suite is really for. Both ways a real run went wrong were visible here
    beforehand: one plan had its registers five semitones apart and came back in a single voice, and
    the other silently gave an eight-syllable outro no notes at all.
    """
    out = []
    if not bands.two:
        out.append("this plan writes every phrase in one register. That does not by itself mean one "
                   "singer — how many voices a song has is decided by the style string — but there "
                   "is nothing here for a marker to disagree with")
    else:
        out.append(f"the plan came with two registers, {bands.separation():.0f} semitones apart. With "
                   f"a LoRA on the text encoder the singers follow them where they agree with the "
                   f"markers, which is what recast makes them do, and a plan that contradicts its "
                   f"markers came back in one voice. The base model mostly keeps one voice whatever "
                   f"the plan says")
    for spot in spots:
        where = f"'{spot.label}' at {_mmss(spot.seconds(score))}"
        named = " + ".join(b.label for b in spot.blocks) or "nothing"
        if spot.wants and not spot.holds:
            out.append(f"{where} carries {named} ({spot.wants} syllables) and the plan gives it no "
                       f"notes at all — those words will not be sung")
        elif spot.wants and spot.holds * 10 < spot.wants * 6:
            out.append(f"{where} has room for {spot.holds} syllables but {named} needs "
                       f"{spot.wants} — about {100 - spot.holds * 100 // spot.wants}% of it will be "
                       f"dropped or slurred")
        if spot.band and spot.was and spot.band != spot.was and not spot.octaves:
            out.append(f"{where} is marked for {spot.voice} ({spot.band}) and the plan puts it in "
                       f"the {spot.was} register")
        singers = {u.voice for b in spot.blocks for u in b.units if u.voice}
        if len(singers) > 1 and len(spot.sung) < 2:
            out.append(f"{where} asks for an exchange between {' and '.join(sorted(singers))}, but "
                       f"the plan gives it {len(spot.sung)} phrase(s) — it cannot hold one")
    return out


def _mmss(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def report(score, spots, notes, bands=None):
    """What landed where, whose voice it is, and what the plan cannot hold."""
    bands = B.find(B.read(score)) if bands is None else bands
    out = [f"{score.bar_count()} bars = {_mmss(score.duration())} at {score.bpm} bpm"]
    if bands.two:
        low, high = bands.seats[B.LOW], bands.seats[B.HIGH]
        out.append(f"two voices as sent: lower sits {low[0]:.0f}-{low[1]:.0f}, "
                   f"upper {high[0]:.0f}-{high[1]:.0f}, "
                   f"{bands.separation():.0f} semitones apart")
    else:
        out.append("one voice — the plan has no second register")
    out += ["", f"{'at':>7}  {'section':<12}{'notes':>6}{'syl':>5}{'band':>6}{'move':>6}  lyrics"]
    for spot in spots:
        named = " + ".join(b.label for b in spot.blocks) or "—"
        move = f"{spot.octaves:+d}oct" if spot.octaves else ""
        band = spot.was or "-"
        if spot.band and spot.band != spot.was and not spot.octaves:
            band += " !"
        out.append(f"{_mmss(spot.seconds(score)):>7}  {spot.label:<12}{spot.holds:>6}"
                   f"{spot.wants:>5}{band:>6}{move:>6}  {named}"
                   + (f"   [{spot.voice}]" if spot.voice else ""))
    if notes:
        out += [""] + ["! " + n for n in notes]
    return "\n".join(out)
