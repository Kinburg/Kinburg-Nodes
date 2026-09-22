"""Which singer sings a phrase, and how to make it the other one.

YuE2 has no field for a voice. Asked in `style` for two of them, it writes both into the one
monophonic `Vocal` line as two pitch bands that never touch — a measured song put the man at MIDI
63-70 and the woman at 73-82, with nothing at all in between — and hands a phrase to whichever band
it was written in. Moving a phrase an octave moves it to the other singer. That was tested the only
way it can be: one chorus transposed down inside a song that kept its other chorus untouched, same
seed, and the moved one came back in the other voice.

**The bands are read off the score, never assumed.** Where they sit depends on what `style` asked
for — a bass and a tenor would not land where a man and a woman did — so the numbers above are one
song's, not a constant, and hardcoding them would be a bug that only shows up on someone else's
music. What generalises is the *shape*: two clusters with an empty gap between them. So `find()`
sorts the phrase medians, cuts at the widest gap, and reports what it found.

**A median, not a range.** Phrase ranges overlap the boundary — a phrase spanning 68-75 is ordinary —
while the medians separate cleanly, because a phrase belongs to one singer as a whole. Classifying on
range would leave the straddlers unassigned; classifying on median puts every phrase somewhere and is
what the model itself appears to do.

**The gap is the signal that there are two voices at all.** A song written for one singer is one
cluster, and `find()` says so rather than inventing a split down the middle of a single voice's
range. The other direction is the failure worth naming: a score with nothing above the boundary has
no upper voice *at all*, not a quieter one — transposing an entire song down an octave took the woman
out of it completely. Bands are a budget, not a preference.
"""
from statistics import median

from . import notation as N

#: Semitones of empty space needed between two clusters of phrase medians before they count as two
#: voices. Three is under the octave the bands actually sat apart by, and above the spread inside one
#: singer's phrases — a single voice wanders a few semitones between phrases, not a minor third.
MIN_GAP = 3

LOW, HIGH = "low", "high"


class Phrase:
    """One line of the Vocal voice: where it sits in time, how many syllables it can hold, and who
    sings it."""

    __slots__ = ("section", "line", "start_bar", "bars", "notes", "pitches", "band")

    def __init__(self, section, line, start_bar, bars, notes, pitches):
        self.section, self.line = section, line
        self.start_bar, self.bars = start_bar, bars
        self.notes, self.pitches = notes, pitches
        self.band = None

    @property
    def silent(self):
        return not self.pitches

    @property
    def middle(self):
        return median(self.pitches) if self.pitches else None

    @property
    def low(self):
        return min(self.pitches) if self.pitches else None

    @property
    def high(self):
        return max(self.pitches) if self.pitches else None

    def seconds(self, score):
        """(start, end) in seconds. The model performs a few percent faster than this, uniformly —
        so these are right for ordering and proportion, and a second or two out for seeking."""
        return (self.start_bar * score.bar_seconds,
                (self.start_bar + self.bars) * score.bar_seconds)

    def __repr__(self):
        return f"<Phrase {self.section} bar{self.start_bar} notes={self.notes} band={self.band}>"


def read(score):
    """Every Vocal line of a score as a `Phrase`, in performance order.

    Silent phrases are kept rather than dropped: an intro of nothing but rests is a real part of the
    song's shape, and a caller lining phrases up against lyric lines needs to see the holes.
    """
    out, bar = [], 0
    for section in score.sections:
        for group in section.groups:
            if group.vocal is None:
                continue
            line = score.lines[group.vocal]
            width = N.bars(line)
            out.append(Phrase(section.label, group.vocal, bar, width,
                              N.attacks(line, score.key), N.pitches(line, score.key)))
            bar += width
    return out


class Bands:
    """The two voices a score was written for — or the one it was, when there is no gap.

    Two different ranges are worth keeping and they are not interchangeable. `spans` is every note
    actually written in the band, and it OVERLAPS its neighbour: a phrase that belongs to the man as
    a whole still reaches up into the woman's notes for a syllable or two. `seats` is where the
    phrases sit — the spread of their medians — and that is what separates cleanly and what a move
    has to aim at. Targeting the middle of `spans` would aim at a point both singers use.
    """

    def __init__(self, split=None, spans=None, seats=None):
        self.split = split                      # the median the two clusters divide at, or None
        self.spans = spans or {}                # band → (lowest, highest) note written in it
        self.seats = seats or {}                # band → (lowest, highest) PHRASE median in it

    @property
    def two(self):
        return self.split is not None

    def of(self, phrase):
        """Which band a phrase belongs to. `None` for a silent one, and for every phrase in a
        single-voice score — there is no choice to report."""
        if phrase.silent or not self.two:
            return None
        return LOW if phrase.middle < self.split else HIGH

    def centre(self, band):
        """Where a phrase of this band typically sits. A move aims here."""
        lo, hi = self.seats[band]
        return (lo + hi) / 2

    def separation(self):
        """Semitones between the two seats — how far apart the singers really are.

        Reported instead of a gap between `spans`, which is routinely negative and says nothing.
        """
        if not self.two:
            return None
        return self.seats[HIGH][0] - self.seats[LOW][1]

    def __repr__(self):
        return f"<Bands split={self.split} seats={self.seats}>"


def find(phrases, min_gap=MIN_GAP):
    """Detect the voice bands from the phrases themselves.

    The cut goes at the widest gap between neighbouring phrase medians, provided that gap is big
    enough to be a second singer rather than one singer's ordinary wandering. Everything else is
    reported as a single voice, which is a real answer and not a failure.
    """
    sung = [p for p in phrases if not p.silent]
    if len(sung) < 2:
        return Bands()
    middles = sorted(p.middle for p in sung)
    gap, at = 0, None
    for a, b in zip(middles, middles[1:]):
        if b - a > gap:
            gap, at = b - a, (a + b) / 2
    if gap < min_gap:
        return Bands()
    bands = Bands(split=at)
    for p in sung:
        p.band = LOW if p.middle < at else HIGH
    for name in (LOW, HIGH):
        mine = [p for p in sung if p.band == name]
        notes = [n for p in mine for n in p.pitches]
        middles = [p.middle for p in mine]
        bands.spans[name] = (min(notes), max(notes)) if notes else (0, 0)
        bands.seats[name] = (min(middles), max(middles)) if middles else (0, 0)
    for p in phrases:
        p.band = bands.of(p)
    return bands


#: How many octaves a move may travel. Four is far past any real band separation and only exists so
#: the search below terminates on a degenerate score.
REACH = 4


def octaves_to(phrase, band, bands):
    """Whole octaves that would move `phrase` into `band`. Zero when it is already there.

    Whole octaves because that is the only transposition that cannot change a pitch class, and the
    bands sit an octave apart anyway — see `notation.shift_octaves`.

    The candidates are *searched*, not computed, and the difference matters. Dividing the distance
    by twelve and rounding looks equivalent and is not: a phrase sitting exactly half an octave from
    the target centre gives `round(-0.5)`, which Python resolves to zero — banker's rounding — and
    the phrase stays in the voice it was supposed to leave. Nothing raises, the score still parses,
    and the only evidence is a line sung by the wrong singer. So the rule here is the one that was
    actually meant: consider only shifts that land on the correct side of the split, and among those
    take the one nearest the band's seat.
    """
    if phrase.silent or not bands.two or band not in bands.seats:
        return 0
    want = bands.centre(band)
    best = None
    for steps in range(-REACH, REACH + 1):
        landed = phrase.middle + 12 * steps
        if (landed < bands.split) != (band == LOW):
            continue
        distance = abs(landed - want)
        if best is None or distance < best[1]:
            best = (steps, distance)
    return best[0] if best else 0


def apply(score, moves):
    """Shift phrases by whole octaves. `moves` is [(phrase, octaves)]; zeroes are skipped.

    Edits go through `Score.replace`, so every line the caller did not name survives byte for byte.
    """
    done = 0
    for phrase, octaves in moves:
        if not octaves:
            continue
        score.replace(phrase.line, N.shift_octaves(score.lines[phrase.line], octaves))
        done += 1
    return done


def _mmss(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def report(score, phrases=None, bands=None):
    """The register map: every phrase, when it happens, how many syllables it holds, and who sings.

    This is the output to read before believing anything else the suite says, because it is the one
    that shows whether the score really has two separated voices or whether they have run together.
    """
    phrases = read(score) if phrases is None else phrases
    bands = find(phrases) if bands is None else bands
    out = [f"{score.meter[0]}/{score.meter[1]}  1/{score.unit.denominator}  {score.bpm} bpm  "
           f"{score.key}   bar {score.bar_seconds:.3f}s   "
           f"{score.bar_count()} bars = {_mmss(score.duration())}"]
    if bands.two:
        lo, hi = bands.seats[LOW], bands.seats[HIGH]
        sl, sh = bands.spans[LOW], bands.spans[HIGH]
        out.append(f"two voices: low sits {lo[0]:.0f}-{lo[1]:.0f} (notes {sl[0]}-{sl[1]}), "
                   f"high sits {hi[0]:.0f}-{hi[1]:.0f} (notes {sh[0]}-{sh[1]})")
        out.append(f"split at {bands.split:.1f}, seats {bands.separation():.0f} semitones apart — "
                   f"the note ranges overlap, the phrases do not")
    else:
        sung = [n for p in phrases for n in p.pitches]
        span = f"{min(sung)}-{max(sung)}" if sung else "silent"
        out.append(f"one voice ({span}) — no gap wide enough to be a second singer")
    out.append("")
    out.append(f"{'section':<12}{'from':>7}{'to':>7}{'bars':>5}{'syl':>5}{'range':>9}  voice")
    for p in phrases:
        start, end = p.seconds(score)
        if p.silent:
            out.append(f"{p.section:<12}{_mmss(start):>7}{_mmss(end):>7}{p.bars:>5}"
                       f"{'-':>5}{'-':>9}  (silent)")
            continue
        out.append(f"{p.section:<12}{_mmss(start):>7}{_mmss(end):>7}{p.bars:>5}{p.notes:>5}"
                   f"{f'{p.low}-{p.high}':>9}  {p.band or '-'}")
    return "\n".join(out)
