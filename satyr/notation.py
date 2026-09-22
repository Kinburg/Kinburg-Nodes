"""YuE2's two-voice ABC, read for analysis and edited in place.

The score is the only conditioning YuE2 has with a time axis. Everything else it is given — the style
string, the lyrics — is one flat blob for the whole song, so *when* anything happens can only be said
here. That makes this module the floor the rest of the suite stands on.

**Parse to look, edit by line.** A full parse-and-reserialise would have to reproduce every spacing
and ordering choice the model made, and the first time it got one wrong it would corrupt a part of
the score nobody asked it to touch. So the text is kept verbatim and the parse is an *index* into
it: sections, groups and the line each one lives on. An edit rewrites those lines and leaves the rest
byte-identical, which is also what makes a diff of an edit readable.

**Octaves, not semitones.** `shift_octaves` is the only transposition offered, and that is a
deliberate limit rather than an unfinished one. The thing worth moving a phrase for is its voice
band, and the bands sit an octave apart; meanwhile an arbitrary transposition has to respell every
note against the key signature, where a wrong accidental is both easy to write and inaudible in the
code review. An octave shift is a letter-case change and cannot change a pitch class.

**A tie is one syllable.** `attacks()` counts note onsets, not note glyphs: `d2-d2` is one sung
syllable held across a barline, and counting it twice is how a phrase silently acquires a syllable
that the singer then has to steal from the next line.

Two invariants of the format are worth knowing before editing anything. Both voices carry the same
number of bars in every group — `Z4` against four written bars is the normal way that is spelled —
and chord symbols live only in the Vocal voice. `parse` does not enforce either, because a parse
that refuses to read a malformed score is useless for diagnosing one; `problems()` reports them
afterwards, for a caller that wants to know before it starts editing.
"""
import re
from fractions import Fraction

#: Where a natural letter sits above C, in semitones.
NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

#: Sharps and flats enter the key signature in these orders.
SHARP_ORDER = "FCGDAE B".replace(" ", "")
FLAT_ORDER = "BEADGCF"

#: key name → signed accidental count (+n sharps, -n flats). Minor keys carry the `m`.
_MAJOR = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
          "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7}
_MINOR = {"A": 0, "E": 1, "B": 2, "F#": 3, "C#": 4, "G#": 5, "D#": 6, "A#": 7,
          "D": -1, "G": -2, "C": -3, "F": -4, "Bb": -5, "Eb": -6, "Ab": -7}

#: An accidental glyph and what it does to a pitch. `None` means "no accidental written".
ACCIDENTAL = {"": None, "^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}

#: Every note length the format admits, counted in `L:` units. Anything else has to be spelled as a
#: tie — `C8-C2` rather than an invented `C10` — so a rewrite that changes a duration has to land on
#: one of these or not happen at all.
SUPPORTED = frozenset({1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48})

#: Chord symbol, inline key change, or one note/rest. Ordered so a quoted string wins over the
#: letters inside it — `"Bmaj7"` must never be read as a B followed by a bar of nonsense.
ELEMENT = re.compile(
    r'"(?P<chord>[^"]*)"'
    r"|\[K:(?P<key>[^\]]+)\]"
    r"|(?P<acc>[_=^]*)(?P<letter>[A-Ga-gz])(?P<octave>[,']*)(?P<units>\d*)(?P<tie>-?)"
)

#: A whole-measure rest: `Z` for one bar, `Z4` for four.
FULL_REST = re.compile(r"^Z(\d*)$")

_HEADER_KEYS = ("X:", "T:", "M:", "L:", "Q:", "K:")


def knows_key(key):
    """Is this a key signature the format can write? C major has no accidentals and is still known —
    the distinction matters, because an empty signature is otherwise indistinguishable from a key
    nobody recognised."""
    name = str(key or "").strip()
    table, core = (_MINOR, name[:-1]) if name.endswith("m") else (_MAJOR, name)
    return core.strip() in table


def key_accidentals(key):
    """Key name → {letter: semitone offset} for the letters the signature touches.

    `K:D#m` is six sharps, so F C G D A E are all raised and B is not. Unknown or modal keys fall
    back to no accidentals, which is wrong but quiet: a pitch read a semitone off still lands in the
    right octave, and the octave is what the voice bands are about.
    """
    name = str(key or "").strip()
    table, core = (_MINOR, name[:-1]) if name.endswith("m") else (_MAJOR, name)
    count = table.get(core.strip())
    if count is None:
        return {}
    if count > 0:
        return {letter: 1 for letter in SHARP_ORDER[:count]}
    return {letter: -1 for letter in FLAT_ORDER[:-count]}


class Note:
    """One sung event. `tie_in` marks a continuation — the same syllable, still sounding."""

    __slots__ = ("pitch", "units", "tie_out", "tie_in", "rest")

    def __init__(self, pitch, units, tie_out, tie_in, rest):
        self.pitch, self.units = pitch, units
        self.tie_out, self.tie_in, self.rest = tie_out, tie_in, rest

    def __repr__(self):
        return f"<rest {self.units}>" if self.rest else f"<{self.pitch} {self.units}{'-' if self.tie_out else ''}>"


def read_notes(line, key=""):
    """One music line → its notes and rests, in order, with MIDI pitches.

    Accidentals propagate the way YuE2's own exporter writes them: by LETTER across octaves, reset at
    every barline. That differs from some ABC readers, which scope an accidental to one octave, and
    getting it wrong would move a note by a semitone — harmless for banding, but this is also what
    the linter reports, so it should be right.
    """
    signature = key_accidentals(key)
    notes, tie_pending = [], False
    for measure in split_measures(line):
        bar = {}
        full = FULL_REST.match(measure.strip())
        if full:
            tie_pending = False
            continue
        for m in ELEMENT.finditer(measure):
            if m.group("chord") is not None or m.group("key") is not None:
                continue
            letter, acc = m.group("letter"), m.group("acc")
            units = int(m.group("units") or 1)
            tie_out = bool(m.group("tie"))
            if letter == "z":
                notes.append(Note(None, units, False, False, True))
                tie_pending = False
                continue
            name = letter.upper()
            if acc:
                offset = ACCIDENTAL[acc]
                bar[name] = offset
            else:
                offset = bar.get(name, signature.get(name, 0))
            octaves = m.group("octave")
            pitch = (60 + NATURAL[name] + (0 if letter.isupper() else 12)
                     + 12 * octaves.count("'") - 12 * octaves.count(",") + offset)
            notes.append(Note(pitch, units, tie_out, tie_pending, False))
            tie_pending = tie_out
    return notes


def attacks(line, key=""):
    """How many syllables this line can carry: onsets only, ties counted once."""
    return sum(1 for n in read_notes(line, key) if not n.rest and not n.tie_in)


def pitches(line, key=""):
    """Every sounding pitch on the line, ties included — this is what a band is measured from."""
    return [n.pitch for n in read_notes(line, key) if not n.rest]


def split_measures(line):
    """Measures of one music line, `Z4` still folded. Empty trailing piece dropped."""
    return [m for m in str(line).split("|") if m.strip()]


def bars(line):
    """Bar count of one music line, with `Zn` expanded. Both voices must agree on this."""
    total = 0
    for measure in split_measures(line):
        full = FULL_REST.match(measure.strip())
        total += int(full.group(1) or 1) if full else 1
    return total


def shift_octaves(line, octaves):
    """Move every note on a music line by whole octaves, leaving everything else untouched.

    Chord symbols, inline key changes, rests, durations and ties all survive verbatim. The rest is
    the one that has to be defended: `z` is a beat of silence and `Z` is a whole bar of it, so an
    upper-casing that treated them as the same letter would turn a quarter rest into four empty bars
    and move every following note in the song. Hence rests return before the case is ever touched.
    """
    if not octaves:
        return line

    def move(m):
        if m.group("chord") is not None or m.group("key") is not None:
            return m.group(0)
        letter = m.group("letter")
        if letter == "z":
            return m.group(0)
        marks = m.group("octave")
        # An ABC octave is spelled three ways — commas below, an implied middle, apostrophes above —
        # so a shift walks that ladder a step at a time rather than doing arithmetic on it.
        steps = marks.count("'") - marks.count(",") + (1 if letter.islower() else 0) + octaves
        letter = letter.lower() if steps >= 1 else letter.upper()
        rung = steps - 1 if steps >= 1 else steps
        marks = "'" * rung if rung > 0 else "," * -rung
        return m.group("acc") + letter + marks + m.group("units") + m.group("tie")

    return ELEMENT.sub(move, line)


class Group:
    """One aligned pair of music lines: the Vocal's bars and the Ins's bars over the same span.

    `span` is every line the group occupies — its two `V:` headers, any `M:`/`K:` they carry, and the
    two music lines — because a group is the smallest thing that can be REMOVED from a score. Take
    out a music line without its header and the next group inherits a stray `V: Vocal`; take out one
    voice without the other and the two run out of step from there to the end of the song.
    """

    __slots__ = ("vocal", "ins", "prefix", "first", "last")

    def __init__(self, first=0):
        self.vocal = None      # line index into Score.lines, or None
        self.ins = None
        self.prefix = []       # indices of any M:/K: lines that opened this group
        self.first = first     # the `V:` line that opens the group
        self.last = first      # the last line belonging to it

    @property
    def span(self):
        return range(self.first, self.last + 1)

    def __repr__(self):
        return f"<Group vocal={self.vocal} ins={self.ins} lines {self.first}-{self.last}>"


class Section:
    """A `% label` and everything under it until the next one."""

    __slots__ = ("label", "line", "groups")

    def __init__(self, label, line):
        self.label, self.line, self.groups = label, line, []

    def __repr__(self):
        return f"<Section {self.label!r} groups={len(self.groups)}>"


class Score:
    """A parsed YuE2 score: the original lines, the header numbers, and an index of the body."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.meter = (4, 4)
        self.unit = Fraction(1, 8)
        self.bpm = 120
        self.key = ""
        self.sections = []

    # ------------------------------------------------------------------------------ time
    @property
    def bar_seconds(self):
        """Seconds per bar. `M:` says how much of a whole note a bar is, `Q:` how fast a quarter is.

        This is why a YuE2 plan can state a duration where an AceStep plan could only wish for one.
        """
        numerator, denominator = self.meter
        return float(Fraction(numerator, denominator) * 4 * Fraction(60, self.bpm))

    def bar_count(self):
        """Bars in the song, read off the Vocal voice. Ins is required to match and is not asked."""
        return sum(bars(self.lines[g.vocal]) for s in self.sections for g in s.groups
                   if g.vocal is not None)

    def duration(self):
        """Seconds the score describes. The model shortens this by a few percent when it performs."""
        return self.bar_count() * self.bar_seconds

    # ------------------------------------------------------------------------------ output
    def text(self):
        return "\n".join(self.lines) + "\n"

    def replace(self, index, line):
        self.lines[index] = line


def parse(text):
    """ABC text → `Score`. Tolerant on purpose: unknown lines are kept and simply not indexed.

    A score that a linter would reject still has to be readable here, because the reason to parse a
    broken score is to find out how it is broken.
    """
    score = Score(str(text or "").splitlines())
    section = None
    voice = None
    group = None
    for i, raw in enumerate(score.lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            section = Section(line[1:].strip(), i)
            score.sections.append(section)
            voice, group = None, None
            continue
        if line.startswith("V:"):
            rest = line[2:].strip()
            name = rest.split()[0] if rest else ""
            if "clef=" in rest or "name=" in rest:      # a header declaration, not a body switch
                continue
            voice = name
            if voice == "Vocal" or group is None:
                group = Group(i)
                if section is None:                      # a body that opened without a `%` label
                    section = Section("", i)
                    score.sections.append(section)
                section.groups.append(group)
            group.last = i
            continue
        if line.startswith(_HEADER_KEYS):
            if line.startswith("M:") and section is None:
                num, _, den = line[2:].partition("/")
                score.meter = (int(num), int(den))
            elif line.startswith("L:") and section is None:
                num, _, den = line[2:].partition("/")
                score.unit = Fraction(int(num), int(den))
            elif line.startswith("Q:") and section is None:
                score.bpm = int(float(line.split("=")[-1]))
            elif line.startswith("K:") and section is None:
                score.key = line[2:].strip()
            elif group is not None:
                group.prefix.append(i)
                group.last = i
            continue
        if group is not None and voice:
            if voice == "Vocal":
                group.vocal = i
            elif voice == "Ins":
                group.ins = i
            group.last = i
    return score


def drop(score, groups):
    """A new `Score` with those groups gone, lines and all. The original is left alone.

    Whole groups, never single lines: a group is one span of the song carried by both voices at once,
    and removing half of it desynchronises them from that point to the end. Everything outside the
    dropped spans — the header, the section labels, every other group — comes through byte for byte,
    so a diff shows the bars that went and nothing else.
    """
    doomed = {i for group in groups for i in group.span}
    if not doomed:
        return parse(score.text())
    kept = [line for i, line in enumerate(score.lines) if i not in doomed]
    # A `%` label whose every group has gone would head an empty section; drop it too.
    text = "\n".join(kept) + "\n"
    fresh = parse(text)
    empty = {section.line for section in fresh.sections if not section.groups}
    if empty:
        text = "\n".join(l for i, l in enumerate(fresh.lines) if i not in empty) + "\n"
        fresh = parse(text)
    return fresh


def problems(score):
    """What is structurally wrong with a score, as a list of sentences. Empty means well-formed.

    These are the two invariants YuE2's own exporter always honours, so a score that breaks one was
    either hand-edited or written by something that did not know the format. Worth knowing BEFORE
    editing, because both failures are the kind that read as musical mistakes rather than structural
    ones: unequal bar counts silently desynchronise the voices from the group onward, and a chord
    symbol in the Ins voice is harmony written where nothing reads it.
    """
    found = []
    for section in score.sections:
        where = f"section '{section.label}'" if section.label else "the opening section"
        for n, group in enumerate(section.groups, 1):
            if group.vocal is None or group.ins is None:
                missing = "Ins" if group.vocal is not None else "Vocal"
                found.append(f"{where}, group {n}: no {missing} line — the voices cannot be lined up")
                continue
            vocal, ins = bars(score.lines[group.vocal]), bars(score.lines[group.ins])
            if vocal != ins:
                found.append(f"{where}, group {n}: Vocal has {vocal} bars and Ins has {ins} — "
                             f"every group must carry the same count in both voices")
            if '"' in score.lines[group.ins]:
                found.append(f"{where}, group {n}: a chord symbol is written in the Ins voice, "
                             f"where nothing reads it — harmony belongs on Vocal")
    return found
