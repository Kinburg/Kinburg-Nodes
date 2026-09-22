"""A MIDI file → a YuE2 two-voice plan.

This is the first real control over the *melody*. Who sings a phrase is settled in the style string
and the plan gets no say ([[yue2-voice-bands]] has the measurements), but what they sing is written
right here — so a tune composed in a DAW, or exported from anywhere that speaks MIDI, can be handed
to YuE2 as the score it performs.

**Nothing here is guessed silently.** Every lossy step reports what it cost: how far notes moved to
reach the grid, how many were dropped to make a line monophonic, a tempo map reduced to one number.
A MIDI quantised in a DAW converts exactly; a live take does not, and the difference has to be
visible before it is audible.

The format's demands are specific and all of them are met here: two voices with equal bar counts in
every group, monophonic lines, note lengths drawn from a fixed set, accidentals written against the
key signature, and bars grouped one to four at a time. `notation.problems()` on the result is the
check that they were.

**A note crossing a barline is one syllable, not two.** ABC has to break it at the bar and join the
halves with a tie, and forgetting the tie is invisible in the text — it reads as two ordinary notes,
the singer is handed a syllable that does not exist, and everything after it in the line shifts. It
is the one error in this file worth naming, because the first draft made it.
"""
import re
from fractions import Fraction

from . import notation as N

#: Note lengths the format admits, longest first — `spend` takes the biggest that fits.
LENGTHS = sorted(N.SUPPORTED, reverse=True)
LETTERS = "CDEFGAB"

#: Grids to consider for `L:`. Coarser is better when it fits: a 1/32 grid makes every rhythm
#: representable and every line unreadable.
GRIDS = (4, 8, 16, 32)

#: How far, as a fraction of one grid unit, notes may sit off the grid on average before a finer
#: one is chosen. A tenth is comfortably inside what a DAW's own quantiser leaves behind.
SNAP = 0.1

#: Words in a track name that say what it carries. Checked before falling back to track order.
VOCAL_WORDS = ("vocal", "voc", "lead", "melody", "sing", "вокал", "голос", "мелод")
INS_WORDS = ("ins", "instr", "accomp", "backing", "chord", "инстр", "аккомп")

#: General MIDI's sixteen instrument families, one per eight programs. Enough to name a part usefully
#: — "Bass", "Strings" — without carrying a table of 128 patch names nobody reads.
FAMILIES = ("Piano", "Chromatic Percussion", "Organ", "Guitar", "Bass", "Strings", "Ensemble",
            "Brass", "Reed", "Pipe", "Synth Lead", "Synth Pad", "Synth FX", "Ethnic",
            "Percussive", "Sound FX")

#: General MIDI puts the drum kit on channel 10, counted from one. Everything on it is percussion,
#: and a plan that hands percussion to the Vocal voice asks the model to sing a drum part — which it
#: will do, at length.
DRUM_CHANNEL = 9

#: A vocal line breathes. Past this share of the song sounding without a break, a part is an
#: arrangement rather than something anyone sings, and saying so is the whole reason to measure it.
BUSY = 0.85


#: General MIDI's 128 programs, in order. The families above are enough to tell one part from
#: another in a list; these are what goes in a style string, where "distortion guitar" and "church
#: organ" are the difference between a rock song and a different rock song.
PROGRAMS = (
    "acoustic grand piano", "bright acoustic piano", "electric grand piano", "honky-tonk piano",
    "electric piano", "electric piano", "harpsichord", "clavinet",
    "celesta", "glockenspiel", "music box", "vibraphone",
    "marimba", "xylophone", "tubular bells", "dulcimer",
    "drawbar organ", "percussive organ", "rock organ", "church organ",
    "reed organ", "accordion", "harmonica", "tango accordion",
    "nylon guitar", "steel guitar", "jazz guitar", "clean electric guitar",
    "muted electric guitar", "overdriven guitar", "distortion guitar", "guitar harmonics",
    "acoustic bass", "electric bass", "electric bass", "fretless bass",
    "slap bass", "slap bass", "synth bass", "synth bass",
    "violin", "viola", "cello", "contrabass",
    "tremolo strings", "pizzicato strings", "orchestral harp", "timpani",
    "string ensemble", "string ensemble", "synth strings", "synth strings",
    "choir aahs", "voice oohs", "synth choir", "orchestra hit",
    "trumpet", "trombone", "tuba", "muted trumpet",
    "french horn", "brass section", "synth brass", "synth brass",
    "soprano sax", "alto sax", "tenor sax", "baritone sax",
    "oboe", "english horn", "bassoon", "clarinet",
    "piccolo", "flute", "recorder", "pan flute",
    "blown bottle", "shakuhachi", "whistle", "ocarina",
    "square lead", "sawtooth lead", "calliope lead", "chiff lead",
    "charang lead", "voice lead", "fifths lead", "bass and lead",
    "new age pad", "warm pad", "polysynth pad", "choir pad",
    "bowed pad", "metallic pad", "halo pad", "sweep pad",
    "rain synth", "soundtrack synth", "crystal synth", "atmosphere synth",
    "brightness synth", "goblins synth", "echoes synth", "sci-fi synth",
    "sitar", "banjo", "shamisen", "koto",
    "kalimba", "bagpipe", "fiddle", "shanai",
    "tinkle bell", "agogo", "steel drums", "woodblock",
    "taiko drum", "melodic tom", "synth drum", "reverse cymbal",
    "guitar fret noise", "breath noise", "seashore", "bird tweet",
    "telephone ring", "helicopter", "applause", "gunshot",
)

#: How many instruments a style string can carry before it stops describing anything. A band is four
#: or five things; a list of twelve is the arrangement, not its character.
MOST = 6


def _text(value):
    """A MIDI text field, decoded the way it was probably written.

    MIDI carries no encoding for its text, so `mido` hands back latin-1 and a Cyrillic track name
    arrives as mojibake — "Скрипка" reads as "Ñêðèïêà". The bytes are recoverable: re-encode and try
    UTF-8, which a modern file uses and which fails loudly on anything else, then cp1251, which is
    what Russian sequencers write. A name that is already plain ASCII survives both untouched, and
    anything that decodes to neither is left exactly as it came.
    """
    try:
        raw = value.encode("latin-1")
    except (AttributeError, UnicodeEncodeError):
        return value
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        out = raw.decode("cp1251")
    except UnicodeDecodeError:
        return value
    # Two Cyrillic letters in a row, not one. A latin-1 name like Cafe-with-an-accent becomes
    # 'Cafй' under cp1251 — a single stray Cyrillic character, which is an artefact rather than a
    # word, and accepting it would mangle a name that was never Russian to begin with.
    return out if re.search(r"[\u0400-\u04ff]{2}", out) else value


def read(path):
    """MIDI → (facts, {track name: [(start, end, pitch)]}, [(tick, label)]).

    Tempo, meter and key are taken from wherever they appear — a type-1 file keeps them in a
    conductor track, a type-0 file mixes them in with the notes, and both are ordinary.
    """
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError(
            "[Satyr Import] reading MIDI needs the 'mido' package, which is not installed in this "
            "ComfyUI's python. Install it with: python -m pip install mido") from exc
    try:
        midi = mido.MidiFile(str(path))
    except Exception as exc:
        raise RuntimeError(f"[Satyr Import] could not read {path!r} as a MIDI file: {exc}") from exc

    facts = {"ppq": midi.ticks_per_beat or 480, "bpm": 120, "meter": (4, 4), "key": "",
             "tempos": 0, "meters": 0, "drums": [], "played": []}
    tracks, markers = {}, []
    for index, track in enumerate(midi.tracks):
        at, sounding, name = 0, {}, ""
        # By CHANNEL, not by track. A type-0 file keeps an entire arrangement in one track and tells
        # the parts apart by their channel, so reading a track as one line hands bass, guitar and
        # violin to the singer at once — which YuE2 will cheerfully perform.
        notes, programs = {}, {}
        for msg in track:
            at += msg.time
            if msg.type == "set_tempo":
                facts["tempos"] += 1
                if facts["tempos"] == 1:
                    facts["bpm"] = max(1, round(mido.tempo2bpm(msg.tempo)))
            elif msg.type == "time_signature":
                facts["meters"] += 1
                if facts["meters"] == 1:
                    facts["meter"] = (msg.numerator, msg.denominator)
            elif msg.type == "key_signature" and not facts["key"]:
                facts["key"] = msg.key
            elif msg.type == "marker":
                markers.append((at, _text(msg.text).strip()))
            elif msg.type == "track_name" and not name:
                name = _text(msg.name).strip()
            elif msg.type == "program_change":
                programs.setdefault(msg.channel, msg.program)
            elif msg.type == "note_on" and msg.velocity > 0:
                sounding.setdefault((msg.channel, msg.note), []).append(at)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                started = sounding.get((msg.channel, msg.note))
                if started:
                    notes.setdefault(msg.channel, []).append((started.pop(0), at, msg.note))
        for channel, found in notes.items():
            label = name or f"track {index}"
            if len(notes) > 1:                            # one track, several parts: name them apart
                what = FAMILIES[programs[channel] // 8] if channel in programs else f"ch{channel + 1}"
                label = f"{label} / {what}" if name else what
            if channel == DRUM_CHANNEL:
                label += " [drums]"
                facts["drums"].append(label)
            elif channel in programs:
                facts["played"].append((len(found), PROGRAMS[programs[channel]]))
            if label in tracks:
                # Two channels of one track sharing a family name. A trailing space told them apart
                # and read like a typo; the channel number is what actually distinguishes them.
                label = f"{label} ch{channel + 1}"
            while label in tracks:
                label += " "
            tracks[label] = sorted(found)
    return facts, tracks, sorted(markers)


def choose(tracks, vocal="", ins=""):
    """Which track is the voice and which the instrument. → (vocal name, ins name, notes).

    A name is matched first, then an index, then the words a musician actually types into a track
    name, and only then the order they appear in. Order alone is a coin flip, and a coin flip about
    which line a song is sung on is not something to make quietly.
    """
    names, said = list(tracks), []

    def find(want, words, taken):
        want = str(want or "").strip()
        if want:
            for name in names:
                if name.lower() == want.lower() and name not in taken:
                    return name
            if want.lstrip("-").isdigit() and 0 <= int(want) < len(names):
                return names[int(want)]
            said.append(f"no track called {want!r} — the file has: {', '.join(names)}")
        for name in names:
            if name not in taken and any(w in name.lower() for w in words):
                return name
        # A drum kit is never a guess worth making, though it can still be asked for by name.
        playable = [n for n in names if n not in taken and "[drums]" not in n]
        return next(iter(playable), None)

    chosen_vocal = find(vocal, VOCAL_WORDS, set())
    chosen_ins = find(ins, INS_WORDS, {chosen_vocal})
    return chosen_vocal, chosen_ins, said


def pick_grid(facts, notes):
    """The `L:` denominator to quantise onto, and how far the notes had to move for it.

    Chosen from the music rather than fixed, because the two failures are opposite and both bad: a
    grid too coarse for the fastest run swallows notes, and one finer than the music needs writes an
    unreadable line and invites a rhythm nobody played. So the COARSEST grid that loses no note and
    leaves the average note close enough to it wins, and only if none qualifies does the finest one
    take it — with the drift reported, because at that point the drift is the story.
    """
    if not notes:
        return 16, 0.0
    measured = []
    for grid in GRIDS:
        unit = facts["ppq"] * 4 / grid
        drift, count, lost = 0.0, 0, 0
        for start, end, _ in notes:
            for tick in (start, end):
                drift += abs(tick / unit - round(tick / unit))
                count += 1
            if round(end / unit) <= round(start / unit):
                lost += 1
        measured.append((grid, drift / max(1, count), lost))
    for grid, drift, lost in measured:
        if not lost and drift <= SNAP:
            return grid, drift
    grid, drift, _ = measured[-1]
    return grid, drift


def monophonic(notes):
    """One note at a time. → (notes, how many were dropped, how many were cut short).

    The highest wins an overlap, which is the usual reading of a melody hidden in a chord. It is
    still a reading: a line written under a held pedal tone comes out as the pedal.
    """
    out, dropped, cut = [], 0, 0
    for start, end, pitch in sorted(notes, key=lambda n: (n[0], -n[2])):
        if out and start < out[-1][1]:
            if pitch <= out[-1][2]:
                dropped += 1
                continue
            out[-1] = (out[-1][0], start, out[-1][2])
            cut += 1
            if out[-1][1] <= out[-1][0]:
                out.pop()
                dropped += 1
                cut -= 1
        out.append((start, end, pitch))
    return [n for n in out if n[1] > n[0]], dropped, cut


#: A semitone offset and the glyph that writes it. A natural sign is needed whenever something
#: earlier in the bar, or the key signature, would otherwise alter the letter.
MARKS = {0: "=", 1: "^", -1: "_", 2: "^^", -2: "__"}


def spell(pitch, signature, bar=None):
    """A MIDI pitch → its ABC token, written against the key signature AND the bar so far.

    The bar matters, and leaving it out is a silent wrong note rather than a malformed score. An
    accidental in this notation holds for the rest of the measure and across octaves — after `^F`,
    every F in that bar sounds sharp — so a later natural F has to be written `=F` or it will be read
    a semitone high. Deciding each note against the key signature alone, which the first draft did,
    gets every note right until a bar contains two spellings of one letter.

    `bar` is the letters already altered in this measure and is updated in place. Pass nothing and
    the note is spelled against the key alone, which is right for a measure's first note.
    """
    bar = {} if bar is None else bar
    wanted = pitch % 12
    best = None
    for letter in LETTERS:
        for offset in (0, 1, -1, 2, -2):
            if (N.NATURAL[letter] + offset) % 12 != wanted:
                continue
            active = bar.get(letter, signature.get(letter, 0))
            plain = offset == active
            # Prefer a note that needs no accidental, then the smallest one, then C before B.
            rank = (0 if plain else 1, abs(offset), LETTERS.index(letter))
            if best is None or rank < best[0]:
                best = (rank, letter, offset, "" if plain else MARKS[offset])
    if best is None:
        return "C"                                        # unreachable for a 0-127 pitch
    _, letter, offset, mark = best
    if mark:
        bar[letter] = offset
    return _octave(mark, letter, pitch, N.NATURAL[letter] + offset)


def _octave(mark, letter, pitch, sounding):
    """The letter case and the comma/apostrophe marks that put a spelling in the right octave."""
    steps = (pitch - 60 - sounding) // 12
    if steps >= 1:
        return mark + letter.lower() + "'" * (steps - 1)
    return mark + letter + "," * (-steps)


def spend(units):
    """A length in units → the lengths the format admits, longest first, summing to it."""
    out = []
    while units > 0:
        for size in LENGTHS:
            if size <= units:
                out.append(size)
                units -= size
                break
        else:
            out.append(1)
            units -= 1
    return out


def _token(text, units, tie):
    return text + (str(units) if units != 1 else "") + ("-" if tie else "")


def render(events, width, signature):
    """One bar of (offset, length, pitch or None, ties onward) → its ABC text.

    The accidental state is born here and dies at the barline, which is exactly the scope the format
    gives it, so every note is spelled knowing what the ones before it in this measure did.
    """
    if not events:
        return f"z{width}"
    parts, at, bar = [], 0, {}
    for offset, length, pitch, onward in events:
        for piece in spend(offset - at):
            parts.append(_token("z", piece, False))
        text = "z" if pitch is None else spell(pitch, signature, bar)
        pieces = spend(length)
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            parts.append(_token(text, piece, False if pitch is None else (onward if last else True)))
            if pitch is not None:
                # the tied remainder repeats the letter, and the accidental is already in force
                text = spell(pitch, signature, bar)
        at = offset + length
    for piece in spend(width - at):
        parts.append(_token("z", piece, False))
    return "".join(parts)


def _bars(notes, width, unit_ticks):
    """Quantised notes → {bar: [(offset, length, pitch, ties onward)]}.

    A note that reaches past a barline is broken at it and the halves are joined with a tie, so the
    singer is given one syllable and not two. The pitch is carried through unspelled: how it is
    written depends on the bar it lands in, and that is not known until the bar is rendered.
    """
    out = {}
    for start, end, pitch in notes:
        first, last = round(start / unit_ticks), round(end / unit_ticks)
        if last <= first:
            last = first + 1
        at = first
        while at < last:
            bar, offset = divmod(at, width)
            take = min(width - offset, last - at)
            at += take
            out.setdefault(bar, []).append((offset, take, pitch, at < last))
    return out


def convert(path, vocal="", ins="", grid=0, group=4):
    """A MIDI file → (ABC text, [report line, ...]). The report is half the point."""
    facts, tracks, markers = read(path)
    said = [f"{len(tracks)} track(s) with notes: " +
            ", ".join(f"{name} ({len(notes)} notes, "
                      f"{min(p for _, _, p in notes)}-{max(p for _, _, p in notes)})"
                      for name, notes in tracks.items())]
    if not tracks:
        raise RuntimeError(f"[Satyr Import] {path!r} has no notes in any track.")

    vocal_name, ins_name, notes_said = choose(tracks, vocal, ins)
    said.extend(notes_said)
    said.append(f"Vocal <- {vocal_name or 'nothing'}; Ins <- {ins_name or 'nothing (silent)'}")
    if vocal_name and "[drums]" in vocal_name:
        said.append(f"{vocal_name!r} is the percussion channel. Its notes are drum numbers, not "
                    f"pitches, so this asks YuE2 to sing a drum kit — it will, and it is funny "
                    f"exactly once. Name the track that carries the tune instead")
    if len(tracks) > 2:
        said.append(f"{len(tracks)} parts to choose from and only two voices — name them in "
                    f"'vocal_track' and 'ins_track' rather than letting the order decide")

    voices, dropped, cut = {}, 0, 0
    for role, name in (("Vocal", vocal_name), ("Ins", ins_name)):
        picked, lost, short = monophonic(tracks.get(name, []))
        voices[role], dropped, cut = picked, dropped + lost, cut + short
    if dropped or cut:
        said.append(f"made monophonic: {dropped} note(s) dropped under a higher one, {cut} cut short")

    every = voices["Vocal"] + voices["Ins"]
    chosen, drift = (grid, 0.0) if grid else pick_grid(facts, every)
    if not grid:
        said.append(f"grid L:1/{chosen}, notes sitting {drift * 100:.0f}% of a unit off it on average"
                    + ("" if drift <= SNAP else " — this take was not quantised, so the written "
                                                "rhythm is an approximation of it"))
    num, den = facts["meter"]
    width = int(Fraction(num, den) / Fraction(1, chosen))
    unit_ticks = facts["ppq"] * 4 / chosen
    signature = N.key_accidentals(facts["key"])
    if facts["key"] and not N.knows_key(facts["key"]):
        said.append(f"key {facts['key']!r} is not one the format knows, so every accidental is "
                    f"written out and K: says C")
    if facts["tempos"] > 1:
        said.append(f"{facts['tempos']} tempo changes: a plan carries one Q:, so "
                    f"{facts['bpm']} bpm was taken from the first and the rest ignored")
    if facts["meters"] > 1:
        said.append(f"{facts['meters']} time signatures: {num}/{den} was taken from the first")

    laid = {role: _bars(picked, width, unit_ticks) for role, picked in voices.items()}
    total = max([b for bars in laid.values() for b in bars] or [-1]) + 1
    if not total:
        raise RuntimeError("[Satyr Import] nothing survived quantising — the file may be empty.")
    labels = {round(tick / unit_ticks) // width: text for tick, text in markers if text}
    said.append(f"{len(labels)} section label(s) from MIDI markers"
                if labels else "no MIDI markers, so the whole plan is one '% verse' section")

    lines = ["X:1", "T:", f"M:{num}/{den}", f"L:1/{chosen}", f"Q:1/4={facts['bpm']}",
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
             'V: Ins clef=treble name="Ins Melody" snm="Inst."',
             f"K:{facts['key'] or 'C'}"]
    bar, opened = 0, False
    while bar < total:
        if bar in labels:
            lines.append(f"% {labels[bar]}")
            opened = True
        elif not opened:
            lines.append("% verse")
            opened = True
        span = min(group, total - bar)
        for ahead in range(1, span):                      # never run a group past the next label
            if bar + ahead in labels:
                span = ahead
                break
        for role in ("Vocal", "Ins"):
            lines.append(f"V: {role}")
            written = [render(sorted(laid[role].get(bar + i, [])), width, signature)
                       for i in range(span)]
            empty = f"z{width}"
            lines.append("Z" + (str(span) if span > 1 else "") + "|" if all(w == empty for w in written)
                         else "|".join(written) + "|")
        bar += span
    text = "\n".join(lines) + "\n"
    said.extend(_sanity(voices["Vocal"], unit_ticks, total * width, vocal_name))
    said.append("no chord symbols: MIDI does not carry them, so use mode 'melody' in YuE2")
    return text, said


def _sanity(notes, unit_ticks, units, name):
    """Does the line handed to the singer look like something a person sings?

    A vocal line breathes, sits at a few notes to the bar, and rests between phrases. An
    arrangement does none of those, and a plan built from one asks YuE2 to sing the whole band —
    which it does, at length and with conviction. The failure is funny once and expensive after
    that, so it is worth one measurement before four minutes of rendering.
    """
    if not notes or not units:
        return []
    sounding = sum(end - start for start, end, _ in notes) / unit_ticks
    said = []
    if sounding / units >= BUSY:
        said.append(f"{name!r} sounds for {sounding / units * 100:.0f}% of the song without a "
                    f"real break. A sung line breathes; this looks like an arrangement, and YuE2 "
                    f"will try to sing all of it. Pick the track that carries the tune")
    per_bar = len(notes) / max(1.0, units / 8)
    if per_bar > 12:
        said.append(f"{name!r} averages {per_bar:.0f} notes per bar, far more than a singer gets "
                    f"through — check that this is the melody and not the accompaniment")
    return said


def style_of(facts):
    """What the MIDI itself can say about the style: the tempo and the band. → one string.

    YuE2 asks for `Language + Genre + Vocal Character + Tempo + Instruments` and a MIDI knows two of
    those exactly. The tempo is written in the file, and the General MIDI programs name every
    instrument in the arrangement — which is worth having, because the style is what YuE2 builds the
    accompaniment from, and guessing "electric guitar" when the file says "distortion guitar" is a
    different song.

    Nothing about language, genre or who is singing comes from here, because a MIDI does not know
    any of it. That half stays the author's, and `style_prefix` on the node is where it goes.

    Instruments are ordered by how much they play, so a four-note triangle does not displace the
    guitar, and the list stops at `MOST`: a style naming a dozen things has stopped describing
    anything.
    """
    bits = [f"{facts['bpm']} BPM"]
    num, den = facts["meter"]
    if (num, den) != (4, 4):
        bits.append(f"{num}/{den} time")
    seen = []
    for _, name in sorted(facts.get("played", []), key=lambda p: -p[0]):
        if name not in seen:
            seen.append(name)
    if facts.get("drums"):
        seen.append("drums")
    bits.extend(seen[:MOST])
    return ", ".join(bits)
