"""MIDI into a YuE2 plan: does the tune survive the trip, and is the cost reported.

Two properties carry the suite. The first is that **every pitch comes back**: convert, parse the ABC
with the pack's own reader, and the note list must equal the MIDI's. That one check covers the key
signature, the accidentals, the octave marks and the letter spelling at once, and none of them can be
wrong quietly — a pitch written a semitone or an octave off still produces a plan that parses.

The second is that **a note crossing a barline is one syllable**. ABC has to break it at the bar and
tie the halves; the first draft of the converter forgot the tie, which reads as two ordinary notes,
hands the singer a syllable that does not exist, and shifts every word after it. Nothing about the
text looks wrong. The check is that the attack count equals the note count.

Everything else here is about the lossy steps being counted rather than hidden: a chord flattened to
its top note, a live take snapped to a grid, a tempo map reduced to one number.
"""
import sys
import tempfile
from pathlib import Path

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "satyr")
N = load_module("kn.satyr.notation", "satyr/notation.py")
M = load_module("kn.satyr.midi", "satyr/midi.py")

check = Checker()
PPQ = 480
TEMP = Path(tempfile.mkdtemp(prefix="satyr_midi_"))


def write(name, tracks, meter=(2, 4), bpm=120, key="", markers=(), extra_tempo=()):
    """A MIDI file from [(track name, [(start, length, pitch)])]. Overlaps are allowed."""
    midi = MidiFile(ticks_per_beat=PPQ)
    head = MidiTrack()
    midi.tracks.append(head)
    head.append(MetaMessage("time_signature", numerator=meter[0], denominator=meter[1], time=0))
    head.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    if key:
        head.append(MetaMessage("key_signature", key=key, time=0))
    at = 0
    for tick, text in sorted(markers):
        head.append(MetaMessage("marker", text=text, time=tick - at))
        at = tick
    for extra in extra_tempo:
        head.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(extra), time=0))
    for label, notes in tracks:
        track = MidiTrack()
        midi.tracks.append(track)
        track.append(MetaMessage("track_name", name=label, time=0))
        events = []
        for start, length, pitch in notes:
            events.append((start, 1, Message("note_on", note=pitch, velocity=90, time=0)))
            events.append((start + length, 0, Message("note_off", note=pitch, velocity=0, time=0)))
        at = 0
        for tick, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
            msg.time = tick - at
            at = tick
            track.append(msg)
    path = TEMP / name
    midi.save(str(path))
    return str(path)


def sung(text, role="Vocal"):
    """The pitches a plan's voice actually starts — ties counted once, as a singer would."""
    score = N.parse(text)
    out = []
    for section in score.sections:
        for group in section.groups:
            line = group.vocal if role == "Vocal" else group.ins
            if line is None:
                continue
            out += [n.pitch for n in N.read_notes(score.lines[line], score.key)
                    if not n.rest and not n.tie_in]
    return out


def fails(call, word):
    """True when `call` raises a RuntimeError whose message contains `word`."""
    try:
        call()
    except RuntimeError as e:
        return word in str(e)
    return False


# ------------------------------------------------------------------------------------ reading
path = write("read.mid", [("Vocal", [(0, PPQ, 69), (PPQ, PPQ, 71)]),
                          ("Ins", [(0, PPQ, 45)])],
             meter=(3, 4), bpm=96, key="F#m", markers=[(0, "intro"), (PPQ * 3, "verse")])
facts, tracks, markers = M.read(path)
check("tempo is read", facts["bpm"] == 96, facts["bpm"])
check("meter is read", facts["meter"] == (3, 4), facts["meter"])
check("key is read", facts["key"] == "F#m", facts["key"])
check("tracks come back by name", sorted(tracks) == ["Ins", "Vocal"], sorted(tracks))
check("notes come back as spans", tracks["Vocal"] == [(0, PPQ, 69), (PPQ, PPQ * 2, 71)],
      tracks["Vocal"])
check("markers come back in order", [m[1] for m in markers] == ["intro", "verse"], markers)
check("a file that is not a MIDI file is explained", fails(lambda: M.read(__file__), "could not read"))


# ------------------------------------------------------------------------------------ choosing
picked = M.choose({"Piano": [], "Lead Vocal": [], "Strings": []})
check("a name that says vocal wins", picked[0] == "Lead Vocal", picked[0])
check("and the instrument is the first one left", picked[1] == "Piano", picked[1])
check("an explicit name is taken", M.choose({"a": [], "b": []}, vocal="b")[0] == "b")
check("so is an index", M.choose({"a": [], "b": []}, vocal="1")[0] == "b")
named = M.choose({"a": [], "b": []}, vocal="nope")
check("a name that matches nothing is reported", any("no track called" in s for s in named[2]),
      named[2])
check("and falls back rather than failing", named[0] == "a", named[0])
check("one track leaves the instrument empty", M.choose({"only": []})[1] is None)


# ------------------------------------------------------------------------------------ the grid
# Notes are (start, END, pitch) — the same shape `read` returns, not (start, length, pitch).
quarters = [(0, PPQ, 60), (PPQ, PPQ * 2, 62)]
sixteenths = [(0, PPQ // 4, 60), (PPQ // 4, PPQ // 2, 62)]
check("a coarse tune gets a coarse grid", M.pick_grid({"ppq": PPQ}, quarters)[0] == 4,
      M.pick_grid({"ppq": PPQ}, quarters))
check("a fast tune gets a fine one", M.pick_grid({"ppq": PPQ}, sixteenths)[0] == 16,
      M.pick_grid({"ppq": PPQ}, sixteenths))
check("and neither had to move", M.pick_grid({"ppq": PPQ}, quarters)[1] == 0.0)
drift = M.pick_grid({"ppq": PPQ}, [(11, PPQ + 11, 60), (PPQ + 19, PPQ * 2 + 19, 62)])[1]
check("a take off the grid is measured, not hidden", drift > 0, drift)
check("an empty part does not crash the guess", M.pick_grid({"ppq": PPQ}, [])[0] == 16)


# ------------------------------------------------------------------------------------ monophony
keep, dropped, cut = M.monophonic([(0, PPQ, 60), (0, PPQ, 64), (0, PPQ, 67)])
check("the top note wins a chord", [n[2] for n in keep] == [67], keep)
check("and the rest are counted", dropped == 2, dropped)
keep, dropped, cut = M.monophonic([(0, PPQ * 2, 60), (PPQ, PPQ * 2, 72)])
check("a lower note already sounding is cut short", [n[2] for n in keep] == [60, 72], keep)
check("and that is counted too", cut == 1, cut)


# ------------------------------------------------------------------------------------ spelling
# One note per bar first: this is the spelling on its own, with no accidental carried in.
for key in ("C", "F#m", "Bb", "D#m", "Eb"):
    every = list(range(48, 85))
    written = "|".join(M.spell(p, N.key_accidentals(key)) + "4" for p in every) + "|"
    check(f"every pitch survives being written in {key}",
          [n.pitch for n in N.read_notes(written, key) if not n.rest] == every, key)

# Then the case the first draft got wrong: an accidental holds to the END of its bar, and across
# octaves, so a later note of the same letter has to say what it is or be read a semitone out.
for key in ("C", "F#m", "Eb"):
    signature = N.key_accidentals(key)
    every = list(range(60, 73)) + list(range(72, 59, -1))
    bar = {}
    written = "".join(M.spell(p, signature, bar) + "4" for p in every) + "|"
    check(f"a whole chromatic bar survives in {key}",
          [n.pitch for n in N.read_notes(written, key) if not n.rest] == every, written[:40])


# ------------------------------------------------------------------------------------ lengths
for units in range(1, 60):
    pieces = M.spend(units)
    check(f"{units} units are spelled exactly", sum(pieces) == units, pieces) if units in (5, 7, 59) \
        else None
check("every piece is a length the format has",
      all(set(M.spend(n)) <= N.SUPPORTED for n in range(1, 200)))
check("nothing is spelled with more pieces than it needs", len(M.spend(48)) == 1, M.spend(48))


# ------------------------------------------------------------------------------------ converting
path = write("tune.mid", [("Vocal", [(0, PPQ // 2, 69), (PPQ // 2, PPQ // 2, 71),
                                     (PPQ, PPQ, 73), (PPQ * 2, PPQ, 74)]),
                          ("Ins", [(0, PPQ, 45), (PPQ, PPQ, 47)])], key="F#m")
text, said = M.convert(path)
score = N.parse(text)
check("the plan is well-formed", N.problems(score) == [], N.problems(score))
check("both voices are there", all(g.vocal is not None and g.ins is not None
                                   for s in score.sections for g in s.groups))
check("every vocal pitch survives", sung(text) == [69, 71, 73, 74], sung(text))
check("every instrumental pitch survives", sung(text, "Ins") == [45, 47], sung(text, "Ins"))
check("the header carries the tempo", score.bpm == 120, score.bpm)
check("and the key", score.key == "F#m", score.key)

# The regression: a note held across two barlines is ONE syllable.
path = write("held.mid", [("Vocal", [(0, PPQ, 60), (PPQ, PPQ * 3, 62), (PPQ * 4, PPQ, 64)])])
text, _ = M.convert(path)
check("a note across barlines stays one syllable", sung(text) == [60, 62, 64], sung(text))
check("and it is written with ties", "-" in text)
check("the plan is still well-formed", N.problems(N.parse(text)) == [])

# A length the format cannot write has to come out as tied pieces.
path = write("odd.mid", [("Vocal", [(0, PPQ * 5 // 4, 60), (PPQ * 5 // 4, PPQ * 3 // 4, 62)])])
text, _ = M.convert(path)
check("an unwritable length is spelled with a tie", sung(text) == [60, 62], sung(text))

# Markers become section labels; without them there is one section.
path = write("marked.mid", [("Vocal", [(0, PPQ, 60), (PPQ * 2, PPQ, 62)])],
             markers=[(0, "intro"), (PPQ * 2, "chorus")])
text, said = M.convert(path)
check("markers become sections", [s.label for s in N.parse(text).sections] == ["intro", "chorus"],
      [s.label for s in N.parse(text).sections])
path = write("plain.mid", [("Vocal", [(0, PPQ, 60)])])
text, said = M.convert(path)
check("no markers means one section", [s.label for s in N.parse(text).sections] == ["verse"])
check("and the report says so", any("no MIDI markers" in s for s in said), said)


# ------------------------------------------------------------------------------------ the report
path = write("messy.mid", [("Vocal", [(0, PPQ, 60), (0, PPQ, 64), (7, PPQ, 67)])],
             extra_tempo=[90, 140])
text, said = M.convert(path)
check("a flattened chord is reported", any("monophonic" in s for s in said), said)
check("tempo changes are reported", any("tempo changes" in s for s in said), said)
check("the grid and its drift are reported", any("grid L:1/" in s for s in said), said)
check("the missing chords are reported", any("no chord symbols" in s for s in said), said)
check("the tracks are listed so the next run can name one",
      any("track(s) with notes" in s for s in said), said)

path = write("silent.mid", [("Vocal", [])])
check("a file with no notes is explained", fails(lambda: M.convert(path), "no notes"))


# ------------------------------------------------------------------------------------ real files
# Everything below came from one real MIDI — a whole band arrangement of a rock song — which the
# converter cheerfully handed to the singer in its entirety. It sang the bass, the guitar and the
# drums. Three separate faults, all of them invisible in a plan that parses:

def type0(name, parts, bpm=145, track_name=""):
    """One track carrying several channels, which is what a type-0 file is and what most exports are.

    `parts` is [(channel, program or None, [(start, length, pitch)])].
    """
    midi = MidiFile(ticks_per_beat=PPQ, type=0)
    track = MidiTrack()
    midi.tracks.append(track)
    if track_name:
        track.append(MetaMessage("track_name", name=track_name, time=0))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    events = []
    for channel, program, notes in parts:
        if program is not None:
            events.append((0, 0, Message("program_change", channel=channel, program=program, time=0)))
        for start, length, pitch in notes:
            events.append((start, 1, Message("note_on", channel=channel, note=pitch,
                                             velocity=90, time=0)))
            events.append((start + length, 0, Message("note_off", channel=channel, note=pitch,
                                                      velocity=0, time=0)))
    at = 0
    for tick, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        msg.time = tick - at
        at = tick
        track.append(msg)
    path = TEMP / name
    midi.save(str(path))
    return str(path)


# 1. A Cyrillic track name arrives as mojibake, because MIDI declares no encoding and mido hands
#    back latin-1. "Скрипка" reads as "Ñêðèïêà" until the bytes are put back and tried as cp1251.
check("a cp1251 name is recovered", M._text("Скрипка".encode("cp1251").decode("latin-1")) == "Скрипка",
      M._text("Скрипка".encode("cp1251").decode("latin-1")))
check("a utf-8 name is recovered", M._text("Скрипка".encode("utf-8").decode("latin-1")) == "Скрипка")
check("plain ASCII is left alone", M._text("Lead Vocal") == "Lead Vocal")
check("a latin-1 name is not mangled into Cyrillic", M._text("Café") == "Café", M._text("Café"))

# 2. A type-0 file separates its parts by CHANNEL. Reading a track as one line hands the melody, the
#    bass and the drums to the singer at once — which is exactly what happened.
melody = [(i * PPQ, PPQ // 2, 69 + (i % 5)) for i in range(0, 16, 2)]
bass = [(i * PPQ // 2, PPQ // 2, 45 + (i % 3)) for i in range(32)]
drums = [(i * PPQ // 4, PPQ // 8, 36 + (i % 3)) for i in range(64)]
band = type0("band.mid", [(0, 40, melody), (1, 33, bass), (9, None, drums)], track_name="Violin")
facts, tracks, _ = M.read(band)
check("one track with three channels reads as three parts", len(tracks) == 3, sorted(tracks))
check("and each keeps only its own notes",
      sorted(len(n) for n in tracks.values()) == [8, 32, 64],
      sorted(len(n) for n in tracks.values()))
check("the General MIDI family names them",
      any("Strings" in n for n in tracks) and any("Bass" in n for n in tracks), sorted(tracks))
check("channel 10 is marked as drums", any("[drums]" in n for n in tracks), sorted(tracks))
check("and the file says which", facts["drums"], facts["drums"])

# 3. A drum kit is never a guess worth making.
picked = M.choose(tracks)
check("the drum channel is never auto-picked", "[drums]" not in (picked[0] or ""), picked[0])
check("the melody is", "Strings" in (picked[0] or ""), picked[0])
check("and the bass becomes the instrument", "Bass" in (picked[1] or ""), picked[1])
check("but it can still be asked for by name",
      M.choose(tracks, vocal=next(n for n in tracks if "[drums]" in n))[0].endswith("[drums]"))

text, said = M.convert(band)
check("a converted band still parses", N.problems(N.parse(text)) == [], N.problems(N.parse(text)))
check("and the choice is offered rather than assumed",
      any("only two voices" in s for s in said), said)

# 4. A part that never stops is an arrangement, not a line anyone sings.
_, said = M.convert(band, vocal=next(n for n in tracks if "Bass" in n))
check("a part that never breathes is reported", any("without a real break" in s for s in said), said)
_, said = M.convert(band, vocal=next(n for n in tracks if "[drums]" in n))
check("and picking the drum kit is reported", any("percussion channel" in s for s in said), said)
_, said = M.convert(band)
check("a melody with rests draws no complaint",
      not any("without a real break" in s for s in said), said)


# ------------------------------------------------------------------------------------ the style
# A MIDI answers two of the five things YuE2 asks for in a style string — the tempo and the band —
# exactly, from the file rather than from a guess. It has nothing to say about the other three.
check("the General MIDI table is complete", len(M.PROGRAMS) == 128, len(M.PROGRAMS))
check("and its names read like a style string", M.PROGRAMS[30] == "distortion guitar",
      M.PROGRAMS[30])

facts, tracks, _ = M.read(band)
style = M.style_of(facts)
check("the tempo comes from the file", style.startswith("145 BPM"), style)
check("the instruments are named, not guessed", "violin" in style and "electric bass" in style, style)
check("and the drums are counted as one of them", "drums" in style, style)
check("4/4 is not worth saying", "4/4" not in style, style)

odd = type0("odd.mid", [(0, 40, melody)], bpm=90)
odd_facts = M.read(odd)[0]
odd_facts["meter"] = (7, 8)
check("an unusual meter is", "7/8 time" in M.style_of(odd_facts), M.style_of(odd_facts))

# Ordered by how much each part actually plays, so a stray note does not displace the band.
crowded = type0("crowded.mid", [(0, 40, [(0, PPQ, 69)]),                       # violin, one note
                                (1, 30, [(i * PPQ // 2, PPQ // 2, 50 + i % 4)  # guitar, many
                                         for i in range(40)])])
loud = M.style_of(M.read(crowded)[0])
check("the part that plays most is named first",
      loud.index("distortion guitar") < loud.index("violin"), loud)
check("the list stops before it becomes an inventory",
      len(M.style_of({"bpm": 120, "meter": (4, 4),
                      "played": [(n, f"thing {n}") for n in range(20)],
                      "drums": []}).split(", ")) <= M.MOST + 2)
check("a file with no programs still gives the tempo",
      M.style_of({"bpm": 100, "meter": (4, 4), "played": [], "drums": []}) == "100 BPM")

check.done()
