"""A YuE2 plan as notes on a grid, and back — the half of Satyr Edit that understands ABC.

The editor (`web/satyr_edit.js`) never sees ABC. It gets a flat model — bars with their widths, notes
with an absolute start and length in `L:` units, chord changes — and sends the same shape back. All
that the format demands lives here, next to `notation`, which already reads it.

**What you did not touch comes back as the text it was.** A group whose bars are all still there, in
order, with the same notes, chords, key and meter, in the same key and meter context, is written out
as the lines it came from, byte for byte. Only what was edited is re-rendered, so a diff of a save
shows the bars that changed and nothing else, and the model gets its own text everywhere else.

**What is re-rendered is written the way ComfyUI's own exporter writes it** — the SheetSage2 → ABC code
in `comfy/audio_encoders/sheetsage2_abc.py`, whose output is what YuE2's plans imitate: at most four
bars to a group and a new group at every section, meter or key change; the chord in force restated at
every barline; key-relative spelling (`^^F`, not `=G`, in D# minor) through core's own `note_to_abc`;
a length the format cannot write split by core's own `_split_duration_units` and tied; an empty bar
folded to `Z`. The model learned that text, so that is the text an edit should look like.

**A note is one sung event, however it is spelled.** `d2-d2` across a barline is one note of length 4
here, and is split and tied again on the way out. A tie onto a different pitch is kept as a second
note marked `j` (joined): one syllable carried onto a new pitch, which counts as no new syllable.

**Chords follow the exporter's reading too.** It restates the chord at every barline, so a bar that
does not open with one has no chord there. The model keeps chord *changes* only — `""` is "no chord
from here" — and the restating is put back on the way out. Keys are changes as well, at any point:
the exporter writes one that lands on a barline as a `K:` line opening a group and one inside a bar
as `[K:]`, and so does this.
"""
from fractions import Fraction

from comfy.audio_encoders import sheetsage2_abc as core

from ..siren.score import _LETTERS, _syllable_chunks
from . import bands as B
from . import layout as L
from . import notation as N

VOICES = ("Vocal", "Ins")

#: YuE2's section vocabulary — SheetSage2's structure labels, the words its plans are written in.
LABELS = ("intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge", "interlude", "solo",
          "instrumental", "outro", "pre-outro", "fade-out", "rap", "theme", "development",
          "variation", "loop", "silence", "preshot", "irregular", "intro and verse",
          "verse and pre-chorus", "pre-chorus and chorus")


# ------------------------------------------------------------------------------------------ reading
def _width(meter, unit):
    """Units in one bar: M:4/4 at L:1/16 is 16. A meter the unit cannot divide is rounded up."""
    units = Fraction(meter[0], meter[1]) / unit
    return max(1, -(-units.numerator // units.denominator))


def _meter(text):
    num, _, den = str(text).strip().partition("/")
    return [int(num), int(den or 4)]


def _read_line(line, key):
    """One music line → its bars, and the key in force at its end.

    A bar is `{"used", "events", "chords", "keys", "key", "rests"}`: events are
    `[offset, units, pitch or None, tie_out]`, chords and inline key changes `[offset, text]`, `key`
    the key the bar opens in, `rests` true for a bar folded into `Zn` (it has no events and takes its
    width from the meter). Accidentals hold by letter across octaves until the barline, and an inline
    `[K:]` resets them, which is how YuE2's exporter writes both.
    """
    out = []
    for measure in N.split_measures(line):
        full = N.FULL_REST.match(measure.strip())
        if full:
            for _ in range(int(full.group(1) or 1)):
                out.append({"used": 0, "events": [], "chords": [], "keys": [], "key": key, "rests": True})
            continue
        bar = {"used": 0, "events": [], "chords": [], "keys": [], "key": key, "rests": False}
        signature, state, at = N.key_accidentals(key), {}, 0
        for m in N.ELEMENT.finditer(measure):
            if m.group("chord") is not None:
                bar["chords"].append([at, m.group("chord")])
                continue
            if m.group("key") is not None:
                key = m.group("key").strip()
                signature, state = N.key_accidentals(key), {}
                if at == 0:
                    bar["key"] = key
                else:
                    bar["keys"].append([at, key])
                continue
            letter, units = m.group("letter"), int(m.group("units") or 1)
            if letter == "z":
                bar["events"].append([at, units, None, False])
            else:
                name, acc = letter.upper(), m.group("acc")
                if acc:
                    state[name] = N.ACCIDENTAL[acc]
                offset = state.get(name, signature.get(name, 0))
                octaves = m.group("octave")
                pitch = (60 + N.NATURAL[name] + (0 if letter.isupper() else 12)
                         + 12 * octaves.count("'") - 12 * octaves.count(",") + offset)
                bar["events"].append([at, units, pitch, bool(m.group("tie"))])
            at += units
        bar["used"] = at
        out.append(bar)
    return out, key


class _Plan:
    """A plan read for editing: the model the editor gets, plus the bookkeeping `save` needs to
    write untouched groups back verbatim (which lines each group occupies, which bars it holds, and
    whether it declares its own meter or key)."""

    def __init__(self, text):
        self.text = str(text or "")
        self.score = score = N.parse(self.text)
        self.unit = score.unit
        self.bars, self.notes, self.chords, self.keys, self.sections, self.warnings = [], [], [], [], [], []
        self.groups, self.labels = [], []        # per original group / per original section
        starts = sorted([s.line for s in score.sections] + [g.first for s in score.sections for g in s.groups])
        self.header_end = starts[0] if starts else len(score.lines)

        meter, key, t = list(score.meter), score.key, 0
        last = {v: None for v in VOICES}         # the note a tie can extend, per voice
        chord, cur_key = "", None
        for si, section in enumerate(score.sections):
            label_line = section.line if score.lines[section.line].strip().startswith("%") else None
            self.labels.append(label_line)
            first = len(self.bars)
            for group in section.groups:
                has_m = has_k = False
                for i in group.prefix:
                    line = score.lines[i].strip()
                    if line.startswith("M:"):
                        meter, has_m = _meter(line[2:]), True
                    elif line.startswith("K:"):
                        key, has_k = line[2:].strip(), True
                vocal, key_after = _read_line(score.lines[group.vocal] if group.vocal is not None else "", key)
                ins, _ = _read_line(score.lines[group.ins] if group.ins is not None else "", key)
                count = max(len(vocal), len(ins))
                if len(vocal) != len(ins):
                    self.warnings.append(f"section '{section.label}': Vocal has {len(vocal)} bar(s) "
                                         f"and Ins {len(ins)} — the short voice is padded with rests")
                after = [n for n in starts if n > group.first]
                end = (after[0] - 1) if after else len(score.lines) - 1
                self.groups.append({"first": len(self.bars), "count": count, "lines": (group.first, end),
                                    "has_m": has_m, "has_k": has_k})
                nominal = _width(meter, self.unit)
                for j in range(count):
                    pair = {"Vocal": vocal[j] if j < len(vocal) else None, "Ins": ins[j] if j < len(ins) else None}
                    here = pair["Vocal"] or pair["Ins"]
                    width = max([nominal] + [p["used"] for p in pair.values() if p])
                    for name, p in pair.items():
                        if p and not p["rests"] and p["used"] != nominal:
                            self.warnings.append(f"bar {len(self.bars) + 1} ({name}) holds {p['used']} "
                                                 f"units where the meter says {nominal}")
                    self.bars.append({"w": width, "m": list(meter), "o": len(self.bars)})
                    if here["key"] != cur_key:
                        self.keys.append({"t": t, "k": here["key"]})
                        cur_key = here["key"]
                    for off, k in here["keys"]:
                        if k != cur_key:
                            self.keys.append({"t": t + off, "k": k})
                            cur_key = k
                    for v, name in enumerate(VOICES):
                        last[name] = self._take(pair[name], v, t, last[name])
                    vb = pair["Vocal"]
                    at_start = next((c for off, c in (vb["chords"] if vb else []) if off == 0), "")
                    if at_start != chord:
                        self.chords.append({"t": t, "c": at_start})
                        chord = at_start
                    for off, c in (vb["chords"] if vb else []):
                        if off > 0 and c != chord:
                            self.chords.append({"t": t + off, "c": c})
                            chord = c
                    if pair["Ins"] and pair["Ins"]["chords"]:
                        self.warnings.append(f"bar {len(self.bars)}: chord symbols in Ins are not part of "
                                             f"the format and are dropped if the bar is edited")
                    t += width
                key = key_after
            self.sections.append({"label": section.label, "n": len(self.bars) - first, "o": si})
        self.total = t

    def _take(self, bar, v, t, carry):
        """Append one bar's notes of voice `v` (starting at absolute `t`); returns the note a tie
        out of this bar would extend, or None."""
        if bar is None:
            return None
        for offset, units, pitch, tie_out in bar["events"]:
            if pitch is None:
                carry = None
                continue
            start = t + offset
            if carry is not None and carry["t"] + carry["d"] == start:
                if carry["p"] == pitch:
                    carry["d"] += units
                    carry = carry if tie_out else None
                    continue
                note = {"v": v, "t": start, "d": units, "p": pitch, "j": 1}
            else:
                note = {"v": v, "t": start, "d": units, "p": pitch, "j": 0}
            self.notes.append(note)
            carry = note if tie_out else None
        if bar["rests"]:
            return None
        return carry

    def model(self):
        """What the editor gets."""
        score = self.score
        bands = B.find(B.read(score))
        return {
            "unit": [self.unit.numerator, self.unit.denominator],
            "bpm": score.bpm,
            "meter": list(score.meter),
            "key": score.key,
            "bars": self.bars,
            "sections": self.sections,
            "notes": self.notes,
            "chords": self.chords,
            "keys": self.keys,
            "bands": ({"split": bands.split, "low": list(bands.seats[B.LOW]), "high": list(bands.seats[B.HIGH])}
                      if bands.two else None),
            "labels": list(LABELS),
            "warnings": self.warnings,
            "seconds": seconds(self.total, self.unit, score.bpm),
        }


def seconds(units, unit, bpm):
    """Duration of `units` of `L:` at `Q:1/4=bpm`."""
    return float(Fraction(units) * Fraction(unit) * 4 * Fraction(60, max(1, int(bpm))))


def load(text):
    """Plan text → the editor's model."""
    return _Plan(text).model()


# ------------------------------------------------------------------------------------- the edit
def _changes(events, total, field):
    """A list of `{"t", field}` changes, sorted, inside the song, the later of two at one point kept,
    and repeats of what is already in force dropped."""
    out, last = [], None
    for e in sorted(events or [], key=lambda e: int(e["t"])):
        t, value = int(e["t"]), str(e.get(field) or "").strip()
        if not 0 <= t < total or '"' in value or "]" in value:
            continue
        if out and out[-1]["t"] == t:
            out.pop()
            last = out[-1][field] if out else None
        if value != last:
            out.append({"t": t, field: value})
            last = value
    return out


def _clean(model):
    """The editor's model, checked and put in order. Raises ValueError on anything that cannot be
    written; repairs what a drag may legitimately leave behind (an overlap, a join with nothing to
    join to, a note past the end) and says so in the returned notes."""
    said = []
    bars = [{"w": int(b["w"]), "m": [int(b["m"][0]), int(b["m"][1])],
             "o": (int(b["o"]) if b.get("o") is not None else None)} for b in model.get("bars") or []]
    if not bars:
        raise ValueError("the plan has no bars left")
    if any(b["w"] < 1 for b in bars):
        raise ValueError("a bar has no length")
    sections = [{"label": " ".join(str(s.get("label", "")).split()), "n": int(s["n"]),
                 "o": (int(s["o"]) if s.get("o") is not None else None)} for s in model.get("sections") or []]
    if sum(s["n"] for s in sections) != len(bars) or any(s["n"] < 0 for s in sections):
        raise ValueError(f"the sections hold {sum(s['n'] for s in sections)} bars but the plan has {len(bars)}")
    total = sum(b["w"] for b in bars)

    notes, clipped = [], 0
    for n in model.get("notes") or []:
        v, t, d, p = int(n["v"]), int(n["t"]), int(n["d"]), int(n["p"])
        if v not in (0, 1) or not 0 <= p <= 127 or d < 1 or not 0 <= t < total:
            raise ValueError(f"a note cannot be written: {n}")
        if t + d > total:
            d, clipped = total - t, clipped + 1
        notes.append({"v": v, "t": t, "d": d, "p": p, "j": 1 if n.get("j") else 0})
    if clipped:
        said.append(f"{clipped} note(s) ran past the end and were shortened")
    trimmed = unjoined = 0
    for v in (0, 1):
        mine = sorted((n for n in notes if n["v"] == v), key=lambda n: n["t"])
        for a, b in zip(mine, mine[1:]):
            if a["t"] + a["d"] > b["t"]:
                a["d"], trimmed = b["t"] - a["t"], trimmed + 1
        prev = None
        for n in mine:
            if n["d"] < 1:
                continue
            if n["j"] and not (prev and prev["t"] + prev["d"] == n["t"]):
                n["j"], unjoined = 0, unjoined + 1
            prev = n
    notes = sorted((n for n in notes if n["d"] >= 1), key=lambda n: (n["v"], n["t"]))
    if trimmed:
        said.append(f"{trimmed} overlapping note(s) were cut short — a voice sings one note at a time")
    if unjoined:
        said.append(f"{unjoined} joined note(s) had nothing to join to and became new syllables")

    keys = _changes(model.get("keys"), total, "k")
    if not keys or keys[0]["t"] != 0:
        raise ValueError("the plan has no key at its start")
    bpm = int(model.get("bpm") or 0)
    if not 20 <= bpm <= 400:
        raise ValueError(f"tempo {bpm} is outside 20-400")
    return {"bars": bars, "sections": sections, "notes": notes, "bpm": bpm, "keys": keys,
            "chords": _changes(model.get("chords"), total, "c")}, said


def _timeline(events, field, starts, bars):
    """Per bar: `((0, value in force at the barline), (offset, change), ...)`."""
    out, value, i = [], "", 0
    for bi, s in enumerate(starts):
        end = s + bars[bi]["w"]
        while i < len(events) and events[i]["t"] <= s:
            value, i = events[i][field], i + 1
        row = [(0, value)]
        while i < len(events) and events[i]["t"] < end:
            value, i = events[i][field], i + 1
            row.append((events[i - 1]["t"] - s, value))
        out.append(tuple(row))
    return out


def _views(plan):
    """Per bar, everything that decides its text: `(w, meter, keys, chords, Vocal pieces, Ins pieces)`.

    `keys` and `chords` are the bar's timelines (what is in force at the barline, then each change); a
    piece is `(offset, units, pitch, tie_out)` — a note clipped to the bar, `tie_out` when it goes on
    sounding after the piece (into the next bar, or onto a joined note). Equal views render to equal
    text, which is how `save` knows a group is untouched.
    """
    bars, starts, t = plan["bars"], [], 0
    for b in bars:
        starts.append(t)
        t += b["w"]
    pieces = [([], []) for _ in bars]
    for v in (0, 1):
        mine = [n for n in plan["notes"] if n["v"] == v]
        for i, n in enumerate(mine):
            joined = i + 1 < len(mine) and mine[i + 1]["j"] and mine[i + 1]["t"] == n["t"] + n["d"]
            a, end = n["t"], n["t"] + n["d"]
            bi = _bar_at(starts, a)
            while a < end:
                b = min(end, starts[bi] + bars[bi]["w"])
                pieces[bi][v].append((a - starts[bi], b - a, n["p"], b < end or bool(joined)))
                a, bi = b, bi + 1
    keys = _timeline(plan["keys"], "k", starts, bars)
    chords = _timeline(plan["chords"], "c", starts, bars)
    return [(b["w"], tuple(b["m"]), keys[i], chords[i], tuple(pieces[i][0]), tuple(pieces[i][1]))
            for i, b in enumerate(bars)]


def _bar_at(starts, t):
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= t:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _accidentals(key):
    """A key as core's `note_to_abc` wants it: one offset per letter, C D E F G A B. Read through
    `notation`, which knows the same signatures and answers an unknown key with none rather than
    raising — a key the format cannot name is then spelled with explicit accidentals throughout."""
    table = N.key_accidentals(key)
    return [table.get(letter, 0) for letter in "CDEFGAB"]


def _render_bar(view, voice):
    """One bar of one voice, as core's exporter writes it."""
    width, _, keys, chords, vocal, ins = view
    pieces = vocal if voice == 0 else ins
    key_at = dict(keys)
    chord_at = dict(chords) if voice == 0 else {}
    cuts = {0, width}
    for off, units, _, _ in pieces:
        cuts.update((off, off + units))
    cuts.update(off for off in list(key_at) + list(chord_at) if 0 < off < width)
    cuts = sorted(cuts)
    accs, state, parts = _accidentals(keys[0][1]), {}, []
    for a, b in zip(cuts, cuts[1:]):
        prefix = ""
        if a > 0 and a in key_at:
            accs, state = _accidentals(key_at[a]), {}
            prefix += f"[K:{key_at[a]}]"
        if chord_at.get(a):
            prefix += f'"{chord_at[a]}"'
        piece = next((p for p in pieces if p[0] <= a < p[0] + p[1]), None)
        if piece is None:
            text, tie = "z", False
        else:
            text = core.note_to_abc(piece[2], accs, state)
            tie = b < piece[0] + piece[1] or piece[3]
        chunks = core._split_duration_units(b - a)
        for i, chunk in enumerate(chunks):
            more = text != "z" and (i + 1 < len(chunks) or tie)
            parts.append((prefix if i == 0 else "") + text + ("" if chunk == 1 else str(chunk)) + ("-" if more else ""))
    return "".join(parts)


def _render_voice(views, voice):
    rendered = [_render_bar(v, voice) for v in views]
    parts, i = [], 0
    while i < len(rendered):
        if not core._is_compressible_full_rest(rendered[i]):
            parts.append(rendered[i] + "|")
            i += 1
            continue
        j = i + 1
        while j < len(rendered) and core._is_compressible_full_rest(rendered[j]):
            j += 1
        parts.append("Z" + (str(j - i) if j - i > 1 else "") + "|")
        i = j
    return "".join(parts)


def _opens(view):
    """The key a bar opens in."""
    return view[2][0][1]


def _closes(view):
    """The key in force when a bar ends."""
    return view[2][-1][1]


def save(base, model):
    """The edited model → (plan text, report lines). Untouched groups keep their original lines."""
    old = _Plan(base)
    new, said = _clean(model)
    bars = new["bars"]
    views = _views(new)
    old_views = _views({"bars": old.bars, "notes": old.notes, "keys": old.keys, "chords": old.chords})
    lines = old.score.lines

    out = list(lines[:old.header_end])
    key0 = new["keys"][0]["k"]
    meter, key = list(old.score.meter), old.score.key
    for i, line in enumerate(out):
        s = line.strip()
        if s.startswith("Q:") and new["bpm"] != old.score.bpm:
            out[i] = line[:line.index("=") + 1] + str(new["bpm"]) if "=" in line else f"Q:1/4={new['bpm']}"
        elif s.startswith("K:") and key0 != old.score.key:
            out[i], key = f"K:{key0}", key0
        elif s.startswith("M:") and bars[0]["m"] != list(old.score.meter):
            out[i], meter = f"M:{bars[0]['m'][0]}/{bars[0]['m'][1]}", bars[0]["m"]

    by_origin = {g["first"]: g for g in old.groups}
    kept_groups = kept_bars = at = 0
    for section in new["sections"]:
        o = section["o"]
        if (o is not None and o < len(old.labels) and old.labels[o] is not None
                and section["label"] == old.sections[o]["label"]):
            out.append(lines[old.labels[o]])
        elif section["label"]:
            out.append(f"% {section['label']}")
        end, pending, i = at + section["n"], [], at
        while i < end:
            g = by_origin.get(bars[i]["o"])
            if g is not None and _verbatim(g, i, end, bars, views, old_views, meter, key):
                if pending:
                    meter, key = _emit(out, pending, views, bars, meter, key)
                    pending = []
                a, b = g["lines"]
                out.extend(lines[a:b + 1])
                i += g["count"]
                meter, key = bars[i - 1]["m"], _closes(views[i - 1])
                kept_groups, kept_bars = kept_groups + 1, kept_bars + g["count"]
                continue
            pending.append(i)
            i += 1
        if pending:
            meter, key = _emit(out, pending, views, bars, meter, key)
        at = end

    newline = "\r\n" if "\r\n" in old.text else "\n"     # a plan pasted on Windows keeps its line ends
    text = newline.join(out) + (newline if old.text.endswith("\n") or not old.text else "")
    total = sum(b["w"] for b in bars)
    report = [f"{len(bars)} bars · {_mmss(seconds(total, old.unit, new['bpm']))} · K:{key0} · Q:1/4={new['bpm']}",
              f"{kept_groups} group(s) kept as written, {len(bars) - kept_bars} bar(s) re-rendered"]
    report += said + [f"⚠ {p}" for p in N.problems(N.parse(text))]
    return text, report


def _verbatim(g, i, end, bars, views, old_views, meter, key):
    """May original group `g`, found at edited bar `i`, be written back as its own lines? Only if all
    its bars are still there, in order, inside one section, unchanged — and the meter and key in force
    before it are what its text assumes when it does not declare them itself."""
    n = g["count"]
    if i + n > end:
        return False
    for j in range(n):
        if bars[i + j]["o"] != g["first"] + j or views[i + j] != old_views[g["first"] + j]:
            return False
    if not g["has_m"] and bars[i]["m"] != list(meter):
        return False
    if not g["has_k"] and _opens(views[i]) != key:
        return False
    return True


def _emit(out, indices, views, bars, meter, key):
    """Write freshly rendered bars grouped the way core's exporter groups them — at most four, and a
    new group wherever the meter or the key changes at a barline. Returns the meter and key after."""
    groups, cur = [], []
    for i in indices:
        if cur and (len(cur) >= 4 or bars[i]["m"] != bars[cur[-1]]["m"]
                    or _opens(views[i]) != _closes(views[cur[-1]])):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    for grp in groups:
        head = grp[0]
        m_changed = bars[head]["m"] != list(meter)
        k_changed = _opens(views[head]) != key
        for v, name in enumerate(VOICES):
            out.append(f"V: {name}")
            if m_changed:
                out.append(f"M:{bars[head]['m'][0]}/{bars[head]['m'][1]}")
            if k_changed:
                out.append(f"K:{_opens(views[head])}")
            out.append(_render_voice([views[i] for i in grp], v))
        meter, key = bars[grp[-1]]["m"], _closes(views[grp[-1]])
    return meter, key


def _matched(holds, wants, pins):
    """Which lyric blocks each section carries: Satyr Score's `align`, run between the sections the
    editor pinned. A pin is the block a section sings, or -1 for a section that sings nothing.

    Pins are what the matcher cannot know. It goes by counts, so a plan that wrote no melody for the
    bridge hands the bridge's words to the chorus after it, and nothing it can count says otherwise.
    A pin that would run backwards against an earlier one is ignored; blocks with no free section
    between two pins ride with the pinned section before them. Returns (blocks per section, the
    sections whose pins took effect).
    """
    n_sec, n_blk = len(holds), len(wants)
    out = [[] for _ in range(n_sec)]
    anchors, last = [], -1
    for s in sorted(pins):
        b = pins[s]
        if 0 <= s < n_sec and isinstance(b, int) and 0 <= b < n_blk and b > last:
            anchors.append((s, b))
            out[s] = [b]
            last = b
    silent = {s for s, b in pins.items() if b == -1 and 0 <= s < n_sec}
    edges = [(-1, -1)] + anchors + [(n_sec, n_blk)]
    for (s0, b0), (s1, b1) in zip(edges, edges[1:]):
        secs = [s for s in range(s0 + 1, s1) if s not in silent]
        blocks = list(range(b0 + 1, b1))
        if secs:
            for s, idx in zip(secs, L.align([wants[b] for b in blocks], [holds[s] for s in secs])):
                out[s] = [blocks[i] for i in idx]
        elif blocks and s0 >= 0:
            out[s0] = out[s0] + blocks
        elif blocks and s1 < n_sec:
            out[s1] = blocks + out[s1]
    return out, {s for s, _ in anchors} | silent


def words(base, model, lyrics, voices):
    """Which lyric block each section carries and its syllables, the way Satyr Score lays them.

    The model is written out exactly as Save would write it and read with Satyr Score's own parts —
    its lyric reader, its block-to-section matcher, its voice bands, its findings — so the editor and
    that node's report agree about which words belong where, who sings them and in which register.
    Sections the editor pinned (`sings` on a section: a block index, or -1 for none) keep what they
    were given and the matcher works around them. A section's `pickup` is how many notes before its
    barline already sing its words. Every syllable knows its `block` (an index into `blocks`) and its
    `unit` — one run of lines by one voice, or one bracketed backing line — so the editor can show who
    the lyrics name for each stretch of notes. How the syllables then land on each section's notes the
    editor decides.
    """
    text, _ = save(base, model)
    score = N.parse(text)
    voices = [v for v in voices or [] if isinstance(v, dict)]
    found, said = L.blocks(lyrics or "", voices)
    table, voice_notes = L.voice_bands(voices)
    phrases = B.read(score)
    bands = B.find(phrases)
    parts = L.sections(score, phrases)
    pins = {i: s["sings"] for i, s in enumerate(model.get("sections") or [])
            if isinstance(s, dict) and isinstance(s.get("sings"), int)}
    pairing, held = _matched([L.held(ph, o, e) for _, ph, o, e in parts], [b.syllables for b in found], pins)
    spots = []
    for (label, ph, opens, lends), idx in zip(parts, pairing):
        spot = L.Spot(label, ph, [found[i] for i in idx], opens, lends)
        spot.was, spot.band = L._settled(spot), table.get(spot.voice)
        spots.append(spot)
    said = list(said) + voice_notes + L._findings(score, spots, bands)
    lines, sections, run = [], [], 0
    for si, spot in enumerate(spots):
        syllables = []
        for at, block in zip(pairing[si], spot.blocks):
            for unit in block.units:
                for line in unit.lines:
                    # A word with no vowel — "в", "з" — is no syllable of its own: it is sung with the
                    # next word, so it is shown on that word's first syllable (on the last one, at a
                    # line's end).
                    lone, opened = "", len(syllables)
                    for word in _LETTERS.findall(line):
                        pieces = _syllable_chunks(word)
                        if not pieces:
                            lone += word + " "
                            continue
                        for i, piece in enumerate(pieces):
                            syllables.append({"s": (lone if i == 0 else "") + piece, "word": lone + word, "first": i == 0,
                                              "last": i == len(pieces) - 1, "line": len(lines), "backing": unit.backing,
                                              "voice": unit.voice, "block": at, "unit": run})
                        lone = ""
                    if lone and len(syllables) > opened:
                        syllables[-1]["s"] += " " + lone.strip()
                    lines.append(line)
                run += 1
        sections.append({"label": spot.label, "blocks": [b.label for b in spot.blocks], "voice": spot.voice,
                         "band": spot.band, "was": spot.was, "wants": spot.wants, "holds": spot.holds,
                         "pickup": spot.opens.onsets if spot.opens else 0,
                         "syllables": syllables, "carries": pairing[si], "pinned": si in held})
    singers = {band: [name for name, b in table.items() if b == band] for band in (B.HIGH, B.LOW)}
    blocks = [{"label": b.label, "voice": next((u.voice for u in b.units if u.voice), ""), "syllables": b.syllables}
              for b in found]
    return {"sections": sections, "lines": lines, "singers": singers, "findings": said, "blocks": blocks}


def _mmss(sec):
    sec = max(0.0, float(sec))
    return f"{int(sec // 60)}:{sec % 60:04.1f}"
