"""The subtitle files: SubStation Alpha with real karaoke, and a plain SubRip beside it.

**Why `.ass` and not just the `.srt` `Save Clip` already writes.** SubRip has one unit — a block of
text between two timestamps — so the best it can do with word timings is flash one word at a time.
ASS has a karaoke primitive (`\\k`) that colours a line *as it is sung*, which is the thing the
author asked for, and per-line styles, which is where one colour per singer lives. Both are read by
every player worth the name, and by `libass`, which is what burns them into a frame later.

Writing the file rather than only burning pixels is deliberate: a `.ass` next to the mp4 can be
restyled — font, size, colours, position — and re-checked in a player in **seconds**, with no
re-encode. That is what makes a wrong alignment cheap to discover.

Three details of the format that are easy to get wrong and expensive to debug, so they are each
pinned by a test:

* **Colour is `&HAABBGGRR`** — blue first, red last, and `AA` is *transparency*, not opacity, so an
  opaque colour ends in `00` at the front. A hex code copied straight from a colour picker comes out
  with red and blue swapped, which looks like a palette bug and is a byte order.
* **`PrimaryColour` is the colour a word becomes once sung**, `SecondaryColour` the colour it waits
  in. The names suggest the opposite and half the karaoke scripts on the internet have them the
  wrong way round.
* **`\\k` is in centiseconds and the values must add up to the event's own length.** Rounding each
  word on its own drifts by up to half a centisecond per word, which over a long line is a visible
  lag; so the durations are taken as *differences of rounded absolute times*, which cannot drift.

Pure stdlib. No torch, no PIL, no model.
"""
import re

#: Fallback colours, in first-appearance order, for voices nobody assigned one to. Chosen to stay
#: apart on a photographic background and to survive the yuv420p chroma subsampling an mp4 does to
#: them — heavily saturated reds and blues bleed at 4:2:0, so these sit off the corners of the gamut.
PALETTE = ["#FFD24A", "#7FD8FF", "#FF8FB1", "#9CE68A", "#C9A6FF", "#FFB27F", "#8AE0D0", "#E8E0C8"]

#: What an unsung word waits in, and what the outline is. Both overridable per track.
DEFAULT_IDLE = "#F2F2F2"
DEFAULT_OUTLINE = "#101010"

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")
#: `Nina = #ff66cc`, `Nina: ff66cc`, `Nina - #ff66cc`. The separator is loose on purpose — this is
#: typed into a node widget by hand, and rejecting a colon is not a useful thing to do to somebody.
_MAP_ROW = re.compile(r"^\s*(.+?)\s*[=:—-]\s*(#?[0-9a-fA-F]{6})\s*$")


def parse_colours(text):
    """`'Nina = #ff66cc'` lines -> `({name_lowered: '#rrggbb'}, notes)`.

    Names are matched case- and space-insensitively, because the same singer is `Nina` in the cast
    and `nina` in a plan row and nobody should have to care.
    """
    out, notes = {}, []
    for lineno, raw in enumerate(str(text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        m = _MAP_ROW.match(line)
        if not m:
            notes.append(f"colour line {lineno} is not 'Name = #rrggbb' and was skipped: {line[:40]!r}")
            continue
        out[_key(m.group(1))] = "#" + m.group(2).lstrip("#").lower()
    return out, notes


def _key(name):
    return " ".join(str(name or "").split()).lower()


def assign_colours(voices, mapping=None):
    """Every voice -> a colour: the one it was given, else the next one off the palette.

    Order matters and is the order the voices are handed in, which the caller takes from the song
    rather than from a `set` — so the lead singer gets the first palette colour on every render of
    the same track instead of a different one each time a hash seed changes.
    """
    mapping = mapping or {}
    out, spare = {}, 0
    for name in voices:
        k = _key(name)
        if not k or k in out:
            continue
        if k in mapping:
            out[k] = mapping[k]
        else:
            out[k] = PALETTE[spare % len(PALETTE)]
            spare += 1
    return out


def ass_colour(hex_colour, alpha=0):
    """`'#ff66cc'` -> `'&H0000CCFF'` — ASS's `AABBGGRR`, alpha being TRANSPARENCY (0 = opaque)."""
    m = _HEX.match(str(hex_colour or "").strip())
    rgb = m.group(1) if m else "FFFFFF"
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    return f"&H{max(0, min(255, int(alpha))):02X}{b}{g}{r}".upper()


def ass_time(seconds):
    """`0:01:02.34` — ASS's clock, which is centiseconds and a single-digit hour.

    Rounded to whole centiseconds first, so 59.999 s carries into the minute instead of writing
    `0:00:60.00`, which players read as an event that never starts.
    """
    cs = int(round(max(0.0, float(seconds)) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def ass_text(text):
    """Escape one run of lyric for an ASS event.

    `{` opens an override block, so a lyric containing one would silently swallow the rest of the
    line; the format's own escape for it is `\\{`. A literal newline ends the event, so it becomes
    `\\N`, and a run of spaces collapses unless it is hard-spaced.
    """
    out = (str(text or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
           .replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\N"))
    return out


def _karaoke(line, lead_in=0.0, sweep=False):
    """One line's `{\\k…}` runs, plus the event's start and end.

    The event starts `lead_in` seconds before the first word so the line can be READ before it is
    sung — with a leading `\\k` of exactly that length, so nothing is highlighted during the wait.

    `sweep` picks `\\kf` (the colour fills across the word) over `\\k` (it flips at the word's
    start). Fill reads better on held notes and worse at speed, which is why it is a choice.
    """
    words = [w for w in line.get("words") or [] if w.get("text")]
    if not words:
        return "", float(line.get("start", 0.0)), float(line.get("end", 0.0))

    tag = "kf" if sweep else "k"
    start = max(0.0, float(words[0]["start"]) - max(0.0, float(lead_in)))
    end = float(words[-1]["end"])

    # Absolute centiseconds from the event's start, so every duration below is a DIFFERENCE of two
    # rounded numbers and the runs are guaranteed to sum to the event's own length.
    def cs(t):
        return int(round((max(start, float(t)) - start) * 100))

    out, at = [], 0
    if cs(words[0]["start"]) > 0:
        out.append("{\\%s%d}" % (tag, cs(words[0]["start"])))
        at = cs(words[0]["start"])
    for i, w in enumerate(words):
        # A gap between two words is its own silent run, or the next word's highlight would start
        # early and the sweep would run ahead of the singer.
        gap = cs(w["start"]) - at
        if gap > 0:
            out.append("{\\%s%d}" % (tag, gap))
            at += gap
        dur = max(0, cs(w["end"]) - at)
        out.append("{\\%s%d}%s" % (tag, dur, ass_text(w["text"])))
        at += dur
    return "".join(out), start, end


def _style_row(name, colour, idle, outline, font, size, play_h, alignment=2, margin_v=None):
    """One `Style:` row. `BorderStyle 1` is outline+shadow, `Alignment 2` is bottom-centre."""
    return ("Style: {n},{f},{sz},{pri},{sec},{out},&H80000000,-1,0,0,0,100,100,0,0,1,"
            "{ow:.1f},{sh:.1f},{al},40,40,{mv},1").format(
        n=name, f=font, sz=int(size),
        pri=ass_colour(colour), sec=ass_colour(idle), out=ass_colour(outline),
        ow=max(1.0, size / 22.0), sh=max(0.0, size / 44.0), al=int(alignment),
        mv=int(margin_v if margin_v is not None else max(24, play_h * 0.06)))


def ass(track, colours=None, width=1920, height=1080, font="Arial", size=None,
        idle=DEFAULT_IDLE, outline=DEFAULT_OUTLINE, lead_in=0.35, sweep=True, alignment=2):
    """A whole `.ass` file, as text.

    `track` is Echo's timing structure — `{"lines": [{start, end, text, voice, words: [...]}]}`.
    One style per voice, named after the voice, so the file stays editable by a human afterwards:
    changing every one of Nina's lines is one row at the top, not a search and replace.

    `size` defaults to a fraction of the frame rather than a constant. A 48 px subtitle is right on
    720p and unreadable on a phone-shaped 1080x1920, and the node has the frame size anyway.
    """
    lines = list(track.get("lines") or [])
    size = int(size or max(20, round(min(int(width), int(height)) / 22.0)))
    colours = colours or {}

    order, seen = [], set()
    for ln in lines:
        k = _key(ln.get("voice"))
        if k and k not in seen:
            seen.add(k)
            order.append(ln.get("voice"))
    palette = assign_colours(order, colours)

    styles = [_style_row("Default", colours.get("default") or PALETTE[0], idle, outline, font,
                         size, height, alignment)]
    for name in order:
        styles.append(_style_row(_safe_style(name), palette[_key(name)], idle, outline, font,
                                 size, height, alignment))

    events = []
    for ln in lines:
        text, start, end = _karaoke(ln, lead_in, sweep)
        if not text or end <= start:
            continue
        style = _safe_style(ln.get("voice")) if _key(ln.get("voice")) else "Default"
        name = ass_text(ln.get("label") or "")
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},{name},0,0,0,,{text}")

    return "\n".join([
        "[Script Info]",
        "; Written by Echo (kinburg-nodes)",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        f"PlayResX: {int(width)}",
        f"PlayResY: {int(height)}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        *styles,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events,
        "",
    ])


_UNSAFE_STYLE = re.compile(r"[,\[\]{}\r\n]")


def _safe_style(name):
    """A style name may not contain a comma — the `Style:` row is comma-separated and one name with
    a comma in it shifts every field after it, which a player reports as a font it cannot find."""
    return _UNSAFE_STYLE.sub(" ", str(name or "")).strip() or "Default"


def srt_time(seconds):
    """`00:00:12,400`. Whole milliseconds, so a value that rounds up carries instead of writing
    `,1000`, which no player accepts. Same rule as `save_video/timeline.py`."""
    ms = int(round(max(0.0, float(seconds)) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def srt(track, per_line=True):
    """A SubRip track off the same timings — the plain file for anything that cannot read ASS.

    `per_line=False` writes one cue per WORD. Not karaoke, and not a substitute for it: it is the
    debugging view, where a word that landed in the wrong place is visible in a text editor without
    opening a player at all.
    """
    out, n = [], 0
    for ln in track.get("lines") or []:
        items = ([ln] if per_line
                 else [{"start": w["start"], "end": w["end"], "text": w["text"]}
                       for w in ln.get("words") or []])
        for it in items:
            text = str(it.get("text") or "").strip()
            if not text or float(it.get("end", 0)) <= float(it.get("start", 0)):
                continue
            n += 1
            out.append(f"{n}\n{srt_time(it['start'])} --> {srt_time(it['end'])}\n{text}\n")
    return "\n".join(out)
