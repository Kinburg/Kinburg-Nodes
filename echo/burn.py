"""Karaoke text painted onto a frame — the half of the job that touches pixels.

**Why this is cheap, which is the only reason it can live inside `Save Clip`'s encode loop.** A line
is laid out and drawn **once**, into two RGBA sprites of identical geometry: one in the colour a word
waits in, one in the colour it becomes. Per frame there is no text layout, no glyph rasterizing and
no outline — only a mask of rectangles saying how far the singing has got, and one composite. The
expensive part happens once per line of the song; the per-frame part is a handful of filled boxes.

That matters because of what `Save Clip` is: it holds a couple of prepared slides and feeds the
encoder frame by frame, precisely so that a three-minute song never becomes a batch of five thousand
frames. Subtitles that re-rendered text every frame would not blow the memory, but they would make
the node's cost scale with fps for no reason. This way a 24 fps render draws each line once.

**The wipe is a mask, not a second drawing.** Words already sung contribute their whole box, the
word being sung contributes a box cut at its own progress, and `Image.composite` picks the hot
sprite inside those boxes and the idle one outside. Both sprites share their alpha — same glyphs,
same outline — so the composite cannot produce a fringe.

**What is on screen is what the author typed.** The word chunks come from `track.display_chunks`,
which keeps every comma and dash, so a line reassembles exactly. Nothing here lowercases or strips
anything for the aligner's benefit — that separation was made back in `translit.py` and this is
where it pays off.

Pure PIL. No torch, no PyAV, no model: the geometry is testable on its own.
"""
import os

from . import subs as S

BOTTOM, TOP, MIDDLE = "bottom", "top", "middle"
POSITIONS = [BOTTOM, TOP, MIDDLE]

#: Everything a burnt subtitle needs to look like itself. `Echo` fills this from its own widgets and
#: puts it on the timing, so a line is styled once and comes out the same in the `.ass` and in the
#: video — two places to set a colour is two places for them to disagree.
DEFAULT_STYLE = {
    "font": "Arial",
    "size": 0,                    # 0 = scale to the frame's short side
    "idle": S.DEFAULT_IDLE,
    "outline": S.DEFAULT_OUTLINE,
    "colors": {},                 # voice (lowered) -> #rrggbb
    "sweep": True,
    "lead_in": 0.35,
    "hold": 0.4,                  # how long a line stays after its last word
    "position": BOTTOM,
    "margin": 0.06,               # fraction of the frame height
    "width": 0.86,                # fraction of the frame width a line may use
    "max_lines": 3,               # how many lines may share the screen
}

#: Where to look for a font named without a path. The Windows directory is last on purpose: a font
#: dropped into ComfyUI's own `models/fonts` should win over one of the same name in the system.
_FONT_DIRS = []


def _font_dirs():
    if _FONT_DIRS:
        return _FONT_DIRS
    dirs = []
    try:
        import folder_paths
        dirs.append(os.path.join(folder_paths.models_dir, "fonts"))
    except Exception:
        pass
    dirs.append(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"))
    dirs.append("/usr/share/fonts")
    _FONT_DIRS.extend(dirs)
    return _FONT_DIRS


def load_font(name, px):
    """A font by name, size in pixels — or the default, which is better than raising mid-render.

    A name is tried as a path, then as a file in each font directory, with and without the
    extensions a `.ttf` name is usually written without. The fallbacks are the two faces that carry
    Cyrillic on the machines this pack runs on.
    """
    from PIL import ImageFont

    px = max(8, int(px))
    tries = []
    for candidate in [name, "Arial", "Segoe UI", "DejaVu Sans"]:
        text = str(candidate or "").strip()
        if not text:
            continue
        tries.append(text)
        if not os.path.splitext(text)[1]:
            flat = text.replace(" ", "")
            tries += [text + ".ttf", flat + ".ttf", flat.lower() + ".ttf"]

    for candidate in tries:
        for base in [""] + _font_dirs():
            try:
                return ImageFont.truetype(os.path.join(base, candidate) if base else candidate, px)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=px)
    except Exception:                                  # Pillow < 10 takes no size
        return ImageFont.load_default()


def style_from(timing, overrides=None):
    """The style stored on a timing by `Echo`, with anything the caller wants to override."""
    out = dict(DEFAULT_STYLE)
    out.update({k: v for k, v in (timing or {}).get("style", {}).items() if v is not None})
    out.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return out


def _rgb(hex_colour, fallback=(255, 255, 255)):
    text = str(hex_colour or "").strip().lstrip("#")
    if len(text) != 6:
        return fallback
    try:
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback


class Painter:
    """Draws a timing's lines onto frames of one size.

    Built once per render. Everything it caches is keyed by line, so the cost of a song is its
    number of LINES and never its number of frames.
    """

    def __init__(self, timing, size, style=None):
        self.lines = list((timing or {}).get("lines") or [])
        self.width, self.height = int(size[0]), int(size[1])
        self.style = style_from(timing, style)

        px = int(self.style["size"]) or max(14, round(min(self.width, self.height) / 22.0))
        self.font = load_font(self.style["font"], px)
        self.stroke = max(1, round(px / 11.0))
        self.max_w = max(32, int(self.width * float(self.style["width"])))

        order = []
        for ln in self.lines:
            voice = ln.get("voice")
            if voice and voice not in order:
                order.append(voice)
        self.colours = S.assign_colours(order, self.style.get("colors"))

        self._layouts = {}
        self._frames = {}
        # Sorted once: `active()` is called for every frame and must not sort a hundred lines each
        # time. Index kept so a cache key can name a line without hashing its text.
        self._sorted = sorted(range(len(self.lines)), key=lambda i: self.lines[i]["start"])

    # ------------------------------------------------------------------------------- what is on
    def active(self, t):
        """The indices of the lines visible at `t`, in the order they should be stacked."""
        lead = float(self.style["lead_in"])
        hold = float(self.style["hold"])
        out = []
        for i in self._sorted:
            ln = self.lines[i]
            if ln["start"] - lead <= t <= ln["end"] + hold:
                out.append(i)
            elif ln["start"] - lead > t:
                break                                  # sorted by start: nothing later can be on
        return out[-int(self.style["max_lines"]):]

    def progress(self, index, t):
        """`(word, fraction)` — how far the wipe has got through line `index` at `t`.

        `word` is the index of the word being sung, `fraction` how far into it. Before the line
        starts that is `(0, 0.0)`; after it ends, every word is lit.
        """
        words = self.lines[index]["words"]
        if not words or t < words[0]["start"]:
            return 0, 0.0
        for i, w in enumerate(words):
            if t < w["end"]:
                span = max(1e-6, w["end"] - w["start"])
                return i, max(0.0, min(1.0, (t - w["start"]) / span))
        return len(words), 0.0

    # --------------------------------------------------------------------------------- drawing
    def _layout(self, index):
        """Lay a line out and draw its two sprites. Once per line, for the whole render."""
        hit = self._layouts.get(index)
        if hit is not None:
            return hit

        from PIL import Image, ImageDraw

        line = self.lines[index]
        words = line["words"]
        ruler = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        widths = [ruler.textlength(w["text"], font=self.font) for w in words]
        # A trailing space belongs to the word for the wipe (the highlight should cross it) but not
        # to the row's width, or a centred line sits visibly off to the left.
        bare = [ruler.textlength(w["text"].rstrip(), font=self.font) for w in words]

        rows, row, used = [], [], 0.0
        for i, w in enumerate(words):
            if row and used + bare[i] > self.max_w:
                rows.append(row)
                row, used = [], 0.0
            row.append(i)
            used += widths[i]
        if row:
            rows.append(row)

        ascent, descent = self.font.getmetrics()
        line_h = ascent + descent + self.stroke * 2
        pad = self.stroke * 2
        row_w = [sum(widths[i] for i in r[:-1]) + bare[r[-1]] for r in rows]
        sprite = (max(1, int(max(row_w)) + pad * 2), max(1, line_h * len(rows) + pad))

        boxes = [None] * len(words)
        base = Image.new("RGBA", sprite, (0, 0, 0, 0))
        hot = Image.new("RGBA", sprite, (0, 0, 0, 0))
        d_base, d_hot = ImageDraw.Draw(base), ImageDraw.Draw(hot)

        idle = _rgb(self.style["idle"]) + (255,)
        edge = _rgb(self.style["outline"]) + (255,)
        voice = self.colours.get(S._key(line.get("voice")), S.PALETTE[0])
        lit = _rgb(voice) + (255,)

        for r, indices in enumerate(rows):
            x = (sprite[0] - row_w[r]) / 2.0
            y = pad // 2 + r * line_h
            for i in indices:
                for draw, fill in ((d_base, idle), (d_hot, lit)):
                    draw.text((x, y), words[i]["text"], font=self.font, fill=fill,
                              stroke_width=self.stroke, stroke_fill=edge)
                boxes[i] = (x, y, x + widths[i], y + line_h)
                x += widths[i]

        hit = {"base": base, "hot": hot, "boxes": boxes, "size": sprite}
        self._layouts[index] = hit
        return hit

    def _painted(self, index, word, frac):
        """The composited sprite for one wipe position, cached at 1/16th of a word."""
        # No floor on this. `max(1, ...)` here lit a sixteenth of the first word during the lead-in,
        # before a note had been sung — small, and exactly the kind of wrongness a karaoke line is
        # watched closely enough to show.
        step = int(round(max(0.0, min(1.0, frac)) * 16))
        key = (index, word, 0 if word >= len(self.lines[index]["words"]) else step)
        hit = self._frames.get(key)
        if hit is not None:
            return hit

        from PIL import Image, ImageDraw

        lay = self._layout(index)
        mask = Image.new("L", lay["size"], 0)
        pen = ImageDraw.Draw(mask)
        for i, box in enumerate(lay["boxes"]):
            if box is None:
                continue
            if i < word:
                pen.rectangle(box, fill=255)
            elif i == word and step > 0:
                x0, y0, x1, y1 = box
                pen.rectangle((x0, y0, x0 + (x1 - x0) * (step / 16.0), y1), fill=255)
        out = Image.composite(lay["hot"], lay["base"], mask)

        if len(self._frames) > 64:                     # a couple of lines' worth of wipe positions
            self._frames.clear()
        self._frames[key] = out
        return out

    def draw(self, pil, t, indices=None):
        """Paint every line active at `t` onto `pil`, in place. Returns it for chaining."""
        indices = self.active(t) if indices is None else indices
        if not indices:
            return pil

        sprites = []
        for i in indices:
            word, frac = self.progress(i, t)
            if not self.style["sweep"] and frac > 0:
                frac = 1.0                             # \k: the word flips at its start
            sprites.append(self._painted(i, word, frac))

        margin = int(self.height * float(self.style["margin"]))
        total = sum(s.height for s in sprites)
        if self.style["position"] == TOP:
            y = margin
        elif self.style["position"] == MIDDLE:
            y = max(0, (self.height - total) // 2)
        else:
            y = max(0, self.height - margin - total)

        for sprite in sprites:
            pil.paste(sprite, ((self.width - sprite.width) // 2, y), sprite)
            y += sprite.height
        return pil
