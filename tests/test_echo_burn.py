"""Karaoke burnt onto a frame — the geometry, and the claim that makes it affordable.

The claim: **a line is laid out and drawn once for the whole render, however many frames it is on
screen for.** That is what lets this run inside `Save Clip`'s encode loop instead of beside it, and
it is asserted by counting layouts after a few hundred frames rather than by trusting the comment.

Everything else here is the wipe: which word is lit at a given moment, that the highlight only ever
grows, and that a line lands inside the frame in every position. PIL only — no torch, no PyAV.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "echo")
load_module("kn.echo.subs", "echo/subs.py")
B = load_module("kn.echo.burn", "echo/burn.py")

from PIL import Image  # noqa: E402

check = Checker()

SIZE = (1280, 720)


def line(words, voice="Nina", label="Verse 1"):
    ws = [{"text": t, "start": a, "end": b} for t, a, b in words]
    return {"start": ws[0]["start"], "end": ws[-1]["end"], "voice": voice, "label": label,
            "section": 0, "text": "".join(w["text"] for w in ws), "words": ws}


TIMING = {"total": 60.0, "lines": [
    line([("Живий! ", 2.0, 2.6), ("Я ", 2.6, 2.9), ("живий", 2.9, 3.8)], "Nina"),
    line([("Похилилася", 8.0, 9.4)], "Gru", "Chorus"),
]}


# ------------------------------------------------------------------------------ the font resolves
font = B.load_font("Arial", 48)
check("a font resolves by bare name", font is not None)
check("...and can measure Cyrillic", font.getbbox("Живий")[2] > 0, font.getbbox("Живий"))
check("a font that does not exist falls back rather than raising",
      B.load_font("NoSuchFontHere", 40) is not None)


# --------------------------------------------------------------------------- what is on screen
p = B.Painter(TIMING, SIZE)
check("nothing is on before the first line", p.active(0.5) == [], p.active(0.5))
check("a line appears lead_in early", p.active(2.0 - 0.2) == [0], p.active(1.8))
check("it is on while sung", p.active(3.0) == [0])
check("it holds briefly after the last word", p.active(3.8 + 0.2) == [0], p.active(4.0))
check("and then it is gone", p.active(5.0) == [], p.active(5.0))
check("the second line comes on by itself", p.active(8.5) == [1], p.active(8.5))

# Two lines at once is legal — a backing echo under its lead — and is stacked, not overdrawn.
duet = B.Painter({"lines": [line([("a", 1.0, 3.0)], "Nina"), line([("b", 1.5, 3.5)], "Gru")]}, SIZE)
check("two overlapping lines are both on", duet.active(2.0) == [0, 1], duet.active(2.0))


# ------------------------------------------------------------------------------------ the wipe
check("before the first word nothing is lit", p.progress(0, 1.9) == (0, 0.0), p.progress(0, 1.9))
w, f = p.progress(0, 2.3)
check("mid-word: the first word, half way", w == 0 and 0.4 < f < 0.6, (w, f))
w, f = p.progress(0, 2.75)
check("the wipe moves to the second word", w == 1 and 0.4 < f < 0.6, (w, f))
w, f = p.progress(0, 9.0)
check("after the line every word is lit", w == 3, (w, f))

# The wipe must never go backwards — that reads as a stutter and is the thing a viewer notices.
seen, backwards = (-1, -1.0), []
for i in range(0, 400):
    t = 1.5 + i * 0.01
    w, f = p.progress(0, t)
    if (w, f) < seen and w != seen[0]:
        backwards.append((t, (w, f), seen))
    seen = (w, f) if w != seen[0] else seen
check("the wipe only ever advances", backwards == [], backwards[:3])


# ---------------------------------------------------- laid out ONCE, however many frames it is on
fresh = B.Painter(TIMING, SIZE)
frame = Image.new("RGB", SIZE, (20, 20, 20))
for i in range(240):                                   # ten seconds at 24 fps: both lines, twice
    fresh.draw(frame.copy(), i / 24.0)
check("two lines cost two layouts, not two hundred and forty",
      len(fresh._layouts) == 2, len(fresh._layouts))
check("the wipe cache is bounded", len(fresh._frames) <= 64 + 4, len(fresh._frames))


# ------------------------------------------------------------------------ it lands in the frame
def painted_box(pil):
    """The bounding box of everything that is not the background."""
    bg = Image.new("RGB", pil.size, (20, 20, 20))
    from PIL import ImageChops
    return ImageChops.difference(pil.convert("RGB"), bg).getbbox()


for position in B.POSITIONS:
    painter = B.Painter(TIMING, SIZE, {"position": position})
    shot = painter.draw(frame.copy(), 3.0)
    box = painted_box(shot)
    check(f"{position}: something was actually drawn", box is not None, box)
    check(f"{position}: entirely inside the frame",
          box and box[0] >= 0 and box[1] >= 0 and box[2] <= SIZE[0] and box[3] <= SIZE[1], box)

bottom = painted_box(B.Painter(TIMING, SIZE).draw(frame.copy(), 3.0))
top = painted_box(B.Painter(TIMING, SIZE, {"position": B.TOP}).draw(frame.copy(), 3.0))
check("bottom really is below top", bottom[1] > top[1], (bottom[1], top[1]))
check("a frame with nothing on it is untouched",
      painted_box(B.Painter(TIMING, SIZE).draw(frame.copy(), 6.0)) is None)


# --------------------------------------------------------------------- the highlight really moves
early = B.Painter(TIMING, SIZE).draw(frame.copy(), 2.05)
late = B.Painter(TIMING, SIZE).draw(frame.copy(), 3.75)
check("the picture at the start of a line differs from the one at its end",
      list(early.getdata()) != list(late.getdata()))

# One voice's colour must actually reach the pixels, or per-singer colour is a lie.
pink = B.Painter(TIMING, SIZE, {"colors": {"nina": "#ff00ff"}}).draw(frame.copy(), 3.79)
check("the singer's colour is on screen once the words are sung",
      any(px[0] > 200 and px[1] < 80 and px[2] > 200 for px in pink.convert("RGB").getdata()))
check("...and is NOT there before they are",
      not any(px[0] > 200 and px[1] < 80 and px[2] > 200 for px in
              B.Painter(TIMING, SIZE, {"colors": {"nina": "#ff00ff"}})
              .draw(frame.copy(), 1.85).convert("RGB").getdata()))


# ------------------------------------------------------------------------------- long lines wrap
longer = {"lines": [line([(w + " ", 1.0 + i * 0.2, 1.2 + i * 0.2)
                          for i, w in enumerate(["Воно"] * 40)], "Nina")]}
tall = B.Painter(longer, SIZE)
lay = tall._layout(0)
check("a line too wide for the frame wraps onto several rows",
      lay["size"][1] > lay["size"][0] * 0.05 and lay["size"][0] <= int(SIZE[0] * 0.86) + 40,
      lay["size"])
check("every word got a box", all(b is not None for b in lay["boxes"]), None)
check("a wrapped line still fits the frame",
      painted_box(tall.draw(frame.copy(), 4.0))[2] <= SIZE[0])


# ---------------------------------------------------------------------------------- degenerate
check("an empty timing paints nothing",
      B.Painter({"lines": []}, SIZE).draw(frame.copy(), 1.0) is not None)
check("a phone-shaped frame gets a bigger relative font",
      B.Painter(TIMING, (1080, 1920)).font.size >= B.Painter(TIMING, (1280, 720)).font.size,
      (B.Painter(TIMING, (1080, 1920)).font.size, B.Painter(TIMING, (1280, 720)).font.size))
check("an explicit size is used verbatim", B.Painter(TIMING, SIZE, {"size": 96}).font.size == 96)

check.done()
