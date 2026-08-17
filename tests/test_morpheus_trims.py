"""Morpheus' `trims`: the tail-drop that lets Orpheus put a cut on the beat.

Two things are worth pinning and neither is the trimming arithmetic itself (that is one expression).

**Widget order.** ComfyUI maps a saved workflow's widget values by POSITION. A new widget inserted
anywhere but the end silently shifts every value after it, so a graph saved before `trims` existed
would come back with `audio` holding what `seam_trim` used to say. It has to be last, and this is
the only place that can notice if it stops being.

**Parsing.** The comma list is the surface a human types into, so it is where typos live, and a typo
in an advanced field must cost a trim rather than a chain that took twenty minutes to sample.

Loads the real pack, so it needs torch and comfy — Morpheus is a sampler and cannot be faked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, load_pack  # noqa: E402

pack = load_pack("kn_trims")
M = sys.modules["kn_trims.morpheus.nodes"]
check = Checker()

Node = pack.NODE_CLASS_MAPPINGS["KinburgMorpheus"]
req = list(Node.INPUT_TYPES()["required"])

# ------------------------------------------------------------------------------- the widget order
check("trims exists as an input", "trims" in req, req[-3:])
check("…and it is the LAST widget", req[-1] == "trims", req[-4:])
check("…the widgets before it are untouched",
      req[-4:-1] == ["live_preview", "llm_keep_loaded", "shots_range"], req[-4:])
check("…and it defaults to blank, so old graphs behave exactly as before",
      Node.INPUT_TYPES()["required"]["trims"][1]["default"] == "")
check("render() accepts it by keyword with a default",
      "trims" in Node.render.__code__.co_varnames and
      Node.render.__defaults__ is not None)

# ------------------------------------------------------------------------------------- parsing
p = M._per_shot_ints
check("blank means no trimming at all", p("", 4) == [0, 0, 0, 0])
check("whitespace is still blank", p("  \n ", 3) == [0, 0, 0])
check("one value applies to every shot", p("12", 3) == [12, 12, 12])
check("the last value repeats", p("12, 2", 4) == [12, 2, 2, 2])
check("extra values are ignored", p("1, 2, 3", 2) == [1, 2])
check("semicolons and spaces split too", p("1; 2 3", 3) == [1, 2, 3])
check("floats round to frames", p("11.6, 0.4", 2) == [12, 0])
check("negatives clamp to zero", p("-5, 3", 2) == [0, 3])
check("a typo costs a trim, not the run", p("12, oops, 4", 3) == [12, 0, 4])
check("zero shots gives an empty list", p("12", 0) == [])

# ----------------------------------------------------------------- the clamp, as the plan applies it
# A shot must keep 5 frames whatever the head and tail trims ask for between them. Mirrors the
# expression in render()'s plan loop; if that changes, this is the reminder to change it here too.
def out_frames(frames, drop, want_cut):
    cut = max(0, min(want_cut, frames - drop - 5))
    return frames - drop - cut, cut


check("an ordinary trim just comes off", out_frames(192, 1, 12) == (179, 12))
check("no trim leaves the shot alone", out_frames(192, 1, 0) == (191, 0))
check("a greedy trim is clamped to leave 5 frames", out_frames(124, 1, 999) == (5, 118))
check("…and never goes negative", out_frames(124, 1, -3)[1] == 0)
check("head and tail trims share the shot",
      out_frames(124, 20, 999) == (5, 99), out_frames(124, 20, 999))

check.done()
