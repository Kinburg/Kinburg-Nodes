"""The one thing `Satyr Music` exists for: getting `cfg_scale` as far as the tokenizer.

YuE2's reference implementation takes a guidance scale (`protocol.py` validates 0–20) and ComfyUI's
node never passes one, so guidance is off — the positive and negative branches are never even built.
The scale is reachable anyway, because `YuE2Tokenizer.tokenize_with_weights` reads it out of its
kwargs; this node's whole job is to put it there. Verified against the real tokenizer loaded from the
checkpoint: `cfg_scale=2.5` comes back as 2.5, the default is 1.0 for full/melody and 1.01 for off.

What the scale amplifies is worth stating, because it is not the obvious thing. The negative branch
is the instruction alone — `[Tags]` and `[Lyrics]` are removed — and the ABC is appended to BOTH
branches afterwards. So the plan cancels out of the difference entirely and the scale pushes the
style and the words against a plan whose authority is unchanged.

Everything here runs against a recording stub rather than the model: what needs pinning is the shape
of the call, and a stub is the only way to assert on an argument that the real path would swallow
silently.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

# `_frames_per_second` reaches into core at call time; stand it up before the node is loaded.
core = types.ModuleType("comfy.text_encoders.yue2")
core.FRAMES_PER_SECOND = 25
sys.modules.setdefault("comfy", types.ModuleType("comfy"))
sys.modules.setdefault("comfy.text_encoders", types.ModuleType("comfy.text_encoders"))
sys.modules["comfy.text_encoders.yue2"] = core

fake_package("kn", "satyr", "siren", "context", "timer", "util")
load_module("kn.util.anytype", "util/anytype.py")
load_module("kn.context.character_card", "context/character_card.py")
load_module("kn.timer.timer_nodes", "timer/timer_nodes.py")
load_module("kn.siren.cast", "siren/cast.py")
load_module("kn.siren.score", "siren/score.py")
load_module("kn.satyr.notation", "satyr/notation.py")
load_module("kn.satyr.bands", "satyr/bands.py")
load_module("kn.satyr.rewrite", "satyr/rewrite.py")
load_module("kn.satyr.layout", "satyr/layout.py")
load_module("kn.satyr.trim", "satyr/trim.py")
Nodes = load_module("kn.satyr.nodes", "satyr/nodes.py")

check = Checker()


class Clip:
    """Records the tokenize call and hands back a conditioning shaped like YuE2's."""

    def __init__(self, frames=500, keep_cfg=True):
        self.seen = None
        self.frames = frames
        self.keep_cfg = keep_cfg

    def tokenize(self, style, **kwargs):
        self.seen = {"style": style, **kwargs}
        out = dict(kwargs)
        if not self.keep_cfg:                       # an older core that ignores the argument
            out["cfg_scale"] = 1.0
        return out

    def encode_from_tokens_scheduled(self, tokens):
        return [[None, {"yue2_frames": self.frames}]]


node = Nodes.KinburgSatyrMusic()
ABC = "X:1\nT:\nM:4/4\n"


# ------------------------------------------------------------------------------------ the point
clip = Clip()
cond, seconds = node.run(clip, "english, female vocal", "[Verse]\nla la\n", ABC, 7, "full",
                         max_duration=360.0, cfg_scale=2.5, temperature=1.0, top_p=0.95,
                         top_k=100, repetition_penalty=1.2)
check("cfg_scale reaches the tokenizer", clip.seen["cfg_scale"] == 2.5, clip.seen["cfg_scale"])
check("and 1.0 passes through unchanged",
      Clip().tokenize("x", cfg_scale=1.0)["cfg_scale"] == 1.0)

# A core that swallowed it would leave guidance off while the node claimed otherwise — the one
# failure this node could have that nothing else would catch.
deaf = Clip(keep_cfg=False)
try:
    node.run(deaf, "s", "l", ABC, 1, "full", 360.0, 2.0, 1.0, 0.95, 100, 1.2)
    caught = False
except RuntimeError as e:
    caught = "ignored cfg_scale" in str(e)
check("a tokenizer that ignores it is caught, not trusted", caught)
check("and 1.0 is never rejected, since nothing has to change",
      node.run(Clip(keep_cfg=False), "s", "l", ABC, 1, "full", 360.0, 1.0, 1.0, 0.95, 100, 1.2)[1])


# ------------------------------------------------------------------------------------ the rest
check("the plan is passed on", clip.seen["abc"] == ABC)
check("the mode is passed as cot", clip.seen["cot"] == "full", clip.seen["cot"])
check("seconds come from the frame count", seconds == 500 / 25, seconds)
check("max_tokens is the duration in frames", clip.seen["max_tokens"] == 9000,
      clip.seen["max_tokens"])
check("a short duration still asks for one token",
      Clip().tokenize("x", **{"max_tokens": max(1, round(0.01 * 25))})["max_tokens"] == 1)
check("sampling settings are passed straight through",
      (clip.seen["temperature"], clip.seen["top_p"], clip.seen["top_k"],
       clip.seen["repetition_penalty"]) == (1.0, 0.95, 100, 1.2))
check("the seed goes with them", clip.seen["seed"] == 7)
check("conditioning comes back as given", cond[0][1]["yue2_frames"] == 500)

# An empty plan has to fall back to `off`, exactly as the core node does — anything else would send
# the model a mode it has no score for.
blank = Clip()
node.run(blank, "s", "l", "   \n ", 1, "full", 360.0, 1.0, 1.0, 0.95, 100, 1.2)
check("an empty plan falls back to off", blank.seen["cot"] == "off", blank.seen["cot"])
melody = Clip()
node.run(melody, "s", "l", ABC, 1, "melody", 360.0, 1.0, 1.0, 0.95, 100, 1.2)
check("melody mode survives a real plan", melody.seen["cot"] == "melody")

check.done()
