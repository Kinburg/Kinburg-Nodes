"""Send Image to Live Log — the frames event it puts on the wire.

The node is one payload wide, so that is what is checked: which frames of a batch are picked (and
that 'all' really is capped), how the block's label is assembled, the note, the warning that stands
in for a picture the encoder refused, and the channel + node id the event is sent on.

`util.images.log_uris` is stubbed. The real one reaches into `local_llm/llm_node.py` for the
thumbnail encoder, which drags comfy and llama-cpp in; what matters here is that the frames handed
to it are the ones the mode asked for, which a recorder shows better than a JPEG would.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, PACK, fake_package, load_module  # noqa: E402

fake_package("kn", "local_llm", "util")

# The stub goes in before the node is loaded, so its `from ..util.images import log_uris` binds to
# this. `seen` keeps the frames of the last call, which is how the frame-picking is verified.
seen = {"frames": [], "max_side": None, "fail": False}
images = types.ModuleType("kn.util.images")


def _log_uris(frames, max_side=320):
    seen["frames"] = list(frames)
    seen["max_side"] = max_side
    return [] if seen["fail"] else [f"data:image/jpeg;base64,#{i}" for i in range(len(frames))]


images.log_uris = _log_uris
sys.modules["kn.util.images"] = images

S = load_module("kn.local_llm.send_image_log_node", "local_llm/send_image_log_node.py")

import torch  # noqa: E402

check = Checker()
node = S.KinburgSendImageToLog()

# ── pick_frames ─────────────────────────────────────────────────────────────────────────────
check("all of a small batch is all of it", S.pick_frames(5, "all") == [0, 1, 2, 3, 4])
check("first is the first", S.pick_frames(9, "first") == [0])
check("last is the last", S.pick_frames(9, "last") == [8])
check("first & last are the ends", S.pick_frames(9, "first & last") == [0, 8])
check("a single frame can't be both ends", S.pick_frames(1, "first & last") == [0])
check("evenly spaced includes both ends",
      S.pick_frames(100, "4 evenly spaced") == [0, 33, 66, 99], S.pick_frames(100, "4 evenly spaced"))
check("...and never more than there are",
      S.pick_frames(3, "16 evenly spaced") == [0, 1, 2], S.pick_frames(3, "16 evenly spaced"))
check("8 evenly spaced really is 8", len(S.pick_frames(400, "8 evenly spaced")) == 8)

wide = S.pick_frames(300, "all")
check(f"all is capped at {S.MAX_FRAMES}", len(wide) == S.MAX_FRAMES, len(wide))
check("...still spanning the whole batch", wide[0] == 0 and wide[-1] == 299, (wide[0], wide[-1]))
check("...and in order, no repeats", wide == sorted(set(wide)))
check("an empty batch picks nothing", S.pick_frames(0, "all") == [])
check("a mode this version doesn't know falls back to all",
      S.pick_frames(4, "every other Thursday") == [0, 1, 2, 3])

# ── build_payload ───────────────────────────────────────────────────────────────────────────
one = torch.rand(1, 32, 24, 3)
batch = torch.rand(120, 32, 24, 3)

p = S.build_payload(one, "before upscale", "all", note="seed 7", max_side=256)
check("the event is a frames event", p["event"] == "frames", p["event"])
check("one frame -> one thumbnail", len(p["images"]) == 1, p["images"])
check("the label is the label", p["label"] == "before upscale", p.get("label"))
check("a single frame says nothing about frames", "of" not in p["label"], p["label"])
check("the note is the block's text", p["text"] == "seed 7", p.get("text"))
check("max_side reaches the encoder", seen["max_side"] == 256, seen["max_side"])
check("the frame handed over is a one-frame batch", tuple(seen["frames"][0].shape) == (1, 32, 24, 3),
      tuple(seen["frames"][0].shape))

p = S.build_payload(batch, "", "8 evenly spaced")
check("a thinned batch says how thin it is", p["label"] == "8 of 120 frames", p.get("label"))
check("...and sends that many", len(p["images"]) == 8, len(p["images"]))
check("no label, no note -> no keys for them", "text" not in p)

p = S.build_payload(batch, "the tail", "last")
check("a single frame of a batch is named by index", p["label"] == "the tail · frame 120 of 120",
      p.get("label"))
check("...and it is the last frame",
      torch.equal(seen["frames"][0], batch[119:120]), tuple(seen["frames"][0].shape))

p = S.build_payload(batch, "  ", "first", note="   ")
check("blank label and note are treated as absent", "label" in p and "text" not in p, p.keys())
check("...the frame count still labels it", p["label"] == "frame 1 of 120", p.get("label"))

bare = S.build_payload(torch.rand(48, 48, 3), "unbatched", "all")
check("a bare [H,W,C] frame is one picture", len(bare["images"]) == 1, bare["images"])
check("...passed on with a batch axis", tuple(seen["frames"][0].shape) == (1, 48, 48, 3),
      tuple(seen["frames"][0].shape))

seen["fail"] = True
p = S.build_payload(one, "", "all", note="seed 7")
check("a picture the encoder refused is said out loud", p["text"].startswith("⚠"), p.get("text"))
check("...without losing the note", p["text"].endswith("seed 7"), p.get("text"))
seen["fail"] = False

# ── run(): the wire ─────────────────────────────────────────────────────────────────────────
sent = []
srv = types.ModuleType("server")
srv.PromptServer = types.SimpleNamespace(
    instance=types.SimpleNamespace(send_sync=lambda ch, d: sent.append((ch, d))))
sys.modules["server"] = srv

out = node.run(one, "tap", "all", True, note="hello", max_side=320, unique_id=42)
check("the image passes straight through", out[0] is one)
check("one event went out", len(sent) == 1, len(sent))
ch, d = sent[0]
check("on the log's channel", ch == "kinburg.llm", ch)
check("tagged with this node's id, so the log titles the block after it", d["id"] == "42", d["id"])
check("carrying the payload", d["event"] == "frames" and d["text"] == "hello", d)

del sys.modules["server"]
out = node.run(one, "tap", "all", True)          # headless: no server to send to
check("no server is not an error", out[0] is one)

# ── caching ─────────────────────────────────────────────────────────────────────────────────
a, b = S.KinburgSendImageToLog.IS_CHANGED(always_run=True), S.KinburgSendImageToLog.IS_CHANGED(always_run=True)
check("always_run: never equal to last time, so it re-runs", a != b, (a, b))
c, e = S.KinburgSendImageToLog.IS_CHANGED(always_run=False), S.KinburgSendImageToLog.IS_CHANGED(always_run=False)
check("off: caches like any other node", c == e, (c, e))

# ── registration ────────────────────────────────────────────────────────────────────────────
check("node is mapped", "KinburgSendImageToLog" in S.NODE_CLASS_MAPPINGS)
check("display name is set",
      S.NODE_DISPLAY_NAME_MAPPINGS["KinburgSendImageToLog"] == "Send Image to Live Log 📜")
spec = S.KinburgSendImageToLog.INPUT_TYPES()
check("it is an output node, so it runs as a leaf", S.KinburgSendImageToLog.OUTPUT_NODE is True)
check("image passes through", S.KinburgSendImageToLog.RETURN_NAMES == ("image",))
check("every frames mode is offered", spec["required"]["frames"][0] == S.FRAMES, spec["required"]["frames"][0])
check("the node knows its own id for the log", spec["hidden"] == {"unique_id": "UNIQUE_ID"})

# The whole point is that the log's renderer reads what this node writes; the two keys it must know
# about are `label` and `text` on a frames event.
js = (PACK / "web" / "llm_log.js").read_text(encoding="utf-8")
check("the renderer reads a note off a frames event",
      'if (ev === "frames")' in js and "block.imagesOnly = true" in js
      and 'typeof d.text === "string"' in js)

check.done()
