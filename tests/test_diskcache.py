"""The shared disk cache, and the Morpheus adapter that used to be all of it.

Both halves matter. The keys decide whether editing one step of a chain re-does the rest of it, and
the store's fp16 round-trip decides whether a cached frame is the same tensor a fresh one was — a
frame that comes back even slightly different would miss every cache downstream of it, forever.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, comfy_on_path, fake_package, load_module  # noqa: E402

comfy_on_path()
import torch  # noqa: E402
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kbcache_")
folder_paths.get_user_directory = lambda: TMP      # redirect the whole cache tree

fake_package("kn", "util", "morpheus")
dc = load_module("kn.util.diskcache", "util/diskcache.py")
mc = load_module("kn.morpheus.cache", "morpheus/cache.py")

check = Checker()

# ------------------------------------------------------------------------------------ the keys
check("a key is stable", dc.key("a", 1, None) == dc.key("a", 1, None))
check("parts cannot run together", dc.key("a", "b") != dc.key("ab", ""))
check("None and empty are the same part", dc.key(None) == dc.key(""))
check("a causal key changes when its ancestor does",
      dc.key(dc.key("s1"), 2) != dc.key(dc.key("s1x"), 2))

t = torch.zeros((1, 4, 4, 3))
check("tensor_key(None)", dc.tensor_key(None) == "none")
check("tensor_key is content-addressed", dc.tensor_key(t) == dc.tensor_key(t.clone()))
check("…and sees the content", dc.tensor_key(t) != dc.tensor_key(t + 1))
check("…and the shape", dc.tensor_key(t) != dc.tensor_key(torch.zeros((1, 4, 5, 3))))
check("tensor_key never raises on junk", dc.tensor_key(object()) == "unhashable")
check("fingerprint(None) is a key, not a crash", dc.fingerprint(None) == dc.key("none"))


class Patched:
    def __init__(self, n):
        self.patches = {"lora_a": 1} if n else {}

    def parameters(self):
        return [torch.zeros(4)]


check("fingerprint sees LoRA patches", dc.fingerprint(Patched(1)) != dc.fingerprint(Patched(0)))

# ---------------------------------------------------------------------------------- the store
st = dc.Store("unit_store", "Unit", limit_bytes=10 ** 9)
check("the folder lands under ComfyUI's user directory", st.dir().startswith(TMP))
check("a miss is None, not an error", st.load(dc.key("nope")) is None)

k = dc.key("entry", 1)
val = torch.rand((1, 8, 8, 3))
check("save reports success as None", st.save(k, {"image": val, "absent": None}) is None)
got = st.load(k)
check("what went in comes back", got is not None and "image" in got)
check("a None tensor is skipped, not stored", "absent" not in got)
check("stored as fp16", got["image"].dtype == torch.float16)
check("the values survive", torch.allclose(got["image"].float(), val, atol=1e-2))
check("require= accepts a matching entry", st.load(k, require="image") is not None)
check("require= rejects one written by an older version", st.load(k, require="video") is None)
check("saving nothing says so", st.save(dc.key("empty"), {"a": None}) == "nothing to write")
check("a non-contiguous view can still be written",
      st.save(dc.key("view"), {"image": torch.rand((1, 8, 8, 3)).movedim(-1, 1)}) is None)

jk = dc.key("text", 1)
check("a json miss is None", st.load_json(jk) is None)
check("json saves", st.save_json(jk, {"text": "привет", "n": 2}) is None)
check("json round-trips, Cyrillic and all", st.load_json(jk) == {"text": "привет", "n": 2})
check("tensors and text never collide", st.path(jk) != st.path(jk, ".json"))

# ---------------------------------------------------------------------------------- pruning
small = dc.Store("prune_store", "Unit", limit_bytes=10 ** 9)
paths = []
for i in range(3):
    small.save(dc.key("p", i), {"image": torch.rand((1, 16, 16, 3))})
    paths.append(small.path(dc.key("p", i)))
    time.sleep(0.01)                       # distinct mtimes, so the LRU order is real
one = os.path.getsize(paths[0])
check("a generous limit deletes nothing", small.prune() == 0 and len(os.listdir(small.dir())) == 3)
freed = small.prune(limit_bytes=int(one * 1.5))
left = [n for n in os.listdir(small.dir()) if n.endswith(".safetensors")]
check("pruning frees bytes", freed >= one)
check("…down to what fits", len(left) == 1)
check("…keeping the most recent", left[0] == os.path.basename(paths[2]))
tiny = dc.Store("tiny_store", "Unit")
tiny.save(dc.key("t"), {"image": torch.rand((1, 16, 16, 3))})
tiny.prune(limit_bytes=1)
check("a limit below one entry empties the folder", not os.listdir(tiny.dir()))

# ------------------------------------------------------------------------- the Morpheus adapter
check("the folder is where it always was",
      mc.cache_dir() == os.path.join(TMP, "kinburg-nodes", "morpheus_dreams"))
mk = mc.key("shot", 1)
video, audio, tail = torch.rand((1, 4, 8, 8)), torch.rand((1, 2, 16)), torch.rand((1, 8, 8, 3))
check("a shot saves", mc.save(mk, video, audio, tail, meta={"shot": 1}) is None)
hit = mc.load(mk)
check("…with all three tensors", hit is not None and set(hit) == {"video", "audio", "handoff"})
check("audio and handoff stay optional",
      mc.save(mc.key("s2"), video) is None and set(mc.load(mc.key("s2"))) == {"video"})
check("an entry with no video is not a hit — the rule that stayed local",
      (dc.Store("morpheus_dreams", "Unit").save(mc.key("videoless"), {"handoff": tail}) is None
       and mc.load(mc.key("videoless")) is None))
check("the helpers are the same ones", mc.tensor_key(tail) == dc.tensor_key(tail))
check("the limit is the shared default", mc.LIMIT_BYTES == dc.DEFAULT_LIMIT)
check("json still works through the adapter",
      mc.save_json(mc.key("j"), {"prompt": "x"}) is None
      and mc.load_json(mc.key("j")) == {"prompt": "x"})
check("prune is still callable with a limit", isinstance(mc.prune(10 ** 9), int))

check.done()
