"""Send Image to Chat — backend."""
import importlib.util
import json
import tempfile
import types
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import COMFY, PACK, comfy_on_path, fake_package, load_module, load_pack  # noqa: E402

comfy_on_path()

IN = Path(tempfile.mkdtemp()) / "input"
IN.mkdir(parents=True)
fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: str(IN)
fp.get_temp_directory = lambda: str(IN / "temp")
fp.get_output_directory = lambda: str(IN / "out")
sys.modules["folder_paths"] = fp

pkg = types.ModuleType("kn")
pkg.__path__ = [str(PACK / "local_llm")]
sys.modules["kn"] = pkg
spec = importlib.util.spec_from_file_location("kn.send_image_node",
                                              str(PACK / "local_llm" / "send_image_node.py"))
sn = importlib.util.module_from_spec(spec)
sys.modules["kn.send_image_node"] = sn
spec.loader.exec_module(sn)

import torch
from PIL import Image

fails = []


def check(label, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + label + (("  " + str(extra)) if extra else ""))
    if not cond:
        fails.append(label)


node = sn.LocalLLMChatSendImage()
DIR = IN / sn.ATT_DIR

# ── _resize_to_mp ───────────────────────────────────────────────────────────────────────────
big = Image.new("RGB", (2000, 2000))          # 4 MP
small = Image.new("RGB", (400, 300))          # 0.12 MP
r = sn._resize_to_mp(big, 1.0)
check("downscales to ~1 MP", 0.9e6 < r.width * r.height <= 1.05e6, (r.width, r.height))
check("never upscales", sn._resize_to_mp(small, 4.0).size == (400, 300))
check("0 means keep it", sn._resize_to_mp(big, 0).size == (2000, 2000))
check("aspect is kept", abs((r.width / r.height) - 1.0) < 0.01, (r.width, r.height))

# ── _save_frames ────────────────────────────────────────────────────────────────────────────
torch.manual_seed(7)
img = torch.rand(1, 64, 48, 3)
refs = sn._save_frames(img, 0)
check("one frame -> one ref", len(refs) == 1, len(refs))
check("the file is really there", (DIR / refs[0]["name"]).is_file(), refs[0])
check("the ref is a ComfyUI file triple",
      refs[0]["subfolder"] == sn.ATT_DIR and refs[0]["type"] == "input", refs[0])

again = sn._save_frames(img, 0)
check("the same picture hashes to the same name", again[0]["name"] == refs[0]["name"],
      (refs[0]["name"], again[0]["name"]))
check("only one file was written", len(list(DIR.glob("*.png"))) == 1,
      [p.name for p in DIR.glob("*.png")])

other = sn._save_frames(torch.rand(1, 64, 48, 3), 0)
check("a different picture gets a different name", other[0]["name"] != refs[0]["name"])

batch = sn._save_frames(torch.rand(3, 32, 32, 3), 0)
check("a batch of 3 -> three refs", len(batch) == 3, len(batch))
check("...all distinct", len({b["name"] for b in batch}) == 3)

shrunk = sn._save_frames(torch.rand(1, 2000, 2000, 3), 0.25)
with Image.open(DIR / shrunk[0]["name"]) as im:
    check("megapixels are applied to the saved copy", im.width * im.height <= 0.26e6, im.size)


# ── run() ───────────────────────────────────────────────────────────────────────────────────
def payload_of(out):
    return json.loads(out["ui"]["kinburg_chatsend"][0])


out = node.run(img, "the active persona", "on button press", 1.0, caption="a red dress",
               shot="shot 3 / end")
p = payload_of(out)
check("run passes the image straight through", out["result"][0] is img)
check("payload carries who and when",
      p["as"] == "the active persona" and p["when"] == "on button press", (p["as"], p["when"]))
a = p["refs"][0]
check("caption rides along", a["caption"] == "a red dress", a)
check("shot rides along", a["shot"] == "shot 3 / end", a)
check("a persona picture is invisible to the model by default", a.get("ctx") is False, a)

p2 = payload_of(node.run(img, "persona 2", "every run", 1.0, caption="c", note_in_context=True))
check("note_in_context lets it into the context", "ctx" not in p2["refs"][0], p2["refs"][0])

p3 = payload_of(node.run(img, "me (user)", "every run", 1.0))
check("what YOU send is always remembered", "ctx" not in p3["refs"][0], p3["refs"][0])
check("no caption -> no caption key", "caption" not in p3["refs"][0], p3["refs"][0])
check("blank caption is treated as none",
      "caption" not in payload_of(node.run(img, "me (user)", "every run", 1.0,
                                           caption="   "))["refs"][0])

# a persona picture with note_in_context but no caption still says *something* happened
p4 = payload_of(node.run(img, "persona 1", "every run", 1.0, note_in_context=True))
check("marker without a caption is still allowed", "ctx" not in p4["refs"][0], p4["refs"][0])


# a failure is reported, not raised
class Boom:
    def detach(self):
        raise RuntimeError("no pixels")


bad = payload_of(node.run(Boom(), "me (user)", "every run", 1.0))
check("a save failure comes back as an error payload", "couldn't save" in bad.get("error", ""),
      bad)

# ── registration ────────────────────────────────────────────────────────────────────────────
check("node is mapped", "LocalLLMChatSendImage" in sn.NODE_CLASS_MAPPINGS)
check("display name is set",
      sn.NODE_DISPLAY_NAME_MAPPINGS["LocalLLMChatSendImage"] == "Send Image to Chat")
opt = sn.LocalLLMChatSendImage.INPUT_TYPES()
check("send_as offers user + 6 personas + active", len(opt["required"]["send_as"][0]) == 8,
      opt["required"]["send_as"][0])

# ── discard(): what 🗑 Clear calls ───────────────────────────────────────────────────────────
att = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("kn.attachments",
                                           str(PACK / "local_llm" / "attachments.py")))
sys.modules["kn.attachments"] = att
att.__loader__.exec_module(att)

mine = sn._save_frames(torch.rand(1, 16, 16, 3), 0)[0]
check("the file exists before", (DIR / mine["name"]).is_file())
d, r = att.discard([mine])
check("discard removes it", d == 1 and r == 0 and not (DIR / mine["name"]).is_file(), (d, r))
d, r = att.discard([mine])
check("discarding a gone file is not an error", (d, r) == (0, 0), (d, r))

# an outsider must survive: the input tree is full of the user's own LoadImage sources
outsider = IN / "someones_photo.png"
Image.new("RGB", (4, 4)).save(outsider)
for ref in ({"name": "../someones_photo.png"},
            {"name": "..\\someones_photo.png"},
            {"name": "sub/someones_photo.png"},
            {"name": ""},
            "not-a-dict"):
    att.discard([ref])
check("nothing outside the attachment folder can be deleted", outsider.is_file())
d, r = att.discard([{"name": "../someones_photo.png"}])
check("...and the attempt is counted as refused", (d, r) == (0, 1), (d, r))

keep = sn._save_frames(torch.rand(1, 16, 16, 3), 0)[0]
att.discard([{"name": "no_such_file.png"}, keep])
check("a mixed batch still deletes what it can", not (DIR / keep["name"]).is_file())

print("\n" + ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
