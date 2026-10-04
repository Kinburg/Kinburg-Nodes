"""Load VAE (Precision): the weights come from the file, at the precision asked for, into a VAE of
their own — from a VAE file or from inside a checkpoint, whatever prefix that checkpoint keeps it under.

ComfyUI is stubbed: what is under test is which weights reach `comfy.sd.VAE` and with what dtype, not
ComfyUI's VAE itself. Every file here is a small safetensors written for the test.
"""
import shutil
import sys
import tempfile
import types
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="kn_vae_"))
FILES = {
    ("vae", "plain.safetensors"): {"encoder.w": torch.ones(2), "decoder.w": torch.full((3,), 2.0)},
    ("checkpoints", "song.safetensors"): {"model.x": torch.zeros(4), "vae.decoder.w": torch.full((3,), 5.0),
                                          "text_encoders.y": torch.zeros(1)},
    ("checkpoints", "sd.safetensors"): {"model.diffusion_model.x": torch.zeros(1),
                                        "first_stage_model.decoder.w": torch.full((3,), 7.0)},
    ("checkpoints", "bare.safetensors"): {"model.x": torch.zeros(1)},
}
for (folder, name), tensors in FILES.items():
    (TMP / folder).mkdir(exist_ok=True)
    save_file(tensors, str(TMP / folder / name), metadata={"made": "for the test"})

paths = types.ModuleType("folder_paths")
paths.get_filename_list = lambda folder: sorted(n for f, n in FILES if f == folder)
paths.get_full_path_or_raise = lambda folder, name: str(TMP / folder / name)
built = []


class FakeVAE:
    def __init__(self, sd=None, metadata=None, dtype=None):
        built.append(self)
        self.sd, self.metadata, self.dtype = sd, metadata, dtype

    def throw_exception_if_invalid(self):
        pass


comfy = types.ModuleType("comfy")
comfy.sd = types.ModuleType("comfy.sd")
comfy.sd.VAE = FakeVAE
comfy.utils = types.ModuleType("comfy.utils")
comfy.utils.load_torch_file = lambda path, return_metadata=False: (
    (load_file(path), {"made": "for the test"}) if return_metadata else load_file(path))
sys.modules.update({"folder_paths": paths, "comfy": comfy, "comfy.sd": comfy.sd, "comfy.utils": comfy.utils})

fake_package("kn", "util")
V = load_module("kn.util.vae_loader", "util/vae_loader.py")
check = Checker()
node = V.KinburgVAELoaderPrecision()

listed = V.sources()
check("VAE files and checkpoints are both offered, each marked with where it lives",
      "vae/plain.safetensors" in listed and "checkpoint/song.safetensors" in listed, listed)
check("fp32 is the default", V.KinburgVAELoaderPrecision.INPUT_TYPES()["required"]["dtype"][1]["default"] == "fp32")

vae = node.run("vae/plain.safetensors", "fp32")[0]
check("a VAE file is loaded whole, at the precision asked for",
      set(vae.sd) == {"encoder.w", "decoder.w"} and vae.dtype is torch.float32, (sorted(vae.sd), vae.dtype))
check("with its metadata", vae.metadata == {"made": "for the test"})
check("from the file's own weights, not a copy rounded on the way in",
      vae.sd["decoder.w"].dtype is torch.float32 and float(vae.sd["decoder.w"][0]) == 2.0)

vae = node.run("checkpoint/song.safetensors", "fp32")[0]
check("a checkpoint gives up its VAE alone, prefix stripped", set(vae.sd) == {"decoder.w"}
      and float(vae.sd["decoder.w"][0]) == 5.0, sorted(vae.sd))
vae = node.run("checkpoint/sd.safetensors", "bf16")[0]
check("Stable Diffusion's layout is found too", set(vae.sd) == {"decoder.w"} and float(vae.sd["decoder.w"][0]) == 7.0)
check("and any precision on the list reaches the VAE", vae.dtype is torch.bfloat16)
check("fp16 as well", node.run("vae/plain.safetensors", "fp16")[0].dtype is torch.float16)

try:
    node.run("checkpoint/bare.safetensors", "fp32")
    check("a checkpoint with no VAE says so", False)
except ValueError as e:
    check("a checkpoint with no VAE says so", "no VAE inside" in str(e), str(e))

check("every load builds a VAE of its own, so nothing else in the graph is touched",
      len({id(v) for v in built}) == len(built) == 4, len(built))

shutil.rmtree(TMP, ignore_errors=True)
check.done()
