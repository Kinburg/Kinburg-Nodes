"""Load VAE (Precision) — a VAE loaded at the precision you name, not the one ComfyUI picks for it.

ComfyUI casts a VAE's weights when it loads it and decodes at that precision, and on any card with
fast fp16 it picks fp16. For an image that is invisible. For audio it is the last step before the
waveform: YuE2's reference pipeline decodes its VAE in fp32 ("BF16 AR/NAR, FP32 VAE") and ships the
weights in fp32, while on an RTX 4070 ComfyUI loads and decodes that same VAE in fp16.

Wrapping the VAE a loader already returned cannot undo that, because its weights were rounded on the
way in. So this loads its own copy from the file's original weights — a VAE file, or the VAE inside
a checkpoint — at the precision asked for. It is a separate object: the checkpoint's VAE and every
other decoder in the graph stay exactly as they were, and two decodes of one latent compare cleanly.

ComfyUI's own modules are imported where they are used, as everywhere in this pack, so the package
still imports without ComfyUI for the registry scan and the tests.
"""
import torch
from safetensors import safe_open

from ..categories import CAT_MODEL

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
#: Where checkpoints keep their VAE: most current models (YuE2 among them), Stable Diffusion's own
#: layout, and Stable Audio's.
PREFIXES = ("vae.", "first_stage_model.", "pretransform.model.")


def sources():
    import folder_paths
    return ([f"vae/{name}" for name in folder_paths.get_filename_list("vae")]
            + [f"checkpoint/{name}" for name in folder_paths.get_filename_list("checkpoints")])


def checkpoint_vae(path):
    """The VAE weights inside a checkpoint, prefix stripped. A safetensors file is read key by key,
    so a multi-gigabyte model does not have to load to get at its VAE."""
    if path.lower().endswith(".safetensors"):
        with safe_open(path, "pt") as f:
            names = list(f.keys())
            for prefix in PREFIXES:
                found = [n for n in names if n.startswith(prefix)]
                if found:
                    return {n[len(prefix):]: f.get_tensor(n) for n in found}
    else:
        import comfy.utils
        sd = comfy.utils.load_torch_file(path)
        for prefix in PREFIXES:
            found = {n[len(prefix):]: t for n, t in sd.items() if n.startswith(prefix)}
            if found:
                return found
    raise ValueError(f"no VAE inside {path} — none of its weights start with {', '.join(PREFIXES)}")


class KinburgVAELoaderPrecision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (sources(), {"tooltip": "A VAE file from models/vae, or a checkpoint whose own VAE you want — the one its loader's VAE output gives you, here at your precision."}),
                "dtype": (list(DTYPES), {"default": "fp32", "tooltip": "The precision the weights are loaded in and the decode runs at.\n\nfp32 is what YuE2's reference pipeline decodes its audio in; ComfyUI picks fp16 for it on any card with fast fp16. Twice the VAE's memory, and a slower decode — negligible next to generating the song."}),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "run"
    CATEGORY = CAT_MODEL
    DESCRIPTION = ("Loads a VAE — from models/vae, or the one inside a checkpoint — at the precision you "
                   "choose instead of the one ComfyUI picks (fp16 on most cards). Its own copy from the "
                   "file's original weights, so the checkpoint's VAE and every other decoder are left "
                   "alone. For decoding audio in fp32, as YuE2's reference does, and for A/B tests of one "
                   "latent decoded both ways. Wire it into the ordinary VAE Decode node.")

    def run(self, source, dtype):
        import comfy.sd
        import comfy.utils
        import folder_paths
        kind, name = source.split("/", 1)
        if kind == "vae":
            sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", name),
                                                       return_metadata=True)
        elif kind == "checkpoint":
            sd, metadata = checkpoint_vae(folder_paths.get_full_path_or_raise("checkpoints", name)), None
        else:
            raise ValueError(f"unknown source {source!r}")
        vae = comfy.sd.VAE(sd=sd, metadata=metadata, dtype=DTYPES[dtype])
        vae.throw_exception_if_invalid()
        return (vae,)


NODE_CLASS_MAPPINGS = {"KinburgVAELoaderPrecision": KinburgVAELoaderPrecision}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgVAELoaderPrecision": "Load VAE (Precision)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
