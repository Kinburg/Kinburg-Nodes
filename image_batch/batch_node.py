"""Unlim Image Batch — concatenate an unlimited number of IMAGE inputs into one batch.

Like Unlim Text Concat, the input list grows on demand (web/image_batch.js): it starts
with `image_1` (required) + `image_2`, and a new empty slot appears whenever you connect
the last one. The execution backend forwards every connected link input — even the ones
not declared in INPUT_TYPES — so the extra `image_N` slots arrive via **kwargs and are
stacked in slot order.

ComfyUI IMAGE tensors are [B, H, W, C] floats in 0..1. A single batch tensor needs every
frame at the same size, so when inputs differ the node reconciles them **without
resampling** (no quality loss) — see `mode`:
  * "as is"           — stack untouched; errors if the inputs aren't already the same size.
  * "crop to smallest"— center-crop every input down to the smallest width/height.
  * "pad to largest"  — center-pad every input up to the largest width/height (`pad_color`).
Inputs with fewer channels (RGB vs RGBA) are padded with opaque alpha so they can stack.
Each input may itself already be a batch — all of them are concatenated.
"""
import re
from ..categories import CAT_IMAGE

# Heavy / ComfyUI-only imports are guarded so the package still imports (and the Registry
# can enumerate nodes) without ComfyUI present. At runtime torch is always available.
try:
    import torch
except Exception:
    pass

_IDX_RE = re.compile(r"^image_(\d+)$")
_MODES = ["as is", "crop to smallest", "pad to largest"]


def _hex_rgb01(value, fallback=(0.0, 0.0, 0.0)):
    """'#RRGGBB' / '#RGB' -> (r, g, b) floats in 0..1; fallback if invalid."""
    s = str(value).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return fallback
    try:
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return fallback


class UnlimImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "mode": (_MODES, {"default": "as is", "tooltip": "How to reconcile inputs of different sizes (no resampling): 'as is' keeps pixels and errors on a size mismatch; 'crop to smallest' center-crops down; 'pad to largest' center-pads up with pad_color."}),
                "pad_color": ("STRING", {"default": "#000000", "tooltip": "Fill color (HEX) for the borders added in 'pad to largest' mode. Ignored otherwise."}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "Skip empty / unconnected inputs (e.g. a bypassed branch) instead of failing, so the batch stays aligned with the rest of the comparison."}),
            },
            "optional": {
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CAT_IMAGE

    @staticmethod
    def _index(key):
        m = _IDX_RE.match(key)
        return int(m.group(1)) if m else 1 << 30

    def run(self, mode="as is", pad_color="#000000", skip_empty=True, **kwargs):
        imgs = []
        for k in sorted((k for k in kwargs if _IDX_RE.match(k)), key=self._index):
            v = kwargs.get(k)
            if v is None:
                continue
            if skip_empty and int(v.shape[0]) == 0:
                continue
            imgs.append(v)
        if not imgs:
            return (None,)

        target_c = max(int(im.shape[-1]) for im in imgs)

        if mode == "crop to smallest":
            th = min(int(im.shape[1]) for im in imgs)
            tw = min(int(im.shape[2]) for im in imgs)
            out = [self._pad_channels(self._center_crop(im, th, tw), target_c) for im in imgs]
        elif mode == "pad to largest":
            th = max(int(im.shape[1]) for im in imgs)
            tw = max(int(im.shape[2]) for im in imgs)
            fill = _hex_rgb01(pad_color)
            out = [self._center_pad(self._pad_channels(im, target_c), th, tw, fill) for im in imgs]
        else:  # "as is" — stack untouched; sizes must already match
            sizes = {(int(im.shape[1]), int(im.shape[2])) for im in imgs}
            if len(sizes) > 1:
                pretty = ", ".join(f"{w}x{h}" for h, w in sorted(sizes))
                raise ValueError(
                    f"Unlim Image Batch: inputs have different sizes ({pretty}). "
                    f"Switch mode to 'crop to smallest' or 'pad to largest'.")
            out = [self._pad_channels(im, target_c) for im in imgs]

        return (out[0] if len(out) == 1 else torch.cat(out, dim=0),)

    @staticmethod
    def _pad_channels(img, c):
        """Pad missing channels (e.g. RGB -> RGBA) with opaque alpha. Lossless."""
        if int(img.shape[-1]) == c:
            return img
        return torch.nn.functional.pad(img, (0, c - int(img.shape[-1])), value=1.0)

    @staticmethod
    def _center_crop(img, th, tw):
        """Center-crop [B,H,W,C] to th x tw (no-op if already <= target). Lossless."""
        H, W = int(img.shape[1]), int(img.shape[2])
        top = max(0, (H - th) // 2)
        left = max(0, (W - tw) // 2)
        return img[:, top:top + th, left:left + tw, :]

    @staticmethod
    def _center_pad(img, th, tw, fill_rgb):
        """Center-pad [B,H,W,C] up to th x tw with a solid fill (no-op if already >=)."""
        B, H, W, C = (int(img.shape[0]), int(img.shape[1]), int(img.shape[2]), int(img.shape[-1]))
        if H >= th and W >= tw:
            return img
        th, tw = max(th, H), max(tw, W)
        canvas = torch.ones((B, th, tw, C), dtype=img.dtype, device=img.device)  # alpha (ch>=3) -> 1.0
        for ch in range(min(3, C)):
            canvas[..., ch] = fill_rgb[ch]
        top = (th - H) // 2
        left = (tw - W) // 2
        canvas[:, top:top + H, left:left + W, :] = img
        return canvas


NODE_CLASS_MAPPINGS = {"UnlimImageBatch": UnlimImageBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"UnlimImageBatch": "Unlim Image Batch"}
