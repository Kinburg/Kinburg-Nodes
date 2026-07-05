"""Image batch editing — insert / remove frames in an IMAGE batch by position.

A ComfyUI IMAGE batch is one tensor [B, H, W, C]. These nodes edit it frame-wise:

* **Image Batch Insert** puts an image (itself possibly a batch) into the batch at a chosen
  spot — start / end / at index / after index. Frames of a different size are reconciled the
  same lossless way as Unlim Image Batch (`mode` / `pad_color`), so the result still stacks.
* **Image Batch Remove** drops `count` frame(s) starting at `index` and also returns what it
  removed. `index` may be negative (-1 = last frame).

For non-image data (or images of differing sizes you'd rather keep separate) use the generic
List Insert / List Remove instead.
"""
try:
    import torch
except Exception:
    torch = None

from ..image_batch.batch_node import UnlimImageBatch, _MODES

_POS = ["at end", "at start", "at index", "after index"]


def _len(t):
    return int(t.shape[0]) if t is not None else 0


class ImageBatchInsert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "position": (_POS, {"default": "at end", "tooltip": "Where to insert. 'at index' inserts before the frame at 'index'; 'after index' inserts after it. 'index' is ignored for start/end."}),
                "index": ("INT", {"default": 0, "min": -100000, "max": 100000, "tooltip": "0-based target frame for 'at index' / 'after index'. Negative counts from the end (-1 = last)."}),
                "mode": (_MODES, {"default": "pad to largest", "tooltip": "How to reconcile different frame sizes (lossless, no resampling): 'as is' errors on a mismatch, 'crop to smallest', or 'pad to largest' (with pad_color)."}),
                "pad_color": ("STRING", {"default": "#000000", "tooltip": "Fill color (HEX) for borders added in 'pad to largest' mode."}),
            },
            "optional": {
                "batch": ("IMAGE", {"tooltip": "The batch to insert into. Leave empty to start a new batch from 'image'."}),
                "image": ("IMAGE", {"tooltip": "Image (or batch) to insert."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("batch", "count")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/image"

    def run(self, position="at end", index=0, mode="pad to largest", pad_color="#000000", batch=None, image=None):
        B = _len(batch)
        if position == "at start":
            pos = 0
        elif position == "at end":
            pos = B
        else:
            i = index if index >= 0 else B + index
            if position == "after index":
                i += 1
            pos = max(0, min(B, i))

        before = batch[:pos] if B else None
        after = batch[pos:] if B else None

        # Reuse Unlim Image Batch to reconcile sizes/channels and concatenate in slot order.
        kwargs, n = {}, 1
        for part in (before, image, after):
            if _len(part) > 0:
                kwargs[f"image_{n}"] = part
                n += 1
        if not kwargs:
            return (None, 0)
        out = UnlimImageBatch().run(mode=mode, pad_color=pad_color, skip_empty=True, **kwargs)[0]
        return (out, _len(out))


class ImageBatchRemove:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": -100000, "max": 100000, "tooltip": "0-based frame to start removing at. Negative counts from the end (-1 = last frame)."}),
                "count": ("INT", {"default": 1, "min": 0, "max": 100000, "tooltip": "How many frames to remove, starting at 'index'."}),
            },
            "optional": {
                "batch": ("IMAGE", {"tooltip": "The batch to remove frame(s) from."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("batch", "removed", "count")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/image"

    def run(self, index=0, count=1, batch=None):
        B = _len(batch)
        if B == 0:
            return (batch, None, 0)
        start = index if index >= 0 else B + index
        start = max(0, min(B, start))
        end = max(start, min(B, start + max(0, int(count))))

        removed = batch[start:end]
        remaining = torch.cat([batch[:start], batch[end:]], dim=0)
        return (
            remaining if _len(remaining) > 0 else None,
            removed if _len(removed) > 0 else None,
            _len(remaining),
        )


NODE_CLASS_MAPPINGS = {
    "ImageBatchInsert": ImageBatchInsert,
    "ImageBatchRemove": ImageBatchRemove,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchInsert": "Image Batch Insert",
    "ImageBatchRemove": "Image Batch Remove",
}
