"""Unlim Image List — collect an unlimited number of IMAGE inputs into an image *list*.

Unlike Unlim Image Batch (which stacks into a single, same-size tensor), this returns a
ComfyUI image list (``OUTPUT_IS_LIST``): every connected input is split into individual
frames and emitted as separate list items, so images of *different sizes* can travel
together. Downstream nodes then process them one at a time — handy for loop / iterator
setups: read the list length, pull an item by index, do the per-image work inside the loop.

The input list grows on demand (web/image_list.js), same as the other Unlim nodes. No
options — it just gathers whatever is connected, in slot order, untouched.
"""
import re

_IDX_RE = re.compile(r"^image_(\d+)$")


class UnlimImageList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image_1": ("IMAGE",)},
            "optional": {"image_2": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/image"

    @staticmethod
    def _index(key):
        m = _IDX_RE.match(key)
        return int(m.group(1)) if m else 1 << 30

    def run(self, **kwargs):
        out = []
        for key in sorted((k for k in kwargs if _IDX_RE.match(k)), key=self._index):
            v = kwargs.get(key)
            if v is None:
                continue
            for i in range(int(v.shape[0])):   # split any batch into single frames
                out.append(v[i:i + 1])         # keep IMAGE 4D: [1, H, W, C]
        return (out,)


NODE_CLASS_MAPPINGS = {"UnlimImageList": UnlimImageList}
NODE_DISPLAY_NAME_MAPPINGS = {"UnlimImageList": "Unlim Image List"}
