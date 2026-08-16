"""Unlim Text Concat — join an unlimited number of string inputs with a separator.

The node ships with `text_1` (required) and `text_2` (optional); the frontend
(web/text_concat.js) grows the input list on demand — connect the last slot and a new
one appears, disconnect and trailing empties collapse. The execution backend passes any
connected (even undeclared) link input through to `run`, so the extra `text_N` slots
arrive via **kwargs and are joined in slot order.
"""
import re
from ..categories import CAT_UTIL

_IDX_RE = re.compile(r"^text_(\d+)$")


class UnlimTextConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"forceInput": True}),
                "separator": ("STRING", {"multiline": True, "default": "\n", "tooltip": "Inserted between the inputs. Multi-line is fine; the default is a newline (one input per line)."}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "Skip empty / unconnected inputs so they don't produce stray separators."}),
            },
            "optional": {
                "text_2": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = CAT_UTIL

    @staticmethod
    def _index(key):
        m = _IDX_RE.match(key)
        return int(m.group(1)) if m else 1 << 30

    def run(self, separator="\n", skip_empty=True, **kwargs):
        parts = []
        for key in sorted((k for k in kwargs if _IDX_RE.match(k)), key=self._index):
            value = kwargs.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                value = str(value)
            if skip_empty and value == "":
                continue
            parts.append(value)
        return (separator.join(parts),)


NODE_CLASS_MAPPINGS = {"UnlimTextConcat": UnlimTextConcat}
NODE_DISPLAY_NAME_MAPPINGS = {"UnlimTextConcat": "Unlim Text Concat"}
