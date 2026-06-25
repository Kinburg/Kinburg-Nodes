"""Color Caption — type a caption and a color.

Outputs a compact one-line JSON `{"caption", "color", "band_color"}` meant to be wired
into the Image Compare node's `captions` input (one caption per line): `color` tints the
text, `band_color` fills the caption band behind it. The JSON is plain enough to be
useful anywhere a string is expected.
"""
import json
import re

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")


def _normalize_hex(value, fallback="#FFFFFF"):
    """Coerce a user/HEX string to '#RRGGBB' upper-case; fall back if invalid."""
    if not value:
        return fallback
    m = _HEX_RE.match(str(value).strip())
    if not m:
        return fallback
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return "#" + h.upper()


class ColorCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"multiline": True, "default": "", "tooltip": "The caption text."}),
                "color": ("STRING", {"default": "#FFFFFF", "tooltip": "Text color as HEX (#RRGGBB)."}),
                "band_color": ("STRING", {"default": "#000000", "tooltip": "Color of the band behind the text, as HEX (#RRGGBB)."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption_json",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/image/compare"

    def run(self, caption, color, band_color):
        return (json.dumps(self._payload(caption, color, band_color), ensure_ascii=False),)

    @staticmethod
    def _payload(caption, color, band_color):
        return {"caption": caption, "color": _normalize_hex(color),
                "band_color": _normalize_hex(band_color, "#000000")}

    @classmethod
    def IS_CHANGED(cls, caption, color, band_color):
        return json.dumps(cls._payload(caption, color, band_color), ensure_ascii=False)


NODE_CLASS_MAPPINGS = {"ColorCaption": ColorCaption}
NODE_DISPLAY_NAME_MAPPINGS = {"ColorCaption": "Color Caption"}
