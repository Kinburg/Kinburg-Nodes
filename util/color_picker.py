"""Color Picker — the same handy swatch/native-picker control as Color Caption, standalone.

Pick a color (palette swatches + a native color picker, or type a HEX) and get it back as a
normalized `#RRGGBB` string plus its R / G / B components.
"""
import re

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")


def _normalize_hex(value, fallback="#FFFFFF"):
    if not value:
        return fallback
    m = _HEX_RE.match(str(value).strip())
    if not m:
        return fallback
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return "#" + h.upper()


def _hex_to_rgb(hexv):
    s = hexv.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


class ColorPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("STRING", {"default": "#FFFFFF", "tooltip": "Pick a color with the swatches / color picker above, or type a HEX value."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("hex", "R", "G", "B")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/util"

    def run(self, color):
        hexv = _normalize_hex(color)
        r, g, b = _hex_to_rgb(hexv)
        return (hexv, r, g, b)


NODE_CLASS_MAPPINGS = {"ColorPicker": ColorPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"ColorPicker": "Color Picker"}
