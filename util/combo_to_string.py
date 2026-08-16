"""Combo to String — turn a COMBO value (e.g. the `Primitive` node in combo mode) into a STRING.

ComfyUI won't let a COMBO output wire into a STRING input: a COMBO output only connects to a
COMBO input, and the two combos must even share a value. So this node's `value` is a COMBO
*widget* (that's how the Primitive delivers its value — it writes into a matching widget), and a
companion frontend patch (web/combo_to_string.js) lets a Primitive's COMBO output connect here.
Whatever value lands in the widget is returned as a STRING.

VALIDATE_INPUTS is permissive so any incoming value passes (the widget's own option list is just
a placeholder — the real value is pushed in by the connected Primitive).
"""

_PLACEHOLDER = "(connect a combo / Primitive)"
from ..categories import CAT_UTIL


class ComboToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ([_PLACEHOLDER], {"tooltip": "Connect a Primitive (combo) output here. Its selected value is passed through as a STRING."}),
            },
        }

    # Accept any value — the connected Primitive pushes in a value that isn't in the placeholder
    # option list, which the default combo check would otherwise reject.
    @classmethod
    def VALIDATE_INPUTS(cls, value):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "run"
    CATEGORY = CAT_UTIL

    def run(self, value):
        if isinstance(value, list):
            value = value[0] if value else ""
        return (str(value),)


NODE_CLASS_MAPPINGS = {"ComboToString": ComboToString}
NODE_DISPLAY_NAME_MAPPINGS = {"ComboToString": "Combo to String"}
