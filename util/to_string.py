"""Any -> String — a tiny converter for values you can't otherwise wire into text nodes.

ComfyUI types outputs strictly: a **COMBO** output (e.g. the `Primitive` node in combo mode,
or an enum-style widget converted to an input) only connects to COMBO inputs, so it can't feed
a `STRING` input like Preview Text or a prompt box — even though the value is just a string.

This node's `value` input is the wildcard type `*`, which accepts any output (COMBO included).
It returns `str(value)`, giving you a real STRING to route wherever you like.
"""


from .anytype import ANY


class AnyToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY, {"tooltip": "Any value — including a COMBO output (e.g. Primitive) that won't connect to STRING inputs directly."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/util"

    def run(self, value):
        # INPUT_IS_LIST isn't set, so `value` is a single item; guard the odd list case anyway.
        if isinstance(value, list):
            value = value[0] if value else ""
        return (str(value),)


NODE_CLASS_MAPPINGS = {"AnyToString": AnyToString}
NODE_DISPLAY_NAME_MAPPINGS = {"AnyToString": "Any to String"}
