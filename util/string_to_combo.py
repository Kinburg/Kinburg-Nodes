"""String to Combo — feed a STRING into another node's dropdown (sampler_name, scheduler, ckpt_name …).

The reverse of Combo to String. A STRING output won't wire into a COMBO input, but the wildcard
`ANY` output does: the frontend lets `*` connect anywhere, and ComfyUI's backend checks a linked
COMBO input's type only, never its value against the option list. So the value must match one of
the target's options exactly — a typo fails in the target node, not here.
"""
from .anytype import ANY
from ..categories import CAT_UTIL


class StringToCombo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True, "tooltip": "The option to select, e.g. 'euler'. Surrounding whitespace (an LLM's trailing newline) is stripped; the rest must match one of the target dropdown's options exactly."}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("combo",)
    OUTPUT_TOOLTIPS = ("Connect to any dropdown (COMBO) input.",)
    FUNCTION = "run"
    CATEGORY = CAT_UTIL

    def run(self, string):
        return (string.strip(),)


NODE_CLASS_MAPPINGS = {"StringToCombo": StringToCombo}
NODE_DISPLAY_NAME_MAPPINGS = {"StringToCombo": "String to Combo"}
