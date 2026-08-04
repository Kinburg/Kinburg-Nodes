"""Card Presets — pick a saved Character / Entity card from a dropdown and emit its block.

Build a library once (Card Save from a photo, or a Character/Entity Card's 💾 save_preset_as),
then reuse any of them here without re-describing the same subject each run. The `card` output is
the same Markdown block the card nodes produce — feed it into Context Collector / an LLM `context`
input. An optional `filter` dropdown narrows the preset list to a single tag (a frontend
convenience; the chosen `preset` alone decides what's emitted). Category ``Kinburg-Nodes/LLM``.
"""
from . import store


class CardPresets:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (store.names(), {"default": store.NONE, "tooltip": "A saved Character/Entity card. Build the library with Card Save (from a photo) or the Character Card / Entity Card nodes (save_preset_as)."}),
            },
            "optional": {
                "filter": ([store.ALL_TAGS] + store.all_tags(), {"default": store.ALL_TAGS, "tooltip": "Narrow the preset list to one tag. Purely a picker convenience — it doesn't change the output. Tag cards from Card Save / the Manage dialog."}),
            },
        }

    # Skip strict combo-membership so presets/tags saved live (without an object_info reload) validate.
    @classmethod
    def VALIDATE_INPUTS(cls, preset, filter=None):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("card",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, preset="", filter=None):
        return (store.render(preset),)


NODE_CLASS_MAPPINGS = {"CardPresets": CardPresets}
NODE_DISPLAY_NAME_MAPPINGS = {"CardPresets": "Card Presets"}
