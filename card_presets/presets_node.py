"""Card Presets — pick a saved Character / Entity card from a dropdown and emit its block.

Build a library once (fill a Character/Entity Card and press 💾 Save preset), then reuse any of
them here without re-describing the same subject each run. The `card` output is the same Markdown
block the card nodes produce — feed it into Context Collector / an LLM `context` input.
Category ``Kinburg-Nodes/LLM``.
"""
from . import store


class CardPresets:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (store.names(), {"default": store.NONE, "tooltip": "A saved Character/Entity card. Save cards from the Character Card / Entity Card nodes (💾 Save preset)."}),
            },
        }

    # Skip strict combo-membership so presets saved live (without an object_info reload) validate.
    @classmethod
    def VALIDATE_INPUTS(cls, preset):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("card",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, preset=""):
        return (store.render(preset),)


NODE_CLASS_MAPPINGS = {"CardPresets": CardPresets}
NODE_DISPLAY_NAME_MAPPINGS = {"CardPresets": "Card Presets"}
