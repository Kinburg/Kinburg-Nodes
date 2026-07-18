"""Grammar Presets — pick a GBNF grammar and emit it as a STRING.

Ships with templates that force an LLM's output into a fixed shape (Character Card / Entity
Card as JSON); add your own from the node UI (persisted on disk by ``store``). Wire the
``grammar`` output into a **Local LLM (GGUF)** node's ``grammar_override`` input, feed that node
a photo + a short prompt ("fill the card from this image"), and the vision model returns exactly
that structure. Category ``Kinburg-Nodes/LLM``.
"""
from . import store


class GrammarPresets:
    @classmethod
    def INPUT_TYPES(cls):
        names = store.list_names()
        default = "Character Card (JSON)" if "Character Card (JSON)" in names else (names[0] if names else store.NONE)
        return {
            "required": {
                "preset": (names, {"default": default, "tooltip": "The GBNF grammar template. Add your own with '➕ Add grammar'. Wire the 'grammar' output into a Local LLM (GGUF) node's grammar_override."}),
            },
        }

    # Skip strict combo-membership so grammars added live (without an object_info reload) validate.
    @classmethod
    def VALIDATE_INPUTS(cls, preset):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("grammar",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, preset=""):
        return (store.get(preset),)


NODE_CLASS_MAPPINGS = {"GrammarPresets": GrammarPresets}
NODE_DISPLAY_NAME_MAPPINGS = {"GrammarPresets": "Grammar Presets"}
