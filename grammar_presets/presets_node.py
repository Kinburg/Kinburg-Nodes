"""Prompt Presets — pick a saved prompt / GBNF grammar pair and emit both, plus the preset's name.

A preset holds a ``prompt`` and a ``grammar``, either of which may be empty: a system prompt that
asks for ``===PROMPT===`` / ``===NOTES===`` sections and the grammar that enforces them are saved as
one entry, so the two can't drift apart. Wire ``grammar`` into a **Local LLM (GGUF)** node's
``grammar_override`` and ``prompt`` into its ``system_override``; an empty field emits "", which that
node reads as "not set". The library lives on disk, not in the workflow, and is edited from the
node's 🗂 Manage dialog. It ships with read-only grammars for the Character / Entity cards and
Siren. The class id is still ``GrammarPresets`` so saved workflows keep loading.
Category ``Kinburg-Nodes/LLM/presets``.
"""
from . import store
from ..categories import CAT_LLM_PRESETS


class GrammarPresets:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (store.list_names(), {"default": store.NONE, "tooltip": "A saved prompt and/or GBNF grammar. Add, edit and rename them with '🗂 Manage'; type in the open list to search it."}),
            },
        }

    # Skip strict combo-membership so presets saved live (without an object_info reload) validate.
    @classmethod
    def VALIDATE_INPUTS(cls, preset):
        return True

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("grammar", "prompt", "name")
    OUTPUT_TOOLTIPS = (
        "The preset's GBNF grammar, \"\" if it has none. Wire into a Local LLM (GGUF) node's grammar_override.",
        "The preset's prompt, \"\" if it has none. A system prompt goes into a Local LLM (GGUF) node's system_override.",
        "The preset's name (\"\" for 🚫 None).",
    )
    FUNCTION = "run"
    CATEGORY = CAT_LLM_PRESETS

    def run(self, preset):
        if preset == store.NONE:
            return ("", "", "")
        p = store.get(preset)
        if p is None:
            raise ValueError(f"Prompt Presets: no preset named '{preset}' in the library. It was renamed "
                             f"or deleted, or this workflow comes from another machine.")
        return (p["grammar"], p["prompt"], preset)


NODE_CLASS_MAPPINGS = {"GrammarPresets": GrammarPresets}
NODE_DISPLAY_NAME_MAPPINGS = {"GrammarPresets": "Prompt Presets"}
