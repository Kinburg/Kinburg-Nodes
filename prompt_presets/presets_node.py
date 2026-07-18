"""Prompt Presets — five flexible preset slots, one prompt-fragment STRING output each.

Each slot has TWO dropdowns: a **category** selector and a **preset** selector (the presets of
the chosen category). Out of the box the five slots default to the classic camera / aesthetics /
light / medium / background categories, but any slot can point at any category — including
user-created ones. Categories and presets are managed from the node's UI and persisted on disk
by ``store``; selecting a preset resolves to its fragment at run time (``🚫 None`` -> "").
Saved "setups" (a named combination of all five slots' category+preset) are a frontend
convenience applied via the ``⚙ setup`` selector.
"""
from . import store


class PromptPresets:
    @classmethod
    def INPUT_TYPES(cls):
        cats = store.category_list()
        req = {}
        for i in range(1, store.N_SLOTS + 1):
            default_cat = store.SLOT_DEFAULT_CATS[i - 1] if i - 1 < len(store.SLOT_DEFAULT_CATS) else (cats[0] if cats else "")
            req[f"cat_{i}"] = (cats, {"default": default_cat, "tooltip": f"Slot {i}: which preset category this dropdown draws from."})
            req[f"preset_{i}"] = (store.options_for(default_cat), {"default": store.NONE, "tooltip": f"Slot {i}: the preset (from its category) — resolves to a prompt fragment."})
        return {"required": req}

    # Skip ComfyUI's strict combo-membership check: categories and presets are dynamic (added
    # live without an object_info reload). Resolution happens against the on-disk store.
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("STRING",) * store.N_SLOTS
    RETURN_NAMES = tuple(
        (store.SLOT_DEFAULT_CATS[i] if i < len(store.SLOT_DEFAULT_CATS) else f"slot_{i + 1}")
        for i in range(store.N_SLOTS))
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/prompt"

    def run(self, **kwargs):
        out = []
        for i in range(1, store.N_SLOTS + 1):
            cat = kwargs.get(f"cat_{i}", "")
            preset = kwargs.get(f"preset_{i}", store.NONE)
            out.append(store.resolve(cat, preset))
        return tuple(out)


NODE_CLASS_MAPPINGS = {"PromptPresets": PromptPresets}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptPresets": "Prompt Presets"}
