"""Prompt Presets — five preset dropdowns (camera / aesthetics / light / medium /
background), one prompt-fragment STRING output each.

Each dropdown lists built-in presets plus any the user has added (managed from the node's
UI and persisted on disk by ``store``). Selecting a preset resolves to its prompt fragment
at run time; ``🚫 None`` resolves to "". Saved "setups" (named combinations of all five
dropdowns) are a frontend convenience that just sets the dropdown values.
"""
from . import store


class PromptPresets:
    @classmethod
    def INPUT_TYPES(cls):
        def combo(cat, tip):
            return (store.options_for(cat), {"default": store.NONE, "tooltip": tip})
        return {
            "required": {
                "camera": combo("camera", "Shot / lens / angle preset."),
                "aesthetics": combo("aesthetics", "Overall art style preset."),
                "light": combo("light", "Lighting preset."),
                "medium": combo("medium", "Medium / material / format preset."),
                "background": combo("background", "Background / setting preset."),
            },
        }

    # Skips ComfyUI's strict combo-membership check so presets added live (without an
    # object_info reload) still validate. Resolution happens against the on-disk store.
    @classmethod
    def VALIDATE_INPUTS(cls, camera, aesthetics, light, medium, background):
        return True

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("camera", "aesthetics", "light", "medium", "background")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/prompt"

    def run(self, camera, aesthetics, light, medium, background):
        return (
            store.resolve("camera", camera),
            store.resolve("aesthetics", aesthetics),
            store.resolve("light", light),
            store.resolve("medium", medium),
            store.resolve("background", background),
        )


NODE_CLASS_MAPPINGS = {"PromptPresets": PromptPresets}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptPresets": "Prompt Presets"}
