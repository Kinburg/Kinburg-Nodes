"""LoRA pair — pick LoRAs (with trigger words) and stack them onto a model in one node.

* `Lora Trigger Loader` is pure config: a searchable LoRA dropdown, a strength, and an optional
  trigger word. Its single `lora` output (type KINBURG_LORA) carries the spec {name, strength,
  trigger} — nothing is loaded here.
* `Lora Unlim Accumulator` takes a model (+ optional CLIP) and a prompt, plus an auto-growing
  list of `lora_*` specs. It loads and applies each LoRA to the model (and CLIP when connected),
  appends the non-empty trigger words to the prompt, and outputs the patched model/CLIP/prompt —
  plus a `triggers` output (just the comma-separated trigger words) to wire into Ouroboros's
  `trigger_words` so they survive the LLM prompt rewrite.

Heavy ComfyUI imports live inside the methods so the package still imports (and the Registry can
enumerate nodes) without ComfyUI present.
"""
import re
from ..categories import CAT_LORA

LORA_TYPE = "KINBURG_LORA"
_IDX_RE = re.compile(r"^lora_(\d+)$")


def _lora_list():
    """The LoRA filenames for the dropdown, or [] when ComfyUI isn't present (registry scan)."""
    try:
        import folder_paths
        return folder_paths.get_filename_list("loras")
    except Exception:
        return []


class LoraTriggerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (_lora_list(), {"tooltip": "LoRA file from ComfyUI/models/loras. The dropdown is searchable."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength applied to the diffusion model (UNET). Can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength applied to CLIP (text encoder). Only used when the accumulator has a CLIP connected. Can be negative."}),
            },
            "optional": {
                "trigger": ("STRING", {"default": "", "tooltip": "Optional trigger word(s) for this LoRA, added to the prompt by Lora Unlim Accumulator. Leave empty if the LoRA needs none."}),
            },
        }

    RETURN_TYPES = (LORA_TYPE,)
    RETURN_NAMES = ("lora",)
    FUNCTION = "run"
    CATEGORY = CAT_LORA

    def run(self, lora_name, strength_model, strength_clip, trigger=""):
        return ({"name": lora_name, "strength_model": float(strength_model),
                 "strength_clip": float(strength_clip), "trigger": (trigger or "").strip()},)


class LoraUnlimAccumulator:
    def __init__(self):
        self._cache = {}  # lora_path -> (state_dict, metadata), so re-runs don't reload from disk

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base diffusion model the LoRAs are applied to."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional CLIP. Connected -> LoRAs are applied to it too (with each LoRA's strength_clip) and returned; left empty -> model-only."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Text prompt (input only). The LoRAs' trigger words are appended to it, in slot order."}),
                "lora_1": (LORA_TYPE,),
                "lora_2": (LORA_TYPE,),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "prompt", "triggers")
    FUNCTION = "run"
    CATEGORY = CAT_LORA

    def run(self, model, clip=None, prompt="", **kwargs):
        # The frontend-wired lora_1, lora_2, … specs arrive via kwargs — apply them in slot order.
        specs = []
        for k in sorted((k for k in kwargs if _IDX_RE.match(k)), key=lambda x: int(_IDX_RE.match(x).group(1))):
            v = kwargs.get(k)
            if isinstance(v, dict) and v.get("name"):
                specs.append(v)

        triggers = []
        for spec in specs:
            sm = float(spec.get("strength_model", spec.get("strength", 1.0)))
            sc = float(spec.get("strength_clip", sm))
            eff_sc = sc if clip is not None else 0.0     # clip strength is moot without a CLIP
            if sm == 0 and eff_sc == 0:
                continue                                 # LoRA off -> skip it AND its trigger word
            model, clip = self._apply(model, clip, spec["name"], sm, eff_sc)
            trig = (spec.get("trigger") or "").strip()
            if trig:
                triggers.append(trig)

        # Triggers go in their own paragraph (blank line) after the prompt, comma-separated among
        # themselves. CLIP normalises the newlines to token boundaries, so this is just for
        # readability — it doesn't change how the model reads the trigger words.
        trigger_block = ", ".join(triggers)
        prompt_clean = prompt.strip() if prompt else ""
        if prompt_clean and trigger_block:
            new_prompt = prompt_clean + "\n\n" + trigger_block
        else:
            new_prompt = prompt_clean or trigger_block
        # `triggers` = just the comma-separated trigger words (no prompt) — wire it into
        # Ouroboros's `trigger_words` so they're always appended AFTER the LLM rewrites the prompt.
        return (model, clip, new_prompt, trigger_block)

    def _apply(self, model, clip, lora_name, strength_model, strength_clip):
        import comfy.sd
        import comfy.utils
        import folder_paths

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        cached = self._cache.get(lora_path)
        if cached is None:
            lora, meta = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
            self._cache[lora_path] = (lora, meta)
        else:
            lora, meta = cached

        model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=meta)
        return model, clip


NODE_CLASS_MAPPINGS = {
    "LoraTriggerLoader": LoraTriggerLoader,
    "LoraUnlimAccumulator": LoraUnlimAccumulator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraTriggerLoader": "Lora Trigger Loader",
    "LoraUnlimAccumulator": "Lora Unlim Accumulator",
}
