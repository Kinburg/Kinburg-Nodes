"""Config-bundle nodes for the Local LLM nodes.

`Local LLM Settings (GGUF)` holds the model / sampling / loader / reasoning / output options and
hands them to an LLM node through a single ``config`` link — keeping the heavy nodes uncluttered.
`Vision Settings (GGUF)` is an optional sub-bundle (mmproj / handler / downscale) that plugs into
the Settings node the same way ``context`` does; connect it only when you want vision.
"""
from .llm_node import (
    LLM_CONFIG, VISION_CONFIG, _base_config_widgets, _list_mmproj, _VISION_HANDLER_LABELS,
)


class LocalLLMSettingsGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _base_config_widgets(),
            "optional": {
                "context": ("STRING", {"forceInput": True, "tooltip": "Reference material appended to the system prompt — e.g. Character Card / Context Collector output. Connect-only (no text field)."}),
                "vision": (VISION_CONFIG, {"tooltip": "Optional vision settings — wire a 'Vision Settings (GGUF)' node here to enable vision (mmproj / handler / downscale). Leave unconnected for text-only."}),
            },
        }

    RETURN_TYPES = (LLM_CONFIG,)
    RETURN_NAMES = ("config",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, vision=None, **kw):
        cfg = dict(kw)  # model, model_path, system_prompt, sampling, loader, reasoning, output, context
        if isinstance(vision, dict):
            cfg.update(vision)  # mmproj / mmproj_path / vision_handler / image_max_side
        return (cfg,)


class LocalLLMVisionSettingsGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mmproj": (_list_mmproj(), {"tooltip": "Projector mmproj .gguf from ComfyUI/models/llm (subfolders included, so you can keep a model + its mmproj together in one folder; mmproj-named files first). Choose the placeholder to type a path in mmproj_path."}),
                "mmproj_path": ("STRING", {"default": "", "tooltip": "Full path to the mmproj .gguf (when mmproj is the placeholder). Surrounding quotes are stripped."}),
                "vision_handler": (_VISION_HANDLER_LABELS, {"default": "auto (MTMD)", "tooltip": "auto (MTMD) is llama.cpp's generic multimodal loader and fits most modern vision GGUFs. Switch to the model's family only if auto fails."}),
                "image_max_side": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 64, "tooltip": "Downscale each image so its longest side is at most this many px before sending. 0 = full size."}),
            },
        }

    RETURN_TYPES = (VISION_CONFIG,)
    RETURN_NAMES = ("vision",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, mmproj, mmproj_path, vision_handler, image_max_side):
        return ({"mmproj": mmproj, "mmproj_path": mmproj_path,
                 "vision_handler": vision_handler, "image_max_side": image_max_side},)


NODE_CLASS_MAPPINGS = {
    "LocalLLMSettingsGGUF": LocalLLMSettingsGGUF,
    "LocalLLMVisionSettingsGGUF": LocalLLMVisionSettingsGGUF,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalLLMSettingsGGUF": "Local LLM Settings (GGUF)",
    "LocalLLMVisionSettingsGGUF": "Vision Settings (GGUF)",
}
