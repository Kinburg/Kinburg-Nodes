"""Get Results (image) — collect all Set Results (image) of a name into one batch.

The frontend's Collect button wires each matching Set's output into this node (in index
order) as image_1, image_2, … inputs; the batching itself reuses Unlim Image Batch.
"""
from ..image_batch.batch_node import UnlimImageBatch
from ..util.text_concat import UnlimTextConcat

_MODES = ["as is", "crop to smallest", "pad to largest"]


class GetAccumImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "IMAGES", "tooltip": "Accumulator name to collect — must match the Set Accumulator (images) nodes. Press 'Collect' to (re)wire after adding/removing Sets."}),
                "mode": (_MODES, {"default": "as is", "tooltip": "How to reconcile inputs of different sizes (same as Unlim Image Batch)."}),
                "pad_color": ("STRING", {"default": "#000000", "tooltip": "Border color (HEX) for 'pad to largest'."}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "Skip empty / bypassed branches so the batch stays valid."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/accumulators"

    def run(self, name="", mode="as is", pad_color="#000000", skip_empty=True, **kwargs):
        # kwargs carries the frontend-wired image_1, image_2, … links — batch them.
        return UnlimImageBatch().run(mode=mode, pad_color=pad_color, skip_empty=skip_empty, **kwargs)


class GetAccumTexts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "TEXTS", "tooltip": "Accumulator name to collect — must match the Set Accumulator (texts) nodes. Press 'Collect' to (re)wire after adding/removing Sets."}),
                "separator": ("STRING", {"multiline": True, "default": "\n", "tooltip": "Joins the collected texts in index order. Multi-line allowed; default is a newline."}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "Skip empty / bypassed branches so they don't leave stray separators."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/accumulators"

    def run(self, name="", separator="\n", skip_empty=True, **kwargs):
        # kwargs carries the frontend-wired text_1, text_2, … links — join them.
        return UnlimTextConcat().run(separator=separator, skip_empty=skip_empty, **kwargs)


NODE_CLASS_MAPPINGS = {"GetAccumImages": GetAccumImages, "GetAccumTexts": GetAccumTexts}
NODE_DISPLAY_NAME_MAPPINGS = {"GetAccumImages": "Get Accumulator (images)", "GetAccumTexts": "Get Accumulator (texts)"}
