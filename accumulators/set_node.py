"""Set Results / Get Results — name-based accumulators (image / text / gen-info pairs).

`Set Results (image)` is a labelled pass-through: connect your final image and give it an
accumulator name + index. Copy the flow and the index auto-increments (frontend). A
`Get Results (image)` with the same name has a **Collect** button that physically wires every
matching Set's output into it, in index order, and batches them — plug it where an image
batch used to go (e.g. Image Compare).

The wiring is plain real links created on the frontend (web/accumulators.js), so execution and
caching are completely standard — no runtime globals, no prompt magic.
"""


class SetAccumImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "name": ("STRING", {"default": "IMAGES", "tooltip": "Accumulator name. A Get Accumulator (images) with the same name collects every Set that shares it."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "tooltip": "Position in the collected batch. Auto-increments when you copy the node."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/accumulators"

    def run(self, image, name="", index=0):
        return (image,)


class SetAccumTexts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "name": ("STRING", {"default": "TEXTS", "tooltip": "Accumulator name. A Get Accumulator (texts) with the same name collects every Set that shares it."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "tooltip": "Position in the joined output. Auto-increments when you copy the node."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/accumulators"

    def run(self, text, name="", index=0):
        return (text,)


class SetAccumPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "name": ("STRING", {"default": "PROMPTS", "tooltip": "Accumulator name. A Get Accumulator (prompts) with the same name collects every Set that shares it."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "tooltip": "Position in the joined output. Auto-increments when you copy the node."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/accumulators"

    def run(self, text, name="", index=0):
        return (text,)


class SetAccumCaptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "name": ("STRING", {"default": "CAPTIONS", "tooltip": "Accumulator name. A Get Accumulator (captions) with the same name collects every Set that shares it."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "tooltip": "Position in the joined output. Auto-increments when you copy the node."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/accumulators"

    def run(self, text, name="", index=0):
        return (text,)


class SetAccumGenInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("GEN_INFO",),
                "name": ("STRING", {"default": "SETTINGS", "tooltip": "Accumulator name. A Get Accumulator (gen info) with the same name collects every Set that shares it."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "tooltip": "Position in the collected bundle. Auto-increments when you copy the node."}),
            },
        }

    RETURN_TYPES = ("GEN_INFO",)
    RETURN_NAMES = ("data",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/accumulators"

    def run(self, data, name="", index=0):
        return (data,)


NODE_CLASS_MAPPINGS = {"SetAccumImages": SetAccumImages, "SetAccumTexts": SetAccumTexts, "SetAccumPrompts": SetAccumPrompts, "SetAccumCaptions": SetAccumCaptions, "SetAccumGenInfo": SetAccumGenInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"SetAccumImages": "Set Accumulator (images)", "SetAccumTexts": "Set Accumulator (texts)", "SetAccumPrompts": "Set Accumulator (prompts)", "SetAccumCaptions": "Set Accumulator (captions)", "SetAccumGenInfo": "Set Accumulator (gen info)"}
