"""Get Results (image) — collect all Set Results (image) of a name into one batch.

The frontend's Collect button wires each matching Set's output into this node (in index
order) as image_1, image_2, … inputs; the batching itself reuses Unlim Image Batch.
"""
import json
import re

from ..image_batch.batch_node import UnlimImageBatch
from ..image_batch.list_node import UnlimImageList
from ..util.text_concat import UnlimTextConcat
from ..util.separators import BLOCK_JOINER
from ..categories import CAT_FLOW_ACCUMULATORS

_MODES = ["as is", "crop to smallest", "pad to largest"]

_DATA_RE = re.compile(r"^data_(\d+)$")
_AUDIO_RE = re.compile(r"^audio_(\d+)$")


def _data_idx(key):
    m = _DATA_RE.match(key)
    return int(m.group(1)) if m else 1 << 30


def _parse_dump(v):
    """A single GEN_INFO dump -> list of {class_type, ord, params} entries."""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return []
    return []


def _dump_is_empty(dump):
    """True if the dump carries no params at all (e.g. an empty / bypassed branch).

    Matches how the Generation Info Filter judges emptiness: a field exists only when some
    entry has a non-empty params dict.
    """
    return not any(isinstance(e, dict) and e.get("params") for e in dump)


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
    CATEGORY = CAT_FLOW_ACCUMULATORS

    def run(self, name="", mode="as is", pad_color="#000000", skip_empty=True, **kwargs):
        # kwargs carries the frontend-wired image_1, image_2, … links — batch them.
        return UnlimImageBatch().run(mode=mode, pad_color=pad_color, skip_empty=skip_empty, **kwargs)


class GetAccumImagesList:
    """Like Get Accumulator (images), but returns a ComfyUI image *list* instead of a batch, so
    accumulated images of different sizes coexist (no crop/pad). Collects from the same Set
    Accumulator (images) nodes by name."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "IMAGES", "tooltip": "Accumulator name to collect — must match the Set Accumulator (images) nodes. Press 'Collect' to (re)wire after adding/removing Sets."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = CAT_FLOW_ACCUMULATORS

    def run(self, name="", **kwargs):
        # kwargs carries the frontend-wired image_1, image_2, … links — emit them as a list
        # (each batch split into single frames), so different sizes travel separately.
        return UnlimImageList().run(**kwargs)


class GetAccumAudio:
    """Get Accumulator (audio) — collect every Set Accumulator (audio) of a name into a LIST.

    A list rather than a batch: two takes are only batchable when they share a length and a sample
    rate, and comparing takes of different lengths is a normal thing to want. Plug it into Siren
    Compare (Audio) where a stack of `audio_*` inputs used to go.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "AUDIO", "tooltip": "Accumulator name to collect — must match the Set Accumulator (audio) nodes. Press 'Collect' to (re)wire after adding/removing Sets."}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "Skip empty / bypassed branches so a muted take doesn't leave a hole in the list."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audios",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = CAT_FLOW_ACCUMULATORS

    def run(self, name="", skip_empty=True, **kwargs):
        # kwargs carries the frontend-wired audio_1, audio_2, … links, in index order.
        out = []
        for key in sorted((k for k in kwargs if _AUDIO_RE.match(k)),
                          key=lambda k: int(_AUDIO_RE.match(k).group(1))):
            v = kwargs.get(key)
            if isinstance(v, list):          # a wired list input arrives already flattened
                out.extend(a for a in v if isinstance(a, dict) and a.get("waveform") is not None)
            elif isinstance(v, dict) and v.get("waveform") is not None:
                out.append(v)
            elif not skip_empty:
                out.append(v)
        return (out,)


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
    CATEGORY = CAT_FLOW_ACCUMULATORS

    def run(self, name="", separator="\n", skip_empty=True, **kwargs):
        # kwargs carries the frontend-wired text_1, text_2, … links — join them.
        return UnlimTextConcat().run(separator=separator, skip_empty=skip_empty, **kwargs)


class GetAccumPrompts:
    """Get Accumulator (prompts) — collect texts into one Image Compare `prompts` string.

    A compare-tuned twin of Get Accumulator (texts): instead of a custom separator it joins
    the collected blocks with a hardcoded '---' line (BLOCK_JOINER), which is exactly how
    Image Compare (HTML) splits multi-line prompts back apart — so the two can't mismatch.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "PROMPTS", "tooltip": "Accumulator name to collect — must match the Set Accumulator (prompts) nodes. Press 'Collect' to (re)wire after adding/removing Sets."}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "Skip empty / bypassed branches so they don't leave stray separators."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "run"
    CATEGORY = CAT_FLOW_ACCUMULATORS

    def run(self, name="", skip_empty=True, **kwargs):
        # kwargs carries the frontend-wired text_1, text_2, … links — join them with a '---'
        # line so Image Compare's prompt splitter lines them up one block per image.
        return UnlimTextConcat().run(separator=BLOCK_JOINER, skip_empty=skip_empty, **kwargs)


class GetAccumCaptions:
    """Get Accumulator (captions) — collect texts into one Image Compare `captions` string.

    Like Get Accumulator (prompts) but joins with a plain newline: Image Compare reads one
    caption per line, so the hardcoded '\\n' separator is the right contract here.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "CAPTIONS", "tooltip": "Accumulator name to collect — must match the Set Accumulator (captions) nodes. Press 'Collect' to (re)wire after adding/removing Sets."}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "Skip empty / bypassed branches so they don't leave blank caption lines."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("captions",)
    FUNCTION = "run"
    CATEGORY = CAT_FLOW_ACCUMULATORS

    def run(self, name="", skip_empty=True, **kwargs):
        # kwargs carries the frontend-wired text_1, text_2, … links — one caption per line.
        return UnlimTextConcat().run(separator="\n", skip_empty=skip_empty, **kwargs)


class GetAccumGenInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "SETTINGS", "tooltip": "Accumulator name to collect — must match the Set Accumulator (gen info) nodes. Press 'Collect' to (re)wire after adding/removing Sets."}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "Skip empty / bypassed branches so they don't add empty blocks downstream."}),
            },
        }

    RETURN_TYPES = ("GEN_INFO_LIST",)
    RETURN_NAMES = ("data",)
    FUNCTION = "run"
    CATEGORY = CAT_FLOW_ACCUMULATORS

    def run(self, name="", skip_empty=True, **kwargs):
        # kwargs carries the frontend-wired data_1, data_2, … GEN_INFO links — bundle them in
        # index order into a JSON list-of-dumps for the Generation Info Filter node.
        dumps = []
        for k in sorted((k for k in kwargs if _DATA_RE.match(k)), key=_data_idx):
            v = kwargs.get(k)
            if v is None:
                continue
            dump = _parse_dump(v)
            if skip_empty and _dump_is_empty(dump):
                continue
            dumps.append(dump)
        return (json.dumps(dumps, ensure_ascii=False),)


# The former "Collect All Accumulators" helper node is gone: collecting only ever served a
# comparison, so its button and its `auto_collect` toggle now live on **Image Compare (HTML)**
# (see web/accumulators.js). Nothing about the Set/Get pairs themselves changed.

NODE_CLASS_MAPPINGS = {"GetAccumImages": GetAccumImages, "GetAccumImagesList": GetAccumImagesList, "GetAccumAudio": GetAccumAudio, "GetAccumTexts": GetAccumTexts, "GetAccumPrompts": GetAccumPrompts, "GetAccumCaptions": GetAccumCaptions, "GetAccumGenInfo": GetAccumGenInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"GetAccumImages": "Get Accumulator (images)", "GetAccumImagesList": "Get Accumulator (images list)", "GetAccumAudio": "Get Accumulator (audio)", "GetAccumTexts": "Get Accumulator (texts)", "GetAccumPrompts": "Get Accumulator (prompts)", "GetAccumCaptions": "Get Accumulator (captions)", "GetAccumGenInfo": "Get Accumulator (gen info)"}
