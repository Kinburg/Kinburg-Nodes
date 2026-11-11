"""Card Presets — pick a saved Character / Entity card from a dropdown and emit its block.

Build a library once (Card Save from a photo, or a Character/Entity Card's 💾 save_preset_as),
then reuse any of them here without re-describing the same subject each run. The `card` output is
the same Markdown block the card nodes produce — feed it into Context Collector / an LLM `context`
input. An optional `filter` dropdown narrows the preset list to a single tag (a frontend
convenience; the chosen `preset` alone decides what's emitted). The `voice` output is the same
hand-off Character Card gives Siren Cast, so a saved band member can drive the music too (empty
for an entity preset, or for a character saved before the voice fields existed). The `picture`
output is the card's saved picture — a character sheet or a location — ready for a reference-image
input such as MiniMax H3 Reference to Video's; None when the card has none. The `cast` output is
the whole card for Morpheus' `cast_N` inputs: name, type, field values and the picture, so Morpheus
can decide per shot who is in it and attach their reference.

No widget may be added here: this node's JS buttons are saved positionally after its widgets, so a
new widget would read a saved graph's button `null` as its value.
Category ``Kinburg-Nodes/LLM``.
"""
import json

import numpy as np
import torch
from PIL import Image

from . import store
from ..context.character_card import VOICE_TYPE
from ..categories import CAT_LLM_PRESETS

# Typed hand-off to Morpheus: {"name", "type", "values", "picture"}.
CAST_TYPE = "KINBURG_CAST"


def _picture(preset):
    path = store.picture_path((store.get(preset) or {}).get("picture"))
    if not path:
        return None
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None]


class CardPresets:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (store.names(), {"default": store.NONE, "tooltip": "A saved Character/Entity card. Build the library with Card Save (from a photo) or the Character Card / Entity Card nodes (save_preset_as)."}),
            },
            "optional": {
                "filter": ([store.ALL_TAGS] + store.all_tags(), {"default": store.ALL_TAGS, "tooltip": "Narrow the preset list to one tag. Purely a picker convenience — it doesn't change the output. Tag cards from Card Save / the Manage dialog."}),
            },
        }

    # Skip strict combo-membership so presets/tags saved live (without an object_info reload) validate.
    @classmethod
    def VALIDATE_INPUTS(cls, preset, filter=None):
        return True

    # The inputs are only a name, so without this an edit made in 🗂 Manage — new fields, a new
    # picture — would keep coming out of the cache as the old card.
    @classmethod
    def IS_CHANGED(cls, preset, filter=None):
        return json.dumps(store.get(preset), sort_keys=True, ensure_ascii=False)

    RETURN_TYPES = ("STRING", VOICE_TYPE, "IMAGE", CAST_TYPE)
    RETURN_NAMES = ("card", "voice", "picture", "cast")
    OUTPUT_TOOLTIPS = (
        "The card's Markdown block — feed Context Collector or an LLM's context input.",
        "The singing voice, for Siren Cast. Empty for an entity card.",
        "The card's saved picture (a character sheet, a location) for a reference-image input. Nothing comes out when the card has no picture — add one in 🗂 Manage or wire 'picture' on the node that saves the card.",
        "The whole card for Morpheus' 'cast_N' inputs. Morpheus puts it into a shot when the shot's prompt names it (outside quotes), and attaches its picture as a reference. A card with no picture goes in as words only — faster, and a location described in text often works well enough.",
    )
    FUNCTION = "run"
    CATEGORY = CAT_LLM_PRESETS

    def run(self, preset="", filter=None):
        picture = _picture(preset)
        p = store.get(preset) if preset != store.NONE else None
        cast = None
        if p:
            values = p.get("values") or {}
            cast = {"name": str(values.get("name") or "").strip() or preset,
                    "type": p.get("type") or "character", "values": values, "picture": picture}
        return (store.render(preset), store.voice(preset), picture, cast)


NODE_CLASS_MAPPINGS = {"CardPresets": CardPresets}
NODE_DISPLAY_NAME_MAPPINGS = {"CardPresets": "Card Presets"}
