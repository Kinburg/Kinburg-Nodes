"""Character Card — fill a few fields, get one tidy character block for the LLM context.

Empty fields are skipped, so you only describe what matters. The `card` output is a small
Markdown block (a heading with the name + a bullet per filled attribute + free-form notes)
meant to be gathered by Context Collector and fed into an LLM node's `context` input, e.g.

    ### Vasya
    - Gender: male
    - Age: 35
    - Eyes: brown
    - Hair color: dark brown

Giving the model such reference cards lets it weave each named character's looks into an
expanded image prompt (see README). Category `Kinburg-Nodes/LLM`.
"""

# (widget name, display label) in output order. All are plain strings; empty ones are dropped.
_FIELDS = [
    ("gender", "Gender"),
    ("age", "Age"),
    ("ethnicity", "Ethnicity / skin"),
    ("eye_color", "Eyes"),
    ("hair_color", "Hair color"),
    ("hair_style", "Hair style"),
    ("build", "Build"),
    ("height", "Height"),
    ("outfit", "Outfit / clothing"),
    ("features", "Distinctive features"),
]


class CharacterCard:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {"default": "", "tooltip": "Leave empty to omit this line from the card."}
        fields = {
            "name": ("STRING", {"default": "", "tooltip": "Character name / label — becomes the card heading and the name the LLM binds attributes to."}),
        }
        for key, label in _FIELDS:
            fields[key] = ("STRING", dict(opt, tooltip=f"{label}. " + opt["tooltip"]))
        fields["notes"] = ("STRING", {"multiline": True, "default": "", "tooltip": "Anything else (personality, accessories, scars…). Added verbatim under the card."})
        return {"required": fields}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("card",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, name="", notes="", **kwargs):
        lines = []
        for key, label in _FIELDS:
            v = (kwargs.get(key) or "").strip()
            if v:
                lines.append(f"- {label}: {v}")
        notes = (notes or "").strip()

        # Nothing filled in -> emit an empty string so Context Collector skips this card.
        if not lines and not notes:
            return ("",)

        header = (name or "").strip() or "Character"
        block = [f"### {header}"] + lines
        if notes:
            block.append(notes)
        return ("\n".join(block),)


NODE_CLASS_MAPPINGS = {"CharacterCard": CharacterCard}
NODE_DISPLAY_NAME_MAPPINGS = {"CharacterCard": "Character Card"}
