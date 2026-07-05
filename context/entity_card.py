"""Entity Card — a free-form sibling of Character Card: one Name + one Description for
anything that isn't a person (an object, interior, building, place, faction, …).

Where Character Card has fixed attribute fields, this node is deliberately loose: you write
the description however you like, and the `card` output is a small Markdown block —

    ### Cafe
    Cluttered with old bronze pitchers on every shelf — the place's signature look.

meant to be gathered by Context Collector and fed into an LLM node's `context` input. Binding
the description under a named heading lets the model latch onto that name: mention "Cafe" in
the request and it weaves the pitchers into the expanded prompt. Category `Kinburg-Nodes/LLM`.
"""


class EntityCard:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "tooltip": "The thing's name / label (e.g. 'Cafe', 'Bronze pitcher', 'Old town hall'). Becomes the card heading the LLM binds the description to. Empty = description is emitted on its own, with no heading."}),
                "description": ("STRING", {"multiline": True, "default": "", "tooltip": "Free-form description of the entity — looks, materials, mood, signature details… Written verbatim under the heading. Empty (with an empty name) => nothing is emitted, so Context Collector skips this card."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("card",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, name="", description=""):
        name = (name or "").strip()
        description = (description or "").strip()

        # Nothing filled in -> emit an empty string so Context Collector skips this card.
        if not name and not description:
            return ("",)
        # No name -> just the free-form text (a loose note, no heading to bind to).
        if not name:
            return (description,)

        block = f"### {name}"
        if description:
            block += "\n" + description
        return (block,)


NODE_CLASS_MAPPINGS = {"EntityCard": EntityCard}
NODE_DISPLAY_NAME_MAPPINGS = {"EntityCard": "Entity Card"}
