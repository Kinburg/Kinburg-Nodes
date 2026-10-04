"""Satyr — YuE2's plan, rewritten so the right singer gets the right words.

The model writes the melody; this suite only moves it. Who sings a phrase follows its register once
the register agrees with the lyric's marker — with a LoRA on the text encoder — so an octave decides
it; how many words fit is carried by its note count. Both are things YuE2 decides on its own and
neither is anything it lets you ask for.
"""
from . import routes  # noqa: F401  (registers Satyr Edit's PromptServer routes on import)
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
