"""Satyr — YuE2's plan, rewritten so the right singer gets the right words.

The model writes the melody; this suite only moves it. Who sings a phrase is carried by its register
and nothing else, so an octave decides it; how many words fit is carried by its note count, so the
notes are cut to the syllables. Both are things YuE2 decides on its own and neither is anything it
lets you ask for.
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
