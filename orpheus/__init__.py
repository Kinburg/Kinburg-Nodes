"""Orpheus — the one whose music moves everything else.

Siren writes the song; Phantas makes the pictures and Morpheus the motion between them. This suite
is the join: it reads a finished track and says **where the cuts fall**, so a music video is edited
to its own music instead of to a shot length someone typed.

Nothing here is a model. The cut points come from the song's own plan when Siren wrote it, and from
the signal when it did not; the arithmetic that reconciles the musical grid with H3's is in
`timing.py`, the measurements are in `detect.py`, and both are tested without audio, without weights
and without a GPU — which is the point of them being separate from the node at all.
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F401

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
