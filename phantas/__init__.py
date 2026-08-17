"""Phantas — the keyframes Morpheus dreams between.

Morpheus' brother writes stills, not motion: a brief becomes a chain of keyframe prompts, and the
sampler renders them on one seed with one style, so the pictures are consistent enough to be the
boundaries of a continuous take. What comes out is a `MORPHEUS_SHOT` chain with the prompts left
empty — Morpheus Storyboard fills those in place and Morpheus renders the video between them.
"""
from .board import NODE_CLASS_MAPPINGS as _B, NODE_DISPLAY_NAME_MAPPINGS as _BD
from .nodes import NODE_CLASS_MAPPINGS as _N, NODE_DISPLAY_NAME_MAPPINGS as _ND

NODE_CLASS_MAPPINGS = {**_B, **_N}
NODE_DISPLAY_NAME_MAPPINGS = {**_BD, **_ND}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
