"""Kinburg-Nodes — a personal collection of custom ComfyUI nodes.

The root package aggregates the node mappings from every sub-collection and
exposes a single WEB_DIRECTORY for all of their frontend extensions.
"""
from .local_llm import (
    NODE_CLASS_MAPPINGS as _LLM_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _LLM_NAMES,
)
from .image_compare import (
    NODE_CLASS_MAPPINGS as _IC_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _IC_NAMES,
)
from .util import (
    NODE_CLASS_MAPPINGS as _UTIL_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _UTIL_NAMES,
)

NODE_CLASS_MAPPINGS = {**_LLM_NODES, **_IC_NODES, **_UTIL_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_LLM_NAMES, **_IC_NAMES, **_UTIL_NAMES}
WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
