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
from .image_batch import (
    NODE_CLASS_MAPPINGS as _IB_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _IB_NAMES,
)
from .util import (
    NODE_CLASS_MAPPINGS as _UTIL_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _UTIL_NAMES,
)
from .timer import (
    NODE_CLASS_MAPPINGS as _TIMER_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _TIMER_NAMES,
)
from .gen_info import (
    NODE_CLASS_MAPPINGS as _GI_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _GI_NAMES,
)
from .report import (
    NODE_CLASS_MAPPINGS as _RPT_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _RPT_NAMES,
)
from .accumulators import (
    NODE_CLASS_MAPPINGS as _ACCUM_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _ACCUM_NAMES,
)
from .collage import (
    NODE_CLASS_MAPPINGS as _COLLAGE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _COLLAGE_NAMES,
)

NODE_CLASS_MAPPINGS = {**_LLM_NODES, **_IC_NODES, **_IB_NODES, **_UTIL_NODES, **_TIMER_NODES, **_GI_NODES, **_RPT_NODES, **_ACCUM_NODES, **_COLLAGE_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_LLM_NAMES, **_IC_NAMES, **_IB_NAMES, **_UTIL_NAMES, **_TIMER_NAMES, **_GI_NAMES, **_RPT_NAMES, **_ACCUM_NAMES, **_COLLAGE_NAMES}
WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
