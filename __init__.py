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
from .loops import (
    NODE_CLASS_MAPPINGS as _LOOP_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _LOOP_NAMES,
)
from .lora import (
    NODE_CLASS_MAPPINGS as _LORA_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _LORA_NAMES,
)
from .prompt_presets import (
    NODE_CLASS_MAPPINGS as _PP_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _PP_NAMES,
)
from .list_ops import (
    NODE_CLASS_MAPPINGS as _LO_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _LO_NAMES,
)
from .llm_server import (
    NODE_CLASS_MAPPINGS as _SRV_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SRV_NAMES,
)
from .context import (
    NODE_CLASS_MAPPINGS as _CTX_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CTX_NAMES,
)
from .gguf_convert import (
    NODE_CLASS_MAPPINGS as _GGUFC_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _GGUFC_NAMES,
)
from .show_text import (
    NODE_CLASS_MAPPINGS as _ST_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _ST_NAMES,
)
from .save_song import (
    NODE_CLASS_MAPPINGS as _SS_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SS_NAMES,
)
from .group_control import (
    NODE_CLASS_MAPPINGS as _GC_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _GC_NAMES,
)
from .vision_judge import (
    NODE_CLASS_MAPPINGS as _VJ_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _VJ_NAMES,
)
from .prompt_variations import (
    NODE_CLASS_MAPPINGS as _PV2_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _PV2_NAMES,
)
from .grammar_presets import (
    NODE_CLASS_MAPPINGS as _GRM_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _GRM_NAMES,
)
from .card_presets import (
    NODE_CLASS_MAPPINGS as _CDP_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CDP_NAMES,
)

NODE_CLASS_MAPPINGS = {**_LLM_NODES, **_IC_NODES, **_IB_NODES, **_UTIL_NODES, **_TIMER_NODES, **_GI_NODES, **_RPT_NODES, **_ACCUM_NODES, **_COLLAGE_NODES, **_LOOP_NODES, **_LORA_NODES, **_PP_NODES, **_LO_NODES, **_SRV_NODES, **_CTX_NODES, **_GGUFC_NODES, **_ST_NODES, **_SS_NODES, **_GC_NODES, **_VJ_NODES, **_PV2_NODES, **_GRM_NODES, **_CDP_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_LLM_NAMES, **_IC_NAMES, **_IB_NAMES, **_UTIL_NAMES, **_TIMER_NAMES, **_GI_NAMES, **_RPT_NAMES, **_ACCUM_NAMES, **_COLLAGE_NAMES, **_LOOP_NAMES, **_LORA_NAMES, **_PP_NAMES, **_LO_NAMES, **_SRV_NAMES, **_CTX_NAMES, **_GGUFC_NAMES, **_ST_NAMES, **_SS_NAMES, **_GC_NAMES, **_VJ_NAMES, **_PV2_NAMES, **_GRM_NAMES, **_CDP_NAMES}
WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
