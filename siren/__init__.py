"""Siren — a section-aware sampler for AceStep music latents (she sings; you pick the bars),
plus the scope that makes two takes comparable by eye."""
from .nodes import NODE_CLASS_MAPPINGS as _N, NODE_DISPLAY_NAME_MAPPINGS as _ND
from .scope import NODE_CLASS_MAPPINGS as _S, NODE_DISPLAY_NAME_MAPPINGS as _SD
from .compare import NODE_CLASS_MAPPINGS as _C, NODE_DISPLAY_NAME_MAPPINGS as _CD

NODE_CLASS_MAPPINGS = {**_N, **_S, **_C}
NODE_DISPLAY_NAME_MAPPINGS = {**_ND, **_SD, **_CD}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
