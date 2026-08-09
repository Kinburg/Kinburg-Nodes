"""Morpheus — chain MiniMax H3 shots of 5-15 s into one long video with sound, and let a local LLM
write the whole storyboard that feeds it."""
from .nodes import NODE_CLASS_MAPPINGS as _N, NODE_DISPLAY_NAME_MAPPINGS as _ND
from .storyboard import NODE_CLASS_MAPPINGS as _S, NODE_DISPLAY_NAME_MAPPINGS as _SD
from .dream_board import NODE_CLASS_MAPPINGS as _B, NODE_DISPLAY_NAME_MAPPINGS as _BD

NODE_CLASS_MAPPINGS = {**_N, **_S, **_B}
NODE_DISPLAY_NAME_MAPPINGS = {**_ND, **_SD, **_BD}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
