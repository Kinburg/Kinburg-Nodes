"""Ouroboros — self-correcting text→image sampler loop + its settings-bundle nodes."""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import stop  # noqa: F401  (registers the /kinburg/ouroboros/stop PromptServer route)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
