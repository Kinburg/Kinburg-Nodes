from . import routes  # noqa: F401  -- importing registers the PromptServer routes
from .presets_node import (
    NODE_CLASS_MAPPINGS as _PP_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _PP_NAMES,
)

NODE_CLASS_MAPPINGS = {**_PP_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_PP_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
