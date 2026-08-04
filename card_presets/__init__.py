from .presets_node import (
    NODE_CLASS_MAPPINGS as _CDP_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CDP_NAMES,
)
from .save_node import (
    NODE_CLASS_MAPPINGS as _SAVE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SAVE_NAMES,
)
from . import routes  # noqa: F401  (registers PromptServer routes on import)

NODE_CLASS_MAPPINGS = {**_CDP_NODES, **_SAVE_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_CDP_NAMES, **_SAVE_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
