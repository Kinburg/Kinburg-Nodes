from .presets_node import (
    NODE_CLASS_MAPPINGS as _GP_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _GP_NAMES,
)
from . import routes  # noqa: F401  (registers PromptServer routes on import)

NODE_CLASS_MAPPINGS = {**_GP_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_GP_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
