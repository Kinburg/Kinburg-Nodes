from . import routes  # noqa: F401  -- importing registers the /kinburg/showtext save route
from .text_node import (
    NODE_CLASS_MAPPINGS as _ST_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _ST_NAMES,
)

NODE_CLASS_MAPPINGS = {**_ST_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_ST_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
