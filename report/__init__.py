from . import routes  # noqa: F401  -- importing registers the PromptServer routes

# This package contributes no nodes; it only adds the report DB + routes.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
