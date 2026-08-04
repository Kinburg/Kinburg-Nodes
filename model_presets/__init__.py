"""Model Library — pick a model and its known-good settings from two dropdowns.

  * **Model Capture** registers a model assembly (loaders + patches, however exotic) by reading it
    out of the graph once; the loaders can then be deleted from the workflow.
  * **Model Select** rebuilds the chosen model from that recipe and emits its saved sampler preset,
    so one node replaces a loader stack plus every "Sampler Settings for the other model".
  * **Settings Save** puts a settings chain into a model's library, with the score and time it was
    measured at.

See replay.py for how a stored recipe becomes live objects (and why no node type is hardcoded).
"""
from .capture_node import (
    NODE_CLASS_MAPPINGS as _CAP_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CAP_NAMES,
)
from .select_node import (
    NODE_CLASS_MAPPINGS as _SEL_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SEL_NAMES,
)
from .save_node import (
    NODE_CLASS_MAPPINGS as _SAV_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SAV_NAMES,
)
from .settings_node import (
    NODE_CLASS_MAPPINGS as _SET_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SET_NAMES,
)
from . import routes  # noqa: F401  (registers PromptServer routes on import)

NODE_CLASS_MAPPINGS = {**_CAP_NODES, **_SEL_NODES, **_SAV_NODES, **_SET_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_CAP_NAMES, **_SEL_NAMES, **_SAV_NAMES, **_SET_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
