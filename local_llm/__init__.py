from .llm_node import (
    NODE_CLASS_MAPPINGS as _LLM_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _LLM_NAMES,
)
from .settings_node import (
    NODE_CLASS_MAPPINGS as _SET_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SET_NAMES,
)
from .chat_node import (
    NODE_CLASS_MAPPINGS as _CHAT_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CHAT_NAMES,
)
from .token_node import (
    NODE_CLASS_MAPPINGS as _TOK_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _TOK_NAMES,
)
from .context_sizer import (
    NODE_CLASS_MAPPINGS as _CSZ_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CSZ_NAMES,
)
from .live_log_node import (
    NODE_CLASS_MAPPINGS as _LLOG_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _LLOG_NAMES,
)
from .send_image_node import (
    NODE_CLASS_MAPPINGS as _SEND_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SEND_NAMES,
)
from . import attachments as _attachments   # registers /kinburg/chat/discard

NODE_CLASS_MAPPINGS = {**_LLM_NODES, **_SET_NODES, **_CHAT_NODES, **_TOK_NODES, **_CSZ_NODES,
                       **_LLOG_NODES, **_SEND_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_LLM_NAMES, **_SET_NAMES, **_CHAT_NAMES, **_TOK_NAMES,
                              **_CSZ_NAMES, **_LLOG_NAMES, **_SEND_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
