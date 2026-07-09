"""Show Text (Markdown) — display any input as text on the node.

Accepts ANY value on ``value`` and renders it as text (str; dict/list → pretty JSON;
bytes decoded). The frontend keeps the shown text inside the workflow (in the node's
``properties``), so it survives switching between workflow tabs — unlike the core Preview
Text node, which resets. A ``markdown`` toggle on the node flips between a rendered markdown
preview and an editable raw textarea *without changing the node size*.

``save_path`` + the 💾 Save button write the current text to a ``.md`` file (via a
PromptServer route in ``routes.py``); ``autosave`` does it automatically on every run. Paths
are resolved under ComfyUI's output dir when relative, always get a ``.md`` extension, and
support ``{date}`` / ``{time}`` / ``{datetime}`` placeholders.

The converted text is also returned as a STRING output (``text``), so the node can sit
inline in a wire and pass the text downstream.
"""
import os
import json
from datetime import datetime

from ..util.anytype import ANY


def _first(v, default=None):
    """Unwrap the 1-element list that INPUT_IS_LIST wraps scalar widgets in."""
    if isinstance(v, list):
        return v[0] if v else default
    return v


def value_to_text(value):
    """Convert any ComfyUI value into a display string. Bare lists are joined so nothing is
    dropped (INPUT_IS_LIST wraps every input in a list)."""
    if isinstance(value, list):
        if len(value) == 1:
            return value_to_text(value[0])
        return "\n\n".join(value_to_text(v) for v in value)
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "replace")
        except Exception:
            return repr(value)
    if isinstance(value, (dict, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    return str(value)


def _expand_tokens(path):
    """Expand {date}/{time}/{datetime} placeholders using the current local time."""
    if "{" not in path:
        return path
    now = datetime.now()
    return (path
            .replace("{datetime}", now.strftime("%Y-%m-%d_%H-%M-%S"))
            .replace("{date}", now.strftime("%Y-%m-%d"))
            .replace("{time}", now.strftime("%H-%M-%S")))


def resolve_md_path(path):
    """Expand tokens, force a ``.md`` extension, resolve relative paths under output/."""
    path = _expand_tokens((path or "").strip())
    if not path:
        return None
    root, ext = os.path.splitext(path)
    if ext.lower() != ".md":
        path = (root + ".md") if ext else (path + ".md")
    if not os.path.isabs(path):
        try:
            import folder_paths
            base = folder_paths.get_output_directory()
        except Exception:
            base = os.path.abspath("output")
        path = os.path.join(base, path)
    return path


def save_markdown(path, text):
    """Write ``text`` to a resolved ``.md`` path (parent dirs created). Returns the path."""
    resolved = resolve_md_path(path)
    if not resolved:
        raise ValueError("save path is empty")
    os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
    with open(resolved, "w", encoding="utf-8", newline="\n") as f:
        f.write(text if text is not None else "")
    return resolved


class KinburgShowText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY, {"tooltip": "Anything — text, a number, a COMBO, a dict/list… it's converted to text for display."}),
            },
            "optional": {
                "markdown": ("BOOLEAN", {"default": True, "label_on": "markdown preview", "label_off": "raw / edit",
                                         "tooltip": "On: render the text as formatted markdown. Off: show an editable raw textarea. Toggling doesn't resize the node."}),
                "save_path": ("STRING", {"default": "", "tooltip": "Path for 💾 Save / autosave. Relative paths go under ComfyUI's output folder; a .md extension is added. {date}/{time}/{datetime} expand to the current date/time."}),
                "autosave": ("BOOLEAN", {"default": False, "label_on": "autosave on run", "label_off": "autosave off",
                                         "tooltip": "Write the text to save_path automatically on every run (needs a non-empty path)."}),
            },
        }

    # Gather a whole batch/list into one view (and one saved file) rather than running per item.
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Kinburg-Nodes/util"

    def run(self, value, markdown=True, save_path="", autosave=False):
        text = value_to_text(value)
        path = _first(save_path, "")
        if _first(autosave, False) and (path or "").strip():
            try:
                save_markdown(path, text)
            except Exception as e:
                print(f"[KinburgShowText] autosave failed: {e}")
        return {"ui": {"kinburg_showtext": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {"KinburgShowText": KinburgShowText}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgShowText": "Show Text (Markdown)"}
