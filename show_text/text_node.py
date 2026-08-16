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

**Freeze / use-saved-text.** The ``value`` input is *lazy* and a ``use_saved_text`` toggle
picks the output source:
  * OFF (default) — evaluate ``value`` (running its upstream, e.g. an LLM) and output that;
    the shown/stored text tracks the input.
  * ON — output the text saved & edited in the node and **do not evaluate ``value``**, so the
    upstream never runs (that's ComfyUI lazy evaluation via ``check_lazy_status``, not just
    caching — it holds even after a restart or with a random seed). Edit the text and re-queue
    to push changes downstream without regenerating.
The edited text rides in a hidden, serialized ``saved_text`` widget (same pattern as the Chat
node's ``history_json``) so it reaches ``run()`` and survives a workflow reload / restart.
"""
import os
import json
from datetime import datetime

from ..util.anytype import ANY
from ..categories import CAT_UTIL


def _first(v, default=None):
    """Unwrap the 1-element list that INPUT_IS_LIST wraps scalar widgets in."""
    if isinstance(v, list):
        return v[0] if v else default
    return v


def value_to_text(value):
    """Convert any ComfyUI value into a display string. Bare lists are joined so nothing is
    dropped (INPUT_IS_LIST wraps every input in a list)."""
    if value is None:
        return ""
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
            "required": {},
            "optional": {
                # Lazy: in "use saved text" mode we don't request it, so its upstream (e.g. an
                # LLM) is never evaluated. Optional so the node can also run with nothing wired.
                "value": (ANY, {"lazy": True, "tooltip": "Anything — text, number, COMBO, dict/list… converted to text for display. LAZY: with 'use saved text' ON, this input (and its upstream, e.g. an LLM) is NOT evaluated."}),
                "markdown": ("BOOLEAN", {"default": True, "label_on": "markdown preview", "label_off": "raw / edit",
                                         "tooltip": "On: render the text as formatted markdown. Off: show an editable raw textarea. Toggling doesn't resize the node."}),
                "save_path": ("STRING", {"default": "", "tooltip": "Path for 💾 Save / autosave. Relative paths go under ComfyUI's output folder; a .md extension is added. {date}/{time}/{datetime} expand to the current date/time."}),
                "autosave": ("BOOLEAN", {"default": False, "label_on": "autosave on run", "label_off": "autosave off",
                                         "tooltip": "Write the text to save_path automatically on every run (needs a non-empty path)."}),
                "use_saved_text": ("BOOLEAN", {"default": False, "label_on": "🔒 saved text (skip upstream)", "label_off": "input (regenerate)",
                                               "tooltip": "Off (default): take the connected input and run its upstream (e.g. the LLM); the shown text tracks the input. On: output the text saved/edited here and DON'T evaluate the input, so the upstream never runs (holds after a restart / with a random seed). Edit the text and re-queue to push it downstream without regenerating."}),
                # Hidden, serialized carrier for the edited text (managed by web/show_text.js).
                # Single-line on purpose (its value still holds multiline text): a multiline
                # widget would leave a stray DOM textarea when hidden.
                "saved_text": ("STRING", {"default": "", "tooltip": "The node's stored text. Managed by the editor and hidden in the UI; it carries the edited text to run() and into the saved workflow."}),
            },
        }

    # Gather a whole batch/list into one view (and one saved file) rather than running per item.
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT_UTIL

    def check_lazy_status(self, value=None, use_saved_text=False, saved_text="",
                          markdown=True, save_path="", autosave=False):
        """Decide whether the (lazy) `value` input needs evaluating.
          * frozen (use_saved_text) → [] — never touch the upstream.
          * `value is None` → [] — the input is UNCONNECTED (the kwarg is omitted, so it
            defaults to None); requesting it would raise a strong-link error. Just run.
          * otherwise → ['value'] — it's connected (arrives as a list like (None,) while
            pending); request it so the upstream runs. On the re-visit it's available and
            ComfyUI drops the request, letting the node execute.
        """
        if bool(_first(use_saved_text, False)):
            return []
        if value is None:
            return []
        return ["value"]

    def run(self, value=None, markdown=True, save_path="", autosave=False,
            use_saved_text=False, saved_text=""):
        if bool(_first(use_saved_text, False)):
            saved = _first(saved_text, "")
            text = saved if isinstance(saved, str) else value_to_text(saved)
        else:
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
