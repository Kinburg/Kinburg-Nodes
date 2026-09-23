"""Any Gate — one link, one toggle: pass the value through, or stop the branch dead.

With the toggle **on** the node is a plain wire: whatever comes in goes out, untouched and
un-inspected (the slots are the pack's wildcard `*`, so any type rides through).

With it **off** the node returns ``ExecutionBlocker(None)`` instead of the value. ComfyUI
propagates a blocker down the graph: every node that consumes this output is skipped silently,
and so is everything behind them — including output nodes, so nothing is saved and nothing is
previewed. The run just ends short; it is not an error and nothing is reported.

The `input` is **lazy**, so the branch *feeding* the gate is skipped as well: with the toggle
off ``check_lazy_status`` asks for nothing and ComfyUI never evaluates the upstream, so a
sampler sitting behind a closed gate does not run at all. That is the whole point of the node —
``Any Switch`` next door routes a value but still computes every branch it is choosing between.

`input` is *required* on purpose. A gate with nothing wired in has nothing to gate; letting it
sit there silently emitting ``None`` downstream hides the mistake, while ComfyUI's own "required
input is missing" says it out loud.
"""

# ComfyUI-only import, guarded so the pack still imports (and the Registry can enumerate nodes)
# without ComfyUI present. `comfy_execution.graph` re-exports the same class.
try:
    from comfy_execution.graph_utils import ExecutionBlocker
except Exception:
    ExecutionBlocker = None

from .anytype import ANY
from ..categories import CAT_UTIL


class AnyGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (ANY, {"lazy": True, "tooltip": "Anything — image, latent, model, string, conditioning… LAZY: with the gate closed this input is not evaluated, so its upstream never runs."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "open (pass through)", "label_off": "closed (stop here)",
                                        "tooltip": "Open: the input passes straight to the output. Closed: nothing leaves this node — every node downstream is skipped silently, and the upstream feeding this gate is never evaluated either."}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = CAT_UTIL
    DESCRIPTION = ("A valve on one link. Toggle it closed and the flow stops here: everything "
                   "downstream is skipped (no saves, no previews, no error), and the branch "
                   "feeding the gate is not evaluated either — the input is lazy, so a sampler "
                   "behind a closed gate never runs. Toggle it open and the node is a plain "
                   "wire. Takes and returns any type.")

    def check_lazy_status(self, input=None, enabled=True):
        """Open: request the input, which makes ComfyUI evaluate its upstream and come back here
        once the value exists (the core drops a request for an input it already has). Closed: ask
        for nothing, so that upstream is never run."""
        return ["input"] if bool(enabled) else []

    def run(self, input=None, enabled=True):
        if bool(enabled):
            return (input,)
        if ExecutionBlocker is None:      # no ComfyUI to block with — pass the hole through
            return (None,)
        # None = block silently: consumers are skipped without an error being reported.
        return (ExecutionBlocker(None),)


NODE_CLASS_MAPPINGS = {"KinburgAnyGate": AnyGate}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgAnyGate": "Any Gate"}
