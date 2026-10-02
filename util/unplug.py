"""Any Unplug — keep the link, drop what travels down it.

The other half of ``Any Gate``, and its opposite. A closed gate stops the *run*: every node past
it is skipped. An unplugged link lets that node run exactly as it always does — it just finds
nothing in that one slot, as if the wire had been pulled out of the socket. Put it between a
Context Collector and a Settings node and the toggle turns the reference material off while the
LLM keeps generating.

What "nothing" is depends on who is listening, so `when_off` picks it — ComfyUI has no way to
un-send a value down a link that exists, something has to arrive:

* ``nothing`` — ``None``, which is what an unconnected optional input holds. Right for IMAGE,
  MODEL, CONDITIONING, a settings bundle… and for text inputs that read their value as
  ``(value or "")``, which is how this pack's own `context` input takes it.
* ``empty text`` — ``""``, for a node that joins or strips the string it is handed before
  looking at it. ``None`` there is an AttributeError; an empty string is a no-op.

The one thing a pulled wire cannot imitate: an optional input left unconnected falls back to the
default in the node's own signature, and that default is not always ``None``. Hand such a node
``nothing`` and it gets ``None`` instead of its default — ``empty text`` is the way out when that
default was a string.

The `input` is lazy, so switching the link off also stops the branch feeding it from running:
the context is not assembled while it is unplugged.
"""
from .anytype import ANY
from ..categories import CAT_UTIL

OFF_VALUES = ["nothing", "empty text"]


class AnyUnplug:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (ANY, {"lazy": True, "tooltip": "Anything — the value this node passes on, or withholds. LAZY: while unplugged it is not evaluated, so the branch feeding it never runs."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "connected", "label_off": "unplugged",
                                        "tooltip": "Connected: the value passes straight through. Unplugged: the node downstream still runs, it just receives nothing in this slot — as if the link had been pulled out."}),
                "when_off": (OFF_VALUES, {"default": "nothing",
                                          "tooltip": "What 'nothing' is while unplugged. 'nothing' = None, what an unconnected optional input holds — right for images, models, conditioning, config bundles and for text read as (value or ''). 'empty text' = \"\", for a node that would strip or join the string before checking it."}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = CAT_UTIL
    DESCRIPTION = ("A switch for one value, not for the run. Unplugged, the node downstream still "
                   "executes — it just gets nothing in that slot, the way it behaves with the "
                   "input left unconnected: an LLM keeps generating, only without the context "
                   "wired through here. Choose what 'nothing' means (None or an empty string) to "
                   "match what is listening. The input is lazy, so an unplugged branch is not "
                   "computed. Use Any Gate instead to stop the run outright.")

    def check_lazy_status(self, input=None, enabled=True, when_off="nothing"):
        """Unplugged: ask for nothing, so the branch feeding this node is never evaluated."""
        return ["input"] if bool(enabled) else []

    def run(self, input=None, enabled=True, when_off="nothing"):
        if bool(enabled):
            return (input,)
        return ("" if when_off == "empty text" else None,)


NODE_CLASS_MAPPINGS = {"KinburgAnyUnplug": AnyUnplug}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgAnyUnplug": "Any Unplug"}
