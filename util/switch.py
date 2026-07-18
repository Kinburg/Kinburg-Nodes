"""Any Switch — forward one of several inputs, chosen by `select`.

The `input_*` slots accept any type and auto-grow (web/switch.js): the node shows the
connected slots plus one spare. `select` (1-based) picks which slot to pass through; the
connected slots are contiguous (input_1, input_2, …), so `select = 2` is the second one.
If `select` points past the connected inputs it is clamped to the nearest valid slot.

Note: this is a plain forwarding switch — ComfyUI evaluates every connected input before
`run`, so the unselected branches are still computed. It routes the value, it does not skip
upstream work.
"""
import re

from .anytype import ANY

MAX_SLOTS = 20
_IDX_RE = re.compile(r"^input_(\d+)$")


class AnySwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": MAX_SLOTS, "tooltip": "Which connected input to pass through (1-based). Clamped to the number of connected inputs."}),
            },
            "optional": {"input_1": (ANY,), "input_2": (ANY,)},
        }

    RETURN_TYPES = (ANY, "INT")
    RETURN_NAMES = ("output", "selected")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/util"

    @staticmethod
    def _index(key):
        m = _IDX_RE.match(key)
        return int(m.group(1)) if m else 1 << 30

    def run(self, select=1, **kwargs):
        # Connected inputs in slot order (only those actually wired arrive in kwargs).
        connected = [(self._index(k), kwargs[k]) for k in kwargs
                     if _IDX_RE.match(k) and kwargs[k] is not None]
        connected.sort(key=lambda p: p[0])
        if not connected:
            print("[AnySwitch] no inputs connected — output is None")
            return (None, 0)
        # `select` is a 1-based slot number; clamp into the connected slots.
        pos = max(1, min(int(select), len(connected)))
        slot_no, value = connected[pos - 1]
        return (value, slot_no)


NODE_CLASS_MAPPINGS = {"KinburgAnySwitch": AnySwitch}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgAnySwitch": "Any Switch"}
