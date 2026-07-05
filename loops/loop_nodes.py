"""Loop nodes — flexible iteration for ComfyUI.

ComfyUI's execution graph is acyclic, so "loops" are built two ways, both used here:

* **For Each** (`For Each (Open)` / `Collect`) — a *map*: Open fans a batch/list out one item
  at a time (OUTPUT_IS_LIST), the graph downstream runs once per item, Collect gathers the
  per-item results back into a batch/list (INPUT_IS_LIST). No state carried between items.

* **Repeat** (`Repeat (Open)` / `Repeat (Close)`) — a real iterative loop with carried state,
  built on ComfyUI's **graph expansion** (`GraphBuilder` / the `"expand"` return): each pass
  the Close node clones the loop body wired back to itself for the next iteration, all inside
  one queue run. State travels through wildcard `*` slots, so any number of any-typed values
  (image, latent, int, string, conditioning, …) ride through the loop.

Both families use auto-growing wildcard slots driven by web/loops.js: the node shows only the
connected slots plus one spare, but the backend always declares MAX_SLOTS of them.
"""

import time

# Heavy / ComfyUI-only imports are guarded so the package still imports (and the Registry can
# enumerate nodes) without ComfyUI present. At runtime these are always available.
try:
    import torch
except Exception:
    torch = None

try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
    from comfy_execution.graph import ExecutionBlocker
except Exception:
    GraphBuilder = None
    is_link = None
    ExecutionBlocker = None

MAX_SLOTS = 10
FLOW_TYPE = "KINBURG_LOOP_FLOW"


from ..util.anytype import ANY


class _ByPassTuple(tuple):
    """RETURN_TYPES wrapper: any index at/after the last returns the last element, so a node can
    return more (dynamic) outputs than it literally declares without an IndexError."""
    def __getitem__(self, index):
        if index >= len(self):
            index = len(self) - 1
        return super().__getitem__(index)


def _value_names(prefix="value", n=MAX_SLOTS):
    return [f"{prefix}_{i}" for i in range(n)]


# ----------------------------------------------------------------- For Each (map over a list)
def _to_items(v):
    """Split one input into per-iteration items: an IMAGE batch [B,H,W,C] -> B single frames,
    a list -> itself, anything else -> a one-item list."""
    if torch is not None and isinstance(v, torch.Tensor) and v.ndim == 4:
        return [v[i:i + 1] for i in range(int(v.shape[0]))]
    if isinstance(v, list):
        return list(v)
    return [v]


def _first(v):
    """Unwrap the 1-element list that INPUT_IS_LIST wraps scalars in; pass anything else."""
    return v[0] if isinstance(v, list) and len(v) == 1 else v


class ListEmit:
    """Leaf that fans each input out as a ComfyUI list (OUTPUT_IS_LIST). For Each (Collect)
    expands into this on its final pass to emit the accumulated items as a per-item list — so
    different-sized images travel separately instead of being forced into one batch. Keeping
    OUTPUT_IS_LIST on this plain (non-expansion) leaf avoids tangling it with the loop's
    graph expansion. Also useful standalone: turn a value holding a Python list into a list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {name: (ANY,) for name in _value_names()}}

    RETURN_TYPES = _ByPassTuple(tuple((ANY,) * MAX_SLOTS))
    RETURN_NAMES = _ByPassTuple(tuple(_value_names("item")))
    OUTPUT_IS_LIST = (True,) * MAX_SLOTS
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/loops"

    def run(self, **kwargs):
        out = []
        for i in range(MAX_SLOTS):
            v = kwargs.get(f"value_{i}")
            out.append(list(v) if isinstance(v, (list, tuple)) else ([] if v is None else [v]))
        return tuple(out)


# --------------------------------------------------------------- Repeat (stateful loop)
class RepeatOpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 4, "min": 1, "max": 100000, "tooltip": "How many times the loop body runs. Keep this a value (not an input) — it's read at expansion time."}),
            },
            "optional": {name: (ANY,) for name in _value_names()},
            # Carried index, set by the Close node on each recursion. NOT dunder-named: a
            # leading "__" would trigger Python name-mangling inside the class and never bind.
            "hidden": {"loop_index": (ANY,)},
        }

    RETURN_TYPES = _ByPassTuple((FLOW_TYPE, "INT") + (ANY,) * MAX_SLOTS)
    RETURN_NAMES = _ByPassTuple(("flow", "index") + tuple(_value_names()))
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/loops"

    def run(self, count, loop_index=0, **kwargs):
        # First pass: index 0 and the initial values. On each later pass the Close node clones
        # this node with loop_index/value_* overridden to the next iteration's state.
        index = int(loop_index or 0)
        values = [kwargs.get(f"value_{i}") for i in range(MAX_SLOTS)]
        return tuple(["stub", index] + values)


class _LoopCloseBase:
    """Shared graph-expansion machinery for the loop Close nodes. To continue a loop it clones
    the whole body (every node feeding Close, plus the Open node) into a fresh subgraph wired
    back into a copy of itself, advancing the Open node's carried state to the next iteration.
    Mirrors the proven graph-expansion loop pattern."""

    def _explore_deps(self, node_id, dynprompt, upstream):
        node = dynprompt.get_node(node_id)
        if "inputs" not in node:
            return
        for k, v in node["inputs"].items():
            if is_link(v):
                parent = v[0]
                if parent not in upstream:
                    upstream[parent] = []
                    self._explore_deps(parent, dynprompt, upstream)
                upstream[parent].append(node_id)

    def _collect(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child in upstream[node_id]:
            if child not in contained:
                contained[child] = True
                self._collect(child, upstream, contained)

    def _read_open_int(self, dynprompt, open_id, key, default):
        """Read a literal INT widget value off the Open node; fall back if it's a link/missing."""
        try:
            v = dynprompt.get_node(open_id)["inputs"].get(key, default)
            return int(v) if not (is_link and is_link(v)) else default
        except Exception:
            return default

    def _clone_body(self, open_id, unique_id, dynprompt):
        """Clone the loop body (Open + everything between it and this Close, plus a 'Recurse'
        copy of this Close) into a fresh GraphBuilder subgraph with the original links rewired.
        Returns (graph, cloned_open, cloned_self). Caller sets the next state on cloned_open."""
        upstream = {}
        self._explore_deps(unique_id, dynprompt, upstream)
        contained = {}
        self._collect(open_id, upstream, contained)
        contained[unique_id] = True
        contained[open_id] = True

        graph = GraphBuilder()
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            node = graph.node(original["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original["inputs"].items():
                if is_link(v) and v[0] in contained:
                    node.set_input(k, graph.lookup_node(v[0]).out(v[1]))
                else:
                    node.set_input(k, v)
        return graph, graph.lookup_node(open_id), graph.lookup_node("Recurse")

    def _expand(self, open_id, unique_id, dynprompt, next_index, values):
        graph, new_open, my_clone = self._clone_body(open_id, unique_id, dynprompt)
        new_open.set_input("loop_index", next_index)
        for i in range(MAX_SLOTS):
            new_open.set_input(f"value_{i}", values[i])
        return {"result": tuple(my_clone.out(i) for i in range(MAX_SLOTS)),
                "expand": graph.finalize()}


class RepeatClose(_LoopCloseBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": (FLOW_TYPE, {"rawLink": True}),
                "index": ("INT", {"forceInput": True, "tooltip": "Wire the matching Repeat (Open)'s 'index' output here (the Auto-pair button does this for you)."}),
            },
            "optional": {name: (ANY,) for name in _value_names()},
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = _ByPassTuple(tuple((ANY,) * MAX_SLOTS))
    RETURN_NAMES = _ByPassTuple(tuple(_value_names()))
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/loops"

    def run(self, flow, index, dynprompt=None, unique_id=None, **kwargs):
        open_id = flow[0]
        values = [kwargs.get(f"value_{i}") for i in range(MAX_SLOTS)]
        count = self._read_open_int(dynprompt, open_id, "count", 100000)
        next_index = int(index) + 1
        if next_index >= count:
            return tuple(values)  # ran `count` times — hand the final state downstream
        return self._expand(open_id, unique_id, dynprompt, next_index, values)


class WhileOpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_iterations": ("INT", {"default": 100, "min": 1, "max": 100000, "tooltip": "Safety cap: the loop can't run more than this many times even if the condition stays True. Keep it a value (read when the loop expands)."}),
            },
            "optional": {name: (ANY,) for name in _value_names()},
            "hidden": {"loop_index": (ANY,)},
        }

    RETURN_TYPES = _ByPassTuple((FLOW_TYPE, "INT") + (ANY,) * MAX_SLOTS)
    RETURN_NAMES = _ByPassTuple(("flow", "index") + tuple(_value_names()))
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/loops"

    def run(self, max_iterations, loop_index=0, **kwargs):
        index = int(loop_index or 0)
        values = [kwargs.get(f"value_{i}") for i in range(MAX_SLOTS)]
        return tuple(["stub", index] + values)


class WhileClose(_LoopCloseBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": (FLOW_TYPE, {"rawLink": True}),
                "index": ("INT", {"forceInput": True, "tooltip": "Wire the matching While (Open)'s 'index' output here (the Auto-pair button does this for you)."}),
                "condition": ("BOOLEAN", {"forceInput": True, "tooltip": "Loop while this is True. Compute it in the loop body (e.g. a comparison). The loop also stops at the Open node's max_iterations."}),
            },
            "optional": {name: (ANY,) for name in _value_names()},
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = _ByPassTuple(tuple((ANY,) * MAX_SLOTS))
    RETURN_NAMES = _ByPassTuple(tuple(_value_names()))
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/loops"

    def run(self, flow, index, condition, dynprompt=None, unique_id=None, **kwargs):
        open_id = flow[0]
        values = [kwargs.get(f"value_{i}") for i in range(MAX_SLOTS)]
        max_iter = self._read_open_int(dynprompt, open_id, "max_iterations", 100000)
        next_index = int(index) + 1
        if (not condition) or next_index >= max_iter:
            return tuple(values)  # condition False or safety cap hit — stop
        return self._expand(open_id, unique_id, dynprompt, next_index, values)


# --------------------------------------------------------------- For Each (map + accumulate)
class ForEachOpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {name: (ANY,) for name in _value_names()},
            # INPUT_IS_LIST so a whole batch / ComfyUI-list arrives intact. The carried index
            # and accumulator bundle are set by the Collect node on each recursion.
            "hidden": {"unique_id": "UNIQUE_ID", "loop_index": (ANY,), "accum": (ANY,)},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = _ByPassTuple((FLOW_TYPE, "INT", "INT") + (ANY,) * MAX_SLOTS)
    RETURN_NAMES = _ByPassTuple(("flow", "index", "total") + tuple(_value_names("element")))
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/loops"

    def run(self, unique_id=None, loop_index=0, accum=None, **kwargs):
        uid = _first(unique_id)
        idx = int(_first(loop_index) or 0)
        acc = _first(accum)
        if not isinstance(acc, dict):
            acc = {}
        # Each connected input arrives as a list (INPUT_IS_LIST); flatten batch frames / list
        # items into a flat item list, then emit the idx-th element of each. Iterate to the
        # SHORTEST input (equal lengths — the common case — drop nothing).
        items = {}
        for i in range(MAX_SLOTS):
            v = kwargs.get(f"value_{i}")
            if v is None:
                continue
            flat = []
            for elem in v:
                flat.extend(_to_items(elem))
            items[i] = flat
        total = min((len(x) for x in items.values()), default=0)
        elements = [(items[i][idx] if (i in items and idx < len(items[i])) else None)
                    for i in range(MAX_SLOTS)]
        # `flow` bundles everything Collect needs (the loop handle): one wire, auto-paired.
        flow = {"open_id": uid, "index": idx, "total": total, "accum": acc}
        return tuple([flow, idx, total] + elements)


class ForEachCollect(_LoopCloseBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"flow": (FLOW_TYPE, {"tooltip": "Connect the matching For Each (Open)'s 'flow' output (the Auto-pair button does this)."})},
            "optional": {name: (ANY,) for name in _value_names("result")},
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = _ByPassTuple(tuple((ANY,) * MAX_SLOTS))
    RETURN_NAMES = _ByPassTuple(tuple(_value_names("collected")))
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/loops"

    def run(self, flow, dynprompt=None, unique_id=None, **kwargs):
        bundle = flow if isinstance(flow, dict) else {}
        open_id = bundle.get("open_id")
        idx = int(bundle.get("index", 0))
        total = int(bundle.get("total", 0))
        # Copy the carried lists, then append this iteration's results (don't mutate shared state).
        acc = {int(k): list(v) for k, v in (bundle.get("accum") or {}).items()}
        for i in range(MAX_SLOTS):
            if f"result_{i}" in kwargs:
                acc.setdefault(i, []).append(kwargs[f"result_{i}"])

        if idx + 1 >= total:
            # Last element done. Return the accumulated items as a plain Python list per slot.
            # We must NOT fan them out here (no OUTPUT_IS_LIST): inside graph expansion ComfyUI
            # flattens list outputs across slots, which corrupts the slot count. Feed the output
            # into a `List Output` node (outside the loop) to get a real per-item ComfyUI list —
            # the 🔗 button wires one up for you.
            return tuple(acc.get(i) for i in range(MAX_SLOTS))

        # Otherwise advance: clone the body, push the next index + grown accumulators into the
        # cloned Open, and recurse. The iterables stay wired to their original source.
        graph, new_open, my_clone = self._clone_body(open_id, unique_id, dynprompt)
        new_open.set_input("loop_index", idx + 1)
        new_open.set_input("accum", acc)
        return {"result": tuple(my_clone.out(i) for i in range(MAX_SLOTS)),
                "expand": graph.finalize()}


# --------------------------------------------------------------- Get by Index (universal)
def _resolve_index(i, n, oob):
    """Map a (possibly negative / out-of-range) index into [0, n) per the out-of-range mode."""
    if n <= 0:
        raise IndexError("Get by Index: the container is empty.")
    i = int(i)
    if i < 0:
        i += n
    if 0 <= i < n:
        return i
    if oob == "wrap":
        return i % n
    if oob == "clamp":
        return 0 if i < 0 else n - 1
    raise IndexError(f"Get by Index: index {i} is out of range for length {n}.")


def _index_into(value, i, oob):
    """Return (item, length) for the index-th element of any indexable value.

    Keeps batch dims for media tensors (IMAGE/MASK -> a 1-frame batch) and for LATENT dicts,
    so the item stays a valid ComfyUI type; lists/tuples/strings return a plain element.
    """
    if value is None:
        return None, 0
    # LATENT (and similar) — a dict whose "samples" is a batched tensor.
    if isinstance(value, dict) and torch is not None and isinstance(value.get("samples"), torch.Tensor):
        n = int(value["samples"].shape[0])
        j = _resolve_index(i, n, oob)
        out = dict(value)
        out["samples"] = value["samples"][j:j + 1]
        return out, n
    if torch is not None and isinstance(value, torch.Tensor):
        n = int(value.shape[0])
        j = _resolve_index(i, n, oob)
        return (value[j:j + 1] if value.ndim >= 3 else value[j]), n
    if isinstance(value, (list, tuple, str)):
        n = len(value)
        return value[_resolve_index(i, n, oob)], n
    try:
        n = len(value)
    except TypeError:
        return value, 1  # not indexable (a scalar) — pass through unchanged
    return value[_resolve_index(i, n, oob)], n


class GetByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY, {"forceInput": True, "tooltip": "Anything indexable: an IMAGE/MASK batch, a LATENT batch, a list, a string, etc."}),
                "index": ("INT", {"default": 0, "min": -100000, "max": 100000, "tooltip": "Element to take. Negative counts from the end (-1 = last)."}),
                "out_of_range": (["clamp", "wrap", "error"], {"default": "clamp", "tooltip": "Index past the end: clamp = nearest valid, wrap = modulo (cycle), error = stop the run."}),
            }
        }

    RETURN_TYPES = (ANY, "INT")
    RETURN_NAMES = ("item", "length")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/loops"

    def run(self, value, index, out_of_range="clamp"):
        item, length = _index_into(value, index, out_of_range)
        return (item, length)


# --------------------------------------------------------------- Delay (debug)
class Delay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY, {"forceInput": True, "tooltip": "Passed straight through after the pause — drop this node into any wire."}),
                "seconds": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 600.0, "step": 0.1, "tooltip": "How long to pause before passing the value on. Handy inside a loop body to watch iterations."}),
            },
            "optional": {
                "label": ("STRING", {"default": "", "tooltip": "Optional tag printed to the console on each pause, e.g. to mark which delay fired."}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/loops"

    def run(self, value, seconds=0.5, label=""):
        s = max(0.0, float(seconds))
        tag = f" [{label}]" if label else ""
        print(f"[Delay]{tag} sleeping {s}s")
        if s:
            time.sleep(s)
        return (value,)


NODE_CLASS_MAPPINGS = {
    "KinburgForEachOpen": ForEachOpen,
    "KinburgForEachCollect": ForEachCollect,
    "KinburgRepeatOpen": RepeatOpen,
    "KinburgRepeatClose": RepeatClose,
    "KinburgWhileOpen": WhileOpen,
    "KinburgWhileClose": WhileClose,
    "KinburgGetByIndex": GetByIndex,
    "KinburgDelay": Delay,
    "KinburgListEmit": ListEmit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KinburgForEachOpen": "For Each (Open)",
    "KinburgForEachCollect": "For Each (Collect)",
    "KinburgRepeatOpen": "Repeat (Open)",
    "KinburgRepeatClose": "Repeat (Close)",
    "KinburgWhileOpen": "While (Open)",
    "KinburgWhileClose": "While (Close)",
    "KinburgGetByIndex": "Get by Index",
    "KinburgDelay": "Delay",
    "KinburgListEmit": "List Output",
}
