"""Generic list editing — insert / remove items in a ComfyUI list, by position.

These operate on ComfyUI *lists* (``INPUT_IS_LIST`` / ``OUTPUT_IS_LIST``) of ANY type, so they
work for images, strings, latents, ints — anything. An item is one list element; a single
value connected to a list input counts as a one-item list.

* **List Insert** inserts the item(s) on the `item` input into `list` at a chosen spot
  (start / end / at index / after index). Feed `item` a multi-item list to insert several.
* **List Remove** drops `count` item(s) starting at `index` (negative counts from the end) and
  also returns the removed items.

For frame-level editing of a same-size IMAGE batch tensor, use Image Batch Insert / Remove.
"""


from ..util.anytype import ANY

_POS = ["at end", "at start", "at index", "after index"]


def _as_list(v):
    """A value coming through INPUT_IS_LIST -> a plain Python list ([] when unconnected)."""
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def _first(v, default):
    """First element of an INPUT_IS_LIST-wrapped scalar (index/count/position), or default."""
    if isinstance(v, list):
        return v[0] if v else default
    return default if v is None else v


class ListInsert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "position": (_POS, {"default": "at end", "tooltip": "Where to insert. 'at index' inserts before the item at 'index'; 'after index' inserts after it. 'index' is ignored for start/end."}),
                "index": ("INT", {"default": 0, "min": -100000, "max": 100000, "tooltip": "0-based target position for 'at index' / 'after index'. Negative counts from the end (-1 = last)."}),
            },
            "optional": {
                "list": (ANY, {"tooltip": "The list to insert into. Leave empty to start a new list from 'item'."}),
                "item": (ANY, {"tooltip": "Item to insert. Connect a multi-item list to insert several at once."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = (ANY, "INT")
    RETURN_NAMES = ("list", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/list"

    def run(self, **kw):
        base = _as_list(kw.get("list"))
        items = _as_list(kw.get("item"))
        position = _first(kw.get("position"), "at end")
        index = int(_first(kw.get("index"), 0))

        n = len(base)
        if position == "at start":
            pos = 0
        elif position == "at end":
            pos = n
        else:
            i = index if index >= 0 else n + index
            if position == "after index":
                i += 1
            pos = max(0, min(n, i))

        out = base[:pos] + items + base[pos:]
        return (out, len(out))


class ListRemove:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": -100000, "max": 100000, "tooltip": "0-based item to start removing at. Negative counts from the end (-1 = last item)."}),
                "count": ("INT", {"default": 1, "min": 0, "max": 100000, "tooltip": "How many items to remove, starting at 'index'."}),
            },
            "optional": {
                "list": (ANY, {"tooltip": "The list to remove item(s) from."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = (ANY, ANY, "INT")
    RETURN_NAMES = ("list", "removed", "count")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/list"

    def run(self, **kw):
        base = _as_list(kw.get("list"))
        index = int(_first(kw.get("index"), 0))
        count = int(_first(kw.get("count"), 1))

        n = len(base)
        start = index if index >= 0 else n + index
        start = max(0, min(n, start))
        end = max(start, min(n, start + max(0, count)))

        removed = base[start:end]
        remaining = base[:start] + base[end:]
        return (remaining, removed, len(remaining))


NODE_CLASS_MAPPINGS = {
    "ListInsert": ListInsert,
    "ListRemove": ListRemove,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ListInsert": "List Insert",
    "ListRemove": "List Remove",
}
