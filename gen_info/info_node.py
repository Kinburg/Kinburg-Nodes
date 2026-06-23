"""Generation Info — inspect the settings of the branch that produced an output.

Pass your sampler / latent / image output through this node. It reads ComfyUI's hidden
PROMPT (the resolved graph), walks upstream from that link, and lists every upstream node's
widget values — both as a human-readable dump on the node (`info`) and as machine-readable
structured data (`data`, a JSON string, type GEN_INFO) for the Generation Info Filter node.

Each entry carries the node's `class_type`, its 1-based occurrence index `ord` among nodes
of the same class (so a repeated node like PrimitiveString can be addressed as
`PrimitiveString[2]`), and its literal widget `params`.
"""
import json
from collections import deque


class AnyType(str):
    """ComfyUI wildcard idiom — compares equal to every type."""
    def __ne__(self, other):
        return False


ANY = AnyType("*")


def _is_link(v):
    """In the API prompt a linked input is [source_node_id, output_slot]."""
    return (isinstance(v, list) and len(v) == 2
            and isinstance(v[0], (str, int)) and not isinstance(v[0], bool)
            and isinstance(v[1], int) and not isinstance(v[1], bool))


def _walk_upstream(prompt, starts, limit=1000):
    """Breadth-first walk of every node upstream of (and including) the start ids."""
    order, seen = [], set()
    q = deque(str(s) for s in starts)
    while q and len(seen) < limit:
        nid = str(q.popleft())
        if nid in seen or nid not in prompt:
            continue
        seen.add(nid)
        order.append(nid)
        for v in prompt[nid].get("inputs", {}).values():
            if _is_link(v):
                q.append(str(v[0]))
    return order


def _fmt_value(v, cap=300):
    s = v if isinstance(v, str) else repr(v)
    s = " ".join(s.split())
    return s if len(s) <= cap else s[:cap] + "…"


class GenerationInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": (ANY, {"tooltip": "Pass your sampler / latent / image output through here — tap it downstream of where the branches converge (e.g. the sampler output) so the upstream walk reaches them all. The node lists every upstream node's widget settings."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (ANY, "STRING", "GEN_INFO")
    RETURN_NAMES = ("passthrough", "info", "data")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Kinburg-Nodes/util"

    def run(self, passthrough, prompt=None, unique_id=None):
        data = self._collect(prompt, unique_id)
        text = self._render(data)
        return {"ui": {"kinburg_geninfo": [text]},
                "result": (passthrough, text, json.dumps(data, ensure_ascii=False))}

    def _collect(self, prompt, unique_id):
        """Return [{class_type, ord, params}, ...] for every upstream node with literals."""
        if not isinstance(prompt, dict) or unique_id is None:
            return []
        uid = str(unique_id[0] if isinstance(unique_id, list) else unique_id)
        src = prompt.get(uid, {}).get("inputs", {}).get("passthrough")
        if not _is_link(src):
            return []
        counts, out = {}, []
        for nid in _walk_upstream(prompt, [src[0]]):
            node = prompt.get(nid, {})
            ctype = node.get("class_type", "?")
            literals = {k: v for k, v in node.get("inputs", {}).items() if not _is_link(v)}
            if not literals:
                continue
            counts[ctype] = counts.get(ctype, 0) + 1
            out.append({"class_type": ctype, "ord": counts[ctype], "params": literals})
        return out

    @staticmethod
    def _render(data):
        if not data:
            return "(no upstream settings — connect 'passthrough' to your sampler / latent / image output)"
        lines = []
        for e in data:
            params = ", ".join(f"{k}: {_fmt_value(v)}" for k, v in e["params"].items())
            lines.append(f"[{e['class_type']}] {params}")
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {"GenerationInfo": GenerationInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"GenerationInfo": "Generation Info"}
