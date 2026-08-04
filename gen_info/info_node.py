"""Generation Info — inspect the settings of the branch that produced an output.

Pass your LATENT (e.g. the sampler's output) through this node. It reads ComfyUI's hidden
PROMPT (the resolved graph), walks upstream from that link, and lists every upstream node's
widget values — both as a human-readable dump on the node (`info`) and as machine-readable
structured data (`data`, a JSON string, type GEN_INFO) for the Generation Info Filter node.

Each entry carries the node's `class_type`, its 1-based occurrence index `ord` among nodes
of the same class (so a repeated node like PrimitiveString can be addressed as
`PrimitiveString[2]`), and its literal widget `params`.

A walk can only read widget LITERALS, so a node that decides things at run time (Chimera resolving
its step split, sigma boundary and per-stage times) can hand what it knows to the optional `extra`
input: it is merged in front of the walked entries into ONE dump, keeping the branch a single Set
Accumulator (gen info) — i.e. a single image downstream. `extra` is an addition to a dump, never a
dump on its own: a branch that skips this node has none of the shared settings, and the Filter's
`differences` mode then keeps every field the other branches have (absent counts as a difference).
"""
import json
from collections import deque


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


def _parse_info(v):
    """A GEN_INFO value is a JSON string of [{class_type, ord, params}] (or an already-parsed list)."""
    if not v:
        return []
    if isinstance(v, list):
        return [e for e in v if isinstance(e, dict)]
    try:
        d = json.loads(v)
    except Exception:
        return []
    return [e for e in d if isinstance(e, dict)] if isinstance(d, list) else []


def _renumber(entries):
    """Recount the per-class `ord` after merging, so every entry stays uniquely addressable by the
    Generation Info Filter's `ClassType[n].param` selectors."""
    counts, out = {}, []
    for e in entries:
        ct = e.get("class_type", "?")
        counts[ct] = counts.get(ct, 0) + 1
        out.append({"class_type": ct, "ord": counts[ct], "params": e.get("params", {}) or {}})
    return out


class GenerationInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": ("LATENT", {"tooltip": "Pass your LATENT through here — tap it downstream of where the branches converge (e.g. the sampler's LATENT output) so the upstream walk reaches them all. The node lists every upstream node's widget settings."}),
            },
            "optional": {
                "extra": ("GEN_INFO", {"tooltip": "Optional GEN_INFO from a node that reports its OWN runtime facts — e.g. Chimera's 'gen_extra_info', which knows the resolved step split, the sigma boundary and the per-stage times that a graph walk can't see (a walk only reads widget literals). It is an ADDITION, not a whole dump: it gets merged IN FRONT of the walked settings into a single one, so the branch still feeds one Set Accumulator (gen info) and stays one image. Wiring such a node straight into the accumulator instead would give that branch a dump with none of the shared settings — and the Filter's 'differences' mode then keeps every field the other branches have, since present-vs-absent counts as a difference."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "STRING", "GEN_INFO")
    RETURN_NAMES = ("passthrough", "info", "data")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Kinburg-Nodes/util"

    def run(self, passthrough, extra=None, prompt=None, unique_id=None):
        data = _renumber(_parse_info(extra) + self._collect(prompt, unique_id))
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
            return "(no upstream settings — connect 'passthrough' to your LATENT, e.g. the sampler output)"
        lines = []
        for e in data:
            params = ", ".join(f"{k}: {_fmt_value(v)}" for k, v in e["params"].items())
            lines.append(f"[{e['class_type']}] {params}")
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {"GenerationInfo": GenerationInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"GenerationInfo": "Generation Info"}
