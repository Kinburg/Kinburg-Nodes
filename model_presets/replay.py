"""Replay a captured loader subgraph — turn a stored recipe back into live MODEL / CLIP / VAE.

A recipe is a verbatim copy of the slice of prompt graph that built a model (see capture_node.py):
the very same ``{node_id: {class_type, inputs}}`` shape ComfyUI itself queues, where an input is
either a literal widget value or a ``[node_id, output_slot]`` link. Replaying it means running
those nodes in dependency order — which is possible precisely because loader and model-patch nodes
are pure functions of their inputs, the same assumption ComfyUI's own output cache rests on.

Nothing here knows the name of a single node type. Classes are looked up in
``nodes.NODE_CLASS_MAPPINGS`` — the one registry that holds V1 and V3, builtin and third-party
alike (nodes.py:2279 / :2307) — so a model that ships tomorrow with its own custom loader replays
without a code change, as long as it was captured.

Two calling conventions have to be honoured:
  * **V1** — instantiate, then ``getattr(obj, cls.FUNCTION)(**inputs)`` → tuple
  * **V3** — ``cls.execute(**inputs)`` is a classmethod returning ``io.NodeOutput``, values in
    ``.result`` (comfy_api/latest/_io.py)
Either may be a coroutine (ComfyUI supports async node functions, execution.py:290), so replay is
async and the calling node's FUNCTION must be ``async def``.

Nodes that need the executor wrapped around them — hidden inputs, list in/out, subgraph expansion,
output nodes — are refused by :func:`unsupported_reason`. That check runs at CAPTURE time, so an
unreplayable bundle is never stored in the first place and the user hears which node and why while
the graph that produced it is still on screen.

**Caching.** Results are memoised by a merkle key over (class_type, literals, upstream keys), and
the key is derived from the recipe alone — so a cache hit skips the node's whole upstream subtree.
That is what makes switching *presets* cheap: the patch nodes' keys change and re-run, the loader's
key does not, so 20 GB of weights are not re-read from disk. Switching *models* misses, as it must.
``purge_others`` (wired to Model Select's ``unload_others``) drops everything the current recipe
doesn't need, for the one-model-resident-at-a-time discipline the rest of this pack follows.
"""
import collections
import hashlib
import inspect
import json

# Merkle key -> output tuple. Bounded so a long session switching models can't stack checkpoints in
# RAM forever; `purge_others` is the explicit form of the same idea.
_CACHE = collections.OrderedDict()
_CACHE_MAX = 24


def _mappings():
    import nodes
    return nodes.NODE_CLASS_MAPPINGS


def _is_v3(cls):
    try:
        from comfy_api.internal import _ComfyNodeInternal
    except Exception:
        return False
    return isinstance(cls, type) and issubclass(cls, _ComfyNodeInternal)


def is_link(v):
    """ComfyUI's prompt link encoding: ``[node_id, output_slot]``. Mirrors
    comfy_execution.graph_utils.is_link, kept local so this module has no import coupling."""
    return isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], int) \
        and isinstance(v[0], (str, int)) and not isinstance(v[0], bool)


def input_spec(cls):
    """``({name: (type, opts)}, {hidden names})`` over required + optional inputs.

    V3 classes carry an ``INPUT_TYPES()`` compatibility shim, so one path serves both conventions;
    the V3 schema is consulted separately in :func:`unsupported_reason` for what the shim drops.
    """
    try:
        it = cls.INPUT_TYPES() or {}
    except Exception:
        return {}, set()
    out = {}
    for cat in ("required", "optional"):
        for name, spec in (it.get(cat) or {}).items():
            if isinstance(spec, (list, tuple)) and spec:
                t = spec[0]
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            else:
                t, opts = spec, {}
            out[name] = (t, opts)
    return out, set((it.get("hidden") or {}).keys())


def unsupported_reason(class_type):
    """Why this node type can't be replayed outside the executor — or ``None`` if it can.

    Deliberately strict: anything that reads execution context or doesn't map to a single
    inputs→outputs call is rejected by name, because the alternative is a bundle that misbehaves
    quietly at generation time. The escape hatch for a genuine rejection is Model Select's
    ``model`` / ``clip`` / ``vae`` override inputs — that assembly simply stays in the graph.
    """
    cls = _mappings().get(class_type)
    if cls is None:
        return f"node type '{class_type}' is not installed"
    if getattr(cls, "OUTPUT_NODE", False):
        return f"'{class_type}' is an output node"
    if getattr(cls, "INPUT_IS_LIST", False):
        return f"'{class_type}' takes its inputs as lists (INPUT_IS_LIST)"
    if any(getattr(cls, "OUTPUT_IS_LIST", None) or ()):
        return f"'{class_type}' returns lists (OUTPUT_IS_LIST)"
    _, hidden = input_spec(cls)
    if hidden:
        return f"'{class_type}' needs execution context ({', '.join(sorted(hidden))})"
    if _is_v3(cls):
        try:
            schema = cls.define_schema()
        except Exception as e:
            return f"'{class_type}' schema unavailable ({e})"
        if getattr(schema, "hidden", None):
            names = ", ".join(str(getattr(h, "name", h)) for h in schema.hidden)
            return f"'{class_type}' needs execution context ({names})"
        if getattr(schema, "is_input_list", False):
            return f"'{class_type}' takes its inputs as lists"
        if getattr(schema, "is_output_node", False):
            return f"'{class_type}' is an output node"
        if getattr(schema, "enable_expand", False):
            return f"'{class_type}' expands into a subgraph"
        if getattr(schema, "has_intermediate_output", False):
            return f"'{class_type}' streams intermediate outputs"
        if any(getattr(o, "is_output_list", False) for o in (getattr(schema, "outputs", None) or ())):
            return f"'{class_type}' returns lists"
        if getattr(cls, "execute", None) is None:
            return f"'{class_type}' has no execute()"
    elif not getattr(cls, "FUNCTION", None):
        return f"'{class_type}' has no FUNCTION"
    return None


def _cacheable(cls):
    """V3 marks nodes whose result must not be reused; honour it (ComfyUI does the same)."""
    if _is_v3(cls):
        try:
            if getattr(cls.define_schema(), "not_idempotent", False):
                return False
        except Exception:
            return False
    return True


async def _call(class_type, inputs):
    """One node, one call — the V1/V3 fork. Returns a plain tuple of its outputs."""
    cls = _mappings()[class_type]
    if _is_v3(cls):
        out = cls.execute(**inputs)
        if inspect.isawaitable(out):
            out = await out
        result = getattr(out, "result", None)
        if result is None:
            result = out if isinstance(out, (list, tuple)) else (out,)
        return tuple(result)
    out = getattr(cls(), cls.FUNCTION)(**inputs)
    if inspect.isawaitable(out):
        out = await out
    if isinstance(out, dict):        # {"ui": …, "result": …} — the classic UI-returning form
        out = out.get("result") or ()
    return tuple(out) if isinstance(out, (list, tuple)) else (out,)


def _digest(payload):
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def recipe_keys(graph):
    """Merkle key per node id, from the recipe alone (no execution).

    Because a key folds in the keys of everything upstream, a hit on one node means its whole
    subtree can be skipped — the property that keeps a preset change from re-reading weights.
    Raises on a cycle or a dangling link so a corrupt recipe fails loudly, not halfway through
    loading a model.
    """
    keys, visiting = {}, set()

    def key(nid):
        nid = str(nid)
        if nid in keys:
            return keys[nid]
        if nid in visiting:
            raise ValueError(f"recipe has a cycle at node {nid}")
        node = graph.get(nid)
        if not isinstance(node, dict):
            raise ValueError(f"recipe refers to missing node {nid}")
        visiting.add(nid)
        parts = {}
        for name, v in sorted((node.get("inputs") or {}).items()):
            parts[name] = ["link", key(v[0]), int(v[1])] if is_link(v) else ["value", v]
        visiting.discard(nid)
        keys[nid] = _digest([node.get("class_type"), parts])
        return keys[nid]

    for nid in graph:
        key(nid)
    return keys


def canonicalize(graph, outputs):
    """Renumber a captured subgraph to ``1..N`` in content order.

    Node ids in a prompt are whatever the canvas happened to assign, so capturing the *same*
    assembly twice would otherwise produce two recipes that differ only in their keys — different
    merkle keys, a pointless reload, and a spurious "the bundle changed". Ordering by merkle key
    makes the recipe a pure function of what the assembly *is*.
    """
    keys = recipe_keys(graph)
    order = sorted(graph, key=lambda nid: (keys[nid], str(nid)))
    remap = {nid: str(i + 1) for i, nid in enumerate(order)}
    new_graph = {}
    for nid in order:
        node = graph[nid]
        inputs = {}
        for name, v in (node.get("inputs") or {}).items():
            inputs[name] = [remap[str(v[0])], int(v[1])] if is_link(v) else v
        new_graph[remap[nid]] = {"class_type": node.get("class_type"), "inputs": inputs}
    new_outputs = {slot: [remap[str(ref[0])], int(ref[1])]
                   for slot, ref in (outputs or {}).items() if is_link(ref)}
    return new_graph, new_outputs


def check(recipe):
    """Every reason this recipe can't be replayed (empty list = good). Used by Capture before it
    saves and by Model Select before it loads, so the same verdict is reached in both places."""
    problems = []
    graph = (recipe or {}).get("nodes") or {}
    outputs = (recipe or {}).get("outputs") or {}
    if not graph:
        return ["recipe has no nodes"]
    for nid in sorted(graph):
        node = graph[nid]
        if not isinstance(node, dict) or not node.get("class_type"):
            problems.append(f"node {nid} has no class_type")
            continue
        why = unsupported_reason(node["class_type"])
        if why:
            problems.append(why)
        for name, v in (node.get("inputs") or {}).items():
            if is_link(v) and str(v[0]) not in graph:
                problems.append(f"node {nid} input '{name}' links to missing node {v[0]}")
    for slot, ref in outputs.items():
        if not is_link(ref):
            problems.append(f"output '{slot}' is not a link")
        elif str(ref[0]) not in graph:
            problems.append(f"output '{slot}' links to missing node {ref[0]}")
    try:
        recipe_keys(graph)
    except ValueError as e:
        problems.append(str(e))
    return problems


def purge(keep=()):
    """Drop cached results, keeping the given merkle keys. Releases the previous model's weights so
    only what the current bundle needs stays referenced."""
    keep = set(keep)
    for k in [k for k in _CACHE if k not in keep]:
        _CACHE.pop(k, None)


def cache_size():
    return len(_CACHE)


async def replay(recipe, purge_others=False):
    """Build a recipe's outputs. Returns ``({slot: object}, notes)``.

    ``notes`` records what actually happened per node (cached / built), so Model Select's ``info``
    output can show whether a run touched the disk at all.
    """
    graph = (recipe or {}).get("nodes") or {}
    outputs = (recipe or {}).get("outputs") or {}
    problems = check(recipe)
    if problems:
        raise RuntimeError("[Model Library] recipe can't be replayed:\n  - " + "\n  - ".join(problems))

    keys = recipe_keys(graph)
    built, notes = {}, []

    async def resolve(nid):
        nid = str(nid)
        if nid in built:
            return built[nid]
        k = keys[nid]
        if k in _CACHE:                      # whole subtree skipped — see the docstring
            _CACHE.move_to_end(k)
            built[nid] = _CACHE[k]
            notes.append(f"{graph[nid]['class_type']} (cached)")
            return built[nid]
        inputs = {}
        for name, v in (graph[nid].get("inputs") or {}).items():
            if is_link(v):
                up = await resolve(v[0])
                slot = int(v[1])
                if slot >= len(up):
                    raise RuntimeError(f"[Model Library] node {v[0]} has no output {slot}")
                inputs[name] = up[slot]
            else:
                inputs[name] = v
        class_type = graph[nid]["class_type"]
        out = await _call(class_type, inputs)
        if _cacheable(_mappings()[class_type]):
            _CACHE[k] = out
            _CACHE.move_to_end(k)
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
        built[nid] = out
        notes.append(f"{class_type} (built)")
        return out

    result = {}
    for slot, ref in outputs.items():
        up = await resolve(ref[0])
        result[slot] = up[int(ref[1])]
    if purge_others:
        purge(keep=set(keys.values()))
    return result, notes
