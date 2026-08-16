"""Model Capture — register a model assembly in the library by reading it out of the graph.

Wire the outputs of a loader stack you already have working — however many nodes it takes, whatever
they are — give it an id, and this node walks its own inputs upstream through the queued prompt and
stores that slice of graph as the model's **recipe** (see replay.py). Then the loaders can be
deleted from the workflow: Model Select rebuilds them from the library.

Two properties worth knowing:

  * **It loads nothing.** The inputs are declared ``lazy`` and :meth:`check_lazy_status` asks for
    none of them, so ComfyUI never executes the upstream loaders (comfy_execution/graph.py:159 —
    a lazy input is not turned into a strong link). The recipe comes from the hidden ``prompt``,
    which carries the whole queued graph (execution.py:210), and a graph description is all that's
    needed. Capturing a 40 GB assembly costs nothing and touches no VRAM.
  * **It refuses what it can't replay**, by name and reason, while the graph that produced it is
    still in front of you. The escape hatch for a genuine rejection is Model Select's own
    ``model`` / ``clip`` / ``vae`` inputs: that one assembly stays in the graph, everything else
    lives in the library.

``mode`` defaults to *preview* — the first run reports what it found and writes nothing, so a
mis-wired capture can't quietly overwrite a good bundle.
"""
from . import replay, store
from ..categories import CAT_MODEL

MODES = ["preview", "register"]

# The bundle's output slots, in report order. `model_negative` exists for checkpoints shipped as a
# model + unconditional-model pair (Ideogram); Chimera takes it on its own `model_negative` input.
SLOTS = ["model", "clip", "vae", "model_negative"]


def collect(prompt, unique_id, slot_links):
    """Walk upstream from this node's links and copy that slice of the prompt verbatim.

    Returns ``(graph, outputs, problems)``. Inputs are copied as they are — a literal stays a
    literal, a ``[node_id, slot]`` link stays a link — which is exactly what makes the recipe
    replayable without interpreting anything.
    """
    graph, problems = {}, []
    pending = [str(ref[0]) for ref in slot_links.values()]
    while pending:
        nid = pending.pop()
        if nid in graph:
            continue
        node = prompt.get(nid)
        if not isinstance(node, dict) or not node.get("class_type"):
            problems.append(f"node {nid} is not in the queued prompt")
            continue
        if nid == str(unique_id):
            problems.append("the capture node feeds itself")
            continue
        inputs = dict(node.get("inputs") or {})
        graph[nid] = {"class_type": node["class_type"], "inputs": inputs}
        for v in inputs.values():
            if replay.is_link(v):
                pending.append(str(v[0]))
    outputs = {slot: [str(ref[0]), int(ref[1])] for slot, ref in slot_links.items()}
    return graph, outputs, problems


def describe(graph, outputs):
    """A human-readable rendering of a recipe — what Capture reports and Manage shows.

    Prints the assembly per slot, following links back so the shape is visible at a glance rather
    than as a wall of JSON.
    """
    lines = []
    for slot in SLOTS:
        ref = (outputs or {}).get(slot)
        if not ref:
            continue
        chain, seen, nid = [], set(), str(ref[0])
        while nid and nid not in seen:
            seen.add(nid)
            node = graph.get(nid)
            if not isinstance(node, dict):
                break
            widgets = {k: v for k, v in (node.get("inputs") or {}).items()
                       if not replay.is_link(v)}
            detail = ", ".join(f"{k}={v}" for k, v in sorted(widgets.items()))
            chain.append(f"{node.get('class_type')}" + (f" ({detail})" if detail else ""))
            ups = [str(v[0]) for v in (node.get("inputs") or {}).values() if replay.is_link(v)]
            nid = ups[0] if len(ups) == 1 else None
            if len(ups) > 1:                     # a join — stop walking, the node list covers it
                chain.append(f"… + {len(ups)} inputs")
        lines.append(f"  {slot}: " + " ← ".join(chain))
    return "\n".join(lines)


class ModelCapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "", "tooltip": "Stable id for this model — the label in Model Select's dropdown and the key its presets hang off. Choose it once; renaming later is done from the Manage dialog so the presets follow."}),
                "mode": (MODES, {"default": "preview", "tooltip": "'preview' reports what it captured and writes NOTHING (run this first). 'register' saves the bundle to the library, replacing this id's recipe while keeping its presets, families and notes."}),
            },
            "optional": {
                # Lazy: check_lazy_status asks for none of these, so wiring a 40 GB assembly in
                # costs no load. The links are read from the hidden `prompt`, not from the objects.
                "model": ("MODEL", {"lazy": True, "tooltip": "The finished MODEL of this assembly — after every patch (shift, CFG override, …). Wire the last node of the chain."}),
                "clip": ("CLIP", {"lazy": True, "tooltip": "The CLIP for this model. For an old checkpoint that carries its own, wire it from the same loader — Capture notices they share a node and Model Select will make one call for all three."}),
                "vae": ("VAE", {"lazy": True, "tooltip": "The VAE for this model (from its own loader, or from the checkpoint)."}),
                "model_negative": ("MODEL", {"lazy": True, "tooltip": "Optional second MODEL for the unconditional pass — for checkpoints shipped as a pair (Ideogram). Comes out of Model Select on its own output, ready for Chimera's 'model_negative'."}),
                "families": ("STRING", {"default": "", "tooltip": "Comma-separated families this model belongs to, e.g. 'flow-1024'. Presets saved as SHARED for a family show up for every model in it — that's how one 'good 12-step recipe' serves five models without being copied five times.\n\nUse the '＋ family' picker below to add one that already exists: a typo here doesn't error, it just quietly stops shared presets from showing up."}),
                "notes": ("STRING", {"multiline": True, "default": "", "tooltip": "Free notes shown in Model Select's info — what this model likes: cfg range, scheduler, quirks."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT_MODEL
    DESCRIPTION = ("Register a model assembly (loaders + patches, however exotic) in the Model "
                   "Library by reading it out of the graph — then delete those loaders and use "
                   "Model Select instead. Knows no node types: it copies the subgraph and replays "
                   "it later through ComfyUI's own registry, so a model that ships tomorrow with "
                   "its own custom loader works with no code change. Costs no VRAM — the inputs "
                   "are lazy, so nothing upstream is actually loaded.")

    @classmethod
    def check_lazy_status(cls, **kwargs):
        # Nothing is ever needed: the recipe is read from the prompt, not from loaded objects.
        return []

    def run(self, model_id="", mode="preview", families="", notes="",
            prompt=None, unique_id=None, **kwargs):
        model_id = (model_id or "").strip()
        prompt = prompt or {}
        me = (prompt.get(str(unique_id)) or {}).get("inputs") or {}
        slot_links = {slot: me[slot] for slot in SLOTS
                      if slot in me and replay.is_link(me[slot])}

        head = f"Model Capture — id '{model_id or '(unset)'}' · {mode}"
        if not slot_links:
            return self._out(f"{head}\n  ⚠ nothing wired — connect at least 'model'.")
        if "model" not in slot_links:
            return self._out(f"{head}\n  ⚠ 'model' is not wired; a bundle without a MODEL is "
                             f"not usable.")

        graph, outputs, problems = collect(prompt, unique_id, slot_links)
        if not problems:
            try:
                graph, outputs = replay.canonicalize(graph, outputs)
            except ValueError as e:
                problems.append(str(e))
        recipe = {"nodes": graph, "outputs": outputs}
        problems += replay.check(recipe)

        lines = [head,
                 f"  {len(graph)} node(s), slots: {', '.join(sorted(outputs)) or 'none'}",
                 describe(graph, outputs)]
        if problems:
            lines.append("  ✕ NOT registered — this assembly can't be rebuilt outside the graph:")
            lines += [f"      - {p}" for p in dict.fromkeys(problems)]
            lines.append("      Keep these loaders in the workflow and wire them into Model "
                         "Select's own model / clip / vae inputs instead.")
            return self._out("\n".join(lines))

        if mode != "register":
            lines.append("  (preview — nothing written. Set mode to 'register' to save.)")
            return self._out("\n".join(lines))
        if not model_id:
            lines.append("  ✕ NOT registered — 'model_id' is empty.")
            return self._out("\n".join(lines))

        existed = store.get_model(model_id) is not None
        # A brand-new family is legitimate, but it's also what a typo looks like — and a typo here
        # fails silently (shared presets just never show up), so it's called out before it bites.
        fam_notes = store.family_warnings(families, exclude_model=model_id)
        store.upsert_model(model_id, recipe=recipe,
                           families=families if families.strip() else None,
                           notes=notes if notes.strip() else None)
        kept = store.get_model(model_id) or {}
        lines.append(f"  ✔ {'updated' if existed else 'registered'} as '{model_id}'"
                     + (f" · {len(kept.get('presets') or {})} preset(s) kept" if existed else ""))
        lines += [f"  ⚠ {w}" for w in fam_notes]
        return self._out("\n".join(lines))

    def _out(self, text):
        print("[Model Capture] " + text.replace("\n", "\n[Model Capture] "))
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {"KinburgModelCapture": ModelCapture}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgModelCapture": "Model Capture 📥"}
