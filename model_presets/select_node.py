"""Model Select — pick a model and one of ITS presets from two dropdowns; get everything wired.

This is the node the whole package exists for. It replaces, in a workflow:

  * the checkpoint / UNET loader, the CLIP loader(s), the VAE loader and any model patches
    (``ModelSamplingAuraFlow``, ``CFGOverride``, …) — rebuilt from the library's recipe
    (see replay.py), so exactly ONE model is ever loaded no matter how many are registered;
  * the pile of ``Sampler Settings`` nodes kept around "for the other model", plus whatever
    switch logic chose between them — the settings live under the model they belong to, so the
    second dropdown can only offer settings that are valid for the first one's pick.

The ``sampler_settings`` output is a ``SAMPLER_CFG`` chain, which means a saved multi-stage preset
drops straight into Chimera's ``stage_a`` (it flattens chains into stages) or into Ouroboros.

An assembly the library can't rebuild — Model Capture refuses those loudly — stays in the graph and
is wired into ``model_override`` / ``clip_override`` / ``vae_override``, which win over the library.
So there is no assembly this node can't serve, only ones it can't store.
"""
import json

from ..ouroboros.nodes import SAMPLER_CFG
from . import replay, store
from ..categories import CAT_MODEL


def _seeded(stages, seed_override):
    """A copy of the chain with the seed forced, when asked. Storing a seed in a preset is right
    (it records what was measured) but reusing it forever is not, so -1 keeps the preset's own."""
    if int(seed_override) < 0:
        return [dict(s) for s in stages]
    out = []
    for s in stages:
        s = dict(s)
        s["seed"] = int(seed_override)
        out.append(s)
    return out


def _stage_warnings(stages):
    """Samplers / schedulers a preset names that this install doesn't have — a preset saved on
    another machine, or after a node pack was removed. Reported, never raised: the run should tell
    you what's wrong, not die at stage two."""
    try:
        import comfy.samplers
        samplers = set(comfy.samplers.KSampler.SAMPLERS)
        schedulers = set(comfy.samplers.KSampler.SCHEDULERS)
    except Exception:
        return []
    out = []
    for i, s in enumerate(stages):
        if s.get("sampler_name") and s["sampler_name"] not in samplers:
            out.append(f"stage {i + 1}: sampler '{s['sampler_name']}' is not installed")
        if s.get("scheduler") and s["scheduler"] not in schedulers:
            out.append(f"stage {i + 1}: scheduler '{s['scheduler']}' is not installed")
    return out


def _stage_line(i, s):
    bits = [f"{s.get('sampler_name', '?')}/{s.get('scheduler', '?')}",
            f"{s.get('steps', '?')} steps", f"cfg {s.get('cfg', '?')}"]
    if float(s.get("denoise", 1.0) or 1.0) < 0.9999:
        bits.append(f"denoise {s['denoise']}")
    if float(s.get("eta", 0) or 0):
        bits.append(f"eta {s['eta']}")
    bits.append(f"seed {s.get('seed', 0)}")
    return f"  stage {chr(ord('A') + i)}: " + " · ".join(str(b) for b in bits)


class ModelSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (store.model_names(), {"default": store.NONE, "tooltip": "A model registered in the library. Build the library with Model Capture: wire a working loader stack into it once, then delete those loaders from the workflow."}),
                "preset": (cls._all_preset_names(), {"default": store.NONE, "tooltip": "Saved sampler settings for THIS model — the list narrows to what's valid for the chosen model (its own presets plus any shared with its families). '🚫 None' loads the model but emits no settings. Save presets with Settings Save."}),
                "seed_override": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "-1 = use the seed stored in the preset. Anything else replaces the seed on every stage — a preset records the seed it was measured with, which is worth keeping but not worth being stuck with."}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8, "tooltip": "Fallback size, used only when the chosen preset has no size of its own. A preset saved with a latent wired into Settings Save carries the real one and wins."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8, "tooltip": "Fallback height — see 'width'."}),
                "unload_others": ("BOOLEAN", {"default": True, "tooltip": "Free every other model first (ComfyUI's resident models plus the library's own cache of previously built bundles), so one model is resident at a time. Turn off on a big GPU to keep switching between models fast at the cost of RAM."}),
            },
            "optional": {
                "model_override": ("MODEL", {"tooltip": "Escape hatch: a MODEL wired here wins over the library. For an assembly Model Capture refused (a merge, a LoRA fork, a node that needs execution context) — keep those loaders in the graph and wire them in; presets keep working."}),
                "clip_override": ("CLIP", {"tooltip": "As 'model_override', for CLIP."}),
                "vae_override": ("VAE", {"tooltip": "As 'model_override', for VAE."}),
            },
        }

    @classmethod
    def _all_preset_names(cls):
        """Every preset name in the library — the frontend narrows this per model. The server can't
        do the narrowing itself: a combo's values are baked when /object_info is served, long before
        anyone picks a model."""
        names = set()
        data = store.full_data()
        for m in data.get("models", {}).values():
            names.update((m.get("presets") or {}).keys())
        names.update((data.get("shared") or {}).keys())
        return [store.NONE] + sorted(names)

    # Live library: presets and models saved without an /object_info reload must still validate.
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, model=None, preset=None, seed_override=-1, width=0, height=0, **kwargs):
        """Edit a recipe or a preset in the library and the dropdown values don't change — so
        without this ComfyUI would hand back the previous run's model and settings. Hash what was
        actually resolved, not what was picked."""
        try:
            _, p, recipe = store.resolve(model, preset)
            return json.dumps([recipe, p, int(seed_override), int(width), int(height)],
                              sort_keys=True, default=str)
        except Exception as e:
            return f"error:{e}"

    # `model_id` last so existing workflows' output indices don't move. It feeds Settings Select,
    # which then narrows its own preset list to this model instead of asking for it twice.
    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "VAE", SAMPLER_CFG, "INT", "INT", "STRING", "GEN_INFO",
                    "STRING")
    RETURN_NAMES = ("model", "model_negative", "clip", "vae", "sampler_settings", "width", "height",
                    "info", "gen_extra_info", "model_id")
    FUNCTION = "run"
    CATEGORY = CAT_MODEL
    DESCRIPTION = ("Pick a model and one of its saved sampler presets from two dropdowns — the "
                   "second lists only what's valid for the first. Rebuilds the model's whole "
                   "assembly (loaders + patches) from the library, so the workflow holds one node "
                   "instead of a loader stack and a pile of Sampler Settings. 'sampler_settings' "
                   "goes straight into Chimera's stage_a or Ouroboros; 'model_negative' carries an "
                   "unconditional-pass model when the bundle has one.")

    async def run(self, model=None, preset=None, seed_override=-1, width=1024, height=1024,
                  unload_others=True, model_override=None, clip_override=None, vae_override=None):
        lines, warnings = [], []
        rec, pre, recipe = store.resolve(model, preset)
        pname = preset if preset and preset != store.NONE else None

        if rec is None and model and model != store.NONE:
            warnings.append(f"model '{model}' is not in the library")
        if pname and pre is None:
            warnings.append(f"preset '{pname}' is not available for '{model}' — no settings emitted")

        # ---- the model itself: overrides win, otherwise replay the recipe
        built, notes = {}, []
        # Skip the rebuild only when every slot the bundle offers is already wired in — including
        # `model_negative`, which has no override input of its own and would otherwise be dropped
        # from an Ideogram-style bundle the moment all three visible overrides were connected.
        covered = (model_override is not None and clip_override is not None
                   and vae_override is not None
                   and "model_negative" not in (recipe or {}).get("outputs", {}))
        need_replay = bool(recipe and (recipe.get("nodes") or {}) and not covered)
        # Freeing before we load is the point; freeing when we're NOT about to load would just evict
        # the models wired into the overrides, which were built upstream in this very run.
        if unload_others and need_replay:
            self._free()
        if need_replay:
            built, notes = await replay.replay(recipe, purge_others=unload_others)
        elif not recipe and model_override is None:
            warnings.append("nothing to load — pick a registered model or wire an override")

        out_model = model_override if model_override is not None else built.get("model")
        out_clip = clip_override if clip_override is not None else built.get("clip")
        out_vae = vae_override if vae_override is not None else built.get("vae")
        out_neg = built.get("model_negative")

        # ---- the settings
        stages = _seeded((pre or {}).get("stages") or [], seed_override)
        warnings += _stage_warnings(stages)
        w = int((pre or {}).get("width") or 0) or int(width)
        h = int((pre or {}).get("height") or 0) or int(height)

        # ---- report
        head = f"Model Select — {model or '(none)'} · preset {pname or 'none'}"
        lines.append(head)
        if rec:
            fams = ", ".join(rec.get("families") or []) or "—"
            lines.append(f"  bundle: {len(recipe.get('nodes') or {})} node(s), slots "
                         f"{', '.join(sorted(recipe.get('outputs') or {})) or 'none'} · "
                         f"families: {fams}")
        if notes:
            n_built = sum(1 for n in notes if "(built)" in n)
            lines.append(f"  rebuilt: {', '.join(notes)}"
                         if len(notes) <= 6 else
                         f"  rebuilt: {len(notes)} node(s), {n_built} run, "
                         f"{len(notes) - n_built} from cache")
        for slot, val, src in (("model", out_model, model_override), ("clip", out_clip, clip_override),
                               ("vae", out_vae, vae_override)):
            if src is not None:
                lines.append(f"  {slot}: from {slot}_override (library ignored)")
            elif val is None:
                warnings.append(f"no {slot.upper()} — the bundle has no '{slot}' slot and nothing "
                                f"is wired into '{slot}_override'")
        if out_neg is not None:
            lines.append("  model_negative: present (wire it into Chimera's 'model_negative')")
        if pre:
            # store.resolve already folded the overrides into `recipe`; re-checking which class
            # names actually exist in the bundle is what turns a stale override into a warning
            # instead of a silently ignored line.
            if pre.get("overrides"):
                hit, missed = [], []
                for key in pre["overrides"]:
                    cls_name = str(key).rpartition(".")[0]
                    (hit if any(n.get("class_type") == cls_name
                                for n in (recipe.get("nodes") or {}).values())
                     else missed).append(str(key))
                if hit:
                    lines.append("  preset overrides: " + ", ".join(
                        f"{k}={pre['overrides'][k]}" for k in hit))
                for k in missed:
                    warnings.append(f"preset override '{k}' matches no node in this bundle")
            lines.append(f"  settings: {len(stages)} stage(s) · {w}×{h}")
            for i, s in enumerate(stages):
                lines.append(_stage_line(i, s))
            if int(seed_override) >= 0:
                lines.append(f"  seed forced to {int(seed_override)} on every stage")
            if pre.get("score") is not None or pre.get("seconds") is not None:
                measured = []
                if pre.get("score") is not None:
                    measured.append(f"score {pre['score']:.2f}")
                if pre.get("seconds") is not None:
                    measured.append(f"{pre['seconds']:.1f}s")
                lines.append("  measured: " + " · ".join(measured))
            if pre.get("notes"):
                lines.append(f"  preset notes: {pre['notes']}")
        else:
            lines.append(f"  settings: none · {w}×{h}")
        if (rec or {}).get("notes"):
            lines.append(f"  model notes: {rec['notes']}")
        for wn in warnings:
            lines.append(f"  ⚠ {wn}")
        info = "\n".join(lines)
        print("[Model Select] " + info.replace("\n", "\n[Model Select] "))

        # `gen_extra_info` adds the CHOICE to a Generation Info dump — a graph walk over widget
        # literals can't see which model a library id resolved to. Separate fields so the
        # Generation Info Filter's 'differences' mode can show exactly what changed between runs.
        params = {"model": model or "", "preset": pname or "none", "size": f"{w}×{h}"}
        if (rec or {}).get("families"):
            params["families"] = ", ".join(rec["families"])
        if pre and pre.get("score") is not None:
            params["preset_score"] = pre["score"]
        gen_extra = json.dumps([{"class_type": "Model Select", "ord": 1, "params": params}],
                               ensure_ascii=False)
        return (out_model, out_neg, out_clip, out_vae, stages, w, h, info, gen_extra,
                model if model and model != store.NONE else "")

    def _free(self):
        """Same discipline as the LLM nodes in this pack: one heavy thing resident at a time."""
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as e:
            print(f"[Model Select] could not free models: {e}")


NODE_CLASS_MAPPINGS = {"KinburgModelSelect": ModelSelect}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgModelSelect": "Model Select 🎛"}
