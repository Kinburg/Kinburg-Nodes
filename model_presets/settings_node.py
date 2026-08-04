"""Settings Select — emit a saved preset's sampler settings, without touching the model.

Model Select couples "which model" to "which settings", which is right for the common case and wrong
for the investigating case: one model, several runs, different settings and seeds, compared
side by side. Two Model Selects would each want to own the model. This node is Model Select minus
the loading — pick a preset, optionally force the seed, out comes a ``SAMPLER_CFG`` chain.

It still needs to know *which model's* presets to offer, since that's what keeps the list valid.
Two ways, and the wired one is better:

  * wire **Model Select's ``model_id`` output** into ``model_id`` — one link, and the picker follows
    whatever that node has selected, so switching model in one place updates every Settings Select
    hanging off it;
  * or pick the model in this node's own ``model`` dropdown, when there's no Model Select to follow.

``label`` is there for comparison runs: feed it to Image Compare's ``captions`` (via a Set/Get
Accumulator) so each image says which preset — and which seed — produced it.
"""
import json

from ..ouroboros.nodes import SAMPLER_CFG
from . import store
from .select_node import ModelSelect, _seeded, _stage_line, _stage_warnings


class SettingsSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (store.model_names(), {"default": store.NONE, "tooltip": "Whose presets to offer. Ignored when 'model_id' is wired — then the list follows that Model Select's own pick."}),
                "preset": (ModelSelect._all_preset_names(), {"default": store.NONE, "tooltip": "A saved preset for the model above (its own, plus any shared with its families). Save presets with Settings Save."}),
                "seed_override": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "-1 = use the seed stored in the preset. Anything else replaces the seed on every stage — this is the field to sweep when you want the same settings at several seeds."}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8, "tooltip": "Fallback size, used only when the chosen preset carries no size of its own."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8, "tooltip": "Fallback height — see 'width'."}),
            },
            "optional": {
                "model_id": ("STRING", {"forceInput": True, "tooltip": "Wire Model Select's 'model_id' output here. It wins over the 'model' dropdown, and the preset list follows it — so the model is chosen in exactly one place."}),
            },
        }

    # The library grows without an /object_info reload, so skip strict combo membership.
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Report when the *library* changed — that's all this has to do.

        Widget values are already part of ComfyUI's cache signature (caching.py:116-122), so
        IS_CHANGED only needs to cover state the graph can't see: a preset edited on disk, which
        moves no widget at all.

        Why a whole-library fingerprint rather than a hash of this preset: IS_CHANGED's inputs are
        resolved with no execution list, so a WIRED ``model_id`` arrives as ``None``
        ("we only want constants in IS_CHANGED", execution.py:88) and the preset can't be pinned
        down. Over-invalidating is the safe direction here — this node loads nothing, so a spurious
        re-run costs almost nothing, while reusing stale settings would quietly falsify a comparison.
        """
        return store.fingerprint()

    RETURN_TYPES = (SAMPLER_CFG, "INT", "INT", "STRING", "STRING", "GEN_INFO")
    RETURN_NAMES = ("sampler_settings", "width", "height", "label", "info", "gen_extra_info")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/model"
    DESCRIPTION = ("Emit a saved sampler preset without loading a model — for running one model at "
                   "several settings / seeds in the same graph. Wire Model Select's 'model_id' in "
                   "and the preset list follows its pick. 'label' is a ready-made caption for "
                   "Image Compare.")

    def run(self, model=None, preset=None, seed_override=-1, width=1024, height=1024,
            model_id=None):
        wired = isinstance(model_id, str) and model_id.strip() != ""
        mid = model_id.strip() if wired else model
        pname = preset if preset and preset != store.NONE else None
        warnings = []

        rec, pre, _ = store.resolve(mid, pname)
        if mid in (None, "", store.NONE):
            warnings.append("no model chosen — pick one, or wire Model Select's 'model_id'")
        elif rec is None:
            warnings.append(f"model '{mid}' is not in the library")
        elif pname and pre is None:
            warnings.append(f"preset '{pname}' is not available for '{mid}' — no settings emitted")

        stages = _seeded((pre or {}).get("stages") or [], seed_override)
        warnings += _stage_warnings(stages)
        w = int((pre or {}).get("width") or 0) or int(width)
        h = int((pre or {}).get("height") or 0) or int(height)

        label = pname or "no preset"
        if int(seed_override) >= 0:
            label += f" · seed {int(seed_override)}"

        lines = [f"Settings Select — {mid or '(no model)'} · preset {pname or 'none'}"
                 + (" (model from wire)" if wired else "")]
        lines.append(f"  settings: {len(stages)} stage(s) · {w}×{h}")
        for i, s in enumerate(stages):
            lines.append(_stage_line(i, s))
        if int(seed_override) >= 0:
            lines.append(f"  seed forced to {int(seed_override)} on every stage")
        if pre and pre.get("overrides"):
            # Overrides retune the model's bundle, and this node never touches the bundle — so say
            # so rather than let someone believe a shift override took effect here.
            lines.append(f"  ⓘ this preset carries {len(pre['overrides'])} bundle override(s) "
                         f"({', '.join(sorted(pre['overrides']))}) — they apply only where the "
                         f"MODEL is built, i.e. in Model Select, not here")
        if pre:
            measured = []
            if pre.get("score") is not None:
                measured.append(f"score {pre['score']:.2f}")
            if pre.get("seconds") is not None:
                measured.append(f"{pre['seconds']:.1f}s")
            if measured:
                lines.append("  measured: " + " · ".join(measured))
            if pre.get("notes"):
                lines.append(f"  preset notes: {pre['notes']}")
        for wn in warnings:
            lines.append(f"  ⚠ {wn}")
        info = "\n".join(lines)
        print("[Settings Select] " + info.replace("\n", "\n[Settings Select] "))

        params = {"model": mid or "", "preset": pname or "none", "size": f"{w}×{h}"}
        if int(seed_override) >= 0:
            params["seed_override"] = int(seed_override)
        if pre and pre.get("score") is not None:
            params["preset_score"] = pre["score"]
        gen_extra = json.dumps([{"class_type": "Settings Select", "ord": 1, "params": params}],
                               ensure_ascii=False)
        return (stages, w, h, label, info, gen_extra)


NODE_CLASS_MAPPINGS = {"KinburgSettingsSelect": SettingsSelect}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSettingsSelect": "Settings Select ⚙"}
