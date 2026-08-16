"""Settings Save — put a Sampler Settings chain into a model's preset library, measurements and all.

Sits inline: ``Sampler Settings → Settings Save → Chimera``. The ``SAMPLER_CFG`` passes through
untouched, so adding it changes nothing about the run — it only records what the run used.

Saving happens when ``preset_name`` is non-empty; clear the field to stop saving. That's the same
rule as ``save_preset_as`` on the Character / Entity Card nodes, and for the same reason: a button
on the frontend can't see values that arrived over a wire, and the values here (score, seconds,
size) all arrive over wires.

The measurement inputs are what make the library more than a notebook. ``score`` takes Ouroboros'
``best_score`` or a number pulled out of the Vision Judge's ``results_json``; ``seconds`` takes
Chimera's own ``seconds``. Once stored, Model Select shows them and the picker can be sorted by
them — "which settings are actually good on this model" stops being something to remember.

A preset saved as ``shared`` lives in the family-wide pool instead of under one model, so one good
recipe serves every model that declares a matching family (see store.presets_for).
"""
from ..ouroboros.nodes import SAMPLER_CFG
from . import store
from ..categories import CAT_MODEL


def _flatten(cfg):
    """A SAMPLER_CFG is a stage dict, a chain of them, or None → always a flat list. Same
    normalisation Chimera does, so what's saved is what the sampler would have run."""
    if isinstance(cfg, list):
        return [s for s in cfg if isinstance(s, dict)]
    if isinstance(cfg, dict):
        return [cfg]
    return []


def _size_from_latent(latent):
    """Latent → pixel size, the way Ouroboros derives it (samples are 8× smaller per side)."""
    try:
        samples = latent["samples"]
        return int(samples.shape[-1]) * 8, int(samples.shape[-2]) * 8
    except Exception:
        return 0, 0


class SettingsSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_settings": (SAMPLER_CFG, {"tooltip": "The chain to save — passes through unchanged. Chain several Sampler Settings and the whole multi-stage recipe is stored as one preset (Chimera flattens it back into stages)."}),
                "model": (store.model_names(), {"default": store.NONE, "tooltip": "Which model these settings belong to. Register models with Model Capture first. Ignored when 'shared' is on, and overridden when 'model_id' is wired."}),
                "preset_name": ("STRING", {"default": "", "tooltip": "Name to save under — SAVING HAPPENS WHEN THIS IS NON-EMPTY, and re-saving the same name overwrites. Clear the field to leave the library alone."}),
            },
            "optional": {
                "model_id": ("STRING", {"forceInput": True, "tooltip": "Wire Model Select's 'model_id' output here and the preset is filed under the model that actually ran — no second pick to keep in sync, and no way to file it under the wrong one by mistake. Wins over the 'model' dropdown."}),
                "shared": ("BOOLEAN", {"default": False, "tooltip": "Save into the family-wide pool instead of under one model. Fill 'families' too — the preset then shows up in Model Select for every model that declares a matching family, so a good generic recipe isn't copied per model."}),
                "families": ("STRING", {"default": "", "tooltip": "Comma-separated families this preset applies to (only meaningful with 'shared'), e.g. 'flow-1024'."}),
                "tags": ("STRING", {"default": "", "tooltip": "Comma-separated labels for filtering a growing library, e.g. 'fast, portrait'."}),
                "set_default": ("BOOLEAN", {"default": False, "tooltip": "Make this the model's default preset — what the picker lands on when you choose the model. Clears the flag on the model's other presets."}),
                "notes": ("STRING", {"multiline": True, "default": "", "tooltip": "What this preset is for / what it traded off. Shown in Model Select's info."}),
                "score": ("FLOAT", {"forceInput": True, "tooltip": "Measured quality — wire Ouroboros' 'best_score', or a Vision Judge score pulled out of its results_json. Stored with the preset and shown in the picker."}),
                "seconds": ("FLOAT", {"forceInput": True, "tooltip": "Measured cost — wire Chimera's 'seconds' output (it times its own sampling, so it can't be skewed by the rest of the graph)."}),
                "latent": ("LATENT", {"tooltip": "Optional: the latent this preset was run at. Its size is stored with the preset and comes back out of Model Select's width / height, so the preset carries the resolution the model likes."}),
            },
        }

    # The model list grows without an /object_info reload, so skip strict combo membership.
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = (SAMPLER_CFG, "STRING")
    RETURN_NAMES = ("sampler_settings", "report")
    FUNCTION = "run"
    CATEGORY = CAT_MODEL
    DESCRIPTION = ("Save a Sampler Settings chain into a model's preset library — with its measured "
                   "score and time, so Model Select can show and sort by what actually worked. "
                   "Pass-through: wire it between Sampler Settings and Chimera / Ouroboros. Saves "
                   "only while 'preset_name' is non-empty.")

    def run(self, sampler_settings=None, model=None, preset_name="", shared=False, families="",
            tags="", set_default=False, notes="", score=None, seconds=None, latent=None,
            model_id=None):
        stages = _flatten(sampler_settings)
        name = (preset_name or "").strip()
        wired = isinstance(model_id, str) and model_id.strip() != ""
        model = model_id.strip() if wired else model
        lines = [f"Settings Save — {len(stages)} stage(s)"
                 + (" · model from wire" if wired else "")]

        if not name:
            lines.append("  (not saving — 'preset_name' is empty)")
            return self._out(sampler_settings, lines)
        if not stages:
            lines.append("  ✕ nothing to save — no sampler stages on the input")
            return self._out(sampler_settings, lines)
        if not shared and (not model or model == store.NONE):
            lines.append("  ✕ no model — wire 'model_id' or pick one (or turn 'shared' on and "
                         "fill 'families')")
            return self._out(sampler_settings, lines)
        if shared and not (families or "").strip():
            lines.append("  ✕ a shared preset needs 'families' — otherwise no model can see it")
            return self._out(sampler_settings, lines)

        w, h = _size_from_latent(latent) if latent is not None else (0, 0)
        preset = {
            "stages": stages,
            "families": families if shared else "",
            "tags": tags,
            "notes": notes,
            "default": bool(set_default) and not shared,
            "width": w,
            "height": h,
            "score": score,
            "seconds": seconds,
        }
        try:
            store.upsert_preset(model, name, preset, shared=bool(shared))
        except ValueError as e:
            lines.append(f"  ✕ {e}")
            return self._out(sampler_settings, lines)

        where = f"shared ({families.strip()})" if shared else f"model '{model}'"
        lines.append(f"  ✔ saved '{name}' under {where}")
        if shared:
            # A typo'd family on a shared preset is the one failure with no symptom: no error, the
            # preset just never appears in any picker. Say it out loud.
            for wn in store.family_warnings(families, exclude_preset=name):
                lines.append(f"  ⚠ {wn}")
        extra = []
        if w and h:
            extra.append(f"{w}×{h}")
        if score is not None:
            extra.append(f"score {float(score):.2f}")
        if seconds is not None:
            extra.append(f"{float(seconds):.1f}s")
        if preset["default"]:
            extra.append("default for this model")
        if extra:
            lines.append("    " + " · ".join(extra))
        return self._out(sampler_settings, lines)

    def _out(self, cfg, lines):
        report = "\n".join(lines)
        print("[Settings Save] " + report.replace("\n", "\n[Settings Save] "))
        return (cfg, report)


NODE_CLASS_MAPPINGS = {"KinburgSettingsSave": SettingsSave}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSettingsSave": "Settings Save 💾"}
