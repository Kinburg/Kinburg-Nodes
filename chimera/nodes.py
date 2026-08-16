"""Chimera — a flexible multi-stage sampler (many heads, one beast, one image).

Runs 2+ **Sampler Settings** bundles as consecutive stages of a SINGLE image, the way the Ouroboros
refine chain does, but with the step budget under explicit control and a physically correct handoff
between stages.

The core idea (and what makes it different from wiring two KSamplers together):

  * **continuous handoff** (default) — ONE sigma schedule of `curve_len` steps is built once, and each
    stage runs a *slice* of it. Stage 1 goes 0→a and leaves the latent sitting at the sigma of step a
    (partially denoised); stage 2 resumes at a with NO fresh noise and finishes a→end. The whole run
    is exactly one traversal of one noise trajectory — just with different samplers / cfg / eta per
    segment. It is the equivalent of chaining two `KSampler (Advanced)` nodes via
    start_at_step / end_at_step, except the schedule can't get out of sync because this node owns it.
  * **restart handoff** — the classic "two KSamplers" pattern: every stage builds its OWN schedule
    and adds its OWN noise, so later stages need `denoise < 1` or they'll obliterate the previous
    stage. Kept because it's a genuinely different (img2img-refine) effect, not a worse one.

Because the node owns the schedule, a stage's `steps` simply means "how many steps THIS stage runs".
`step_split` picks WHERE the boundary falls; the four modes are the same decision expressed in
different units, so the one to reach for is whichever unit you're thinking in:

  * **stage steps** — each stage runs exactly the `steps` it declares; the curve is their sum (or
    `total_steps`, if you set it — then a shorter sum ends the run early, leaving residual noise).
  * **at step**    — `total_steps` cut at `handoff_step`: stage 1 gets that many steps, the rest the
    remainder. Same unit as 'stage steps', but the boundary lives on THIS node, so you can sweep it
    without touching the Sampler Settings.
  * **at percent** — the same cut, given as a percentage OF THE STEPS (60% of 30 → 18 / 12).
  * **at sigma**   — the cut lands where the NOISE LEVEL crosses `handoff_sigma`. The one to use when
    the stages run different schedulers, or when you want the boundary to stay put as the step count
    changes: step indices aren't comparable across schedules, noise levels are.

Optional per-stage overrides for everything after stage 1: `model_b` (different LoRA stack /
refiner), `positive_b` / `negative_b` (refine on a different prompt).
"""
import json
import time

import torch

from ..ouroboros.nodes import (
    SAMPLER_CFG, _stage_extra_options, _sample_stage, build_dual_guider, guider_model,
)
from ..timer.timer_nodes import _format_elapsed
from ..categories import CAT_CHIMERA

SPLIT_MODES = ["stage steps", "at step", "at percent", "at sigma"]
# Older workflows stored the first names these modes had; map them so nothing breaks on load.
LEGACY_SPLIT = {"manual": "stage steps", "percent": "at percent", "sigma": "at sigma"}
HANDOFF_MODES = ["continuous", "restart"]


# ------------------------------------------------------------------------------------- stage prep
def _flatten_cfg(cfg):
    """A SAMPLER_CFG input is a stage dict, a chain (list of them), or None → always a flat list."""
    if isinstance(cfg, list):
        return [s for s in cfg if isinstance(s, dict)]
    if isinstance(cfg, dict):
        return [cfg]
    return []


def _norm_stage(sp):
    """Fill in every dial a stage needs, so a hand-built or older bundle can't KeyError us."""
    return {
        "seed": int(sp.get("seed", 0)),
        "steps": int(sp.get("steps", 20)),
        "cfg": float(sp.get("cfg", 7.0)),
        "sampler_name": sp.get("sampler_name", "euler"),
        "scheduler": sp.get("scheduler", "normal"),
        "denoise": float(sp.get("denoise", 1.0)),
        "eta": float(sp.get("eta", 1.0)),
        "s_noise": float(sp.get("s_noise", 1.0)),
        "s_churn": float(sp.get("s_churn", 0.0)),
        "solver_type": sp.get("solver_type", "midpoint"),
    }


def _stage_label(stg):
    return f"{stg['sampler_name']}/{stg['scheduler']}"


# ---------------------------------------------------------------------------------------- sigmas
def _build_sigmas(model, stage, steps, denoise):
    """The sigma schedule for `steps` steps of `stage`'s sampler+scheduler. Goes through the stock
    `comfy.samplers.KSampler` so the denoise trick (build steps/denoise, keep the tail) and the
    discard-penultimate handling stay bit-identical to every other sampler node."""
    import comfy.samplers
    ks = comfy.samplers.KSampler(model, steps=int(steps), device=model.load_device,
                                 sampler=stage["sampler_name"], scheduler=stage["scheduler"],
                                 denoise=float(denoise), model_options=model.model_options)
    return ks.sigmas


def _sigma_index(sigmas, target, curve_len):
    """First step index whose sigma has fallen to `target` (schedules run high→low). Clamped to
    1..curve_len-1 so both sides of the split always get at least one step."""
    target = float(target)
    for i in range(1, int(curve_len)):
        if float(sigmas[i]) <= target:
            return i
    return max(1, int(curve_len) - 1)


def _split_counts(curve_len, declared, mode, pct, sigmas, handoff_sigma, handoff_step=0):
    """Steps per stage. 'stage steps' = as declared (clamped to the curve); the other three name a
    boundary for the FIRST stage — in steps, in percent OF STEPS, or at a noise level — and the
    remainder is shared out among the rest in proportion to what they declared (so a 3-stage chain
    still works even though the boundary only names one split)."""
    n = len(declared)
    curve_len = max(1, int(curve_len))
    if n == 1:
        want = int(declared[0]) if mode == "stage steps" else curve_len
        return [max(0, min(want, curve_len))]

    if mode == "stage steps":
        out, used = [], 0
        for d in declared:
            c = max(0, min(int(d), curve_len - used))
            out.append(c)
            used += c
        return out

    if mode == "at step":
        a = int(handoff_step) if int(handoff_step) > 0 else int(round(curve_len / 2.0))
    elif mode == "at percent":
        a = int(round(curve_len * float(pct) / 100.0))
    else:
        a = _sigma_index(sigmas, handoff_sigma, curve_len)
    a = max(1, min(curve_len - 1, a))
    rest, tail = curve_len - a, declared[1:]
    if n == 2:
        return [a, rest]
    # 3+ stages: hand out `rest` proportionally, keeping ≥1 step each while it lasts.
    weights = [max(1, int(t)) for t in tail]
    total_w = sum(weights) or len(weights)
    counts, acc = [a], 0
    for i, w in enumerate(weights):
        left_after = len(weights) - i - 1
        if i == len(weights) - 1:
            c = rest - acc
        else:
            c = int(round(rest * w / total_w))
            c = max(0, min(c, rest - acc - left_after))
        counts.append(max(0, c))
        acc += max(0, c)
    return counts


# ------------------------------------------------------------------------------------- sampling
def _sample_slice(model, latent, positive, negative, stg, sigmas, seed, add_noise, x0_out=None,
                  guider=None):
    """Run ONE slice of a shared schedule. `add_noise=False` continues an in-progress trajectory
    (zero noise — exactly what SamplerCustom does with add_noise disabled), so the latent's existing
    noise level must match `sigmas[0]`. `seed` still drives the stochastic samplers' internal noise
    sampler, so each stage's ancestral/SDE randomness stays independently controllable.
    With a `guider`, the model, conditioning and cfg all come from it instead."""
    import comfy.model_management
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview
    name = stg["sampler_name"]
    extra = _stage_extra_options(name, stg["eta"], stg["s_noise"], stg["s_churn"], stg["solver_type"])
    sampler_obj = comfy.samplers.ksampler(name, extra) if extra else comfy.samplers.sampler_object(name)
    ref = guider_model(guider) or model
    out = latent.copy()
    latent_image = comfy.sample.fix_empty_latent_channels(ref, latent["samples"])
    if sigmas is None or sigmas.numel() < 2:  # nothing to run (0 steps / denoise≤0) → pass through
        out["samples"] = latent_image
        return out
    if add_noise:
        noise = comfy.sample.prepare_noise(latent_image, seed, latent.get("batch_index"))
    else:
        noise = torch.zeros(latent_image.shape, dtype=latent_image.dtype,
                            layout=latent_image.layout, device="cpu")
    callback = latent_preview.prepare_callback(ref, sigmas.shape[-1] - 1, x0_out)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    if guider is not None:
        samples = guider.sample(noise, latent_image, sampler_obj, sigmas,
                                denoise_mask=latent.get("noise_mask"), callback=callback,
                                disable_pbar=disable_pbar, seed=seed)
        out["samples"] = samples.to(comfy.model_management.intermediate_device())
    else:
        out["samples"] = comfy.sample.sample_custom(
            model, noise, stg["cfg"], sampler_obj, sigmas, positive, negative, latent_image,
            noise_mask=latent.get("noise_mask"), callback=callback, disable_pbar=disable_pbar,
            seed=seed)
    return out


def _tail_for_scheduler(model, stage, master, start, count, curve_len, denoise):
    """Continuous handoff with a DIFFERENT scheduler on a later stage: rebuild the curve with that
    stage's scheduler over the same length, take its tail, and pin the first sigma to the handoff
    value the latent actually carries (a mismatch there is what produces the classic 'refiner
    artifacts'). `cummin` keeps the result monotonically non-increasing. Falls back to the master
    tail if the alternate curve doesn't line up."""
    try:
        alt = _build_sigmas(model, stage, curve_len, denoise)
        if alt is None or alt.shape[-1] != master.shape[-1]:
            return master[start:start + count + 1], False
        tail = alt[start:start + count + 1].clone().to(master.device)
        tail[0] = master[start]
        return torch.cummin(tail, dim=0).values, True
    except Exception as e:
        print(f"[Chimera] scheduler splice failed ({e}) — using the master schedule for this stage")
        return master[start:start + count + 1], False


# ------------------------------------------------------------------------------------------- node
class KinburgChimeraSampler:
    """Multi-stage sampler: run several Sampler Settings bundles over one image, one shared schedule."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "stage_a": (SAMPLER_CFG, {"tooltip": "First stage — a 'Sampler Settings' bundle. Wire a second one into 'stage_b' for two-stage sampling (a chain of Sampler Settings also works and is flattened into stages, left→right). NOTE: 'seed_mode' / 'seed_step' on the Settings node are Ouroboros-loop dials and are ignored here — only 'seed' is used."}),
                "handoff": (HANDOFF_MODES, {"default": "continuous", "tooltip": "How stage 2+ picks up from stage 1.\n\n• continuous (recommended) — ONE schedule is built and each stage runs a slice of it; later stages add NO fresh noise and resume exactly where the previous one stopped. The run is one honest traversal of one noise trajectory with different samplers/cfg/eta per segment. The shared curve takes its scheduler + denoise from STAGE A (a later stage's own denoise is ignored; a different scheduler is spliced in — see the report).\n\n• restart — the classic 'two KSamplers' pattern: every stage builds its own schedule and adds its own noise, so stage 2+ NEEDS denoise<1 or it will destroy stage 1's image. A different effect (img2img refine), not a better or worse one."}),
                "step_split": (SPLIT_MODES, {"default": "stage steps", "tooltip": "WHERE the boundary between the stages falls. The last three are the same decision in different units — pick whichever unit you're thinking in.\n\n• stage steps — each stage runs exactly the 'steps' it declares; the curve is their sum (unless total_steps overrides it).\n\n• at step — cut total_steps at 'handoff_step': stage 1 gets that many steps, the rest gets the remainder. Same unit as above, but the boundary is on THIS node, so you can sweep it without editing the Sampler Settings.\n\n• at percent — the same cut as a percentage OF THE STEPS: 60% of 30 steps → 18 / 12. (It divides steps, not the sigma range.)\n\n• at sigma — the cut lands where the NOISE LEVEL crosses 'handoff_sigma'. Use it when the stages run different schedulers, or to keep the boundary in the same place as you change the step count: step numbers aren't comparable across schedules, noise levels are.\n\nThe last three ignore the 'steps' set inside the Sampler Settings nodes."}),
                "total_steps": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Length of the shared schedule. 0 = auto (the sum of the stages' own 'steps').\n\nIn the 'stage steps' split, setting this LONGER than the sum leaves the tail of the curve unwalked, so the result keeps residual noise on purpose (reported). Shorter than the sum → the last stages get clipped."}),
                "handoff_step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "'at step' split only: how many steps the FIRST stage runs; everything left goes to the rest. 0 = halfway. Clamped so both sides always get at least one step."}),
                "split_percent": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "'at percent' split only: the share OF THE STEPS given to the FIRST stage (60 → 18/12 out of 30). Both sides always get at least one step."}),
                "handoff_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": "'at sigma' split only: hand over to the next stage once the noise level has fallen to this sigma.\n\nThe scale is MODEL-dependent: flow-matching models (Flux / SD3 / Krea) run 1.0 → 0, so the useful range is 0..1 and the 0.5 default sits mid-trajectory; SD / SDXL run from about 14.6, where you'd want single digits instead. The report prints the curve's actual range and the sigma the split landed on, so one run tells you the scale you're working in."}),
            },
            "optional": {
                "stage_b": (SAMPLER_CFG, {"tooltip": "Second stage — another 'Sampler Settings' bundle. Leave empty to run stage A alone (a plain single-pass sampler)."}),
                "model_negative": ("MODEL", {"tooltip": "Second model for the UNCONDITIONAL (negative) pass — for checkpoints shipped as a model + uncond-model pair, such as Ideogram. The dual-model guider is built PER STAGE from that stage's conditioning and cfg, so 'positive_b' and per-stage cfg keep working (wiring a ready-made 'Dual Model CFG Guider' into 'guider' instead would freeze both). NOTE: at cfg 1.0 the guider skips the unconditional pass entirely and this model does nothing — use a cfg above 1."}),
                "guider": ("GUIDER", {"tooltip": "Escape hatch for ANY custom guider (Dual CFG, CFG++, a hand-rolled one…). It carries its own model, conditioning and cfg, so this node's 'positive' / 'negative' / 'positive_b' and every stage's cfg are IGNORED while it's wired (the report says so). The sigma curve is built from the guider's own model. Takes precedence over 'model_negative'. For the common Ideogram case use 'model_negative' instead — it keeps the per-stage prompts working."}),
                "sigmas": ("SIGMAS", {"tooltip": "External noise schedule, replacing the one this node builds. 'total_steps' and every stage's scheduler and denoise are then IGNORED (reported) — the curve is given. Splitting still works exactly the same: all four 'step_split' modes just slice the supplied curve, so you can pair any scheduler node with staged sampling."}),
                "model_b": ("MODEL", {"tooltip": "Optional model for stage 2+ — a different LoRA stack or a refiner. It must share the noise schedule of the main model (same family): in 'continuous' handoff the shared curve is always built from the MAIN model, since that's the space the in-flight latent's noise level lives in."}),
                "positive_b": ("CONDITIONING", {"tooltip": "Optional positive conditioning for stage 2+ — e.g. polish on a shorter prompt, or without the style tokens. Falls back to 'positive'."}),
                "negative_b": ("CONDITIONING", {"tooltip": "Optional negative conditioning for stage 2+. Falls back to 'negative'."}),
                "verbose": ("BOOLEAN", {"default": True, "tooltip": "Print the split report (curve, per-stage step ranges and sigma boundaries, warnings) to the console. The same text is always on the 'report' output."}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "STRING", "GEN_INFO", "STRING", "FLOAT")
    RETURN_NAMES = ("latent", "handoff_latent", "handoff_denoised", "report", "gen_extra_info",
                    "time", "seconds")
    FUNCTION = "run"
    CATEGORY = CAT_CHIMERA
    DESCRIPTION = ("Multi-stage sampler: run two (or more) 'Sampler Settings' bundles over ONE image "
                   "with an explicit step budget. 'continuous' handoff builds a single sigma schedule "
                   "and gives each stage a slice of it — stage 2 resumes exactly where stage 1 stopped "
                   "without re-noising (like chaining KSampler Advanced by start/end step), so each "
                   "segment can use its own sampler, cfg and eta. Put the boundary where you like: "
                   "from each stage's own steps, at a step number, at a percentage of the steps, or "
                   "at a noise level (sigma). Also reports its OWN sampling time "
                   "(per stage and total), measured inside the node — unlike Start/Stop Timer it "
                   "can't be skewed by how ComfyUI schedules other branches of the graph. "
                   "'gen_extra_info' carries what it resolved at run time; wire it into a "
                   "Generation Info node's 'extra' input to add it to that branch's settings dump.")

    def run(self, model, positive, negative, latent_image, stage_a, handoff, step_split,
            total_steps, split_percent, handoff_sigma, handoff_step=0, stage_b=None, model_b=None,
            positive_b=None, negative_b=None, model_negative=None, guider=None, sigmas=None,
            verbose=True):
        if not (isinstance(latent_image, dict) and latent_image.get("samples") is not None):
            raise RuntimeError("[Chimera] No valid latent — wire a latent into 'latent_image'.")
        step_split = LEGACY_SPLIT.get(step_split, step_split)   # workflows saved under the old names
        raw = _flatten_cfg(stage_a) + _flatten_cfg(stage_b)
        if not raw:
            raise RuntimeError("[Chimera] No sampler settings — wire a 'Sampler Settings' node into "
                               "'stage_a'.")
        stages = [_norm_stage(s) for s in raw]
        n = len(stages)
        names = [chr(ord("A") + i) for i in range(n)]
        declared = [s["steps"] for s in stages]
        curve_len = int(total_steps) if int(total_steps) > 0 else sum(max(0, d) for d in declared)
        curve_len = max(1, curve_len)
        notes, lines = [], []
        continuous = (handoff == "continuous")
        base_denoise = stages[0]["denoise"]

        # A wired GUIDER owns the model, the conditioning AND the cfg, so everything this node knows
        # about those is superseded — including the per-stage overrides. `model_negative` is the
        # narrower (and usually better) route: the guider is rebuilt per stage below, so per-stage
        # prompts and cfg survive.
        if guider is not None and model_negative is not None:
            notes.append("both 'guider' and 'model_negative' are wired — the guider wins; "
                         "'model_negative' is unused")
            model_negative = None
        if guider is not None:
            notes.append("a GUIDER is wired: it carries its own model, conditioning and cfg, so this "
                         "node's positive / negative / positive_b and every stage's cfg are ignored")
        curve_model = guider_model(guider) or model
        if model_negative is not None:
            flat = [names[i] for i, s in enumerate(stages) if abs(s["cfg"] - 1.0) < 1e-6]
            if flat:
                notes.append(f"'model_negative' is wired but stage(s) {', '.join(flat)} run at "
                             f"cfg 1.0 — the dual guider skips the unconditional pass there, so the "
                             f"second model does NOTHING. Raise those stages' cfg above 1.")

        # A shared schedule is only meaningful in continuous mode; in restart mode each stage builds
        # its own, so we still need SOME curve for the 'sigma' split — stage A's is the right one.
        # An external SIGMAS input replaces it outright.
        external_sigmas = sigmas is not None and getattr(sigmas, "numel", lambda: 0)() >= 2
        if external_sigmas:
            master = sigmas.to(curve_model.load_device)
            curve_len = int(master.shape[-1]) - 1
            notes.append(f"external sigmas ({curve_len} steps, {float(master[0]):.4f} → "
                         f"{float(master[-1]):.4f}) — total_steps and every stage's scheduler and "
                         f"denoise are ignored")
        else:
            master = _build_sigmas(curve_model, stages[0], curve_len, base_denoise)
        if master is None or master.numel() < 2:
            msg = (f"[Chimera] stage A denoise {base_denoise} leaves nothing to sample — "
                   f"returning the input latent untouched.")
            print(msg)
            out = {k: v for k, v in latent_image.items()}
            noop = [{"class_type": "Chimera", "ord": 1,
                     "params": {"status": f"no-op — stage A denoise {base_denoise}"}}]
            return (out, out, out, msg, json.dumps(noop, ensure_ascii=False),
                    _format_elapsed(0.0, "auto"), 0.0)
        avail = int(master.shape[-1]) - 1
        if avail < curve_len:  # e.g. a scheduler that can't produce that many distinct sigmas
            notes.append(f"schedule yielded {avail} steps, not {curve_len} — using {avail}")
            curve_len = avail

        counts = _split_counts(curve_len, declared, step_split, split_percent, master, handoff_sigma,
                               handoff_step)
        walked = sum(counts)
        if step_split == "stage steps" and walked < curve_len:
            notes.append(f"stages walk {walked} of {curve_len} scheduled steps — the run stops at "
                         f"sigma {float(master[walked]):.4f}, so the result KEEPS residual noise "
                         f"(intentional if you set total_steps above the sum; otherwise raise a "
                         f"stage's steps)")
        elif step_split == "stage steps" and sum(max(0, d) for d in declared) > curve_len:
            notes.append(f"declared steps {sum(max(0, d) for d in declared)} exceed the curve "
                         f"({curve_len}) — later stages were clipped")
        if step_split != "stage steps":
            notes.append(f"'{step_split}' split — the stages' own 'steps' values are ignored")
        # Inpainting caveat: a continued stage passes ZERO noise, and comfy reuses that same tensor to
        # re-noise the frozen region behind a denoise_mask each step (samplers.py, KSamplerX0Inpaint) —
        # so the masked-out area is blended un-noised from stage 2 on. Stock `SamplerCustom` with
        # add_noise disabled behaves identically; 'restart' handoff, where every stage makes its own
        # noise, does not have the problem.
        if continuous and n > 1 and latent_image.get("noise_mask") is not None:
            notes.append("the input latent carries a NOISE MASK: in continuous handoff stage 2+ runs "
                         "on zero noise, so the frozen region is no longer re-noised as the steps "
                         "progress (same as SamplerCustom with add_noise disabled). For a masked "
                         "refine, use the 'restart' handoff — each stage then makes its own noise.")
        if n > 2 and step_split != "stage steps":
            notes.append(f"{n} stages with a single split point — stage A got its share, the "
                         f"remaining {curve_len - counts[0]} steps were divided among the rest in "
                         f"proportion to their declared steps")

        curve_src = ("external sigmas" if external_sigmas
                     else f"{_stage_label(stages[0])}, denoise {base_denoise:.2f}")
        if guider is not None:
            curve_src += " · GUIDER"
        elif model_negative is not None:
            curve_src += " · dual-model CFG"
        header = (f"Chimera — {handoff} handoff · {step_split} split · curve {curve_len} steps "
                  f"({curve_src}) · "
                  f"sigma {float(master[0]):.4f} → {float(master[curve_len]):.4f}")
        lines.append(header)
        if step_split == "at sigma":
            lines.append(f"  requested handoff sigma {float(handoff_sigma):.4f} → landed on step "
                         f"{counts[0]} (sigma {float(master[counts[0]]):.4f})")

        latent_cur = {k: v for k, v in latent_image.items()}
        latent_cur["samples"] = latent_image["samples"].clone()
        handoff_latent = handoff_denoised = None
        x0_out = {}
        start = 0
        total_secs = 0.0   # measured around the sampling calls themselves — see the note below
        # `gen_extra_info` is an ADDITION to a Generation Info dump, not a replacement for one: it
        # holds only what this node resolved at run time, which a graph walk over widget literals
        # can't see (the actual split, the sigma boundary, which stage used which settings, the
        # per-stage times). Feed it to Generation Info's `extra` input; that merged dump is what
        # goes to Set Accumulator (gen info) → Get → Generation Info Filter → Image Compare.
        # Reported as separate fields rather than one blob so the Filter's 'differences' mode can
        # show exactly what changed between the compared runs.
        head = {"handoff": handoff, "step_split": step_split,
                "steps": f"{walked} / {curve_len}",
                "curve": (f"{curve_src} · "
                          f"sigma {float(master[0]):.4f} → {float(master[curve_len]):.4f}")}
        if guider is not None:
            head["guidance"] = "external guider"
        elif model_negative is not None:
            head["guidance"] = "dual-model CFG (separate unconditional model)"
        stage_entries = []

        for i, stg in enumerate(stages):
            c = counts[i]
            if c <= 0:
                lines.append(f"  stage {names[i]}: 0 steps — skipped")
                continue
            m_i = model_b if (i > 0 and model_b is not None) else model
            p_i = positive_b if (i > 0 and positive_b is not None) else positive
            n_i = negative_b if (i > 0 and negative_b is not None) else negative
            over = []
            if m_i is not model:
                over.append("model_b")
            if p_i is not positive:
                over.append("positive_b")
            if n_i is not negative:
                over.append("negative_b")
            # A wired guider is used as-is; `model_negative` builds one per stage, so THIS stage's
            # conditioning and cfg are the ones baked into it.
            g_i = guider
            if g_i is None and model_negative is not None:
                g_i = build_dual_guider(m_i, model_negative, p_i, n_i, stg["cfg"])
                if g_i is not None:
                    over.append("dual-model CFG")
            elif g_i is not None:
                over.append("guider")

            if continuous:
                spliced = False
                # An external curve has no scheduler to rebuild a tail from, so stages just slice it.
                if i == 0 or external_sigmas or stg["scheduler"] == stages[0]["scheduler"]:
                    sl = master[start:start + c + 1]
                else:
                    sl, spliced = _tail_for_scheduler(curve_model, stg, master, start, c, curve_len,
                                                      base_denoise)
                want_x0 = (i == 0 and n > 1)
                t0 = time.perf_counter()
                latent_cur = _sample_slice(m_i, latent_cur, p_i, n_i, stg, sl, stg["seed"],
                                           add_noise=(i == 0),
                                           x0_out=(x0_out if want_x0 else None), guider=g_i)
                secs = time.perf_counter() - t0
                s_hi, s_lo = float(sl[0]), float(sl[-1])
                desc = (f"{_stage_label(stg)} · steps {start + 1}-{start + c} ({c}) · "
                        f"cfg {stg['cfg']} · eta {stg['eta']} · seed {stg['seed']} · "
                        f"sigma {s_hi:.4f} → {s_lo:.4f}")
                params = {"sampler": stg["sampler_name"], "scheduler": stg["scheduler"],
                          "steps": f"{start + 1}-{start + c} ({c})", "cfg": stg["cfg"],
                          "eta": stg["eta"], "seed": stg["seed"],
                          "sigma": f"{s_hi:.4f} → {s_lo:.4f}"}
                if i == 0:
                    params["denoise"] = base_denoise   # the shared curve's denoise, set by stage A
                if i > 0:
                    desc += " · continues (no added noise)"
                if spliced:
                    desc += f" · scheduler '{stg['scheduler']}' spliced onto the handoff sigma"
                    params["scheduler_spliced"] = True
                if i > 0 and stg["denoise"] < 0.9999:
                    notes.append(f"stage {names[i]} denoise {stg['denoise']:.2f} is IGNORED in "
                                 f"continuous handoff — the shared curve owns the noise level "
                                 f"(stage A's denoise {base_denoise:.2f} defines it). Switch "
                                 f"handoff to 'restart' if you wanted a re-noised refine pass.")
            else:
                stg_run = dict(stg)
                stg_run["steps"] = c
                t0 = time.perf_counter()
                latent_cur = _sample_stage(m_i, latent_cur, p_i, n_i, stg_run, stg["seed"], guider=g_i)
                secs = time.perf_counter() - t0
                desc = (f"{_stage_label(stg)} · {c} steps · cfg {stg['cfg']} · "
                        f"denoise {stg['denoise']:.2f} · eta {stg['eta']} · seed {stg['seed']} · "
                        f"own schedule, own noise")
                params = {"sampler": stg["sampler_name"], "scheduler": stg["scheduler"],
                          "steps": c, "cfg": stg["cfg"], "denoise": stg["denoise"],
                          "eta": stg["eta"], "seed": stg["seed"]}
                if i > 0 and stg["denoise"] > 0.9999:
                    notes.append(f"stage {names[i]} runs at denoise 1.00 in 'restart' handoff — it "
                                 f"re-noises from scratch and DISCARDS stage {names[i - 1]}'s "
                                 f"result. Lower its denoise, or use the 'continuous' handoff.")
            if over:
                desc += " · " + ", ".join(over)
                params["overrides"] = ", ".join(over)
            params["time"] = _format_elapsed(secs, "auto")
            desc += f" · {params['time']}"
            total_secs += secs
            lines.append(f"  stage {names[i]}: {desc}")
            stage_entries.append({"class_type": "Chimera stage",
                                  "ord": len(stage_entries) + 1, "params": params})
            if i == 0 and n > 1:
                handoff_latent = {k: v for k, v in latent_cur.items()}
                if continuous and "x0" in x0_out:
                    try:
                        d = latent_cur.copy()
                        d["samples"] = curve_model.model.process_latent_out(x0_out["x0"].cpu())
                        handoff_denoised = d
                    except Exception as e:
                        print(f"[Chimera] could not build handoff_denoised: {e}")
            start += c

        if handoff_latent is None:  # single stage → the handoff outputs mirror the result
            handoff_latent = latent_cur
        if handoff_denoised is None:
            # No x0 to show: in restart handoff stage A already ran to sigma 0, so its latent IS its
            # clean result; if a continuous run somehow produced no callback, the latent is the best
            # we have either way.
            handoff_denoised = handoff_latent
        # Timing is taken around the sampling calls themselves, inside this node, so it can't be
        # skewed by how ComfyUI schedules the rest of the graph (the failure mode of wrapping a
        # branch in Start/Stop Timer). It ends up honest because `sample_custom` moves the result to
        # the intermediate device before returning, which waits for the GPU. It covers SAMPLING ONLY
        # — CLIP encode and VAE decode happen in other nodes — but the first stage does absorb the
        # model load when the checkpoint isn't resident yet, since Comfy loads it inside that call.
        elapsed = _format_elapsed(total_secs, "auto")
        if walked > 0 and total_secs > 0:
            lines.append(f"  total sampling time: {elapsed} · {walked} steps · "
                         f"{total_secs / walked:.2f} s/step")
        head["time"] = elapsed
        gen_extra = json.dumps([{"class_type": "Chimera", "ord": 1, "params": head}] + stage_entries,
                               ensure_ascii=False)
        for w in notes:
            lines.append(f"  ⚠ {w}")
        report = "\n".join(lines)
        if verbose:
            print("[Chimera] " + report.replace("\n", "\n[Chimera] "))
        return (latent_cur, handoff_latent, handoff_denoised, report, gen_extra,
                elapsed, float(total_secs))


NODE_CLASS_MAPPINGS = {"KinburgChimeraSampler": KinburgChimeraSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgChimeraSampler": "Chimera (Multi-Sampler) 🦁"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
