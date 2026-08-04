"""Ouroboros — a self-correcting text→image sampler loop.

The snake eats its tail: an LLM expands the prompt → an image is sampled → a vision **critic**
scores it and returns concrete advice + negative-prompt terms → the advice feeds the next prompt
revision → repeat, until the score reaches a target (or max iterations). The best image across all
iterations is returned, plus the whole trajectory in the exact shape **Image Compare** wants.

Design (settled with the author):
  * ONE orchestrator node runs a plain Python `while` loop inside `run()` — no graph-expansion loop
    needed, state is just local variables. Generation is owned internally via ComfyUI's own
    `common_ksampler` / `CLIPTextEncode` / `VAEDecode`, so the node is a KSampler variant.
  * Modular settings, matching the repo's "settings bundle" idiom:
      - **Sampler Settings** → SAMPLER_CFG (chainable: wire them in series for a multi-stage refine
        pipeline — 2+ stages = draft then polish, critic judges only the final image)
      - **Critic Settings (GGUF)** → CRITIC (embeds a vision LLM_CONFIG + the evaluation rules)
      - the expander is a plain Local LLM Settings (GGUF) `LLM_CONFIG`.
  * **Fixed seed** + a fixed start latent so only the prompt (and accumulated negative) varies —
    a clean optimization signal. Two LLM configs (expander + critic); on low VRAM the same vision
    model file should back both. VRAM discipline is governed by the node's **low_vram** toggle: when
    ON we free ComfyUI models before each LLM call AND unload the worker after it (before the next
    diffusion) — strictly one model resident at a time — and AUTO-SIZE each call's `n_ctx` to what
    the request actually needs (the Settings-node `n_ctx` becomes the ceiling; measured per role
    from the previous iteration's prompt tokens). OFF keeps everything resident at the full `n_ctx`
    (fastest, no reloads). This toggle overrides the `unload_*` flags on the LLM Settings nodes.

Reuses the Vision LLM Judge machinery (multi-criteria grammar + parsing) and the Local LLM worker.
"""
import json
import random
import re
import time

import torch

from ..local_llm.llm_node import (
    LLM_CONFIG, build_llm_request, _generate_and_format, _shutdown_worker,
)
from ..vision_judge.judge_node import (
    _grammar_tail, _parse_criteria, _recover_obj, _clamp_int, _clean_tags, DEFAULT_CRITERIA,
)
from . import stop

# Custom socket types for the settings bundles (any matching string connects).
CRITIC = "KINBURG_CRITIC"
SAMPLER_CFG = "KINBURG_SAMPLER_CFG"

# Suggested system prompt for the ENHANCER — not applied automatically: paste it into the
# enhancer's own Local LLM Settings (GGUF) `system_prompt` (that node configures how to expand).
EXPANDER_SYSTEM = (
    "You are an expert prompt engineer for text-to-image models. You turn a user's intent into one "
    "strong, detailed image-generation prompt, faithful to every subject, attribute and relation in "
    "the intent. You work inside an automatic refine loop: from the second round on, the message "
    "includes a vision judge's numbered 'required fixes' for the previous image (and flaws to avoid) "
    "— your TOP priority is to apply EVERY fix with a concrete, visible change to the prompt, not a "
    "generic re-expansion, while keeping the intent intact. Reason internally; the visible answer "
    "must be the final image prompt ONLY — no analysis, no preamble."
)

CRITIC_SYSTEM = (
    "You are a strict, consistent image-quality judge inside an automatic refinement loop. You are "
    "shown ONE image plus evaluation criteria and the prompt it was generated from. Score it "
    "objectively, then give concrete, actionable advice on how to change the GENERATION PROMPT "
    "(not the image directly) to raise the weakest criteria, and list negative-prompt terms for "
    "the flaws you see. Reply with ONLY a single JSON object — no prose, no markdown, no fences."
)

DEFAULT_ADVICE_STYLE = "one or two concrete, actionable sentences"


# ----------------------------------------------------------------------------- critic grammar
def _build_critic_grammar(keys):
    """GBNF forcing the loop critic's verdict. Multi-criteria (keys present):
        {"scores": {k: int, …}, "advice": "<string>", "negative_add": [<string>…]}
    Single (no criteria): {"score": int, "advice": …, "negative_add": …}. Reuses the proven
    tags/string/int/ws productions of the Vision Judge grammar (via _grammar_tail)."""
    tail = _grammar_tail()
    if keys:
        body = ' ws "," '.join('ws "\\"' + k + '\\"" ws ":" ws int' for k in keys)
        root = ('root ::= ws "{" ws "\\"scores\\"" ws ":" ws "{" ' + body +
                ' ws "}" ws "," ws "\\"advice\\"" ws ":" ws string ws "," '
                'ws "\\"negative_add\\"" ws ":" ws tags ws "}" ws')
    else:
        root = ('root ::= ws "{" ws "\\"score\\"" ws ":" ws int ws "," '
                'ws "\\"advice\\"" ws ":" ws string ws "," '
                'ws "\\"negative_add\\"" ws ":" ws tags ws "}" ws')
    return root + "\n" + tail + "\n"


def _parse_critic(text, lo, hi, keys):
    """Parse the critic verdict → {score(float), scores, advice, negative_add}. None if no object."""
    obj = _recover_obj(text)
    if not isinstance(obj, dict):
        return None
    advice = obj.get("advice", "")
    advice = (advice if isinstance(advice, str) else str(advice)).strip()
    neg = _clean_tags(obj.get("negative_add", []))
    if keys:
        sc = obj.get("scores", {})
        if not isinstance(sc, dict):
            sc = {}
        scores = {k: _clamp_int(sc.get(k, lo), lo, hi, lo) for k in keys}
        mean = sum(scores.values()) / len(scores) if scores else float(lo)
        overall = round(max(lo, min(hi, mean)), 2)
        return {"score": overall, "scores": scores, "advice": advice, "negative_add": neg}
    return {"score": float(_clamp_int(obj.get("score", lo), lo, hi, lo)), "scores": {},
            "advice": advice, "negative_add": neg}


def _median(xs):
    xs = sorted(xs)
    k = len(xs)
    if not k:
        return 0.0
    return xs[k // 2] if k % 2 else (xs[k // 2 - 1] + xs[k // 2]) / 2.0


def _aggregate_verdicts(verdicts, keys, lo, hi):
    """Self-consistency: median score(s) across samples; advice from the sample nearest the median
    score; negative_add = union (dedup, order-preserving)."""
    if len(verdicts) == 1:
        return verdicts[0]
    scores = [v["score"] for v in verdicts]
    med = _median(scores)
    if keys:
        agg_scores = {k: int(round(_median([v["scores"].get(k, lo) for v in verdicts]))) for k in keys}
        agg_scores = {k: max(lo, min(hi, s)) for k, s in agg_scores.items()}
        overall = round(sum(agg_scores.values()) / len(agg_scores), 2) if agg_scores else float(lo)
    else:
        agg_scores, overall = {}, round(med, 2)
    pick = min(verdicts, key=lambda v: abs(v["score"] - med))
    seen, neg = set(), []
    for v in verdicts:
        for t in v["negative_add"]:
            if t.lower() not in seen:
                seen.add(t.lower()); neg.append(t)
    return {"score": overall, "scores": agg_scores, "advice": pick["advice"], "negative_add": neg}


# -------------------------------------------------------------------------------- text helpers
def _merge_negative(current, add_list):
    """Accumulate negative-prompt terms (comma/newline separated), dedup case-insensitively."""
    parts = [p.strip() for p in re.split(r"[,\n]", current or "") if p.strip()]
    seen = {p.lower() for p in parts}
    for a in add_list:
        a = str(a).strip()
        if a and a.lower() not in seen:
            parts.append(a); seen.add(a.lower())
    return ", ".join(parts)


SEED_MODES = ["fixed", "random", "increment", "decrement"]
_SEED_MAX = 0xffffffffffffffff


def _seed_for(base, mode, step, i, rng):
    """Per-iteration seed. fixed = same; random = next draw from an RNG seeded by `base` (varies
    per iteration yet reproducible per run); increment/decrement = base ± i*step (wrapped)."""
    if mode == "random":
        return rng.randint(0, _SEED_MAX)
    if mode == "increment":
        return (base + i * step) % (_SEED_MAX + 1)
    if mode == "decrement":
        return (base - i * step) % (_SEED_MAX + 1)
    return base  # fixed


# `solver_type` has TWO unrelated vocabularies in k-diffusion, and each sampler validates only its
# own — handing 'midpoint' to seeds_2 raises "solver_type must be 'phi_1' or 'phi_2'". Which family a
# sampler speaks is told by the default in its signature.
SOLVER_FAMILIES = ({"midpoint", "heun"}, {"phi_1", "phi_2"})
SOLVER_TYPES = ["midpoint", "heun", "phi_1", "phi_2"]
_solver_warned = set()


def _solver_family(default):
    for fam in SOLVER_FAMILIES:
        if default in fam:
            return fam
    return None


def _stage_extra_options(sampler_name, eta, s_noise, s_churn, solver_type):
    """Build the k-diffusion sampler kwargs, keeping ONLY those the chosen sampler actually accepts.
    We introspect `sample_<name>`'s signature: eta/s_noise apply to ancestral & SDE samplers, s_churn
    to euler/heun/dpm_2, solver_type to the dpmpp_2m_sde and SEEDS/exp families — passing an
    unsupported one would TypeError. Samplers with no `sample_<name>` (uni_pc, ddim) → empty dict
    (handled by sampler_object).

    `solver_type` needs one extra check beyond "does this sampler take it": the value has to come from
    the SAME family the sampler validates against, or it raises. A value from the other family is
    dropped so the sampler keeps its own default, rather than failing the run."""
    try:
        import inspect
        import comfy.k_diffusion.sampling as kds
        fn = getattr(kds, f"sample_{sampler_name}", None)
        params = dict(inspect.signature(fn).parameters) if fn is not None else {}
    except Exception:
        params = {}
    extra = {}
    if "eta" in params:
        extra["eta"] = float(eta)
    if "s_noise" in params:
        extra["s_noise"] = float(s_noise)
    if "s_churn" in params:
        extra["s_churn"] = float(s_churn)
    if "solver_type" in params:
        fam = _solver_family(params["solver_type"].default)
        want = str(solver_type)
        if fam is not None and want in fam:
            extra["solver_type"] = want
        elif fam is not None:
            key = (sampler_name, want)
            if key not in _solver_warned:   # once per combination, not once per step
                _solver_warned.add(key)
                print(f"[Kinburg] '{sampler_name}' only accepts solver_type "
                      f"{sorted(fam)} — '{want}' is from the other family and was ignored; "
                      f"the sampler's default '{params['solver_type'].default}' is used.")
    return extra


def guider_model(guider):
    """The ModelPatcher a GUIDER samples with — every CFGGuider keeps it as `model_patcher`."""
    return getattr(guider, "model_patcher", None)


def build_dual_guider(model, model_negative, positive, negative, cfg):
    """A guider that runs the positive pass on `model` and the UNCONDITIONAL pass on a second model —
    what checkpoints shipped as a model + uncond-model pair (Ideogram) need. Built here per call, so
    the conditioning and cfg are whatever the caller resolved for THIS stage/iteration; wiring a
    ready-made 'Dual Model CFG Guider' node in instead would freeze both.

    Returns None if this ComfyUI has no such guider, so the caller can fall back to normal CFG.
    NOTE: the guider skips the uncond model entirely at cfg 1.0 — the second model does nothing there."""
    try:
        from comfy_extras.nodes_custom_sampler import Guider_DualModel
    except Exception as e:
        print(f"[Kinburg] this ComfyUI has no dual-model guider ({e}) — sampling with normal CFG")
        return None
    g = Guider_DualModel(model, model_negative)
    g.set_conds(positive, negative)
    g.set_cfg(float(cfg))
    return g


def _sample_stage(model, latent, positive, negative, stg, seed, guider=None):
    """Run ONE sampling stage via ComfyUI's custom-sampler API (so per-sampler knobs — eta, s_noise,
    s_churn, solver_type — are reachable; `common_ksampler` can't pass them). Sigmas come from a stock
    `KSampler` object so the denoise trick and discard-penultimate handling stay bit-identical to the
    classic path; the sampler is `ksampler(name, extra)` with `extra` filtered to what the sampler
    accepts (falling back to `sampler_object` for uni_pc/ddim). Returns a latent dict; a denoise≤0
    stage is a pass-through. With a `guider`, the model/conditioning/cfg come from it instead."""
    import comfy.model_management
    import comfy.samplers
    import comfy.sample
    import comfy.utils
    import latent_preview
    name = stg["sampler_name"]
    extra = _stage_extra_options(name, stg["eta"], stg["s_noise"], stg["s_churn"], stg["solver_type"])
    sampler_obj = comfy.samplers.ksampler(name, extra) if extra else comfy.samplers.sampler_object(name)
    sched_model = guider_model(guider) or model
    ks = comfy.samplers.KSampler(sched_model, steps=stg["steps"], device=sched_model.load_device,
                                 sampler=name, scheduler=stg["scheduler"], denoise=stg["denoise"],
                                 model_options=sched_model.model_options)
    sigmas = ks.sigmas
    out = latent.copy()
    latent_image = comfy.sample.fix_empty_latent_channels(sched_model, latent["samples"])
    if sigmas is None or sigmas.numel() < 2:  # denoise≤0 → empty/1-elem sigmas → nothing to sample
        out["samples"] = latent_image
        return out
    noise = comfy.sample.prepare_noise(latent_image, seed, latent.get("batch_index"))
    callback = latent_preview.prepare_callback(sched_model, sigmas.shape[-1] - 1)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    if guider is not None:
        samples = guider.sample(noise, latent_image, sampler_obj, sigmas,
                                denoise_mask=latent.get("noise_mask"), callback=callback,
                                disable_pbar=disable_pbar, seed=seed)
        samples = samples.to(comfy.model_management.intermediate_device())
    else:
        samples = comfy.sample.sample_custom(model, noise, stg["cfg"], sampler_obj, sigmas,
                                             positive, negative, latent_image,
                                             noise_mask=latent.get("noise_mask"),
                                             callback=callback, disable_pbar=disable_pbar, seed=seed)
    out["samples"] = samples
    return out


def _thumb_b64(image, max_side=896):
    """JPEG data-URI preview of an IMAGE tensor [1,H,W,3] (0-1 float) for the live log. Sized so
    it's actually assessable (≈512px tall for 16:9 at max_side=896) yet still light over the
    websocket (~100-150 KB vs megabytes for a full-res PNG). "" on failure (log works without it)."""
    try:
        import io
        import base64
        import numpy as np
        from PIL import Image
        arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]
        arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
        im = Image.fromarray(arr)
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"[Ouroboros] preview encode failed: {e}")
        return ""


def _emit(payload):
    """Push a JSON event to the frontend (the Ouroboros Live Log node listens for it). Fire-and-
    forget, thread-safe (send_sync uses call_soon_threadsafe); no-op if the server isn't present."""
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("kinburg.ouroboros", payload)
    except Exception:
        pass


def _now():
    """Wall-clock HH:MM:SS for log timestamps."""
    return time.strftime("%H:%M:%S")


def _ctx_summary(stats):
    """Condense a raw LLM stats dict (filled by _generate_and_format) into the compact context-fill
    figures the live log / report show: prompt + generated tokens, total used, the n_ctx limit, the
    fill %, and truncation flags. None when there's nothing usable (e.g. the call errored)."""
    if not isinstance(stats, dict) or not stats:
        return None
    n_ctx = int(stats.get("n_ctx", 0) or 0)
    out = int(stats.get("output_tokens", 0) or 0)
    used = int(stats.get("context_used", 0) or 0)
    prompt = int(stats.get("prompt_tokens", 0) or 0)
    if used <= 0:  # worker couldn't read the KV fill — estimate from the text token counts
        prompt = prompt or (int(stats.get("sys_tokens", 0) or 0) + int(stats.get("user_tokens", 0) or 0))
        used = prompt + out
    if n_ctx <= 0 and used <= 0:
        return None
    finish = stats.get("finish_reason", "")
    return {"prompt": prompt, "output": out, "used": used, "n_ctx": n_ctx,
            "pct": round(100.0 * used / n_ctx, 1) if n_ctx > 0 else 0.0,
            "prompt_pct": round(100.0 * prompt / n_ctx, 1) if n_ctx > 0 else 0.0,
            "finish": finish, "truncated": finish == "length",
            "over": n_ctx > 0 and used >= n_ctx,
            # 'tight' = the PROMPT alone nearly fills n_ctx → genuinely need a bigger window. Based on
            # the prompt (not total used) so it doesn't false-fire in low_vram auto mode, where n_ctx
            # is intentionally sized close to prompt + generation budget.
            "tight": n_ctx > 0 and prompt >= 0.9 * n_ctx}


def _ctx_line(c, label):
    """Human one-liner for the console / report (or '' when there's no context data)."""
    if not c:
        return ""
    s = f"{label} ctx {c['used']}/{c['n_ctx']} ({c['pct']}%)  prompt {c['prompt']} + gen {c['output']}"
    warn = []
    if c["truncated"]:
        warn.append("output truncated — raise max_tokens")
    if c["over"] or c["tight"]:
        warn.append("prompt ≈ context limit — raise n_ctx")
    if warn:
        s += "  ⚠ " + "; ".join(warn)
    return s


def _bucket_ctx(need, cap, bucket=512):
    """Round a token requirement up to the next `bucket`, clamped to the ceiling `cap`. Used by
    low_vram auto-sizing to pick each LLM call's n_ctx: small enough to save KV-cache VRAM, rounded
    up so it isn't razor-thin, never above the Settings-node n_ctx."""
    cap = int(cap)
    if need <= 0:
        return cap
    v = ((int(need) + bucket - 1) // bucket) * bucket
    return max(1, min(cap, v))


def _auto_nctx(peak_prompt, peak_out, max_tokens, cap):
    """low_vram context size for a role, from what it has ACTUALLY used so far. Reserve = the
    largest prompt seen so far + room for the next prompt to grow by ~one generation (the enhancer
    folds its last output back into the next prompt) + an output budget scaled to the OUTPUT the
    model actually produces — NOT the `max_tokens` ceiling, which is often set far higher than the
    few hundred tokens these calls emit (that ceiling only caps the reserve). Bucketed to 512 and
    clamped to the Settings n_ctx. A rare underestimate is caught by the retry-at-ceiling below."""
    max_tokens = int(max_tokens)
    out_reserve = min(max_tokens, max(512, int(peak_out) * 2))
    need = int(peak_prompt) + int(peak_out) + out_reserve + 128
    return _bucket_ctx(need, cap)


LOG_MODES = ["streaming", "per step", "per iteration"]


def _append_triggers(prompt, triggers):
    """Guarantee LoRA trigger words survive the LLM rewrite: append any comma-separated trigger
    not already present (case-insensitive) to the end of the (expanded) prompt."""
    triggers = (triggers or "").strip()
    if not triggers:
        return prompt
    low = (prompt or "").lower()
    add = [t.strip() for t in triggers.split(",") if t.strip() and t.strip().lower() not in low]
    if not add:
        return prompt
    base = (prompt or "").rstrip().rstrip(",")
    tail = ", ".join(add)
    # Triggers go on their own line (blank line before) — readable in the log, and CLIP-safe
    # (same convention as Lora Unlim Accumulator; newlines are just token boundaries).
    return (base + "\n\n" + tail) if base.strip() else tail


def _advice_points(advice):
    """Split the critic's advice into discrete fixes (on newlines, semicolons, and sentence
    boundaries) so they can be presented as a numbered checklist the enhancer works through one by
    one, rather than one easy-to-skim blob."""
    text = (advice or "").strip()
    if not text:
        return []
    parts = re.split(r"\n+|;\s+|(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", text)
    return [p.strip().rstrip(".").strip() for p in parts if p and p.strip()]


def _reviser_user(intent, current, advice, history, history_n=4, score=None, score_max=None,
                  scores=None, avoid=None):
    """The message fed to the expander LLM to produce the next prompt. On a refine round the judge's
    feedback is presented as a prominent, numbered 'required fixes' block placed LAST — right before
    the output instruction — because models weight the end of the prompt most, so the enhancer
    actually acts on it (with the previous score and weakest criterion to aim at, plus `avoid` — the
    critic's flaw terms — so the enhancer can positively phrase against them, which matters on
    turbo/distilled models that barely honour the negative prompt). `history_n` caps how many recent
    iterations are recapped as 'already tried' context (0 = none)."""
    points = _advice_points(advice)
    revising = bool(points)
    lines = []
    if revising:
        lines.append("You are REVISING an image-generation prompt inside an automatic refine loop. "
                     "A vision judge scored the previous image and listed the fixes below. Apply "
                     "EVERY fix with a concrete, visible change to the prompt, while keeping the "
                     "original intent and all of its subjects intact.")
    else:
        lines.append("Turn the following intent into a strong, detailed image-generation prompt, "
                     "staying faithful to it.")

    lines.append(f"Original intent (preserve every subject, attribute and relation): {intent}")
    if current and current.strip() != intent.strip():
        lines.append(f"Current prompt to revise:\n{current}")
    if history and history_n > 0:
        tried = "; ".join(f"(score {h['score']}) {h['prompt'][:120]}" for h in history[-history_n:])
        lines.append(f"Already tried — do not repeat or regress to these: {tried}")

    if revising:
        weak = ""
        if isinstance(scores, dict) and scores:
            lo_val = min(scores.values())
            weakest = [k for k, v in scores.items() if v == lo_val]
            if weakest:
                weak = f", weakest: {', '.join(weakest)}"
        hdr = "▶ JUDGE FEEDBACK — required fixes"
        if score is not None and score_max:
            hdr += f" (previous score {score}/{score_max}{weak})"
        block = [hdr + ":"]
        block += ([f"{i + 1}. {p}" for i, p in enumerate(points)] if len(points) > 1 else [points[0]])
        avoid_terms = [str(t).strip() for t in (avoid or []) if str(t).strip()]
        if avoid_terms:
            block.append("Also steer the prompt AWAY from these flaws the judge saw (phrase it so "
                         "they don't appear): " + ", ".join(avoid_terms) + ".")
        block.append("Make a concrete change in the prompt for EACH point above — do not merely "
                     "re-expand the current prompt.")
        lines.append("\n".join(block))
        lines.append("Output ONLY the revised image prompt.")
    else:
        lines.append("Output ONLY the prompt, no preamble.")

    return "\n\n".join(lines)


def _critic_user(rubric, crit, keys, lo, hi, prompt, advice_style):
    """The message fed to the vision critic LLM alongside the image."""
    parts = []
    if rubric and rubric.strip():
        parts.append(rubric.strip())
    parts.append(f"The image was generated from this prompt:\n{prompt}")
    if keys:
        crit_lines = "\n".join(f'  - "{k}"' + (f': {d}' if d else '') for (k, _l, d) in crit)
        parts.append(f"Score EACH criterion as an integer {lo}-{hi} (higher is better):\n{crit_lines}")
        parts.append(
            f'Return a JSON object with: "scores" (object with exactly those keys), '
            f'"advice" ({advice_style} on how to change the PROMPT to raise the LOWEST-scoring '
            f'criteria while keeping the intent), and "negative_add" (array of short lowercase '
            f'negative-prompt terms for visible flaws, e.g. "extra fingers", "watermark"; may be empty).')
    else:
        parts.append(
            f'Score the image {lo}-{hi}. Return a JSON object with: "score" (integer), '
            f'"advice" ({advice_style} on how to improve the prompt), and "negative_add" '
            f'(array of short lowercase negative-prompt terms; may be empty).')
    return "\n\n".join(parts)


# ================================================================================ settings nodes
class KinburgSamplerSettings:
    """Bundle the KSampler dials into a SAMPLER_CFG for the Ouroboros node."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Fixed seed for the whole optimization — keep it constant so only the prompt varies (clean signal)."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed_mode": (SEED_MODES, {"default": "fixed", "tooltip": "How the seed changes each iteration: fixed = same seed (only the prompt varies — cleanest signal); random = a new seed each iteration (reproducible from this seed); increment / decrement = seed ± seed_step·iteration. NOTE: any non-fixed mode mixes 'the prompt improved' with 'the seed got lucky'."}),
                "seed_step": ("INT", {"default": 1, "min": 1, "max": 0xffffffff, "tooltip": "Step for the increment / decrement seed modes (ignored for fixed / random)."}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Stochasticity for ancestral / SDE samplers (euler_ancestral, dpmpp_*_sde, …). 0 = deterministic ODE (most repeatable); higher = more randomness / variation. NO effect on deterministic samplers like euler or dpmpp_2m."}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Multiplier for the extra noise added on stochastic steps. ~1.0 is standard; >1 adds grain / fine detail, <1 is smoother. Applies to ancestral / SDE (and churn-driven) samplers only."}),
                "s_churn": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Adds stochasticity to otherwise-deterministic samplers (euler, heun, dpm_2) by re-injecting noise between steps. 0 = off (default). Small values can add detail; too high adds artifacts. Ignored by samplers that don't support it."}),
                "solver_type": (SOLVER_TYPES, {"default": "midpoint", "tooltip": "Solver variant, for the samplers that have one — and there are TWO separate vocabularies:\n\n• 'midpoint' (smoother) / 'heun' (sharper, can be noisier) — for the dpmpp_2m_sde family.\n\n• 'phi_1' / 'phi_2' — for the SEEDS and exp_heun families (seeds_2, exp_heun_2_x0, exp_heun_2_x0_sde).\n\nA value from the wrong family is ignored (the sampler keeps its own default) and noted in the console, so picking one can never break a run. Samplers with no solver_type ignore this entirely."}),
            },
            "optional": {
                "sampler_settings": (SAMPLER_CFG, {"tooltip": "Wire a PREVIOUS Sampler Settings here to CHAIN them → a multi-stage refine pipeline inside Ouroboros. This node appends its own settings as the next stage (left→right = stage order). Leave empty for a single (classic) stage. In Ouroboros, 2+ stages = refine mode: stage 1 drafts, later stages polish the previous latent (use denoise<1 on them); the critic judges only the FINAL image."}),
            },
        }

    RETURN_TYPES = (SAMPLER_CFG,)
    RETURN_NAMES = ("sampler_settings",)
    FUNCTION = "build"
    CATEGORY = "Kinburg-Nodes/sampling"
    DESCRIPTION = ("KSampler settings bundle for the Ouroboros self-correcting sampler. Chain several "
                   "(optional 'sampler_settings' input) → a multi-stage refine pipeline. Image "
                   "dimensions come from the latent wired into Ouroboros.")

    def build(self, seed, steps, cfg, sampler_name, scheduler, denoise, seed_mode="fixed", seed_step=1,
              eta=1.0, s_noise=1.0, s_churn=0.0, solver_type="midpoint", sampler_settings=None):
        stage = {"seed": int(seed), "steps": int(steps), "cfg": float(cfg),
                 "sampler_name": sampler_name, "scheduler": scheduler, "denoise": float(denoise),
                 "seed_mode": seed_mode, "seed_step": int(seed_step),
                 "eta": float(eta), "s_noise": float(s_noise), "s_churn": float(s_churn),
                 "solver_type": solver_type}
        # Accumulate: the upstream chain (dict = one prior stage, or an already-built list) + this
        # stage. Always returns a LIST — Ouroboros normalizes single-or-list, so it stays back-compatible.
        chain = []
        if isinstance(sampler_settings, list):
            chain.extend(s for s in sampler_settings if isinstance(s, dict))
        elif isinstance(sampler_settings, dict):
            chain.append(sampler_settings)
        chain.append(stage)
        return (chain,)


class KinburgCriticSettings:
    """Bundle the loop critic's model + evaluation rules into a CRITIC. Embeds a vision LLM_CONFIG
    (a Local LLM Settings (GGUF) WITH a Vision Settings mmproj)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (LLM_CONFIG, {"tooltip": "A 'Local LLM Settings (GGUF)' WITH a 'Vision Settings (GGUF)' (mmproj) — the critic needs vision."}),
                "criteria": ("STRING", {"multiline": True, "default": DEFAULT_CRITERIA, "tooltip": "Criteria the critic scores each image on (one per line, 'name: description'). Pre-filled; clear for a single overall score."}),
                "score_min": ("INT", {"default": 1, "min": 0, "max": 100}),
                "score_max": ("INT", {"default": 5, "min": 1, "max": 100}),
            },
            "optional": {
                "rubric": ("STRING", {"multiline": True, "default": "Strictly and critically evaluate the generated image.", "tooltip": "Free-text guidance / tone for the critic."}),
                "system_prompt": ("STRING", {"multiline": True, "default": CRITIC_SYSTEM, "tooltip": "The critic's persona (system prompt). Default pre-filled; blank → built-in default."}),
                "advice_style": ("STRING", {"default": DEFAULT_ADVICE_STYLE, "tooltip": "How the advice should read, e.g. 'a detailed paragraph' or 'one short sentence'."}),
                "samples": ("INT", {"default": 1, "min": 1, "max": 9, "tooltip": "Self-consistency: judge each image N times and take the median score (N× critic cost)."}),
                "image_downscale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.5, "tooltip": "Shrink the image the critic sees by this factor before judging (2 = half-size, 4 = quarter). Fewer image tokens → faster judging and less VRAM, at the cost of fine detail — good when pixel-level precision isn't needed. 1.0 = full resolution (never upscales past the Vision Settings 'image_max_side' cap)."}),
            },
        }

    RETURN_TYPES = (CRITIC,)
    RETURN_NAMES = ("critic_settings",)
    FUNCTION = "build"
    CATEGORY = "Kinburg-Nodes/LLM"
    DESCRIPTION = "Critic (vision judge) settings bundle for the Ouroboros loop — model + scoring rules."

    def build(self, config, criteria, score_min, score_max,
              rubric="", system_prompt="", advice_style="", samples=1, image_downscale=1.0):
        return ({
            "config": config,
            "criteria": criteria or "",
            "rubric": rubric or "",
            "score_min": int(score_min),
            "score_max": int(score_max),
            "system_prompt": (system_prompt or "").strip() or CRITIC_SYSTEM,
            "advice_style": (advice_style or "").strip() or DEFAULT_ADVICE_STYLE,
            "samples": max(1, int(samples)),
            "image_downscale": max(1.0, float(image_downscale)),
        },)


# =============================================================================== the orchestrator
class KinburgOuroboros:
    """Ouroboros — generate → judge → revise → repeat, keeping the best image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "latent": ("LATENT", {"tooltip": "Starting latent — sets the image dimensions (wire an Empty Latent, matching your model: SD/SDXL vs SD3/Flux). For img2img/hires-feedback, feed a real latent and set denoise<1 in Sampler Settings. Reused each iteration so only the prompt varies."}),
                "user_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Your intent — what you want. The enhancer improves it; the critic anchors advice to it."}),
                "negative": ("STRING", {"multiline": True, "default": "", "tooltip": "Base negative prompt; the critic's flaw terms accumulate onto it each iteration."}),
                "enhancer_settings": (LLM_CONFIG, {"tooltip": "A 'Local LLM Settings (GGUF)' for the prompt-enhancer LLM — its model/temperature AND its system_prompt define how the prompt is expanded. On low VRAM, point it at the SAME model file as the critic."}),
                "critic_settings": (CRITIC, {"tooltip": "A 'Critic Settings (GGUF)' bundle — the vision judge + scoring rules."}),
                "sampler_settings": (SAMPLER_CFG, {"tooltip": "A 'Sampler Settings' bundle — OR a chain of them (wire Sampler Settings nodes in series). One = classic single pass. Two or more = REFINE mode: each iteration runs the stages in order (stage 1 drafts, later stages polish the previous latent with their own denoise / sampler / eta); the critic judges only the FINAL image."}),
                "target_score": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "The loop stops once the overall score reaches this (on the critic's score_min..score_max scale)."}),
                "max_iterations": ("INT", {"default": 6, "min": 1, "max": 100, "tooltip": "Hard cap on refinement rounds."}),
                "low_vram": ("BOOLEAN", {"default": True, "tooltip": "Low-VRAM mode (default ON). ON: strictly one model in VRAM at a time — frees ComfyUI models before each LLM call, unloads the LLM worker after it, and AUTO-SIZES each LLM's context (n_ctx) to what the request actually needs, so the n_ctx set on the LLM Settings nodes becomes the MAXIMUM. Best when the GPU can't hold the diffusion model + the LLM at once. OFF: keep everything resident (fastest, no reloads) and use the full n_ctx from the Settings nodes — for GPUs with VRAM to spare. NOTE: inside the loop this overrides the unload_* toggles on the LLM Settings nodes."}),
            },
            "optional": {
                "model_negative": ("MODEL", {"tooltip": "Second model for the UNCONDITIONAL (negative) pass — for checkpoints shipped as a model + uncond-model pair, such as Ideogram. The loop builds the dual-model guider itself on every iteration, so the freshly rewritten prompt is still what gets encoded (wiring a ready-made 'Dual Model CFG Guider' node here instead would freeze the conditioning and defeat the whole loop). NOTE: at cfg 1.0 the guider skips the uncond pass entirely and this model does nothing — give the Sampler Settings a cfg above 1."}),
                "enhancer_history": ("INT", {"default": 4, "min": 0, "max": 50, "tooltip": "How many of the most recent iterations (prompt + score) are recapped to the enhancer LLM as 'already tried, do not repeat' context. Higher = more memory of what failed (longer prompt, more tokens); 0 = no history (each rewrite sees only the intent, current prompt and the latest critic advice). The latest advice is always sent regardless."}),
                "trigger_words": ("STRING", {"forceInput": True, "tooltip": "Comma-separated words ALWAYS appended to the enhanced prompt (e.g. LoRA triggers from Lora Unlim Accumulator's 'triggers' output), so the LLM rewrite can't drop them."}),
                "full_console_log": ("BOOLEAN", {"default": True, "tooltip": "Console/terminal verbosity only (not the Live Log node or 'report'). On: per iteration also prints the enhanced prompt, advice and negative additions. Off: just a one-line score per iteration. The full trace is always in the 'report' output and the Live Log node."}),
                "log_mode": (LOG_MODES, {"default": "streaming", "tooltip": "How the Ouroboros Live Log node updates: 'streaming' is like 'per step' but the enhancer's prompt types out token by token as it's written (text only — the critic is grammar-constrained and can't stream); 'per step' posts each stage the moment it finishes (enhanced prompt → generated image → critic verdict), each timestamped; 'per iteration' posts one combined entry after the whole iteration (the older behaviour)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "IMAGE", "STRING", "FLOAT", "STRING", "INT",
                    "STRING", "STRING", "GEN_SETTINGS")
    RETURN_NAMES = ("images", "prompts", "judge_data", "best_image", "best_prompt", "best_score",
                    "report", "iterations", "captions", "times", "settings_data")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/sampling"
    DESCRIPTION = ("Self-correcting text→image loop: an LLM expands the prompt, an image is sampled, "
                   "a vision critic scores it and advises how to improve the prompt, and it repeats "
                   "until the target score (or max iterations). Outputs the whole trajectory (wire "
                   "images/prompts/judge_data straight into Image Compare) plus the best image/prompt.")

    def run(self, model, clip, vae, latent, user_prompt, negative, enhancer_settings,
            critic_settings, sampler_settings, target_score, max_iterations, low_vram=True,
            enhancer_history=4, trigger_words="", full_console_log=True, log_mode="per step",
            model_negative=None, unique_id=None):
        import comfy.model_management as mm
        from nodes import CLIPTextEncode, VAEDecode
        stop.clear_stop(unique_id)  # drop any stale stop request from a previous run
        stream = (log_mode == "streaming")  # also type the enhancer prompt out token by token
        live = log_mode in ("per step", "streaming")  # post each stage as it finishes (vs one per-iteration entry)
        log = lambda m: print(f"[Ouroboros {_now()}] {m}")
        try:
            from comfy.utils import ProgressBar
        except Exception:
            ProgressBar = None

        cs = critic_settings if isinstance(critic_settings, dict) else {}
        vcfg = cs.get("config") or {}
        rubric = cs.get("rubric", "")
        target = float(target_score)
        lo = int(cs.get("score_min", 1))
        hi = int(cs.get("score_max", 5))
        if hi < lo:
            lo, hi = hi, lo
        critic_system = cs.get("system_prompt") or CRITIC_SYSTEM
        advice_style = cs.get("advice_style") or DEFAULT_ADVICE_STYLE
        samples = max(1, int(cs.get("samples", 1)))
        crit_downscale = max(1.0, float(cs.get("image_downscale", 1.0) or 1.0))  # shrink the image the critic sees

        crit = _parse_criteria(cs.get("criteria", ""))
        keys = [k for (k, _l, _d) in crit]
        critic_grammar = _build_critic_grammar(keys)

        # Sampler settings may be a single bundle (classic) OR a CHAIN (Sampler Settings wired in
        # series → a list). One stage = the old behaviour; 2+ = REFINE mode — stage 1 drafts, later
        # stages polish the previous latent with their own denoise/sampler/eta. The critic ALWAYS
        # judges the FINAL image of the chain; there is no intermediate judging.
        if isinstance(sampler_settings, list):
            _raw_stages = [s for s in sampler_settings if isinstance(s, dict)]
        elif isinstance(sampler_settings, dict):
            _raw_stages = [sampler_settings]
        else:
            _raw_stages = []
        if not _raw_stages:
            _raw_stages = [{}]
        stages_norm = []
        for _sp in _raw_stages:
            _s_seed = int(_sp.get("seed", 0))
            stages_norm.append({
                "seed": _s_seed,
                "steps": int(_sp.get("steps", 20)),
                "cfg": float(_sp.get("cfg", 7.0)),
                "sampler_name": _sp.get("sampler_name", "euler"),
                "scheduler": _sp.get("scheduler", "normal"),
                "denoise": float(_sp.get("denoise", 1.0)),
                "seed_mode": _sp.get("seed_mode", "fixed"),
                "seed_step": int(_sp.get("seed_step", 1)),
                "eta": float(_sp.get("eta", 1.0)),
                "s_noise": float(_sp.get("s_noise", 1.0)),
                "s_churn": float(_sp.get("s_churn", 0.0)),
                "solver_type": _sp.get("solver_type", "midpoint"),
                "rng": random.Random(_s_seed),  # per-stage reproducible source for seed_mode="random"
            })
        n_stages = len(stages_norm)
        # Stage-0 aliases keep the existing single-stage console/report lines unchanged.
        st0 = stages_norm[0]
        steps, cfg_scale = st0["steps"], st0["cfg"]
        sampler_name, scheduler, denoise = st0["sampler_name"], st0["scheduler"], st0["denoise"]
        if n_stages > 1:
            print(f"[Ouroboros] refine chain — {n_stages} stages: " + " → ".join(
                f"{s['sampler_name']}/{s['scheduler']} {s['steps']}st d{s['denoise']}" for s in stages_norm))
        if model_negative is not None:
            # The dual guider short-circuits to a single model at cfg 1.0, so a second model wired in
            # at that cfg silently does nothing at all.
            flat = [s for s in stages_norm if abs(s["cfg"] - 1.0) < 1e-6]
            if flat:
                print("[Ouroboros] ⚠ model_negative is wired but "
                      f"{len(flat)} of {n_stages} stage(s) run at cfg 1.0 — the guider skips the "
                      "unconditional pass there, so the second model does nothing. Raise cfg above 1.")

        # Generation needs a valid latent. A broken critic does NOT block us — we still generate
        # and return the image(s), stopping the loop with a loud report on the first critic failure.
        if not (isinstance(latent, dict) and latent.get("samples") is not None):
            raise RuntimeError("[Ouroboros] No valid latent — wire an Empty Latent (matching your model) into 'latent'.")

        base_latent = latent
        try:
            _sh = base_latent["samples"].shape
            lat_h, lat_w = int(_sh[-2]) * 8, int(_sh[-1]) * 8
        except Exception:
            lat_h, lat_w = 0, 0
        clip_enc = CLIPTextEncode()
        vae_dec = VAEDecode()

        intent = (user_prompt or "").strip()
        triggers = (trigger_words or "").strip()
        current_prompt = intent
        current_negative = (negative or "").strip()
        last_advice = ""
        last_score = None       # previous iteration's overall score + per-criterion scores, fed to
        last_scores = None      # the enhancer so its feedback block can name the weakest criteria
        last_negative_add = None  # the critic's last flaw terms → 'avoid' hint for the enhancer
        history = []
        frames, used_prompts, jd_items, report = [], [], [], []
        captions, times, settings = [], [], []
        best = None
        low_vram = bool(low_vram)
        # low_vram auto-n_ctx state: the largest prompt AND output each role has actually used so far
        # (measured). Next load is sized from these; 0 → first load runs at the full ceiling to
        # establish a baseline.
        peak_enh = peak_crit = 0            # peak prompt tokens per role
        peak_enh_out = peak_crit_out = 0    # peak generated tokens per role
        enhancer_history = max(0, int(enhancer_history))
        hist_keep = max(6, enhancer_history)  # buffer must retain at least what the enhancer recaps
        n = max(1, int(max_iterations))
        pbar = ProgressBar(n) if ProgressBar else None
        _emit({"type": "start", "id": str(unique_id), "total": n, "ts": _now(), "mode": log_mode})

        for i in range(n):
            if mm.processing_interrupted():
                report.append(f"interrupted at iteration {i + 1}")
                break
            if stop.stop_requested(unique_id):
                stop.clear_stop(unique_id)
                report.append(f"⏹ stopped by user before iteration {i + 1} "
                              f"(returning {len(frames)} image(s) so far)")
                log(f"⏹ stopped by user — returning {len(frames)} image(s)")
                _emit({"type": "stopped", "id": str(unique_id), "i": i + 1, "ts": _now()})
                break

            # Per-stage seeds for this iteration (computed once here so seed_mode="random" draws once).
            stage_seeds = [_seed_for(s["seed"], s["seed_mode"], s["seed_step"], i, s["rng"])
                           for s in stages_norm]
            iter_seed = stage_seeds[0]  # stage-1 seed drives the existing per-iteration log lines

            # 1) Enhancer LLM → next prompt (iter 0: improve the intent; later: apply last advice).
            #    The enhancer's Local LLM Settings drives its persona (system_prompt) and context;
            #    we only force plain-text output. LoRA trigger words are then guaranteed-appended.
            enh_cfg = dict(enhancer_settings) if isinstance(enhancer_settings, dict) else {}
            enh_cfg["output_format"] = "text"
            enh_cfg["grammar"] = ""
            enh_cap = int(enh_cfg.get("n_ctx", 4096) or 4096)  # Settings-node n_ctx = the ceiling
            reviser_user = _reviser_user(intent, current_prompt, last_advice, history, enhancer_history,
                                         score=last_score, score_max=hi, scores=last_scores,
                                         avoid=last_negative_add)
            refined = current_prompt
            enh_stats = {}
            if low_vram and peak_enh > 0:  # size to what the enhancer has actually used, ≤ ceiling
                enh_cfg["n_ctx"] = _auto_nctx(peak_enh, peak_enh_out, int(enh_cfg.get("max_tokens", 512) or 512), enh_cap)

            # In 'streaming' mode the enhancer types its rewrite into the Live Log token by token.
            enh_token_cb = None
            if stream:
                def enh_token_cb(piece, _i=i):
                    _emit({"type": "prompt_delta", "id": str(unique_id), "i": _i + 1, "delta": piece})

            err, ctx = build_llm_request(enh_cfg, reviser_user)
            if err:
                report.append(f"#{i + 1} enhancer error: {err}")
            else:
                if stream:
                    ctx["req"]["stream_text"] = True
                    # max_tokens rides along so the log can count the streamed tokens against the
                    # enhancer's ceiling live ("142/512 tok") instead of waiting for the ctx line.
                    _emit({"type": "stage", "stage": "prompt", "open": True, "id": str(unique_id),
                           "i": i + 1, "total": n, "ts": _now(), "seed": iter_seed,
                           "max_tokens": int(ctx["max_tokens"])})
                out = _generate_and_format(
                    ctx["req"], ctx["load_sig"], ctx["max_tokens"],
                    low_vram, low_vram,   # free comfy + kill worker (before diffusion) in low_vram mode
                    ctx["directive"], ctx["strip_think"], ctx["answer_marker"], ctx["help"],
                    token_cb=enh_token_cb, show_progress=False, stats=enh_stats)
                t = out[0]
                if (low_vram and isinstance(t, str) and t.startswith("[ERROR]") and "context" in t.lower()
                        and int(enh_cfg.get("n_ctx", 0) or 0) < enh_cap):
                    enh_cfg["n_ctx"] = enh_cap  # auto-size under-shot → retry once at the full ceiling
                    e2, ctx = build_llm_request(enh_cfg, reviser_user)
                    if not e2:
                        if stream:
                            ctx["req"]["stream_text"] = True
                        out = _generate_and_format(
                            ctx["req"], ctx["load_sig"], ctx["max_tokens"], low_vram, low_vram,
                            ctx["directive"], ctx["strip_think"], ctx["answer_marker"], ctx["help"],
                            token_cb=enh_token_cb, show_progress=False, stats=enh_stats)
                        t = out[0]
                if isinstance(t, str) and t.strip() and not t.startswith("[ERROR]"):
                    refined = t.strip()
                elif isinstance(t, str) and t.startswith("[ERROR]"):
                    report.append(f"#{i + 1} enhancer: {t}")
            if low_vram:
                peak_enh = max(peak_enh, int(enh_stats.get("prompt_tokens", 0) or 0))
                peak_enh_out = max(peak_enh_out, int(enh_stats.get("output_tokens", 0) or 0))
            enh_ctx = _ctx_summary(enh_stats)
            refined = _append_triggers(refined or current_prompt, triggers)
            if full_console_log:
                flat = " ".join(refined.split())  # one-line for the console glance
                shown = flat if len(flat) <= 200 else flat[:200] + "…"
                log(f"iter {i + 1}/{n}  seed={iter_seed}  prompt: {shown}")
                if enh_ctx:
                    log(f"    ⓘ {_ctx_line(enh_ctx, 'enhancer')}")
            if live:  # stage 1 done — post the enhanced prompt immediately
                _emit({"type": "stage", "stage": "prompt", "id": str(unique_id), "i": i + 1,
                       "total": n, "ts": _now(), "seed": iter_seed, "prompt": refined, "ctx": enh_ctx})

            # 2) Encode → 3) sample (one pass per stage, chaining latents) → 4) decode. Timed as a
            #    whole — this is the "generation time". In refine mode every stage runs each iteration;
            #    only the FINAL latent is decoded and shown to the critic.
            gen_t0 = time.perf_counter()
            pos = clip_enc.encode(clip, refined)[0]
            neg = clip_enc.encode(clip, current_negative)[0]
            latent_cur = {"samples": base_latent["samples"].clone()}
            for si, stg in enumerate(stages_norm):
                # With a second (unconditional) model, the dual guider is rebuilt HERE — after the
                # rewrite has been encoded — so every iteration judges the prompt it actually used.
                g = (build_dual_guider(model, model_negative, pos, neg, stg["cfg"])
                     if model_negative is not None else None)
                latent_cur = _sample_stage(model, latent_cur, pos, neg, stg, stage_seeds[si], guider=g)
            latent_out = latent_cur
            image = vae_dec.decode(vae, latent_out)[0]
            gen_seconds = time.perf_counter() - gen_t0
            thumb = _thumb_b64(image)  # encode once; reused by whichever event carries the image
            if live:  # stage 2 done — post the image the moment it's decoded
                _emit({"type": "stage", "stage": "image", "id": str(unique_id), "i": i + 1,
                       "total": n, "ts": _now(), "gen_seconds": round(gen_seconds, 2), "thumb": thumb})

            # Keep the generated frame immediately — generation is independent of the critic, so a
            # critic failure never costs us the image. The score field is filled in below.
            frames.append(image)
            used_prompts.append(refined)
            captions.append(f"Iteration {i + 1}")
            times.append(f"{gen_seconds:.2f} s")
            # iter_settings[1] MUST stay the score entry — it's patched in-place below.
            iter_settings = [
                {"key": "Ouroboros.iteration", "value": str(i + 1)},
                {"key": "Ouroboros.score", "value": "pending"},
                {"key": "Ouroboros.size", "value": f"{lat_w}x{lat_h}"},
            ]
            if n_stages == 1:
                iter_settings += [
                    {"key": "Ouroboros.seed", "value": str(stage_seeds[0])},
                    {"key": "Ouroboros.steps", "value": str(st0["steps"])},
                    {"key": "Ouroboros.cfg", "value": str(st0["cfg"])},
                    {"key": "Ouroboros.sampler", "value": f"{st0['sampler_name']} / {st0['scheduler']}"},
                    {"key": "Ouroboros.denoise", "value": str(st0["denoise"])},
                ]
            else:
                for si, stg in enumerate(stages_norm):
                    iter_settings.append({"key": f"Ouroboros.stage{si + 1}", "value":
                        f"{stg['sampler_name']}/{stg['scheduler']} · {stg['steps']}st · "
                        f"cfg {stg['cfg']} · denoise {stg['denoise']} · seed {stage_seeds[si]}"})
            settings.append(iter_settings)

            # 5) Critic (with optional self-consistency samples).
            verdicts = []
            crit_err = ""
            crit_ctx = None
            # Optional downscale of the image the critic sees: fewer image tokens → faster judging,
            # at some loss of fine detail. Applied as an effective image_max_side derived from the
            # image's real longest side, never upscaling past whatever the vision config already caps.
            crit_max_side = None
            if crit_downscale > 1.0:
                long_side = max(lat_w, lat_h)
                if long_side <= 0:
                    try:
                        _ish = image.shape
                        long_side = max(int(_ish[-2]), int(_ish[-3]))
                    except Exception:
                        long_side = 0
                if long_side > 0:
                    crit_max_side = max(64, int(round(long_side / crit_downscale)))
            for s in range(samples):
                crit_cfg = dict(vcfg)
                crit_cfg["context"] = ""
                if crit_max_side is not None:
                    _cap = int(crit_cfg.get("image_max_side", 1024) or 0)
                    crit_cfg["image_max_side"] = min(_cap, crit_max_side) if _cap > 0 else crit_max_side
                crit_cfg["max_tokens"] = max(int(crit_cfg.get("max_tokens", 512) or 512), 256 + 24 * len(keys))
                crit_cap = int(crit_cfg.get("n_ctx", 4096) or 4096)  # Settings-node n_ctx = the ceiling
                if low_vram and peak_crit > 0:  # measured prompt includes the image tokens (grammar → exact usage)
                    crit_cfg["n_ctx"] = _auto_nctx(peak_crit, peak_crit_out, int(crit_cfg["max_tokens"]), crit_cap)
                cu = _critic_user(rubric, crit, keys, lo, hi, refined, advice_style)
                cerr, cctx = build_llm_request(crit_cfg, cu, image=image,
                                               system_override=critic_system,
                                               grammar_override=critic_grammar)
                if cerr:
                    crit_err = cerr
                    report.append(f"#{i + 1} critic error: {cerr}")
                    continue
                crit_stats = {}
                cout = _generate_and_format(
                    cctx["req"], cctx["load_sig"], cctx["max_tokens"],
                    low_vram and s == 0,               # unload comfy once, before the first sample
                    low_vram and s == samples - 1,     # kill worker after the last sample (before next diffusion)
                    cctx["directive"], cctx["strip_think"], cctx["answer_marker"], cctx["help"],
                    show_progress=False, stats=crit_stats)
                ctext = cout[0]
                if (low_vram and isinstance(ctext, str) and ctext.startswith("[ERROR]")
                        and "context" in ctext.lower() and int(crit_cfg.get("n_ctx", 0) or 0) < crit_cap):
                    crit_cfg["n_ctx"] = crit_cap  # auto-size under-shot → retry this sample at the full ceiling
                    cerr2, cctx = build_llm_request(crit_cfg, cu, image=image,
                                                    system_override=critic_system,
                                                    grammar_override=critic_grammar)
                    if not cerr2:
                        cout = _generate_and_format(
                            cctx["req"], cctx["load_sig"], cctx["max_tokens"],
                            low_vram and s == 0, low_vram and s == samples - 1,
                            cctx["directive"], cctx["strip_think"], cctx["answer_marker"], cctx["help"],
                            show_progress=False, stats=crit_stats)
                        ctext = cout[0]
                crit_ctx = _ctx_summary(crit_stats) or crit_ctx  # keep the latest sample's fill
                if low_vram:
                    peak_crit = max(peak_crit, int(crit_stats.get("prompt_tokens", 0) or 0))
                    peak_crit_out = max(peak_crit_out, int(crit_stats.get("output_tokens", 0) or 0))
                v = (_parse_critic(ctext, lo, hi, keys)
                     if isinstance(ctext, str) and not ctext.startswith("[ERROR]") else None)
                if v:
                    verdicts.append(v)
                elif isinstance(ctext, str) and ctext.startswith("[ERROR]"):
                    crit_err = ctext
                    report.append(f"#{i + 1} critic: {ctext}")

            # No verdict this round → there's no scoring signal to optimize on, so STOP the loop.
            # But we keep the frame we already generated (above) and return everything so far; the
            # failure is surfaced loudly in `report` + console. The run does NOT hard-fail — an
            # image is still worth returning.
            if not verdicts:
                _shutdown_worker()  # free the (possibly half-loaded) critic worker's VRAM
                msg = (f"critic could not judge iteration {i + 1} — stopping the loop (no scoring "
                       f"signal). It needs a VISION GGUF + a COMPATIBLE mmproj (same model family, "
                       f"e.g. a gemma3 mmproj with a gemma3 model). Last error: {crit_err or 'see console'}")
                print(f"[Ouroboros] {msg}")
                report.append(f"#{i + 1} ⚠ {msg}")
                if enh_ctx:
                    report.append(f"    ⓘ {_ctx_line(enh_ctx, 'enhancer')}")
                captions[-1] = f"Iteration {i + 1} (critic failed)"
                iter_settings[1]["value"] = "critic failed"
                if best is None:
                    best = {"score": None, "prompt": refined, "negative": current_negative, "image": image}
                _emit({"type": "error", "id": str(unique_id), "i": i + 1, "total": n, "ts": _now(),
                       "seed": iter_seed, "message": msg, "prompt": refined,
                       "gen_seconds": round(gen_seconds, 2), "ctx": enh_ctx,
                       "thumb": ("" if live else thumb)})  # in live mode the image stage already sent it
                break

            verdict = _aggregate_verdicts(verdicts, keys, lo, hi)
            score = verdict["score"]
            iter_settings[1]["value"] = f"{score} / {hi}"
            jd_items.append({"index": i, "score": score, "score_max": hi,
                             "tags": verdict["negative_add"], "comment": verdict["advice"],
                             "scores": verdict["scores"]})
            is_best = best is None or best["score"] is None or score > best["score"]
            if is_best:
                best = {"score": score, "prompt": refined, "negative": current_negative, "image": image}
            brk = " ·".join(f" {k} {verdict['scores'][k]}" for k in keys) if keys else ""
            # Console (live): always a one-line score; full_console_log adds advice + negatives.
            log(f"iter {i + 1}/{n}  seed={iter_seed}  ★{score}/{hi}{brk}  {gen_seconds:.1f}s"
                + ("  ← best" if is_best else ""))
            if full_console_log and verdict["advice"]:
                log(f"    ↳ advice: {verdict['advice']}")
            if full_console_log and verdict["negative_add"]:
                log(f"    ⊖ negative += {', '.join(verdict['negative_add'])}")
            if full_console_log and crit_ctx:
                log(f"    ⓘ {_ctx_line(crit_ctx, 'critic')}")
            # report (full log): score line + seed/time, then advice / negative on their own lines.
            report.append(f"#{i + 1}  seed {iter_seed}  ★{score}/{hi}{brk}  ({gen_seconds:.1f}s)"
                          + ("  ← best" if is_best else ""))
            if verdict["advice"]:
                report.append(f"    ↳ {verdict['advice']}")
            if verdict["negative_add"]:
                report.append(f"    ⊖ negative += {', '.join(verdict['negative_add'])}")
            for _lbl, _c in (("enhancer", enh_ctx), ("critic", crit_ctx)):
                if _c:
                    report.append(f"    ⓘ {_ctx_line(_c, _lbl)}")
            # Live log event. Per-step: post just the verdict now (prompt+image already sent above).
            # Per-iteration: post one combined entry with the full prompt + thumbnail.
            if live:
                _emit({"type": "stage", "stage": "verdict", "id": str(unique_id), "i": i + 1,
                       "total": n, "ts": _now(), "score": score, "score_max": hi,
                       "scores": verdict["scores"], "advice": verdict["advice"],
                       "negative_add": verdict["negative_add"], "is_best": is_best, "ctx": crit_ctx})
            else:
                _emit({"type": "iteration", "id": str(unique_id), "i": i + 1, "total": n,
                       "ts": _now(), "seed": iter_seed, "score": score, "score_max": hi,
                       "scores": verdict["scores"], "advice": verdict["advice"],
                       "negative_add": verdict["negative_add"], "prompt": refined,
                       "gen_seconds": round(gen_seconds, 2), "is_best": is_best, "thumb": thumb,
                       "ctx": enh_ctx, "ctx_crit": crit_ctx})
            if pbar:
                try:
                    pbar.update_absolute(i + 1)
                except Exception:
                    pass

            if score >= target:
                report.append(f"→ target {target} reached at iteration {i + 1}")
                break

            # Prep next iteration.
            current_negative = _merge_negative(current_negative, verdict["negative_add"])
            last_advice = verdict["advice"]
            last_score = score
            last_scores = verdict["scores"]
            last_negative_add = verdict["negative_add"]
            current_prompt = refined
            history.append({"prompt": refined, "score": score, "advice": verdict["advice"]})
            history = history[-hist_keep:]

        if not frames:
            return self._abort("No iterations completed.")

        images = torch.cat(frames, dim=0)
        prompts_out = "\n---\n".join(used_prompts)
        judge_data = json.dumps(jd_items, ensure_ascii=False)
        best_image = best["image"] if best else frames[0]
        best_prompt = best["prompt"] if best else current_prompt
        best_score = float(best["score"]) if best and best["score"] is not None else 0.0
        _emit({"type": "done", "id": str(unique_id), "iterations": len(frames),
               "best_score": best_score, "best_prompt": best_prompt, "ts": _now()})
        log(f"done — {len(frames)} iteration(s), best ★{best_score}/{hi}")
        return (images, prompts_out, judge_data, best_image, best_prompt, best_score,
                "\n".join(report), len(frames),
                "\n".join(captions), "\n".join(times),
                json.dumps(settings, ensure_ascii=False))

    @staticmethod
    def _abort(msg):
        print(f"[Ouroboros] {msg}")
        black = torch.zeros((1, 64, 64, 3))
        return (black, "", "[]", black, "", 0.0, f"[ERROR] {msg}", 0, "", "", "[]")


class KinburgOuroborosLog:
    """UI-only live log for the Ouroboros loop. It has no inputs/outputs and never runs on the
    backend — the whole display lives in web/ouroboros_log.js, which listens for the
    `kinburg.ouroboros` websocket events the Ouroboros node emits (start / iteration / error /
    stopped / done) and renders each iteration's thumbnail + score + prompt + advice live."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "noop"
    CATEGORY = "Kinburg-Nodes/sampling"
    OUTPUT_NODE = False
    DESCRIPTION = ("Live log for the Ouroboros sampler: shows each iteration's thumbnail, seed, "
                   "score, prompt and advice as the loop runs. Drop it anywhere on the canvas — "
                   "it needs no connections.")

    def noop(self):
        return ()


NODE_CLASS_MAPPINGS = {
    "KinburgOuroboros": KinburgOuroboros,
    "KinburgCriticSettings": KinburgCriticSettings,
    "KinburgSamplerSettings": KinburgSamplerSettings,
    "KinburgOuroborosLog": KinburgOuroborosLog,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KinburgOuroboros": "Ouroboros (Self-Correcting Sampler) 🐍",
    "KinburgCriticSettings": "Critic Settings (GGUF)",
    "KinburgSamplerSettings": "Sampler Settings",
    "KinburgOuroborosLog": "Ouroboros Live Log 🐍📜",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
