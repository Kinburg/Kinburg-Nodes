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
      - **Sampler Settings** → SAMPLER_CFG
      - **Critic Settings (GGUF)** → CRITIC (embeds a vision LLM_CONFIG + the evaluation rules)
      - the expander is a plain Local LLM Settings (GGUF) `LLM_CONFIG`.
  * **Fixed seed** + a fixed start latent so only the prompt (and accumulated negative) varies —
    a clean optimization signal. Two LLM configs (expander + critic); on low VRAM the same vision
    model file should back both so the worker stays warm (no reload). VRAM discipline: when a
    config's `unload_comfy_models` is on, we free ComfyUI models before each LLM call AND kill the
    worker after it (before the next diffusion) — strictly one model resident at a time.

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
    "You are an expert image-prompt engineer. You rewrite prompts to maximize image quality while "
    "staying faithful to the user's original intent. Output ONLY the final prompt text — no "
    "explanations, no quotes, no markdown, no preamble."
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


LOG_MODES = ["per step", "per iteration"]


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


def _reviser_user(intent, current, advice, history):
    """The message fed to the expander LLM to produce the next prompt."""
    lines = [f"Original intent (stay faithful to this): {intent}"]
    if current and current.strip() != intent.strip():
        lines.append(f"Current prompt: {current}")
    if advice and advice.strip():
        lines.append(f"Judge feedback to address: {advice.strip()}")
    if history:
        tried = "; ".join(f"(score {h['score']}) {h['prompt'][:120]}" for h in history[-4:])
        lines.append(f"Already tried (do not repeat these): {tried}")
    if advice and advice.strip():
        lines.append("Rewrite the image-generation prompt to address the feedback while staying "
                     "faithful to the original intent. Output ONLY the new prompt, no preamble.")
    else:
        lines.append("Improve and expand this into a strong, detailed image-generation prompt, "
                     "staying faithful to the intent. Output ONLY the prompt, no preamble.")
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
            }
        }

    RETURN_TYPES = (SAMPLER_CFG,)
    RETURN_NAMES = ("sampler_settings",)
    FUNCTION = "build"
    CATEGORY = "Kinburg-Nodes/sampling"
    DESCRIPTION = ("KSampler settings bundle for the Ouroboros self-correcting sampler. "
                   "Image dimensions come from the latent wired into Ouroboros.")

    def build(self, seed, steps, cfg, sampler_name, scheduler, denoise, seed_mode="fixed", seed_step=1):
        return ({"seed": int(seed), "steps": int(steps), "cfg": float(cfg),
                 "sampler_name": sampler_name, "scheduler": scheduler, "denoise": float(denoise),
                 "seed_mode": seed_mode, "seed_step": int(seed_step)},)


class KinburgCriticSettings:
    """Bundle the loop critic's model + evaluation rules into a CRITIC. Embeds a vision LLM_CONFIG
    (a Local LLM Settings (GGUF) WITH a Vision Settings mmproj)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (LLM_CONFIG, {"tooltip": "A 'Local LLM Settings (GGUF)' WITH a 'Vision Settings (GGUF)' (mmproj) — the critic needs vision."}),
                "criteria": ("STRING", {"multiline": True, "default": DEFAULT_CRITERIA, "tooltip": "Criteria the critic scores each image on (one per line, 'name: description'). Pre-filled; clear for a single overall score."}),
                "target_score": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "The loop stops once the overall score reaches this."}),
                "score_min": ("INT", {"default": 1, "min": 0, "max": 100}),
                "score_max": ("INT", {"default": 5, "min": 1, "max": 100}),
            },
            "optional": {
                "rubric": ("STRING", {"multiline": True, "default": "Strictly and critically evaluate the generated image.", "tooltip": "Free-text guidance / tone for the critic."}),
                "system_prompt": ("STRING", {"multiline": True, "default": CRITIC_SYSTEM, "tooltip": "The critic's persona (system prompt). Default pre-filled; blank → built-in default."}),
                "advice_style": ("STRING", {"default": DEFAULT_ADVICE_STYLE, "tooltip": "How the advice should read, e.g. 'a detailed paragraph' or 'one short sentence'."}),
                "samples": ("INT", {"default": 1, "min": 1, "max": 9, "tooltip": "Self-consistency: judge each image N times and take the median score (N× critic cost)."}),
            },
        }

    RETURN_TYPES = (CRITIC,)
    RETURN_NAMES = ("critic_settings",)
    FUNCTION = "build"
    CATEGORY = "Kinburg-Nodes/LLM"
    DESCRIPTION = "Critic (vision judge) settings bundle for the Ouroboros loop — model + scoring rules + target."

    def build(self, config, criteria, target_score, score_min, score_max,
              rubric="", system_prompt="", advice_style="", samples=1):
        return ({
            "config": config,
            "criteria": criteria or "",
            "rubric": rubric or "",
            "target_score": float(target_score),
            "score_min": int(score_min),
            "score_max": int(score_max),
            "system_prompt": (system_prompt or "").strip() or CRITIC_SYSTEM,
            "advice_style": (advice_style or "").strip() or DEFAULT_ADVICE_STYLE,
            "samples": max(1, int(samples)),
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
                "critic_settings": (CRITIC, {"tooltip": "A 'Critic Settings (GGUF)' bundle — the vision judge + scoring rules + target."}),
                "sampler_settings": (SAMPLER_CFG, {"tooltip": "A 'Sampler Settings' bundle."}),
                "max_iterations": ("INT", {"default": 6, "min": 1, "max": 100, "tooltip": "Hard cap on refinement rounds."}),
            },
            "optional": {
                "trigger_words": ("STRING", {"forceInput": True, "tooltip": "Comma-separated words ALWAYS appended to the enhanced prompt (e.g. LoRA triggers from Lora Unlim Accumulator's 'triggers' output), so the LLM rewrite can't drop them."}),
                "full_console_log": ("BOOLEAN", {"default": True, "tooltip": "Console/terminal verbosity only (not the Live Log node or 'report'). On: per iteration also prints the enhanced prompt, advice and negative additions. Off: just a one-line score per iteration. The full trace is always in the 'report' output and the Live Log node."}),
                "log_mode": (LOG_MODES, {"default": "per step", "tooltip": "How the Ouroboros Live Log node updates: 'per step' posts each stage the moment it finishes (enhanced prompt → generated image → critic verdict), each timestamped; 'per iteration' posts one combined entry after the whole iteration (the older behaviour)."}),
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
            critic_settings, sampler_settings, max_iterations, trigger_words="", full_console_log=True,
            log_mode="per step", unique_id=None):
        import comfy.model_management as mm
        from nodes import common_ksampler, CLIPTextEncode, VAEDecode
        stop.clear_stop(unique_id)  # drop any stale stop request from a previous run
        live = (log_mode == "per step")  # emit each stage as it finishes vs one entry per iteration
        log = lambda m: print(f"[Ouroboros {_now()}] {m}")
        try:
            from comfy.utils import ProgressBar
        except Exception:
            ProgressBar = None

        cs = critic_settings if isinstance(critic_settings, dict) else {}
        vcfg = cs.get("config") or {}
        rubric = cs.get("rubric", "")
        target = float(cs.get("target_score", 4.0))
        lo = int(cs.get("score_min", 1))
        hi = int(cs.get("score_max", 5))
        if hi < lo:
            lo, hi = hi, lo
        critic_system = cs.get("system_prompt") or CRITIC_SYSTEM
        advice_style = cs.get("advice_style") or DEFAULT_ADVICE_STYLE
        samples = max(1, int(cs.get("samples", 1)))

        crit = _parse_criteria(cs.get("criteria", ""))
        keys = [k for (k, _l, _d) in crit]
        critic_grammar = _build_critic_grammar(keys)

        sp = sampler_settings if isinstance(sampler_settings, dict) else {}
        base_seed = int(sp.get("seed", 0))
        steps = int(sp.get("steps", 20))
        cfg_scale = float(sp.get("cfg", 7.0))
        sampler_name = sp.get("sampler_name", "euler")
        scheduler = sp.get("scheduler", "normal")
        denoise = float(sp.get("denoise", 1.0))
        seed_mode = sp.get("seed_mode", "fixed")
        seed_step = int(sp.get("seed_step", 1))
        seed_rng = random.Random(base_seed)  # reproducible per-run source for seed_mode="random"

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
        history = []
        frames, used_prompts, jd_items, report = [], [], [], []
        captions, times, settings = [], [], []
        best = None
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

            iter_seed = _seed_for(base_seed, seed_mode, seed_step, i, seed_rng)

            # 1) Enhancer LLM → next prompt (iter 0: improve the intent; later: apply last advice).
            #    The enhancer's Local LLM Settings drives its persona (system_prompt) and context;
            #    we only force plain-text output. LoRA trigger words are then guaranteed-appended.
            enh_cfg = dict(enhancer_settings) if isinstance(enhancer_settings, dict) else {}
            enh_cfg["output_format"] = "text"
            enh_cfg["grammar"] = ""
            reviser_user = _reviser_user(intent, current_prompt, last_advice, history)
            refined = current_prompt
            err, ctx = build_llm_request(enh_cfg, reviser_user)
            if err:
                report.append(f"#{i + 1} enhancer error: {err}")
            else:
                out = _generate_and_format(
                    ctx["req"], ctx["load_sig"], ctx["max_tokens"],
                    ctx["unload_comfy"], ctx["unload_comfy"],   # free comfy + kill worker (before diffusion) on low VRAM
                    ctx["directive"], ctx["strip_think"], ctx["answer_marker"], ctx["help"],
                    show_progress=False)
                t = out[0]
                if isinstance(t, str) and t.strip() and not t.startswith("[ERROR]"):
                    refined = t.strip()
                elif isinstance(t, str) and t.startswith("[ERROR]"):
                    report.append(f"#{i + 1} enhancer: {t}")
            refined = _append_triggers(refined or current_prompt, triggers)
            if full_console_log:
                flat = " ".join(refined.split())  # one-line for the console glance
                shown = flat if len(flat) <= 200 else flat[:200] + "…"
                log(f"iter {i + 1}/{n}  seed={iter_seed}  prompt: {shown}")
            if live:  # stage 1 done — post the enhanced prompt immediately
                _emit({"type": "stage", "stage": "prompt", "id": str(unique_id), "i": i + 1,
                       "total": n, "ts": _now(), "seed": iter_seed, "prompt": refined})

            # 2) Encode → 3) sample → 4) decode (timed — this is the "generation time").
            gen_t0 = time.perf_counter()
            pos = clip_enc.encode(clip, refined)[0]
            neg = clip_enc.encode(clip, current_negative)[0]
            latent_in = {"samples": base_latent["samples"].clone()}
            latent_out = common_ksampler(model, iter_seed, steps, cfg_scale, sampler_name, scheduler,
                                         pos, neg, latent_in, denoise=denoise)[0]
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
            iter_settings = [
                {"key": "Ouroboros.iteration", "value": str(i + 1)},
                {"key": "Ouroboros.score", "value": "pending"},
                {"key": "Ouroboros.seed", "value": str(iter_seed)},
                {"key": "Ouroboros.steps", "value": str(steps)},
                {"key": "Ouroboros.cfg", "value": str(cfg_scale)},
                {"key": "Ouroboros.sampler", "value": f"{sampler_name} / {scheduler}"},
                {"key": "Ouroboros.denoise", "value": str(denoise)},
                {"key": "Ouroboros.size", "value": f"{lat_w}x{lat_h}"},
            ]
            settings.append(iter_settings)

            # 5) Critic (with optional self-consistency samples).
            verdicts = []
            crit_err = ""
            for s in range(samples):
                crit_cfg = dict(vcfg)
                crit_cfg["context"] = ""
                crit_cfg["max_tokens"] = max(int(crit_cfg.get("max_tokens", 512) or 512), 256 + 24 * len(keys))
                cu = _critic_user(rubric, crit, keys, lo, hi, refined, advice_style)
                cerr, cctx = build_llm_request(crit_cfg, cu, image=image,
                                               system_override=critic_system,
                                               grammar_override=critic_grammar)
                if cerr:
                    crit_err = cerr
                    report.append(f"#{i + 1} critic error: {cerr}")
                    continue
                cout = _generate_and_format(
                    cctx["req"], cctx["load_sig"], cctx["max_tokens"],
                    cctx["unload_comfy"] and s == 0,               # unload comfy once, before the first sample
                    cctx["unload_comfy"] and s == samples - 1,     # kill worker after the last sample (before next diffusion)
                    cctx["directive"], cctx["strip_think"], cctx["answer_marker"], cctx["help"],
                    show_progress=False)
                ctext = cout[0]
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
                captions[-1] = f"Iteration {i + 1} (critic failed)"
                iter_settings[1]["value"] = "critic failed"
                if best is None:
                    best = {"score": None, "prompt": refined, "negative": current_negative, "image": image}
                _emit({"type": "error", "id": str(unique_id), "i": i + 1, "total": n, "ts": _now(),
                       "seed": iter_seed, "message": msg, "prompt": refined,
                       "gen_seconds": round(gen_seconds, 2),
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
            # report (full log): score line + seed/time, then advice / negative on their own lines.
            report.append(f"#{i + 1}  seed {iter_seed}  ★{score}/{hi}{brk}  ({gen_seconds:.1f}s)"
                          + ("  ← best" if is_best else ""))
            if verdict["advice"]:
                report.append(f"    ↳ {verdict['advice']}")
            if verdict["negative_add"]:
                report.append(f"    ⊖ negative += {', '.join(verdict['negative_add'])}")
            # Live log event. Per-step: post just the verdict now (prompt+image already sent above).
            # Per-iteration: post one combined entry with the full prompt + thumbnail.
            if live:
                _emit({"type": "stage", "stage": "verdict", "id": str(unique_id), "i": i + 1,
                       "total": n, "ts": _now(), "score": score, "score_max": hi,
                       "scores": verdict["scores"], "advice": verdict["advice"],
                       "negative_add": verdict["negative_add"], "is_best": is_best})
            else:
                _emit({"type": "iteration", "id": str(unique_id), "i": i + 1, "total": n,
                       "ts": _now(), "seed": iter_seed, "score": score, "score_max": hi,
                       "scores": verdict["scores"], "advice": verdict["advice"],
                       "negative_add": verdict["negative_add"], "prompt": refined,
                       "gen_seconds": round(gen_seconds, 2), "is_best": is_best, "thumb": thumb})
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
            current_prompt = refined
            history.append({"prompt": refined, "score": score, "advice": verdict["advice"]})
            history = history[-6:]

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
