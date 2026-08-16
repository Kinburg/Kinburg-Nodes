"""Siren — a section-aware sampler for AceStep music latents (she sings; you pick the bars).

An AceStep audio latent is a **one-dimensional strip of time**: `[B, 64, T]`, one frame per 40 ms
(the 1.5 VAE upsamples a frame to 1920 samples at 48 kHz — 25 frames per second). That single fact is
what a generic latent sampler can't exploit and what these two nodes are for: if you can name a
stretch of the strip in *seconds*, you can regenerate only that stretch, or append to it, and leave
the rest of the take alone.

Two nodes, one job each:

  * **Siren Section** turns "from 0:47.5 to 1:02, snapped to bars, with a 0.35 s crossfade" into a
    denoise mask on the latent, and — in `extend` mode — grows the latent along the time axis first.
    It writes the mask into the latent's own `noise_mask`, which is ComfyUI's standard key, so the
    result also works with the stock samplers; Siren is not required to use it. Chain several to mark
    several sections at once.
  * **Siren (Music Sampler)** samples it, with the staged handling from Chimera (a chain of *Sampler
    Settings* = consecutive stages, so cfg can be high early and low late) but none of Chimera's
    split machinery — each stage simply runs the steps it declares.

Why the mask math is safe: ComfyUI's masked path (`KSamplerX0Inpaint`) re-pins the frozen frames
every step with `sigma * noise + (1 - sigma) * original`, which for a flow model like AceStep is the
exact forward interpolation — so the untouched audio stays consistent with the noise level the
sampler is working at, and no boundary artifacts are introduced. `reshape_mask` handles the 1-D case
(linear interpolation), so nothing here needs custom sampling code.

The one rule that follows from that math: **a masked run cannot use Chimera's continuous handoff.**
A continuing stage passes zero noise and inherits the *in-flight* latent as its `latent_image`, so the
frozen region would be re-pinned to a partially-denoised reference with no noise term — correct at the
handoff step and drifting after it. With a section present, every stage therefore runs its own
schedule and its own noise (each stage finishes at sigma≈0, where the reference IS the clean latent).
Without a section, stages run continuous, exactly like Chimera.
"""
import json
import time

import torch

from ..chimera.nodes import (
    _build_sigmas, _flatten_cfg, _norm_stage, _sample_slice, _split_counts, _stage_label,
    _tail_for_scheduler,
)
from ..ouroboros.nodes import SAMPLER_CFG, SOLVER_TYPES
from ..timer.timer_nodes import _format_elapsed
from ..categories import CAT_SIREN

SIREN_SECTION = "KINBURG_SIREN_SECTION"
SECTION_MODES = ["retake", "extend"]
EXTEND_SIDES = ["end", "start"]
SNAP_MODES = ["off", "beat", "bar"]

# AceStep 1.5: the VAE turns one latent frame into 1920 samples at 48 kHz (Oobleck strides
# 2·4·4·6·10, comfy/sd.py) — exactly 40 ms, so 25 latent frames per second. `ACEAudio15`'s
# `temporal_downscale_ratio` of 1764 is the same 40 ms expressed at 44.1 kHz and is only used to
# rescale empty latents authored for another model, so it is NOT the number to divide by here.
# A wired VAE always wins over this default: AceStep 1.0 runs at a different frame rate.
DEFAULT_FPS = 25.0


# ------------------------------------------------------------------------------------- formatting
def _mmss(sec):
    """m:ss.s — the unit you actually think in when pointing at a spot in a track."""
    sec = max(0.0, float(sec))
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def _bars_at(sec, bpm, beats_per_bar, origin=0.0):
    """Position in bars, for the report. None when there's no tempo to measure against."""
    if bpm <= 0 or beats_per_bar <= 0:
        return None
    bar = (60.0 / float(bpm)) * int(beats_per_bar)
    return (float(sec) - float(origin)) / bar


# ------------------------------------------------------------------------------------------ time
def _fps_from_vae(vae):
    """Latent frames per second straight from the VAE: sample rate ÷ samples per latent frame.
    Only the audio VAEs carry both as plain numbers (the image/video ones put tuples or lambdas in
    `upscale_ratio`), so anything else returns None and the caller falls back to the widget."""
    if vae is None:
        return None
    try:
        sr = getattr(vae, "audio_sample_rate", None)
        ratio = getattr(vae, "upscale_ratio", None)
        if isinstance(sr, (int, float)) and isinstance(ratio, (int, float)) and sr > 0 and ratio > 0:
            return float(sr) / float(ratio)
    except Exception:
        pass
    return None


def _grid_seconds(snap, bpm, beats_per_bar):
    """Length of one snap unit in seconds, or None when snapping is off / there's no tempo."""
    if snap == "off" or float(bpm) <= 0:
        return None
    beat = 60.0 / float(bpm)
    return beat * max(1, int(beats_per_bar)) if snap == "bar" else beat


def _snap_sec(sec, grid, origin):
    """Nearest grid line at or after 0. `origin` moves the grid for tracks with a pickup intro."""
    if grid is None or grid <= 0:
        return max(0.0, float(sec))
    return max(0.0, float(origin) + round((float(sec) - float(origin)) / grid) * grid)


# ------------------------------------------------------------------------------------------ mask
def _build_mask(windows, frames, fps, samples):
    """The denoise mask for `windows` over a `frames`-long latent.

    ComfyUI's convention (samplers.py, `KSamplerX0Inpaint`): **1 = this frame may change, 0 = pinned
    to the input latent**. Values in between are a real crossfade in latent space, which is what
    `fade` buys — a hard mask edge is a visible seam in an image but an audible CLICK in audio.

    The ramp sits OUTSIDE the window, so the range you asked for is rewritten in full and the seam is
    spread over the neighbouring audio instead.

    Shape matters: `comfy.utils.reshape_mask` does not reshape its input in the 1-D-latent branch, so
    the tensor has to arrive as [N, C, T] already. For a 4-D latent (AceStep 1.0, `[B, 8, 16, T]`) the
    same time profile is repeated across the other axis."""
    prof = torch.zeros(frames, dtype=torch.float32)
    for w in windows:
        i0 = max(0, min(frames, int(round(float(w["start"]) * fps))))
        i1 = max(i0, min(frames, int(round(float(w["end"]) * fps))))
        cur = torch.zeros(frames, dtype=torch.float32)
        if i1 > i0:
            cur[i0:i1] = 1.0
        f = max(0, int(round(float(w.get("fade", 0.0)) * fps)))
        if f > 0 and i1 > i0:
            lo = max(0, i0 - f)
            if i0 > lo:   # strictly-between values: 0 and 1 already belong to the neighbours
                cur[lo:i0] = torch.linspace(0.0, 1.0, i0 - lo + 2)[1:-1]
            hi = min(frames, i1 + f)
            if hi > i1:
                cur[i1:hi] = torch.linspace(1.0, 0.0, hi - i1 + 2)[1:-1]
        prof = torch.maximum(prof, cur)
    if samples.ndim >= 4:
        return prof.view(1, 1, 1, frames).repeat(1, 1, int(samples.shape[-2]), 1)
    return prof.view(1, 1, frames)


def _pad_latent(samples, frames):
    """Latent content for a stretch added by `extend`. Zeros: at denoise 1.0 (the only sensible
    setting for extend — there is nothing there to preserve) the schedule starts from pure noise and
    this value provably cannot matter, since a flow model's initial x is `sigma*noise +
    (1-sigma)*latent` with sigma = 1. At denoise < 1 it does leak in, which the sampler reports as a
    warning rather than silently papering over. (AceStep ships its own silence latent —
    `comfy.ldm.ace.ace_step15.get_silence_latent` — if a better prior is ever wanted here.)"""
    shape = list(samples.shape)
    shape[-1] = int(frames)
    return torch.zeros(shape, dtype=samples.dtype, device=samples.device)


# --------------------------------------------------------------------------------- section node
class KinburgSirenSection:
    """Mark a stretch of an audio latent as the part to (re)generate, in seconds or bars."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The audio latent to mark up. For a RETAKE this is the latent of a take you already have (decode it once to hear where the section you want to replace actually sits); for EXTEND it is the take to grow."}),
                "mode": (SECTION_MODES, {"default": "retake", "tooltip": "• retake — mark the stretch between 'start_sec' and 'end_sec' as free to regenerate; everything else is frozen. The latent's length does not change.\n\n• extend — GROW the latent by 'extend_sec' seconds and mark only the new part as free. The existing take is frozen but still visible to the model (attention runs over the whole strip), so the new part is written to follow on from it."}),
                "start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 0.01, "tooltip": "'retake' only: where the section starts, in seconds from the top of the track."}),
                "end_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 0.01, "tooltip": "'retake' only: where the section ends, in seconds. 0 = run to the end of the track (handy for 'redo everything from the last chorus on')."}),
                "extend_sec": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 2000.0, "step": 0.01, "tooltip": "'extend' only: how many seconds to add.\n\nIMPORTANT: the 'duration' you gave TextEncodeAceStepAudio1.5 describes the WHOLE track and is baked into the tokens, so after extending you should raise it to the new total and re-encode — otherwise the model is being told the song is shorter than the strip it is writing on. The sampler prints the new length so you have the number to enter."}),
                "extend_at": (EXTEND_SIDES, {"default": "end", "tooltip": "'extend' only: which end to grow. 'start' prepends (an intro) and shifts every section marked earlier in the chain later by the same amount, so their timings stay on the music."}),
                "fade_sec": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Crossfade at each edge of the section, in seconds. The ramp lies OUTSIDE the window, so the range you named is rewritten in full and the join is spread into the neighbouring audio.\n\nThis is the difference between a seamless replacement and an audible CLICK: in an image a hard mask edge is just a visible line, in audio it is a transient. 0.2-0.5 s is a good range; 0 for a hard cut."}),
                "snap": (SNAP_MODES, {"default": "bar", "tooltip": "Quantize the section's edges to the musical grid built from 'bpm' / 'beats_per_bar'.\n\n• bar (recommended) — cut on the bar line. Replacing a section that starts mid-bar is the usual reason a retake refuses to sit in the groove.\n• beat — finer, for fills and pickups.\n• off — exactly the seconds you typed."}),
                "bpm": ("INT", {"default": 120, "min": 0, "max": 300, "tooltip": "Tempo for the snap grid — use the same value you gave TextEncodeAceStepAudio1.5. 0 disables snapping regardless of the 'snap' setting."}),
                "beats_per_bar": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "Beats per bar for the snap grid (the 'timesignature' on TextEncodeAceStepAudio1.5). One bar at 120 bpm in 4/4 is 2 s = exactly 50 latent frames."}),
                "grid_origin_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 0.01, "tooltip": "Where bar 1 begins, in seconds. Leave at 0 unless the track opens with a pickup or a bit of silence — then the grid is offset from the file start and snapping to 0 would land between bars."}),
                "latent_fps": ("FLOAT", {"default": DEFAULT_FPS, "min": 1.0, "max": 1000.0, "step": 0.01, "tooltip": "Latent frames per second, used to turn seconds into frame indices. 25 is correct for AceStep 1.5 (the VAE maps one frame to 1920 samples at 48 kHz = 40 ms). Wire the 'vae' input instead of touching this — it reads the true rate from the model and covers AceStep 1.0, which is different."}),
            },
            "optional": {
                "section": (SIREN_SECTION, {"tooltip": "Wire a PREVIOUS Siren Section here to chain them — several sections marked at once (two retakes, or an extend plus a retake of the tail). Each node adds its own window and rebuilds the mask over all of them, so overlapping windows merge cleanly."}),
                "vae": ("VAE", {"tooltip": "The audio VAE, used ONLY to read the true latent frame rate (sample rate ÷ samples per frame) instead of trusting 'latent_fps'. Nothing is encoded or decoded here. Recommended: it makes the seconds exact on any audio model."}),
            },
        }

    RETURN_TYPES = ("LATENT", SIREN_SECTION, "STRING")
    RETURN_NAMES = ("latent", "section", "report")
    FUNCTION = "build"
    CATEGORY = CAT_SIREN
    DESCRIPTION = ("Mark a stretch of an AceStep audio latent as the part to regenerate — a section "
                   "retake — or grow the latent to extend the track. Edges can snap to the bar/beat "
                   "grid and are crossfaded so the join doesn't click. The mask is written to the "
                   "latent's standard 'noise_mask', so the output works with the stock samplers too; "
                   "Siren (Music Sampler) just also reports what it found.")

    def build(self, latent, mode, start_sec, end_sec, extend_sec, extend_at, fade_sec, snap, bpm,
              beats_per_bar, grid_origin_sec, latent_fps, section=None, vae=None):
        if not (isinstance(latent, dict) and latent.get("samples") is not None):
            raise RuntimeError("[Siren Section] No valid latent — wire an audio latent into 'latent'.")
        samples = latent["samples"]
        vae_fps = _fps_from_vae(vae)
        fps = vae_fps if vae_fps else max(1.0, float(latent_fps))
        notes, lines = [], []
        if vae_fps is None and vae is not None:
            notes.append("the wired VAE doesn't expose an audio frame rate — falling back to "
                         "'latent_fps'. Is it an image VAE?")
        if latent.get("type") not in (None, "audio"):
            notes.append(f"the latent's type is '{latent.get('type')}', not 'audio' — the time axis "
                         f"is taken to be the last dimension, which may not be what you want")
        prev = section if isinstance(section, dict) else {}
        windows = [dict(w) for w in prev.get("windows", []) if isinstance(w, dict)]
        if latent.get("noise_mask") is not None and not windows:
            notes.append("the input latent already carried a noise_mask — it is REPLACED by the one "
                         "this node builds")
        frames = int(samples.shape[-1])
        grid = _grid_seconds(snap, bpm, beats_per_bar)
        origin = float(grid_origin_sec)
        fade = max(0.0, float(fade_sec))
        label = f"{mode} #{len(windows) + 1}"

        if mode == "extend":
            add = int(round(max(0.0, float(extend_sec)) * fps))
            if add <= 0:
                notes.append("'extend_sec' rounds to 0 frames — nothing was added and no section was "
                             "marked, so the sampler will regenerate the whole track")
            else:
                pad = _pad_latent(samples, add)
                if extend_at == "start":
                    samples = torch.cat([pad, samples], dim=-1)
                    shift = add / fps
                    for w in windows:   # the old audio moved later; keep earlier marks on the music
                        w["start"] = float(w["start"]) + shift
                        w["end"] = float(w["end"]) + shift
                    win = {"start": 0.0, "end": add / fps, "fade": fade, "label": label}
                    if windows:
                        notes.append(f"prepended {add / fps:.2f} s — the {len(windows)} section(s) "
                                     f"marked upstream were shifted later by the same amount")
                else:
                    samples = torch.cat([samples, pad], dim=-1)
                    win = {"start": frames / fps, "end": (frames + add) / fps, "fade": fade,
                           "label": label}
                frames += add
                windows.append(win)
                lines.append(f"  extend at the {extend_at}: +{add} frames ({add / fps:.2f} s) → "
                             f"track is now {_mmss(frames / fps)} ({frames} frames)")
                lines.append(f"  ⓘ set 'duration' on TextEncodeAceStepAudio1.5 to "
                             f"{frames / fps:.2f} and re-encode — it is baked into the tokens")
        else:
            dur = frames / fps
            s_raw = max(0.0, float(start_sec))
            e_raw = float(end_sec) if float(end_sec) > 0 else dur
            s, e = _snap_sec(s_raw, grid, origin), _snap_sec(e_raw, grid, origin)
            if grid is not None and (abs(s - s_raw) > 1e-6 or abs(e - e_raw) > 1e-6):
                lines.append(f"  snapped to the {snap} grid ({grid:.4f} s): "
                             f"{s_raw:.2f}→{s:.2f} s, {e_raw:.2f}→{e:.2f} s")
            s = min(s, dur)
            e = min(e, dur)
            if e <= s:
                if grid is not None and s + grid <= dur:
                    e = s + grid
                    notes.append(f"start and end snapped onto the same grid line — the section was "
                                 f"widened to one {snap} ({grid:.3f} s)")
                else:
                    raise RuntimeError(
                        f"[Siren Section] empty section: {s:.2f} s → {e:.2f} s on a "
                        f"{dur:.2f} s track. Check 'start_sec' / 'end_sec' (0 = to the end).")
            windows.append({"start": s, "end": e, "fade": fade, "label": label})

        out = {k: v for k, v in latent.items()}
        out["samples"] = samples
        mask = _build_mask(windows, frames, fps, samples)
        out["noise_mask"] = mask
        free = float(mask.mean()) if mask.numel() else 0.0

        head = (f"Siren Section — {len(windows)} section(s) · {fps:.4g} frames/s "
                f"({'vae' if vae_fps else 'latent_fps'}) · track {_mmss(frames / fps)} "
                f"({frames} frames) · {free * 100:.1f}% free to change")
        lines.insert(0, head)
        for w in windows:
            i0, i1 = int(round(w["start"] * fps)), int(round(w["end"] * fps))
            bar_a = _bars_at(w["start"], bpm, beats_per_bar, origin)
            bars = (f" · bars {bar_a + 1:.2f}-{_bars_at(w['end'], bpm, beats_per_bar, origin) + 1:.2f}"
                    if bar_a is not None else "")
            lines.append(f"  {w['label']}: {_mmss(w['start'])} → {_mmss(w['end'])} "
                         f"(frames {i0}-{i1}, {i1 - i0}){bars} · fade {w['fade']:.2f} s "
                         f"({int(round(w['fade'] * fps))} frames each side)")
        for w in notes:
            lines.append(f"  ⚠ {w}")
        report = "\n".join(lines)
        print("[Siren Section] " + report.replace("\n", "\n[Siren Section] "))
        return (out, {"windows": windows, "fps": fps, "frames": frames,
                      "bpm": int(bpm), "beats_per_bar": int(beats_per_bar)}, report)


# ----------------------------------------------------------------------------- schedule helpers
SAME_AS_A = "same as stage A"


def _resume_index(sigmas, curve_len, target):
    """First step index whose sigma has fallen to `target` — where a partial run picks up. `0` means
    nothing is skipped (the whole curve runs). Clamped so at least one step always remains."""
    target = float(target)
    for i in range(0, int(curve_len)):
        if float(sigmas[i]) <= target:
            return i
    return max(0, int(curve_len) - 1)


def _stage_ranges(counts, start_index, curve_len):
    """`[lo, hi)` in the shared curve for each stage, with everything before `start_index` skipped.
    A stage that lies entirely before the resume point comes back empty (`hi == lo`) and is reported
    as skipped rather than silently folded into its neighbour."""
    out, lo = [], 0
    for c in counts:
        hi = min(int(curve_len), lo + max(0, int(c)))
        out.append((max(lo, int(start_index)), hi))
        lo = hi
    return [(a, b) if b > a else (a, a) for a, b in out]


def _landmarks(sigmas, curve_len):
    """Sigma a quarter, half and three quarters of the way through the curve. Worth printing because
    the ENDPOINTS of a flow schedule are always 1→0 whatever `shift` is set to — only the middle
    moves. With AceStep's default shift of 3.0 the halfway figure is 0.750; at shift 1.0 it is 0.500.
    So one glance at this line says whether a ModelSamplingAuraFlow node is actually in the path."""
    out = []
    for frac in (0.25, 0.5, 0.75):
        i = max(0, min(int(curve_len), int(round(curve_len * frac))))
        out.append(f"{int(frac * 100)}%={float(sigmas[i]):.3f}")
    return " ".join(out)


def _inline_chain(seed, steps, cfg, sampler_name, scheduler, stage_b_steps, stage_b_cfg,
                  stage_b_sampler, stage_b_scheduler, stage_b_seed, eta, s_noise, s_churn,
                  solver_type):
    """The node's own widgets as a stage chain, in the same shape a `Sampler Settings` bundle has.
    `steps` is the WHOLE schedule and `stage_b_steps` is carved out of its tail, so 50 / 10 means
    40 + 10 — you never have to add the stages up yourself. `denoise` is fixed at 1.0: the partial-run
    dial on this node is `resume_from_sigma`, which says the same thing in a unit that means
    something musically."""
    b = max(0, int(stage_b_steps))
    a = max(1, int(steps) - b)
    common = {"seed": int(seed), "scheduler": scheduler, "denoise": 1.0, "eta": float(eta),
              "s_noise": float(s_noise), "s_churn": float(s_churn), "solver_type": solver_type}
    chain = [dict(common, steps=a, cfg=float(cfg), sampler_name=sampler_name)]
    if b > 0:
        chain.append(dict(common, steps=b, cfg=float(stage_b_cfg),
                          sampler_name=(sampler_name if stage_b_sampler == SAME_AS_A
                                        else stage_b_sampler),
                          scheduler=(scheduler if stage_b_scheduler == SAME_AS_A
                                     else stage_b_scheduler),
                          seed=(int(seed) if int(stage_b_seed) < 0 else int(stage_b_seed))))
    return chain


# --------------------------------------------------------------------------------- sampler node
class KinburgSirenSampler:
    """Sampler for AceStep music latents: staged cfg, and section-aware when a mask is present."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_audio": ("LATENT", {"tooltip": "The audio latent to sample. Straight from 'Empty Ace Step 1.5 Latent Audio' for a fresh track, or through a 'Siren Section' node to regenerate only part of an existing take."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Noise seed. NOTE this is NOT the seed that decides the song: the musical idea comes from the audio-codes LLM inside TextEncodeAceStepAudio1.5, which has its OWN seed. Change that one for a different piece; change this one for a different noise draw of the same piece.\n\nWhen comparing settings, hold both fixed — otherwise you're comparing two different tracks."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000, "tooltip": "Length of the WHOLE schedule. 50 is the shipped default for acestep_v1.5_xl_base / _sft; the turbo variants want 8.\n\n'stage_b_steps' is carved out of this number, not added to it — 50 with stage_b_steps 10 means 40 + 10, so you never add the stages up by hand."}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Guidance for stage A. Shipped defaults: 6.0 for xl_base, 7.0 for xl_sft, and 1.0 for the turbo variants.\n\nOn AceStep this is a trade, not a quality dial: high cfg locks the lyrics and the structure but squeezes the sound, low cfg lets the timbre breathe but slurs the words. At exactly 1.0 the unconditional pass is skipped entirely, so the negative input stops doing anything at all."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "'euler' is what every shipped AceStep 1.5 template uses. An ancestral / SDE sampler adds variety, but its extra noise on the tail is audible as hiss — if you want one, put it on stage A and finish on a deterministic sampler via 'stage_b_sampler'."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "'simple' is the shipped default for AceStep 1.5. One schedule is built for the whole run, so this applies to every stage."}),
                "stage_b_steps": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "How many of the 'steps' the SECOND stage takes, off the tail. 0 = a single stage (start here).\n\nTwo stages exist for one reason: cfg can be high while the lyrics and structure are being decided and lower while the sound is being finished. AceStep's schedule is heavily top-loaded (with shift 3.0 the halfway step is still at sigma 0.750), so the boundary has to be LATE to land in polishing territory — out of 50 steps, 10 puts it at sigma 0.429 and 15 at 0.562. Much earlier and the second stage starts rewriting the arrangement instead.\n\nIgnored when a section is marked — see the tooltip on 'resume_from_sigma'."}),
                "stage_b_cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Guidance for the second stage. Lower than 'cfg' is the point — try 4.5 against a stage-A 6.0. Ignored when 'stage_b_steps' is 0."}),
                "resume_from_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "0 = off: start from pure noise, i.e. a fresh take.\n\nAbove 0: RESUME an existing take from this noise level instead — the whole track, or only the marked stretch when a Siren Section is wired. This is the retake dial, and it replaces fiddling with 'denoise': the step count is derived for you, so the run walks exactly the tail of the native schedule and takes proportionally less time.\n\nOn a 50-step curve at shift 3.0:\n  0.43 → tidy up the performance, groove intact (10 steps)\n  0.51 → same musical idea, different performance (12)\n  0.56 → noticeably different take (14)\n  0.71 → almost a new section (22)\n  1.00 → completely new (50)\n\nThe scale is not linear in how much survives — the report prints the step it landed on. Needs something to resume FROM: on an empty latent it only weakens the start, which the node warns about."}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True, "tooltip": "Stochasticity for ancestral / SDE samplers. NO effect on 'euler' or other deterministic samplers, which is what AceStep normally runs."}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True, "tooltip": "Multiplier on the noise added at stochastic steps. Ancestral / SDE / churn samplers only."}),
                "s_churn": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True, "tooltip": "Re-injects noise into otherwise deterministic samplers (euler, heun, dpm_2). 0 = off, which is what the AceStep templates use."}),
                "solver_type": (SOLVER_TYPES, {"default": "midpoint", "advanced": True, "tooltip": "Solver variant for the samplers that have one. A value from the wrong family is ignored (noted in the console) rather than breaking the run, so this can't hurt."}),
                "stage_b_sampler": ([SAME_AS_A] + list(comfy.samplers.KSampler.SAMPLERS), {"advanced": True, "tooltip": "Let the second stage use a different sampler. The useful case: an ancestral sampler on stage A for variety, then a deterministic one here so the tail doesn't come out hissy."}),
                "stage_b_scheduler": ([SAME_AS_A] + list(comfy.samplers.KSampler.SCHEDULERS), {"advanced": True, "tooltip": "Let the second stage walk a different scheduler's shape over its part of the run — e.g. 'simple' to lay down the structure and 'beta' to spend the tail differently.\n\nOne curve is still shared, so this can't just be swapped in: the alternate scheduler is rebuilt over the same length, its tail sliced out, and its first sigma PINNED to the level the latent actually carries at the handoff (a mismatch there is what produces classic 'refiner seam' artifacts), then forced monotonic. Same splice Chimera does. The report says when it happened."}),
                "stage_b_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "advanced": True, "tooltip": "Separate noise seed for the second stage. -1 = use stage A's.\n\nIt only does something when 'stage_b_sampler' is ancestral or SDE: a continuing stage adds NO fresh noise to the latent, so the seed reaches nothing but the stochastic samplers' own internal noise generator. With euler it is inert.\n\nIn a RETAKE (a section mask) the rule is different — there only one stage runs anyway, and it draws its own noise from stage A's seed."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Print the report (curve landmarks, stages, section coverage, warnings) to the console. The same text is always on the 'report' output."}),
            },
            "optional": {
                "section": (SIREN_SECTION, {"tooltip": "Output of a 'Siren Section' node — informational here (it feeds the report and 'gen_extra_info'). What actually drives the sampling is the mask that node writes into the LATENT, so the input that matters is 'latent_audio'. Wiring this while forgetting the latent is a mistake the node warns about."}),
                "sampler": (SAMPLER_CFG, {"tooltip": "Optional 'Sampler Settings' bundle (or a chain of them, or a recipe out of the model library's Settings Select / Model Select). While wired it REPLACES this node's own sampling widgets — steps, cfg, samplers, scheduler and the stage split all come from the bundle, and the report says so.\n\nUse it to drive Siren from a stored per-model recipe. For everything else the widgets are simpler: they can't disagree with themselves, and they drop the dials that do nothing here ('seed_mode' / 'seed_step' are Ouroboros loop controls, and a later stage's 'denoise' and 'scheduler' are ignored because one schedule is shared)."}),
                "sigmas": ("SIGMAS", {"tooltip": "External noise schedule, replacing the one this node builds: 'steps', 'scheduler' and any stage denoise are then ignored, though the stages' step counts still decide how the curve is split between them. 'resume_from_sigma' still applies on top."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "GEN_INFO", "STRING", "FLOAT")
    RETURN_NAMES = ("latent", "report", "gen_extra_info", "time", "seconds")
    FUNCTION = "run"
    CATEGORY = CAT_SIREN
    DESCRIPTION = ("Sampler for AceStep music latents, with the dials it actually needs on the node. "
                   "'steps' is the whole schedule and 'stage_b_steps' carves the tail off it, so cfg "
                   "can be high while the lyrics and structure are decided and lower while the sound "
                   "is finished. 'resume_from_sigma' is the retake dial — it resumes an existing take "
                   "from a chosen noise level and derives the step count itself, so a retake costs "
                   "proportionally less time. Wire a 'Siren Section' latent in and the retake applies "
                   "to just that stretch of the track. A 'Sampler Settings' bundle can be wired to "
                   "override the widgets, which is how a stored per-model recipe drives it. Reports "
                   "the curve's midpoints (the only place 'shift' is visible), per-stage times and "
                   "section coverage; 'gen_extra_info' feeds Generation Info's 'extra' input.")

    def run(self, model, positive, negative, latent_audio, seed, steps, cfg, sampler_name, scheduler,
            stage_b_steps, stage_b_cfg, resume_from_sigma, eta=1.0, s_noise=1.0, s_churn=0.0,
            solver_type="midpoint", stage_b_sampler=SAME_AS_A, stage_b_scheduler=SAME_AS_A,
            stage_b_seed=-1, verbose=True, section=None, sampler=None, sigmas=None):
        import comfy.sample
        if not (isinstance(latent_audio, dict) and latent_audio.get("samples") is not None):
            raise RuntimeError("[Siren] No valid latent — wire an audio latent into 'latent_audio'.")
        notes, lines = [], []
        # A wired bundle wins outright rather than merging with the widgets: a half-and-half rule
        # would be impossible to read off the node. The widgets are the simple path; the bundle is
        # how a stored per-model recipe (Settings Select / Model Select) drives this node.
        wired = _flatten_cfg(sampler)
        if wired:
            stages = [_norm_stage(s) for s in wired]
            src = f"wired SAMPLER_CFG ({len(stages)} stage(s))"
            notes.append("a SAMPLER_CFG is wired — this node's own sampling widgets are IGNORED: "
                         "steps, cfg, the samplers, the scheduler and the stage split all come from "
                         "the bundle. Unwire it to use the widgets.")
        else:
            stages = _inline_chain(seed, steps, cfg, sampler_name, scheduler, stage_b_steps,
                                   stage_b_cfg, stage_b_sampler, stage_b_scheduler, stage_b_seed,
                                   eta, s_noise, s_churn, solver_type)
            src = "node widgets"
            if int(steps) - max(0, int(stage_b_steps)) < 1:
                notes.append(f"'stage_b_steps' ({int(stage_b_steps)}) leaves nothing for stage A out "
                             f"of {int(steps)} steps — stage A was given 1, so the curve is "
                             f"{sum(s['steps'] for s in stages)} steps rather than {int(steps)}")
        n = len(stages)
        names = [chr(ord("A") + i) for i in range(n)]

        # The empty-latent fixups the stock samplers do: correct the channel count and, for a latent
        # authored against a different model's frame rate, the length. Done once here (the helpers
        # called per stage don't get the ratios), and the keys are stripped from the output the same
        # way SamplerCustom strips them, so a downstream node can't apply the correction twice.
        samples = comfy.sample.fix_empty_latent_channels(
            model, latent_audio["samples"], latent_audio.get("downscale_ratio_spacial"),
            latent_audio.get("downscale_ratio_temporal"))
        latent_cur = {k: v for k, v in latent_audio.items()}
        latent_cur["samples"] = samples.clone()
        mask = latent_cur.get("noise_mask")
        masked = mask is not None

        sec_info = section if isinstance(section, dict) else {}
        # A section carries the rate it measured (from a VAE, ideally). Without one there is nothing
        # here to read it from, so the report says the seconds are assumed rather than known.
        fps_known = isinstance(sec_info.get("fps"), (int, float)) and float(sec_info["fps"]) > 0
        fps = float(sec_info["fps"]) if fps_known else DEFAULT_FPS
        frames = int(samples.shape[-1])
        duration = frames / fps
        if latent_audio.get("type") not in (None, "audio"):
            notes.append(f"the latent's type is '{latent_audio.get('type')}', not 'audio' — this "
                         f"node assumes the LAST dimension is time")
        if sec_info and not masked:
            notes.append("a 'section' is wired but the latent carries NO noise_mask — the whole "
                         "track will be regenerated. Wire the Section node's 'latent' output into "
                         "'latent_audio' as well; the section input alone changes nothing.")
        if masked and int(sec_info.get("frames", frames)) != frames:
            notes.append(f"the section was built for a {sec_info.get('frames')}-frame latent but "
                         f"this one is {frames} — the mask is stretched to fit, so the timings "
                         f"have moved. Did a node between them change the length?")

        external = sigmas is not None and getattr(sigmas, "numel", lambda: 0)() >= 2
        curve_len = max(1, sum(max(0, s["steps"]) for s in stages))
        if external:
            master = sigmas.to(model.load_device)
            curve_len = int(master.shape[-1]) - 1
            notes.append(f"external sigmas ({curve_len} steps, {float(master[0]):.4f} → "
                         f"{float(master[-1]):.4f}) — 'steps', the scheduler and any stage denoise "
                         f"are ignored, though the stages' step counts still decide how the curve is "
                         f"split between them")
        else:
            master = _build_sigmas(model, stages[0], curve_len, stages[0]["denoise"])
        if master is None or master.numel() < 2:
            msg = (f"[Siren] stage A denoise {stages[0]['denoise']} leaves nothing to sample — "
                   f"returning the input latent untouched.")
            print(msg)
            out = {k: v for k, v in latent_audio.items()}
            noop = [{"class_type": "Siren", "ord": 1,
                     "params": {"status": f"no-op — stage A denoise {stages[0]['denoise']}"}}]
            return (out, msg, json.dumps(noop, ensure_ascii=False),
                    _format_elapsed(0.0, "auto"), 0.0)
        avail = int(master.shape[-1]) - 1
        if avail < curve_len:
            notes.append(f"schedule yielded {avail} steps, not {curve_len} — using {avail}")
            curve_len = avail
        counts = _split_counts(curve_len, [s["steps"] for s in stages], "stage steps", 0.0,
                               master, 0.0)
        if sum(counts) < curve_len:
            notes.append(f"stages walk {sum(counts)} of {curve_len} scheduled steps — the run stops "
                         f"at sigma {float(master[sum(counts)]):.4f}, so the result keeps residual "
                         f"noise (on audio that is hiss). Raise a stage's steps.")

        # `resume_from_sigma` skips the HEAD of the curve: the run picks an existing take up at that
        # noise level instead of starting from scratch. The step count falls out of it, so a retake
        # walks exactly the tail of the native schedule and costs proportionally less time — the same
        # result as the old "steps = total x denoise" arithmetic, without the arithmetic.
        resume = float(resume_from_sigma)
        start_index = 0
        if resume > 0.0:
            start_index = _resume_index(master, curve_len, resume)
            lines.append(f"  resume from sigma {resume:.3f} → step {start_index} of {curve_len} "
                         f"(sigma {float(master[start_index]):.4f}), "
                         f"{curve_len - start_index} steps to run")
            if bool(torch.count_nonzero(samples) == 0):
                notes.append("'resume_from_sigma' is set but the latent is EMPTY — there is nothing "
                             "to resume from, so all it does is lower the starting noise level and "
                             "weaken the result. Use 0 for a fresh take, or feed a take you already "
                             "have (through a Siren Section, for a retake).")
            if stages[0]["denoise"] < 0.9999:
                notes.append(f"'resume_from_sigma' and the wired stage-A denoise "
                             f"({stages[0]['denoise']:.2f}) COMPOUND — the bundle already trimmed the "
                             f"curve and this trims what is left again. Use one or the other.")
            if _is_extend(sec_info):
                notes.append("'resume_from_sigma' on an EXTEND: the added stretch starts from zeros "
                             "rather than pure noise, which can bleed in as a dead patch. Extend "
                             "wants 0 (a full run) — the mask, not the schedule, is what protects "
                             "the audio you already have.")
        ranges = _stage_ranges(counts, start_index, curve_len)

        # A masked run has to finish at sigma 0: only there is the frozen region's reference the CLEAN
        # latent. Handing an in-flight latent to a second stage would re-pin the frozen audio to a
        # partially-denoised reference with no noise term — right at the handoff step, drifting after
        # it. So with a section exactly one stage runs, and it takes the whole remaining tail.
        if masked:
            live = [i for i, (a, b) in enumerate(ranges) if b > a]
            if len(live) > 1:
                notes.append(f"a section is marked, so only ONE stage can run — a masked run has to "
                             f"finish at sigma 0 for the frozen audio to stay consistent. Stage "
                             f"{names[live[0]]} took the whole tail; "
                             f"{', '.join(names[i] for i in live[1:])} skipped. Set 'stage_b_steps' "
                             f"to 0 for retakes.")
            if live:
                keep = live[0]
                ranges = [(a, a) for a, _ in ranges]
                ranges[keep] = (start_index, curve_len)

        free = float(mask.float().mean()) if masked else 1.0
        scope = (f"section: {free * 100:.1f}% of {_mmss(duration)} free to change"
                 if masked else f"whole track ({_mmss(duration)})")
        curve_src = "external sigmas" if external else f"{stages[0]['scheduler']}, {curve_len} steps"
        rate = f"{frames} frames @ {fps:.4g}/s" + ("" if fps_known else " (assumed: AceStep 1.5)")
        lines.insert(0, f"Siren — {scope} · {n} stage(s) from {src} · curve {curve_src} · "
                        f"sigma {float(master[0]):.4f} → {float(master[curve_len]):.4f} "
                        f"[{_landmarks(master, curve_len)}] · {rate}")
        if masked:
            for w in sec_info.get("windows", []):
                lines.append(f"  {w.get('label', 'section')}: {_mmss(w['start'])} → "
                             f"{_mmss(w['end'])} · fade {float(w.get('fade', 0.0)):.2f} s")

        head = {"scope": scope, "stages": f"{n} from {src}",
                "curve": (f"{curve_src} · sigma {float(master[0]):.4f} → "
                          f"{float(master[curve_len]):.4f} [{_landmarks(master, curve_len)}]"),
                "length": f"{duration:.2f} s ({frames} frames @ {fps:.4g}/s)"}
        if resume > 0.0:
            head["resume_from_sigma"] = (f"{resume:.3f} → step {start_index}/{curve_len} "
                                         f"(sigma {float(master[start_index]):.4f})")
        if masked:
            head["sections"] = "; ".join(
                f"{w['start']:.2f}-{w['end']:.2f}s (fade {float(w.get('fade', 0.0)):.2f})"
                for w in sec_info.get("windows", [])) or "mask without section metadata"
        stage_entries = []
        total_secs = 0.0

        first = True
        for i, stg in enumerate(stages):
            lo, hi = ranges[i]
            if hi <= lo:
                lines.append(f"  stage {names[i]}: no steps in its range — skipped")
                continue
            sl = master[lo:hi + 1]
            count = hi - lo
            span = f"steps {lo + 1}-{hi} ({count})"
            # Only the first stage that actually runs draws noise; a later one continues the
            # trajectory it was handed. With a section there is only ever one, so it always draws.
            add_noise = first
            if not first:
                span += ", continues"
            first = False
            # A different scheduler on a later stage can't simply replace the shared curve — the
            # latent is sitting at a particular noise level. Rebuild that scheduler over the same
            # length, take its tail, and pin its first sigma to the level actually carried.
            if i > 0 and stg["scheduler"] != stages[0]["scheduler"] and not external:
                sl, spliced = _tail_for_scheduler(model, stg, master, lo, count, curve_len,
                                                  stages[0]["denoise"])
                if spliced:
                    span += f", '{stg['scheduler']}' spliced on"
                else:
                    notes.append(f"stage {names[i]}'s scheduler '{stg['scheduler']}' could not be "
                                 f"spliced onto the shared curve — stage A's "
                                 f"'{stages[0]['scheduler']}' was used for it instead")
            elif i > 0 and stg["scheduler"] != stages[0]["scheduler"]:
                notes.append(f"stage {names[i]} asks for scheduler '{stg['scheduler']}', but the "
                             f"curve came in on the 'sigmas' input — there is no scheduler to "
                             f"rebuild it from, so it is ignored")
            t0 = time.perf_counter()
            latent_cur = _sample_slice(model, latent_cur, positive, negative, stg, sl, stg["seed"],
                                       add_noise=add_noise)
            secs = time.perf_counter() - t0
            total_secs += secs
            elapsed_i = _format_elapsed(secs, "auto")
            lines.append(f"  stage {names[i]}: {_stage_label(stg)} · {span} · cfg {stg['cfg']} · "
                         f"eta {stg['eta']} · seed {stg['seed']} · sigma {float(sl[0]):.4f} → "
                         f"{float(sl[-1]):.4f} · {elapsed_i}")
            stage_entries.append({"class_type": "Siren stage", "ord": len(stage_entries) + 1,
                                  "params": {"sampler": stg["sampler_name"],
                                             "scheduler": stg["scheduler"], "steps": span,
                                             "cfg": stg["cfg"], "eta": stg["eta"],
                                             "denoise": stg["denoise"], "seed": stg["seed"],
                                             "sigma": f"{float(sl[0]):.4f} → {float(sl[-1]):.4f}",
                                             "time": elapsed_i}})

        # Same contract as SamplerCustom: the correction the ratios describe has been applied, so the
        # keys must not travel on to whatever samples this latent next.
        latent_cur.pop("downscale_ratio_temporal", None)
        latent_cur.pop("downscale_ratio_spacial", None)
        elapsed = _format_elapsed(total_secs, "auto")
        if total_secs > 0:
            lines.append(f"  total sampling time: {elapsed} · {duration:.1f} s of audio · "
                         f"{total_secs / max(1e-6, duration):.2f} s per audio second")
        head["time"] = elapsed
        gen_extra = json.dumps([{"class_type": "Siren", "ord": 1, "params": head}] + stage_entries,
                               ensure_ascii=False)
        for w in notes:
            lines.append(f"  ⚠ {w}")
        report = "\n".join(lines)
        if verbose:
            print("[Siren] " + report.replace("\n", "\n[Siren] "))
        return (latent_cur, report, gen_extra, elapsed, float(total_secs))


def _is_extend(sec_info):
    """Did the section chain grow the latent? Only an extend leaves a window whose label says so."""
    return any(str(w.get("label", "")).startswith("extend")
               for w in sec_info.get("windows", []) if isinstance(w, dict))


NODE_CLASS_MAPPINGS = {
    "KinburgSirenSampler": KinburgSirenSampler,
    "KinburgSirenSection": KinburgSirenSection,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KinburgSirenSampler": "Siren (Music Sampler) 🧜",
    "KinburgSirenSection": "Siren Section (Audio Window) 🧜",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
