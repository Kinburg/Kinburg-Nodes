"""Morpheus — a chain of dreams in, one long video (with sound) out.

Named for the god who *shapes* dreams: they flow into one another instead of starting and stopping,
which is what this node does to MiniMax H3 shots — and "morphing" is what the seams are called in
video production anyway.

H3 generates ~5-15 s per run (its `length` grid is 17k+5 frames at a fixed 24 fps, trained on
124-362 of them) and ComfyUI ships no extend/continue node for it, so the only way to a minute of
video is to run it several times and hand the last frame of each shot to the next one as its first
keyframe. That loop is what this pair of nodes is:

  * **Morpheus Dream** — one link of the chain: prompt + duration, with optional `start_frame` /
    `end_frame` keyframes. Chain them like `Sampler Settings` (wire the `shots` output into the next
    node's `shots` input); left-to-right is shot order. One dream = one shot; the sockets keep the
    video-production word because that is what the tooltips and the report talk about.
  * **Morpheus (Video Sampler)** — resolves the chain, samples every shot, decodes it, feeds the
    handoff frame forward, and concatenates everything into one IMAGE batch + one AUDIO track.

How a shot's first frame is decided (this is the whole design in three lines):

  1. `start_frame` wired            → that image (hard override, always wins)
  2. else `link = continue`         → the last **generated** frame of the previous shot
  3. else (`link = cut`, or shot 1) → no keyframe at all: pure text-to-video

So three keyframes and two shots ("1→2", "2→3") work; so does a pure-text chain where every shot
inherits the previous shot's tail; so does any mix of the two.

Things this node has to get right that a hand-wired graph does not:

  * **fps is not a parameter.** 24 is baked into the model (frame grid *and* audio latent length),
    so duration is given in seconds and snapped to the grid; 24.0 comes back out as a FLOAT, which is
    the type `Create Video` takes — the output drops in with no conversion node.
  * **progress.** One bar for the whole run: total = shots x steps, advanced monotonically, so it
    never resets to 0 per shot (a cached shot jumps its slice at once).
  * **loudness.** `VAEDecodeAudio` normalises per decode (std*5), so decoding shots one by one would
    step the level at every seam. Here the normalisation is computed once over the whole track, and
    seams get equal-length fade-out/fade-in ramps (never an overlapping crossfade — that would shift
    the audio against the picture).
  * **the duplicated seam frame.** A continuing shot's first frame *is* the previous shot's last
    frame, so one of the two is dropped (with 1/24 s trimmed off that shot's audio to match).
  * **RAM.** The output batch is pre-allocated once (1344x768 float32 is 12.4 MB per frame, ~1.5 GB
    per 5-second shot) and filled shot by shot, so the peak is the final tensor plus one shot
    instead of double.
  * **iterating.** Sampled latents are cached on disk under a causal key (see `cache.py`), so
    editing shot 5 of 8 re-samples 5..8 and replays 1..4.

Not in v1, deliberately: reference images / audio (`MiniMaxH3ReferenceToVideo`). They cannot be
combined with keyframes anyway — `comfy/model_base.py` lets `minimax_refs` overwrite the keyframes'
`cond_video_latents`, so a shot is either fl2va or ref2va, never both.
"""
import logging
import time

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
import latent_preview
import node_helpers
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise

from . import cache as shot_cache
from ..local_llm.llm_node import LLM_CONFIG, _shutdown_worker as _shutdown_llm
from ..timer.timer_nodes import _format_elapsed

try:  # ComfyUI without MiniMax H3 support: fail with a readable message, not on import
    from comfy_extras import nodes_minimax_h3 as h3
except Exception as _e:  # pragma: no cover
    h3 = None
    _H3_ERR = str(_e)

MORPHEUS_SHOT = "KINBURG_MORPHEUS_SHOT"
LINK_MODES = ["continue", "cut"]
# kept here rather than imported from refine.py: that module imports storyboard.py, which imports
# this one, so touching it at import time would close a cycle
REFINE_SCOPES = ["auto", "off", "opening", "full"]
AUDIO_MODES = ["concat", "mute"]
CACHE_MODES = ["disk", "off"]
DEFAULT_AUD_SR = 44100


def _require_h3():
    if h3 is None:
        raise RuntimeError("This ComfyUI has no MiniMax H3 support (comfy_extras/nodes_minimax_h3.py "
                           f"failed to import: {_H3_ERR}). Update ComfyUI.")


# ------------------------------------------------------------------------------------- chain utils
def _flatten_shots(shots):
    """A MORPHEUS_SHOT input is one shot dict, an already-built chain, or None → always a flat list."""
    if isinstance(shots, list):
        return [s for s in shots if isinstance(s, dict)]
    if isinstance(shots, dict):
        return [shots]
    return []


def _frames_for(duration):
    """Seconds → the frame count H3 will actually run (17k+5 grid at 24 fps)."""
    _require_h3()
    return h3.align_frame_count(max(5, int(round(float(duration) * h3.FPS))))


def _seconds(frames):
    return frames / float(h3.FPS)


def _snap32(v):
    return max(32, int(round(v / 32.0)) * 32)


def _parse_range(text, n):
    """"" → all shots; "3" → just shot 3; "2-4" → shots 2..4 (1-based, clamped)."""
    text = (text or "").strip()
    if not text:
        return 1, n
    try:
        if "-" in text:
            a, b = text.split("-", 1)
            lo = int(a) if a.strip() else 1
            hi = int(b) if b.strip() else n
        else:
            lo = hi = int(text)
    except ValueError:
        raise ValueError(f"shots_range: expected '' or '3' or '2-4', got {text!r}")
    lo, hi = max(1, min(lo, n)), max(1, min(hi, n))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _fp16_round(img):
    """Every handoff frame goes through fp16 so the fresh path and the cached path are identical.

    `.contiguous()` is not cosmetic: a real video VAE hands back frames as a permuted view, `.to()`
    preserves those strides, and safetensors refuses to write a non-contiguous tensor — which is
    exactly how the shot cache silently wrote nothing on the first real run."""
    return img.detach().to("cpu", torch.float16).clamp(0.0, 1.0).to(torch.float32).contiguous()


# =========================================================================================== nodes
class KinburgMorpheusDream:
    """One link of a Morpheus storyboard: prompt + duration (+ optional keyframes)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "tooltip": "What happens in THIS shot. With both keyframes wired, describe the transition between them."}),
                "duration": ("FLOAT", {"default": 5.17, "min": 0.2, "max": 15.1, "step": 0.1,
                                       "tooltip": "Shot length in seconds. Snapped UP to the model's frame grid — 17k+5 frames at 24 fps, i.e. steps of 0.71 s: … 4.46, 5.17, 5.88, 6.58 … s. The trained range is 124-362 frames = 5.17-15.08 s; anything shorter still runs but is out of distribution. The 'info' output shows the length you actually get."}),
                "link": (LINK_MODES, {"default": "continue",
                                      "tooltip": "How this shot starts when NO 'start_frame' is wired:\n\n• continue — inherit the last generated frame of the previous shot (a continuous take; motion still resets slightly at the seam, since the model only sees a still frame).\n\n• cut — start from nothing: text-to-video. A hard montage cut.\n\nA wired 'start_frame' always wins over both."}),
                "seed_offset": ("INT", {"default": 0, "min": 0, "max": 0xffffffff,
                                        "tooltip": "Added to this shot's seed. Bump it to re-roll THIS shot only — it changes the shot's cache key, so every earlier shot still replays from cache (later ones re-sample, because their first frame changes)."}),
            },
            "optional": {
                "shots": (MORPHEUS_SHOT, {"tooltip": "Wire the PREVIOUS 'Morpheus Dream' here to chain them. Left→right = shot order. Leave empty for the first shot."}),
                "start_frame": ("IMAGE", {"tooltip": "First frame of this shot. Overrides 'link'. NOTE: it is STRETCHED to the canvas (no crop), so its aspect ratio should match — the sampler's 'canvas' = auto takes the canvas from this image."}),
                "end_frame": ("IMAGE", {"tooltip": "Last frame of this shot — the target the shot moves toward (cover-cropped to the canvas). The next shot still inherits the GENERATED tail frame, not this image, so the two stay visually identical."}),
                "keyframe_strength": ("FLOAT", {"default": 0.999, "min": 0.900, "max": 1.000, "step": 0.001, "advanced": True,
                                                "tooltip": "How hard the model is held to its keyframes (the model's visual_cond_noise_aug). 1.0 = exact; lower mixes noise into the keyframe latent and lets the shot drift from it. Worth lowering a hair on a long chain, where re-encoded handoff frames accumulate colour/detail drift. Advanced — leave at 0.999 unless you are chasing that."}),
                "refine": (REFINE_SCOPES, {"default": "auto",
                                           "tooltip": "Whether the sampler's own vision LLM (its optional 'llm_config' input) reworks this shot just before it is sampled, when the real first frame finally exists.\n\n• auto — rework the OPENING when this shot's first frame is one the writer never saw (an inherited tail: its text was written against a forecast); write the whole shot when there is no proper prompt at all; otherwise leave it alone.\n\n• off — hands off. Use this when you pasted a finished prompt yourself.\n\n• opening — always rewrite [Scene Overview] and the first beat from the real frame.\n\n• full — always rewrite the shot from the frames, keeping the style and negative sections.\n\nIgnored entirely when no 'llm_config' is wired into the sampler."}),
            },
        }

    RETURN_TYPES = (MORPHEUS_SHOT, "STRING")
    RETURN_NAMES = ("shots", "info")
    FUNCTION = "build"
    CATEGORY = "Kinburg-Nodes/sampling"
    DESCRIPTION = ("One dream of a Morpheus storyboard. Chain several (optional 'shots' input) and "
                   "feed the chain to 'Morpheus (Video Sampler)'. Keyframes are optional: with "
                   "link = continue a shot starts from the previous shot's last generated frame.")

    def build(self, prompt, duration, link, seed_offset, shots=None,
              start_frame=None, end_frame=None, keyframe_strength=0.999, refine="auto"):
        _require_h3()
        chain = _flatten_shots(shots)
        frames = _frames_for(duration)
        shot = {
            "prompt": prompt,
            "frames": frames,
            "link": link if link in LINK_MODES else "continue",
            "seed_offset": int(seed_offset),
            "keyframe_strength": float(keyframe_strength),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "refine": refine if refine in REFINE_SCOPES else "auto",
        }
        chain.append(shot)

        idx = len(chain)
        start = "wired image" if start_frame is not None else (
            "inherited" if (idx > 1 and shot["link"] == "continue") else "text only")
        info = (f"shot {idx} · {frames} frames · {_seconds(frames):.2f} s · "
                f"start: {start} · end: {'wired image' if end_frame is not None else 'open'}")
        return (chain, info)


class KinburgMorpheus:
    """Samples a MORPHEUS_SHOT chain into one continuous video + audio track."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shots": (MORPHEUS_SHOT, {"tooltip": "The chain out of the last 'Morpheus Dream'."}),
                "model": ("MODEL", {"tooltip": "The H3 model. 'MiniMax H3 Sigma Shift' is applied here automatically if the model doesn't already carry it."}),
                "clip": ("CLIP", {"tooltip": "H3's text encoder (it also sees the keyframes)."}),
                "vae": ("VAE", {"tooltip": "The VIDEO vae."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 4096, "step": 32,
                                  "tooltip": "Canvas width for EVERY shot (they are concatenated, so one canvas for the whole storyboard). Rounded to a multiple of 32. H3's own budget is a 768 short edge with a 768*1344 area cap; going over it is slow and out of distribution, and the report says so."}),
                "height": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32,
                                   "tooltip": "Canvas height. Match the aspect ratio of your keyframes: H3 STRETCHES the first frame onto the canvas and does not crop it, so a mismatch distorts the whole shot. The report warns when it spots one."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 1000,
                                  "tooltip": "Ignored when a 'sigmas' input is wired."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple",
                                                                   "tooltip": "Ignored when a 'sigmas' input is wired."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler",
                                                                    "tooltip": "Ignored when a 'sampler' input is wired."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "Base seed. Each shot samples with seed + shot index + its own seed_offset, so no two shots get the same noise."}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01, "advanced": True,
                                          "tooltip": "H3's video flow shift (drives the sigma schedule). Applied here unless the wired model already went through 'MiniMax H3 Sigma Shift'."}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01, "advanced": True}),
                "seam_trim": ("INT", {"default": 1, "min": 0, "max": 24, "advanced": True,
                                      "tooltip": "Frames dropped from the HEAD of every shot that starts from a keyframe.\n\n1 (default) removes the duplicate — a continuing shot's first frame is the previous shot's last frame.\n\nHigher values also cut the re-acceleration: the model is handed a still frame, which carries position but no velocity, so every shot eases its motion in from rest and the subject appears to speed up again at each seam. 3-6 usually swallows it; each frame costs 1/24 s of that shot. The shot's audio is trimmed by the same amount, so sound stays in sync.\n\nCosts nothing to try: trimming is done at decode time, so cached shots are re-used."}),
                "audio": (AUDIO_MODES, {"default": "concat", "advanced": True,
                                        "tooltip": "concat = decode each shot's audio, trim it to the shot's exact length, ramp the seams and normalise the whole track once. mute = silent track of the right length (still valid for Create Video). Needs 'audio_vae' wired; without it the track is silent."}),
                "seam_fade_ms": ("INT", {"default": 40, "min": 0, "max": 500, "advanced": True,
                                         "tooltip": "Fade-out/fade-in ramp on each audio seam, in ms. Kills the click where two independently generated soundtracks meet. Not a crossfade: nothing overlaps, so the audio never shifts against the picture."}),
                "cache": (CACHE_MODES, {"default": "disk", "advanced": True,
                                        "tooltip": "disk = cache each shot's sampled LATENTS under user/kinburg-nodes/minimax_shots (~7 MB per shot). Editing shot 5 then re-samples 5..N and replays 1..4. off = always re-sample."}),
                "cache_tag": ("STRING", {"default": "", "advanced": True,
                                         "tooltip": "Free text folded into every cache key. The key covers architecture + LoRA patches, NOT the exact weight file — so if you swap to another H3 checkpoint of the same size, bump this to invalidate."}),
                "live_preview": ("BOOLEAN", {"default": True,
                                             "tooltip": "Stream the in-loop LLM's writing to an 'LLM Live Log' node, one labelled block per shot ('refine 2/4 (opening)'). Only does anything when 'llm_config' is wired."}),
                "llm_keep_loaded": ("BOOLEAN", {"default": False, "advanced": True,
                                                "tooltip": "Leave the LLM in memory between shots instead of shutting its worker down after every call. Faster (no reload per seam) but it holds its VRAM and RAM while H3 samples — on 12 GB with a 26B model that means OOM. Off is the safe default: the LLM is loaded, used and killed around each shot."}),
                "shots_range": ("STRING", {"default": "", "advanced": True,
                                           "tooltip": "Render only part of the chain: '' = all, '3' = shot 3, '2-4' = shots 2..4. Shots after the range are skipped entirely; shots before it are still needed for the handoff frame (free if cached). Handy while you design the opening shots."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "The AUDIO vae. Without it the video comes out silent."}),
                "llm_config": (LLM_CONFIG, {"tooltip": "Optional 'Local LLM Settings (GGUF)' WITH a 'Vision Settings (GGUF)' mmproj. Wire it and the writer gets to see the frame each shot really starts on: just before sampling a shot whose first frame was inherited, the LLM reworks that shot's opening from the actual pixels instead of the forecast it was written against. Per-shot control is the 'refine' widget on 'Morpheus Dream' (the Storyboard node sets it for you). Leave empty and shots are sampled exactly as they arrive."}),
                "sigmas": ("SIGMAS", {"tooltip": "Optional explicit schedule (overrides steps/scheduler). Build it from a model that already has the sigma shift applied, or it won't match."}),
                "sampler": ("SAMPLER", {"tooltip": "Optional explicit sampler (overrides sampler_name)."}),
                "noise": ("NOISE", {"tooltip": "Optional noise source; its seed becomes the base seed, and each shot still gets its own offset. Leave empty to use the 'seed' widget."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "last_frame", "report", "prompts")
    OUTPUT_TOOLTIPS = ("Every frame of the storyboard, in order.",
                       "The whole soundtrack, normalised once and seam-ramped.",
                       "24.0 — the model's fixed rate, as a FLOAT so it drops straight into 'Create Video'.",
                       "The last frame — the cover, or the start_frame of a later run.",
                       "Per-shot table: frames, seam trim, seed, write/sample times, cache hits, warnings.",
                       "The prompts that were ACTUALLY sampled, ---separated — after any in-loop rework, so this is the text that made this video. Same format 'prompt_overrides' takes on the Storyboard node.")
    FUNCTION = "render"
    CATEGORY = "Kinburg-Nodes/sampling"
    DESCRIPTION = ("Samples a chain of 'Morpheus Dream' nodes into one long video with sound: each "
                   "shot's last frame becomes the next shot's first keyframe. Handles the 24 fps "
                   "frame grid, the seam frame, one global audio normalisation, and a disk cache of "
                   "sampled latents so editing one shot doesn't re-run the whole storyboard.")

    # ------------------------------------------------------------------------------------ helpers
    @staticmethod
    def _decode_video(vae, video_lat):
        img = vae.decode(video_lat.to(torch.float32))
        if img.ndim == 5:  # [B, T, H, W, C] → one flat batch of frames
            img = img.reshape(-1, *img.shape[-3:])
        return img.to("cpu", torch.float32)

    @staticmethod
    def _decode_audio(audio_vae, audio_lat, drop_frames, want, sr):
        """One shot's soundtrack, trimmed to exactly `want` samples. No normalisation here — that
        happens once for the whole track, or the level would step at every seam.

        `want` is measured from the storyboard's cumulative frame position, not from this shot's
        length: rounding each shot independently drifts by up to half a sample per seam, and over a
        long chain that walks the sound off the picture."""
        wav = audio_vae.decode(audio_lat.to(torch.float32)).movedim(-1, 1).to("cpu", torch.float32)
        if drop_frames:
            wav = wav[..., int(round(drop_frames * sr / h3.FPS)):]
        if wav.shape[-1] > want:
            wav = wav[..., :want]
        elif wav.shape[-1] < want:
            pad = torch.zeros(wav.shape[:-1] + (want - wav.shape[-1],), dtype=wav.dtype)
            wav = torch.cat([wav, pad], dim=-1)
        return wav.contiguous()

    @staticmethod
    def _callback(pbar, previewer, n_units, base):
        """Per-step callback that advances the storyboard-wide bar from `base` instead of from 0.
        A failed preview decode must never kill a 20-minute run, hence the swallow."""
        def cb(step, x0, x, total_steps):
            preview = None
            if previewer is not None:
                try:
                    xx = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                    preview = previewer.decode_latent_to_preview_image("JPEG", xx)
                except Exception:
                    preview = None
            pbar.update_absolute(base + step + 1, n_units, preview)
        return cb

    @staticmethod
    def _ramp_seams(chunks, ms, sr):
        n = int(round(sr * ms / 1000.0))
        if n <= 0:
            return
        for i, c in enumerate(chunks):
            k = min(n, c.shape[-1] // 2)
            if k <= 0:
                continue
            ramp = torch.linspace(0.0, 1.0, k, dtype=c.dtype)
            if i > 0:
                c[..., :k] *= ramp
            if i < len(chunks) - 1:
                c[..., -k:] *= ramp.flip(0)

    # ------------------------------------------------------------------------------------- render
    def render(self, shots, model, clip, vae, width, height, steps, scheduler, sampler_name,
               seed, shift_video, shift_audio, seam_trim, audio, seam_fade_ms, cache,
               cache_tag, live_preview=True, llm_keep_loaded=False, shots_range="",
               audio_vae=None, sigmas=None, sampler=None, noise=None, llm_config=None,
               unique_id=None):
        _require_h3()
        chain = _flatten_shots(shots)
        if not chain:
            raise ValueError("No shots wired — chain at least one 'Morpheus Dream'.")

        warn, t_run = [], time.time()
        lo, hi = _parse_range(shots_range, len(chain))
        cw, ch = _snap32(width), _snap32(height)
        if cw * ch > h3.MAX_PIXELS:
            warn.append(f"canvas {cw}x{ch} is above the model's {h3.MAX_PIXELS // 1000}k-pixel budget "
                        f"({h3.BASE_SHORT_EDGE} short edge, 768*1344 area) — expect artifacts and a slow run")

        # --- model / schedule --------------------------------------------------------------------
        pre_shift = (model.model_options.get("transformer_options", {})
                     .get("minimax_h3_sigma_shift_video"))
        if pre_shift is None:
            m = h3.MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio)).result[0]
            shift_note = f"applied here (video {shift_video:g} / audio {shift_audio:g})"
        else:
            m = model
            shift_note = f"already on the model (video {pre_shift:g}) — widgets ignored"

        if sigmas is None:
            sig = comfy.samplers.KSampler(m, steps=int(steps), device=m.load_device,
                                          sampler=sampler_name, scheduler=scheduler,
                                          denoise=1.0, model_options=m.model_options).sigmas
            sched_note = f"built here — {int(steps)} steps, {scheduler}"
        else:
            sig = sigmas
            sched_note = f"wired — {sig.shape[-1] - 1} steps"
            if pre_shift is None:
                warn.append("'sigmas' is wired but the model carried no sigma shift, so the schedule "
                            "was almost certainly built on an unshifted model — either wire 'MiniMax "
                            "H3 Sigma Shift' before the scheduler, or drop the sigmas input")
        sampler_obj = sampler if sampler is not None else comfy.samplers.sampler_object(sampler_name)
        base_seed = int(getattr(noise, "seed", seed)) if noise is not None else int(seed)

        # --- plan: frames per shot, what gets dropped, what lands in the output ------------------
        plan = []
        trim = max(0, int(seam_trim))
        for i, s in enumerate(chain):
            has_start = s.get("start_frame") is not None or (i > 0 and s["link"] == "continue")
            # frame 0 duplicates the previous shot's tail; the frames after it are the model
            # accelerating its motion from a standstill, because a keyframe carries no velocity
            drop = min(trim, s["frames"] - 5) if (i > 0 and has_start) else 0
            plan.append({"frames": s["frames"], "drop": drop, "out": s["frames"] - drop,
                         "in_range": lo <= i + 1 <= hi})
            if not (124 <= s["frames"] <= 362):
                warn.append(f"shot {i + 1}: {s['frames']} frames ({_seconds(s['frames']):.2f} s) is "
                            f"outside the model's trained 124-362 frame range")
            img = s.get("start_frame")
            if img is not None:
                r_img, r_c = int(img.shape[2]) / int(img.shape[1]), cw / ch
                if abs(r_img - r_c) / r_c > 0.02:
                    warn.append(f"shot {i + 1}: start_frame is {img.shape[2]}x{img.shape[1]} "
                                f"({r_img:.2f}) but the canvas is {cw}x{ch} ({r_c:.2f}) — H3 stretches "
                                f"the first frame, it does not crop, so this shot will be distorted")
        total_out = sum(p["out"] for p in plan if p["in_range"])

        # which shots actually run: the ones in the output range, plus any earlier shot whose tail
        # frame the next one inherits. Decided up front so the progress bar knows its true total.
        for i in range(len(chain)):
            plan[i]["next_needs"] = (i + 1 < len(chain) and i + 1 < hi
                                     and chain[i + 1].get("start_frame") is None
                                     and chain[i + 1]["link"] == "continue")
            plan[i]["process"] = i + 1 <= hi and (plan[i]["in_range"] or plan[i]["next_needs"])

        # --- cache keys (causal: a shot's key folds in the previous shot's) ----------------------
        env = shot_cache.key(
            shot_cache.fingerprint(getattr(m.model, "diffusion_model", m.model)),
            shot_cache.fingerprint(getattr(clip, "cond_stage_model", None), getattr(clip, "patcher", None)),
            shot_cache.fingerprint(getattr(vae, "first_stage_model", None)),
            cw, ch, base_seed, sampler_name if sampler is None else "wired-sampler",
            shot_cache.tensor_key(sig.cpu() if hasattr(sig, "cpu") else None),
            shift_video if pre_shift is None else pre_shift, shift_audio, cache_tag,
        )
        def shot_key(prev_key, i, s, prompt):
            """A shot's key. Computed IN the loop, not up front: with an LLM in the loop the prompt
            is written from the previous shot's rendered frame, so it doesn't exist until then."""
            return shot_cache.key(prev_key, i, prompt, s["frames"], s["link"], s["seed_offset"],
                                  s["keyframe_strength"], shot_cache.tensor_key(s.get("start_frame")),
                                  shot_cache.tensor_key(s.get("end_frame")))

        # --- the in-loop writer (optional): it exists to replace forecasts with real frames --------
        refine, emit = None, None
        if llm_config:
            from . import refine as refine_mod  # deferred: refine -> storyboard -> nodes is a cycle
            refine = refine_mod
            if live_preview:
                try:
                    from server import PromptServer
                    nid = str(unique_id[0] if isinstance(unique_id, list) else unique_id)

                    def emit(payload):
                        try:
                            PromptServer.instance.send_sync("kinburg.llm", {"id": nid, **payload})
                        except Exception:
                            pass
                except Exception:
                    emit = None

        # --- the loop ---------------------------------------------------------------------------
        sr = int(getattr(audio_vae, "audio_sample_rate_output",
                         getattr(audio_vae, "audio_sample_rate", DEFAULT_AUD_SR))) if audio_vae else DEFAULT_AUD_SR
        want_audio = audio == "concat" and audio_vae is not None
        if audio == "concat" and audio_vae is None:
            warn.append("audio = concat but no 'audio_vae' is wired — the track is silent")

        # ONE progress bar for the whole storyboard: `shots x steps` units, only ever moving
        # forward. Per-shot bars were the first thing that confused a real run — the bar reached
        # 100%, reset, and did it again, so "nearly done" meant nothing.
        steps_each = max(1, int(sig.shape[-1]) - 1)
        n_units = max(1, sum(1 for p in plan if p["process"])) * steps_each
        pbar = comfy.utils.ProgressBar(n_units)
        try:
            previewer = latent_preview.get_previewer(m.load_device, m.model.latent_format)
        except Exception:
            previewer = None

        images, cursor, handoff, rows, wav_chunks = None, 0, None, [], []
        hits, done_units, cache_errs = 0, 0, []
        prompts_out, prev_key, wrote_any = [], env, False
        for i, s in enumerate(chain):
            if i + 1 > hi:
                break
            comfy.model_management.throw_exception_if_processing_interrupted()
            p, t_shot = plan[i], time.time()
            seed_i = (base_seed + i + s["seed_offset"]) & 0xffffffffffffffff
            next_needs = p["next_needs"]
            prompt, rmode, t_write = s["prompt"], "—", 0.0
            if not p["process"]:
                # still has to advance the causal chain, or every later key would shift
                prev_key = shot_key(prev_key, i, s, prompt)
                logging.info(f"[Morpheus] shot {i + 1}/{len(chain)} — skipped (outside "
                             f"shots_range, nothing downstream needs it)")
                continue

            start = s.get("start_frame")
            inherited = False
            if start is None and i > 0 and s["link"] == "continue":
                start, inherited = handoff, handoff is not None
            end = s.get("end_frame")

            # --- rework this shot's text now that its real first frame exists --------------------
            if refine is not None:
                scope = refine.resolve_scope(s.get("refine", "auto"), prompt, inherited,
                                             start is not None)
                if scope in ("opening", "full"):
                    t0 = time.time()
                    tag = f"refine {i + 1}/{len(chain)} ({scope})"
                    logging.info(f"[Morpheus] shot {i + 1}/{len(chain)} — {tag}")
                    text_key = shot_cache.key(prev_key, i, "refine", scope, prompt,
                                              shot_cache.tensor_key(start),
                                              shot_cache.tensor_key(end))
                    new_prompt, err = refine.refine_prompt(
                        llm_config, prompt, scope, start, end, _seconds(p["frames"]),
                        direction=prompt, unload_comfy=not llm_keep_loaded, emit=emit, tag=tag,
                        cache_key=text_key, use_cache=cache == "disk")
                    t_write = time.time() - t0
                    if err:
                        warn.append(f"shot {i + 1}: the in-loop writer failed ({err}) — the shot was "
                                    f"sampled with the prompt it arrived with")
                        rmode = f"{scope} failed"
                    elif new_prompt:
                        prompt, rmode, wrote_any = new_prompt, scope, True
                    if not llm_keep_loaded:
                        # kill the worker before H3 comes back: on 12 GB they cannot coexist
                        try:
                            _shutdown_llm()
                        except Exception:
                            pass

            key_i = shot_key(prev_key, i, s, prompt)
            prev_key = key_i
            entry = shot_cache.load(key_i) if cache == "disk" else None
            cached = entry is not None
            if cached:
                hits += 1
                video_lat, audio_lat = entry["video"], entry.get("audio")
                logging.info(f"[Morpheus] shot {i + 1}/{len(chain)} — cached ({p['frames']} frames)")
            else:
                logging.info(f"[Morpheus] shot {i + 1}/{len(chain)} — sampling {p['frames']} frames "
                             f"({_seconds(p['frames']):.2f} s) at {cw}x{ch}, seed {seed_i}")
                cond, latent = h3.MiniMaxH3ImageToVideo.execute(
                    clip=clip, vae=vae, prompt=prompt, width=cw, height=ch,
                    length=p["frames"], first_frame=start, last_frame=end).result
                if (start is not None or end is not None) and s["keyframe_strength"] < 1.0:
                    cond = node_helpers.conditioning_set_values(
                        cond, {"minimax_visual_cond_noise_aug": s["keyframe_strength"]})

                guider = Guider_Basic(m)
                guider.set_conds(cond)
                gen = Noise_RandomNoise(seed_i)
                cb = self._callback(pbar, previewer, n_units, done_units)
                out = guider.sample(gen.generate_noise(latent), latent["samples"], sampler_obj, sig,
                                    denoise_mask=None, callback=cb,
                                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed_i)
                out = out.to(comfy.model_management.intermediate_device())
                video_lat, audio_lat = out.unbind()
            done_units += steps_each  # a cached shot jumps its whole slice at once
            pbar.update_absolute(done_units, n_units)

            # decode only when the pixels are actually needed: for the output, or for the next
            # shot's keyframe when the cache didn't already hand us that frame
            have_handoff = cached and entry.get("handoff") is not None
            decoded, tail = None, None
            if p["in_range"] or (next_needs and not have_handoff):
                decoded = self._decode_video(vae, video_lat)
                tail = _fp16_round(decoded[-1:])
            if next_needs:
                handoff = entry["handoff"].to(torch.float32) if have_handoff else tail

            if p["in_range"]:
                frames = decoded[p["drop"]:]
                if images is None:
                    images = torch.empty((total_out,) + tuple(frames.shape[1:]), dtype=torch.float32)
                if frames.shape[1:] != images.shape[1:]:
                    raise RuntimeError(f"shot {i + 1} decoded to {tuple(frames.shape[1:])} but the "
                                       f"storyboard is {tuple(images.shape[1:])} — all shots must "
                                       f"share one canvas")
                room = images.shape[0] - cursor
                if frames.shape[0] > room:  # only if the VAE's frame math ever disagrees with h3's
                    warn.append(f"shot {i + 1} decoded to {frames.shape[0]} frames, {frames.shape[0] - room} "
                                f"more than planned — the tail was cut to fit the storyboard")
                    frames = frames[:room]
                at = cursor
                images[cursor:cursor + frames.shape[0]] = frames
                cursor += frames.shape[0]
                if want_audio:
                    # exact sample span of this shot's slot in the finished track
                    want = int(round(cursor * sr / h3.FPS)) - int(round(at * sr / h3.FPS))
                    wav_chunks.append(
                        torch.zeros((1, 2, want)) if audio_lat is None else
                        self._decode_audio(audio_vae, audio_lat, p["drop"], want, sr))

            if not cached and cache == "disk":
                # the metadata is the forensic trail: when a finished video looks wrong, the first
                # question is always "what did this shot actually get?"
                err = shot_cache.save(key_i, video_lat, audio_lat, tail,
                                      {"shot": i + 1, "frames": p["frames"], "seed": seed_i,
                                       "first_frame": ("wired" if s.get("start_frame") is not None
                                                       else ("inherited" if start is not None else "none")),
                                       "last_frame": "frame" if end is not None else "none",
                                       "refined": rmode, "prompt": prompt[:1500]})
                if err:
                    cache_errs.append(f"shot {i + 1}: {err.splitlines()[0][:160]}")
            del video_lat, audio_lat, decoded

            prompts_out.append(prompt)
            rows.append((i + 1, p, seed_i, time.time() - t_shot - t_write, t_write, cached, rmode,
                         "wired" if s.get("start_frame") is not None else
                         ("inherited" if (i > 0 and s["link"] == "continue") else "text")))

        if images is None:
            raise RuntimeError(f"shots_range {shots_range!r} selected nothing to render")
        if cursor != images.shape[0]:  # a shot decoded to a different length than planned
            images = images[:cursor]

        # --- audio: one normalisation for the whole track, ramps at the seams -------------------
        if wav_chunks:
            self._ramp_seams(wav_chunks, seam_fade_ms, sr)
            track = torch.cat(wav_chunks, dim=-1)
            std = torch.std(track, dim=[1, 2], keepdim=True) * 5.0
            std[std < 1.0] = 1.0
            track = track / std
        else:
            track = torch.zeros((1, 2, int(round(images.shape[0] * sr / h3.FPS))))
        audio_out = {"waveform": track, "sample_rate": sr}

        if cache == "disk":
            shot_cache.prune()

        if cache_errs:
            warn.append("the shot cache could NOT be written, so nothing will replay next run: "
                        + "; ".join(cache_errs))
        report = self._report(rows, images, cw, ch, sr, shift_note, sched_note, sampler_name,
                              sampler, base_seed, cache, hits, lo, hi, len(chain), warn,
                              time.time() - t_run, want_audio, seam_fade_ms,
                              llm_config is not None, wrote_any, seam_trim)
        logging.info(f"[Morpheus] storyboard done: {images.shape[0]} frames "
                     f"({_seconds(images.shape[0]):.2f} s) in "
                     f"{_format_elapsed(time.time() - t_run, 'human')}")
        last = images[-1:].clone()
        return (images, audio_out, float(h3.FPS), last, report, "\n\n---\n\n".join(prompts_out))

    # ------------------------------------------------------------------------------------- report
    @staticmethod
    def _report(rows, images, cw, ch, sr, shift_note, sched_note, sampler_name, sampler,
                base_seed, cache, hits, lo, hi, n_shots, warn, elapsed, want_audio, fade_ms,
                has_llm, wrote_any, trim):
        n = images.shape[0]
        out = [f"Morpheus — {n} frames = {_seconds(n):.2f} s @ {h3.FPS} fps · {cw}x{ch}",
               f"shots: {n_shots} in the chain, rendered {lo}-{hi}"
               + (f" ({hits} from cache)" if hits else ""),
               f"sigma shift: {shift_note}",
               f"schedule: {sched_note}",
               f"sampler: {'wired SAMPLER' if sampler is not None else sampler_name}",
               f"seeds: base {base_seed} + shot index + per-shot offset",
               f"seam trim: {trim} frame(s) off each continuing shot",
               f"audio: {'decoded, one global normalisation, ' + str(fade_ms) + ' ms seam ramps, ' + str(sr) + ' Hz' if want_audio else 'silent'}",
               f"in-loop writer: " + ("wired" + (" — some shots were reworked from their real first "
                                                 "frame" if wrote_any else " — nothing needed reworking")
                                      if has_llm else "not wired (shots sampled as they arrived)"),
               f"cache: {cache}" + (f" ({shot_cache.cache_dir()})" if cache == "disk" else ""),
               f"total: {_format_elapsed(elapsed, 'human')}",
               "",
               "  #  frames    dur   start       refined     write     sample",
               "  " + "-" * 62]
        for idx, p, seed_i, secs, w_secs, cached, rmode, start in rows:
            mark = "" if p["in_range"] else "  (handoff only)"
            drop = f" -{p['drop']}" if p["drop"] else "   "
            out.append(f"  {idx:<2} {p['frames']:>5}{drop} {_seconds(p['out']):>6.2f}s "
                       f"{start:<11} {rmode:<11} "
                       f"{(_format_elapsed(w_secs, 'human') if w_secs else '—'):<9} "
                       f"{('cached' if cached else _format_elapsed(secs, 'human'))}{mark}")
        if warn:
            out += ["", "warnings:"] + [f"  · {w}" for w in warn]
        return "\n".join(out)


NODE_CLASS_MAPPINGS = {
    "KinburgMorpheusDream": KinburgMorpheusDream,
    "KinburgMorpheus": KinburgMorpheus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KinburgMorpheusDream": "Morpheus Dream 🌙",
    "KinburgMorpheus": "Morpheus (Video Sampler) 🌙",
}
