"""Phantas 🎞 — renders a board's keyframes and hands Morpheus the chain to dream between.

The sampling half. It walks the board once, generating one still per keyframe, and returns a
`MORPHEUS_SHOT` chain whose prompts are left EMPTY — which is what tells `Morpheus Storyboard`
"these are yours to write". Neighbouring shots share the *same tensor object* for the frame between
them, because that frame is one picture playing two parts: the end of one shot and the start of the
next.

**Consistency is structural here, not cosmetic.** A shot is generated as the movement from its first
keyframe to its last, so two neighbours that disagree do not look "slightly different" — they
describe a shot that morphs halfway through. A shared seed is the weakest of the levers (same
starting noise, but the trajectories diverge within a few steps), so it is the floor rather than the
mechanism. The real ones, in order:

  * the identical style block the board stamps on every prompt,
  * an **anchor**: an already-rendered frame fed back in as a reference, which is what `reference`
    turns on.

Two reference mechanisms are supported, both of them stock ComfyUI, because they are what different
model families actually offer:

  * **redux** — the anchor goes through CLIP-Vision and `StyleModelApply` (Flux Redux). Works on
    anything in the Flux dev family, Krea included, which is the point: it does not require giving
    up the model you like. Applied with `attn_bias`, so `reference_strength` below 1.0 attenuates
    the reference's grip and leaves the text in charge — at full strength Redux tends to redraw its
    reference and every frame comes out the same picture.
  * **edit** — the anchor is VAE-encoded and appended as a `reference_latents` entry, the way
    Kontext and Qwen-Image-Edit read their input image. Strongest identity carry, but it means
    running an edit model.

The anchor itself is chosen by `anchor`: the first frame holds global identity, the previous frame
holds the seam, and both together do both at the cost of one more encode. A wired `reference_image`
is always added on top — including for frame 1, which otherwise has nothing to hold onto.

Rendered frames are cached on disk under a causal key. That is not only a speed feature: ComfyUI's
Cancel raises inside the sampler and discards the whole run, so without a cache a stopped board
loses every frame it had finished. With one, Cancel *is* the stop button — restart and the finished
frames come straight back. It is also what makes `redo` cheap.
"""
import json
import logging
import random
import time

import torch

import node_helpers

from . import timing
from .board import PHANTAS_BOARD
from ..morpheus.nodes import MORPHEUS_SHOT
from ..ouroboros.nodes import SAMPLER_CFG, _sample_stage, _seed_for
from ..util import diskcache
from ..util.images import fp16_round, log_uris
from ..categories import CAT_PHANTAS

REFERENCE_MODES = ["off", "redux", "edit"]
ANCHOR_MODES = ["first frame", "previous frame", "first + previous"]
CACHE_MODES = ["disk", "off"]

_store = diskcache.Store("phantas_frames", "Phantas")


def parse_selection(text, n):
    """Which frames to re-roll. `""` → none. `"3"` → just 3. `"2-4"` → 2,3,4. `"1,4-6"` → the union.

    1-based on the way in, 0-based on the way out, clamped to the board. Empty means *nothing is
    forced* rather than everything: a frame that is simply absent from the cache renders anyway, so
    "all of them" is already the behaviour of a cold run."""
    text = (text or "").strip()
    if not text:
        return set()
    out = set()
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                lo = int(a) if a.strip() else 1
                hi = int(b) if b.strip() else n
            else:
                lo = hi = int(part)
        except ValueError:
            raise ValueError(f"redo: expected '' or '3' or '2-4' or '1,4-6', got {part!r}")
        if hi < lo:
            lo, hi = hi, lo
        out.update(range(max(1, lo), min(n, hi) + 1))
    return {i - 1 for i in out}


def anchors_for(i, images, mode, external):
    """Which already-existing pictures frame `i` is conditioned on.

    Frame 0 has no rendered neighbour, so in every mode it gets only whatever was wired in — which
    is precisely why `reference_image` exists."""
    refs = [external] if external is not None else []
    if i > 0:
        if mode in ("first frame", "first + previous") and images:
            refs.append(images[0])
        if mode in ("previous frame", "first + previous"):
            refs.append(images[i - 1])
    # first == previous on frame 1: conditioning on the same picture twice only doubles its weight
    out = []
    for r in refs:
        if not any(r is o for o in out):
            out.append(r)
    return out


def apply_reference(cond, refs, mode, strength, style_model=None, clip_vision=None, vae=None):
    """Attach the anchors to a conditioning, by whichever mechanism the model family offers."""
    if mode == "off" or not refs:
        return cond
    if mode == "redux":
        if style_model is None or clip_vision is None:
            raise ValueError("reference = 'redux' needs both a 'style_model' (Flux Redux) and a "
                             "'clip_vision' input. Load them with StyleModelLoader and "
                             "CLIPVisionLoader, or set reference = 'off'.")
        from nodes import StyleModelApply
        apply = StyleModelApply().apply_stylemodel
        for ref in refs:
            out = clip_vision.encode_image(ref)
            # attn_bias, not multiply: it biases how hard the model attends to the reference tokens,
            # which is the dial that keeps the prompt in charge of what the frame SHOWS
            cond = apply(cond, style_model, out, float(strength), "attn_bias")[0]
        return cond
    if mode == "edit":
        if vae is None:
            raise ValueError("reference = 'edit' needs the VAE (it is already required).")
        lats = [vae.encode(ref[:, :, :, :3]) for ref in refs]
        return node_helpers.conditioning_set_values(cond, {"reference_latents": lats}, append=True)
    raise ValueError(f"reference: '{mode}' is not one of {REFERENCE_MODES}")


def _stages(sampler_settings):
    """SAMPLER_CFG arrives as one dict or an already-chained list; normalise to a list."""
    if isinstance(sampler_settings, list):
        out = [s for s in sampler_settings if isinstance(s, dict)]
    elif isinstance(sampler_settings, dict):
        out = [sampler_settings]
    else:
        out = []
    if not out:
        raise ValueError("Phantas needs a 'Sampler Settings' bundle wired into sampler_settings.")
    return out


class KinburgPhantas:
    """Renders a Phantas board into keyframes and a Morpheus shot chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "board": (PHANTAS_BOARD, {"tooltip": "The board from 'Phantas Storyboard'."}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "sampler_settings": (SAMPLER_CFG, {"tooltip": "A 'Sampler Settings' bundle — the same one Ouroboros and Chimera take. Chain two of them for a draft-then-polish pass on every frame (give the later stage denoise < 1). Its seed_mode decides whether the frames share a seed; 'fixed' (the default) is what keeps them related."}),
                "width": ("INT", {"default": 960, "min": 64, "max": 8192, "step": 8, "tooltip": "Keyframe width. Match the aspect ratio you will render the video at — these pictures are also what the video model's writer looks at."}),
                "height": ("INT", {"default": 544, "min": 64, "max": 8192, "step": 8, "tooltip": "Keyframe height."}),
                "reference": (REFERENCE_MODES, {"default": "off", "tooltip": "How already-rendered frames are fed back in, to keep the sequence one world:\n\n• off — text and seed only. The weakest, and on a keyframe chain it usually shows.\n\n• redux — CLIP-Vision + a Flux Redux style model. Works with any Flux dev model, Krea included.\n\n• edit — the anchor goes in as a reference latent, the way Kontext and Qwen-Image-Edit read an input image. The strongest, and it means running one of those models."}),
                "reference_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How hard the anchor pulls, in 'redux' mode (applied as an attention bias). 1.0 is full Redux, which tends to redraw its reference and make every frame the same picture; lower values keep the identity while leaving the prompt in charge of the shot. Ignored by the other modes."}),
                "anchor": (ANCHOR_MODES, {"default": "first + previous", "tooltip": "Which rendered frame is used as the anchor:\n\n• first frame — global identity, but neighbours can still drift apart.\n• previous frame — tight seams, but drift accumulates down a long chain.\n• first + previous — both, at the cost of one more encode per frame.\n\nA wired 'reference_image' is always added on top, and is the only anchor frame 1 can have."}),
                "cache": (CACHE_MODES, {"default": "disk", "tooltip": "Cache rendered frames on disk under a causal key. Worth leaving on: ComfyUI's Cancel discards the whole run, so without this a stopped board loses every finished frame — with it, cancel and re-run picks up where it stopped. It is also what makes 'redo' cheap."}),
                "redo": ("STRING", {"default": "", "tooltip": "Re-roll these keyframes, 1-based: '3', '2-4', '1,4-6'. Empty = re-roll nothing (a frame missing from the cache still renders, so a cold board renders in full either way).\n\nA listed frame is generated again with its seed shifted by 'redo_seed_offset' — the same seed and the same prompt would only give the same picture back. The frames AFTER it re-render too, because they were anchored to the picture it used to be; the ones before it stay cached."}),
                "redo_seed_offset": ("INT", {"default": 1, "min": 0, "max": 0xffffffff, "tooltip": "Added to the seed of every keyframe named in 'redo'. Bump it again for another roll of the same frame."}),
            },
            "optional": {
                "style_model": ("STYLE_MODEL", {"tooltip": "Flux Redux, for reference = 'redux'."}),
                "clip_vision": ("CLIP_VISION", {"tooltip": "The CLIP-Vision encoder Redux was trained with (sigclip vision 384), for reference = 'redux'."}),
                "first_frame": ("IMAGE", {"tooltip": "Use this picture AS keyframe 1 instead of generating it — a photo, a frame from an earlier render, a picture out of a chat. It is resized to width×height and anchors everything after it."}),
                "reference_image": ("IMAGE", {"tooltip": "An extra anchor added to every frame, including the first. This is how a character reference or a location plate gets into the whole board."}),
                "negative": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative prompt. Empty = the board's own [NEGATIVE] block, which is what the style bible wrote."}),
                "live_preview": ("BOOLEAN", {"default": True, "tooltip": "Push each finished keyframe to a 'Kinburg Live Log' node as soon as it is decoded. No wiring — drop a log node anywhere."}),
                "cache_tag": ("STRING", {"default": "", "tooltip": "Any text you like, folded into the cache key. Two different checkpoints of the same architecture and size look identical to the cache; put something here to tell them apart."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (MORPHEUS_SHOT, "IMAGE", "STRING", "GEN_SETTINGS", "STRING")
    RETURN_NAMES = ("shots", "images", "captions", "settings_data", "report")
    OUTPUT_TOOLTIPS = ("The Morpheus chain: every shot bounded by two of these keyframes, prompts "
                       "left empty. Wire it into 'Morpheus Storyboard', which fills them in place.",
                       "The keyframes themselves, as one batch.",
                       "'Keyframe n' per image, for Image Compare.",
                       "Per-image settings for Image Compare.",
                       "What was rendered, what came from cache, and how long each took.")
    FUNCTION = "render"
    CATEGORY = CAT_PHANTAS
    DESCRIPTION = ("Renders a Phantas board's keyframes on one seed and one style, and returns the "
                   "Morpheus shot chain that runs between them.")

    def render(self, board, model, clip, vae, sampler_settings, width, height, reference,
               reference_strength, anchor, cache, redo, redo_seed_offset=1, style_model=None, clip_vision=None,
               first_frame=None, reference_image=None, negative="", live_preview=True,
               cache_tag="", unique_id=None):
        from nodes import CLIPTextEncode, EmptyLatentImage, VAEDecode

        if not isinstance(board, dict) or not board.get("frames"):
            raise ValueError("Phantas: the 'board' input is empty — wire a 'Phantas Storyboard'.")
        frames_plan = board["frames"]
        shots_plan = board.get("shots") or []
        n = len(frames_plan)
        if len(shots_plan) != n - 1:
            raise ValueError(f"Phantas: the board has {n} keyframes but {len(shots_plan)} shots; "
                             f"a chain of {n} frames has exactly {n - 1}.")
        stages = _stages(sampler_settings)
        neg_text = (negative or "").strip() or board.get("negative", "")
        forced = parse_selection(redo, n)
        use_cache = cache == "disk"

        emit = None
        if live_preview:
            nid = str(unique_id) if unique_id is not None else "phantas"

            def emit(payload):
                try:
                    from server import PromptServer
                    PromptServer.instance.send_sync("kinburg.llm", {"id": nid, **payload})
                except Exception:  # pragma: no cover - a headless run has no server
                    pass

        clip_enc, vae_dec = CLIPTextEncode(), VAEDecode()
        base_latent = EmptyLatentImage().generate(int(width), int(height), 1)[0]
        external = fp16_round(reference_image) if reference_image is not None else None

        # Everything that changes what a frame looks like but is not the frame's own prompt.
        env = diskcache.key(
            diskcache.fingerprint(getattr(model, "model", model)),
            diskcache.fingerprint(getattr(clip, "cond_stage_model", None), getattr(clip, "patcher", None)),
            diskcache.fingerprint(getattr(vae, "first_stage_model", None)),
            width, height, neg_text, reference, reference_strength, anchor, cache_tag,
            diskcache.fingerprint(style_model), diskcache.tensor_key(external),
            json.dumps(stages, sort_keys=True, default=str))

        rng = random.Random(int(stages[0].get("seed", 0)))
        images, settings, report, prev_key = [], [], [], env
        t_run = time.time()
        for i, fr in enumerate(frames_plan):
            prompt = fr.get("prompt", "")
            # Worked out for EVERY frame, cached or not, so a 'random' seed_mode walks the same
            # sequence whether or not the disk had the picture already.
            bump = int(redo_seed_offset) if i in forced else 0
            seeds = [(_seed_for(int(s.get("seed", 0)), s.get("seed_mode", "fixed"),
                                int(s.get("seed_step", 1)), i, rng) + bump) % (2 ** 64)
                     for s in stages]
            # causal: this frame is conditioned on EARLIER frames, so its key must carry them all —
            # which is also what makes `redo` work. A re-rolled frame changes the keys of every
            # frame after it, because they were anchored to the picture it used to be.
            key_i = diskcache.key(prev_key, i, prompt, seeds,
                                  diskcache.tensor_key(first_frame) if i == 0 else "")
            prev_key = key_i

            if i == 0 and first_frame is not None:
                img = fp16_round(_fit(first_frame, int(width), int(height)))
                report.append(f"frame 1/{n}: wired in")
            else:
                hit = _store.load(key_i, require="image") if use_cache else None
                if hit is not None:
                    img = hit["image"].to(torch.float32)
                    report.append(f"frame {i + 1}/{n}: from cache")
                else:
                    t0 = time.time()
                    pos = clip_enc.encode(clip, prompt)[0]
                    refs = anchors_for(i, images, anchor, external)
                    pos = apply_reference(pos, refs, reference, reference_strength,
                                          style_model, clip_vision, vae)
                    neg = clip_enc.encode(clip, neg_text)[0]
                    lat = base_latent
                    for stg, s_seed in zip(stages, seeds):
                        lat = _sample_stage(model, lat, pos, neg, stg, s_seed)
                    img = fp16_round(vae_dec.decode(vae, lat)[0])
                    took = time.time() - t0
                    if use_cache:
                        _store.save(key_i, {"image": img}, meta={"frame": i + 1, "seed": seeds[0]})
                    report.append(f"frame {i + 1}/{n}: rendered in {took:.1f}s "
                                  f"(seed {seeds[0]}, {len(refs)} anchor(s))")
            images.append(img)
            if emit:
                emit({"event": "frames", "label": f"Phantas · keyframe {i + 1}/{n}",
                      "images": log_uris([img])})
            settings.append([
                {"key": "Phantas.keyframe", "value": f"{i + 1} of {n}"},
                {"key": "Phantas.seed", "value": str(seeds[0])},
                {"key": "Phantas.size", "value": f"{int(width)}x{int(height)}"},
                {"key": "Phantas.reference", "value": f"{reference} · {anchor}"
                                                      + (f" @ {reference_strength}" if reference == "redux" else "")},
                {"key": "Phantas.sampler", "value": f"{stages[0].get('sampler_name')} / "
                                                    f"{stages[0].get('scheduler')} / "
                                                    f"{stages[0].get('steps')} steps"},
            ])

        chain = []
        for i, s in enumerate(shots_plan):
            chain.append({
                "prompt": "",                       # empty = Morpheus Storyboard writes this one
                "beat": s.get("beat", ""),
                "frames": int(s.get("frames") or timing.frames_for(timing.DEFAULT_SECONDS)),
                "link": (board.get("links") or ["continue"] * len(shots_plan))[i],
                "seed_offset": 0,
                "keyframe_strength": 0.999,
                # the SAME object on both sides of a boundary: one picture, two parts
                "start_frame": images[i],
                "end_frame": images[i + 1],
                "refine": "auto",
            })

        shot_frames = [c["frames"] for c in chain]
        report.append(timing.describe(shot_frames))
        report.extend("⚠ " + w for w in timing.range_warnings(shot_frames))
        report.append(f"{n} keyframes in {time.time() - t_run:.1f}s")
        if use_cache:
            _store.prune()
        logging.info(f"[Phantas] {n} keyframes, {len(chain)} shots")

        batch = torch.cat(images, dim=0) if len(images) > 1 else images[0]
        captions = "\n".join(f"Keyframe {i + 1}" for i in range(n))
        return (chain, batch, captions, json.dumps(settings), "\n".join(report))


def _fit(image, width, height):
    """A wired picture onto the board's canvas: cover-cropped, so nothing is stretched."""
    import comfy.utils
    img = image[:1]
    _, h, w, _ = img.shape
    if (w, h) == (width, height):
        return img
    scale = max(width / float(w), height / float(h))
    nw, nh = max(width, int(round(w * scale))), max(height, int(round(h * scale)))
    up = comfy.utils.common_upscale(img.movedim(-1, 1), nw, nh, "lanczos", "disabled")
    y, x = (nh - height) // 2, (nw - width) // 2
    return up[:, :, y:y + height, x:x + width].movedim(1, -1)


NODE_CLASS_MAPPINGS = {"KinburgPhantas": KinburgPhantas}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgPhantas": "Phantas 🎞"}
