"""Audio SR — bandwidth extension for a finished mix, at 48 kHz, with a progress bar that works.

AudioSR is a latent-diffusion model that invents the top end of a recording rather than filtering it:
fed a track that dies at 11 kHz it writes plausible 11-24 kHz content back in. The model is vendored
under `vendor/audiosr` (see `vendor/NOTICE.md`); this node is the part that makes it behave inside
ComfyUI, and there were three specific things to fix.

**Progress.** The wrapper this replaces called `model_management.get_progress_state()` and
`comfy.model_management.update_progress()` — neither of which exists in ComfyUI — inside a bare
`except Exception: pass`, so it silently did nothing on every call and the node looked hung for
minutes. The time goes in the DDIM loop, `ddim_steps` iterations per chunk, so that is where the bar
is driven from: `vendor`'s `ddim.py` carries one `STEP_HOOK` and this node fills it in. The bar
therefore counts `chunks x steps`, which is the real unit of work.

**Cancelling.** For the same reason: interruption used to be checked only *between* chunks, so a
cancel could sit unhonoured for the length of a chunk. The hook checks every step.

**The speechbrain landmine.** Unrelated to this model, but this is where it went off — see
`util/imports.py`. One line at the top of `run()` defuses it.

Chunking is upstream's: overlapping windows, a Hann crossfade over the overlap, and each chunk
rescaled back to its own input peak (the model does not preserve level). The one deliberate
difference is that the plan is computed up front, so the last chunk cannot come out short and the
progress total is known before the first step rather than guessed.
"""
import gc
import time

import torch

from ..timer.timer_nodes import _format_elapsed
from ..util.imports import defuse_lazy_modules

SR = 48000                 # what AudioSR outputs, always
FOLDER = "AudioSR"         # ComfyUI/models/AudioSR
MODEL_NAMES = ("basic", "speech")
DTYPES = ("fp32", "fp16", "bf16")
_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# The model consumes audio in whole multiples of this: `make_batch_for_super_resolution` pads a chunk
# up to the next 5.12 s, so a chunk length that is not a multiple of it wastes compute on padding.
CHUNK_QUANTUM = 5.12

# `audiosr.pipeline.seed_everything` calls `np.random.seed`, which raises above 2**32-1 — so this is
# not the usual 0xffffffffffffffff. The widget is capped so ComfyUI's randomiser cannot land outside
# the range, and anything arriving on the wire is folded in `_seed32` (a 64-bit song seed from Siren
# Cast is exactly the sort of thing that gets wired here).
SEED_MAX = 0xffffffff

ST_MS = "mid/side (keep the image)"
ST_MONO = "sum to mono"
STEREO = [ST_MS, ST_MONO]

_cache = {"key": None, "model": None}


# ------------------------------------------------------------------------------------- vendored pkg
def _vendor_on_path():
    """Put `vendor/` on `sys.path` so the model imports as a plain top-level `audiosr`.

    The vendored code uses absolute imports, which cannot name this pack — its folder has a hyphen in
    it. Rewriting 76 files to relative imports would touch every line of a third-party tree for no
    gain, so the prefix was rewritten to `audiosr` once and the directory holding it goes on the path.
    An `audiosr` that is already importable (a real pip install) is left to win, so this never
    shadows a deliberate installation."""
    import os
    import sys
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    if here not in sys.path:
        sys.path.append(here)
    return here


# -------------------------------------------------------------------------------- chunk arithmetic
def _chunk_plan(total, chunk, overlap):
    """`[(start, end), …]` covering `total` samples, computed up front.

    Every window is exactly `chunk` long — the tail one is pulled BACK to end at `total` rather than
    padded with silence, so the model never spends a step denoising zeros and the last seconds of a
    track get the same treatment as the rest. Consecutive windows advance by `chunk - overlap`, and
    the final window may overlap its neighbour by more than that, which the crossfade handles the
    same way."""
    chunk = max(1, int(chunk))
    overlap = max(0, min(int(overlap), chunk - 1))
    if total <= chunk:
        return [(0, total)]
    step = chunk - overlap
    starts = list(range(0, max(1, total - chunk + 1), step))
    if starts[-1] + chunk < total:
        starts.append(total - chunk)
    return [(s, min(total, s + chunk)) for s in starts]


def _ramp(idx, length, count, fade_in, fade_out):
    """The crossfade window for chunk `idx` of `count`. Ones in the middle; the leading edge fades in
    unless this is the first chunk, the trailing edge fades out unless it is the last. Paired with the
    running weight sum in `run()` this makes the overlap a true average, so a join can neither dip nor
    double however the windows land."""
    ramp = torch.ones(int(length))
    if fade_in is None:
        return ramp
    n = min(len(fade_in), int(length) // 2)
    if not n:
        return ramp
    if idx > 0:
        ramp[:n] = fade_in[:n]
    if idx < count - 1:
        ramp[-n:] = fade_out[-n:]
    return ramp


def _seed32(seed):
    """(seed numpy will accept, note). Folded rather than clamped, so different inputs stay different
    seeds instead of all collapsing onto the maximum."""
    raw = int(seed)
    out = raw % (SEED_MAX + 1)
    if out == raw:
        return out, ""
    return out, (f"seed {raw} is above numpy's limit of {SEED_MAX} and was folded to {out} — "
                 f"AudioSR's seed_everything() calls np.random.seed(). Same input, same result; "
                 f"just not the number on the widget.")


def _to_work(wave, mode):
    """Stereo in → (the one channel the model gets, the side channel to keep, note).

    AudioSR is mono, so a stereo mix has to be reduced somehow. Measured on a real 3-minute take: the
    source had an L/R correlation of +0.45 and a side/mid RMS of **0.61** — a wide mix — and summing
    it to mono threw all of that away for good. Mid/side keeps it: only `mid` is rewritten, `side` is
    carried through untouched, and the two are recombined afterwards.

    The consequence, stated plainly: everything the model invents above the source's roll-off lands
    in the middle, because `side` has no content up there and none is invented for it. Highs come out
    centred. That is how plenty of real records sit, and it is a great deal better than the whole
    image collapsing."""
    if wave.shape[1] == 1:
        return wave, None, ""
    left, right = wave[:, 0:1], wave[:, 1:2]
    if mode == ST_MONO:
        return wave.mean(dim=1, keepdim=True), None, (
            f"{wave.shape[1]} channels were summed to mono — the stereo image is gone. "
            f"'{ST_MS}' keeps it.")
    note = ""
    if wave.shape[1] > 2:
        note = f"{wave.shape[1]} channels — only the first two are used for mid/side"
    return (left + right) / 2, (left - right) / 2, note


def _from_work(processed, side):
    """Put the image back: L = mid + side, R = mid - side. Exactly inverts `_to_work`."""
    if side is None:
        return processed
    return torch.cat([processed + side, processed - side], dim=1)


def _low_band_gain(out, ref, sr, upto=10000.0):
    """The gain that puts `out`'s energy BELOW `upto` back where `ref`'s was.

    Not overall RMS: the model genuinely adds energy up top, and matching totals would turn the whole
    track down to pay for it. Below the roll-off the model measured transparent (-0.4 dB at 8-12 kHz),
    so any drift down there is drift, not content — on the take we measured it was -1.2 dB at 0-4 kHz
    and -1.7 dB at 4-8 kHz, which is audible as the mix losing body."""
    def energy(x):
        spec = torch.fft.rfft(x.reshape(-1).to(torch.float32))
        bins = torch.fft.rfftfreq(x.reshape(-1).shape[0], 1.0 / sr)
        return float((spec[bins < upto].abs() ** 2).sum())
    a, b = energy(ref), energy(out)
    if a <= 0 or b <= 0:
        return 1.0
    return float((a / b) ** 0.5)


def _fades(overlap, device=None):
    """The two halves of a Hann window — one to fade a chunk in, one to fade the last one out.

    `periodic=True` on purpose: only for the periodic window does `w[i] + w[i + n]` come to exactly 1,
    which is what makes the two halves a true crossfade. Upstream builds a symmetric one, whose halves
    sum to about 0.988 at the centre of the join — a 1.2% dip, hidden here by the weight
    normalisation but no reason to introduce. `overlap = 0` gives empty ramps and a hard butt-join,
    which is what upstream defaults to."""
    if overlap <= 0:
        return None, None
    w = torch.hann_window(2 * int(overlap), periodic=True, device=device)
    return w[:int(overlap)], w[int(overlap):]


# --------------------------------------------------------------------------------- model plumbing
def _model_files():
    try:
        import folder_paths
        try:
            folder_paths.add_model_folder_path(
                FOLDER.lower(), str(__import__("pathlib").Path(folder_paths.models_dir) / FOLDER))
        except Exception:
            pass
        found = [f for f in folder_paths.get_filename_list(FOLDER.lower())
                 if f.lower().endswith((".safetensors", ".ckpt", ".pth", ".bin", ".sft"))]
        return found or ["(nothing in ComfyUI/models/AudioSR)"]
    except Exception:
        return ["(folder_paths unavailable)"]


def _load(path, model_name, device, dtype):
    """Build the LatentDiffusion and put the local checkpoint in it.

    Deliberately NOT `audiosr.pipeline.build_model`: that function ignores the `ckpt_path` it is
    given and calls `download_checkpoint()` instead, so it would fetch a copy of a file already
    sitting in `ComfyUI/models/AudioSR`."""
    _vendor_on_path()
    from audiosr.latent_diffusion.models.ddpm import LatentDiffusion
    from audiosr.utils import default_audioldm_config

    config = default_audioldm_config(model_name)
    config["model"]["params"]["device"] = device
    model = LatentDiffusion(**config["model"]["params"])

    if str(path).lower().endswith((".safetensors", ".sft")):
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
        state = blob.get("state_dict", blob)

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    if dtype is not torch.float32:
        model = model.to(dtype)
    model = model.to(device)
    return model, len(missing), len(unexpected)


def _cached(path, model_name, device, dtype, keep):
    key = (str(path), model_name, str(device), str(dtype))
    if _cache["key"] == key and _cache["model"] is not None:
        return _cache["model"], True
    _release()
    model, missing, unexpected = _load(path, model_name, device, dtype)
    if keep:
        _cache.update(key=key, model=model)
    return (model, missing, unexpected), False


def _release():
    if _cache["model"] is not None:
        _cache.update(key=None, model=None)
        gc.collect()
        try:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------------------- the node
class KinburgAudioSR:
    """Bandwidth extension for a finished mix — 48 kHz out, chunked, with real progress."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The mix to extend. Resampled to 48 kHz and summed to mono first — AudioSR is a mono model, so a stereo image does not survive this. Upscale before you widen, not after."}),
                "checkpoint": (_model_files(), {"tooltip": "A checkpoint from ComfyUI/models/AudioSR. 'basic' is for music, 'speech' for voice — the variant is read from the file name, so keep the shipped names.\n\nThe fp32 files are about 6 GB each; see 'dtype' if that is tight."}),
                "steps": ("INT", {"default": 50, "min": 10, "max": 500, "tooltip": "DDIM steps per chunk. This is the whole cost: total work is chunks x steps, and the progress bar counts exactly that.\n\n50 is a sane working value; upstream's own default is 200, which is four times the wait for a difference you will struggle to hear on a mix."}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "How hard the model is held to the input. Higher stays closer to what you fed it; lower invents more top end. 3.5 is upstream's default."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX, "control_after_generate": True, "tooltip": "Noise seed. Worth holding fixed while you compare steps or chunk lengths, or you are comparing two different inventions of the high end.\n\nCapped at 2^32-1, not the usual 2^64-1: AudioSR's seed_everything() calls numpy, which refuses anything larger. A bigger value arriving on the wire — a song seed from Siren Cast, say, which really is 64-bit — is folded into range rather than raising, and the report says what was actually used."}),
                "chunk_seconds": ("FLOAT", {"default": 15.36, "min": 5.12, "max": 30.72, "step": 0.01, "tooltip": "How much audio goes through the model at once. The default is 3 x 5.12 s because the batch builder pads every chunk up to a multiple of 5.12 s — a length that is not a multiple of it pays for denoising silence.\n\nLonger chunks mean fewer joins and more VRAM."}),
                "overlap_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "How much neighbouring chunks share, crossfaded with a Hann pair (which sums to 1, so the join neither dips nor doubles).\n\n0 is a hard butt-join, which is upstream's default and is audible on sustained material. 1 s is cheap insurance; more only costs compute, since overlapped audio is processed twice."}),
                "stereo": (STEREO, {"tooltip": "What to do with a stereo mix, since AudioSR is a mono model.\n\n• mid/side (recommended) — only the mid channel goes through the model; side is carried through untouched and the two are recombined. The image survives. Everything invented above the source's roll-off lands in the centre, because side has no content up there — highs come out centred, which is how plenty of records sit.\n\n• sum to mono — what the model wants, and what the wrapper this replaces did. Measured on a real take: an L/R correlation of +0.45 and a side/mid RMS of 0.61 became 1.00 and 0.00. The whole image, gone. Here to A/B against.\n\nMono in is untouched either way. Never run L and R separately: two independent diffusion passes decorrelate, and the invented top comes out phasey instead of wide."}),
                "match_level": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Put the output's energy BELOW 10 kHz back where the input's was.\n\nNot an overall level match — the model genuinely adds energy up top, and matching totals would turn the whole track down to pay for it. Below the roll-off it measured transparent (-0.4 dB at 8-12 kHz), so drift down there is drift: on the take we measured it was -1.2 dB at 0-4 kHz and -1.7 dB at 4-8 kHz, which reads as the mix losing body. The gain applied is printed."}),
                "dtype": (DTYPES, {"advanced": True, "tooltip": "Compute precision. fp32 is what the checkpoints ship as and what to keep unless VRAM says otherwise; fp16 roughly halves the model's footprint, bf16 is the safer half-precision on RTX 30-series and up."}),
                "keep_loaded": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Hold the model in VRAM between runs. On is right while you iterate; off frees about 6 GB after every run and pays the load time again next time."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Print the report to the console. The same text is always on the 'report' output."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/audio"
    DESCRIPTION = ("AudioSR bandwidth extension: takes a mix that dies early and writes plausible "
                   "high end back in, at 48 kHz. A stereo mix keeps its image: only the MID "
                   "channel goes through the model and side is carried through, because summing to "
                   "mono was measured to take an L/R correlation of +0.45 and a side/mid of 0.61 "
                   "down to 1.00 and 0.00. Chunked with a Hann crossfade, each chunk rescaled to "
                   "its own input peak. The progress bar counts chunks x DDIM steps and "
                   "a cancel is honoured every step rather than every chunk — both of which come "
                   "from a hook in the vendored sampler, since the diffusion loop is where the time "
                   "actually goes.")

    def run(self, audio, checkpoint, steps, guidance_scale, seed, chunk_seconds, overlap_seconds,
            stereo=ST_MS, match_level=True, dtype="fp32", keep_loaded=True, verbose=True):
        # speechbrain, imported by some other pack, makes inspect.getmodule() raise process-wide on
        # Windows — and this node's stack calls into inspect. See util/imports.py.
        defuse_lazy_modules()
        import comfy.model_management as mm
        import comfy.utils
        import folder_paths
        import numpy as np

        _vendor_on_path()
        from audiosr.latent_diffusion.models import ddim as ddim_mod
        from audiosr.pipeline import make_batch_for_super_resolution, seed_everything

        notes, lines = [], []
        wave = audio.get("waveform") if isinstance(audio, dict) else None
        rate = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
        if wave is None or not rate:
            raise RuntimeError("[Audio SR] no audio on the input — expected ComfyUI's AUDIO dict "
                               "with 'waveform' and 'sample_rate'.")
        if checkpoint.startswith("("):
            raise RuntimeError(f"[Audio SR] {checkpoint} — put an AudioSR checkpoint in "
                               f"ComfyUI/models/{FOLDER} (the fp32 safetensors is about 6 GB).")
        path = folder_paths.get_full_path(FOLDER.lower(), checkpoint) or checkpoint

        # 48 kHz and one channel is what the model takes. Resample first so mid/side is computed on
        # the audio the model will actually see.
        wave = wave.detach().to(torch.float32).cpu()
        if wave.ndim == 2:
            wave = wave.unsqueeze(0)
        if rate != SR:
            import torchaudio
            wave = torchaudio.transforms.Resample(rate, SR)(wave)
            lines.append(f"  resampled {rate} Hz → {SR} Hz")
        work, side, note = _to_work(wave, stereo)
        if note:
            notes.append(note)
        if wave.shape[1] > 1:
            m, s_ = wave.mean(1), (wave[:, 0] - wave[:, 1]) / 2
            lines.append(f"  input: {wave.shape[1]} channels · L/R correlation "
                         f"{float(torch.corrcoef(torch.stack([wave[0, 0], wave[0, 1]]))[0, 1]):+.3f} "
                         f"· side/mid {float(s_.pow(2).mean().sqrt() / m.pow(2).mean().sqrt().clamp(min=1e-9)):.3f}"
                         f" → {stereo}")
        total = int(work.shape[-1])
        seconds_in = total / SR

        chunk_n = int(round(float(chunk_seconds) * SR))
        over_n = int(round(float(overlap_seconds) * SR))
        if abs(float(chunk_seconds) / CHUNK_QUANTUM - round(float(chunk_seconds) / CHUNK_QUANTUM)) > 1e-6:
            padded = (round(float(chunk_seconds) / CHUNK_QUANTUM + 0.5)) * CHUNK_QUANTUM
            notes.append(f"chunk_seconds {float(chunk_seconds):.2f} is not a multiple of "
                         f"{CHUNK_QUANTUM} s, so every chunk is padded up to {padded:.2f} s before "
                         f"it is denoised — you pay for that silence on every chunk")
        plan = _chunk_plan(total, chunk_n, over_n)
        fade_in, fade_out = _fades(min(over_n, chunk_n // 2))

        device = mm.get_torch_device()
        want = _DTYPE.get(dtype, torch.float32)
        mm.unload_all_models()               # ~6 GB is about to land; give it the room
        t0 = time.perf_counter()
        loaded, from_cache = _cached(path, _variant(checkpoint), device, want, keep_loaded)
        if from_cache:
            model = loaded
            lines.append("  model already resident")
        else:
            model, missing, unexpected = loaded
            lines.append(f"  loaded {checkpoint} in {_format_elapsed(time.perf_counter() - t0, 'auto')}"
                         + (f" · {missing} missing / {unexpected} unexpected key(s)"
                            if missing or unexpected else ""))
        seed, seed_note = _seed32(seed)
        if seed_note:
            notes.append(seed_note)
        seed_everything(seed)

        out = torch.zeros_like(work)
        weight = torch.zeros_like(work)
        pbar = comfy.utils.ProgressBar(max(1, len(plan) * int(steps)))
        done = [0]

        def _step(i, total_steps):
            """One hook, two jobs: move the shared bar, and let a cancel land inside a chunk."""
            mm.throw_exception_if_processing_interrupted()
            pbar.update_absolute(min(done[0] + i + 1, len(plan) * int(steps)))

        t_sample = time.perf_counter()
        ddim_mod.STEP_HOOK = _step
        try:
            for idx, (start, end) in enumerate(plan):
                piece = work[:, :, start:end]
                peak = float(piece.abs().max()) + 1e-8
                batch, dur = make_batch_for_super_resolution(
                    None, waveform=piece.squeeze(0).numpy())
                with torch.no_grad():
                    got = model.generate_batch(
                        batch, unconditional_guidance_scale=float(guidance_scale),
                        ddim_steps=int(steps), duration=dur)
                if isinstance(got, np.ndarray):
                    got = torch.from_numpy(got)
                got = got.detach().to(torch.float32).cpu().reshape(1, 1, -1)[:, :, :end - start]
                # The model does not preserve level, so each chunk goes back to the peak it came in
                # with — otherwise the crossfades step in loudness from chunk to chunk.
                got = got / (float(got.abs().max()) + 1e-8) * peak

                ramp = _ramp(idx, end - start, len(plan), fade_in, fade_out)
                out[:, :, start:end] += got * ramp
                weight[:, :, start:end] += ramp
                done[0] += int(steps)
        finally:
            ddim_mod.STEP_HOOK = None
            if not keep_loaded:
                _release()

        secs = time.perf_counter() - t_sample
        out = torch.nan_to_num(out / weight.clamp(min=1e-6))

        # Level BEFORE the image goes back on: the model's drift is in the mid channel, and matching
        # it there keeps mid and side in the proportion the mix was written with. Matching after the
        # recombination would scale side too and leave the width where the drift put it.
        if match_level:
            gain = _low_band_gain(out, work, SR)
            out = out * gain
            lines.append(f"  level: {20 * torch.log10(torch.tensor(gain)).item():+.2f} dB to put "
                         f"the sub-10 kHz energy back where the input had it")
        out = _from_work(out, side)
        peak_out = float(out.abs().max())
        if peak_out > 1.0:
            out = out / peak_out
            notes.append(f"the recombined mix peaked at {peak_out:.3f} and was scaled down to fit — "
                         f"mid + side can exceed either on its own")
        out = out.clamp(-1.0, 1.0)
        if side is not None:
            m2, s2 = out.mean(1), (out[:, 0] - out[:, 1]) / 2
            lines.append(f"  output: 2 channels · L/R correlation "
                         f"{float(torch.corrcoef(torch.stack([out[0, 0], out[0, 1]]))[0, 1]):+.3f} "
                         f"· side/mid "
                         f"{float(s2.pow(2).mean().sqrt() / m2.pow(2).mean().sqrt().clamp(min=1e-9)):.3f}")

        head = (f"Audio SR — {checkpoint} · {stereo} · {_mmss(seconds_in)} of audio · "
                f"{len(plan)} chunk(s) of "
                f"{chunk_n / SR:.2f} s, {over_n / SR:.2f} s overlap · {int(steps)} steps × "
                f"{len(plan)} = {len(plan) * int(steps)} · cfg {float(guidance_scale)} · "
                f"seed {seed} · {dtype} · {_format_elapsed(secs, 'auto')} "
                f"({secs / max(1e-6, seconds_in):.2f} s per audio second)")
        for i, (start, end) in enumerate(plan):
            lines.append(f"  chunk {i + 1}: {_mmss(start / SR)} → {_mmss(end / SR)}")
        report = "\n".join([head] + lines + [f"  ⚠ {n}" for n in notes])
        if verbose:
            print("[Audio SR] " + report.replace("\n", "\n[Audio SR] "))
        return ({"waveform": out, "sample_rate": SR}, report)


def _variant(filename):
    """`basic` or `speech`, off the file name — the two ship with different configs and the wrong
    one builds a model the checkpoint does not fit."""
    low = str(filename).lower()
    return "speech" if "speech" in low else "basic"


def _mmss(sec):
    sec = max(0.0, float(sec))
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


NODE_CLASS_MAPPINGS = {"KinburgAudioSR": KinburgAudioSR}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgAudioSR": "Audio SR (48 kHz Upscale) 🔊"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
