"""Audio SR: the chunk geometry and the crossfade, which are the parts that can be silently wrong.

A gap of one sample between two windows is a click; a window that runs past the end is silence fed to
a diffusion model at full price; a crossfade whose weights do not sum to 1 is a dip at every join.
None of that needs the 6 GB checkpoint, so all of it is pinned here. What is NOT covered, and cannot
be without a GPU and a real mix: whether the model actually sounds better.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import PACK, Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "audio_sr", "timer", "util")
load_module("kn.util.anytype", "util/anytype.py")
load_module("kn.util.imports", "util/imports.py")
load_module("kn.timer.timer_nodes", "timer/timer_nodes.py")
asr = load_module("kn.audio_sr.nodes", "audio_sr/nodes.py")

check = Checker()
SR = asr.SR


def _coverage(total, chunk, overlap):
    """Rebuild what `run()` accumulates: the per-sample weight every window contributes."""
    plan = asr._chunk_plan(total, chunk, overlap)
    fade_in, fade_out = asr._fades(min(overlap, chunk // 2))
    weight = torch.zeros(total)
    for idx, (start, end) in enumerate(plan):
        weight[start:end] += asr._ramp(idx, end - start, len(plan), fade_in, fade_out)
    return plan, weight


# ------------------------------------------------------------------------------- the chunk geometry
check("audio shorter than one chunk is one chunk, not a padded one",
      asr._chunk_plan(1000, 4000, 500) == [(0, 1000)])
check("audio exactly one chunk long is still one chunk",
      asr._chunk_plan(4000, 4000, 500) == [(0, 4000)])

for total, chunk, overlap in ((48000 * 137, 48000 * 15, 48000), (100, 30, 10), (100, 30, 0),
                              (301, 100, 7), (48000 * 60, 48000 * 5, 24000), (5, 4, 1)):
    plan, weight = _coverage(total, chunk, overlap)
    label = f"{total}/{chunk}/{overlap}"
    check(f"[{label}] every sample is covered by at least one window",
          bool((weight > 0).all()), int((weight <= 0).sum()))
    check(f"[{label}] no window runs past the end — nothing is denoised as silence",
          plan[-1][1] == total and all(e <= total for _, e in plan), plan[-1])
    check(f"[{label}] windows are in order and none is empty",
          all(b > a for a, b in plan) and all(plan[i][0] < plan[i + 1][0]
                                              for i in range(len(plan) - 1)))
    check(f"[{label}] every window is a full chunk, so the cost per chunk is constant",
          all(e - s == chunk for s, e in plan) or len(plan) == 1,
          sorted({e - s for s, e in plan}))

plan = asr._chunk_plan(100, 30, 10)
starts = [s for s, _ in plan]
steps = [b - a for a, b in zip(starts, starts[1:])]
check("consecutive windows advance by chunk minus overlap, bar the pulled-back tail",
      steps[:-1] == [20] * (len(steps) - 1) and steps[-1] <= 20, steps)
check("the tail window is pulled BACK to the end rather than padded",
      plan[-1] == (70, 100), plan[-1])

# --------------------------------------------------------------------------------- the crossfade
fi, fo = asr._fades(64)
check("a Hann pair sums to exactly 1 across the overlap — the join cannot dip or double",
      torch.allclose(fi + fo, torch.ones(64), atol=1e-6), float((fi + fo).min()))
check("...which a SYMMETRIC window does not do, and upstream builds a symmetric one",
      abs(float((torch.hann_window(128, periodic=False)[:64]
                 + torch.hann_window(128, periodic=False)[64:]).min()) - 1.0) > 0.01)
check("no overlap means no ramps, i.e. a hard butt-join", asr._fades(0) == (None, None))
check("a fade is monotonic, in and out",
      bool((fi[1:] >= fi[:-1]).all()) and bool((fo[1:] <= fo[:-1]).all()))

check("the first chunk does not fade IN — a track must not start from silence",
      float(asr._ramp(0, 1000, 3, fi, fo)[0]) == 1.0)
check("...and the last does not fade OUT",
      float(asr._ramp(2, 1000, 3, fi, fo)[-1]) == 1.0)
check("a middle chunk fades at both edges",
      float(asr._ramp(1, 1000, 3, fi, fo)[0]) < 0.01
      and float(asr._ramp(1, 1000, 3, fi, fo)[-1]) < 0.01)
check("the only chunk of a one-chunk track is left flat",
      bool((asr._ramp(0, 1000, 1, fi, fo) == 1.0).all()))
check("a fade longer than half the chunk is trimmed, not wrapped around",
      len(asr._ramp(0, 10, 3, fi, fo)) == 10
      and bool((asr._ramp(1, 10, 3, fi, fo) <= 1.0).all()))

# The reconstruction divides by the accumulated weight, so a constant signal has to come back
# constant — that is the property that says the crossfade is an average and not a sum.
_, weight = _coverage(48000 * 137, 48000 * 15, 48000)
recon = torch.zeros(48000 * 137)
plan, _ = _coverage(48000 * 137, 48000 * 15, 48000)
fi2, fo2 = asr._fades(48000)
for idx, (start, end) in enumerate(plan):
    recon[start:end] += asr._ramp(idx, end - start, len(plan), fi2, fo2)  # a constant 1.0 "chunk"
check("a constant signal survives the overlap-add unchanged — no dips, no doubling",
      torch.allclose(recon / weight.clamp(min=1e-6), torch.ones(48000 * 137), atol=1e-5),
      float((recon / weight.clamp(min=1e-6)).min()))

# ------------------------------------------------------------------------------------------- seed
# `audiosr.pipeline.seed_everything` calls `np.random.seed`, which raises above 2**32-1. The widget
# used to allow 2**64-1, so ComfyUI's own randomiser produced a ValueError on its own.
import numpy as np  # noqa: E402

check("the widget cannot ask for a seed numpy will refuse",
      asr.KinburgAudioSR.INPUT_TYPES()["required"]["seed"][1]["max"] == asr.SEED_MAX
      and asr.SEED_MAX == 0xffffffff,
      hex(asr.KinburgAudioSR.INPUT_TYPES()["required"]["seed"][1]["max"]))
for raw in (0, 42, asr.SEED_MAX, asr.SEED_MAX + 1, 32949217759700, 0xffffffffffffffff):
    got, note = asr._seed32(raw)
    np.random.seed(got)                      # the call that used to raise
    check(f"seed {raw} → {got}, which numpy accepts",
          0 <= got <= asr.SEED_MAX and bool(note) == (got != raw), (got, bool(note)))
check("in-range seeds are passed through untouched and unmentioned",
      asr._seed32(12345) == (12345, ""))
check("out-of-range ones are FOLDED, not clamped — two big seeds stay two seeds",
      asr._seed32(asr.SEED_MAX + 1)[0] != asr._seed32(asr.SEED_MAX + 2)[0])
check("...and the fold is deterministic, so a wired 64-bit song seed still reproduces",
      asr._seed32(32949217759700)[0] == asr._seed32(32949217759700)[0] == 2523632084)
check("the report says which number was actually used",
      "folded" in asr._seed32(1 << 40)[1] and "np.random.seed" in asr._seed32(1 << 40)[1])

# --------------------------------------------------------------------------------- stereo handling
# Measured on a real 3-minute take: the source had an L/R correlation of +0.45 and a side/mid RMS of
# 0.61, and summing to mono took those to 1.00 and 0.00 — the whole image, for good. Mid/side keeps
# it, so the round trip has to be exact or the image comes back wrong in a way nobody would spot.
w = torch.randn(1, 2, 4096)
mid, side, _ = asr._to_work(w, asr.ST_MS)
check("mid is the sum and side the difference, halved",
      torch.allclose(mid[:, 0], (w[:, 0] + w[:, 1]) / 2, atol=1e-6)
      and torch.allclose(side[:, 0], (w[:, 0] - w[:, 1]) / 2, atol=1e-6))
check("...and putting them back is EXACT, so an untouched mid returns the input unchanged",
      torch.allclose(asr._from_work(mid, side), w, atol=1e-6),
      float((asr._from_work(mid, side) - w).abs().max()))
check("the model still only ever sees one channel", mid.shape[1] == 1)
check("a change to mid alone leaves the side energy exactly where it was",
      torch.allclose(((asr._from_work(mid * 0.5, side)[:, 0]
                       - asr._from_work(mid * 0.5, side)[:, 1]) / 2), side[:, 0], atol=1e-6))

mono, no_side, note = asr._to_work(w, asr.ST_MONO)
check("'sum to mono' keeps the old behaviour and says what it costs",
      no_side is None and mono.shape[1] == 1 and "image is gone" in note, note)
check("mono in is passed through untouched by either mode",
      all(asr._to_work(torch.randn(1, 1, 64), m)[1] is None for m in asr.STEREO))
check("more than two channels is reported rather than silently truncated",
      "only the first two" in asr._to_work(torch.randn(1, 4, 64), asr.ST_MS)[2])

# The level match aims below the roll-off, where the model measured transparent — matching total
# energy would turn the track down to pay for the top end it just added.
ref = torch.randn(1, 1, SR * 2)
check("a quieter output is brought back up",
      abs(asr._low_band_gain(ref * 0.5, ref, SR) - 2.0) < 0.01,
      asr._low_band_gain(ref * 0.5, ref, SR))
check("...and an identical one is left alone",
      abs(asr._low_band_gain(ref, ref, SR) - 1.0) < 1e-4)
hi = ref + 0.5 * torch.sin(2 * torch.pi * 18000 * torch.arange(SR * 2) / SR)
check("energy added ABOVE the crossover does not drag the gain down — that is the point",
      abs(asr._low_band_gain(hi, ref, SR) - 1.0) < 0.02, asr._low_band_gain(hi, ref, SR))
check("silence in or out cannot produce a divide-by-zero",
      asr._low_band_gain(torch.zeros(1, 1, 128), ref, SR) == 1.0
      and asr._low_band_gain(ref, torch.zeros(1, 1, 128), SR) == 1.0)

# ------------------------------------------------------------------------------------ node wiring
check("the variant is read off the file name, since the two ship different configs",
      (asr._variant("audiosr_speech_fp32.safetensors"), asr._variant("audiosr_basic_fp32.safetensors"),
       asr._variant("whatever.ckpt")) == ("speech", "basic", "basic"))
check("48 kHz is the output rate, fixed", SR == 48000)
check("the chunk quantum matches what the batch builder pads to", asr.CHUNK_QUANTUM == 5.12)
check("the default chunk length is a whole number of those quanta, so nothing is padded",
      abs(asr.KinburgAudioSR.INPUT_TYPES()["required"]["chunk_seconds"][1]["default"]
          / asr.CHUNK_QUANTUM - 3) < 1e-9)

import inspect  # noqa: E402
req = set(asr.KinburgAudioSR.INPUT_TYPES()["required"])
params = set(inspect.signature(asr.KinburgAudioSR.run).parameters)
check("every required input is a run() parameter", req <= params, sorted(req - params))
check("it returns audio and a report", asr.KinburgAudioSR.RETURN_NAMES == ("audio", "report"))
check("the vendored sampler carries the progress hook, defaulting to off",
      "STEP_HOOK" in (PACK / "audio_sr/vendor/audiosr/latent_diffusion/models/ddim.py").read_text(
          encoding="utf-8"))
check("...and the node is what fills it in, then puts it back",
      "ddim_mod.STEP_HOOK = _step" in inspect.getsource(asr.KinburgAudioSR.run)
      and "ddim_mod.STEP_HOOK = None" in inspect.getsource(asr.KinburgAudioSR.run))
check("the interrupt is checked inside the step hook, not between chunks",
      "throw_exception_if_processing_interrupted" in inspect.getsource(asr.KinburgAudioSR.run))
check("the landmine is defused before anything else runs",
      inspect.getsource(asr.KinburgAudioSR.run).index("defuse_lazy_modules()")
      < inspect.getsource(asr.KinburgAudioSR.run).index("import comfy"))
check("the vendored package keeps its attribution note",
      "MIT" in (PACK / "audio_sr/vendor/NOTICE.md").read_text(encoding="utf-8"))

# CLAP was removed: built by LatentDiffusion, never read, 0.80 GB of the checkpoint and a
# HuggingFace round-trip at import time. Pinned so a future re-vendor cannot bring it back unnoticed.
VENDOR = PACK / "audio_sr/vendor/audiosr"
check("audiosr/clap is gone", not (VENDOR / "clap").exists())
check("...and nothing imports it at module scope any more",
      not [f.name for f in VENDOR.rglob("*.py")
           for ln in f.read_text(encoding="utf-8").splitlines()
           if ln.startswith(("from audiosr.clap", "import audiosr.clap"))],
      [f.name for f in VENDOR.rglob("*.py")
       for ln in f.read_text(encoding="utf-8").splitlines()
       if ln.startswith(("from audiosr.clap", "import audiosr.clap"))])
check("...nor does the diffusion model still construct it",
      "self.clap = CLAP" not in (VENDOR / "latent_diffusion/models/ddpm.py").read_text(
          encoding="utf-8"))
check("the conditioner that used it is still present, just unbuildable without the package",
      "class CLAPAudioEmbeddingClassifierFreev2" in
      (VENDOR / "latent_diffusion/modules/encoders/modules.py").read_text(encoding="utf-8"))
check("the removal is written down where the next person will look",
      "0 missing" in (PACK / "audio_sr/vendor/NOTICE.md").read_text(encoding="utf-8"))

check.done()
