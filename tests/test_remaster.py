"""Remaster: loudness to a target, true peaks under a ceiling, and an honest stop when the target
would take more limiting than allowed.

Every signal here is synthetic and built so the right answer is known: a sine whose samples miss its
crest, a tone at a level BS.1770 pins down, quiet noise with clicks that only a limiter can lift.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "save_song")
R = load_module("kn.save_song.remaster_node", "save_song/remaster_node.py")

check = Checker()
RATE = 48000
torch.manual_seed(7)


def tone(freq, amp, seconds, phase=0.0):
    t = torch.arange(round(seconds * RATE), dtype=torch.float64) / RATE
    return (amp * torch.sin(2 * math.pi * freq * t + phase)).float()


def stereo(mono):
    return torch.stack([mono, mono])


def db(x):
    return 20 * math.log10(x)


# ------------------------------------------------------------------------------- measurement
# A quarter-rate sine started at 45 degrees never samples its crest: every sample sits at 0.707.
CREST = stereo(tone(RATE / 4, 1.0, 1.0, math.pi / 4))
check("the samples of a quarter-rate sine miss its crest", abs(float(CREST.abs().max()) - 0.7071) < 0.001,
      float(CREST.abs().max()))
check("and the true peak finds it between them", abs(float(R.true_peaks(CREST, RATE).max()) - 1.0) < 0.02,
      float(R.true_peaks(CREST, RATE).max()))
# Long enough to cross two block seams, oversampled in blocks and then in one piece.
LONG = stereo(tone(RATE / 4, 1.0, 12.0, math.pi / 4) + 0.1 * torch.randn(RATE * 12))
blocked, size = R.true_peaks(LONG, RATE), R.BLOCK
R.BLOCK = LONG.shape[-1]
whole = R.true_peaks(LONG, RATE)
R.BLOCK = size
check("oversampling in blocks reads every seam as the whole song would",
      float((blocked - whole).abs().max()) < 1e-5, float((blocked - whole).abs().max()))

TONE = stereo(tone(997, 0.1, 5.0))       # -20 dBFS on both channels
check("a -20 dBFS stereo tone measures -20 LUFS", abs(R.lufs(TONE, RATE) + 20.0) < 0.15, R.lufs(TONE, RATE))
check("silence has no loudness", R.lufs(torch.zeros(2, RATE), RATE) is None)
check("nor does a clip shorter than one gating block", R.lufs(TONE[:, :RATE // 5], RATE) is None)

# Ten seconds at -30 LUFS, ten at -20: the loudness range is the ten between them.
STEPS = stereo(torch.cat([tone(997, 0.1 / math.sqrt(10), 10.0), tone(997, 0.1, 10.0)]))
spread = R.loudness_range(STEPS, RATE)
check("a song in two levels ten LU apart has a range of about ten", 9.0 < spread < 10.5, spread)
check("a steady tone has none", R.loudness_range(TONE, RATE) < 0.1, R.loudness_range(TONE, RATE))
check("and a clip under one short-term window has no range", R.loudness_range(TONE[:, :RATE], RATE) is None)


# ---------------------------------------------------------------------------------- limiter
peak = torch.full((RATE,), -40.0, dtype=torch.float64)
peak[RATE // 2] = 0.0                      # one peak, 10 dB over a -10 dB ceiling
cut = R._cut(peak, 0.0, -10.0, RATE)
ahead = round(R.LOOKAHEAD * RATE)
check("the peak is brought down to the ceiling", float(cut[RATE // 2]) >= 10.0 - 1e-9, float(cut[RATE // 2]))
check("and the gain is already coming down before it arrives", float(cut[RATE // 2 - ahead // 2]) > 4.0,
      float(cut[RATE // 2 - ahead // 2]))
check("but not before the lookahead", float(cut[RATE // 2 - ahead - 1]) == 0.0)
later = RATE // 2 + ahead + round(0.1 * RATE)
check("afterwards it releases at a steady rate", abs(float(cut[later]) - (10.0 - R.RELEASE * 0.1)) < 0.3,
      float(cut[later]))
check("and lets go completely", float(cut[-1]) == 0.0)


# ----------------------------------------------------------------------------------- master
quiet = stereo(tone(997, 0.01, 6.0))      # -40 LUFS, nothing anywhere near the ceiling
out, facts = R.master(quiet, RATE, -14.0, -1.0, 6.0)
check("a quiet song is turned up to the target", abs(facts["after"] + 14.0) < 0.1, facts["after"])
check("by the plain difference, with no limiting", abs(facts["gain"] - 26.0) < 0.2 and facts["cut"] < 0.01,
      (facts["gain"], facts["cut"]))

loud = stereo(tone(997, 0.99, 6.0))
out, facts = R.master(loud, RATE, -14.0, -1.0, 6.0)
check("a loud one is turned down to it", abs(facts["after"] + 14.0) < 0.1, facts["after"])
check("which no peak survives above the ceiling", facts["peak"] <= -1.0 + 1e-6, facts["peak"])

# Noise with a click near full scale every second: the clicks stop a plain gain from reaching the
# target, so the target is set to need three decibels of limiting. The noise carries the loudness, as
# a song's body does, so taking the clicks down costs almost none of it.
bed = 0.05 * torch.randn(RATE * 8)
for at in range(RATE // 2, RATE * 8, RATE):
    bed[at:at + 32] += 0.95 * torch.hann_window(32)
clicks = torch.stack([bed, 0.5 * bed])
level, top = R.lufs(clicks, RATE), db(float(R.true_peaks(clicks, RATE).max()))
target = level + (-1.0 - top) + 3.0
out, facts = R.master(clicks, RATE, target, -1.0, 6.0)
check("loudness that needs limiting is reached", abs(facts["after"] - target) < 0.15, (facts["after"], target))
check("with the true peak held at the ceiling", facts["peak"] <= -1.0 + 1e-6
      and db(float(R.true_peaks(out, RATE).max())) <= -1.0 + 1e-3, facts["peak"])
check("using about the limiting it needed, within the cap", 2.5 < facts["cut"] <= 6.0, facts["cut"])
check("both channels keep one gain, so the image does not move",
      float((out[1] - 0.5 * out[0]).abs().max()) < 1e-5)
check("and the report says so", "limiter up to" in R.describe(facts, target, 6.0))

out, facts = R.master(clicks, RATE, target, -1.0, 1.0)
check("a target past the cap stops short of it", facts["after"] < target - 1.0, (facts["after"], target))
check("using no more limiting than allowed", facts["cut"] <= 1.0 + 1e-6, facts["cut"])
check("still under the ceiling", facts["peak"] <= -1.0 + 1e-6, facts["peak"])
said = R.describe(facts, target, 1.0)
check("and the report says how far short, and why", "stopped at" in said and "more than the 1 allowed" in said, said)

out, facts = R.master(torch.zeros(2, RATE * 2), RATE, -14.0, -1.0, 6.0)
check("silence passes through untouched", facts is None and float(out.abs().max()) == 0.0)


# ------------------------------------------------------------------------------------- node
types = R.KinburgRemaster.INPUT_TYPES()["required"]
check("defaults: -14 LUFS, -1 dBTP, 6 dB of limiting",
      (types["target_lufs"][1]["default"], types["true_peak"][1]["default"], types["max_limiting"][1]["default"])
      == (-14.0, -1.0, 6.0))
batch = torch.stack([stereo(tone(997, 0.01, 4.0)), stereo(tone(997, 0.5, 4.0))])
done, report = R.KinburgRemaster().run({"waveform": batch, "sample_rate": RATE}, -14.0, -1.0, 6.0)
check("a batch keeps its shape and rate", done["waveform"].shape == batch.shape and done["sample_rate"] == RATE)
check("and every clip in it lands on the target",
      all(abs(R.lufs(clip, RATE) + 14.0) < 0.1 for clip in done["waveform"]))
check("with a report per clip", report.count("clip ") == 2 and report.count("after:") == 2, report)

check.done()
