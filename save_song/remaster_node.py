"""Remaster — a finished song brought to a loudness target, its true peaks held under a ceiling.

Neither music model masters what it writes: a YuE2 or AceStep take comes out at whatever level it
lands on, and Save Song writes exactly that. This node goes between them. It measures the song, turns
it up or down to the target, and lets a limiter catch the peaks the turning-up pushed over the
ceiling. No compressor, on purpose: the takes arrive already mixed, and squashing them further is a
choice to make by ear, not a default.

**Loudness is measured, not guessed.** Integrated loudness is BS.1770-4, the gated K-weighted mean
that streaming services normalise to, and it is torchaudio's own implementation. The loudness range
in the report is EBU Tech 3342's, from 3 s short-term loudness on the same K-weighting: how far the
quiet parts sit from the loud ones, which is the first thing a limiter eats.

**Peaks are measured between the samples.** A waveform can swing higher between two samples than at
either of them, and a codec or a DAC reproduces that swing. So the ceiling holds the TRUE peak, read
from the signal oversampled four times as BS.1770 prescribes, not the sample values.

**The limiter sees the whole song at once**, which a real-time one cannot. The gain every sample
needs is known in advance, so it comes down over the 5 ms before a peak instead of clipping its front
edge, and recovers at a steady 40 dB a second after it. Both channels share one gain, so nothing moves
in the stereo image.

**When the target is out of reach, it stops short rather than squash.** Bringing a dynamic take up to
-14 LUFS can take far more peak reduction than sounds good. `max_limiting` caps it: the song comes out
as loud as the cap allows, and the report says how far short it fell and what the target would cost.
"""
import math

import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from ..categories import CAT_AUDIO

#: Seconds the limiter looks ahead: long enough to ramp the gain down over a peak's front edge rather
#: than clip it, short enough not to blunt a drum hit.
LOOKAHEAD = 0.005
#: dB a second the gain recovers at once a peak has passed.
RELEASE = 40.0
#: EBU Tech 3342's short-term window, and the step it is read at, for the loudness range.
SHORT_TERM, SHORT_STEP = 3.0, 0.1
#: Samples oversampled at a time, so a five-minute song never needs four copies of itself in memory.
BLOCK = 1 << 18
#: BS.1770 channel weights: left, right, centre, then the two surrounds.
WEIGHTS = (1.0, 1.0, 1.0, 1.41, 1.41)


def lufs(wave, rate):
    """Integrated loudness of a [C, T] clip, or None when there is nothing to gate: silence, or less
    than the 0.4 s of one gating block."""
    if wave.shape[-1] < round(0.4 * rate):
        return None
    value = float(AF.loudness(wave, rate))
    return value if math.isfinite(value) else None


def loudness_range(wave, rate):
    """LRA in LU, or None for a clip shorter than one short-term window."""
    span, step = round(SHORT_TERM * rate), round(SHORT_STEP * rate)
    if wave.shape[-1] < span:
        return None
    # The K-weighting torchaudio's `loudness` applies, so the range and the loudness agree.
    weighted = AF.highpass_biquad(AF.treble_biquad(wave, rate, 4.0, 1500.0, 1 / math.sqrt(2)), rate, 38.0, 0.5)
    power = F.pad(weighted.double().square(), (1, 0)).cumsum(-1)
    starts = torch.arange(0, wave.shape[-1] - span + 1, step, device=wave.device)
    mean = (power[:, starts + span] - power[:, starts]) / span
    gains = torch.tensor(WEIGHTS[:wave.shape[0]], dtype=mean.dtype, device=mean.device)
    short = -0.691 + 10 * torch.log10((gains[:, None] * mean).sum(0).clamp_min(1e-20))
    short = short[short > -70.0]
    if short.numel() == 0:
        return None
    gate = -0.691 + 10 * math.log10(float(torch.pow(10.0, (short + 0.691) / 10).mean())) - 20.0
    short = short[short > gate]
    return float(torch.quantile(short, 0.95) - torch.quantile(short, 0.10))


def true_peaks(wave, rate):
    """[C, T]: the highest each channel swings between each sample and the next, from the signal
    oversampled four times (twice from 96 kHz up). Done in blocks, each padded with its neighbours'
    samples so a seam is interpolated exactly as the whole song would be."""
    factor = 4 if rate < 96000 else 2
    pad = 16                    # well past the resampling kernel's reach
    size = wave.shape[-1]
    out = []
    for s in range(0, size, BLOCK):
        e = min(s + BLOCK, size)
        a, b = max(0, s - pad), min(size, e + pad)
        up = AF.resample(wave[:, a:b], 1, factor)[:, (s - a) * factor:(e - a) * factor]
        out.append(up.abs().reshape(wave.shape[0], e - s, factor).amax(-1))
    return torch.cat(out, -1)


def _ahead_max(x, k):
    """max(x[n : n + k]) for every n of a non-negative 1-D tensor, by doubling the window: about
    log2(k) elementwise maxima. `max_pool1d` at stride 1 does the same in O(T·k) and took eight
    seconds on a five-minute song."""
    m, span = F.pad(x, (0, k - 1)), 1
    while span * 2 <= k:
        m = torch.maximum(m[:-span], m[span:])
        span *= 2
    return torch.maximum(m[:x.shape[-1]], m[k - span:k - span + x.shape[-1]])


def _cut(peak_db, gain, ceiling, rate):
    """dB the limiter takes off at each sample, so that peak + gain - cut never tops the ceiling.

    What each sample needs is known up front. It is spread back over the lookahead, so the gain is
    already down when a peak arrives; released linearly after it, where `cummax` of the need plus a
    rising ramp is exactly "the largest earlier need, less what has been released since"; and smoothed
    by a moving average no longer than the lookahead. That keeps every peak covered, because each
    sample inside the average already has the peak within its own look ahead.
    """
    ahead = max(1, round(LOOKAHEAD * rate))
    need = _ahead_max((peak_db + (gain - ceiling)).clamp_min(0), ahead + 1)
    ramp = torch.arange(need.shape[-1], dtype=need.dtype, device=need.device) * (RELEASE / rate)
    held = torch.cummax(need + ramp, -1).values - ramp
    return F.avg_pool1d(F.pad(held[None, None], (ahead, 0), mode="replicate"), ahead + 1, 1)[0, 0]


def master(wave, rate, target, ceiling, most):
    """One [C, T] clip → (the mastered clip, what was measured and done), or (the clip, None) when
    there is nothing to measure."""
    before = lufs(wave, rate)
    if before is None:
        return wave, None
    peak_db = 20 * torch.log10(true_peaks(wave, rate).amax(0).double().clamp_min(1e-10))
    top = float(peak_db.max())
    reach = ceiling + most - top            # the most gain `most` dB of limiting can carry
    gain = min(target - before, reach)
    # Limiting takes a little loudness off, so the gain is nudged up after it; a few rounds settle.
    for _ in range(4):
        cut = _cut(peak_db, gain, ceiling, rate)
        out = wave * torch.pow(10.0, (gain - cut) / 20).to(wave.dtype)
        applied, after = gain, lufs(out, rate)
        if abs(target - after) < 0.05 or (after < target and gain >= reach):
            break
        gain = min(gain + target - after, reach)
    peak = 20 * math.log10(max(float(true_peaks(out, rate).max()), 1e-10))
    if peak > ceiling:                      # the interpolation can still poke a hair over
        out = out * 10 ** ((ceiling - peak) / 20)
        peak, after = ceiling, lufs(out, rate)
    return out, {"before": before, "after": after, "top": top, "peak": peak, "gain": applied,
                 "cut": float(cut.max()), "share": float((cut > 1.0).double().mean()),
                 "need": top + target - before - ceiling,
                 "range_before": loudness_range(wave, rate), "range_after": loudness_range(out, rate)}


def describe(facts, target, most):
    """The report for one clip."""
    def line(level, peak, spread):
        return f"{level:.1f} LUFS, true peak {peak:+.1f} dBTP" + ("" if spread is None else f", range {spread:.1f} LU")
    said = [f"before: {line(facts['before'], facts['top'], facts['range_before'])}",
            f"after:  {line(facts['after'], facts['peak'], facts['range_after'])}",
            f"gain {facts['gain']:+.1f} dB, " + (f"limiter up to {facts['cut']:.1f} dB, over 1 dB on "
                                                  + (f"{facts['share']:.0%}" if facts["share"] >= 0.01 else "under 1%")
                                                  + " of the song" if facts["cut"] >= 0.1 else "no limiting")]
    if target - facts["after"] > 0.2:
        said.append(f"! {target:g} LUFS would need about {facts['need']:.1f} dB of peak limiting, more than "
                    f"the {most:g} allowed, so it stopped at {facts['after']:.1f} LUFS. Raise max_limiting "
                    f"to go further, at the cost of punch")
    return "\n".join(said)


class KinburgRemaster:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The finished song — after Audio SR, if you use it, so the true peaks are measured at the rate that gets saved."}),
                "target_lufs": ("FLOAT", {"default": -14.0, "min": -40.0, "max": -5.0, "step": 0.5, "tooltip": "Integrated loudness to land on, BS.1770.\n\n-14 is what Spotify and YouTube normalise to: a louder master is simply turned down by the platform, so it buys nothing there and costs dynamics. Go higher only for a file that is played as it is."}),
                "true_peak": ("FLOAT", {"default": -1.0, "min": -9.0, "max": 0.0, "step": 0.1, "tooltip": "Ceiling in dBTP — measured between the samples, not on them.\n\n-1 leaves room for an MP3 or Opus encoder, which overshoots the source a little; a ceiling at 0 can clip after encoding."}),
                "max_limiting": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5, "tooltip": "The most dB the limiter may take off a peak to reach the target.\n\nA dynamic take can need much more than sounds good. Over the cap the node stops short of the target, at the loudest level the cap allows, and the report says what the target would have cost. 0 = no limiting: the song is only turned up until its true peak meets the ceiling."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "run"
    CATEGORY = CAT_AUDIO
    DESCRIPTION = ("Brings a finished song to a loudness target (BS.1770, -14 LUFS by default) and holds its "
                   "true peaks under a ceiling (-1 dBTP) with a lookahead limiter that sees the whole song "
                   "at once. No compressor. If the target needs more limiting than 'max_limiting' allows, "
                   "it stops short and says so. The report gives loudness, true peak and loudness range "
                   "before and after. Put it before Save Song.")

    def run(self, audio, target_lufs, true_peak, max_limiting):
        wave, rate = audio["waveform"], int(audio["sample_rate"])
        outs, said = [], []
        for i in range(wave.shape[0]):
            out, facts = master(wave[i], rate, target_lufs, true_peak, max_limiting)
            outs.append(out)
            text = (describe(facts, target_lufs, max_limiting) if facts
                    else "silent or shorter than 0.4 s — passed through unchanged")
            said.append(text if wave.shape[0] == 1 else f"clip {i + 1}:\n{text}")
        report = "\n\n".join(said)
        print("[Remaster] " + report.replace("\n", "\n[Remaster] "))
        return ({"waveform": torch.stack(outs), "sample_rate": rate}, report)


NODE_CLASS_MAPPINGS = {"KinburgRemaster": KinburgRemaster}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgRemaster": "Remaster (Loudness) 🎚️"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
