"""Where the music actually changes: onsets, tempo, the downbeat, and the seams between sections.

Everything here is signal processing, and that is a deliberate line. `Siren Score` opens with "No LLM
pass, nothing to hallucinate" and Phantas keeps the planner away from seconds for the same reason: a
beat is a measurable thing, exact to the millisecond, and a model asked to *look* at a spectrogram
image gets about two pixels per beat and guesses. So this module finds the candidate moments and
scores them, and the LLM's job — later, elsewhere — is only to choose among them and say what
happens in each shot.

The mel matrix comes from `siren/scope.py` rather than a second implementation. That is not tidiness:
its `/(n_fft/2)**2` normalisation is the difference between dB that mean something absolute and dB
that move with the window size, which was measured once already and should not have to be again.

Four measurements, in the order they depend on one another:

  * **onsets** — half-wave-rectified spectral flux summed across the mel bands. Rectified because
    only energy *arriving* is an onset; a decay is not.
  * **tempo** — autocorrelation of that envelope, weighted by a log-normal prior around 120 BPM.
    The prior is what stops the classic octave error, where a track at 128 comes back as 64 because
    the envelope correlates just as well at twice the lag.
  * **the downbeat** — the phase offset that puts the most onset energy on bar lines. Tempo without
    phase is a grid in the wrong place, which is worse than no grid.
  * **structure** — a novelty curve: how different the next few seconds are from the last few. This
    is what finds the drop, and it is the one cue strong enough to be worth cutting *away* from an
    even shot length for.

None of it is trustworthy on every input, and pretending otherwise is the real failure mode: a
ballad with no rhythm section has no beat grid to find. So every estimate comes back with a
confidence, and the node's report prints it instead of quietly handing back a wrong grid.
"""
import math

import torch

from ..siren.scope import _spectrogram_power, _to_db
from .timing import cue

#: Envelope resolution. 512 samples at 44.1 kHz is 11.6 ms a frame — comfortably finer than the
#: ~30 ms at which two attacks stop being separable by ear, and coarse enough that a 4-minute track
#: is 20k frames rather than a quarter of a million.
DEFAULT_HOP = 512
DEFAULT_N_FFT = 2048
#: Fewer bands than Scope draws with: this is measuring change over time, not showing pitch detail,
#: and 128 bands over 1025 bins keeps every band populated at the default n_fft.
DEFAULT_N_MELS = 128

#: The tempo search. Below 60 and above 200 BPM the autocorrelation is really finding half or double
#: of something else.
BPM_LO, BPM_HI = 60.0, 200.0
#: Width of the log-normal tempo prior, in octaves. 0.9 is broad enough not to drag a genuine 75 BPM
#: track up to 120, and narrow enough to break the 64-versus-128 tie.
TEMPO_PRIOR_WIDTH = 0.9
TEMPO_PRIOR_CENTRE = 120.0

EPS = 1e-9


# ------------------------------------------------------------------------------------ the signal
def to_mono(audio):
    """ComfyUI's AUDIO dict → `([T] float32, sample_rate)`.

    Summed to mono on purpose: an onset is an onset in both channels, and a stereo-wide synth pad
    that differs between them is exactly the kind of thing that should NOT read as a transient.
    """
    wave = audio["waveform"] if isinstance(audio, dict) else audio
    sr = int(audio["sample_rate"]) if isinstance(audio, dict) else 44100
    w = wave.detach().float().cpu()
    while w.dim() > 2:                      # [B, C, T] → [C, T], first item only
        w = w[0]
    if w.dim() == 2:
        w = w.mean(dim=0)
    return w.contiguous(), sr


def onset_envelope(sig, sr, n_fft=DEFAULT_N_FFT, n_mels=DEFAULT_N_MELS, hop=DEFAULT_HOP):
    """`(env [F], db [bands, F], fps)` — how much new energy arrives in each frame.

    Summing rectified dB differences rather than raw power differences is what makes a hi-hat in a
    quiet passage count at all: in linear power a snare in the chorus is a thousand times any of it,
    and the whole verse reads as flat.
    """
    if sig.numel() < int(n_fft):
        z = torch.zeros(1)
        return z, torch.zeros(1, 1), float(sr) / float(hop)
    db = _to_db(_spectrogram_power(sig, sr, "mel", n_fft, n_mels, hop))
    flux = (db[:, 1:] - db[:, :-1]).clamp(min=0.0).sum(dim=0)
    env = torch.cat([flux.new_zeros(1), flux])
    return env, db, float(sr) / float(hop)


def _robust_unit(x, fps=None, window_sec=1.0):
    """Scale to roughly 0..1, where 1 is a *typical loud event* rather than the loudest sample.

    The scale is a high quantile of the per-second maxima, and the "per-second" part is the whole
    trick. A plain quantile of every frame does not survive a sparse track: eight seconds carrying
    six hits is 690 frames of which 6 are events, so the 98th percentile is the *fourteenth loudest
    frame* — which is background. Everything then gets divided by background noise, clipped at 1.0,
    and the detector reports a hit roughly every 300 ms of silence. (Measured: 13 onsets found where
    6 existed, and 29 where there were 3.)

    Taking the loudest frame in each second first, and a high quantile of *those*, gives a number
    that means "how loud a real event is around here": every window with an event contributes one,
    silent stretches sink to the bottom of the quantile, and a single clipped transient moves one
    window out of a hundred — which is the outlier resistance the plain max lacks.
    """
    if x.numel() == 0:
        return x
    pos = x.clamp(min=0.0)
    peak = float(pos.max())
    if not math.isfinite(peak) or peak <= EPS:
        return torch.zeros_like(x)
    w = max(1, int(round(float(window_sec) * fps))) if fps else max(1, x.numel() // 32)
    if w > 1 and pos.numel() >= 2 * w:
        maxima = pos[:(pos.numel() // w) * w].view(-1, w).max(dim=1).values
        scale = float(torch.quantile(maxima, 0.9)) if maxima.numel() > 1 else peak
    else:
        scale = peak
    scale = min(peak, max(scale, peak / 20.0))     # never let silence set the scale
    return (pos / scale).clamp(0.0, 1.0)


# -------------------------------------------------------------------------------------- the tempo
def estimate_tempo(env, fps, bpm_lo=BPM_LO, bpm_hi=BPM_HI):
    """`(bpm, confidence)` from the onset envelope's autocorrelation.

    Confidence is the normalised autocorrelation at the winning lag — literally "how much the
    envelope looks like itself one beat later", 0 to 1. A four-on-the-floor mix lands high; a rubato
    piano ballad lands near zero, which is the honest answer and the one the report prints.

    Both are meaningless without the prior. The envelope of any track correlates at least as well at
    twice the beat period as at the beat, so an unweighted peak pick returns half the tempo about as
    often as the tempo.
    """
    if env.numel() < 8 or fps <= 0:
        return 0.0, 0.0
    x = env - env.mean()
    n = int(2 ** math.ceil(math.log2(max(4, 2 * x.numel()))))
    spec = torch.fft.rfft(x, n=n)
    r = torch.fft.irfft(spec * spec.conj(), n=n)[:x.numel()].real
    if r.numel() < 2 or r[0] <= EPS:
        return 0.0, 0.0

    lo_lag = max(1, int(round(fps * 60.0 / float(bpm_hi))))
    hi_lag = min(r.numel() - 1, int(round(fps * 60.0 / float(bpm_lo))))
    if hi_lag <= lo_lag:
        return 0.0, 0.0

    lags = torch.arange(lo_lag, hi_lag + 1, dtype=torch.float32)
    bpms = 60.0 * fps / lags
    prior = torch.exp(-0.5 * (torch.log2(bpms / TEMPO_PRIOR_CENTRE) / TEMPO_PRIOR_WIDTH) ** 2)
    best = int(torch.argmax(r[lo_lag:hi_lag + 1] * prior).item())
    lag = lo_lag + best
    return float(60.0 * fps / lag), float((r[lag] / r[0]).clamp(0.0, 1.0).item())


def estimate_phase(env, fps, period_sec):
    """Where the bar actually starts, in seconds from the top of the track.

    Folding the whole envelope onto one period and taking the loudest bin: the downbeat is simply
    the phase that most onset energy agrees on. Cheap, and it does not care whether the track begins
    with silence, a pickup, or a fade-in — all of which defeat "assume it starts at zero".
    """
    if env.numel() == 0 or fps <= 0 or not period_sec or period_sec <= 0:
        return 0.0
    period = float(period_sec)
    bins = max(1, int(round(period * fps)))
    if bins < 2 or env.numel() < bins:
        return 0.0

    # The phase is computed in SECONDS and only then binned. Folding by `frame_index % bins` looks
    # equivalent and is not: a bar is almost never a whole number of frames, so that fold runs at
    # `bins/fps` instead of at `period` and walks away from the music. Measured on a 2-minute track
    # at 128 BPM — bar 1.875 s against a 1.869 s integer fold — the accents smeared over a third of
    # a second and the downbeat came back at 0.26 s when the truth was 0.00.
    t = torch.arange(env.numel(), dtype=torch.float64) / float(fps)
    idx = (torch.remainder(t, period) / period * bins).long().clamp(0, bins - 1)
    fold = torch.zeros(bins).index_add_(0, idx, env.clamp(min=0.0))
    return float(int(torch.argmax(fold).item()) * period / bins)


def phase_for(found, period_sec):
    """The downbeat offset for a grid of `period_sec`, anchored to the bar grid.

    `detect` measures the phase of one **bar**, which is the well-defined musical quantity. Using
    that offset with a longer cut unit is a bug with no symptom: a bar phase of 1.86 s against a
    2-bar step of 3.75 s puts every line one bar late, so the whole video cuts on the backbeat and
    nothing in the report looks wrong.

    So a longer grid is not re-folded freely — it is only allowed to choose **which bar** starts the
    phrase. Candidates are `offset + k·bar`, and the winner is the one carrying the most onset
    energy. That guarantees the phrase grid sits on bar lines whatever the envelope says, and leaves
    the fold to decide the one thing it can actually know.
    """
    bar = float(found.get("bar_seconds") or 0.0)
    off = float(found.get("offset") or 0.0)
    period = float(period_sec or 0.0)
    if bar <= 0 or period <= 0:
        return off
    k_max = max(1, int(round(period / bar)))
    if k_max < 2:
        return off

    env, fps = found.get("envelope"), float(found.get("fps") or 0.0)
    if env is None or env.numel() == 0 or fps <= 0:
        return off
    t = torch.arange(env.numel(), dtype=torch.float64) / fps
    e = env.clamp(min=0.0).double()
    tol = 0.5 / fps + 1e-9
    best, best_score = off, None
    for k in range(k_max):
        cand = off + k * bar
        near = (torch.remainder(t - cand, period) <= tol) | \
               (torch.remainder(cand - t, period) <= tol)
        score = float(e[near].sum())
        if best_score is None or score > best_score:
            best, best_score = cand % period, score
    return best


# ------------------------------------------------------------------------------------ the onsets
def pick_onsets(env, fps, sensitivity=0.5, min_gap=0.12, median_window=1.0, normalize=True):
    """`[(t, strength)]` — local maxima that stand above their own neighbourhood.

    The threshold is a *moving* median rather than one number for the track. A fixed threshold tuned
    on the chorus finds nothing in the verse, which is how a detector comes back with a beat grid
    that covers half the song.

    `normalize=False` for a curve that is already 0..1 — the novelty curve, which this is reused on.
    Re-scaling it would reintroduce exactly the bug `_robust_unit` was written to fix, from the other
    side: a track with one real structural change has no "typical loud event" to divide by.
    """
    if env.numel() < 3 or fps <= 0:
        return []
    unit = _robust_unit(env, fps) if normalize else env.clamp(0.0, 1.0)
    w = max(3, int(round(float(median_window) * fps)) | 1)
    pad = w // 2
    padded = torch.nn.functional.pad(unit.view(1, 1, -1), (pad, pad), mode="reflect")
    local = padded.unfold(2, w, 1).median(dim=-1).values.view(-1)[:unit.numel()]
    # sensitivity 0 -> everything above the local median; 1 -> only what towers over it
    thresh = local + float(sensitivity) * (1.0 - local).clamp(min=0.0) * 0.5

    mid = unit[1:-1]
    hits = ((mid >= unit[:-2]) & (mid > unit[2:]) & (mid > thresh[1:-1])).nonzero().view(-1) + 1
    # The first frames compare against the transform's own reflect padding, so a track that starts
    # on a downbeat grows a phantom onset at t=0. That is an artefact of the window, not an event.
    hits = hits[hits >= 2]
    if hits.numel() == 0:
        return []
    peaks = [(float(i) / fps, float(v)) for i, v in zip(hits.tolist(), unit[hits].tolist())]

    # Enforce the minimum gap by flux strength, not by time order: two attacks inside the gap are one
    # event and only one of them should survive. Which one is decided by the flux, and the flux is
    # measured in dB — so for a quiet grace note into a loud hit the winner is normally the *first*,
    # because silence→quiet is a bigger jump in decibels than quiet→loud. That is the right answer
    # for cutting: the sound starts there.
    peaks.sort(key=lambda p: -p[1])
    kept = []
    for t, s in peaks:
        if all(abs(t - u) >= float(min_gap) for u, _ in kept):
            kept.append((t, s))
    return sorted(kept)


# ---------------------------------------------------------------------------------- the structure
def structure_novelty(db, fps, window_sec=4.0, bands=16):
    """`novelty [F]` — how unlike the previous few seconds the next few are, 0..1.

    The distance is the **mean absolute change per band, in dB**, which is both the sensitive choice
    and the meaningful one. A cosine distance is the textbook answer and is wrong here: two mean
    spectra are vectors of large positive numbers pointing in almost the same direction, so cosine
    sits at 0.999 through a drop and the curve is flat. "The average band moved by 9 dB" is a
    statement about the music.

    This is what finds a drop, a key change, the moment the drums enter — the events worth breaking
    an even shot length for, as against the thousand drum hits that are not.
    """
    if db.dim() != 2 or db.shape[1] < 4 or fps <= 0:
        return torch.zeros(max(1, db.shape[-1] if db.dim() == 2 else 1))
    b, frames = db.shape
    fac = max(1, b // int(bands))
    keep = (b // fac) * fac
    coarse = db[:keep].reshape(b // fac, fac, frames).mean(dim=1)

    w = max(2, int(round(float(window_sec) * fps)))
    if frames <= 2 * w:
        return torch.zeros(frames)
    csum = torch.cat([torch.zeros(coarse.shape[0], 1), coarse.cumsum(dim=1)], dim=1)
    idx = torch.arange(w, frames - w)
    before = (csum[:, idx] - csum[:, idx - w]) / w
    after = (csum[:, idx + w] - csum[:, idx]) / w

    nov = torch.zeros(frames)
    nov[w:frames - w] = (before - after).abs().mean(dim=0)

    # Scaled by its own maximum, NOT by `_robust_unit`. That helper answers "how loud is a typical
    # event", which is the right question for onsets (a song has thousands) and the wrong one here:
    # a song has one or two structural changes, so a quantile over windows lands in the background
    # and the curve gets divided by noise. A novelty curve genuinely does have one biggest moment.
    peak = float(nov.max())
    return nov / peak if math.isfinite(peak) and peak > EPS else nov


# ------------------------------------------------------------------------------------ all together
def detect(audio, sensitivity=0.5, bpm=0.0, beats_per_bar=4, n_fft=DEFAULT_N_FFT,
           n_mels=DEFAULT_N_MELS, hop=DEFAULT_HOP, novelty_window=4.0, min_gap=0.12):
    """Everything the planner needs from a waveform, plus how much to trust it.

    `bpm` given (which is the normal case for a song Siren wrote — you typed the tempo) skips the
    estimate entirely and only the phase is measured. Detection is for tracks that arrived from
    somewhere else.
    """
    sig, sr = to_mono(audio)
    total = sig.numel() / float(sr) if sr else 0.0
    env, db, fps = onset_envelope(sig, sr, n_fft, n_mels, hop)

    if float(bpm) > 0:
        found_bpm, confidence, source = float(bpm), 1.0, "given"
    else:
        found_bpm, confidence = estimate_tempo(env, fps)
        source = "detected"

    beat_sec = 60.0 / found_bpm if found_bpm > 0 else 0.0
    bar_sec = beat_sec * max(1, int(beats_per_bar)) if beat_sec else 0.0
    offset = estimate_phase(env, fps, bar_sec) if bar_sec else 0.0

    nov = structure_novelty(db, fps, novelty_window)
    onsets = pick_onsets(env, fps, sensitivity, min_gap)

    # A section change is ONE moment, so it is taken as a peak of the novelty curve rather than as
    # "every onset that happens to sit on a high plateau" — the curve is smoothed over several
    # seconds by construction, so the naive test marked eleven consecutive beats around a single
    # drop as eleven separate section changes, and the planner then had eleven equal candidates
    # where the music has one. The peak picker is reused wholesale, with a gap measured in shots:
    # two structural changes four seconds apart are one change with a fill in it.
    seams = pick_onsets(nov, fps, sensitivity=0.45, min_gap=8.0, median_window=12.0, normalize=False)
    promoted = {}
    for t, strength in seams:
        near = [o for o in onsets if abs(o[0] - t) <= 1.0]
        at = min(near, key=lambda o: abs(o[0] - t))[0] if near else t
        promoted[round(at, 4)] = min(1.0, 0.6 + 0.4 * strength)

    cues = []
    for t, s in onsets:
        key = round(t, 4)
        if key in promoted:
            cues.append(cue(t, promoted.pop(key), "section", "change of section"))
        else:
            cues.append(cue(t, 0.15 + 0.45 * s, "onset", "accent" if s > 0.7 else "beat"))
    # a seam with no transient under it is still a seam — a pad swell, a filter opening
    for t, strength in promoted.items():
        cues.append(cue(t, strength, "section", "change of section"))
    cues.sort(key=lambda c: c["t"])

    return {"total": total, "sample_rate": sr, "fps": fps, "bpm": found_bpm, "bpm_source": source,
            "confidence": confidence, "beats_per_bar": int(beats_per_bar), "bar_seconds": bar_sec,
            "offset": offset, "cues": cues, "envelope": env, "novelty": nov}


def confidence_note(found):
    """The one line of the report that decides whether to believe the rest of it."""
    c, bpm = found["confidence"], found["bpm"]
    if found["bpm_source"] == "given":
        return f"tempo {bpm:.1f} BPM (yours, not measured), downbeat at {found['offset']:.2f} s"
    if bpm <= 0:
        return ("no tempo could be measured — the track has no steady pulse the autocorrelation can "
                "find. Type the bpm, or cut on cues alone.")
    word = "strong" if c >= 0.35 else ("usable" if c >= 0.18 else "WEAK")
    tail = ("" if c >= 0.18 else
            "  — treat the bar grid as a guess: type the bpm if you know it, or expect cuts to sit "
            "off the beat.")
    return f"tempo {bpm:.1f} BPM detected, confidence {c:.2f} ({word}), downbeat at {found['offset']:.2f} s{tail}"
