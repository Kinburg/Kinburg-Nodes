"""Siren Scope — turn audio into an image you can actually compare.

The point of this node is not "draw a pretty spectrogram" — it is to make two takes **comparable**,
so the picture can go straight into **Image Compare** alongside everything else. Three decisions
follow from that, and they are the reason this doesn't just call matplotlib:

  * **The dB scale is fixed, never auto-ranged.** `db_floor` / `db_ceiling` are absolute (0 dB = full
    scale), so a quiet take renders dark instead of being silently boosted to fill the frame. Auto-
    normalising each image independently is exactly what makes two spectrograms un-comparable.
  * **The pixel grid is fixed.** `width_px` always spans the whole clip, so column *x* is the same
    moment in every render of the same-length track, and two images line up when flipped between.
  * **No hidden margins.** Every pixel is signal (plus the optional label strip), so an A/B flip
    doesn't jitter.

Rendering is plain torch: a colour LUT and a 5x7 bitmap font, no plotting library, no fonts on disk.

**`audio_b` turns it into a difference view** — `A - B` in dB, black where the two are identical and
bright where they diverge. That is the direct way to check a Siren retake did what it claims: the
frozen part of the track should come out pure black, and only the marked section should light up.
"""
import json

import torch
import torch.nn.functional as F

MODES = ["mel spectrogram", "linear spectrogram", "waveform", "mel + waveform"]
COLORMAPS = ["magma", "viridis", "gray"]
CHANNEL_MODES = ["mix to mono", "left", "right", "stack both"]
# How much the difference view averages before comparing, as (frequency-band pooling, time in
# seconds). 'musical' is the default because a raw bin-for-bin difference between two independent
# takes is dominated by phase and noise-floor randomness — none of which is audible.
DIFF_DETAIL = ["musical", "fine (raw bins)", "coarse"]
DETAIL_TILES = {"fine (raw bins)": (1, 0.0), "musical": (8, 0.12), "coarse": (24, 0.5)}
# Anything quieter than this far below the loudest point in either take is treated as no change.
DIFF_GATE_BELOW_PEAK = 55.0

# The dB window a shipped data matrix is quantised over (see `db_matrix_png`). The top sits a little
# ABOVE full scale on purpose: with the normalisation in `_spectrogram_power` a full-scale tone lands
# near -5 dB, and clipping the ceiling would flatten the loudest content into "no difference" in a
# diff. 126 dB over 16 bits is still 0.0019 dB per step.
DATA_DB_MIN, DATA_DB_MAX = -120.0, 6.0

# Piecewise-linear colour ramps. Enough anchors to read like the real thing without shipping a LUT.
RAMPS = {
    "magma": [(0.00, 0.00, 0.02), (0.28, 0.07, 0.38), (0.68, 0.21, 0.38),
              (0.95, 0.48, 0.24), (0.99, 0.85, 0.60), (1.00, 0.99, 0.85)],
    "viridis": [(0.27, 0.00, 0.33), (0.23, 0.32, 0.55), (0.13, 0.57, 0.55),
                (0.37, 0.79, 0.38), (0.99, 0.91, 0.14)],
    "gray": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
    # Difference view: identical = black, so anything that changed is the only thing that glows.
    "_diff": [(0.30, 0.85, 1.00), (0.10, 0.35, 0.60), (0.00, 0.00, 0.00),
              (0.65, 0.25, 0.10), (1.00, 0.65, 0.20)],
}

# 5x7 bitmap font — only the glyphs a time axis needs.
FONT = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "01110", "00000", "00000", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}
GLYPH_W, GLYPH_H = 5, 7


# ------------------------------------------------------------------------------------- drawing
def _colorize(x, ramp):
    """[H, W] in 0..1 → [H, W, 3] through a piecewise-linear ramp."""
    anchors = torch.tensor(RAMPS[ramp], dtype=torch.float32)
    n = anchors.shape[0] - 1
    pos = x.clamp(0.0, 1.0) * n
    lo = pos.floor().clamp(0, n - 1).long()
    frac = (pos - lo.float()).unsqueeze(-1)
    return anchors[lo] * (1.0 - frac) + anchors[lo + 1] * frac


def _draw_text(img, text, x, y, scale, color):
    """Blit `text` at (x, y) into an [H, W, 3] image. Clipped, so it can't raise near an edge."""
    h, w = img.shape[0], img.shape[1]
    col = torch.tensor(color, dtype=img.dtype)
    for ch in str(text):
        glyph = FONT.get(ch)
        if glyph is None:
            x += (GLYPH_W + 1) * scale
            continue
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit != "1":
                    continue
                y0, x0 = y + r * scale, x + c * scale
                y1, x1 = min(h, y0 + scale), min(w, x0 + scale)
                if y0 < h and x0 < w and y1 > 0 and x1 > 0:
                    img[max(0, y0):y1, max(0, x0):x1] = col
        x += (GLYPH_W + 1) * scale
    return x


def _tick_label(sec):
    """m:ss for the axis — the unit you point at a track with."""
    sec = max(0.0, float(sec))
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def _nice_step(duration, width_px):
    """A time step that leaves labels legible: aim for one every ~150 px, snapped to something a
    human reads without counting (1, 2, 5, 10, 15, 30 s, then whole minutes)."""
    want = duration * 150.0 / max(1, width_px)
    for step in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
        if step >= want:
            return float(step)
    return 600.0


# --------------------------------------------------------------------------------------- audio
def _pick_channels(wave, mode):
    """[C, T] → a list of (name, [T]) panels."""
    c = wave.shape[0]
    if c == 1:
        return [("mono", wave[0])]
    if mode == "left":
        return [("left", wave[0])]
    if mode == "right":
        return [("right", wave[min(1, c - 1)])]
    if mode == "stack both":
        return [("left", wave[0]), ("right", wave[min(1, c - 1)])]
    return [("mono mix", wave.mean(dim=0))]


def _spectrogram_power(sig, sample_rate, kind, n_fft, n_mels, hop):
    """Linear power spectrogram [bands, frames], normalised so 0 dB means full scale.

    Kept linear so that pooling can average ENERGY — averaging decibels instead would let a single
    quiet bin drag a whole tile down.

    The `/(n_fft/2)**2` is not cosmetic. An unnormalised FFT scales with the window length, so the
    raw mel power of ordinary music peaks around **+44 dB** and a full-scale sine around +54 — which
    meant 8% of every picture was pinned flat against a 0 dBFS ceiling, losing exactly the loudest
    and most informative part, and reading as "no difference" in a diff. It also made the scale move
    with `n_fft`: +54 at 2048 versus +61 at 4096, so two renders at different window sizes were not
    comparable at all. Measured after normalising: a full-scale sine sits at −6.4 dB (2048) / −5.1
    (4096) — near 0 as it should be, and near enough to each other to compare."""
    import torchaudio
    if kind == "mel":
        tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(sample_rate), n_fft=int(n_fft), hop_length=int(hop),
            n_mels=int(n_mels), power=2.0, center=True)
    else:
        tf = torchaudio.transforms.Spectrogram(
            n_fft=int(n_fft), hop_length=int(hop), power=2.0, center=True)
    return tf(sig.unsqueeze(0).float()).squeeze(0) / ((int(n_fft) / 2.0) ** 2)


def _to_db(power):
    """Absolute dB, 0 dB = full scale. No per-signal normalisation anywhere — that is what keeps two
    renders comparable.

    The clamp sits at -140 dB rather than the usual -100: normalising the power in
    `_spectrogram_power` moved everything down by about 60 dB, which would have left an ordinary
    track peaking near -16 dB with its floor only 84 dB below instead of far out of the way."""
    return 10.0 * torch.log10(power.clamp(min=1e-14))


def _pool(power, f_factor, t_factor):
    """Average energy over [f_factor x t_factor] tiles, then hold the result at the original size.

    This is the whole trick behind a readable difference view. Two independent diffusion runs never
    agree bin-for-bin — their fine detail and noise floor are uncorrelated — so a raw bin difference
    is a field of speckle that says nothing about whether the takes SOUND different. Averaging energy
    over a musically-sized tile (a fraction of a beat, a fraction of an octave) throws that noise away
    and leaves the differences a listener would actually notice."""
    f, t = int(f_factor), int(t_factor)
    if f <= 1 and t <= 1:
        return power
    x = power.unsqueeze(0).unsqueeze(0)
    fh = max(1, min(f, power.shape[0]))
    tw = max(1, min(t, power.shape[1]))
    pooled = F.avg_pool2d(x, kernel_size=(fh, tw), stride=(fh, tw), ceil_mode=True)
    return F.interpolate(pooled, size=power.shape, mode="nearest").squeeze(0).squeeze(0)


def _rms_env_db(sig, columns, window):
    """Short-term RMS in dB, one value per output column — the loudness contour, not the waveform."""
    t = sig.shape[0]
    # The window is a real duration, NOT the width of an output column: clamping it to the column
    # width (as an earlier version did) silently disabled the smoothing on short clips, where a
    # column is only a few milliseconds wide.
    per = max(1, min(int(window), max(1, t // 8)))
    usable = (t // per) * per
    frames = sig[:usable].view(-1, per).pow(2).mean(dim=1)
    env = F.interpolate(frames.view(1, 1, -1), size=int(columns), mode="linear",
                        align_corners=False).view(-1)
    return 10.0 * torch.log10(env.clamp(min=1e-12))


def _resize(panel, h, w):
    """[H, W] → [h, w]. Antialiased when available: without it, downscaling 11k spectrogram frames
    to 1280 px drops transients between columns instead of averaging them in."""
    x = panel.unsqueeze(0).unsqueeze(0)
    try:
        out = F.interpolate(x, size=(int(h), int(w)), mode="bilinear", align_corners=False,
                            antialias=True)
    except (TypeError, RuntimeError):
        out = F.interpolate(x, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    return out.squeeze(0).squeeze(0)


def db_matrix_png(sig, sample_rate, columns, n_fft, n_mels, duration=None):
    """The mel spectrogram of `sig` as a **data** image, not a picture: [n_mels, columns] of dB
    packed 16-bit into the R (high byte) and G (low byte) channels of a PNG.

    This exists so a viewer can do the *drawing*. Measured on a 3-minute track, the FFT is 0.02 s of
    a 0.23 s render — the rest is colouring and encoding, which is exactly the part that has to be
    redone every time somebody moves a colour or dB control. Shipping the matrix once (about 580 kB,
    less than two finished colour PNGs) lets a page recolour, re-window and even diff any pair of
    tracks by plain array arithmetic, and keeps the spectrogram itself computed in ONE place — here —
    rather than reimplemented in JavaScript where the two would be free to drift apart.

    `duration` sets the time base: pass the longest track's length and every matrix in a comparison
    shares one time axis, so column x is the same instant on all of them and a diff lines up."""
    import numpy as np
    from PIL import Image
    if duration and duration > 0:
        want = int(round(float(duration) * sample_rate))
        if want > sig.shape[0]:
            sig = torch.cat([sig, torch.zeros(want - sig.shape[0], dtype=sig.dtype)])
        elif want < sig.shape[0]:
            sig = sig[:want]
    hop = int(min(max(sig.shape[0] // max(1, int(columns) * 2), 128), int(n_fft)))
    db = _to_db(_spectrogram_power(sig, sample_rate, "mel", n_fft, n_mels, hop))
    db = _resize(db, int(n_mels), int(columns))
    q = ((db.clamp(DATA_DB_MIN, DATA_DB_MAX) - DATA_DB_MIN) / (DATA_DB_MAX - DATA_DB_MIN)
         * 65535.0).round().clamp(0, 65535).to(torch.int32).numpy()
    rgb = np.stack([(q >> 8).astype("uint8"), (q & 255).astype("uint8"),
                    np.zeros_like(q, dtype="uint8")], axis=-1)
    return Image.fromarray(rgb), {"bands": int(n_mels), "cols": int(columns),
                                  "db_min": DATA_DB_MIN, "db_max": DATA_DB_MAX}


def _envelope(sig, width):
    """Per-column min/max of the waveform — the shape a DAW draws, not a decimated sample line
    (which would alias badly and hide clipping)."""
    t = sig.shape[0]
    if t < width:
        sig = F.interpolate(sig.view(1, 1, -1), size=int(width), mode="linear",
                            align_corners=False).view(-1)
        return sig, sig
    per = t // int(width)
    trimmed = sig[:per * int(width)].view(int(width), per)
    return trimmed.min(dim=1).values, trimmed.max(dim=1).values


def _waveform_panel(sig, h, w, color):
    """A filled min/max envelope on a dark bed, with a centre line."""
    img = torch.zeros(int(h), int(w), 3, dtype=torch.float32)
    img[:, :] = torch.tensor([0.03, 0.03, 0.05])
    lo, hi = _envelope(sig, w)
    mid = (h - 1) / 2.0
    top = (mid - hi.clamp(-1, 1) * mid).round().long().clamp(0, int(h) - 1)
    bot = (mid - lo.clamp(-1, 1) * mid).round().long().clamp(0, int(h) - 1)
    img[int(mid), :] = torch.tensor([0.18, 0.18, 0.22])
    col = torch.tensor(color, dtype=torch.float32)
    rows = torch.arange(int(h)).unsqueeze(1)
    fill = (rows >= top.unsqueeze(0)) & (rows <= bot.unsqueeze(0))
    img[fill] = col
    return img


def _loudness_diff_panel(a, b, h, w, span_db, sample_rate):
    """Loudness difference over time: up = A louder, down = B louder, flat line = same level.

    Emphatically NOT a sample-by-sample subtraction. Two independent takes of the same piece disagree
    in phase everywhere, so `a - b` comes out nearly as loud as the takes themselves and paints the
    whole panel — it measures phase, not anything you can hear. Comparing short-term RMS instead
    answers the question a listener actually has: where is one of these louder than the other."""
    img = torch.zeros(int(h), int(w), 3, dtype=torch.float32)
    img[:, :] = torch.tensor([0.03, 0.03, 0.05])
    # 50 ms is roughly how long the ear integrates loudness over. Shorter windows start reporting
    # millisecond timing shifts — real, but not something anyone hears as "one of these is louder".
    win = max(64, int(sample_rate * 0.05))
    da = _rms_env_db(a, w, win) - _rms_env_db(b, w, win)
    mid = int((h - 1) / 2)
    img[mid, :] = torch.tensor([0.22, 0.22, 0.26])
    frac = (da / max(1e-6, float(span_db))).clamp(-1.0, 1.0)
    ends = (mid - frac * mid).round().long().clamp(0, int(h) - 1)
    rows = torch.arange(int(h)).unsqueeze(1)
    lo = torch.minimum(ends, torch.tensor(mid)).unsqueeze(0)
    hi = torch.maximum(ends, torch.tensor(mid)).unsqueeze(0)
    fill = (rows >= lo) & (rows <= hi)
    warm = torch.tensor([1.00, 0.65, 0.20])           # A louder
    cool = torch.tensor([0.30, 0.85, 1.00])           # B louder
    colors = torch.where((frac > 0).unsqueeze(-1), warm, cool)     # [w, 3]
    img[fill] = colors.unsqueeze(0).expand(int(h), -1, -1)[fill]
    return img


def _metrics(wave, sample_rate):
    """Plain numbers about one take. Facts only — nothing here judges anything.

    These go out as GEN_INFO so Generation Info merges them with the sampler settings and Image
    Compare's 'differences' mode tables exactly which of them moved between two runs. For anyone who
    doesn't read spectrograms, that table is the answer to 'how do these two differ', and a picture
    never will be."""
    x = wave[0].mean(dim=0) if wave.shape[1] > 1 else wave[0, 0]
    n = x.shape[0]
    peak = float(x.abs().max())
    rms = float(x.pow(2).mean().sqrt())
    to_db = lambda v: 20.0 * float(torch.log10(torch.tensor(max(float(v), 1e-10))))
    spec = torch.stft(x, n_fft=2048, hop_length=1024, window=torch.hann_window(2048),
                      return_complex=True).abs().pow(2)
    freqs = torch.linspace(0, sample_rate / 2, spec.shape[0])
    total = spec.sum().clamp(min=1e-12)
    centroid = float((spec.sum(dim=1) * freqs).sum() / spec.sum(dim=1).sum().clamp(min=1e-12))
    band = lambda lo, hi: float(spec[(freqs >= lo) & (freqs < hi)].sum() / total) * 100.0
    frame = max(1, int(sample_rate * 0.05))
    fr = x[:(n // frame) * frame].view(-1, frame).pow(2).mean(dim=1).sqrt()
    quiet = float((fr < max(peak, 1e-9) * 10 ** (-60 / 20)).float().mean()) * 100.0
    out = {
        "duration": f"{n / sample_rate:.2f} s",
        "peak": f"{to_db(peak):.1f} dBFS",
        "rms": f"{to_db(rms):.1f} dBFS",
        "crest": f"{to_db(peak) - to_db(rms):.1f} dB",
        "brightness (centroid)": f"{centroid:.0f} Hz",
        "low <250Hz": f"{band(0, 250):.1f}%",
        "mid 250Hz-4kHz": f"{band(250, 4000):.1f}%",
        "high >4kHz": f"{band(4000, sample_rate / 2 + 1):.1f}%",
        "near-silence": f"{quiet:.1f}%",
        "clipped samples": int((wave.abs() >= 0.999).sum()),
    }
    if wave.shape[1] > 1:
        l, r = wave[0, 0], wave[0, 1]
        lc, rc = l - l.mean(), r - r.mean()
        corr = float((lc * rc).sum() / (lc.norm() * rc.norm()).clamp(min=1e-9))
        out["stereo correlation"] = f"{corr:.3f}"
        out["side energy"] = f"{to_db(float(((l - r) / 2).pow(2).mean().sqrt())):.1f} dBFS"
    return out


def _grid(img, duration, bpm, beats_per_bar, origin, step_sec, diff=False, ruler=True):
    """Bar / beat lines and the plain time ruler, drawn straight onto the panel.

    Alpha-blended rather than max'd: `maximum` against a bright spectrogram barely shifts the pixel,
    so the line vanishes exactly where the music is loud — which is where you most need to know which
    bar you're looking at. Cool cyan reads clearly against magma/viridis; the difference view gets a
    neutral grey instead, so a grid line can't be mistaken for the blue "B was louder" side."""
    h, w = img.shape[0], img.shape[1]
    if duration <= 0:
        return
    bar_col = torch.tensor([0.80, 0.82, 0.86] if diff else [0.45, 0.85, 1.00])
    beat_col = torch.tensor([0.45, 0.47, 0.52] if diff else [0.25, 0.55, 0.70])

    def line(x, color, alpha):
        if 0 <= x < w:
            img[:, x] = img[:, x] * (1.0 - alpha) + color * alpha

    def col_at(sec):
        return int(round(float(sec) / duration * (w - 1)))

    if float(bpm) > 0:
        beat = 60.0 / float(bpm)
        bar = beat * max(1, int(beats_per_bar))
        t, i = float(origin), 0
        while t <= duration and i < 20000:
            line(col_at(t), bar_col, 0.60)
            b = t + beat
            while b < t + bar - 1e-6 and b <= duration:
                line(col_at(b), beat_col, 0.30)
                b += beat
            t += bar
            i += 1
    elif ruler:
        # The fallback ruler lines exist to be read together with the numbers underneath, so they
        # follow `time_labels`. With both off the panel is pure signal — what you want when the image
        # is going to be diffed pixel for pixel rather than looked at.
        t = 0.0
        while t <= duration:
            line(col_at(t), bar_col, 0.45)
            t += step_sec


def _label_strip(width, duration, step_sec, scale):
    """The time ruler under the panels."""
    h = (GLYPH_H + 5) * scale
    strip = torch.zeros(h, int(width), 3, dtype=torch.float32)
    strip[:, :] = torch.tensor([0.05, 0.05, 0.07])
    t = 0.0
    while t <= duration + 1e-6:
        x = int(round(t / max(1e-9, duration) * (width - 1)))
        strip[0:2 * scale, max(0, min(int(width) - 1, x))] = torch.tensor([0.6, 0.6, 0.65])
        text = _tick_label(t)
        tw = len(text) * (GLYPH_W + 1) * scale
        tx = min(int(width) - tw, max(0, x - tw // 2))
        _draw_text(strip, text, tx, 3 * scale, scale, (0.72, 0.72, 0.78))
        t += step_sec
    return strip


# ------------------------------------------------------------------------------------------ node
class KinburgSirenScope:
    """Audio → an image built for A/B comparison: absolute dB, fixed pixel grid, optional diff."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The audio to draw — straight out of VAEDecodeAudio, or a LoadAudio for something off disk."}),
                "mode": (MODES, {"default": "mel + waveform", "tooltip": "• mel spectrogram — frequency over time on a perceptual scale. The one to read structure, drop-outs and a band-limited top end off.\n• linear spectrogram — the same in linear frequency: worse for musical detail, better for spotting a hard low-pass or a resampling artifact.\n• waveform — min/max envelope, the DAW view. Best for level, silence and clipping.\n• mel + waveform — both stacked, sharing one time axis."}),
                "colormap": (COLORMAPS, {"default": "magma", "tooltip": "Ramp for the spectrogram. 'magma' has the most even perceptual spacing of the three, so equal steps in dB look like equal steps in brightness. Ignored in difference mode, which always uses its own diverging ramp."}),
                "width_px": ("INT", {"default": 1280, "min": 64, "max": 8192, "tooltip": "Image width. The whole clip always spans this width, so column x is the same moment in every render of a track of the same length — which is what lets two images be flipped between in Image Compare without anything shifting."}),
                "height_px": ("INT", {"default": 320, "min": 32, "max": 4096, "tooltip": "Height of the panel area (the time ruler is added below it). With 'mel + waveform' the spectrogram takes 70% and the waveform 30%."}),
                "db_floor": ("FLOAT", {"default": -80.0, "min": -200.0, "max": 0.0, "step": 1.0, "tooltip": "Black point, in dB relative to full scale. ABSOLUTE, never auto-fitted: a quiet take renders dark rather than being boosted to fill the frame, which is the whole reason two of these can be compared. Raise toward -60 to see only what's audibly present; drop toward -100 to inspect the noise floor."}),
                "db_ceiling": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 60.0, "step": 1.0, "tooltip": "White point in dBFS. 0 = digital full scale. Leave it there unless you're chasing something specific — moving it breaks comparability with images already rendered."}),
                "channels": (CHANNEL_MODES, {"default": "mix to mono", "tooltip": "Which channel to draw. 'stack both' draws left and right as separate panels — the way to see a stereo image collapse or one side dropping out."}),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300, "tooltip": "Draw the musical grid: bright lines on bar boundaries, dim ones on beats. Use the same tempo you gave TextEncodeAceStepAudio1.5, and the lines land exactly where Siren Section's 'snap: bar' would cut — so you can pick a retake window off the picture. 0 = plain time ruler instead."}),
                "beats_per_bar": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "Beats per bar for the grid (the 'timesignature' on the text encoder)."}),
                "grid_origin_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 0.01, "tooltip": "Where bar 1 starts, for a track that opens with a pickup or a moment of silence. Same meaning as on Siren Section — keep the two equal."}),
                "time_labels": ("BOOLEAN", {"default": True, "tooltip": "Add the m:ss ruler under the panels. Turn it off for a clean image to overlay or diff pixel-for-pixel."}),
                "diff_detail": (DIFF_DETAIL, {"default": "musical", "tooltip": "DIFFERENCE MODE ONLY ('audio_b' wired): how much to average before comparing.\n\nTwo independent takes never agree bin-for-bin — their fine detail and noise floor are uncorrelated — so comparing raw bins gives a field of speckle that says nothing about whether they SOUND different. Averaging energy over a musically-sized tile throws that away.\n\n• musical (default) — ~1/8 of the mel bands x 120 ms. What a listener notices.\n• coarse — ~1/24 of the bands x 500 ms. Only broad shifts in balance survive.\n• fine (raw bins) — no averaging. Honest but usually unreadable; useful only when the two clips share actual samples, i.e. checking a Siren retake."}),
                "diff_span_db": ("FLOAT", {"default": 6.0, "min": 0.5, "max": 60.0, "step": 0.5, "tooltip": "DIFFERENCE MODE ONLY: how many dB of difference reaches full colour. 6 dB is a large, obvious change in a band; drop to 2-3 to bring out nuance, raise it if the picture is saturated everywhere.\n\nThis is the dial that decides whether small differences are visible at all — leaving it wide makes two similar takes look identical when they aren't."}),
                "n_fft": ("INT", {"default": 2048, "min": 256, "max": 16384, "advanced": True, "tooltip": "FFT window. Larger = finer frequency detail and blurrier timing; smaller = sharper transients, coarser pitch."}),
                "n_mels": ("INT", {"default": 192, "min": 16, "max": 1024, "advanced": True, "tooltip": "Mel bands. Only the mel modes use it.\n\n192 rather than 256 because at the default n_fft of 2048 there are only 1025 frequency bins to share out: ask for 256 mel bands and two of them come out with no bins at all, i.e. two permanently dead rows in the picture. 192 is the most that stays fully populated. Raise n_fft to 4096 first if you want more bands than that."}),
                "label_scale": ("INT", {"default": 2, "min": 1, "max": 6, "advanced": True, "tooltip": "Pixel size of the ruler text. Raise it for a wide image you'll be looking at zoomed out."}),
            },
            "optional": {
                "audio_b": ("AUDIO", {"tooltip": "Wire a second take here to switch to DIFFERENCE mode: A minus B in dB, black where the two are identical and bright where they diverge (warm = A louder, cool = B louder).\n\nThis is the direct check that a Siren retake did what it says — the frozen part of the track should come out pure BLACK and only the marked section should light up. If black bleeds into the section, the fade is eating it; if the frozen part glows, something re-noised audio it shouldn't have.\n\nThe two clips are compared over the shorter of the two lengths, which is reported."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "GEN_INFO")
    RETURN_NAMES = ("image", "report", "gen_extra_info")
    FUNCTION = "render"
    CATEGORY = "Kinburg-Nodes/audio"
    DESCRIPTION = ("Render audio as an image built for A/B comparison: an absolute dB scale that is "
                   "never auto-fitted, a fixed pixel grid so two takes line up when flipped between, "
                   "and an optional bar/beat grid that lands exactly where Siren Section would cut. "
                   "Feeds Image Compare like any other image. Wire a second clip into 'audio_b' for a "
                   "difference view that averages over musical tiles first, so it shows differences "
                   "you could hear instead of phase noise. 'gen_extra_info' carries plain numbers — "
                   "loudness, crest, brightness, band balance, stereo — into a Generation Info dump, "
                   "which is usually a far clearer answer to 'how do these two differ' than a picture.")

    def render(self, audio, mode, colormap, width_px, height_px, db_floor, db_ceiling, channels,
               bpm, beats_per_bar, grid_origin_sec, time_labels, diff_detail, diff_span_db,
               n_fft, n_mels, label_scale, audio_b=None):
        if not (isinstance(audio, dict) and audio.get("waveform") is not None):
            raise RuntimeError("[Siren Scope] No valid audio — wire an AUDIO into 'audio'.")
        wave = audio["waveform"].detach().cpu().float()
        sr = int(audio.get("sample_rate", 44100))
        if wave.ndim == 2:                      # [C, T] → [1, C, T]
            wave = wave.unsqueeze(0)
        notes, lines = [], []

        wave_b = None
        if audio_b is not None and isinstance(audio_b, dict) and audio_b.get("waveform") is not None:
            wave_b = audio_b["waveform"].detach().cpu().float()
            if wave_b.ndim == 2:
                wave_b = wave_b.unsqueeze(0)
            sr_b = int(audio_b.get("sample_rate", sr))
            if sr_b != sr:
                import torchaudio
                wave_b = torchaudio.functional.resample(wave_b, sr_b, sr)
                notes.append(f"'audio_b' was {sr_b} Hz and was resampled to {sr} Hz to line up")
            n = min(wave.shape[-1], wave_b.shape[-1])
            if wave.shape[-1] != wave_b.shape[-1]:
                notes.append(f"the two clips differ in length "
                             f"({wave.shape[-1] / sr:.2f} s vs {wave_b.shape[-1] / sr:.2f} s) — "
                             f"compared over the shorter {n / sr:.2f} s")
            wave, wave_b = wave[..., :n], wave_b[..., :n]

        duration = wave.shape[-1] / float(sr)
        db_lo, db_hi = float(db_floor), float(db_ceiling)
        if db_hi <= db_lo:
            db_hi = db_lo + 1.0
            notes.append("'db_ceiling' was not above 'db_floor' — nudged up by 1 dB")
        # Cap the analysis rate at ~2 frames per output column: computing 11k spectrogram frames for
        # a 1280 px image just to average them away is wasted work, and too coarse a hop aliases.
        hop = int(min(max(wave.shape[-1] // max(1, int(width_px) * 2), 128), int(n_fft)))
        step_sec = _nice_step(duration, int(width_px))
        diff = wave_b is not None
        f_pool, t_secs = DETAIL_TILES.get(diff_detail, DETAIL_TILES["musical"])
        t_pool = max(1, int(round(t_secs * sr / hop)))
        gate_used, loudest_change = db_lo, 0.0

        images = []
        for b in range(wave.shape[0]):
            panels = []
            chans = _pick_channels(wave[b], channels)
            chans_b = _pick_channels(wave_b[b if b < wave_b.shape[0] else 0], channels) if diff \
                else [None] * len(chans)
            want_spec = mode != "waveform"
            want_wave = mode in ("waveform", "mel + waveform")
            per = max(1, int(height_px) // max(1, len(chans)))
            for (name, sig), pair in zip(chans, chans_b):
                spec_h = per if not want_wave else max(1, int(per * 0.7))
                wave_h = max(1, per - spec_h) if want_spec else per
                if want_spec:
                    kind = "linear" if mode == "linear spectrogram" else "mel"
                    pa = _spectrogram_power(sig, sr, kind, n_fft, n_mels, hop)
                    if diff:
                        pb = _spectrogram_power(pair[1], sr, kind, n_fft, n_mels, hop)
                        # Floor both takes at the same level BEFORE pooling. Gating afterwards
                        # instead leaves the near-empty bins — where two takes disagree by tens of dB
                        # about essentially nothing — to dominate the picture, and pooling actually
                        # lifts them past the gate. Flooring first makes anything below it read as
                        # identical, which it audibly is. Measured on synthetic takes: inaudible
                        # noise 50 dB down drops from painting the frame to 0% of tiles, while a real
                        # 10 dB shift in the top end still comes through at 9.4 dB.
                        gate = max(db_lo, float(torch.maximum(_to_db(pa).max(), _to_db(pb).max()))
                                   - DIFF_GATE_BELOW_PEAK)
                        gate_used = gate
                        floor_lin = 10.0 ** (gate / 10.0)
                        pa = _pool(pa.clamp(min=floor_lin), f_pool, t_pool)
                        pb = _pool(pb.clamp(min=floor_lin), f_pool, t_pool)
                        d = _to_db(pa) - _to_db(pb)
                        span = max(0.5, float(diff_span_db))
                        norm = (d.clamp(-span, span) / span + 1.0) / 2.0
                        ramp = "_diff"
                        loudest_change = max(loudest_change, float(d.abs().max()))
                    else:
                        norm = ((_to_db(pa) - db_lo) / (db_hi - db_lo)).clamp(0.0, 1.0)
                        ramp = colormap
                    panel = _colorize(_resize(norm.flip(0), spec_h, int(width_px)).clamp(0, 1), ramp)
                    _grid(panel, duration, bpm, beats_per_bar, grid_origin_sec, step_sec, diff, time_labels)
                    panels.append(panel)
                if want_wave:
                    if diff:
                        panel = _loudness_diff_panel(sig, pair[1], wave_h, int(width_px),
                                                     diff_span_db, sr)
                    else:
                        panel = _waveform_panel(sig, wave_h, int(width_px), (0.45, 0.80, 0.95))
                    _grid(panel, duration, bpm, beats_per_bar, grid_origin_sec, step_sec, diff, time_labels)
                    panels.append(panel)
            img = torch.cat(panels, dim=0) if len(panels) > 1 else panels[0]
            if time_labels:
                img = torch.cat([img, _label_strip(int(width_px), duration, step_sec,
                                                   int(label_scale))], dim=0)
            images.append(img)

        out = torch.stack(images, dim=0).clamp(0.0, 1.0)

        # Facts, not verdicts: numbers you'd otherwise have to open an editor to read.
        peak = float(wave.abs().max())
        peak_db = 20.0 * torch.log10(torch.tensor(max(peak, 1e-10)))
        rms = float(wave.pow(2).mean().sqrt())
        rms_db = 20.0 * torch.log10(torch.tensor(max(rms, 1e-10)))
        clipped = int((wave.abs() >= 0.999).sum())
        lines.append(f"Siren Scope — {'DIFFERENCE (A-B)' if diff else mode} · "
                     f"{_tick_label(duration)} ({duration:.2f} s) @ {sr} Hz · "
                     f"{wave.shape[1]} ch · {out.shape[2]}x{out.shape[1]} px")
        lines.append(f"  scale {db_lo:.0f}..{db_hi:.0f} dBFS (absolute) · n_fft {n_fft} · "
                     f"hop {hop} · ruler every {step_sec:g} s" +
                     (f" · grid {bpm} bpm {beats_per_bar}/4" if bpm > 0 else ""))
        lines.append(f"  peak {float(peak_db):.1f} dBFS · rms {float(rms_db):.1f} dBFS" +
                     (f" · {clipped} clipped samples" if clipped else ""))
        info = _metrics(wave, sr)
        if diff:
            d = (wave - wave_b).abs()
            same = float((d < 1e-4).float().mean()) * 100.0
            lines.append(f"  difference: tiles {f_pool} band(s) x {t_pool * hop / sr * 1000:.0f} ms "
                         f"('{diff_detail}') · +-{float(diff_span_db):.1f} dB reaches full colour · "
                         f"gated below {gate_used:.0f} dB")
            lines.append(f"  largest tile difference {loudest_change:.1f} dB · "
                         f"{same:.1f}% of raw samples identical")
            if same < 1.0:
                lines.append("  note: two independent takes share almost no samples even when they "
                             "sound alike — that figure is only meaningful for a Siren retake, where "
                             "the frozen part IS the same samples.")
            info_b = _metrics(wave_b, sr)
            for k in info:
                if info[k] != info_b.get(k):
                    lines.append(f"  {k}: A {info[k]}  vs  B {info_b.get(k)}")
            info = {f"A {k}": v for k, v in info.items()}
            info.update({f"B {k}": v for k, v in info_b.items()})
        if peak < 1e-4:
            notes.append("this clip is effectively SILENT (peak below -80 dBFS)")
        for w in notes:
            lines.append(f"  ⚠ {w}")
        report = "\n".join(lines)
        print("[Siren Scope] " + report.replace("\n", "\n[Siren Scope] "))
        gen_extra = json.dumps([{"class_type": "Siren Scope", "ord": 1, "params": info}],
                               ensure_ascii=False)
        return (out, report, gen_extra)


NODE_CLASS_MAPPINGS = {"KinburgSirenScope": KinburgSirenScope}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSirenScope": "Siren Scope (Audio → Image) 🧜"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
