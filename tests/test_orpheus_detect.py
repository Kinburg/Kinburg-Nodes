"""Orpheus' ear: onsets, tempo, the downbeat and the section seam, against signals we built.

Synthetic audio is the whole point of this suite. A real song has no ground truth — you can listen
and say "about there", which is exactly the standard this module exists to beat. A click track has a
tempo that is *known to the millisecond*, so "detected 128.3 against a true 128" is a measurement
rather than an impression, and an octave error cannot hide behind "well, it sounds plausible".

Needs torch (this is signal processing), but no model, no GPU and no audio files on disk.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "context", "orpheus", "phantas", "siren", "timer", "util")
load_module("kn.util.anytype", "util/anytype.py")
load_module("kn.context.character_card", "context/character_card.py")
load_module("kn.timer.timer_nodes", "timer/timer_nodes.py")
load_module("kn.siren.cast", "siren/cast.py")
load_module("kn.siren.scope", "siren/scope.py")
load_module("kn.phantas.timing", "phantas/timing.py")
load_module("kn.orpheus.timing", "orpheus/timing.py")
D = load_module("kn.orpheus.detect", "orpheus/detect.py")

check = Checker()
SR = 44100
G = torch.Generator().manual_seed(20260817)


def burst(n, lo=0.0, hi=1.0, gain=1.0):
    """A short noise burst, optionally band-limited by a crude one-pole pair, as a percussive hit."""
    x = torch.randn(n, generator=G)
    if hi < 1.0:                                   # cheap low-pass: running mean
        k = max(1, int(1.0 / max(hi, 1e-3)))
        x = torch.nn.functional.avg_pool1d(x.view(1, 1, -1), k, 1, k // 2).view(-1)[:n]
    if lo > 0.0:                                   # cheap high-pass: subtract the running mean
        k = max(1, int(1.0 / max(lo, 1e-3)))
        m = torch.nn.functional.avg_pool1d(x.view(1, 1, -1), k, 1, k // 2).view(-1)[:n]
        x = x - m
    env = torch.exp(-torch.arange(n, dtype=torch.float32) / (0.012 * SR))
    return (x * env * gain)[:n]


def track(times, seconds, gains=None, lo=0.0, hi=1.0, floor=0.0005):
    """A mono track with a hit at each time. `floor` is a whisper of noise, because a digital-silent
    background is not a thing any real detector meets."""
    n = int(seconds * SR)
    x = torch.randn(n, generator=G) * floor
    L = int(0.05 * SR)
    for i, t in enumerate(times):
        s = int(round(t * SR))
        if 0 <= s < n:
            seg = burst(L, lo, hi, 1.0 if gains is None else gains[i])
            x[s:s + len(seg)] += seg[:max(0, n - s)]
    return x


def as_audio(sig, sr=SR):
    return {"waveform": sig.view(1, 1, -1), "sample_rate": sr}


def beat_times(bpm, seconds, offset=0.0):
    step = 60.0 / bpm
    n = int((seconds - offset) / step)
    return [offset + i * step for i in range(n)]


# ------------------------------------------------------------------------------------ the signal
check("to_mono unwraps [B, C, T]", D.to_mono(as_audio(torch.zeros(1000)))[0].shape == (1000,))
check("to_mono returns the sample rate", D.to_mono(as_audio(torch.zeros(10), 48000))[1] == 48000)
st = {"waveform": torch.stack([torch.ones(100), -torch.ones(100)]).view(1, 2, 100), "sample_rate": SR}
check("to_mono sums a stereo pair", float(D.to_mono(st)[0].abs().max()) < 1e-6)
check("to_mono takes a bare tensor too", D.to_mono(torch.zeros(2, 500))[0].shape == (500,))

# --------------------------------------------------------------------------------- the envelope
sig = track(beat_times(120, 20.0), 20.0)
env, db, fps = D.onset_envelope(sig, SR)
check("the envelope runs at sr/hop", abs(fps - SR / 512.0) < 1e-6, fps)
check("the envelope covers the track", abs(env.numel() / fps - 20.0) < 0.2, env.numel() / fps)
check("the envelope is non-negative", bool((env >= 0).all()))
check("a silent track gives a flat envelope",
      float(D.onset_envelope(torch.zeros(SR * 3), SR)[0].max()) < 1e-6)
check("audio shorter than one window does not crash",
      D.onset_envelope(torch.zeros(100), SR)[0].numel() >= 1)

# ------------------------------------------------------------------------------------- the tempo
for bpm in (90.0, 120.0, 128.0, 140.0, 174.0):
    s = track(beat_times(bpm, 30.0), 30.0)
    e, _, f = D.onset_envelope(s, SR)
    got, conf = D.estimate_tempo(e, f)
    check(f"{bpm:g} BPM click track is detected ({got:.1f})", abs(got - bpm) <= 3.0, f"{got:.2f}")
    check(f"…with real confidence ({conf:.2f})", conf >= 0.25, f"{conf:.3f}")

# the octave error the prior exists to break: 64 BPM correlates as well as 128 does
s = track(beat_times(128.0, 40.0), 40.0)
e, _, f = D.onset_envelope(s, SR)
got, _ = D.estimate_tempo(e, f)
check(f"128 BPM does not come back as 64 ({got:.1f})", abs(got - 128.0) <= 3.0, f"{got:.2f}")
check("a slow track is not dragged up to the prior's centre",
      abs(D.estimate_tempo(*D.onset_envelope(track(beat_times(72.0, 40.0), 40.0), SR)[::2])[0] - 72.0)
      <= 3.0)

# a track with no pulse must SAY so rather than invent one
noise = torch.randn(SR * 20, generator=G) * 0.1
e, _, f = D.onset_envelope(noise, SR)
_, conf = D.estimate_tempo(e, f)
check(f"white noise yields no confident tempo ({conf:.3f})", conf < 0.18, f"{conf:.3f}")
check("an empty envelope is handled", D.estimate_tempo(torch.zeros(2), 86.0) == (0.0, 0.0))

# ---------------------------------------------------------------------------------- the downbeat
# every beat hit, every 4th accented — the fold must land on the accent, not just on any beat
bpm, bpb, off = 120.0, 4, 0.37
times = beat_times(bpm, 40.0, offset=off)
gains = [1.6 if i % bpb == 0 else 0.55 for i in range(len(times))]
s = track(times, 40.0, gains)
e, _, f = D.onset_envelope(s, SR)
bar = 60.0 / bpm * bpb
phase = D.estimate_phase(e, f, bar)
err = min(abs(phase - off), abs(phase - off + bar), abs(phase - off - bar))
check(f"the downbeat phase is found within a frame ({phase:.3f} vs {off})", err <= 2.5 / f,
      f"{err * 1000:.0f} ms")
check("phase without a period is zero", D.estimate_phase(e, f, 0) == 0.0)
check("phase of an empty envelope is zero", D.estimate_phase(torch.zeros(0), 86.0, 2.0) == 0.0)

# The fold must run at the BAR, not at the nearest whole number of frames. 128 BPM is the case that
# exposes it: a bar is 1.875 s = 161.5 frames, so an integer fold runs 5.7 ms short and, over two
# minutes, walks a third of a second away from the music — which put the downbeat at 0.26 s on a
# track whose downbeat is at 0.00.
bpm2, bpb2 = 128.0, 4
bar2 = 60.0 / bpm2 * bpb2
t2 = beat_times(bpm2, 120.0)
g2 = [1.7 if i % bpb2 == 0 else 0.5 for i in range(len(t2))]
e2, _, f2 = D.onset_envelope(track(t2, 120.0, g2), SR)
p2 = D.estimate_phase(e2, f2, bar2)
err2 = min(abs(p2 - 0.0), abs(bar2 - p2))
check(f"a 2-minute track at 128 BPM keeps its downbeat ({p2:.3f} s)", err2 <= 3.0 / f2,
      f"{err2 * 1000:.0f} ms")
check("the returned phase is inside one bar", 0.0 <= p2 < bar2 + 1e-9, p2)

# ------------------------------------------------------------------------------------- the onsets
want = [0.5, 1.25, 2.0, 3.5, 5.0, 6.75]
s = track(want, 8.0)
e, _, f = D.onset_envelope(s, SR)
got = D.pick_onsets(e, f, sensitivity=0.3)
check(f"every hit is found ({len(got)} for {len(want)})", len(got) == len(want), [round(t, 3) for t, _ in got])
if len(got) == len(want):
    worst = max(abs(g - w) for (g, _), w in zip(got, want))
    check(f"…each within 40 ms of the truth ({worst * 1000:.0f} ms)", worst <= 0.04, worst)
check("onsets come out in time order", [t for t, _ in got] == sorted(t for t, _ in got))
check("strengths are 0..1", all(0.0 <= v <= 1.0 for _, v in got))

# two attacks inside the gap are one event, and the louder one wins
s = track([1.0, 1.04, 3.0], 5.0, gains=[0.4, 1.5, 1.0])
e, _, f = D.onset_envelope(s, SR)
got = D.pick_onsets(e, f, sensitivity=0.3, min_gap=0.12)
check(f"a double attack collapses to one ({len(got)})", len(got) == 2, [round(t, 3) for t, _ in got])
# …and it lands where the SOUND STARTS, not on the louder of the two attacks. The flux is measured
# in dB, so silence→quiet-hit is a bigger jump than quiet-hit→loud-hit; the quiet grace note at 1.00
# therefore wins over the loud one at 1.04. For cutting picture that is the right answer, and it is
# pinned here because the opposite is the intuitive guess.
check("…landing where the sound begins, not on the louder attack",
      got and abs(got[0][0] - 1.0) <= 0.025, got[:1])

check("silence has no onsets", D.pick_onsets(*D.onset_envelope(torch.zeros(SR * 3), SR)[::2]) == [])
check("a two-frame envelope does not crash", D.pick_onsets(torch.zeros(2), 86.0) == [])
check("higher sensitivity never finds MORE onsets",
      len(D.pick_onsets(e, f, 0.9)) <= len(D.pick_onsets(e, f, 0.1)))

# ---------------------------------------------------------------------------------- the structure
# low rumble for 15 s, bright hits for 15 s: the seam is at 15
lo_half = track(beat_times(120, 15.0), 15.0, lo=0.0, hi=0.02)
hi_half = track(beat_times(120, 15.0), 15.0, lo=0.3, hi=1.0)
s = torch.cat([lo_half, hi_half])
e, db, f = D.onset_envelope(s, SR)
nov = D.structure_novelty(db, f, window_sec=3.0)
peak = float(torch.argmax(nov).item()) / f
check(f"the section seam is found at {peak:.2f} s (true 15.0)", abs(peak - 15.0) <= 1.0, peak)
check("novelty is 0..1", float(nov.min()) >= 0.0 and float(nov.max()) <= 1.0 + 1e-6)
check("a uniform track has no strong seam",
      float(D.structure_novelty(*D.onset_envelope(track(beat_times(120, 30.0), 30.0), SR)[1::-1][::-1][:2],
                                window_sec=3.0).max()) <= 1.0)
check("novelty on a too-short track is zeros",
      float(D.structure_novelty(torch.zeros(16, 3), 86.0).abs().max()) == 0.0)

# ------------------------------------------------------------------------------------ end to end
times = beat_times(128.0, 60.0)
gains = [1.5 if i % 4 == 0 else 0.5 for i in range(len(times))]
half = len(times) // 2
sig = torch.cat([track(times[:half], 30.0, gains[:half], lo=0.0, hi=0.05),
                 track([t - 30.0 for t in times[half:]], 30.0, gains[half:], lo=0.25)])
found = D.detect(as_audio(sig), sensitivity=0.35)
check(f"detect finds the tempo ({found['bpm']:.1f})", abs(found["bpm"] - 128.0) <= 3.0)
check("detect reports where the tempo came from", found["bpm_source"] == "detected")
check("detect measures the track length", abs(found["total"] - 60.0) < 0.1, found["total"])
check("detect returns cues", len(found["cues"]) > 20, len(found["cues"]))
check("every cue is a timing.cue", all({"t", "strength", "kind", "label"} <= set(c) for c in found["cues"]))
check("cues are sorted and inside the track",
      all(0 <= c["t"] <= found["total"] for c in found["cues"]) and
      [c["t"] for c in found["cues"]] == sorted(c["t"] for c in found["cues"]))
sect = [c for c in found["cues"] if c["kind"] == "section"]
check(f"the timbre change is marked as a section ({len(sect)} cue(s))", len(sect) >= 1,
      [round(c["t"], 1) for c in sect])
check("…near the halfway seam", any(abs(c["t"] - 30.0) <= 2.5 for c in sect),
      [round(c["t"], 1) for c in sect])
check("a section cue outranks an ordinary beat",
      min((c["strength"] for c in sect), default=0) >
      max((c["strength"] for c in found["cues"] if c["kind"] == "onset"), default=1))

given = D.detect(as_audio(sig), bpm=128.0)
check("a given bpm is used verbatim", given["bpm"] == 128.0 and given["bpm_source"] == "given")
check("…and the bar length follows from it", abs(given["bar_seconds"] - 1.875) < 1e-6)
check("…and the phase is still measured", given["offset"] >= 0.0)

# ------------------------------------------------- the phase of a LONGER grid, anchored to the bar
# Every bar accented, plus a much bigger hit every 2nd bar: the 2-bar grid must pick the right bar.
bpm3, bpb3 = 120.0, 4
bar3 = 60.0 / bpm3 * bpb3                          # 2.0 s
t3 = beat_times(bpm3, 90.0, offset=0.5)
g3 = [(2.2 if (i // bpb3) % 2 == 0 else 1.2) if i % bpb3 == 0 else 0.4 for i in range(len(t3))]
found3 = D.detect(as_audio(track(t3, 90.0, g3)), bpm=bpm3, beats_per_bar=bpb3)
check(f"the bar phase is right ({found3['offset']:.3f} vs 0.5)",
      min(abs(found3["offset"] - 0.5), abs(bar3 - abs(found3["offset"] - 0.5))) <= 3.0 / found3["fps"],
      found3["offset"])
two_bar = 2 * bar3
p3 = D.phase_for(found3, two_bar)
check(f"the 2-bar phase lands on the strong bar ({p3:.3f} vs 0.5)", abs(p3 - 0.5) <= 3.0 / found3["fps"], p3)
check("…and it is a bar line, not an arbitrary offset",
      abs(((p3 - found3["offset"]) / bar3) - round((p3 - found3["offset"]) / bar3)) < 1e-6, p3)
check("a one-bar period returns the bar phase untouched",
      D.phase_for(found3, bar3) == found3["offset"])
check("phase_for survives a missing envelope", D.phase_for({"bar_seconds": 2.0, "offset": 0.5}, 8.0) == 0.5)
check("phase_for survives no tempo at all", D.phase_for({}, 8.0) == 0.0)

check("confidence_note says 'yours' for a typed bpm", "yours" in D.confidence_note(given))
check("confidence_note names the number for a detected one", "detected" in D.confidence_note(found))
weak = D.detect(as_audio(torch.randn(SR * 20, generator=G) * 0.1))
note = D.confidence_note(weak)
check("a weak detection warns in words", "WEAK" in note or "no tempo" in note, note)
check("the note is printable on a cp1251 console", note.encode("cp1251") is not None)

check.done()
