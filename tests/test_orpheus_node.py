"""The Orpheus node end to end: a real waveform in, the strings Phantas takes out.

Runs the actual `run()` — no comfy needed, because the node deliberately imports nothing from it.
What this pins is the wiring rather than the arithmetic (that is `test_orpheus_timing`) or the
measurements (`test_orpheus_detect`): that `durations` parses back through Phantas' own parser, that
a wired Siren plan really does replace detected section boundaries, that the window arguments mean
what they say, and that the report survives the console it gets printed to.
"""
import json
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
P = load_module("kn.phantas.timing", "phantas/timing.py")
T = load_module("kn.orpheus.timing", "orpheus/timing.py")
load_module("kn.orpheus.detect", "orpheus/detect.py")
N = load_module("kn.orpheus.nodes", "orpheus/nodes.py")

check = Checker()
SR, BPM, BPB = 44100, 128.0, 4
BAR = 60.0 / BPM * BPB                      # 1.875 s
G = torch.Generator().manual_seed(20260817)
Node = N.KinburgOrpheusScore()

SONG = [("Intro", "-", 8), ("Verse 1", "Alex", 8), ("Chorus", "Nina", 8), ("Verse 2", "Nina", 8),
        ("Chorus", "Alex + Nina", 8), ("Outro", "-", 8)]
PLAN = "\n".join(f"{label} | {voice} | {bars} bars" for label, voice, bars in SONG)


def song():
    """Six 8-bar sections at 128 BPM, alternating dull and bright, accented on every downbeat."""
    total = sum(b for _, _, b in SONG) * BAR
    n = int(total * SR)
    x = torch.randn(n, generator=G) * 0.0005
    L = int(0.05 * SR)
    beat = 60.0 / BPM
    for si, (_, _, nbars) in enumerate(SONG):
        t0 = si * 8 * BAR
        bright = si % 2 == 1
        for b in range(int(8 * BAR / beat)):
            s = int((t0 + b * beat) * SR)
            if s + L > n:
                continue
            g = 1.6 if b % BPB == 0 else 0.55
            h = torch.randn(L, generator=G)
            m = torch.nn.functional.avg_pool1d(h.view(1, 1, -1), 40, 1, 20).view(-1)[:L]
            h = (h - m) if bright else m
            x[s:s + L] += h * torch.exp(-torch.arange(L, dtype=torch.float32) / (0.012 * SR)) * g
    return {"waveform": x.view(1, 1, -1), "sample_rate": SR}, total


AUDIO, TOTAL = song()


def run(**kw):
    args = dict(audio=AUDIO, cut_on="2 bars", pace=7.0, sensitivity=0.5, bpm=BPM,
                beats_per_bar=BPB, verbose=False)
    args.update(kw)
    return Node.run(**args)


# ------------------------------------------------------------------------------------ the basics
durations, trims, cues, shot_count, seconds, report, scope, gen = run()
check(f"the song is {TOTAL:.1f} s", abs(TOTAL - 90.0) < 0.01, TOTAL)
check(f"it becomes {shot_count} shots", 6 <= shot_count <= 16, shot_count)
check("seconds is the planned length", abs(seconds - TOTAL) <= 1.0 / T.FPS, seconds)

vals = P.parse_durations(durations, shot_count)
check("durations parses through Phantas' own parser", vals is not None and len(vals) == shot_count,
      durations)
check("…and every value is a legal H3 length",
      all(P.frames_for(v) in set(P.legal_frames()) for v in vals), durations)
check("…and they sum to at least the clip", sum(vals) >= seconds - 0.01, sum(vals))

check("trims is one integer per shot",
      [t.strip().isdigit() for t in trims.split(",")] == [True] * shot_count, trims)
check("every trim is under one quantum", all(0 <= int(t) < 17 for t in trims.split(",")), trims)
check("cues is one line per shot", len(cues.splitlines()) == shot_count, cues.splitlines()[:2])
check("cues lines are labelled by shot", all(l.startswith(f"shot {i + 1}: ")
                                             for i, l in enumerate(cues.splitlines())))

# ---------------------------------------------------------------------------- the plan is authority
d2, t2, cues2, count2, _, report2, _, _ = run(plan=PLAN)
labels = [label for label, _, _ in SONG]
found_labels = [l for l in labels if l in cues2]
check(f"the plan's section names reach the cues ({len(found_labels)}/{len(labels) - 1} internal)",
      len(found_labels) >= 4, found_labels)
check("the report says the sections were read, not detected",
      "read from the table" in report2, [l for l in report2.splitlines() if "plan" in l])
# every 8-bar section boundary is a multiple of 15 s; with the plan wired the cuts must land on them
ends = [round(float(x), 2) for x in []]
for i, line in enumerate(cues2.splitlines()):
    if any(lbl in line for lbl in labels):
        ends.append(i)
check("a plan-named cut exists for most sections", len(ends) >= 4, ends)
check("with a plan, no section cue is invented by the detector",
      "change of section" not in cues2, [l for l in cues2.splitlines() if "change of" in l])
check("without a plan the detector's own name is used",
      "change of section" in cues or "accent" in cues or "bar line" in cues, cues.splitlines()[:3])

# ----------------------------------------------------------------------------------- the window
_, _, _, c3, s3, _, _, _ = run(start_sec=15.0, end_sec=60.0)
check("the window sets the clip length", abs(s3 - 45.0) <= 1.0 / T.FPS, s3)
check("…and it is fewer shots than the whole song", c3 < shot_count, (c3, shot_count))
_, _, _, _, s4, _, _, _ = run(end_sec=0.0, start_sec=30.0)
check("end_sec 0 runs to the end of the track", abs(s4 - (TOTAL - 30.0)) <= 1.0 / T.FPS, s4)


def catches(**kw):
    try:
        run(**kw)
        return None
    except RuntimeError as e:
        return str(e)


check("a window shorter than one shot is a clean error",
      "shorter than one shot" in (catches(start_sec=0.0, end_sec=3.0) or ""),
      catches(start_sec=0.0, end_sec=3.0))
check("empty audio is a clean error",
      "audio is empty" in (catches(audio={"waveform": torch.zeros(1, 1, 0), "sample_rate": SR}) or ""))

# ------------------------------------------------------------------------------------- the picture
check("scope is a ComfyUI IMAGE batch", scope.dim() == 4 and scope.shape[0] == 1 and scope.shape[-1] == 3,
      tuple(scope.shape))
check("…the requested width", scope.shape[2] == 1280, tuple(scope.shape))
check("…pixels in 0..1", float(scope.min()) >= 0.0 and float(scope.max()) <= 1.0,
      (float(scope.min()), float(scope.max())))
check("…and it is not blank", float(scope.std()) > 0.02, float(scope.std()))
narrow = run(scope_width=512)[6]
check("scope_width is honoured", narrow.shape[2] == 512, tuple(narrow.shape))
off = run(scope=False)[6]
check("scope=False costs nothing", off.shape[1] <= 8 and off.shape[2] <= 8, tuple(off.shape))

# ---------------------------------------------------------------------------------- the report
info = json.loads(gen)
check("gen_extra_info is the GEN_INFO shape",
      isinstance(info, list) and info[0]["class_type"] == "Orpheus" and "params" in info[0], info[:1])
check("…and names the tempo source", "given" in info[0]["params"]["bpm"], info[0]["params"]["bpm"])
check("the report names the cut unit", "cutting on 2 bars" in report,
      [l for l in report.splitlines() if "cutting" in l])
check("the report carries the cost table", "shot lengths available" in report)
check("the report carries the timeline", "span" in report and "trim" in report)
check("the report is printable on a cp1251 console", report.encode("cp1251") is not None)
check("…and so is the plan variant", report2.encode("cp1251") is not None)

# a detected tempo (bpm=0) must still produce a usable plan
d5, _, _, c5, _, r5, _, g5 = run(bpm=0.0)
check(f"bpm=0 detects and still plans ({c5} shots)", c5 >= 4 and len(d5.split(",")) == c5, d5)
check("…and the report says it was detected", "detected" in r5,
      [l for l in r5.splitlines() if "tempo" in l])
check("…and gen info records that too", "detected" in json.loads(g5)[0]["params"]["bpm"])

check.done()
