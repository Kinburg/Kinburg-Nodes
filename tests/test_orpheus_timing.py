"""Orpheus' clock: cues onto a musical grid, the grid onto H3's, and the trim that reconciles them.

The claim this module makes is a number: **a cut lands within half a frame of the beat it was
written for**, on any tempo, for any shot length H3 allows. That is what most of this suite is
about — the rest pins the tail rule (never strand an unrenderable remainder) and the absolute-frame
bookkeeping that stops a three-minute clip from walking off its own soundtrack.

Pure arithmetic: no torch, no comfy, no audio. The detector that turns a waveform into cues is a
separate concern and is not exercised here.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "context", "orpheus", "phantas", "siren", "timer", "util")
load_module("kn.util.anytype", "util/anytype.py")
load_module("kn.context.character_card", "context/character_card.py")
load_module("kn.timer.timer_nodes", "timer/timer_nodes.py")
load_module("kn.siren.cast", "siren/cast.py")
P = load_module("kn.phantas.timing", "phantas/timing.py")
T = load_module("kn.orpheus.timing", "orpheus/timing.py")

check = Checker()
LEGAL = set(P.legal_frames())
HALF_FRAME = 0.5 / T.FPS


def bars(bpm, beats_per_bar=4):
    return T.bar_seconds(bpm, beats_per_bar)


# ------------------------------------------------------------------ why this module exists at all
# If a shot length could simply BE a whole number of bars, none of the trimming would be needed.
# At 128 BPM — an ordinary dance tempo — not one of the six bar-counts that fit H3's band is a legal
# shot length. This is the collision the whole module is built around, so it is asserted rather than
# asserted-in-a-docstring.
bar128 = bars(128)
check("128 BPM: a bar is 1.875 s", abs(bar128 - 1.875) < 1e-9, bar128)
fits = [n for n in range(1, 20) if P.MIN_SECONDS - 1e-9 <= n * bar128 <= P.MAX_SECONDS + 1e-9]
check("128 BPM: six bar-counts fit H3's band", fits == [3, 4, 5, 6, 7, 8], fits)
on_grid = [n for n in fits if int(round(n * bar128 * T.FPS)) % 17 == 5]
check("128 BPM: NONE of them is a legal H3 length", on_grid == [], on_grid)

# 120 BPM is the exception that proves it is a coincidence, not a rule: 4 bars = 8.00 s = 192 frames.
check("120 BPM: 4 bars happens to land on the grid exactly",
      int(round(4 * bars(120) * T.FPS)) == 192 and 192 % 17 == 5)
sweep = [b for b in range(70, 181) if any(
    P.MIN_SECONDS <= n * bars(b) <= P.MAX_SECONDS and int(round(n * bars(b) * T.FPS)) % 17 == 5
    for n in range(1, 20))]
check(f"most tempi have no exact bar/shot coincidence at all ({len(sweep)}/111 do)",
      len(sweep) < 40, sweep[:12])

# ------------------------------------------------------------------------------------ the grid
check("bar_seconds is Siren's, 4/4 at 120 = 2 s", abs(bars(120) - 2.0) < 1e-9)
check("bar_seconds: 3/4 is shorter", abs(T.bar_seconds(120, 3) - 1.5) < 1e-9)
check("bar_seconds: no tempo, no bar", T.bar_seconds(0, 4) is None)

check("unit: beat", abs(T.unit_seconds(2.0, 4, "beat") - 0.5) < 1e-9)
check("unit: bar", abs(T.unit_seconds(2.0, 4, "bar") - 2.0) < 1e-9)
check("unit: 2 bars", abs(T.unit_seconds(2.0, 4, "2 bars") - 4.0) < 1e-9)
check("unit: phrase", abs(T.unit_seconds(2.0, 4, "phrase (4 bars)") - 8.0) < 1e-9)
check("unit: an unknown name falls back to one bar", abs(T.unit_seconds(2.0, 4, "??") - 2.0) < 1e-9)
check("unit: no bar means no unit", T.unit_seconds(None, 4, "bar") is None)
check("every CUT_UNIT resolves", all(T.unit_seconds(2.0, 4, u) for u in T.CUT_UNITS))

g = T.grid_times(2.0, 9.0)
check("grid spans the track", g == [0.0, 2.0, 4.0, 6.0, 8.0], g)
g = T.grid_times(2.0, 9.0, offset=0.5)
check("grid respects the downbeat phase", g == [0.5, 2.5, 4.5, 6.5, 8.5], g)
check("grid never goes negative", T.grid_times(2.0, 5.0, offset=-0.4)[0] >= 0)
g = T.grid_times(2.0, 100.0, lo=3.0, hi=7.0)
check("grid honours a window", g == [4.0, 6.0], g)
check("an empty window gives nothing", T.grid_times(2.0, 100.0, lo=7.0, hi=3.0) == [])
check("no step, no grid", T.grid_times(0, 10.0) == [])

# ------------------------------------------------------- what a cut unit costs, and why it varies
# The finding that makes this worth reporting rather than assuming: at 128 BPM one 4-bar phrase
# (7.50 s) wastes ten times as much as two of them (15.00 s), purely from where each lands against
# H3's grid. Same song, same tempo, same unit — different multiple.
phrase128 = T.unit_seconds(bar128, 4, "phrase (4 bars)")
opts = T.unit_options(phrase128)
check("128 BPM phrase: two shot lengths are legal", [o["units"] for o in opts] == [1, 2], opts)
check("…one phrase is 7.50 s and wastes 12 frames",
      opts[0]["span"] == 7.5 and opts[0]["trim_frames"] == 12, opts[0])
check("…two phrases is 15.00 s and wastes 2",
      opts[1]["span"] == 15.0 and opts[1]["trim_frames"] == 2, opts[1])
check("…which is 6.2% against 0.6%",
      abs(opts[0]["waste"] - 0.0625) < 0.002 and abs(opts[1]["waste"] - 0.0055) < 0.002,
      [round(o["waste"], 4) for o in opts])
check("cheapest_option finds the 2-phrase length", T.cheapest_option(phrase128)["units"] == 2)
check("every option is a legal H3 length", all(o["gen_frames"] in LEGAL for o in opts))
check("every option's gen covers its span", all(o["gen_frames"] >= o["span_frames"] for o in opts))

check("a unit past the ceiling has no legal multiple", T.unit_options(20.0) == [])
check("…and cheapest_option says None, not an error", T.cheapest_option(20.0) is None)
check("…and describe_options says so in words", "impossible" in T.describe_options(20.0))
check("no step, no options", T.unit_options(0) == [] and T.cheapest_option(None) is None)
check("describe_options lists every option",
      len(T.describe_options(phrase128, "4 bars").splitlines()) == len(opts) + 1)
check("a 120 BPM phrase costs nothing at all",
      T.cheapest_option(T.unit_seconds(bars(120), 4, "phrase (4 bars)"))["trim_frames"] == 0)

# -------------------------------------------------------------------------------------- the cues
c = T.cue(3.5, 0.8, "onset", "drop")
check("a cue carries its label", c["t"] == 3.5 and c["label"] == "drop")
check("strength is clamped to 0..1", T.cue(1, 4.0)["strength"] == 1.0 and T.cue(1, -2)["strength"] == 0.0)
check("a negative time is clamped", T.cue(-3)["t"] == 0.0)

# two onsets either side of one downbeat are one candidate, and the stronger one names it
snapped = T.snap_cues([T.cue(3.97, 0.4, "onset", "kick"), T.cue(4.03, 0.9, "onset", "snare")], 2.0)
check("cues on the same line merge", len(snapped) == 1, snapped)
check("…onto the line itself", abs(snapped[0]["t"] - 4.0) < 1e-9)
check("…keeping the strongest label", snapped[0]["label"] == "snare", snapped[0])
check("…and the strongest strength", abs(snapped[0]["strength"] - 0.9) < 1e-9)
weak_first = T.snap_cues([T.cue(4.03, 0.9, "onset", "snare"), T.cue(3.97, 0.4, "onset", "kick")], 2.0)
check("merge order does not matter", weak_first[0]["label"] == "snare" and
      abs(weak_first[0]["strength"] - 0.9) < 1e-9, weak_first[0])
check("snapped cues come out sorted",
      [round(x["t"], 6) for x in T.snap_cues([T.cue(8.1), T.cue(2.1), T.cue(6.1)], 2.0)] == [2.0, 6.0, 8.0])
check("no step = cues pass through, sorted",
      [x["t"] for x in T.snap_cues([T.cue(5), T.cue(1)], 0)] == [1.0, 5.0])
check("a cue past the end is dropped", T.snap_cues([T.cue(99)], 2.0, total=10.0) == [])

# ------------------------------------------------------------------------------- picking the cuts
def spans(shots):
    return [s["span"] for s in shots]


def legal_band(shots, allow_short_last=False):
    ok = True
    for i, s in enumerate(shots):
        last = i == len(shots) - 1
        if s["span"] < P.MIN_SECONDS - 1e-6 and not (last and allow_short_last):
            ok = False
        if s["span"] > P.MAX_SECONDS + 1e-6:
            ok = False
    return ok


b = T.plan_cuts(60.0)
check("no cues: the first boundary is the start", b[0]["t"] == 0.0)
check("no cues: the last boundary is the end", abs(b[-1]["t"] - 60.0) < 1e-9)
check("no cues: boundaries increase", all(b[i]["t"] < b[i + 1]["t"] for i in range(len(b) - 1)))
check("no cues, no grid: every cut is 'free'", all(x["via"] == "free" for x in b[1:-1]))
check("no cues: shots sit in H3's band", legal_band(T.shots_from_cuts(b)), spans(T.shots_from_cuts(b)))

# a track shorter than one shot is one (short) shot, not a crash
one = T.plan_shots(3.0)
check("a track shorter than a shot is one shot", len(one) == 1 and abs(one[0]["span"] - 3.0) < 0.05)
check("…and says so", any("shorter than H3's minimum" in n for n in T.warnings(one)), T.warnings(one))
check("a zero-length track plans nothing", T.plan_shots(0.0) == [])

# cues win when they are in range
cues = [T.cue(t, 1.0, "section", f"section {i}") for i, t in enumerate([7.0, 14.0, 21.0, 28.0])]
sh = T.plan_shots(28.0, cues, preferred=7.0)
check("cues become the cuts", [round(s["end"], 2) for s in sh] == [7.0, 14.0, 21.0, 28.0],
      [s["end"] for s in sh])
check("each cut names its cue", [s["cue"]["label"] for s in sh[:-1]] == ["section 0", "section 1", "section 2"],
      [s["cue"] for s in sh[:-1]])

# a strong cue outranks a weak one closer to the ideal length — by the strength DIFFERENCE times
# the pull (here 0.9 × 2.0 = 1.8 s), and no further. 32 s at 8 s a shot makes the target exactly 8.
strong_near = [T.cue(7.0, 1.0, "section", "drop"), T.cue(8.0, 0.1, "onset", "hat")]
sh = T.plan_shots(32.0, strong_near, preferred=8.0, cue_pull=2.0)
check("a strong cue beats a weak one nearer the target", abs(sh[0]["end"] - 7.0) <= HALF_FRAME,
      sh[0]["end"])
# …but not from four seconds away, which is more deviation than its strength can buy
strong_far = [T.cue(5.3, 1.0, "section", "drop"), T.cue(9.9, 0.9, "onset", "hat")]
sh = T.plan_shots(40.0, strong_far, preferred=10.0, cue_pull=2.0)
check("…and does not beat one from four seconds away", abs(sh[0]["end"] - 9.9) <= HALF_FRAME,
      sh[0]["end"])

# a cue outside the band cannot be taken, however strong
sh = T.plan_shots(40.0, [T.cue(2.0, 1.0, "section", "too soon")], preferred=7.0)
check("a cue inside the minimum is unreachable", sh[0]["span"] >= P.MIN_SECONDS - 1e-6, sh[0]["span"])
check("…and the cut is not credited to a cue", sh[0]["via"] != "cue" and sh[0]["cue"] is None)

# the grid is the fallback, before the raw ideal length
sh = T.plan_shots(40.0, [], preferred=7.0, step=2.0, offset=0.0)
check("with no cues the cut lands on a grid line",
      all(abs(s["end"] / 2.0 - round(s["end"] / 2.0)) < 1e-6 for s in sh[:-1]), [s["end"] for s in sh])
check("…and is credited to the grid, not to nothing",
      all(s["via"] == "grid" for s in sh[:-1]), [s["via"] for s in sh])
# a window narrower than the step can hold no line at all — that is 'free', and it is reported
sh = T.plan_shots(40.0, [], preferred=7.0, step=40.0, offset=0.0)
check("a grid too coarse to fit the window falls through to 'free'",
      any(s["via"] == "free" for s in sh[:-1]), [s["via"] for s in sh])

# ------------------------------------------------------------------------------------ the tail rule
worst_tail, cases = 99.0, 0
rng = random.Random(11)
for _ in range(300):
    total = rng.uniform(20.0, 240.0)
    pref = rng.uniform(P.MIN_SECONDS, 12.0)
    sh = T.plan_shots(total, [], preferred=pref, step=rng.choice([None, 1.875, 2.0]))
    cases += 1
    worst_tail = min(worst_tail, sh[-1]["span"])
    if not legal_band(sh):
        worst_tail = -1.0
        break
check(f"no plan ever strands a tail below the minimum ({cases} random tracks)",
      worst_tail >= P.MIN_SECONDS - 1e-6, f"{worst_tail:.3f} s")

check("a track just over the ceiling still splits in two",
      len(T.plan_shots(P.MAX_SECONDS + 0.4)) == 2)
check("…and neither half is unrenderable", legal_band(T.plan_shots(P.MAX_SECONDS + 0.4)))
check("one shot's worth, asked for as one shot, stays one shot",
      len(T.plan_shots(P.MAX_SECONDS, preferred=P.MAX_SECONDS)) == 1)
# …and the same length asked for at 7 s a shot becomes two, which is the bug that started this:
# stopping merely because the remainder FITS in one shot ignored `preferred` entirely.
check("…but at 7 s a shot the same length splits", len(T.plan_shots(P.MAX_SECONDS, preferred=7.0)) == 2)
check("a 14 s tail at 7 s a shot is two shots, not one",
      len(T.plan_shots(28.0, cues, preferred=7.0)) == 4)

check("shots_remaining respects the wish", T.shots_remaining(30.0, 7.0) == 4)
check("shots_remaining respects the ceiling", T.shots_remaining(60.0, 60.0) == 4, T.shots_remaining(60.0, 60.0))
check("shots_remaining respects the floor", T.shots_remaining(20.0, 1.0) == 3, T.shots_remaining(20.0, 1.0))
check("shots_remaining never returns zero", T.shots_remaining(1.0, 7.0) == 1)

# ------------------------------------------------------------- musical time → what H3 must render
sh = T.plan_shots(28.0, cues, preferred=7.0)
check("gen is always a legal H3 length", all(s["gen_frames"] in LEGAL for s in sh),
      [s["gen_frames"] for s in sh])
check("gen is never shorter than the span", all(s["gen_frames"] >= s["span_frames"] for s in sh))
check("trim is under one quantum", all(0 <= s["trim_frames"] < 17 for s in sh),
      [s["trim_frames"] for s in sh])
check("trim is exactly the overshoot",
      all(s["trim_frames"] == s["gen_frames"] - s["span_frames"] for s in sh))

# absolute positions: a seam is the same frame no matter what came before it
check("shots meet exactly, frame for frame",
      all(sh[i]["end_frame"] == sh[i + 1]["start_frame"] for i in range(len(sh) - 1)))
check("the spans sum to the whole track in frames",
      sum(s["span_frames"] for s in sh) == sh[-1]["end_frame"] - sh[0]["start_frame"])
check("the last shot ends at the track's end", abs(sh[-1]["end"] - 28.0) < HALF_FRAME)

# rounding must not accumulate: 40 shots off a grid that is not frame-aligned
odd = [T.cue(i * 6.4321, 1.0, "section") for i in range(1, 41)]
sh = T.plan_shots(40 * 6.4321, odd, preferred=6.4)
check("40 seams still meet exactly",
      all(sh[i]["end_frame"] == sh[i + 1]["start_frame"] for i in range(len(sh) - 1)))
check("…and the last one has not drifted",
      abs(sh[-1]["end"] - 40 * 6.4321) <= HALF_FRAME, sh[-1]["end"] - 40 * 6.4321)

# --------------------------------------------------------- THE claim: cuts land on the beat
worst, cases = 0.0, 0
rng = random.Random(3)
for _ in range(300):
    bpm = rng.uniform(70.0, 180.0)
    bpb = rng.choice([3, 4, 4, 4, 6])
    unit = rng.choice(T.CUT_UNITS)
    step = T.unit_seconds(T.bar_seconds(bpm, bpb), bpb, unit)
    offset = rng.uniform(0.0, step)
    total = rng.uniform(60.0, 200.0)
    lines = T.grid_times(step, total, offset=offset)
    downs = [T.cue(t, rng.choice([0.3, 0.6, 1.0]), "onset") for t in lines[::max(1, len(lines) // 24)]]
    sh = T.plan_shots(total, downs, preferred=rng.uniform(5.5, 11.0), step=step, offset=offset)
    cases += 1
    if not sh or not legal_band(sh) or any(s["gen_frames"] not in LEGAL for s in sh):
        worst = 99.0
        break
    for s in sh[:-1]:                      # the final boundary is the track end, not a beat
        if s["via"] == "free":             # …and a 'free' cut is the one case that is NOT on the
            continue                       #    music, which is exactly why it is reported as such
        k = round((s["end"] - offset) / step)
        worst = max(worst, abs(s["end"] - (offset + k * step)))
check(f"every musical cut lands within half a frame of its grid line ({cases} random songs)",
      worst <= HALF_FRAME + 1e-9, f"{worst * 1000:.1f} ms")
check("half a frame at 24 fps is 20.8 ms", abs(HALF_FRAME - 0.020833) < 1e-5)

check("frame_drift is bounded by half a frame",
      T.frame_drift(T.plan_cuts(120.0, [T.cue(i * 3.777, 1.0) for i in range(1, 30)])) <= HALF_FRAME + 1e-9)

# ---------------------------------------------------------------------------------- what comes out
sh = T.plan_shots(28.0, cues, preferred=7.0)
d = T.format_durations(sh)
check("durations is the comma list Phantas takes", d.count(",") == len(sh) - 1, d)
check("durations reports the GENERATED length, not the span",
      all(abs(float(x) - s["gen"]) <= 0.005 + 1e-9 for x, s in zip(d.split(", "), sh)), d)
check("…which is not the span, or there would be nothing to trim",
      any(abs(s["gen"] - s["span"]) > 1e-6 for s in sh))
check("durations parses back through Phantas' own parser",
      P.parse_durations(d, len(sh)) is not None and
      all(P.frames_for(v) == s["gen_frames"] for v, s in zip(P.parse_durations(d, len(sh)), sh)), d)

tr = T.format_trims(sh)
check("trims is a comma list of frames", all(x.strip().isdigit() for x in tr.split(",")), tr)
check("trims is blank when nothing is trimmed",
      T.format_trims([dict(s, trim_frames=0) for s in sh]) == "")

cu = T.format_cues(sh)
check("cues has one line per shot", len(cu.splitlines()) == len(sh), cu)
check("cues names the reason", "section 0" in cu, cu.splitlines()[0])
check("cues is NOT the beats format (it is labelled per shot)", cu.startswith("shot 1: "))

tl = T.timeline(sh)
check("timeline has one row per shot", len(tl.splitlines()) == len(sh))
check("timeline shows both clocks", "span" in tl and "gen" in tl and "trim" in tl)
check("timeline survives an empty plan", T.timeline([]) == "no shots")

de = T.describe(sh)
check("describe reports the overhead", "overhead" in de and "trimmed" in de, de)
check("describe survives an empty plan", T.describe([]) == "no shots")

# ------------------------------------------------------------------------------------- warnings
check("a clean plan is quiet", T.warnings(T.plan_shots(28.0, cues, preferred=7.0), total=28.0) == [],
      T.warnings(T.plan_shots(28.0, cues, preferred=7.0), total=28.0))
free = T.plan_shots(60.0, [])
check("a cut with no music under it is reported",
      any("not on the music" in n for n in T.warnings(free)), T.warnings(free))
# a bar-line cut is still ON the music and must NOT be apologised for, or the report cries wolf on
# every instrumental stretch and stops being read
grid_only = T.plan_shots(60.0, [], preferred=7.0, step=2.0)
check("a bar-line cut is not a warning", T.warnings(grid_only, total=60.0) == [],
      T.warnings(grid_only, total=60.0))
check("warnings never raise on junk", isinstance(T.warnings([]), list))

# --------------------------------------------------------------------- printable on the console
# `Siren Cast` prints its report under `verbose` and the node built on this module will too. A
# Windows console on a Cyrillic locale is cp1251: it has the em-dash and the middle dot, and it does
# NOT have '→'. So every string this module hands back has to survive that encoder, or "print the
# timeline" becomes a UnicodeEncodeError on the machine the pack is developed on.
sh = T.plan_shots(195.0, [T.cue(t, 1.0, "section", f"section {i}")
                          for i, t in enumerate(range(15, 196, 15))], preferred=7.0, step=7.5)
printable = True
for name, text in [("timeline", T.timeline(sh)), ("describe", T.describe(sh)),
                   ("format_cues", T.format_cues(sh)), ("warnings", "\n".join(T.warnings(sh, 195.0))),
                   ("durations", T.format_durations(sh)), ("trims", T.format_trims(sh))]:
    try:
        text.encode("cp1251")
    except UnicodeEncodeError as e:
        printable = False
        check(f"{name} is printable on a cp1251 console", False, e)
check("every report string survives a cp1251 console", printable)

check.done()
