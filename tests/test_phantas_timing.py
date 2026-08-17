"""Phantas' clock: the frame grid, the three counting units, and laying a target length onto shots.

Pure arithmetic, so this suite is fast and imports nothing heavy — except for one check that pulls
in ComfyUI's real H3 helper to prove the mirrored `align_frames` still agrees with it. Without that
check the mirror could drift from the thing it mirrors and nothing would notice.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, comfy_on_path, fake_package, load_module  # noqa: E402

fake_package("kn", "phantas")
T = load_module("kn.phantas.timing", "phantas/timing.py")

check = Checker()

# --------------------------------------------------------------------------------------- the grid
legal = T.legal_frames()
check("fifteen legal shot lengths", len(legal) == 15, legal[:3])
check("range ends are the trained bounds", legal[0] == 124 and legal[-1] == 362)
check("every legal length is on the 17k+5 grid", all(f % 17 == 5 for f in legal))
check("the step is 0.708 s", abs(T.seconds_for(legal[1]) - T.seconds_for(legal[0]) - 17 / 24) < 1e-9)
check("MIN/MAX seconds match the bounds",
      abs(T.MIN_SECONDS - 124 / 24) < 1e-9 and abs(T.MAX_SECONDS - 362 / 24) < 1e-9)

check("align rounds up onto the grid", T.align_frames(125) == 141 and T.align_frames(124) == 124)
check("align never goes below 5", T.align_frames(0) == 5 and T.align_frames(-9) == 5)
check("frames_for snaps a typed duration UP", T.frames_for(5.0) == 124 and T.frames_for(5.2) == 141)
check("frames_for(6.0) is the documented 158", T.frames_for(6.0) == 158)
check("8.0 s is exactly on the grid", T.frames_for(8.0) == 192 and 192 % 17 == 5)
check("default duration snaps to the shortest shot", T.frames_for(T.DEFAULT_SECONDS) == 124)

check("nearest rounds DOWN when down is closer", T.nearest_frames(132) == 124)
check("nearest rounds UP when up is closer", T.nearest_frames(133) == 141)
check("nearest ties round up", T.nearest_frames(132.5) == 141)
check("nearest clamps below", T.nearest_frames(10) == 124)
check("nearest clamps above", T.nearest_frames(9999) == 362)
check("nearest always returns a legal length",
      all(T.nearest_frames(n) in legal for n in range(0, 500, 7)))

# the mirror: `align_frames` must agree with the real thing it copies, everywhere
try:
    comfy_on_path()
    from comfy_extras.nodes_minimax_h3 import FPS as _real_fps, align_frame_count as _real
    check("align_frames matches h3.align_frame_count over 5..400",
          all(T.align_frames(n) == _real(max(5, n)) for n in range(0, 400)))
    check("FPS matches h3", T.FPS == _real_fps)
except Exception as e:  # pragma: no cover - a checkout without H3 support
    print(f"  --   h3 comparison skipped ({type(e).__name__}: {e})")

# ------------------------------------------------------------------------------ counting units
check("frames: 4 pictures = 3 shots", T.resolve_count("frames", 4) == (4, 3))
check("scenes: 3 scenes = 4 pictures", T.resolve_count("scenes", 3) == (4, 3))
check("scenes: 2 scenes = 3 pictures", T.resolve_count("scenes", 2) == (3, 2))
check("the two units are the same number ±1",
      all(T.resolve_count("frames", s + 1) == T.resolve_count("scenes", s) for s in range(1, 20)))


def catches(fn, *a, **kw):
    """The message of the ValueError, or None if it did not raise."""
    try:
        fn(*a, **kw)
        return None
    except ValueError as e:
        return str(e)


check("frames: one picture is not a board", "at least 2 keyframes" in (catches(T.resolve_count, "frames", 1) or ""))
check("scenes: zero scenes rejected", "at least 1 scene" in (catches(T.resolve_count, "scenes", 0) or ""))
check("an unknown mode is rejected", "not one of" in (catches(T.resolve_count, "shots", 3) or ""))

# duration as the third unit
nf, ns = T.resolve_count("duration", 0, target_seconds=31.0)
check("duration: 31 s at the default becomes 6 shots / 7 pictures", (nf, ns) == (7, 6), (nf, ns))
check("duration: the shot count is always feasible",
      all(T.check_target(T.resolve_count("duration", 0, target_seconds=t)[1], t) is None
          for t in [5.2, 8, 12, 20, 31, 45, 60, 120]))
check("duration: a longer preferred shot means fewer shots",
      T.resolve_count("duration", 0, 60.0, preferred=12.0)[1]
      < T.resolve_count("duration", 0, 60.0, preferred=5.17)[1])
check("duration below one shot is rejected",
      "shorter than a single shot" in (catches(T.resolve_count, "duration", 0, 4.0) or ""))

# ----------------------------------------------------------------- the over-determination error
msg = catches(T.check_target, 4, 12.0)
check("12 s of 4 scenes is an error, not a clamp", msg is not None)
check("…and the error states the legal band", msg is not None and "20.67-60.33" in msg, msg)
check("a target inside the band is accepted", T.check_target(4, 30.0) is None)
check("the band edges are inclusive",
      T.check_target(3, 3 * T.MIN_SECONDS) is None and T.check_target(3, 3 * T.MAX_SECONDS) is None)
check("too LONG for the count is refused too", catches(T.check_target, 2, 40.0) is not None)

# ------------------------------------------------------------------------------- laying out time
f = T.plan_frames(4)
check("no target: every shot gets the preferred length", f == [124, 124, 124, 124], f)
check("no target: weights still shape the clip",
      T.plan_frames(3, weights=[1, 3, 1]) != T.plan_frames(3, weights=[1, 1, 1]))

f = T.plan_frames(3, weights=[1, 2, 1], target_seconds=24.0)
check("weighted: all lengths legal", all(x in legal for x in f), f)
check("weighted: the heavy shot is the longest", f[1] == max(f) and f[1] > f[0], f)
check("weighted: symmetric weights give symmetric shots", f[0] == f[2], f)
check("weighted: total within half a step of the target",
      abs(T.seconds_for(sum(f)) - 24.0) <= 0.36, T.describe(f))

check("equal weights split evenly", len(set(T.plan_frames(5, target_seconds=40.0))) <= 2)

# skewed weights near the ceiling: one shot pins to the maximum and the rest must make up hundreds
# of frames between them, one 17-frame quantum at a time
skew = T.plan_frames(12, weights=[5] + [1] * 11, target_seconds=12 * T.MAX_SECONDS - 1)
check("skewed weights still reach a target near the ceiling",
      abs(T.seconds_for(sum(skew)) - (12 * T.MAX_SECONDS - 1)) <= 0.36, T.describe(skew))

# the ±0.35 s promise, over the whole achievable band
worst, cases = 0.0, 0
rng = random.Random(7)
for _ in range(400):
    n = rng.randint(1, 12)
    t = rng.uniform(n * T.MIN_SECONDS, n * T.MAX_SECONDS)
    w = [rng.choice([1, 1, 2, 3, 5]) for _ in range(n)]
    got = T.plan_frames(n, weights=w, target_seconds=t)
    cases += 1
    if len(got) != n or any(x not in legal for x in got):
        worst = 99.0
        break
    worst = max(worst, abs(T.seconds_for(sum(got)) - t))
check(f"any achievable target is hit within 0.36 s ({cases} random cases)", worst <= 0.36, f"{worst:.3f} s")

check("a target at the very bottom of the band works",
      T.plan_frames(4, target_seconds=4 * T.MIN_SECONDS) == [124] * 4)
check("a target at the very top of the band works",
      T.plan_frames(4, target_seconds=4 * T.MAX_SECONDS) == [362] * 4)
check("an impossible target raises from plan_frames too",
      catches(T.plan_frames, 4, None, 12.0) is not None)

# a planner that miscounts must not break the clock
check("weights of the wrong length fall back", len(T.plan_frames(4, weights=[2, 2])) == 4)
check("weights of None fall back", T.plan_frames(3, weights=None) == T.plan_frames(3, weights=[1, 1, 1]))
check("garbage weights fall back", T.plan_frames(3, weights=["x", -4, None]) == T.plan_frames(3))
check("normalize keeps positive values", T.normalize_weights([1, 2.5, 0, -1], 4) == [1.0, 2.5, 1.0, 1.0])

# ------------------------------------------------------------------------------ typed durations
check("empty durations = let the planner decide", T.parse_durations("", 3) is None)
check("whitespace only is still empty", T.parse_durations("   \n ", 3) is None)
check("one value applies to every shot", T.parse_durations("5.17", 3) == [5.17] * 3)
check("the last value repeats", T.parse_durations("5.17, 8", 4) == [5.17, 8.0, 8.0, 8.0])
check("extra values are ignored", T.parse_durations("5, 6, 7", 2) == [5.0, 6.0])
check("semicolons and spaces split too", T.parse_durations("5; 6 7", 3) == [5.0, 6.0, 7.0])
check("a non-number is a clean error", "is not a number" in (catches(T.parse_durations, "5, soon", 2) or ""))

# ------------------------------------------------------------------------------------- reporting
check("range_warnings is quiet inside the range", T.range_warnings([124, 362]) == [])
check("range_warnings flags a short shot", len(T.range_warnings([107, 141])) == 1)
check("range_warnings names the shot", "shot 2" in T.range_warnings([124, 400])[0])
check("describe reports the total", "→  10.33 s total" in T.describe([124, 124]), T.describe([124, 124]))
check("describe survives an empty board", T.describe([]) == "no shots")

check.done()
