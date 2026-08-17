"""The clock of a Phantas board: how many keyframes, and how long each shot between them runs.

A keyframe sits **between** shots — frame k ends shot k-1 and starts shot k — so a board of N
keyframes is a chain of N-1 shots and the two numbers are always one apart. `frames`, `scenes` and
`duration` are therefore not three schemes but three *units for the same variable*: name the
pictures, name the shots, or name the clock and let the shot count fall out of it.

Everything here is integer arithmetic on H3's frame grid, and that is the point of the module
existing at all. A shot's length is not continuous: frame counts must satisfy `n % 17 == 5`, and the
model is trained on 124-362 of them, so a shot may be exactly one of **fifteen** lengths —

    5.17  5.88  6.58  7.29  8.00  8.71  9.42  10.13  10.83  11.54  12.25  12.96  13.67  14.38  15.08

— spaced 17 frames (0.708 s) apart. Two consequences shape the whole design:

  * **The LLM must never name seconds.** It names *weights* — how much change each transition
    carries — and the seconds are laid out here. A planner that writes "5.2 s" gets 5.88 s from the
    grid and the total silently drifts; a planner that writes weights cannot be wrong about time.
  * **A target length and a shot count over-determine each other.** n shots can only add up to
    between n×5.17 s and n×15.08 s. Asking for 12 s of four scenes is not a preference to be
    clamped, it is a contradiction, and it is raised as one.

Within the achievable band any total is reachable to within half a grid step — ±0.35 s — because
every shot moves in the same 17-frame quantum.

Pure stdlib on purpose: no torch, no comfy. `align_frames` mirrors
`comfy_extras.nodes_minimax_h3.align_frame_count` rather than importing it, so the arithmetic runs
in a headless test; the suite asserts the two agree across the whole range, which is the only thing
that keeps the mirror honest.
"""
import math
import re

FPS = 24
_GRID, _PHASE = 17, 5             # H3 runs 17k+5 frames
MIN_FRAMES, MAX_FRAMES = 124, 362  # …and is trained on this many of them
MIN_SECONDS = MIN_FRAMES / float(FPS)   # 5.1667
MAX_SECONDS = MAX_FRAMES / float(FPS)   # 15.0833
DEFAULT_SECONDS = 5.17            # the shortest trained shot, and Morpheus's own default

COUNT_MODES = ["frames", "scenes", "duration"]


# ------------------------------------------------------------------------------------- the grid
def align_frames(n):
    """The smallest legal frame count that is at least `n`.

    Mirrors `h3.align_frame_count` (`while n % 17 != 5: n += 1`) in closed form."""
    n = max(5, int(n))
    return n + ((_PHASE - n) % _GRID)


def nearest_frames(n):
    """The legal frame count CLOSEST to `n`, clamped to the trained range (ties round up).

    Distinct from `frames_for` on purpose. A duration somebody typed is snapped UP — ask for 6 s and
    getting 5.88 s would be a lie. A share worked out by the planner is snapped to whichever side is
    closer, because rounding fifteen shares up would push the total past the target every time.
    """
    n = int(math.floor(float(n) + 0.5))   # half-UP: python's round() would send 132.5 down to 132
    up = align_frames(n)
    down = up - _GRID
    best = down if (down >= MIN_FRAMES and (n - down) < (up - n)) else up
    return min(MAX_FRAMES, max(MIN_FRAMES, best))


def frames_for(seconds):
    """Seconds → the frame count H3 will actually run. Snaps UP, exactly as Morpheus does."""
    return align_frames(int(round(float(seconds) * FPS)))


def seconds_for(frames):
    return int(frames) / float(FPS)


def legal_frames():
    """Every frame count a shot may have — the fifteen of them."""
    return list(range(MIN_FRAMES, MAX_FRAMES + 1, _GRID))


def range_warnings(frames):
    """One line per shot that falls outside the trained range. Out-of-range still runs, so this
    warns rather than raises — the same call Morpheus makes about the same numbers."""
    out = []
    for i, f in enumerate(frames):
        if not (MIN_FRAMES <= f <= MAX_FRAMES):
            out.append(f"shot {i + 1}: {f} frames ({seconds_for(f):.2f} s) is outside the model's "
                       f"trained {MIN_FRAMES}-{MAX_FRAMES} frame range")
    return out


# ------------------------------------------------------------------------------- how many of what
def resolve_count(mode, count, target_seconds=0.0, preferred=DEFAULT_SECONDS):
    """`(keyframes, shots)` from whichever unit was typed in.

    `frames` names the pictures, `scenes` names the shots, `duration` names neither and works the
    shot count out from the clock — landing inside the band the grid allows for that length, as
    close to `preferred` seconds a shot as it can get.
    """
    if mode == "frames":
        n_frames = int(count)
        if n_frames < 2:
            raise ValueError("frames: a board needs at least 2 keyframes — a shot runs BETWEEN two "
                             "of them, so one picture is not a board.")
        n_shots = n_frames - 1
    elif mode == "scenes":
        n_shots = int(count)
        if n_shots < 1:
            raise ValueError("scenes: a board needs at least 1 scene.")
        n_frames = n_shots + 1
    elif mode == "duration":
        t = float(target_seconds)
        if t < MIN_SECONDS:
            raise ValueError(f"duration: {t:.2f} s is shorter than a single shot — {MIN_SECONDS:.2f} s "
                             f"is the shortest H3 runs.")
        lo = int(math.ceil(t / MAX_SECONDS - 1e-9))        # fewer shots than this cannot stretch to t
        hi = int(math.floor(t / MIN_SECONDS + 1e-9))       # more than this cannot fit inside t
        want = int(round(t / max(MIN_SECONDS, float(preferred))))
        n_shots = min(hi, max(lo, max(1, want)))
        n_frames = n_shots + 1
    else:
        raise ValueError(f"count_mode: '{mode}' is not one of {COUNT_MODES}")
    return n_frames, n_shots


def check_target(n_shots, target_seconds):
    """Raise unless `n_shots` shots can actually add up to `target_seconds`.

    Never clamps. On a scale whose smallest step is 5.17 s, quietly rounding "12 s" up to the legal
    20.67 s hands back a clip nearly twice the length that was asked for, and nothing on screen
    would say so.
    """
    lo, hi = n_shots * MIN_SECONDS, n_shots * MAX_SECONDS
    if not (lo - 1e-6 <= float(target_seconds) <= hi + 1e-6):
        raise ValueError(
            f"target_length {float(target_seconds):.2f} s is impossible for {n_shots} shot(s): they "
            f"can only add up to {lo:.2f}-{hi:.2f} s. Change the count or change the target.")


# ------------------------------------------------------------------------------- laying out time
def normalize_weights(weights, n_shots):
    """`n_shots` positive weights. Anything missing, short, long or unusable becomes an equal
    share: a planner that miscounts must not be able to break the clock."""
    out = []
    for i in range(n_shots):
        try:
            v = float(weights[i])
        except Exception:
            v = 1.0
        out.append(v if v > 0 else 1.0)
    return out


def plan_frames(n_shots, weights=None, target_seconds=0.0, preferred=DEFAULT_SECONDS):
    """Frame counts for `n_shots` shots: all legal, split by weight, as close to the target as the
    grid allows.

    The two halves answer different questions, and conflating them silently throws the weights away:

      * **with a target** there is a fixed budget to divide, so the weights set each shot's *share*
        of it and the residue is nudged until the sum lands.
      * **without one** there is no budget, so `preferred` is the AVERAGE shot and each shot takes
        its share of that. Dividing `n × preferred` instead would fail at the default: `preferred`
        is the shortest legal shot, so the budget would already be at the floor of the band and
        every weight would clamp back to the same 5.17 s.
    """
    if n_shots < 1:
        raise ValueError("plan_frames: a board needs at least one shot.")
    w = normalize_weights(weights, n_shots)
    if not (target_seconds and float(target_seconds) > 0):
        mean = sum(w) / float(n_shots)
        return [nearest_frames(float(preferred) * FPS * x / mean) for x in w]

    check_target(n_shots, target_seconds)
    total = int(round(float(target_seconds) * FPS))
    total = min(n_shots * MAX_FRAMES, max(n_shots * MIN_FRAMES, total))

    share = sum(w)
    ideal = [total * x / share for x in w]
    frames = [nearest_frames(v) for v in ideal]

    # Residue: every shot moves in the same 17-frame quantum, so the total lands within half a step
    # of the target. Each nudge goes to whichever shot is furthest from its ideal share in that
    # direction, which keeps the *proportions* honest while the sum converges.
    # The cap has to allow every shot to cross the WHOLE band one quantum at a time: skewed weights
    # push one shot to the ceiling and leave the rest to make up hundreds of frames between them.
    diff = total - sum(frames)
    for _ in range(n_shots * ((MAX_FRAMES - MIN_FRAMES) // _GRID + 1) + 4):
        if abs(diff) < _GRID / 2.0:
            break
        step = _GRID if diff > 0 else -_GRID
        cands = [i for i in range(n_shots) if MIN_FRAMES <= frames[i] + step <= MAX_FRAMES]
        if not cands:
            break
        i = max(cands, key=lambda j: (ideal[j] - frames[j]) * (1 if step > 0 else -1))
        frames[i] += step
        diff -= step
    return frames


def parse_durations(text, n_shots):
    """`""` → None (nobody typed anything, so the planner decides). `"5.17"` → the same for every
    shot. `"5.17, 8"` → per shot with the last value repeating — the convention Morpheus already
    uses for its own `durations` and `links`."""
    toks = [t for t in re.split(r"[,;\s]+", (text or "").strip()) if t]
    if not toks:
        return None
    vals = []
    for tok in toks:
        try:
            vals.append(max(0.2, float(tok)))
        except ValueError:
            raise ValueError(f"durations: '{tok}' is not a number (use '5.17' or '5.17, 8, 5.17')")
    return [vals[min(i, len(vals) - 1)] for i in range(n_shots)]


def describe(frames):
    """One line for the report: what each shot got, and what that adds up to."""
    if not frames:
        return "no shots"
    per = " · ".join(f"{seconds_for(f):.2f}" for f in frames)
    return (f"{len(frames)} shot(s): {per} s  →  {seconds_for(sum(frames)):.2f} s total "
            f"({sum(frames)} frames)")
