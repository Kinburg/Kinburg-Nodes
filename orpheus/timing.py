"""Orpheus' clock: where a cut falls when the *music* decides it, and what H3 must generate to allow
it.

The whole module exists because of one collision. H3's shot lengths move in a 17-frame quantum —
0.708 s, fifteen legal lengths, all of it worked out in `phantas/timing.py` and imported from there
rather than restated. A bar of music at 120 BPM is 2.000 s. **The two grids are incommensurable**: no
whole number of bars is a legal shot length in general, so a cut that has to land on a downbeat, if
all you may choose is a shot length, arrives up to **±0.354 s** away. At 120 BPM that is 0.7 of a
beat. It is the difference between "cut to the music" and "the editor missed".

So this module never asks the H3 grid to land on the beat. It does the opposite:

  1. the cut is chosen in **musical** time — on a cue (a section boundary, a drop, a vocal entry) or,
     failing that, on a bar line;
  2. H3 is then asked for the **smallest legal length that is at least that long**;
  3. the overshoot comes back as a **trim in frames**, to be dropped at assembly.

Generate long, cut on the beat. Morpheus already trims at decode time (`seam_trim`), so the
machinery exists; what is new is that the trim is per shot and musically derived. The cost is honest
and worth knowing: the overshoot averages half a quantum, ~0.35 s a shot, and a trimmed shot ends
before it reaches its planned end keyframe — which is why this belongs with `anchor = continuous`,
where the shot is pulled by its end frame but starts from the previous shot's real tail.

**Boundaries are absolute frame indices from the top of the track**, never a running sum of per-shot
rounding. Rounding each span on its own drifts up to half a frame per seam and walks the picture off
the music over a three-minute clip — the same mistake Morpheus already avoids when it measures each
shot's audio slot from the cumulative position rather than from its own length.

Pure stdlib, like the module it borrows the grid from: no torch, no comfy, no audio. Everything here
is decided by numbers, so all of it runs and is tested headless — the detector that *produces* cues
from a waveform is a separate concern and lives elsewhere. `bar_seconds` is imported from
`siren/cast.py`, whose docstring calls it "the one place this arithmetic lives"; a second copy here
is exactly how the plan Siren aimed at and the seconds Orpheus computes would drift apart.
"""
import math

from ..phantas.timing import (FPS, MAX_FRAMES, MAX_SECONDS, MIN_FRAMES, MIN_SECONDS,
                              align_frames, range_warnings, seconds_for)
from ..siren.cast import _bar_seconds as bar_seconds, _mmss as mmss

EPS = 1e-6

#: The unit a cut may land on. Same vocabulary as `Siren Section`'s `snap`, plus the two longer
#: groupings, because a shot is 3-7 bars at a normal tempo and snapping to a single bar there gives
#: the picker far more lines than it can use.
CUT_UNITS = ["bar", "beat", "2 bars", "phrase (4 bars)"]
_UNIT_BARS = {"bar": 1.0, "2 bars": 2.0, "phrase (4 bars)": 4.0}

#: The average shot to aim for. Sits low in H3's 5.17-15.08 s band on purpose: music video cutting
#: lives at 6-8 s a shot, and the picker is free to run longer when the music gives it nowhere to cut.
DEFAULT_PREFERRED = 7.0

#: How many seconds of deviation a cue's *strength* buys it. Between two candidates it is the
#: strength DIFFERENCE that pays: at 2.0 a section boundary (1.0) outranks a weak onset (0.1) up to
#: 1.8 s further from the ideal length, and loses beyond that. Past a couple of seconds the shot
#: lengths themselves start to look arbitrary, which reads worse on screen than a cut one bar early.
DEFAULT_CUE_PULL = 2.0

#: How a cut was arrived at. `cue` is the music naming the moment; `grid` is a bar line, which is
#: still musical and still correct; `free` means neither fitted inside H3's band and the cut fell
#: where the length demanded — the only one of the three worth warning about.
VIA = ("cue", "grid", "free")


# --------------------------------------------------------------------------------------- the cues
def cue(t, strength=1.0, kind="onset", label=""):
    """One candidate cut point. `strength` (0..1) is *importance*, not loudness — the caller decides
    that a section boundary outranks a drum hit, because only the caller knows which is which."""
    return {"t": max(0.0, float(t)), "strength": max(0.0, min(1.0, float(strength))),
            "kind": str(kind), "label": str(label)}


def unit_seconds(bar_sec, beats_per_bar, unit):
    """The grid step named by `unit`, in seconds. None when there is no tempo to measure against —
    in which case cues still work and only the *fallback* line has nowhere to snap to."""
    if not bar_sec or float(bar_sec) <= 0:
        return None
    bar = float(bar_sec)
    if unit == "beat":
        n = max(1, int(beats_per_bar or 4))
        return bar / n
    return bar * _UNIT_BARS.get(unit, 1.0)


def unit_options(step, min_sec=MIN_SECONDS, max_sec=MAX_SECONDS):
    """Every shot length available when cutting on `step`, and what each one costs in trim.

    This exists because the cost is **not uniform and not small**. At 128 BPM a 4-bar phrase is
    7.50 s, which snaps up to 8.00 s — 12 frames, 6.2% of everything generated thrown away. Two
    phrases is 15.00 s, which snaps to 15.083 s — **2 frames, 0.6%**. Same song, same grid, ten
    times the waste, decided entirely by which unit you cut on.

    Nothing can be done about that (the two grids are what they are), but it can be *shown*, so the
    `cut_on` choice is made with the number in front of you instead of discovered in a render queue.
    """
    if not step or float(step) <= 0:
        return []
    step = float(step)
    out = []
    for k in range(1, int(float(max_sec) / step) + 2):
        span = k * step
        if not (float(min_sec) - EPS <= span <= float(max_sec) + EPS):
            continue
        span_frames = int(round(span * FPS))
        gen = align_frames(span_frames)
        out.append({"units": k, "span": span, "span_frames": span_frames, "gen_frames": gen,
                    "gen": seconds_for(gen), "trim_frames": gen - span_frames,
                    "waste": (gen - span_frames) / float(gen)})
    return out


def cheapest_option(step, min_sec=MIN_SECONDS, max_sec=MAX_SECONDS):
    """The shot length on this grid that wastes the least. None when the grid offers none at all —
    a unit longer than H3's ceiling has no legal multiple, which is a real answer, not an error."""
    opts = unit_options(step, min_sec, max_sec)
    return min(opts, key=lambda o: (o["trim_frames"], o["span"])) if opts else None


def describe_options(step, label=""):
    """The `cut_on` cost table, for the report."""
    opts = unit_options(step)
    if not opts:
        return (f"cutting on {label or f'{step:.2f} s'} is impossible: no whole number of them "
                f"lands inside H3's {MIN_SECONDS:.2f}-{MAX_SECONDS:.2f} s band")
    rows = [f"  {o['units']}x = {o['span']:5.2f} s  ->  gen {o['gen']:5.2f} s  "
            f"trim {o['trim_frames']:>2} fr ({100 * o['waste']:.1f}%)" for o in opts]
    return f"shot lengths available on {label or f'{step:.2f} s'}:\n" + "\n".join(rows)


def grid_times(step, total, offset=0.0, lo=None, hi=None):
    """Every grid line in `[lo, hi]` (defaults: the whole track). `offset` is the downbeat phase —
    where bar 1 actually starts, which is almost never sample zero."""
    if not step or float(step) <= 0:
        return []
    step, offset = float(step), float(offset)
    lo = 0.0 if lo is None else max(0.0, float(lo))
    hi = float(total) if hi is None else min(float(total), float(hi))
    if hi < lo:
        return []
    k0 = int(math.ceil((lo - offset) / step - EPS))
    k1 = int(math.floor((hi - offset) / step + EPS))
    return [offset + k * step for k in range(k0, k1 + 1) if offset + k * step >= -EPS]


def snap_cues(cues, step, offset=0.0, total=None):
    """Move every cue onto the nearest grid line, then merge the ones that land together.

    Merging is the point, not tidiness: a kick and a snare 30 ms apart are two onsets and one
    downbeat, and without this the picker sees two candidates at the same instant and the stronger
    one's label is the one that survives. Strength is taken as the max and the label comes from
    whichever cue was strongest, so a section boundary keeps its name when a drum hit lands on it.
    """
    if not step or float(step) <= 0:
        return sorted((dict(c) for c in cues), key=lambda c: c["t"])
    step, offset = float(step), float(offset)
    merged = {}
    for c in cues:
        k = int(math.floor((float(c["t"]) - offset) / step + 0.5))
        t = offset + k * step
        if t < -EPS or (total is not None and t > float(total) + EPS):
            continue
        prev = merged.get(k)
        if prev is None or float(c["strength"]) > prev["strength"]:
            merged[k] = dict(c, t=t)
            if prev is not None:
                merged[k]["strength"] = max(float(c["strength"]), prev["strength"])
        else:
            prev["strength"] = max(prev["strength"], float(c["strength"]))
    return [merged[k] for k in sorted(merged)]


# ------------------------------------------------------------------------------ picking the cuts
def _pick(lo, hi, target, cues, cue_pull, step, offset):
    """The one cut inside `[lo, hi]`, and how it was arrived at.

    A cue is scored against the ideal length rather than simply taken: `strength × pull` is what its
    importance is worth in seconds of deviation, so between two candidates the stronger one may sit
    further from the ideal length and still win — by their strength difference times the pull, and
    no further. Ties go to the earlier cut: a shot that runs long is a shot that drags.
    """
    best, best_score = None, None
    for c in cues:
        t = float(c["t"])
        if not (lo - EPS <= t <= hi + EPS):
            continue
        score = float(c["strength"]) * float(cue_pull) - abs(t - target)
        if best_score is None or score > best_score + EPS:
            best, best_score = c, score
    if best is not None:
        return min(hi, max(lo, float(best["t"]))), best, "cue"

    # Nothing named to cut on: fall back to the bar grid, which is still musical, and only then to
    # the raw ideal length. A long instrumental stretch has bar lines even with no transient worth a
    # name — but a window narrower than the grid step can contain no line at all, and that case is
    # what `free` exists to report rather than hide.
    lines = grid_times(step, hi, offset=offset, lo=lo, hi=hi) if step else []
    if lines:
        return min(lines, key=lambda t: (abs(t - target), t)), None, "grid"
    return min(hi, max(lo, target)), None, "free"


def shots_remaining(remaining, preferred, min_sec=MIN_SECONDS, max_sec=MAX_SECONDS):
    """How many shots the rest of the track should become.

    The same question `phantas.timing.resolve_count` answers in `duration` mode, asked again at
    every seam — and it has to be asked at every seam rather than once at the top, because a cut
    taken on a cue is never exactly the ideal length and the remainder has to be re-divided around
    it. `round(remaining / preferred)` is the wish; the feasible band `[⌈r/max⌉, ⌊r/min⌋]` is the
    law. Stopping merely because the remainder *fits* in one shot is what turns the last 14 s of a
    song into a single 14-second shot when 7 was asked for.
    """
    r = float(remaining)
    lo_n = max(1, int(math.ceil(r / float(max_sec) - EPS)))
    hi_n = max(lo_n, int(math.floor(r / float(min_sec) + EPS)))
    want = int(round(r / max(float(min_sec), float(preferred))))
    return min(hi_n, max(lo_n, max(1, want)))


def plan_cuts(total, cues=(), preferred=DEFAULT_PREFERRED, start=0.0, cue_pull=DEFAULT_CUE_PULL,
              step=None, offset=0.0, min_sec=MIN_SECONDS, max_sec=MAX_SECONDS):
    """Walk the track and return the shot boundaries: `n_shots + 1` records, first at `start`, last
    at `total`.

    Each cut's window is `[t + min_sec, t + max_sec]` narrowed by what the *rest* of the track still
    has to do. With `n` shots left to place, the remainder after this cut must be divisible into
    `n-1` of them — so the window shrinks to `[total − (n−1)·max, total − (n−1)·min]` as well.
    That single rule covers the whole tail: it can neither strand 1.2 s that no shot can render, nor
    leave 40 s that one shot cannot reach. It is always satisfiable, because `n` was chosen from the
    feasible band in the first place.
    """
    total, start = float(total), max(0.0, float(start))
    min_sec, max_sec = float(min_sec), float(max_sec)
    out = [{"t": start, "cue": None, "via": "free"}]
    if total - start <= EPS:
        return out

    t = start
    for _ in range(4096):        # a cut advances by min_sec, so the bound is unreachable and cheap
        n = shots_remaining(total - t, preferred, min_sec, max_sec)
        if n <= 1:
            break
        lo = max(t + min_sec, total - (n - 1) * max_sec)
        hi = min(t + max_sec, total - (n - 1) * min_sec)
        if hi < lo - EPS:        # the band arithmetic forbids this; belt-and-braces
            break
        lo, hi = min(lo, hi), max(lo, hi)
        target = min(hi, max(lo, t + (total - t) / n))
        cut, chosen, via = _pick(lo, hi, target, cues, cue_pull, step, offset)
        if cut <= t + EPS:       # a cue sitting exactly on `t`, or a degenerate window
            cut, chosen, via = target, None, "free"
        out.append({"t": cut, "cue": chosen, "via": via})
        t = cut

    out.append({"t": total, "cue": None, "via": "free"})
    return out


# ------------------------------------------------------------- musical time → what H3 must render
def shots_from_cuts(boundaries):
    """Boundaries → one record per shot, carrying both clocks.

    `span_frames` is measured between **absolute** frame positions, so the seam at 2:41 is at the
    same frame no matter how many shots came before it. `gen_frames` is what H3 runs (snapped up
    onto its grid) and `trim_frames` is what comes off the tail to put the cut back on the beat.
    """
    if len(boundaries) < 2:
        return []
    pos = [int(round(float(b["t"]) * FPS)) for b in boundaries]
    shots = []
    for i in range(len(pos) - 1):
        span = pos[i + 1] - pos[i]
        gen = align_frames(span)
        shots.append({
            "index": i,
            "start": pos[i] / float(FPS),
            "end": pos[i + 1] / float(FPS),
            "start_frame": pos[i],
            "end_frame": pos[i + 1],
            "span_frames": span,
            "span": span / float(FPS),
            "gen_frames": gen,
            "gen": seconds_for(gen),
            "trim_frames": gen - span,
            "trim": (gen - span) / float(FPS),
            "cue": boundaries[i + 1]["cue"],
            "via": boundaries[i + 1]["via"],
        })
    return shots


def plan_shots(total, cues=(), **kw):
    """`plan_cuts` and `shots_from_cuts` in one call — what the node actually uses."""
    return shots_from_cuts(plan_cuts(total, cues, **kw))


# ---------------------------------------------------------------------------------- what comes out
def format_durations(shots):
    """The `durations` string Phantas and Morpheus already take. **The generated length, not the
    musical span** — this is what the sampler is being told to render; the span is what survives
    the trim."""
    return ", ".join(f"{s['gen']:.2f}" for s in shots)


def format_trims(shots):
    """The per-shot trim in frames, in the comma-list shape `durations` and `links` use. Blank when
    nothing needs trimming, so a graph that ignores trimming stays byte-identical."""
    if not any(s["trim_frames"] for s in shots):
        return ""
    return ", ".join(str(s["trim_frames"]) for s in shots)


def format_cues(shots):
    """One line per shot naming *why* the cut is there.

    Deliberately NOT the `beats` string. `Morpheus Storyboard` skips its own planning call when
    `beats` is filled ("your lines win"), so feeding it "0:48 — drop" as a shot's whole direction is
    how you get a shot that films a drop and nothing else. This is for reading, and for merging into
    `beats` by hand where it earns its place.
    """
    lines = []
    for s in shots:
        lines.append(f"shot {s['index'] + 1}: {mmss(s['start'])}-{mmss(s['end'])} — {_why(s, shots)}")
    return "\n".join(lines)


def _why(shot, shots):
    """What to call the cut that ends this shot, in a report. The last shot ends because the track
    does, which is not the same thing as the music offering nothing — and reading "no cue in range"
    against the final shot of every clip would train you to ignore the words."""
    c = shot["cue"]
    if c is not None:
        return c["label"] or c["kind"]
    if shot["index"] == len(shots) - 1:
        return "end of track"
    return "bar line" if shot["via"] == "grid" else "no cue in range"


def timeline(shots):
    """The block the report opens with: every shot, both clocks, and what is thrown away.

    Plain hyphens rather than an arrow on purpose. `Siren Cast` prints its report to the console
    under `verbose`, this node will do the same, and a Windows console on a Cyrillic locale is
    cp1251 — which has the em-dash and the middle dot but **not** `→`. One nice-looking character
    would turn "print my timeline" into a UnicodeEncodeError on the machine this pack is written on.
    """
    if not shots:
        return "no shots"
    rows = []
    for s in shots:
        why = _why(s, shots)
        rows.append(f"  {s['index'] + 1:>2}  {mmss(s['start'])} - {mmss(s['end'])}   "
                    f"span {s['span']:5.2f} s   gen {s['gen']:5.2f} s ({s['gen_frames']:>3} fr)   "
                    f"trim {s['trim_frames']:>2} fr ({s['trim']:.2f} s)   {why}")
    return "\n".join(rows)


def describe(shots):
    """One line: how many shots, how long the cut is, and what the trimming costs."""
    if not shots:
        return "no shots"
    span = sum(s["span_frames"] for s in shots)
    gen = sum(s["gen_frames"] for s in shots)
    trim = gen - span
    return (f"{len(shots)} shot(s): {span / float(FPS):.2f} s of clip from "
            f"{gen / float(FPS):.2f} s generated — {trim} frames ({trim / float(FPS):.2f} s) trimmed, "
            f"{100.0 * trim / gen:.1f}% overhead")


def frame_drift(boundaries):
    """The worst distance, in seconds, between a cut as the music wanted it and the frame it can
    actually land on. Bounded by half a frame (20.8 ms at 24 fps) — which is the number that says
    the approach works, so it goes in the report rather than being assumed."""
    worst = 0.0
    for b in boundaries:
        t = float(b["t"])
        worst = max(worst, abs(t - round(t * FPS) / float(FPS)))
    return worst


def warnings(shots, total=None):
    """Everything worth saying out loud about a plan that still ran.

    Nothing here raises. A shot outside H3's trained range renders — it just renders worse — and a
    board that refuses to come back at all because one section was 4 s long is more annoying than one
    that says so.
    """
    notes = list(range_warnings([s["gen_frames"] for s in shots]))
    short = [s["index"] + 1 for s in shots if s["span_frames"] < MIN_FRAMES]
    if short:
        notes.append(f"shot(s) {', '.join(map(str, short))} are shorter than H3's minimum "
                     f"{MIN_SECONDS:.2f} s — the music asked for a cut the model cannot render that "
                     f"close. Raise 'preferred_length' or cut on a longer unit.")
    long_ = [s["index"] + 1 for s in shots if s["gen_frames"] > MAX_FRAMES]
    if long_:
        notes.append(f"shot(s) {', '.join(map(str, long_))} exceed H3's {MAX_SECONDS:.2f} s ceiling "
                     f"after snapping up — split them with an extra cue.")
    # A `grid` cut is a bar line and needs no apology — it is still on the music. `free` is the one
    # worth saying out loud: neither a cue nor a bar line fitted inside H3's band there, so that
    # seam is the only kind that can land off the beat.
    free = [s["index"] + 1 for s in shots[:-1] if s["via"] == "free"]
    if free:
        notes.append(f"shot(s) {', '.join(map(str, free))} were cut where the length demanded, not "
                     f"on the music — nothing fitted inside H3's {MIN_SECONDS:.2f}-{MAX_SECONDS:.2f} s "
                     f"band there. Cut on a shorter unit, or give the detector more sensitivity.")
    if total is not None and shots:
        gap = float(total) - shots[-1]["end"]
        if abs(gap) > 1.0 / FPS:
            notes.append(f"the plan ends {gap:+.2f} s from the end of the track")
    return notes
