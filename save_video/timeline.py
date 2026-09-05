"""Where each picture starts and stops — everything Save Clip decides before a pixel is touched.

Kept free of torch, PyAV and PIL so the arithmetic can be tested on its own: a slideshow that
drifts away from its song is not something you want to discover by watching a three-minute render.

**The one rule the rest follows from: the song says how long, the plan only says in what
proportion.** A Siren plan is written in *bars* (`Verse 1 | Nina | 16 bars`), and bars are only
seconds once you know the tempo — which lives on other nodes, and the moment the two disagree the
pictures slide off the music. Meanwhile the true length of the track is right there in the audio
input. So the plan is read at a nominal tempo, used for its RATIOS, and then scaled so the last
slide ends exactly with the last sample. The scale factor goes in the report: a plan that came out
5% short is worth knowing about, and it can no longer cost you sync.

The other consequence is that anything shaped like a length works here — bars, seconds, `m:ss`, or
the plain comma list of seconds `Orpheus (Audio -> Shots)` writes.
"""
import re

#: The tempo the bar rows are read at. Any value works — a plan written entirely in bars is scaled
#: to the audio afterwards, so it cancels out. It only shows through when ONE plan mixes bars with
#: literal seconds, which nothing in this pack writes.
NOMINAL_BPM = 120.0
NOMINAL_BEATS = 4

LAYOUT_ORDER = "one slide per section"
LAYOUT_LABEL = "by section label (the chorus comes back)"
LAYOUT_EVEN = "even (ignore the plan)"
LAYOUTS = [LAYOUT_ORDER, LAYOUT_LABEL, LAYOUT_EVEN]

#: A bare comma list of seconds — what Orpheus hands Phantas and Morpheus. Recognised only when the
#: text has no table pipe in it at all, so a real plan is never mistaken for one.
_DURATION_LIST = re.compile(r"^[\s\d.,]+$")


def mmss(seconds):
    """`0:12.40`. Hundredths, because a cut half a second late is visible and 0:12 hides it."""
    cs = int(round(max(0.0, float(seconds)) * 100))     # round FIRST: 59.999 s is 1:00.00, not 0:60.00
    m, cs = divmod(cs, 6000)
    return f"{m}:{cs / 100:05.2f}"


def srt_time(seconds):
    """`00:00:12,400` — SubRip's own clock. Whole milliseconds all the way down, so a fraction that
    rounds up carries into the seconds instead of writing a four-digit `,1000` no player accepts."""
    ms = int(round(max(0.0, float(seconds)) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def sections(plan):
    """A plan block -> ``[{label, weight}]`` in order, plus notes about what was dropped.

    ``weight`` is in nominal seconds and is never used as an absolute — :func:`lay_out` scales the
    whole list onto the song. Two shapes are accepted:

      * the Siren table (``label | voice | 16 bars | extra``), parsed by **Siren Cast's own**
        parser, so a row this reads and a row the sampler reads can never disagree;
      * a bare comma list of seconds (``3.50, 4.00, 8.00``) — Orpheus's ``durations``.
    """
    text = str(plan or "").strip()
    if not text:
        return [], []

    if "|" not in text and _DURATION_LIST.match(text):
        rows, notes = [], []
        for i, cell in enumerate(c.strip() for c in text.replace("\n", ",").split(",")):
            if not cell:
                continue
            try:
                secs = float(cell)
            except ValueError:
                notes.append(f"'{cell}' is not a length and was skipped")
                continue
            if secs > 0:
                rows.append({"label": f"shot {len(rows) + 1}", "weight": secs})
        if not rows:
            notes.append("the plan looked like a list of seconds but held no usable length")
        return rows, notes

    from ..siren.cast import _parse_plan          # local: keeps this module import-light
    parsed, notes = _parse_plan(text, NOMINAL_BPM, NOMINAL_BEATS)
    return [{"label": r["label"], "weight": r["seconds"]} for r in parsed], list(notes)


def section_spans(rows, total):
    """The plan's own sections on the song's clock: ``[{start, end, label}]``.

    Where the WORDS are — which is a different question from which picture is on screen, and the
    reason subtitles are timed off this rather than off the segments. One still over a whole song is
    a single segment; the song still has six sections, and the lyrics still belong to them.
    """
    weights = [max(0.0, float(r["weight"])) for r in rows or []]
    planned = sum(weights)
    if not weights or planned <= 0:
        return []
    scale = float(total) / planned
    out, at = [], 0.0
    for r, w in zip(rows, weights):
        out.append({"start": at, "end": at + w * scale, "label": r["label"]})
        at += w * scale
    out[-1]["end"] = float(total)          # kill the rounding dust
    return out


def _apportion(weights, count):
    """Hand out `count` indivisible slides over `weights`, one each minimum, largest remainder."""
    n = len(weights)
    if count <= n:
        return [1] * n
    total = sum(weights) or float(n)
    exact = [1 + (count - n) * (w / total) for w in weights]
    out = [int(e) for e in exact]
    for i in sorted(range(n), key=lambda i: exact[i] - out[i], reverse=True)[:count - sum(out)]:
        out[i] += 1
    return out


def _merge(segments):
    """Fuse neighbours showing the same picture.

    Not cosmetic: a boundary between a slide and itself would otherwise get a crossfade, and
    dissolving a picture into a copy of itself is a half-second of nothing that reads as a stutter.
    """
    out = []
    for s in segments:
        if out and out[-1]["slide"] == s["slide"]:
            out[-1]["end"] = s["end"]
            out[-1]["label"] = f"{out[-1]['label']} + {s['label']}".strip(" +")
        else:
            out.append(dict(s))
    return out


def lay_out(rows, slides, layout, total):
    """-> ``([{start, end, slide, label}], notes)`` covering exactly ``[0, total]``.

    Which picture lands where, in three rules that answer three different intentions:

      * **one slide per section** — the picture changes on the section boundary. With more slides
        than sections the extra ones subdivide the longest sections; with fewer, they cycle.
      * **by section label** — every row called `Chorus` gets the SAME slide, so the chorus shot
        comes back the way it does in a cut music video. This is the only mode that needs nothing
        but the labels the plan already carries.
      * **even** — the plan is ignored and the song is split equally. Also what happens with no
        plan wired at all.
    """
    n = max(1, int(slides))
    total = float(total)
    notes = []

    if not rows or layout == LAYOUT_EVEN:
        if rows and layout == LAYOUT_EVEN:
            notes.append(f"the plan's {len(rows)} section(s) were ignored — 'even' was chosen")
        segs = [{"start": total * i / n, "end": total * (i + 1) / n, "slide": i, "label": ""}
                for i in range(n)]
        return segs, notes

    # The plan's own clock, scaled onto the song. Everything downstream is in real seconds.
    weights = [max(0.0, float(r["weight"])) for r in rows]
    planned = sum(weights)
    if planned <= 0:
        notes.append("every section in the plan had a length of 0 — the song was split evenly instead")
        segs, _ = lay_out([], n, LAYOUT_EVEN, total)
        return segs, notes
    scale = total / planned
    if abs(scale - 1.0) > 0.02:
        notes.append(f"the plan runs {planned:.1f} s and the audio is {total:.1f} s, so the "
                     f"sections were stretched x{scale:.3f}. The pictures still land on the "
                     f"section boundaries, they are just not where the plan's bpm would put them.")

    if layout == LAYOUT_LABEL:
        order, index = [], {}
        for r in rows:
            key = " ".join(str(r["label"]).split()).lower()
            if key not in index:
                index[key] = len(order)
                order.append(key)
        picks = [index[" ".join(str(r["label"]).split()).lower()] % n for r in rows]
        if len(order) > n:
            notes.append(f"{len(order)} distinct section label(s) against {n} slide(s) — the "
                         f"pictures cycle, so two different sections share one")
        counts = None
    else:
        counts = _apportion(weights, n)
        picks = None
        if n < len(rows):
            notes.append(f"{n} slide(s) over {len(rows)} section(s) — the pictures cycle")

    spans = section_spans(rows, total)
    segs, slide = [], 0
    for i, span in enumerate(spans):
        if picks is not None:                                  # by label
            segs.append({**span, "slide": picks[i]})
            continue
        k = counts[i]
        for j in range(k):                                     # in order, subdividing if asked
            width = span["end"] - span["start"]
            segs.append({"start": span["start"] + width * j / k,
                         "end": span["start"] + width * (j + 1) / k,
                         "slide": (slide % n) if n >= len(rows) else (i % n),
                         "label": span["label"] if k == 1 else f"{span['label']} {j + 1}/{k}"})
            slide += 1

    segs[-1]["end"] = total
    return _merge(segs), notes


def frame_track(segments, fps, total, crossfade=0.0):
    """One entry per encoded frame: ``(t, a, b, mix)``.

    ``a`` / ``b`` are SEGMENT indices and ``mix`` is how far the dissolve from ``a`` to ``b`` has
    got (0 = nothing but ``a``). ``t`` is the frame's centre in seconds, which is what a Ken Burns
    move reads its progress off.

    The dissolve is **centred on the cut**, the way an editor's cross-dissolve is: half before the
    boundary, half after, so the moment the two pictures are equal is the moment the music turns.
    It is clamped to 40% of the shorter neighbour, because a 1-second fade across a 1.5-second
    section is not a transition, it is the whole shot.
    """
    fps = float(fps)
    n_frames = max(1, int(round(float(total) * fps)))
    if not segments:
        return [(i / fps, 0, 0, 0.0) for i in range(n_frames)]

    widths = []
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i + 1]
        widths.append(max(0.0, min(float(crossfade),
                                   0.4 * (a["end"] - a["start"]),
                                   0.4 * (b["end"] - b["start"]))))

    track, k = [], 0
    for i in range(n_frames):
        t = (i + 0.5) / fps
        while k + 1 < len(segments) and t >= segments[k]["end"]:
            k += 1
        a = b = k
        mix = 0.0
        # Only the boundary AHEAD can be dissolving here, or the one behind — never both, since a
        # width is capped at 40% of each side.
        for j, w in ((k, widths[k] if k < len(widths) else 0.0),
                     (k - 1, widths[k - 1] if 0 <= k - 1 < len(widths) else 0.0)):
            if w <= 0:
                continue
            boundary = segments[j]["end"]
            if boundary - w / 2 <= t < boundary + w / 2:
                a, b = j, j + 1
                mix = (t - (boundary - w / 2)) / w
                break
        track.append((t, a, b, mix))
    return track


# Zoom in / zoom out with a drift, cycled so four slides in a row do not all move the same way.
# (zoom_in?, x drift, y drift) with the drift in fractions of the room the zoom leaves over.
_MOVES = ((True, 1.0, 0.3), (False, -1.0, -0.3), (True, -0.6, 1.0), (False, 0.6, -1.0))


def ken_burns(index, amount):
    """The move for segment `index`: ``(zoom_start, zoom_end, (x0, y0), (x1, y1))``.

    Zoom is a factor over the frame; the centres are in -1..1, where +-1 is as far as the zoom's
    own slack allows. Deterministic in the index — a re-render of the same clip is the same clip.
    """
    amount = max(0.0, float(amount))
    if amount <= 0:
        return 1.0, 1.0, (0.0, 0.0), (0.0, 0.0)
    zoom_in, dx, dy = _MOVES[index % len(_MOVES)]
    lo, hi = 1.0, 1.0 + amount
    z0, z1 = (lo, hi) if zoom_in else (hi, lo)
    return z0, z1, (-dx / 2, -dy / 2), (dx / 2, dy / 2)


def srt(spans, lyric_sections):
    """A SubRip track: each section's own lines under its own span, from :func:`section_spans`.

    **Not** over the visual segments, which was the first version and was wrong: one still over a
    whole song is a single segment labelled `Intro + Verse 1 + Chorus + ...`, and nothing matched
    it — so the case the node exists for was the case that lost its subtitles. Where the pictures
    change and where the words are are two different questions.

    Matched by LABEL in order rather than by position, because a plan carries rows the lyrics never
    had — the instrumental blocks Siren Score adds to reach the target length. A section with no
    sung lines (a solo, a break) simply gets no cue rather than an empty one.
    """
    pending = list(lyric_sections or [])
    out, n = [], 0
    for seg in spans:
        want = " ".join(str(seg.get("label") or "").split()).lower()
        hit = None
        for i, sec in enumerate(pending):
            if " ".join(str(sec.get("label") or "").split()).lower() == want:
                hit = pending.pop(i)
                break
        text = "\n".join(ln for ln in (hit or {}).get("text", []) if str(ln).strip())
        if not text.strip():
            continue
        n += 1
        out.append(f"{n}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{text}\n")
    return "\n".join(out)


def report(segments, slides, fps, size, total, notes):
    """The block the node prints and puts on its `report` output.

    Plain hyphens, no arrows: this prints to a console that is cp1251 on the machine the pack is
    written on, and one nice-looking character would turn a report into a UnicodeEncodeError.
    """
    head = (f"Save Clip - {len(segments)} segment(s) · {mmss(total)} ({total:.1f} s) @ "
            f"{fps} fps · {size[0]}x{size[1]} · {slides} slide(s)")
    lines = [f"  {mmss(s['start'])} - {mmss(s['end'])}  slide {s['slide'] + 1:<3} "
             f"{s['end'] - s['start']:6.2f} s  {s.get('label') or ''}".rstrip()
             for s in segments]
    return "\n".join([head] + lines + [f"  ! {w}" for w in notes])
