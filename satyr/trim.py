"""Cutting the stretches where nobody sings.

This is the lever that works. Measured on a real plan: YuE2 wrote 4:33 for a lyric AceStep had sung
in 3:00, and **61 of its 171 bars — 98 seconds, 36% of the song — carried no sung note at all**. A
40-second introduction, a 29-second outro, an interlude. Dropping only those bars brought the plan to
3:45 with every one of its 326 sung notes untouched, and the take came back shorter, livelier and
closer to what was wanted. The model obeys a bar count.

**Nothing sung is ever touched.** The unit of the cut is a whole group — one span of the song carried
by both voices at once — and only groups whose Vocal line holds no note at all are candidates. So the
melody, the words, the registers and the two singers all come through exactly as the model wrote
them; what goes is silence.

**Which end survives depends on where the stretch sits.** An introduction leads *into* the singing,
so its last bars are the ones worth keeping — they are the approach, and cutting from the front keeps
the hand-off. Everything else leads *out* of the singing it follows, so the first bars are kept
instead and the trailing repetition goes. The model tends to write the tail of a section as
repetition of its own figure, which is exactly what can be spared.

**A stretch is a run, not a section.** An introduction is sometimes one `%` label and sometimes four
in a row, and a limit that applied per label would cut a four-label intro to four times the length
asked for. So consecutive silent sections are gathered first and the limit applies to the run.

Sections are classified by what is actually in them, not by their labels. A `% verse` the model wrote
with no vocal is an instrumental passage whatever it is called, and an `% interlude` that sings is
not a gap. The label vocabulary is fixed (`comfy/audio_encoders/sheetsage2.py` has all 23 of them)
but it describes the model's reading of the music, not whether a voice is present.
"""
from . import notation as N

INTRO, OUTRO, BETWEEN = "intro", "outro", "between"


def sings(score, section):
    """Does anyone sing in this section? The only question that decides whether it can be cut."""
    return any(N.attacks(score.lines[group.vocal], score.key)
               for group in section.groups if group.vocal is not None)


def stretches(score):
    """The runs of silent sections, as (kind, [section, ...]) in performance order.

    The first run is the introduction and the last is the outro only when nothing sings before or
    after them — a song that opens on a voice has no introduction to cut, however long its first
    instrumental section is.
    """
    voiced = [sings(score, section) for section in score.sections]
    first = next((i for i, v in enumerate(voiced) if v), None)
    last = next((i for i in range(len(voiced) - 1, -1, -1) if voiced[i]), None)
    if first is None:
        return []                       # nothing sings anywhere; there is no song to protect
    runs, current = [], []
    for i, section in enumerate(score.sections):
        if voiced[i]:
            if current:
                runs.append(current)
                current = []
            continue
        current.append((i, section))
    if current:
        runs.append(current)
    out = []
    for run in runs:
        head = run[0][0]
        kind = INTRO if head < first else OUTRO if head > last else BETWEEN
        out.append((kind, [section for _, section in run]))
    return out


def _keep(score, sections, seconds, from_end):
    """Groups to DROP so the run lasts about `seconds`.

    The cut lands on group boundaries, because a group is the smallest span both voices share — so a
    plan written in four-bar groups can only be trimmed four bars at a time, and a limit is met from
    below rather than exactly.

    One exception to "at most", and it is there because the literal reading surprises people: a
    stretch made of a single six-second block, asked for five, would vanish entirely. So a limit
    above zero always keeps one group. Zero still means remove it, which is the way to say that.
    """
    groups = [g for section in sections for g in section.groups if g.vocal is not None]
    order = list(reversed(groups)) if from_end else groups
    budget, drop = seconds, []
    for group in order:
        cost = N.bars(score.lines[group.vocal]) * score.bar_seconds
        if cost <= budget + 1e-9:
            budget -= cost
        else:
            drop.append(group)
    if seconds > 0 and order and len(drop) == len(order):
        drop.remove(order[0])
    return drop


def trim(score, intro=12.0, outro=12.0, between=8.0):
    """Cut every silent stretch to its limit. → (new score, [row, ...]).

    A limit of 0 removes the stretch outright; one longer than the stretch leaves it alone. Each row
    is (kind, labels, bars before, bars after) for the report.
    """
    limits = {INTRO: intro, OUTRO: outro, BETWEEN: between}
    doomed, rows = [], []
    for kind, sections in stretches(score):
        limit = max(0.0, float(limits.get(kind, 0.0)))
        drop = _keep(score, sections, limit, from_end=(kind == INTRO))
        had = sum(N.bars(score.lines[g.vocal]) for s in sections for g in s.groups
                  if g.vocal is not None)
        went = sum(N.bars(score.lines[g.vocal]) for g in drop)
        rows.append((kind, " + ".join(s.label or "?" for s in sections), had, had - went))
        doomed.extend(drop)
    return (N.drop(score, doomed) if doomed else score), rows


def sung_notes(score):
    """Every note attack in the Vocal voice. A trim must not change this number."""
    return sum(N.attacks(score.lines[g.vocal], score.key)
               for s in score.sections for g in s.groups if g.vocal is not None)


def _mmss(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def report(before, after, rows):
    """What was cut, and the proof that nothing sung went with it."""
    out = [f"{before.bar_count()} bars {_mmss(before.duration())}  ->  "
           f"{after.bar_count()} bars {_mmss(after.duration())}"
           f"   ({_mmss(before.duration() - after.duration())} shorter)"]
    kept, lost = sung_notes(before), sung_notes(after)
    out.append(f"sung notes {kept} -> {lost}" + ("   (untouched)" if kept == lost else
                                                 "   !! SUNG NOTES WERE LOST"))
    if rows:
        out += ["", f"{'stretch':<10}{'bars':>6}{'->':>5}{'seconds':>9}  sections"]
        for kind, labels, had, left in rows:
            out.append(f"{kind:<10}{had:>6}{left:>5}{left * before.bar_seconds:>9.1f}  {labels}")
    else:
        out.append("")
        out.append("nothing to cut — no stretch of this plan is silent in the Vocal voice")
    problems = N.problems(after)
    if problems:
        out += [""] + ["! " + p for p in problems]
    return "\n".join(out)
