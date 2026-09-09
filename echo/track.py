"""What gets aligned, in what window — and what the times mean once they come back.

This is the half of Echo that decides the *problem*; `align.py` only solves it. Keeping them apart
is not tidiness: the windowing is what stops a three-minute song from costing three minutes of
attention, and it is arithmetic, so it can be tested without weights.

**The unit is a plan row, not a line and not a section of the lyric.** A Siren plan row
(`Verse 1 | Nina | 16 bars`) is the smallest thing that has both a *place in the song* and a *set of
words*, and `Siren Score` may have split one row's lyric into several sub-sections so two voices
alternate inside it. Those sub-sections are consecutive audio, so they align together and only
differ in which voice a line belongs to. Hence: a **block** = one plan row = one aligner call.

**Why the window matters more than anything else here.** `wav2vec2` is a transformer over 20 ms
frames, so its attention is quadratic in the length of what it is handed. A whole three-minute song
is ~9 000 frames and that is where the memory goes — the same wall the author hit with a video
turned into a frame batch, arrived at from the other side. One block is 15-40 s, i.e. 750-2 000
frames, and the cost stops depending on the song's length at all. The plan is what makes this
possible: it already says roughly where every section is.

**Roughly is the operative word, and it is the author's own complaint.** The plan says where a
section was *asked* to be; the model sang it somewhere else. So every window is the plan's span plus
`slack` on each side, and what comes back is allowed to disagree with the plan — that disagreement
is the *output*, not an error. A block's corrected span is where its words actually are.

**Shown and aligned are different strings** — see `translit.py`. This module carries the pair all
the way through, and reassembles the display side from the original line so that every comma and
dash the author typed survives into the subtitle.

Pure stdlib plus one import from `siren`, which is where a plan row is defined. Nothing from
`save_video`, deliberately: `Save Clip` will import Echo, and two packages that import each other
are a circular import waiting for the one day somebody moves a symbol.
"""
import re

from . import translit as R

#: A window is the plan's span plus this fraction of the span on each side, never less than
#: `SLACK_MIN`. Proportional because the plan is scaled onto the audio, so its error is
#: proportional too: a 4-second intro cannot be 20 seconds late, and a 40-second verse can.
SLACK = 0.15
SLACK_MIN = 3.0

#: With no plan at all there is nothing to window on, so the lyric is spread over the song by
#: syllable count and every window is opened this wide. It works, and it is several times the cost.
SLACK_NO_PLAN = 0.6

#: Where `siren.cast` reads bars from, at a tempo that cancels out — the same convention
#: `save_video/timeline.py` uses, and for the same reason: only the ratios are ever used.
NOMINAL_BPM = 120.0
NOMINAL_BEATS = 4

#: The wire type the timing travels on. Named so `Save Clip` and `Orpheus` can declare an input of
#: it without importing anything that needs torch.
TIMING_TYPE = "KINBURG_ECHO_TIMING"

_DURATION_LIST = re.compile(r"^[\s\d.,]+$")


def plan_rows(plan):
    """A plan block -> `[{label, voice, weight}]`, plus notes.

    Two shapes, the same two `Save Clip` takes: a Siren table (which carries the **voice**, and that
    is the column this whole feature's colouring rests on) or Orpheus' bare comma list of seconds
    (which does not). `weight` is nominal and only ever used as a ratio.
    """
    text = str(plan or "").strip()
    if not text:
        return [], []

    if "|" not in text and _DURATION_LIST.match(text):
        rows, notes = [], []
        for cell in (c.strip() for c in text.replace("\n", ",").split(",")):
            if not cell:
                continue
            try:
                secs = float(cell)
            except ValueError:
                notes.append(f"'{cell}' is not a length and was skipped")
                continue
            if secs > 0:
                rows.append({"label": f"shot {len(rows) + 1}", "voice": "", "weight": secs})
        if rows:
            notes.append("the plan is a list of lengths with no labels and no voices, so the lyric "
                         "is matched to it by ORDER and every line gets the same colour. Wire "
                         "Siren Score's plan instead to get either back.")
        return rows, notes

    from ..siren.cast import _parse_plan          # local: keeps this module import-light
    parsed, notes = _parse_plan(text, NOMINAL_BPM, NOMINAL_BEATS)
    return ([{"label": r["label"], "voice": r["voice_raw"], "weight": r["seconds"]} for r in parsed],
            list(notes))


def spans(rows, total):
    """Plan rows -> `[{label, voice, start, end}]` on the song's own clock.

    The plan is read for its proportions and scaled so the last row ends on the last sample — the
    rule `save_video/timeline.py` states at length, and for the same reason: bars are only seconds
    once you know a tempo, and the tempo lives on a different node.
    """
    weights = [max(0.0, float(r.get("weight") or 0.0)) for r in rows or []]
    planned = sum(weights)
    if not weights or planned <= 0:
        return []
    scale = float(total) / planned
    out, at = [], 0.0
    for r, w in zip(rows, weights):
        out.append({"label": r.get("label") or "", "voice": r.get("voice") or "",
                    "start": at, "end": at + w * scale})
        at += w * scale
    out[-1]["end"] = float(total)
    return out


def display_chunks(line, words):
    """The pieces a karaoke line is drawn in, one per word, which **concatenate back to `line`**.

    Every space, comma and em-dash between two words has to belong to one of them, or the lyric on
    screen is a word list rather than a line. Each word takes what follows it up to the next word;
    the first also takes whatever came before it, which is where an opening dash or quote lives.
    """
    line = str(line or "")
    if not words:
        return []
    edges = [0] + [int(w["at"]) for w in words[1:]] + [len(line)]
    return [line[edges[i]:edges[i + 1]] for i in range(len(words))]


def _voice_of(section, row_voice, voices=()):
    """Who sings this section, in the order the four sources deserve to be trusted.

    A sub-section carries its singer explicitly (`[Gru]` on its own line) — that is an instruction
    and wins. Otherwise the section's own header is asked, against the **roster**: `[Chorus - Nina]`
    names Nina and the plan row for that chorus may well say `Nina + Gru`, which is true of the row
    and not of this line. Falling back to the row's cell before reading the header is how a solo
    verse inside a duet chorus ends up coloured as the duet.

    Only then the plan row, and only with no roster wired the header's raw text — which may be a
    description rather than a name (`[Chorus - powerful FEMALE belt]`), and is still better than
    nothing, because two sections described differently really are two different voices.
    """
    from ..siren.score import _names_in

    names = [n for n in (section.get("names") or []) if str(n).strip()]
    if names:
        return str(names[0]).strip()
    if voices:
        named = _names_in(section.get("marker"), voices)
        if named:
            return str(named[0]).strip()
    row = str(row_voice or "").strip()
    if row and row not in ("-", "—", "–"):
        return row
    return str(section.get("marker") or "").strip()


def _label_key(text):
    return " ".join(str(text or "").split()).lower()


def build(lyrics, plan, total, lang=R.LANG_AUTO, slack=SLACK, voices=()):
    """Everything the aligner needs: `(job, notes)`.

    `job["blocks"]` is one entry per plan row, in order, each with the window to search and the
    lines to find in it. A row with no lyric is kept as an instrumental block — dropping it would
    lose the fact that the song has a solo there, which is exactly what a corrected timeline is for.
    """
    from ..siren.score import _split_sections     # local: siren pulls in the whole cast machinery

    notes = []
    total = float(total)
    rows, row_notes = plan_rows(plan)
    notes.extend(row_notes)

    secs, split_notes = _split_sections(lyrics, voices)
    notes.extend(split_notes)

    # The lyric's language, decided once over everything, so an undecidable word (`вода`, `гора`)
    # is spelled the way the rest of the song is rather than by a coin toss.
    whole = " ".join(" ".join(s.get("text") or []) for s in secs)
    fallback = R.guess_language(whole) or "uk"
    if lang != R.LANG_AUTO:
        fallback = lang

    # Consecutive lyric sections sharing a label are ONE plan row split for alternating voices —
    # `Siren Score` makes those, and they are consecutive audio, so they align together. Grouping is
    # by ADJACENCY, never by label across the song: a chorus sung three times is three groups, and
    # collecting them by name put all three choruses' words into the first chorus's window — the
    # words then showed up minutes early, on top of somebody else's line, and the last chorus got
    # nothing at all. `sub` is what marks a continuation, so it is what the grouping asks.
    groups = []
    for s in secs:
        same = groups and _label_key(groups[-1][0].get("label")) == _label_key(s.get("label"))
        if same and s.get("sub"):
            groups[-1].append(s)
        else:
            groups.append([s])

    if not rows:
        # No plan: spread the lyric over the song by syllables. Every window is wide open, which
        # works and costs several times as much — say so rather than let it look normal.
        notes.append("no plan was wired, so the sections were spread over the song by syllable "
                     "count and every search window was opened wide. It works; it is slower and "
                     "less certain than wiring Siren Score's plan, which already knows where the "
                     "sections are.")
        rows = [{"label": g[0].get("label") or f"section {i + 1}", "voice": "",
                 "weight": max(1, sum(s.get("syl") or 0 for s in g))}
                for i, g in enumerate(groups)]
        slack = SLACK_NO_PLAN

    placed = spans(rows, total)
    # Matched in ORDER, one occurrence consumed at a time — the same rule `Save Clip`'s .srt uses,
    # and the only one that survives a repeated label. A row that matches nothing is instrumental.
    pending = list(groups)
    blocks = []
    for span in placed:
        key = _label_key(span["label"])
        mine = None
        for i, g in enumerate(pending):
            if _label_key(g[0].get("label")) == key:
                mine = pending.pop(i)
                break
        width = max(0.0, span["end"] - span["start"])
        pad = max(float(SLACK_MIN), width * max(0.0, float(slack)))
        block = {
            "label": span["label"], "voice": span["voice"],
            "start": span["start"], "end": span["end"],
            "window": (max(0.0, span["start"] - pad), min(total, span["end"] + pad)),
            "lines": [],
        }
        for section in mine or []:
            voice = _voice_of(section, span["voice"], voices)
            for text in section.get("text") or []:
                ws = R.words(text, lang, fallback)
                if not ws:
                    continue
                chunks = display_chunks(text, ws)
                block["lines"].append({
                    "text": text, "voice": voice, "label": span["label"],
                    "words": [{"text": c, "key": w["key"]} for c, w in zip(chunks, ws)],
                })
        blocks.append(block)

    for g in pending:
        notes.append(f"the lyric has a '{g[0].get('label')}' section that no plan row is called — "
                     f"its words were not aligned. The markers in the lyrics and the labels in the "
                     f"plan have to be the same words; wire both from Siren.")

    # A label with MORE sung sections than the plan has rows for it is the failure that used to be
    # silent: the surplus ones can only be matched to a row that belongs to a different part of the
    # song. Counted over sung groups only, and only in that direction — a plan with five `Break`
    # rows against one `[Break]` marker in the lyrics is an ordinary instrumental arrangement, and
    # warning about it every run taught nothing except to stop reading the warnings.
    from collections import Counter
    want = Counter(_label_key(r["label"]) for r in rows)
    sung_groups = [g for g in groups if any((s.get("text") or []) for s in g)]
    have = Counter(_label_key(g[0].get("label")) for g in sung_groups)
    for key in sorted(have):
        if have[key] > want[key]:
            notes.append(f"'{key}' is sung {have[key]} time(s) in the lyrics but the plan has only "
                         f"{want[key]} row(s) for it. Repeated sections are matched in order, so "
                         f"the surplus one is timed against the wrong part of the song.")

    sung = sum(1 for b in blocks if b["lines"])
    if not sung:
        notes.append("not one plan row matched a section of the lyric, so there is nothing to "
                     "align. Check that both came from the same Siren graph.")

    spell = R.spellable(whole, lang, fallback)
    if spell < 0.9:
        notes.append(f"only {spell * 100:.0f}% of the words can be written in the aligner's "
                     f"alphabet — the rest (digits, symbols) will be carried on their neighbours' "
                     f"timing rather than found. Spell numbers out to fix it.")

    return {"total": total, "language": fallback, "blocks": blocks,
            "spellable": spell, "sung_blocks": sung}, notes


def _carry(words, lo, hi):
    """Give a time to every word the aligner could not be asked about.

    A word with no romanized form (a digit, an emoji, a line of dots) is never sent to the model, so
    it comes back with nothing — but it is still on screen and still sung. It takes the space
    between its neighbours; a run of them shares that space evenly. Losing this is how a subtitle
    ends up with a word that never lights up.
    """
    n = len(words)
    i = 0
    while i < n:
        if words[i].get("start") is not None:
            i += 1
            continue
        j = i
        while j < n and words[j].get("start") is None:
            j += 1
        left = words[i - 1]["end"] if i > 0 and words[i - 1].get("end") is not None else lo
        right = words[j]["start"] if j < n and words[j].get("start") is not None else hi
        if right <= left:
            right = left
        step = (right - left) / max(1, j - i)
        for k in range(i, j):
            words[k]["start"] = left + step * (k - i)
            words[k]["end"] = left + step * (k - i + 1)
            words[k]["score"] = 0.0
            words[k]["carried"] = True
        i = j


def finish(job, min_word=0.06):
    """The aligned job -> the track everything downstream reads.

    Called after `align.py` has filled `start`/`end`/`score` on the words it could find. What is
    left to do is the bookkeeping that keeps a subtitle file legal: carry the words that were never
    sent, stop a line from running backwards, and give every word a width a player can draw.

    The track:
        `{total, language, lines: [{start, end, text, voice, label, score, words: [...]}],
          sections: [{label, voice, start, end, sung}], quiet: [(start, end)]}`
    """
    total = float(job.get("total") or 0.0)
    lines, sections, stranded = [], [], []

    for index, block in enumerate(job.get("blocks") or []):
        lo, hi = block.get("window", (block["start"], block["end"]))
        first, last = None, None
        for ln in block["lines"]:
            words = ln["words"]
            _carry(words, lo, hi)
            # Monotonic and non-degenerate. A player given end <= start draws nothing at all, and a
            # karaoke run of zero length makes the sweep jump a word.
            at = None
            for w in words:
                w["start"] = max(0.0, min(total, float(w.get("start") or 0.0)))
                w["end"] = max(w["start"] + float(min_word), float(w.get("end") or 0.0))
                if at is not None and w["start"] < at:
                    w["start"] = at
                    w["end"] = max(w["end"], at + float(min_word))
                at = w["end"]
            # After the monotonic pass, so a trimmed tail cannot leave the line running backwards.
            cut = _trim_tail(words)
            if cut > 0.05:
                stranded.append((ln["label"], cut))
            found = [w for w in words if not w.get("carried")]
            lines.append({
                "start": words[0]["start"], "end": words[-1]["end"],
                "text": ln["text"], "voice": ln["voice"], "label": ln["label"],
                # The section's INDEX, not only its name. A chorus sung three times is three
                # sections with one label, so anything that groups lines by label alone silently
                # averages all three together.
                "section": index,
                "score": (sum(w.get("score") or 0.0 for w in found) / len(found)) if found else 0.0,
                "found": len(found), "words": words,
            })
            first = lines[-1]["start"] if first is None else min(first, lines[-1]["start"])
            last = lines[-1]["end"] if last is None else max(last, lines[-1]["end"])

        sections.append({
            "index": index,
            "label": block["label"], "voice": block["voice"], "sung": bool(block["lines"]),
            # A sung block's real span is where its words are; an instrumental one has nothing to
            # measure, so it keeps the plan's guess and is marked as a guess by `sung`.
            "start": first if first is not None else block["start"],
            "end": last if last is not None else block["end"],
            "planned": (block["start"], block["end"]),
        })

    _place_instrumentals(sections, total)
    lines.sort(key=lambda ln: (ln["start"], ln["end"]))
    scored = [ln for ln in lines if ln["found"]]
    return {
        "total": total, "language": job.get("language") or "",
        "lines": lines, "sections": sections,
        "score": (sum(ln["score"] for ln in scored) / len(scored)) if scored else 0.0,
        "spellable": job.get("spellable", 1.0),
        "quiet": _quiet(lines, total),
        "collisions": collisions(lines),
        "stranded": stranded,
    }


#: How far a section may begin before the previous one's last word ends. A backing echo overlaps its
#: lead by design and a singer breathes for less than half a second; more than this and the two are
#: not two sections.
ANCHOR_OVERLAP = 0.4

#: Below this confidence a block is not trusted to anchor the next one. Without the gate, one bad
#: placement pushes every section after it and a single failure becomes a cascade — which is a worse
#: failure than the one anchoring exists to prevent.
ANCHOR_TRUST = 0.4


def anchor(block, prev_end=None, drift=0.0, slack=SLACK, total=None):
    """The window to search one block in, given where the previous one's words really ended.

    Two things are being carried, and they answer different halves of the same problem:

    * **`drift`** — how far the last trusted section turned out to be from where the plan put it.
      The author's own take drifted +1.5 s at the first verse and +17.7 s by the second, growing
      monotonically: a plan is not wrong at random, it is wrong *progressively*, so the next
      section's best guess is its planned position plus what the last one taught us.
    * **`prev_end`** — a hard floor. A section cannot begin before the previous one stopped singing,
      and this is the half that matters most. It is what turns "the Outro was never sung, so its
      words were laid confidently on top of the chorus" into "the Outro was searched in the outro,
      found nothing, and said so" — a wrong answer becoming a reported non-answer.

    **The drift is applied asymmetrically, and that is not a detail.** A first version shifted the
    whole window by it and came out *worse*: on the author's take the drift grew to +17.7 s at the
    second verse and then SHRANK to +13.8 and +8.4, so a window whose start had been pushed forward
    by the last measurement began after the section it was looking for. Drift is therefore only ever
    allowed to make a window BIGGER — forward on the ceiling, and backwards on the floor only when
    it is negative. The costs are not symmetric: a window that starts too early is slower, a window
    that starts too late is wrong.

    Tightening the back is the floor's job instead, and the floor is a fact rather than an estimate.

    The ceiling also keeps room for a block's own words to run longer than planned — on that take
    every section did — so a window can never come out too small to hold what is being looked for.
    """
    start, end = float(block["start"]), float(block["end"])
    width = max(0.0, end - start)
    pad = max(float(SLACK_MIN), width * max(0.0, float(slack)))
    span = float(total) if total is not None else end

    lo = start + min(0.0, float(drift)) - pad
    if prev_end is not None:
        lo = max(lo, float(prev_end) - ANCHOR_OVERLAP)
    lo = max(0.0, lo)
    hi = max(end + float(drift) + pad, lo + width * 1.5 + pad)
    return lo, min(span, max(hi, lo))


def lines_in(track, start, end, share=0.2, floor=0.2):
    """The lines sung during `[start, end)`, in time order.

    A line counts when it overlaps the span by a fifth of itself, or by `floor` seconds, whichever
    is smaller — so a line that merely clips a boundary is listed for **both** neighbours rather
    than being awarded to one of them. That is right for context and would be wrong for subtitles:
    nothing here decides what is on screen, only what a shot is *about*.
    """
    out = []
    for ln in track.get("lines") or []:
        span = min(ln["end"], float(end)) - max(ln["start"], float(start))
        if span > 0 and span >= min(float(floor), float(share) * (ln["end"] - ln["start"])):
            out.append(ln)
    return sorted(out, key=lambda ln: ln["start"])


def shot_lyrics(track, spans, instrumental="(instrumental - no words)", quote='"'):
    """`[(start, end)]` -> one line of text per span, **numbered by shot and never timed**.

    The block a storyboard planner is given so that shot 7 knows what is being sung over it. Two
    rules are baked in rather than left to the caller, because both are ways this goes wrong:

    * **No seconds, anywhere.** Phantas' planner emits *weights* and the grid is applied afterwards
      — a model that has been shown timestamps starts reasoning in them and its lengths silently
      stop landing on H3's frame grid. Shots are named by their number.
    * **The singer is named on every line.** A duet where both voices are quoted anonymously reads
      as one person talking to themselves, and the cast block is the mechanism that holds identity
      across frames; the two have to agree about who is present.
    """
    rows = []
    for i, (start, end) in enumerate(spans, 1):
        here = lines_in(track, start, end)
        if not here:
            rows.append(f"{i}. {instrumental}")
            continue
        parts = []
        for ln in here:
            text = " ".join(str(ln.get("text") or "").split())
            if not text:
                continue
            voice = " ".join(str(ln.get("voice") or "").split())
            parts.append(f"{voice} - {quote}{text}{quote}" if voice else f"{quote}{text}{quote}")
        rows.append(f"{i}. " + ("; ".join(parts) if parts else instrumental))
    return rows


def collisions(lines, tolerance=0.15):
    """Pairs of lines from DIFFERENT sections that are on screen at the same time.

    This is the shape of a misplacement, and it is the one thing a confidence score does not show.
    A section searched in a window that does not contain it gets crammed against the window's edge —
    and lands on top of whichever section really is singing there. Two voices' lines then appear
    together, in two colours, while the real performance of one of them passes with no subtitle at
    all. Overlapping lines *within* one section are ordinary (a backing echo under a lead), so only
    cross-section overlaps are counted.
    """
    out = []
    ordered = sorted(lines, key=lambda ln: ln["start"])
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b["start"] >= a["end"] - float(tolerance):
                break
            if a.get("section") != b.get("section"):
                out.append({"a": a["label"], "b": b["label"], "at": b["start"],
                            "seconds": min(a["end"], b["end"]) - b["start"]})
    return out


#: A line's last word may run this many times longer than the rest of the line's own pace predicts
#: before it is treated as having absorbed the silence after it rather than having been held.
TAIL_FACTOR = 3.0

#: What it is trimmed back to, as a multiple of that predicted length. Not to the prediction itself:
#: a singer really does hold a final note longer than the words before it, and cutting to the bare
#: prediction would clip every line that ends on one.
TAIL_KEEP = 1.5

#: Never trim a last word below this. A three-word line gives a noisy rate, and a subtitle that
#: flashes is worse than one that lingers.
TAIL_FLOOR = 1.0


def _trim_tail(words, factor=TAIL_FACTOR, keep=TAIL_KEEP, floor=TAIL_FLOOR):
    """Pull back a final word that ran on into the silence. Returns the seconds removed.

    **Why this is needed at all.** A block is aligned with a `*` star token at each end so the model
    can account for audio the transcript does not cover. Entering that star costs something, and at
    the end of a section it is often cheaper for the Viterbi path to hold the last word's final
    character across the instrumental than to pay for the star — so the last word swallows the gap.
    On the author's own render the closing line of a verse was sung by 0:34.5 and stayed on screen
    until 0:39.9, over the top of the next section's first line.

    The bound comes from the line itself rather than from a constant: the words before it say how
    fast this line is being sung, and the last word is allowed a generous multiple of that. A line
    that really does end on a held note keeps it; one whose last word is three times longer than the
    line's own pace was not held, it was stranded.
    """
    if len(words) < 2:
        return 0.0
    last, body = words[-1], words[:-1]
    chars = sum(len(str(w.get("text") or "").strip()) for w in body)
    spent = sum(max(0.0, w["end"] - w["start"]) for w in body)
    if chars <= 0 or spent <= 0:
        return 0.0

    want = max(float(floor), (spent / chars) * max(1, len(str(last.get("text") or "").strip())))
    have = last["end"] - last["start"]
    if have <= want * float(factor):
        return 0.0
    trimmed = have - max(float(floor), want * float(keep))
    last["end"] -= trimmed
    return trimmed


def _place_instrumentals(sections, total):
    """Give every instrumental section the gap between the singing on either side of it.

    An instrumental block has no words, so nothing measured it — it kept the plan's own absolute
    position. Once the sung sections around it have moved by twenty-five seconds that position is
    simply wrong, and it is also the easiest thing here to get right: an instrumental section is
    **whatever is between two sung ones**, and by now both of those are known.

    Left undone this is not a cosmetic error. It filled the corrected plan's overlap warning with
    the names of five `Break`s and `Solo`s that had not moved, so the one genuine overlap in that
    song was buried among them — a warning that cries wolf is a warning nobody reads. And the plan
    is an OUTPUT: those rows go on to place pictures in `Save Clip` and cuts in `Orpheus`.

    A run of several instrumental rows shares its gap in proportion to their planned lengths, which
    is the only information anyone has about them.
    """
    n = len(sections)
    i = 0
    while i < n:
        if sections[i].get("sung"):
            i += 1
            continue
        j = i
        while j < n and not sections[j].get("sung"):
            j += 1
        lo = sections[i - 1]["end"] if i > 0 else 0.0
        hi = max(lo, sections[j]["start"] if j < n else float(total))
        widths = [max(1e-6, s["planned"][1] - s["planned"][0]) for s in sections[i:j]]
        share = (hi - lo) / sum(widths)
        at = lo
        for s, w in zip(sections[i:j], widths):
            s["start"], s["end"] = at, at + w * share
            at = s["end"]
        sections[j - 1]["end"] = hi          # kill the rounding dust on the last of the run
        i = j
    return sections


def _quiet(lines, total, gap=4.0):
    """The stretches with no words in them, longer than `gap`.

    Not decoration: this is where a music video's instrumental shots go, and it is a thing only the
    alignment knows — the plan's idea of an instrumental section is where one was *asked* for.
    """
    out, at = [], 0.0
    for ln in sorted(lines, key=lambda x: x["start"]):
        if ln["start"] - at >= gap:
            out.append((at, ln["start"]))
        at = max(at, ln["end"])
    if total - at >= gap:
        out.append((at, total))
    return out
