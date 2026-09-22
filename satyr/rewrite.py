"""Making a phrase hold exactly the syllables that belong to it.

The plan's note count is the lyric's syllable count — that is not a metaphor. A measured song put 47
notes under a chorus of 47 syllables, and its one bracketed line, four syllables long, got a phrase
of exactly four notes to itself. Where the two disagree the words slide: a chorus handed 54 notes for
49 syllables pushes every line a little further ahead of the bar it was written for until a voice
change, which can only happen at a phrase boundary, lands in the middle of a word. That is audible,
and it is arithmetic, so it can be removed by arithmetic.

**Only the disagreeing measures are rewritten.** A phrase that already holds the right number of
syllables is returned untouched, and inside one that does not, every measure the surgery did not
need to open comes back as its own original characters. The point is the same as everywhere else in
this suite: a diff of a rewrite should show the notes that changed and nothing else.

**Duration is never created or destroyed.** Both operations work inside one measure and conserve its
total, because a measure that no longer adds up to its meter is not a bar the format can hold — and
the failure would not be local, it would shift every following bar of the song.

* To gain a syllable, split the longest note into two halves of the same pitch. A long held note
  becoming two shorter ones is the smallest musical change that adds an onset, and repeating a pitch
  is exactly what a second syllable on one note wants.
* To lose one, merge the shortest adjacent pair into a single note carrying the first pitch. The
  first is kept because that is where the syllable starts.

Both are refused rather than approximated when the arithmetic will not close: a duration that is not
one the format admits (`notation.SUPPORTED`) is not written, so a pair summing to seven units simply
does not merge and the search moves on to another pair. This is why `refit` reports what it reached
instead of promising what it was asked for — a phrase of four sixteenths in a 2/4 bar has nowhere to
put a fifth syllable, and saying so is more useful than silently writing a bar that does not scan.

**Merging loses a pitch, and that is a real cost.** Shortening a phrase flattens its melody a little,
every time. Lengthening does not — a split repeats a note it already had. So a caller with a choice
should prefer to give a long line to a long phrase rather than shorten the phrase, and the report
says how many merges it had to do so that choice can be made with the number in hand.
"""
from . import notation as N


def _tokens(measure):
    """One measure → its elements in order, as dicts the renderer understands.

    Anything that is not a note or a rest — a chord symbol, an inline key change — is carried through
    as opaque text, so the surgery cannot lose harmony it does not understand.
    """
    out, at = [], 0
    for m in N.ELEMENT.finditer(measure):
        if m.start() > at:
            out.append({"kind": "raw", "text": measure[at:m.start()]})
        if m.group("chord") is not None or m.group("key") is not None:
            out.append({"kind": "raw", "text": m.group(0)})
        else:
            out.append({"kind": "rest" if m.group("letter") == "z" else "note",
                        "acc": m.group("acc"), "letter": m.group("letter"),
                        "octave": m.group("octave"), "units": int(m.group("units") or 1),
                        "tie": bool(m.group("tie"))})
        at = m.end()
    if at < len(measure):
        out.append({"kind": "raw", "text": measure[at:]})
    return out


def _render(tokens):
    parts = []
    for t in tokens:
        if t["kind"] == "raw":
            parts.append(t["text"])
        else:
            parts.append(t["acc"] + t["letter"] + t["octave"]
                         + (str(t["units"]) if t["units"] != 1 else "") + ("-" if t["tie"] else ""))
    return "".join(parts)


def _onsets(tokens, tied_in=False):
    """(indices a syllable starts on, whether the measure ties into the next one).

    `tied_in` is not optional decoration. A tie is the one thing in this notation that reaches across
    a barline: `…A-|A2…` is one held syllable, and a measure scanned on its own counts that second A
    as a fresh onset. Getting it wrong does not produce a wrong-looking line — it makes a phrase
    appear to hold more syllables than it does, so `refit` sets about merging notes out of a phrase
    that was already correct. That is exactly what it did before this argument existed.
    """
    out = []
    tied = tied_in
    for i, t in enumerate(tokens):
        if t["kind"] == "rest":
            tied = False
        elif t["kind"] == "note":
            if not tied:
                out.append(i)
            tied = t["tie"]
    return out, tied


def _scan(grids):
    """Per-measure onset indices for a whole line, threading ties across the barlines."""
    out, tied = [], False
    for g in grids:
        if g is None:
            out.append([])
            tied = False
            continue
        indices, tied = _onsets(g, tied)
        out.append(indices)
    return out


def _tie_state(grids):
    """Whether each measure is tied into from the one before it."""
    out, tied = [], False
    for g in grids:
        out.append(tied)
        if g is None:
            tied = False
            continue
        _, tied = _onsets(g, tied)
    return out


def _split_one(tokens):
    """Split the longest splittable note in place. True if anything changed.

    The halves have to be durations the format admits, which rules out splitting a single unit and
    makes an odd length split unevenly (3 becomes 2 and 1) rather than not at all.
    """
    best, best_units = None, 0
    for i, t in enumerate(tokens):
        if t["kind"] != "note" or t["units"] < 2:
            continue
        first, second = (t["units"] + 1) // 2, t["units"] // 2
        if first not in N.SUPPORTED or second not in N.SUPPORTED:
            continue
        if t["units"] > best_units:
            best, best_units = i, t["units"]
    if best is None:
        return False
    t = tokens[best]
    first, second = (t["units"] + 1) // 2, t["units"] // 2
    # The tie belongs to whatever now ends the note; the new head is a plain onset.
    head = dict(t, units=first, tie=False)
    tail = dict(t, units=second)
    tokens[best:best + 1] = [head, tail]
    return True


def _pairs(tokens, tied_in=False):
    """Adjacent separately-struck notes whose lengths add to a duration the format admits.

    Returns [(index of the first, combined units)]. A pair with something between its two notes is
    skipped: that something is a rest or a chord change, and closing over it would move music.
    """
    out = []
    onsets, _ = _onsets(tokens, tied_in)
    for a, b in zip(onsets, onsets[1:]):
        if b != a + 1:
            continue
        total = tokens[a]["units"] + tokens[b]["units"]
        if total in N.SUPPORTED:
            out.append((a, total))
    return out


def _merged(tokens, at, total):
    merged = list(tokens)
    keep, drop = merged[at], merged[at + 1]
    merged[at:at + 2] = [dict(keep, units=total, tie=drop["tie"])]
    return merged


def _merge_one(tokens, tied_in=False):
    """Merge one adjacent pair of separately-struck notes. True if anything changed.

    Which pair is not obvious, and taking the shortest — the first thing this did — dead-ends. The
    format has no note of five units or seven, so a bar of 2,1,2,3 that merges its first two lands on
    3,2,3, where both remaining pairs sum to five and nothing more can go. Merging the *second* pair
    instead gives 2,3,3, which collapses the whole way to a single eight. Same bar, same rule,
    different answer, purely because of the order.

    So the pair is chosen by how many legal pairs it leaves behind, and only then by being short. One
    step of lookahead is enough for the lengths this format allows, and it is cheap: a measure holds
    a handful of notes, not a search space.
    """
    options = _pairs(tokens, tied_in)
    if not options:
        return False
    at, total = max(options, key=lambda p: (len(_pairs(_merged(tokens, *p), tied_in)), -p[1]))
    tokens[:] = _merged(tokens, at, total)
    return True


def _drop_one(tokens, tied_in=False):
    """Silence the last onset of the measure that can go. True if anything changed.

    The last resort, and it is reported separately because it is the one operation that removes a
    sound rather than reshaping one. It exists because merging cannot reach everything: a phrase
    built of long notes tied across the barlines — `d8-|d6f2|e8-|e4z2g2` — has exactly one onset in
    each measure and therefore no adjacent pair anywhere, so a merge-only rewrite cannot shorten it
    by even a single syllable. Turning a note into a rest can, and it is what an arranger does with a
    line that runs out of words: the phrase simply stops sooner.

    Taken from the end, because a syllable dropped off the tail of a phrase is the one a listener
    misses least. A note tying into the next one is left alone — silencing it would only hand the
    onset to its neighbour and change nothing.
    """
    onsets, _ = _onsets(tokens, tied_in)
    for i in reversed(onsets):
        if tokens[i]["tie"]:
            continue
        tokens[i] = {"kind": "rest", "acc": "", "letter": "z", "octave": "",
                     "units": tokens[i]["units"], "tie": False}
        # Two rests in a row are legal but noisy; fold them when the format allows the total.
        for a in (i - 1, i):
            if 0 <= a and a + 1 < len(tokens) and tokens[a]["kind"] == tokens[a + 1]["kind"] == "rest":
                total = tokens[a]["units"] + tokens[a + 1]["units"]
                if total in N.SUPPORTED:
                    tokens[a:a + 2] = [dict(tokens[a], units=total)]
                    break
        return True
    return False


def refit(line, want):
    """Rewrite one music line to carry `want` syllables. Returns (line, reached, cost).

    `reached` is what the line actually holds afterwards, which is the answer when the target was
    impossible — a phrase of one sung bar cannot be given nine syllables, and saying so beats writing
    a bar that does not scan. `cost` counts the three operations separately, because they do not cost
    the same: a `split` repeats a note the phrase already had and takes nothing away, a `merged`
    gives up one pitch, and a `dropped` gives up a note entirely.
    """
    cost = {"split": 0, "merged": 0, "dropped": 0}
    measures = N.split_measures(line)
    if not measures or want < 0:
        return line, N.attacks(line), cost
    grids = [None if N.FULL_REST.match(m.strip()) else _tokens(m) for m in measures]
    have = sum(len(c) for c in _scan(grids))
    if have == want or all(g is None for g in grids):
        return line, have, cost

    # Work on whichever measure is currently densest (to thin) or sparsest (to fill), so the change
    # is spread over the phrase instead of mangling one bar.
    while have != want:
        counts = _scan(grids)
        live = [i for i, g in enumerate(grids) if g is not None]
        if not live:
            break
        filling = have < want
        live.sort(key=lambda i: len(counts[i]), reverse=not filling)
        if filling:
            moved = next((True for i in live if _split_one(grids[i])), False)
            cost["split"] += 1 if moved else 0
        else:
            # A measure's first note may be the far end of a tie, so whether it can start a merge
            # depends on what the measure before it did.
            ties = _tie_state(grids)
            moved = next((True for i in live if _merge_one(grids[i], ties[i])), False)
            if moved:
                cost["merged"] += 1
            else:
                ties = _tie_state(grids)
                moved = next((True for i in live if _drop_one(grids[i], ties[i])), False)
                cost["dropped"] += 1 if moved else 0
        if not moved:
            break
        have += 1 if filling else -1

    rebuilt = [m if g is None else _render(g) for m, g in zip(measures, grids)]
    # An untouched measure keeps its own characters; only the rewritten ones are regenerated.
    out = "|".join(orig if _render(_tokens(orig)) == new else new
                   for orig, new in zip(measures, rebuilt)) + "|"
    return out, have, cost


def capacity(line):
    """The most syllables this phrase can hold: its sung units, each shrunk to the shortest note.

    Counted over the notes, NOT the bars, and the difference is the whole point. A phrase of four
    bars looks like room for thirty-two syllables at `L:1/16`, but if three of those bars are rests
    it can hold eight — the rests are where the phrase is silent, and filling them would not lengthen
    the line, it would move it. A caller sizing a lyric line against a phrase needs the number that
    is true.
    """
    total = 0
    for measure in N.split_measures(line):
        if N.FULL_REST.match(measure.strip()):
            continue
        total += sum(t["units"] for t in _tokens(measure) if t["kind"] == "note")
    return total
