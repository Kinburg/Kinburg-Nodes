"""Siren Score — read the lyrics, write Siren Cast's plan. No LLM pass, nothing to hallucinate.

Everything the plan needs is already in the lyrics. The section list is there, in order; who sings
each one is in the marker (`[Verse 1 - Keen Burg - melodic female vocal]`); and how long a section
should be is a function of how many lines it has. The only thing an LLM adds to that is the chance of
getting it wrong — and the ways it gets it wrong are specific and were all observed in one run: it
invents sections that are not in the text, it pads the table to reach a bar budget it was told about,
and it writes a two-word member's name as one word.

So the length of the song comes OUT of the text here rather than being imposed on it. A lyric with six
short sections honestly runs about 70 seconds; asking for 180 does not make it longer, it makes
something fill the gap. Want three minutes — ask the lyrics pass for more sections.

**Two voices at once is the one thing the model cannot do**, and the reason is structural: the plan is
one audio code per 200 ms, and the caption is one description. `[Chorus - Keen Burg + Gru BNik]`
therefore asks for a caption holding two contradictory timbres over the same frames, and what comes
back is their average — measured once as "two female vocals rather than a man and a woman". What DOES
work is one voice at a time, which is what per-line markers inside a section express:

    [Bridge - Gru BNik + Keen Burg - shouted male and piercing female]
    [Gru BNik - deep aggressive growls]
    ЦЕ НЕ ЛОГІКА!
    [Keen Burg - high-pitched piercing screams]
    ЯКЩО Я ЗНИКНУ?!

Those inner markers SPLIT the section, one sub-section per voice, so the plan alternates instead of
blending. A header duet with no inner markers to split on becomes a lead voice plus a short "with
male backing harmonies" note — one clear timbre, and the harmonies said in words.

**Lengths run from the target backwards, not from a rate forwards.** `pad_to_seconds` says how long
the song is, `tail_bars` takes its slice off the end, instrumental sections take a fixed count each,
and everything left is shared among the sung sections **in proportion to their syllables**. The
singing rate is therefore an OUTPUT — the report prints what it came to, and says so when it lands
outside what a singer can do.

That is the way round it has to be. With a rate as the input, the slack between the words and the
wanted length hid *inside* the vocal sections, where the model fills it rather than singing slower:
a 6-second intro came back 40 seconds long and ate the first verse. With the length as the input,
slack can only go where it is asked for — the tail — and if there is too much of it the rate says so.
"""
import re

from .cast import (
    _bar_seconds, _beats, _mmss, _num, _resolve_voice, _roster, _voices_in_order,
)
from ..context.character_card import VOICE_TYPE

# Every bar count the Voice Plan grammar can write. Sections are snapped to these so a table built
# here and a table written by the LLM are the same kind of object.
ALLOWED_BARS = (2, 4, 6, 8, 12, 16, 24, 32)

# How many comma clauses a section's caption addition may carry. A guard against exactly what was
# measured: a Bridge with four inner markers accumulated 8 clauses / 233 characters, all of it inside
# the cfg delta, and the take came back sung by one indistinct voice.
MAX_CLAUSES = 4

DUET_SPLIT = "lead + split on inner markers"
DUET_ASIS = "both names in one section"
DUETS = [DUET_SPLIT, DUET_ASIS]

# The instrumental tail, alternating Solo / Break so the model is asked for a jam rather than one long
# undefined stretch.
#
# Its purpose changed once `lyrics_in_negative` went off. Before that, total length was the strongest
# variable measured (172 s good, 136 s bad, 100 s worse) and the tail was how a short lyric reached a
# song's worth of length. With the lyrics guided, the model follows the text closely enough that it no
# longer needs the room — and fills a long tail by repeating the last phrase instead. So the tail is
# now a small deliberate ending, not a length filler. Both findings are real; they were measured in
# different configurations, and this is the one that holds now.
PAD_LABELS = ("Solo", "Break")

# Tail lengths, in bars. A COMBO rather than a duration target on purpose: the tail used to be
# whatever was left over between the lyrics and a target length, which is how a six-section lyric
# ended up with forty bars of instrumental behind it. Measured with `lyrics_in_negative` off, a long
# tail is actively bad — the model repeats the last phrase to fill it and starts clipping the ends of
# held notes — while a short one buys a last chorus, a proper outro and a clean ending instead of the
# track stopping dead on the final word. So the choice is small and explicit.
TAIL_BARS = ["0", "2", "4", "6", "8", "12", "16", "24", "32"]

PAD_END = "after the vocals"
PAD_ENDS = "intro + outro"
PAD_BETWEEN = "between sections"
PAD_PLACEMENTS = [PAD_END, PAD_ENDS, PAD_BETWEEN]

# Section types an instrumental can follow without cutting a lyric in half. A verse running into its
# own pre-chorus is one gesture; after a chorus or a bridge is where a real record breathes.
PAD_AFTER = ("Chorus", "Post-Chorus", "Bridge")

# The band a sung line has to land in, and both ends are measured rather than guessed. The take that
# worked came out at 3.28 syllables a second; a verse given 2.34 was FILLED with instrumental instead
# of being sung slower, and the symptom was a 40-second intro swallowing the first verse. So the floor
# sits between those two. The ceiling is where words start being clipped.
SANE_RATE = (2.5, 8.0)

# What the take that worked came out at: 328 syllables over 100 s of sung sections. Used only to
# suggest the target length a lyric actually wants, when the one asked for does not suit it.
GOOD_RATE = 3.3

# Section synonyms → the canonical labels. ORDER MATTERS: `pre-chorus` and `post-chorus` have to be
# tested before `chorus`, or a pre-chorus becomes a chorus. Matched as a PREFIX of the marker, which
# is what keeps an annotation line like `[cold mechanical beat]` from being read as a section.
CANON = (
    (("pre-chorus", "prechorus", "pre chorus", "предприпев"), "Pre-Chorus"),
    (("post-chorus", "postchorus", "post chorus"), "Post-Chorus"),
    (("intro", "вступление", "вступ"), "Intro"),
    (("verse", "куплет"), "Verse"),
    (("chorus", "refrain", "hook", "припев", "рефрен"), "Chorus"),
    (("bridge", "chaos", "бридж", "переход", "перехід"), "Bridge"),
    (("solo", "instrumental", "соло"), "Solo"),
    (("break", "breakdown", "interlude", "drop", "проигрыш", "програш"), "Break"),
    (("outro", "ending", "coda", "finale", "финал", "фінал", "кода"), "Outro"),
)

# Labels that normally carry a vocal — a 0-line one of these is almost always "repeat the chorus".
VOCAL_LABELS = ("Verse", "Chorus", "Pre-Chorus", "Post-Chorus", "Bridge")

# Words that make an annotation line be about the VOICE rather than the arrangement.
_VOCAL_WORDS = ("vocal", "voice", "scream", "belt", "whisper", "rap", "sung", "singing", "spoken",
                "choir", "harmon", "falsetto", "growl", "shout", "chant", "ad-lib", "adlib",
                "вокал", "голос", "шепот", "шепіт", "крик")

# 'female' contains 'male', so male is matched with a lookbehind and female is tested first.
_FEMALE_RE = re.compile(r"female|woman|women|girl|жен|жіно|жино", re.I)
_MALE_RE = re.compile(r"(?<!fe)male|\bman\b|\bmen\b|\bboy\b|муж|чолов", re.I)

_BRACKETED = re.compile(r"^\[(.+)\]\s*$")
# A line wholly inside round brackets. Two kinds live in there and NEITHER adds duration:
# production notes ("(Distorted bassline, haunting atmospheric guitar)") are not sung at all, and
# backing echoes ("(Живий!)", "(Наша сила...)") are sung *over* the line above rather than after it.
# Counting them as sung lines inflated an instrumental intro to 16 syllables of imaginary singing.
_PARENS = re.compile(r"^\((.+)\)\s*$")
_NUM_AFTER = re.compile(r"^[\s.#:)-]*([1-9])")
_WORD = re.compile(r"[0-9a-zA-Zа-яёА-ЯЁіїєґІЇЄҐ]+")

# Words that only glue names together. Left on their own after a name is removed they describe
# nothing, so a fragment made only of these goes too.
_JOINERS = {"and", "with", "together", "both", "plus", "feat", "ft", "duet", "vs", "trading",
            "и", "та", "разом", "вместе", "спільно"}


# -------------------------------------------------------------------------------- text inspection
def _gender_of(text):
    """'female' / 'male' / None. Female first — 'female' contains 'male'."""
    t = str(text or "")
    if _FEMALE_RE.search(t):
        return "female"
    if _MALE_RE.search(t):
        return "male"
    return None


_VOWELS = re.compile(r"[aeiouyаеёиоуыэюяіїєAEIOUYАЕЁИОУЫЭЮЯІЇЄ]+")
_LETTERS = re.compile(r"[^\W\d_]+", re.UNICODE)


def _syllables(text):
    """Vowel groups, with English's silent final 'e' dropped.

    Exact for the Slavic languages (one vowel is one syllable) and close enough for English, where a
    trailing silent 'e' is the only common systematic overcount. This is what section lengths are
    computed from, because syllables are what a singer spends time on — a line of four words and a
    line of twelve do not take the same number of bars, and counting lines pretends they do."""
    total = 0
    for word in _LETTERS.findall(str(text or "")):
        n = len(_VOWELS.findall(word))
        if n > 1 and word[-1] in "eE":
            n -= 1
        total += n
    return total


def _vocalish(text):
    return any(w in str(text or "").lower() for w in _VOCAL_WORDS)


def _canon_label(head):
    """(label, rest) when `head` starts with a section synonym, else (None, head).

    A prefix test, not a search: `[Chorus - massive wall of guitars]` is a section and
    `[wall of guitars, no chorus pad]` is an annotation, and only a prefix test tells them apart."""
    low = head.strip().lower()
    for words, label in CANON:
        for w in words:
            if low.startswith(w):
                rest = head.strip()[len(w):]
                m = _NUM_AFTER.match(rest)
                if m:
                    return f"{label} {m.group(1)}", rest[m.end():]
                return label, rest
    return None, head


def _keys_of(name):
    """The forms a member can be referred to by: the full name, and the first name on its own."""
    parts = str(name or "").strip().split()
    if not parts:
        return []
    return [" ".join(parts).lower()] + ([parts[0].lower()] if len(parts) > 1 else [])


def _names_in(text, voices):
    """Members named in `text`, ordered by where they appear in it — so a duet marker keeps the
    order it was written in, and the first one named is the one that leads."""
    hits = []
    low = str(text or "").lower()
    for v in voices:
        name = (v.get("name") or "").strip()
        best = None
        for key in _keys_of(name):
            m = re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", low)
            if m and (best is None or m.start() < best):
                best = m.start()
        if best is not None:
            hits.append((best, name))
    return [n for _, n in sorted(hits)]


# ------------------------------------------------------------------------------ reading the sheet
def _new(label, marker, sub=False, names=(), inherit=()):
    return {"label": label, "marker": str(marker).strip(" -–—:,/"), "notes": [], "lines": 0,
            "syl": 0, "sub": sub, "names": list(names), "inherit": list(inherit)}


def _split_sections(lyrics, voices=(), split=True):
    """Lyrics → [{label, marker, notes, lines, sub, names, inherit}] in order, plus notes.

    A bracketed line whose text starts with a section word opens a section. A bracketed line that
    NAMES a member opens a sub-section of the current one (that is the alternation the model can
    actually sing). Any other bracketed line is an annotation of whatever is open — which is where
    the voice often hides, under a header that describes only the drums. Everything else is a sung
    line, and sung lines are what a section's length is computed from."""
    out, notes, stray = [], [], 0
    for raw in str(lyrics or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _PARENS.match(line)
        if m:                                  # a round-bracket line: an annotation, never a lyric
            if out:
                out[-1]["notes"].append(m.group(1).strip())
            continue
        m = _BRACKETED.match(line)
        if m:
            inner = m.group(1)
            label, rest = _canon_label(inner)
            if label:
                out.append(_new(label, rest))
                continue
            named = _names_in(inner, voices) if (split and voices) else []
            if named and out:
                prev = out[-1]
                if prev["lines"] == 0 and not prev["sub"]:
                    # The header had no lines of its own — it was only announcing this exchange, so
                    # it becomes the first sub-section and hands its arrangement notes down.
                    out[-1] = _new(prev["label"], inner, sub=True, names=named,
                                   inherit=[prev["marker"]] + prev["notes"])
                elif prev["sub"] and prev["lines"] == 0:
                    out[-1] = _new(prev["label"], inner, sub=True, names=named,
                                   inherit=prev["inherit"])
                else:
                    out.append(_new(prev["label"], inner, sub=True, names=named))
                continue
            if out:
                out[-1]["notes"].append(inner.strip())
            continue
        label, rest = _canon_label(line.rstrip(":"))
        if label and line.rstrip().endswith(":"):     # an unbracketed `Verse 1:` header
            out.append(_new(label, rest))
            continue
        if out:
            out[-1]["lines"] += 1
            out[-1]["syl"] += _syllables(line)
        else:
            stray += 1
    if stray:
        notes.append(f"{stray} line(s) before the first section marker were ignored — a section "
                     f"starts at a line like '[Verse 1 - Nina]'")
    return out, notes


# --------------------------------------------------------------------------------------- lengths
def _apportion(weights, floors, avail):
    """Share `avail` bars among sections in proportion to `weights` (their syllables).

    This is the whole length model, and it runs the other way round from the obvious one. Rather than
    "pick a singing rate, see how long the song comes out", the song's length is GIVEN — the target
    minus the tail — and the words are spread across it. The rate is then a consequence, which is the
    honest shape: a rate dial let the slack hide inside the vocal sections, where the model fills it
    instead of singing slower, and that is exactly how a 6-second intro came back 40 seconds long.

    Everything lands on an even number of bars (the plan's unit) and never below its floor. Largest
    remainder decides who gets the odd 2-bar units, with the index as a stable tiebreak so the same
    lyric always apportions the same way. Returns the shares; the caller checks the total, since the
    floors can make it exceed `avail` on a target that is simply too short for the words."""
    n = len(weights)
    if not n:
        return []
    total = float(sum(weights)) or 1.0
    units = max(0, int(avail) // 2)                    # everything is counted in 2-bar units
    floor_u = [max(1, -(-int(f) // 2)) for f in floors]
    if units <= sum(floor_u):
        return [2 * f for f in floor_u]                # floors alone already fill it, or overflow
    raw = [units * (w / total) for w in weights]

    # Largest remainder FIRST, floors after. The other order is what makes an 11-syllable line come
    # out shorter than a 9-syllable one: once both have been lifted to the same floor, the leftover
    # unit follows the fractional part, which the smaller line can easily win. Plain Hamilton on the
    # raw quotas cannot invert — a larger weight has a larger quota, so an equal integer part implies
    # a larger remainder too.
    share = [int(r) for r in raw]
    left = units - sum(share)
    for i in sorted(range(n), key=lambda i: (-(raw[i] - int(raw[i])), i))[:left]:
        share[i] += 1
    share = [max(share[i], floor_u[i]) for i in range(n)]

    # Lifting to the floors can overshoot; take the difference back off the longest sections, which
    # is the only direction that keeps the order intact.
    over = sum(share) - units
    while over > 0:
        i = max((i for i in range(n) if share[i] > floor_u[i]), default=None,
                key=lambda i: (share[i], -i))
        if i is None:
            break
        share[i] -= 1
        over -= 1
    return [2 * u for u in share]


def _snap_bars(want, floor):
    """Nearest bar count the grammar allows, never below `floor`. **Ties round UP** — 5 lines at 2
    bars each wants 10, which sits exactly between 8 and 12, and of the two failure modes giving the
    singer too little room is the worse one: a rushed vocal eats words, a roomy one only drags."""
    cands = [b for b in ALLOWED_BARS if b >= max(1, int(floor))] or [max(ALLOWED_BARS)]
    return min(cands, key=lambda b: (abs(b - float(want)), -b))


# --------------------------------------------------------------------------------------- captions
def _pad_split(total, block):
    """`total` bars of instrumental as FEW rows as it takes, each at most `block` long.

    Not many short ones. Every row in the plan is its own LM decode with its own caption, so a tail
    chopped into seventeen 2-bar rows is seventeen restarts — ten audio codes each, no room for the
    model to develop anything, and a stutter at every seam. Merging them by hand was the first thing
    that sounded better, so the node does it: 40 bars at a 16-bar block becomes 14 + 13 + 13, not
    twenty rows of 2.

    Parts are kept to whole even bar counts and shared out as evenly as the total allows, so no row
    is left as a stub."""
    total, block = max(0, int(total)), max(2, int(block))
    if total < 2:
        return []
    n = max(1, -(-total // block))               # ceil, so no part is ever LONGER than the block
    n = min(n, total // 2)                       # never more rows than there are 2-bar units
    unit = total // 2
    base, extra = divmod(unit, n)
    return [2 * (base + (1 if i < extra else 0)) for i in range(n)]


def _pad_rows(sizes):
    """The instrumental rows themselves, alternating Solo / Break so the model is asked for a jam
    rather than one long undefined stretch."""
    return [{"label": PAD_LABELS[i % len(PAD_LABELS)], "voice": "-", "bars": int(b), "extra": "",
             "lines": 0, "syl": 0, "how": "padding to length", "sub": False}
            for i, b in enumerate(sizes)]


def _place_padding(rows, pads, placement):
    """Put the instrumental blocks where the chosen placement says.

    The trade-off is not musical taste, it is where a lyric can be interrupted. AceStep gets the words
    with **no timing in them** — they are matched against the plan — so a gap in the middle asks the
    model to hold the line until the singing resumes, and if it doesn't, everything after it shifts.
    Padding at the ends cannot do that: it sits outside the lyric entirely. That is also the only
    layout with a take behind it, which is why it stays the default even though it is the least
    song-shaped of the three."""
    if not pads or not rows:
        return rows + pads
    if placement == PAD_ENDS:
        # ONE block opens, the rest is the outro. A fraction of the total was fine when the tail was
        # forty bars; with the small explicit tails this dial now sees, a third of two rows is zero
        # and the mode quietly did nothing.
        head = 1 if len(pads) >= 2 else 0
        return pads[:head] + rows + pads[head:]
    if placement == PAD_BETWEEN:
        # Boundaries after a chorus or a bridge — and only the LAST of a run of same-label
        # sub-sections, or an exchange of four Bridge lines would earn four solos.
        slots = [i + 1 for i, r in enumerate(rows)
                 if r["label"].split()[0] in PAD_AFTER
                 and (i + 1 >= len(rows) or rows[i + 1]["label"] != r["label"])]
        if slots:
            share, extra = divmod(len(pads) - 1, len(slots))   # one block opens the record
            out, used = list(pads[:1]), 1
            for i, row in enumerate(rows):
                out.append(row)
                if i + 1 in slots:
                    n = share + (extra if slots.index(i + 1) == len(slots) - 1 else 0)
                    out.extend(pads[used:used + n])
                    used += n
            return out + pads[used:]
    return rows + pads


def _arrangement(sec, include_vocal):
    """What goes in the plan's 4th column — i.e. onto that section's caption and nowhere else.

    `[Chorus - massive explosion of sound, wall of distorted guitars]` is a real instruction about
    that section. So is `[Intro - deep hypnotic SPOKEN WORD]`: when the voice was resolved to a
    member, the marker's wording about HOW they sing here is still worth keeping — the card says who
    the singer is, the marker says what they do in this section. It is only skipped when that same
    wording already became the voice cell itself, which would just repeat it."""
    src = list(sec["inherit"]) + [sec["marker"]] + list(sec["notes"])
    if not include_vocal:
        src = [s for s in src if not _vocalish(s)]
    return ", ".join(p.strip(" -–—:,/") for p in src if p.strip(" -–—:,/"))


def _trim_redundant(text, tags, names=()):
    """What of a marker is worth pasting onto the section's caption.

    Two things come out. **Every band member's name**, because AceStep's caption is a music
    description and "Keen Burg" is not one — a marker written as `[Verse 1 - Keen Burg - hypnotic
    female vocal]` must contribute the delivery, not the credit. All of them, not just this section's
    singer: a sub-section inherits its parent's `[Bridge - Gru BNik + Keen Burg]` header, and the
    name that did NOT end up singing is exactly as meaningless to the model as the one that did.
    **And any fragment whose every word
    the card already says**: `hypnotic melodic FEMALE vocal` adds nothing over a card reading "female
    vocal, melodic, hypnotic", while `[Intro - deep hypnotic spoken word]` adds "spoken word" and
    stays. Fragment by fragment, so one new word doesn't drag four redundant ones in with it."""
    text = str(text or "")
    for name in names:
        # Full name FIRST, then the first name on its own. The other order removes "Keen" and
        # leaves "Burg" stranded, which then reads as a word nobody has ever heard of.
        for key in _keys_of(name):
            text = re.sub(r"(?<!\w)" + re.escape(key) + r"(?!\w)", " ", text, flags=re.I)
    have = {w.lower() for w in _WORD.findall(tags or "")} | _JOINERS
    keep, seen = [], set()
    for frag in text.replace(" - ", ", ").split(","):
        f = " ".join(frag.split()).strip(" -–—:;/+&")
        words = [w.lower() for w in _WORD.findall(f)]
        if not words or all(w in have for w in words):
            continue
        if f.lower() in seen:      # inner markers repeat themselves across an exchange
            continue
        seen.add(f.lower())
        keep.append(f)
    return ", ".join(keep)


def _cap_clauses(text, limit=MAX_CLAUSES):
    """(text, dropped). A long caption is not a detailed caption: everything here lands inside the
    cfg delta, so eight clauses of contradictory instruction guide harder than one clear one."""
    frags = [f.strip() for f in str(text or "").split(",") if f.strip()]
    if len(frags) <= limit:
        return ", ".join(frags), 0
    return ", ".join(frags[:limit]), len(frags) - limit


# --------------------------------------------------------------------------------- who sings what
def _voice_for(sec, voices, roster):
    """A section's voice cell: a roster name, an inferred one, the marker's own words, or '-'.

    Steps, most certain first. The last two are why this works on lyrics written before any of this
    existed: a marker that says `deep hypnotic spoken word - MALE vocal` still produces a usable
    caption fragment even with nobody wired in."""
    if sec["names"]:                       # an inner marker already said, by name
        return list(sec["names"]), "named in the inner marker", []
    text = " ".join([sec["marker"]] + sec["notes"])
    hit = _resolve_voice(text, roster)     # a clean `Nina` or `Nina + Alex` cell
    if hit["names"] and not hit["verbatim"]:
        return list(hit["names"]), "name in the marker", []
    named = _names_in(text, voices)        # …or a name buried in prose
    if named:
        return named, "name in the marker", []

    want = _gender_of(text)
    if want:
        match = [v for v in voices
                 if (_gender_of(v.get("tags")) or _gender_of(v.get("gender"))) == want
                 and (v.get("name") or "").strip()]
        if len(match) == 1:
            return [match[0]["name"]], f"{want.upper()} in the marker → {match[0]['name']}", []
        if len(match) > 1:
            names = ", ".join(v["name"] for v in match)
            return ([match[0]["name"]], f"{want.upper()} → {match[0]['name']}",
                    [f"{sec['label']}: the marker says {want.upper()} and {len(match)} members "
                     f"match ({names}) — took {match[0]['name']}. Put the name in the marker to "
                     f"decide it yourself."])

    vocal = [n for n in sec["notes"] if _vocalish(n)]
    if _vocalish(sec["marker"]):
        vocal.insert(0, sec["marker"])
    if vocal:
        note = []
        if roster:
            note = [f"{sec['label']}: no member named or identified in the marker — its own wording "
                    f"was used as the vocal description instead"]
        return [", ".join(vocal)], "marker text, verbatim", note
    return [], "no vocal in the marker", []


def _backing(names, voices):
    """The short note that stands in for the voices a lead is singing over. Deliberately short and
    in words: the plan has ONE timbre per moment, so the harmonies can only be described, and a long
    description is what made a duet come back as a single indistinct voice."""
    genders = []
    for name in names:
        v = next((x for x in voices if (x.get("name") or "").strip() == name), None)
        g = _gender_of((v or {}).get("tags")) or _gender_of((v or {}).get("gender"))
        if g and g not in genders:
            genders.append(g)
    if len(genders) == 1:
        return f"with {genders[0]} backing harmonies"
    return "with backing harmonies"


class KinburgSirenScore:
    """Lyrics → Siren Cast's plan, deterministically. Optionally checks an LLM's table instead."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"forceInput": True, "tooltip": "The lyrics, with '[Verse 1 - ...]' style markers. The SAME text that goes to Siren Cast — wire one source into both.\n\nWhat the markers need: the section name FIRST (Intro / Verse 1 / Pre-Chorus / Chorus / Post-Chorus / Bridge / Solo / Break / Outro, or a synonym), then ideally the member's name, then how they sing it. 'MALE'/'FEMALE' works instead of a name when the roster has exactly one of each.\n\nA bracketed line that NAMES a member splits the section — that is how an exchange between two singers is written, and it is the only way the model can actually do two voices. Any other bracketed line is an annotation of the current section."}),
                "bpm": ("INT", {"default": 120, "min": 10, "max": 300, "tooltip": "Tempo — only used to report the resulting length in seconds. The plan itself is written in bars, so Siren Cast converts it with ITS bpm; wire both from one source or the report will lie to you."}),
                "timesignature": ("STRING", {"default": "4", "tooltip": "Beats per bar, for the same reason as 'bpm'. A plain field rather than a dropdown because a combo input cannot accept the STRING a text parser hands it — anything with a digit in it is read (4, '4', '4/4')."}),
                "pad_to_seconds": ("FLOAT", {"default": 150.0, "min": 10.0, "max": 2000.0, "step": 1.0, "tooltip": "How long the song should be, in seconds — the reference the whole plan is built from.\n\nThe bars it buys are shared out like this: instrumental sections (a marker with no sung lines) take 'instrumental_bars' each, 'tail_bars' comes off the end, and everything left goes to the sung sections IN PROPORTION TO THEIR SYLLABLES.\n\nSo the singing rate is not a dial here, it is a consequence — the report prints what it came out at. If that lands outside 2-8 syllables a second the node says so and names the length this lyric would actually suit, because a section given more room than its words need does not get sung slower: the model FILLS it, and at the top of a song that reads as the intro running on."}),
                "tail_bars": (TAIL_BARS, {"default": "4", "tooltip": "Bars of instrumental added after the vocals, so the track does not stop dead on the last word. 0 = none.\n\nMeasured with 'lyrics_in_negative' off, which is the setting that makes the voices land: a SHORT tail (2-8 bars) gives a last chorus, a proper outro and a clean ending. A LONG one is actively bad — the model repeats the last phrase over and over to fill it, and starts eating the ends of notes the singer should be holding. It also lengthens the intro, which is the other thing to listen for.\n\nThis replaced a target-duration dial. That one made the tail whatever was left over between the lyrics and the target, which is how a six-section lyric ended up with forty bars of instrumental behind it."}),
                "pad_placement": (PAD_PLACEMENTS, {"default": PAD_END, "advanced": True, "tooltip": "Where the instrumental blocks go. The trade-off is not taste — it is where a lyric can be interrupted. AceStep gets the words with NO timing in them, matched against the plan, so a gap in the middle asks the model to hold the line until the singing resumes; if it doesn't, everything after it shifts.\n\n• after the vocals — everything at the end. The least song-shaped, and the only one with a take behind it, which is why it is the default.\n• intro + outro — a third opens the record, the rest is an outro jam. Still entirely outside the lyric, so it carries the same zero risk and sounds more like a record. Try this second.\n• between sections — one block opens, the rest go after choruses and bridges (never inside a verse running into its own pre-chorus), remainder at the end. The most song-shaped and the only one that can make the lyric drift."}),
                "pad_block_bars": ("INT", {"default": 16, "min": 2, "max": 64, "advanced": True, "tooltip": "Longest instrumental section the padding may use. The tail is split into as FEW rows as it takes, each at most this, sharing the bars out evenly — 40 bars at 16 becomes 14 + 13 + 13, not twenty rows of 2.\n\nWhy it matters: every row is its own LM decode with its own caption, so a tail chopped into 2-bar rows is a restart every ten audio codes — no room for the model to develop anything, and a seam at each one. Merging them by hand was the first thing that sounded better, so this does it by default. Raise it for one long jam, lower it for a tail that changes character more often."}),
                "min_bars": ("INT", {"default": 4, "min": 2, "max": 32, "advanced": True, "tooltip": "Floor for a whole section. Below about 4 bars a section is too short for a voice to establish itself.\n\nSub-sections made by inner markers are exempt — an exchange of single shouted lines is meant to be short, and they get a floor of 2 bars instead."}),
                "instrumental_bars": ("INT", {"default": 4, "min": 2, "max": 32, "advanced": True, "tooltip": "Length for a section with no sung lines at all (an instrumental intro, a solo, a break). There is no line count to derive it from, so it is simply this."}),
                "duets": (DUETS, {"default": DUET_SPLIT, "advanced": True, "tooltip": "What to do when a section names more than one singer. The plan carries ONE audio code per 200 ms and the caption is one description, so two timbres over the same frames come back as their average — measured once as 'two female vocals' where a man and a woman were asked for.\n\n• lead + split on inner markers (recommended) — a bracketed line naming a member starts a sub-section, so the voices ALTERNATE, which the model can do. A header duet with nothing to split on becomes the first-named singer plus a short 'with male backing harmonies' note.\n\n• both names in one section — writes both names in one cell, i.e. asks for the blend. Here to A/B against, not because it works."}),
                "arrangement_notes": ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Put the non-vocal part of a marker into the plan's 4th column, which appends it to that section's caption — '[Chorus - massive explosion of sound, wall of distorted guitars]' is a real instruction about the chorus and this is the only place it fits.\n\nEverything here lands inside Siren Cast's cfg delta, so it is guided as hard as the voice is: at most " + str(MAX_CLAUSES) + " clauses per section are kept and the rest is reported. If a take comes back muddy or sung by one indistinct voice, turning this OFF is the cheapest thing to try."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Print the report to the console. The same text is always on the 'report' output."}),
            },
            "optional": {
                "voice_1": (VOICE_TYPE, {"tooltip": "A band member — Character Card's 'voice' output. Wire the SAME cards you wire into Siren Cast: this node matches marker names against them and reads their gender when a marker only says MALE / FEMALE.\n\nWith nothing wired the markers' own wording is used as the vocal description, which still works — it just can't say 'Nina', and inner markers can't split a section."}),
                "voice_2": (VOICE_TYPE,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("plan", "report")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/sampling"
    DESCRIPTION = ("Builds Siren Cast's plan from the lyrics instead of asking an LLM for it: the "
                   "section list and order come from the '[Verse 1 - ...]' markers, the voice from "
                   "the name in the marker (or MALE/FEMALE matched against the wired Character "
                   "Cards, or the marker's own words), and each section's length from how many "
                   "lines it has. So the song is exactly as long as its lyrics, which is the one "
                   "thing a bar budget cannot make true. Two singers in one section are ALTERNATED "
                   "where per-line markers allow it and reduced to a lead plus backing where they "
                   "don't, because one description over the same frames comes back as a blend. "
                   "Wire an LLM's table into 'plan' instead and it is passed through and audited.")

    def run(self, lyrics, bpm, timesignature, pad_to_seconds, tail_bars, pad_placement,
            pad_block_bars, min_bars, instrumental_bars, duets, arrangement_notes, verbose=True,
            **kwargs):
        beats = _beats(timesignature)
        bar = _bar_seconds(bpm, beats) or 0.0
        voices = _voices_in_order(kwargs)
        roster, notes = _roster(voices)
        secs, split_notes = _split_sections(lyrics, voices, split=duets == DUET_SPLIT)
        notes.extend(split_notes)
        if not secs:
            raise RuntimeError(
                "[Siren Score] no section markers found in the lyrics. A section starts at a line "
                "like '[Verse 1 - Nina]' or '[Chorus - powerful FEMALE belt]' — a bracketed line "
                "whose first word is a section name (Intro / Verse / Pre-Chorus / Chorus / "
                "Post-Chorus / Bridge / Solo / Break / Outro, or a synonym).")

        # Every name in the band, for stripping out of the captions — see `_trim_redundant`.
        cast_names = [(v.get("name") or "").strip() for v in voices if (v.get("name") or "").strip()]

        rows, lines = [], []
        for sec in secs:
            names, how, warn = _voice_for(sec, voices, roster)
            notes.extend(warn)
            backing = ""
            if len(names) > 1 and duets == DUET_SPLIT:
                # One timbre per moment. The lead sings; the rest are said in words.
                backing = _backing(names[1:], voices)
                notes.append(f"{sec['label']}: {' + '.join(names)} sing together — {names[0]} was "
                             f"made the lead and the rest became '{backing}'. Two voices over the "
                             f"same frames come back as a blend; write per-line markers like "
                             f"'[{names[1]} - …]' inside the section to have them alternate instead.")
                names = names[:1]
            voice = " + ".join(names) if names else "-"

            extra, dropped = "", 0
            if arrangement_notes:
                verbatim = how == "marker text, verbatim"
                extra = _arrangement(sec, include_vocal=not verbatim)
                if not verbatim:
                    hit = _resolve_voice(voice, roster)
                    extra = _trim_redundant(extra, hit["add"], cast_names or hit["names"])
                extra, dropped = _cap_clauses(", ".join(x for x in (backing, extra) if x))
            elif backing:
                extra = backing
            if dropped:
                notes.append(f"{sec['label']}: {dropped} arrangement clause(s) past the first "
                             f"{MAX_CLAUSES} were dropped — everything in that column is guided as "
                             f"hard as the voice, and a long caption is not a detailed one")
            if sec["lines"] == 0 and sec["label"].split()[0] in VOCAL_LABELS:
                notes.append(f"{sec['label']} has no sung lines — if that meant 'repeat the "
                             f"chorus', write the lines out: AceStep has no repeat, so this section "
                             f"became {int(instrumental_bars)} bars of instrumental")
            rows.append({"label": sec["label"], "voice": voice, "bars": 0, "extra": extra,
                         "lines": sec["lines"], "syl": sec["syl"], "how": how, "sub": sec["sub"],
                         "floor": 2 if sec["sub"] else int(min_bars)})

        # ---------------------------------------------------------------- lengths, from the budget
        # The target length and the tail are given; what is left belongs to the words, shared out in
        # proportion to how many syllables each section carries. Sections with no sung lines are not
        # in that share — they are a fixed `instrumental_bars` and come off the top.
        tail = max(0, _beats(tail_bars, 0) if isinstance(tail_bars, str) else int(tail_bars))
        want_total = int(round(float(pad_to_seconds) / bar)) if bar > 0 else 0
        fixed = [r for r in rows if r["syl"] == 0]
        sung = [r for r in rows if r["syl"] > 0]
        for r in fixed:
            r["bars"] = max(2, int(instrumental_bars))
        avail = want_total - tail - sum(r["bars"] for r in fixed)
        if not sung:
            notes.append("no section has any sung lines — nothing to apportion")
        else:
            shares = _apportion([r["syl"] for r in sung], [r["floor"] for r in sung], avail)
            for r, b in zip(sung, shares):
                r["bars"] = b
            over = sum(shares) - max(0, avail)
            if over > 0:
                notes.append(
                    f"the words do not fit: their floors alone need {sum(shares)} bars but only "
                    f"{max(0, avail)} were left after the {tail}-bar tail and "
                    f"{sum(r['bars'] for r in fixed)} bars of instrumental sections. The song came "
                    f"out {_num(over * bar)} s longer than the {_num(pad_to_seconds)} s asked for — "
                    f"raise the target, shorten the tail, or lower 'min_bars'.")

        sung_bars = sum(r["bars"] for r in rows)
        sung = sum(r["lines"] for r in rows)
        syl = sum(r["syl"] for r in rows)
        # The rate that actually lands on the WORDS — over the sections that have syllables in them,
        # not over the whole plan. Diluting it with instrumental bars hides exactly the thing this
        # figure exists to catch: a vocal section given more room than its words need gets FILLED,
        # not sung slower, and at the top of a song "filled" reads as the intro running on.
        vocal_bars = sum(r["bars"] for r in rows if r["syl"] > 0)
        actual = (syl / (vocal_bars * bar)) if vocal_bars and bar else 0.0
        pad_bars = 0
        if tail >= 2:
            sizes = _pad_split(tail, max(2, int(pad_block_bars)))
            rows = _place_padding(rows, _pad_rows(sizes), pad_placement)
            pad_bars = sum(sizes)
            notes.append(f"the words hold {sung_bars} bars ({_num(sung_bars * bar)} s); "
                         f"{'/'.join(str(b) for b in sizes)} bars of instrumental were added "
                         f"({pad_placement}) so the track does not stop dead on the last word.")
            if pad_bars > sung_bars // 2:
                notes.append(f"{pad_bars} instrumental bars against {sung_bars} sung is a long tail. "
                             f"Measured with 'lyrics_in_negative' off: a long tail makes the model "
                             f"repeat the last phrase over and over to fill it, and eat the ends of "
                             f"held notes. 2-8 bars gives a last chorus and a proper outro; more "
                             f"gives a loop.")

        built = "".join(f"{r['label']} | {r['voice']} | {r['bars']} bars"
                        + (f" | {r['extra']}" if r["extra"] else "") + "\n" for r in rows)
        built += "END\n"

        out_plan = built
        src = f"{len(rows)} section(s) read from the lyrics"

        total_bars = sum(r["bars"] for r in rows)
        seconds = total_bars * bar
        at = 0.0
        for r in rows:
            dur = r["bars"] * bar
            lines.append(f"  {_mmss(at)} → {_mmss(at + dur)}  {r['label']:<12} {r['voice']:<26} "
                         f"{r['bars']:>2} bars  ({r['lines']} line(s), {r['syl']} syl) · "
                         f"{r['how']}"
                         + (f" · +{r['extra'][:44]}" if r["extra"] else ""))
            at += dur
        head = (f"Siren Score — {src} · {total_bars} bars = {_mmss(seconds)} ({_num(seconds)} s) "
                f"@ {int(bpm)} bpm {beats}/4 · {len(voices)} voice(s) wired · "
                f"{sung} line(s) / {syl} syllable(s)"
                + (f" at {_num(round(actual, 2))} syllables/s over the sung sections"
                   if actual else "")
                + (f" · +{pad_bars} bars instrumental" if pad_bars else ""))
        # The singing rate is an OUTPUT now, not a dial — it falls out of the target length, which is
        # what makes it the number that says whether the target suits the words. Too slow and the
        # model FILLS the room rather than singing slower, and at the top of a song that reads as the
        # intro running on. So out of range it is explained, with the target this lyric would suit.
        if actual and not SANE_RATE[0] <= actual <= SANE_RATE[1]:
            slow = actual < SANE_RATE[0]
            wants = syl / GOOD_RATE + (tail + sum(r["bars"] for r in rows if r["syl"] == 0)) * bar
            worst = (max if slow else min)(
                (r for r in rows if r["syl"] > 0), default=None,
                key=lambda r: r["bars"] * bar / max(r["syl"], 1))
            where = (f" The {'roomiest' if slow else 'tightest'} is {worst['label']}: "
                     f"{worst['syl']} syllables over {_num(worst['bars'] * bar)} s = "
                     f"{_num(round(worst['syl'] / (worst['bars'] * bar), 2))} syllables/s."
                     if worst else "")
            notes.append(
                f"the words land at {_num(round(actual, 2))} syllables/s, "
                f"{'below' if slow else 'above'} the {_num(SANE_RATE[0])}-{_num(SANE_RATE[1])} a "
                f"singer works in.{where} "
                + (f"Sections have more room than their words need, and the model FILLS the rest "
                   f"rather than singing slower. This lyric wants about {_num(round(wants))} s, not "
                   f"{_num(pad_to_seconds)}: lower the target, or lengthen the tail so the slack "
                   f"sits somewhere deliberate."
                   if slow else
                   f"The words are crammed, so they will be rushed or clipped. This lyric wants "
                   f"about {_num(round(wants))} s, not {_num(pad_to_seconds)}: raise the target, or "
                   f"shorten the tail."))
        report = "\n".join([head] + lines + [f"  ⚠ {w}" for w in notes])
        if verbose:
            print("[Siren Score] " + report.replace("\n", "\n[Siren Score] "))
        return (out_plan, report)



NODE_CLASS_MAPPINGS = {"KinburgSirenScore": KinburgSirenScore}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSirenScore": "Siren Score (Lyrics → Plan) 🧜"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
