"""Who is in a shot, and the ref2va prompt that tells H3 so.

Card Presets hands Morpheus its cast as `KINBURG_CAST` dicts — name, card type, the card's field
values and an optional picture. Per shot:

* a card is IN the shot when its name appears in the shot's text as a whole word, with the case as
  written, outside quotes. Case, because a name can also be an English word ("Keen"); quotes,
  because the sung line and text printed on clothes are quoted and can hold the names too (a band
  called "Keen Burg No Death"). A negation ("without Burg") still counts — name only who is in
  frame.
* every card in the shot gets a `subject_definitions` line; one with a picture is also attached as
  a reference image and named `<Picture K>`, numbered in wiring order among the attached ones.
* the shot's own text becomes `[Shot 1]` of `detailed_description` — unless it already carries a
  `[Shot 1]` line, in which case whatever stands before it (the Storyboard's style paragraph) is
  the description's opening, the way the official examples lay it out.

The section layout is H3's own ref2va caption format, the one ComfyUI's MiniMax H3 reference
templates are written in. The character-sheet wording is what the author found by trial to stop the
model drawing a sheet's four views as four people, or opening the video on the sheet itself.

Pure Python: no torch, no comfy, so it is tested headless.
"""
import re

from ..context.character_card import _FIELDS

_QUOTED = re.compile(r'"[^"]*"|“[^”]*”|«[^»]*»|„[^“”]*[“”]')


def roster(cards):
    """The usable cards in wiring order, plus a note for every one that was dropped: a card with
    no name can never be named by a shot, and a second card under the same name would make which
    picture a shot gets depend on luck."""
    out, seen, notes = [], set(), []
    for n, c in enumerate(cards, 1):
        if not c.get("name"):
            notes.append(f"cast card {n} has no name, so no shot can ever name it — skipped")
        elif c["name"] in seen:
            notes.append(f"'{c['name']}' is wired twice — the first card is used")
        else:
            seen.add(c["name"])
            out.append(c)
    return out, notes


def describe(cast):
    """The cast as an LLM writer is told it: names to use verbatim and what each one is. Looks are
    left out on purpose — the reference pictures and the stamped definitions carry them, and a
    writer that has read a description re-describes it in every shot."""
    lines = []
    for c in cast:
        v = c.get("values") or {}
        if c.get("type") == "entity":
            desc = " ".join(str(v.get("description") or "").split())[:200]
            lines.append(f"- {c['name']} (a place or thing)" + (f": {desc}" if desc else ""))
        else:
            who = ", ".join(x for x in (str(v.get("gender") or "").strip(), str(v.get("age") or "").strip()) if x)
            notes = " ".join(str(v.get("notes") or "").split())[:200]
            lines.append(f"- {c['name']} (a person" + (f", {who}" if who else "") + ")"
                         + (f": {notes}" if notes else ""))
    return "\n".join(lines)


_LYRIC_ROW = re.compile(r"^\s*(\d+)\s*[.)]\s*(?:\[([^\]]*)\]\s*)?(.*)$")
_SUNG = re.compile(r'(?:([^";]+?)\s+-\s+)?"([^"]*)"')


def parse_shot_lyrics(text):
    """Orpheus' `lyrics` output → {shot number: [(singer, words), …]}.

    One line per shot, `3. [Chorus] Keen - "words"; Burg - "words"` (the section in brackets is
    for the planner and is skipped here), or `4. (instrumental - no words)`.
    The singer may be missing when the plan had no voice column; the words are kept anyway."""
    out = {}
    for line in (text or "").splitlines():
        m = _LYRIC_ROW.match(line)
        if not m:
            continue
        sung = [(" ".join(v.split()), " ".join(t.split())) for v, t in _SUNG.findall(m.group(3))]
        out[int(m.group(1))] = [(v, t) for v, t in sung if t]
    return out


def links_by_section(text, first, count):
    """Per-shot links for shots `first`…`first + count - 1` out of Orpheus' `lyrics`: a shot in the
    same section as the one before it continues that take, a new section starts on a cut. Rows
    without a section (instrumental) count as one section of their own."""
    sec = {int(m.group(1)): (m.group(2) or "").strip()
           for m in map(_LYRIC_ROW.match, (text or "").splitlines()) if m}
    return ["continue" if k > 1 and sec.get(k, "") == sec.get(k - 1, "") else "cut"
            for k in range(first, first + count)]


def places_in_order(frames, links, count):
    """Which of `count` places each shot is in when they are simply played in order: the clip's
    length is shared out evenly and a shot belongs to the share its middle falls in. A shot that
    continues the previous one stays where that one was — a place can only change on a cut."""
    total, at, out = sum(frames) or 1, 0, []
    for i, f in enumerate(frames):
        k = min(count - 1, int((at + f / 2) / total * count))
        out.append(out[-1] if (i and links[i] == "continue") else k)
        at += f
    return out


def pick_place(line, places):
    """(place or None, the line without its tag) for one line of a plan.

    The planner is asked to open a line with the place in round brackets — `(Roof) Keen sings…`.
    A line that names a place anywhere outside quotes counts too; one that names none returns
    None and the caller keeps the previous shot's place."""
    m = re.match(r"\s*\(([^)]*)\)\s*", line or "")
    if m:
        tag = m.group(1).strip().lower()
        for p in places:
            if p["name"].lower() == tag:
                return p, line[m.end():]
    named = present(places, line)
    return (named[0] if named else None), line or ""


def present(cast, text):
    """The cards whose name the shot's text mentions outside quotes, in wiring order."""
    bare = _QUOTED.sub(" ", text or "")
    return [c for c in cast
            if c["name"] and re.search(r"(?<!\w)" + re.escape(c["name"]) + r"(?!\w)", bare)]


def _features(values):
    return "; ".join(f"{label}: {v}" for key, label in _FIELDS
                     if (v := str(values.get(key) or "").strip()))


def _description(values):
    return str(values.get("description") or "").strip()


_SINGS = re.compile(r'\b(?:sings?|[Ss]ung)\b\s*[:,]?\s*["“«„]')


def assemble(scene, cards, song=False):
    """(prompt, pictures): the shot's ref2va prompt, and the pictures to attach in `<Picture>` order.

    `song` — the shot hears a slice of the song. The slice only guides H3's own soundtrack, it does
    not replace it, so wherever the slice is silent (after a line, in a vocal stem without music)
    the model is free to go on singing: the mouth keeps moving under no words. The soundscape
    section is where the official layout says what is heard, so that is where it is told to stop."""
    defs, keep, pictures = [], [], []
    for n, c in enumerate(cards, 1):
        name, values = c["name"], c.get("values") or {}
        person = c.get("type") != "entity"
        k = None
        if c.get("picture") is not None:
            pictures.append(c["picture"])
            k = len(pictures)
        if person:
            feats = _features(values)
            if k:
                defs.append(f"<Subject {n}> is {name}, the person in the multi-angle character reference "
                            f"sheet <Picture {k}>, which shows the same single person from the front, the "
                            f"back, in profile and in close-up. Reference role: identity and costume "
                            f"anchor." + (f" Features to follow exactly: {feats}." if feats else "")
                            + f" Do not carry over the backdrop or the multi-view layout of <Picture {k}>.")
                keep.append(f"<Subject {n}> (appears in [Shot 1]): fully_preserved - face, hair and outfit "
                            f"of <Picture {k}>. {name} appears once, as one person.")
                keep.append(f"<Picture {k}> layout: weak_reference only - the backdrop and the multi-view "
                            f"sheet arrangement are discarded entirely.")
            else:
                defs.append(f"<Subject {n}> is {name}." + (f" Features: {feats}." if feats else ""))
        else:
            desc = _description(values)
            if k:
                defs.append(f"<Subject {n}> is {name}, shown in <Picture {k}>. Reference role: appearance "
                            f"anchor." + (f" {desc}" if desc else ""))
                keep.append(f"<Subject {n}> (appears in [Shot 1]): fully_preserved - as shown in "
                            f"<Picture {k}>.")
            else:
                defs.append(f"<Subject {n}> is {name}." + (f" {desc}" if desc else ""))

    names = [c["name"] for c in cards]
    who = (" featuring " + (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0])
           if names else "")
    parts = []
    if defs:
        parts.append("subject_definitions:\n" + "\n".join(defs))
    parts.append(f"summary:\n[reference generation] A single continuous shot{who}.")
    if keep:
        parts.append("retention_analysis:\n" + "\n".join(keep))
    scene = (scene or "").strip()
    parts.append("detailed_description:\n" + (scene if re.search(r"(?m)^\[Shot 1\]", scene)
                                              else "[Shot 1] " + scene))
    if song:
        parts.append("overall_soundscape:\n" + (
            "The only voice is the quoted line, sung once. After its last word the singer closes "
            "their mouth and stays silent: no more words, no humming, no ad-libs."
            if _SINGS.search(scene) else
            "Nobody sings or speaks in this shot: every mouth stays closed."))
    return "\n\n".join(parts), pictures
