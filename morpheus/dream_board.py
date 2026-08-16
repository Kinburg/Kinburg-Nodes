"""Morpheus Dream Board 🌙 — a chat becomes a storyboard.

The bridge between `Local LLM Chat (GGUF)` and `Morpheus Storyboard 🌙`. You talk to a character,
pictures accumulate in the conversation, and this node turns the part of it you pick into the three
things Storyboard eats: the keyframes, one direction line per shot, and the shot lengths.

**Pictures define the shots, not the other way round.** In Morpheus a keyframe physically sits
*between* shots — frame 2 ends shot 1 and starts shot 2 — so a picture in the middle of a hand-drawn
selection could never be a boundary at all. Turning it round removes the whole problem:

    picture 1 ····· messages ····· picture 2 ····· messages ····· picture 3
              └──────  shot 1  ──────┘        └──────  shot 2  ──────┘

So K included pictures give K-1 bounded shots, no shot can ever hold three keyframes, and nothing
has to be forbidden or silently dropped. Messages before the first picture join shot 1 (that is
where a scene gets set up); messages after the last one become text-only shots at the end, split by
whatever breaks you placed among them.

The cost, and it is the real limitation of this version: a shot boundary must land on a picture. A
long stretch of story with no picture in it cannot be cut into several shots, because the interior
boundary would have no keyframe and Storyboard fills boundaries from the first one onwards with no
gaps. If a span runs past H3's ~15 s, put another picture in the chat.

**Beats are verbatim for now** — the picked messages, whitespace-collapsed, speaker-labelled the
same way chat_node does it. Storyboard's own planning call is skipped when `beats` is filled ("your
lines win"), so what the shot writer reads is exactly what you selected. If the writing comes out
poorly, the next step is a small LLM pass here that rewrites each shot's messages into one
direction line — which is why nothing in this file assumes the text came straight from the chat.

The node holds a SNAPSHOT of the conversation, pulled by the frontend's Update History button, in
its own `board_state`. It has to: there is no graph link to the chat (a link would put this node
below the chat's blocked output, so rendering a video would need the chat to run first), so at
execution time the snapshot is all there is. Note that this means the picked part of the chat is
saved inside the workflow twice — once in the chat node, once here.
"""
import json
import os
import re

from ..local_llm.attachments import att_base, resolve_refs
from ..categories import CAT_MORPHEUS

MORPHEUS_SHOT = "KINBURG_MORPHEUS_SHOT"   # mirrors morpheus/nodes.py; importing it would pull torch

BOARD_VERSION = 1
DEFAULT_DURATION = 5.17          # H3's shortest trained shot, and Storyboard's default
LINK_MODES = ["continue", "cut"]
DEFAULT_LINK = "continue"


def _parse_state(raw):
    """board_state -> the snapshot and the picks. Never raises: a hand-edited or truncated state
    degrades to an empty board instead of breaking the graph."""
    try:
        st = json.loads(raw) if raw else {}
    except Exception:
        st = {}
    if not isinstance(st, dict):
        st = {}
    msgs = st.get("msgs")
    return {
        "msgs": [m for m in msgs if isinstance(m, dict)] if isinstance(msgs, list) else [],
        "skip": st.get("skip") if isinstance(st.get("skip"), list) else [],
        "noimg": st.get("noimg") if isinstance(st.get("noimg"), list) else [],
        "breaks": st.get("breaks") if isinstance(st.get("breaks"), list) else [],
        "dur": st.get("dur") if isinstance(st.get("dur"), list) else [],
        "lnk": st.get("lnk") if isinstance(st.get("lnk"), list) else [],
    }


def _speaker(m):
    """Same convention as chat_node._speaker_of, so a beat reads like the transcript it is."""
    if m.get("r") == "u":
        return "User"
    return str(m.get("p") or "Assistant")


def _line(m):
    """One message as one line — beats are positional, so a newline would shift every later shot."""
    text = re.sub(r"\s+", " ", str(m.get("t") or "")).strip()
    return f"{_speaker(m)}: {text}" if text else ""


def _ints(xs):
    out = set()
    for x in xs or ():
        try:
            out.add(int(x))
        except (TypeError, ValueError):
            pass
    return out


def plan_board(msgs, skip=(), noimg=(), breaks=()):
    """Derive the shot list. Pure, so it carries the design's whole rule set on its own.

    Returns (shots, boundaries, notes) where a shot is
    ``{"msgs": [indices], "start": ref|None, "end": ref|None, "tail": bool}``
    and `boundaries` is the ordered list of picture refs that become the keyframe batch.
    """
    skip, brk = _ints(skip), _ints(breaks)
    dead = {str(x) for x in (noimg or ())}

    bounds = []                       # (message index, picture ref), in conversation order
    for i, m in enumerate(msgs):
        for a in (m.get("img") or ()):
            if isinstance(a, dict) and a.get("name") and str(a["name"]) not in dead:
                bounds.append((i, a))

    shots, notes = [], []
    kept = lambda lo, hi: [i for i in range(lo, hi) if i not in skip and _line(msgs[i])]

    for k in range(max(0, len(bounds) - 1)):
        # Shot 1 reaches back to the top of the chat; every other starts on its own keyframe. The
        # upper bound is exclusive, so the message that SENT the next picture sets up the next shot.
        lo = 0 if k == 0 else bounds[k][0]
        shots.append({"msgs": kept(lo, bounds[k + 1][0]),
                      "start": bounds[k][1], "end": bounds[k + 1][1], "tail": False})

    # Anything past the last picture runs on text. With fewer than two pictures there are no bounded
    # shots at all, so the whole conversation is that tail (one picture still opens it as a keyframe).
    tail_from = bounds[-1][0] if len(bounds) >= 2 else 0
    tail = kept(tail_from, len(msgs))
    if tail:
        cuts = sorted(b for b in brk if tail[0] < b <= tail[-1])
        group = []
        for i in tail:
            if i in cuts and group:
                shots.append({"msgs": group, "start": None, "end": None, "tail": True})
                group = []
            group.append(i)
        shots.append({"msgs": group, "start": None, "end": None, "tail": True})

    ignored = sorted(b for b in brk if not (tail and tail[0] < b <= tail[-1]))
    if ignored:
        notes.append("breaks ignored inside a keyframed span (a boundary there would need a "
                     "keyframe): " + ", ".join(f"#{b}" for b in ignored))
    return shots, [ref for _, ref in bounds], notes


def _durations(dur, n):
    """Per-shot seconds as Storyboard's `durations` takes them: a comma list, last value repeating.
    Emitted at full length rather than relying on that repeat, so the table and the text agree."""
    vals = []
    for x in dur or ():
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if v > 0:
            vals.append(v)
    if not vals:
        vals = [DEFAULT_DURATION]
    return [vals[i] if i < len(vals) else vals[-1] for i in range(max(0, n))]


def _link_list(lnk, n):
    """Per-shot link modes for Storyboard's `links`, same last-value-repeats shape as durations.
    A shot bounded by two keyframes ignores it — a wired start frame always wins — so this only
    really speaks for the text-only shots."""
    vals = [str(x) for x in (lnk or ()) if str(x) in LINK_MODES] or [DEFAULT_LINK]
    return [vals[i] if i < len(vals) else vals[-1] for i in range(max(0, n))]


def _snapped(seconds):
    """What H3 will really run, for the report. Best-effort: the grid lives in the H3 nodes, which
    may not be importable here, and being unable to say is no reason to fail."""
    try:
        from .nodes import _frames_for, _seconds
        return _seconds(_frames_for(seconds))
    except Exception:
        return None


def _build_chain(shots, secs, lnks, beats, kf):
    """The shots as a MORPHEUS_SHOT chain, with each shot's own keyframes attached.

    This is the output worth wiring: one link instead of four parallel lists whose alignment is
    positional and therefore silent when it breaks. It also frees the keyframes from Storyboard's
    left-to-right batch rule — here a shot names its own start and end, so a text-only shot can sit
    between two keyframed ones without the boundary in the middle needing a frame of its own.

    ``prompt`` is left EMPTY on purpose: that is what tells Storyboard "this one is yours to write".
    Both frames are attached even though the sampler may only be conditioned on the end — the LLM
    has to SEE both to describe the movement between them, and Storyboard's `anchor` decides what is
    actually wired.
    """
    from .nodes import _frames_for      # lazy: importing the H3 nodes pulls in comfy + torch
    chain, k = [], 0
    for s, sec, lnk, beat in zip(shots, secs, lnks, beats):
        start = end = None
        if not s["tail"] and kf is not None:
            start, end = kf[k:k + 1], kf[k + 1:k + 2]
            k += 1
        chain.append({
            "prompt": "",
            "beat": beat,
            "frames": _frames_for(sec),
            "link": lnk if lnk in LINK_MODES else DEFAULT_LINK,
            "seed_offset": 0,
            "keyframe_strength": 0.999,
            "start_frame": start,
            "end_frame": end,
            "refine": "auto",
        })
    return chain


def _load_keyframes(refs):
    """Boundary pictures -> one IMAGE batch, or (None, notes) when there are none.

    A batch has to be uniform, and chat pictures are whatever was generated or pasted, so the FIRST
    one sets the canvas (it is the picture that decides the video's shape anyway) and the rest are
    cover-cropped to it. Anything cropped hard enough to matter is called out in the report.
    """
    notes = []
    if not refs:
        return None, notes
    paths, missing = resolve_refs(refs)
    if missing:
        # The board holds a snapshot, so a picture taken out of the chat after Update History leaves
        # a name here with nothing behind it. Say where it looked — the alternative is guessing at
        # whether the file, the folder or the reference is what went wrong.
        try:
            where = att_base()
        except Exception:
            where = "(input folder unavailable)"
        raise ValueError("keyframe file is gone: " + ", ".join(sorted(set(missing)))
                         + f"\nlooked in: {where}"
                         + "\nPress ⟳ Update History to re-read the chat — most likely the picture "
                           "was removed from the conversation after the board snapshotted it. If it "
                           "should still be there, re-run the branch that generated it: the "
                           "filename is a hash of the pixels, so the same file comes back.")

    import numpy as np
    import torch
    from PIL import Image

    frames, canvas = [], None
    for p in paths:
        with Image.open(p) as im:
            pil = im.convert("RGB")
            if canvas is None:
                canvas = pil.size
            elif pil.size != canvas:
                cw, ch = canvas
                ar, want = pil.width / pil.height, cw / ch
                if abs(ar - want) > 0.02:
                    notes.append(f"{os.path.basename(p)} is {pil.width}x{pil.height}, cropped to "
                                 f"{cw}x{ch}")
                s = max(cw / pil.width, ch / pil.height)          # cover, then centre-crop
                pil = pil.resize((max(1, round(pil.width * s)), max(1, round(pil.height * s))),
                                 Image.Resampling.LANCZOS)
                l, t = (pil.width - cw) // 2, (pil.height - ch) // 2
                pil = pil.crop((l, t, l + cw, t + ch))
            frames.append(np.asarray(pil, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(frames)), notes


class KinburgDreamBoard:
    """Chat → keyframes + beats + durations, ready for Morpheus Storyboard."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Carried by the board's own DOM widget, like the chat's chat_state — see the
                # module docstring. The frontend splices the auto-created widget out.
                "board_state": ("STRING", {"default": "", "tooltip": "The snapshot of the conversation and what you picked from it, managed by the board. You don't edit this directly."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (MORPHEUS_SHOT, "IMAGE", "STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("shots", "keyframes", "beats", "durations", "links", "shot_count", "report")
    OUTPUT_TOOLTIPS = (
        "The whole storyboard as one chain — wire it into Storyboard's 'shots' and leave its "
        "keyframes / durations / links / shot_count alone. Each shot carries its own keyframes, "
        "length, link mode and direction, and an EMPTY prompt, which is what tells Storyboard to "
        "write it. One wire instead of four lists that only line up by position. Route 'beats' into "
        "a Show Text as well if you want to read or hand-edit the directions — Storyboard's 'beats' "
        "input overrides what is in the chain, line by line.",
        "The picked pictures in conversation order — wire into Storyboard's 'keyframes'. They are "
        "consumed there as shot boundaries, which is exactly the order they are in here. None when "
        "you picked no pictures, which Storyboard reads as a text-only chain.",
        "One line per shot, in Storyboard's 'beats' format. Route it through a 'Show Text' node "
        "first if you want to read or hand-edit it before it is written up.",
        "Per-shot seconds for Storyboard's 'durations'.",
        "Per-shot link mode for Storyboard's 'links' — what a shot with no start keyframe does: "
        "'continue' inherits the previous shot's last frame, 'cut' starts fresh from text.",
        "How many shots the board came to — wire into Storyboard's 'shot_count' so the text-only "
        "shots at the end are written too (its own default would stop at the keyframes).",
        "The shot table: which pictures bound each shot, how many messages went in, and anything "
        "that was dropped or cropped.")
    FUNCTION = "build"
    CATEGORY = CAT_MORPHEUS
    DESCRIPTION = ("Turns a Local LLM Chat conversation into a Morpheus storyboard: the pictures "
                   "you pick become shot boundaries and the messages between them become each "
                   "shot's direction. Feeds Morpheus Storyboard 🌙.")

    def build(self, board_state="", unique_id=None):
        st = _parse_state(board_state)
        msgs = st["msgs"]
        shots, bounds, notes = plan_board(msgs, st["skip"], st["noimg"], st["breaks"])
        if not shots:
            raise ValueError("Nothing on the board yet — press Update History, then pick at least "
                             "two pictures (or some messages) to make a shot from.")

        keyframes, knotes = _load_keyframes(bounds)
        notes.extend(knotes)

        secs = _durations(st["dur"], len(shots))
        lnks = _link_list(st["lnk"], len(shots))
        lines = [" ".join(_line(msgs[i]) for i in s["msgs"]) for s in shots]
        beats = "\n".join(lines)
        chain = _build_chain(shots, secs, lnks, lines, keyframes)

        rows = ["shot  seconds        keyframes            link       messages"]
        for n, (s, sec, lnk) in enumerate(zip(shots, secs, lnks), 1):
            snap = _snapped(sec)
            when = f"{sec:g}" + (f" → {snap:.2f}" if snap and abs(snap - sec) > 0.005 else "")
            if s["tail"]:
                kf = "text only"
            else:
                kf = f"{os.path.splitext(s['start']['name'])[0][:10]} → " \
                     f"{os.path.splitext(s['end']['name'])[0][:10]}"
            # Shot 1 opens the sequence and a bounded shot is pinned by its frame, so `link` has
            # nothing to say in either case — showing it there would just invite fiddling.
            shown = "—" if (n == 1 or not s["tail"]) else lnk
            rows.append(f"{n:<5} {when:<14} {kf:<20} {shown:<10} {len(s['msgs'])}")
        rows.append("")
        rows.append(f"{len(shots)} shot(s), {len(bounds)} keyframe(s) from {len(msgs)} message(s)")
        if not beats.strip():
            rows.append("⚠ every beat line is empty — Storyboard will invent all of it from its "
                        "brief; pick some messages, or write a brief there.")
        rows.extend("⚠ " + n for n in notes)

        return (chain, keyframes, beats, ", ".join(f"{s:g}" for s in secs), ", ".join(lnks),
                len(shots), "\n".join(rows))


NODE_CLASS_MAPPINGS = {"KinburgDreamBoard": KinburgDreamBoard}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgDreamBoard": "Morpheus Dream Board 🌙"}
