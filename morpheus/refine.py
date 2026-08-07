"""In-loop prompt refinement: the writer finally gets to see the frame the shot really starts on.

`Morpheus Storyboard` writes every shot before anything is sampled, so a shot whose first frame is
inherited is written against a **forecast** — the previous shot's `[END STATE]` sentence, which is the
writer's own guess about a frame that does not exist yet. When the guess and the rendered frame
disagree, the shot is conditioned on one state and told about another, and the model splits the
difference.

Putting a vision LLM inside the sampler's loop removes the guess: by the time shot N is about to be
sampled, shot N-1 has been decoded, so the real first frame is in hand — together with the planned
last keyframe, if there is one. That turns every continuing shot from "text mode, blind" into the
two-image job, which is the mode that works best.

Two scopes, because rewriting everything throws away work that was fine:

* **opening** (the default) — only `[Scene Overview]` and the first beat are rewritten. The forecast
  was the *only* thing wrong: the pacing, the camera, the audio and the target were planned against
  real information. ~80 tokens of generation instead of ~350, and the plan survives verbatim.
* **full** — the five shot-owned blocks are written from scratch, keeping the style and negative
  sections of the existing prompt. For shots that never had a real prompt to begin with.

Sections 1 and 6 are never touched: the style bible has to stay byte-identical across the whole
storyboard or the look drifts mid-video.
"""
import re

from . import cache as shot_cache
from .storyboard import (ANCHOR_CLAUSE, KinburgMorpheusStoryboard, SHOT_SYSTEM_MORPH,
                         SHOT_SYSTEM_START, _LABELS_SHOT, _blocks, dedupe_dialogue)

SCOPES = ["auto", "off", "opening", "full"]

REFINE_OPENING_SYSTEM = """You are a director of photography correcting ONE detail of a shot prompt for the MiniMax H3 video model.

The shot was written before it was filmed, so its opening was a guess. You are now given the frame the shot ACTUALLY starts on (and, if there is one, the frame it must end on). The rest of the prompt — pacing, camera, sound, target — was planned properly and is not yours to change.

Rewrite only the opening, to match what is really in the first image:
- Read the first image: subject, wardrobe, materials, pose, framing, light.
- Say what is ALREADY true there. Never describe that state being arrived at, and never mention anything that happened before it.
- The shot opens MID-MOVEMENT: state that the subject is already travelling at speed and keep it steady. Never write "begins to", "starts to", "picks up speed" or "accelerates from".
- Keep the wording, subject names and register of the existing prompt. Change no other section.
- If the beat you are replacing contained a spoken line, keep that line in the beat, once, with its voice specification — verbatim, in whatever language and alphabet it was written in. Never translate it, never move it elsewhere, never duplicate it.

Whatever language the input is in, answer in ENGLISH — except for a spoken line, which keeps its own language and alphabet, verbatim. Use EXACTLY these two labelled blocks and nothing else — no preamble, no commentary:

[SITUATION]: one sentence — what is already true as the shot opens, naming who or what is on screen and where.
[BEAT 1]: the replacement text for the first beat only, without its timing bracket — what happens in the shot's first stretch, starting from the state above."""

_SECTION = re.compile(r"^[ \t]*(\d)\.[ \t]*\[([^\]]+)\][ \t]*:?[ \t]*", re.M)
_BEAT_LINE = re.compile(r"^(\s*\[[^\]]*\][^:\n]*:)(.*)$", re.M)


# ------------------------------------------------------------------------------------ prompt surgery
def split_sections(prompt):
    """{index: (title, body)} for an assembled six-section prompt, {} if it isn't one."""
    hits = list(_SECTION.finditer(prompt or ""))
    out = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(prompt)
        out[int(m.group(1))] = (m.group(2).strip(), prompt[m.end():end].strip())
    return out


def _join(sections):
    return "\n\n".join(f"{i}. [{sections[i][0]}]: {sections[i][1]}"
                       if not sections[i][1].startswith("[")
                       else f"{i}. [{sections[i][0]}]:\n{sections[i][1]}"
                       for i in sorted(sections))


def swap_first_beat(storyboard, new_text):
    """Replace the text of the first beat line, keeping its timing bracket and label."""
    new_text = (new_text or "").strip()
    if not new_text:
        return storyboard
    # a model that echoed the whole line ("[0s-2s] Beat 1: x") keeps only its payload
    m = _BEAT_LINE.match(new_text)
    if m:
        new_text = m.group(2).strip()
    if not _BEAT_LINE.search(storyboard or ""):
        return storyboard
    return _BEAT_LINE.sub(lambda mm: f"{mm.group(1)} {new_text}", storyboard, count=1)


def splice_opening(prompt, situation, beat1):
    """New `[Scene Overview]` + first beat, everything else verbatim."""
    sec = split_sections(prompt)
    if 2 not in sec:
        return None
    if situation:
        sec[2] = (sec[2][0], (situation.strip() + " " + ANCHOR_CLAUSE).strip())
    if 3 in sec and beat1:
        sec[3] = (sec[3][0], swap_first_beat(sec[3][1], beat1))
        # a rewritten beat can introduce a line the audio block already carries
        if 5 in sec:
            cleaned = dedupe_dialogue(sec[3][1], sec[5][1])
            if cleaned:
                sec[5] = (sec[5][0], cleaned)
            else:
                sec.pop(5)
    return _join(sec)


def rebuild_full(prompt, body, anchored):
    """Shot-owned sections replaced from a five-block answer; style and negative kept as they were."""
    sec = split_sections(prompt)
    style = sec.get(1, ("Style and Aesthetic", ""))[1]
    negative = sec.get(6, ("Negative Prompt/Constraints", ""))[1]
    scene = body.get("SITUATION", "").strip()
    if anchored and scene:
        scene = scene + " " + ANCHOR_CLAUSE
    out = {}
    if style:
        out[1] = ("Style and Aesthetic", style)
    if scene:
        out[2] = ("Scene Overview", scene)
    if body.get("STORYBOARD"):
        out[3] = ("Storyboard", body["STORYBOARD"].strip())
    if body.get("CAMERA"):
        out[4] = ("Camera", body["CAMERA"].strip())
    audio = dedupe_dialogue(body.get("STORYBOARD", ""), body.get("AUDIO", ""))
    if audio:
        out[5] = ("Audio & Voice", audio)
    if negative:
        out[6] = ("Negative Prompt/Constraints", negative)
    return _join(out) if out else None


# --------------------------------------------------------------------------------------- scope logic
def resolve_scope(declared, prompt, start_was_forecast, has_start):
    """The scope this shot actually runs at.

    `auto` follows what the writer could know: a first frame it never saw (an inherited tail) means
    the opening is a guess and gets corrected; a prompt with no assembled sections was never really
    written, so it gets written now. Everything else is left alone."""
    declared = declared if declared in SCOPES else "auto"
    if declared != "auto":
        return declared
    if not split_sections(prompt or ""):
        return "full" if has_start else "off"
    return "opening" if start_was_forecast else "off"


# -------------------------------------------------------------------------------------- the call
def refine_prompt(cfg, prompt, scope, start_img, end_img, seconds, direction="",
                  unload_comfy=False, emit=None, tag="refine", cache_key=None, use_cache=True):
    """(new_prompt, error) — `new_prompt` is None when nothing changed.

    The answer is cached under `cache_key` because the prompt IS the sampler's cache key: text that
    came out even one character different on a re-run would re-sample a five-minute shot."""
    # nothing to look at means nothing to correct — the whole point is reading a real frame
    if scope not in ("opening", "full") or (start_img is None and end_img is None):
        return None, None

    hit = shot_cache.load_json(cache_key) if (use_cache and cache_key) else None
    if hit and hit.get("prompt"):
        if emit:
            emit({"event": "start", "label": tag + " (cached)"})
            emit({"event": "done", "text": hit["prompt"], "label": tag + " (cached)"})
        return hit["prompt"], None

    imgs = [im for im in (start_img, end_img) if im is not None]
    ask = KinburgMorpheusStoryboard._ask  # one LLM seam for both nodes
    if scope == "opening":
        parts = [f"The shot is {seconds:.2f} seconds long.",
                 "The first image is the frame this shot really starts on."
                 + (" The second image is the frame it must end on." if end_img is not None else ""),
                 "This is the prompt as it was planned:\n" + (prompt or ""),
                 "Rewrite only its opening so it matches the first image."]
        text, err = ask(cfg, REFINE_OPENING_SYSTEM, "\n\n".join(parts), imgs, unload_comfy, tag,
                        emit)
        if err:
            return None, err
        b = _blocks(text, ["SITUATION", "BEAT 1"])
        if not b:
            return None, "the answer had no [SITUATION] / [BEAT 1] labels"
        out = splice_opening(prompt, b.get("SITUATION", ""), b.get("BEAT 1", ""))
        if out is None:
            return None, "the existing prompt has no [Scene Overview] section to replace"
    else:
        system = SHOT_SYSTEM_MORPH if end_img is not None else SHOT_SYSTEM_START
        parts = [f"Shot length: {seconds:.2f} seconds. The beats must span 0s to {seconds:.2f}s.",
                 ("Direction for this shot: " + direction) if direction else "",
                 "The images are this shot's real first frame"
                 + (" and the frame it must end on." if end_img is not None else "."), ]
        text, err = ask(cfg, system, "\n\n".join(p for p in parts if p), imgs, unload_comfy, tag,
                        emit)
        if err:
            return None, err
        body = _blocks(text, _LABELS_SHOT)
        if not body.get("STORYBOARD"):
            return None, "the answer had no [STORYBOARD] label"
        out = rebuild_full(prompt, body, anchored=True)
        if out is None:
            return None, "nothing usable came back"

    if use_cache and cache_key and out:
        shot_cache.save_json(cache_key, {"prompt": out, "scope": scope})
    return out, None
