"""Morpheus Storyboard — one idea (+ any keyframes you happen to have) in, a whole chain of dreams out.

This is the LLM half of Morpheus: it writes one MiniMax-format prompt per shot and hands the
finished `MORPHEUS_SHOT` chain to the sampler, so the manual loop of "crop keyframes → ask the VLM
about each pair → paste into N Dream nodes" collapses into one node.

**Keyframes are consumed as shot boundaries, left to right.** That single rule covers every mix the
author asked for, with no modes to pick:

    K frames, N shots → boundary i is known while i < K
      K = 0            every shot text-only (shot 1 is pure t2v)
      K = 1            shot 1 starts on the frame, the rest are text-only fantasy
      K = N + 1        every shot bounded by two hard frames
      1 < K < N + 1    the first K-1 shots are bounded, then it runs on text

The prompts follow the format MiniMax itself recommends (numbered [Style] / [Scene Overview] /
[Storyboard] / [Camera] / [Audio & Voice] / [Negative] sections, with `[0s-1.5s] Shot n:` beat lines
inside the shot). Two things about it are worth knowing, because they shape the whole node:

  * H3 reads those timings as **pacing and order, not as a clock** — nothing binds "at 2.0s" to a
    frame. So beats are generated, exact seconds are never promised, and real timing control comes
    from where the SHOT boundaries fall.
  * the [Style] / [Scene Overview] / [Negative] blocks must be **identical in every shot** or the
    look drifts mid-sequence. So they are written once by a text-only "style bible" call and then
    stamped onto every shot; only the storyboard/camera/audio blocks are per shot.

Continuity without keyframes comes from the shots themselves: every call also returns an
`[END STATE]` line describing its own last frame, which is fed to the next shot as its starting
situation. That is what keeps a text-only chain from wandering off.

Everything the LLM writes is cached on disk under a causal key (see `cache.py`), because the prompts
ARE the sampler's cache key: a prompt that changes on every run would re-sample every shot, at
several minutes each.
"""
import json
import logging
import os
import re
import time

import torch

import comfy.utils

from . import cache as shot_cache
from .nodes import (MORPHEUS_SHOT, KinburgMorpheusDream, LINK_MODES, _frames_for, _log_uris,
                    _require_h3, _seconds, h3)
from ..local_llm.llm_node import (LLM_CONFIG, PLACEHOLDER, UNLOAD_MODES, _generate_and_format,
                                  _resolve_path, _shutdown_worker, build_llm_request,
                                  resolve_unload)
from ..timer.timer_nodes import _format_elapsed

ANCHOR_MODES = ["continuous", "plan"]
CACHE_MODES = ["disk", "off"]
MAX_SHOTS = 64

STYLE_SYSTEM = """You are a film director writing the STYLE BIBLE for a short video sequence that will be generated one shot at a time. These blocks are stamped onto EVERY shot's prompt unchanged, so they may only contain what is true in every single shot.

CRITICAL: never describe the sequence's story, its beginning, its ending, or the stages of any change. Each shot is generated separately, and a prompt that mentions the whole arc makes the model replay that arc inside one shot.

Answer with EXACTLY these four labelled blocks, in this order, and nothing else:

[STYLE]: one paragraph, look and craft only — genre or reference, lens, depth of field, lighting, colour grade, grain/texture, atmosphere. No story, no camera moves, no shot list.
[SUBJECT]: one or two sentences of INVARIANTS only — who or what is on screen, their identity and wardrobe, the vehicle or props, the location and time of day. Written as a standing description ("a professional cyclist in a navy skinsuit on a sunlit coastal highway"), never as a story and never naming a start or end state.
[AUDIO BED]: one sentence for the constant sound world every shot shares (ambience plus score character).
[NEGATIVE]: a comma-separated list of things to avoid. Always include text, subtitles, logos and watermarks; add only faults — blur, artifacts, distorted anatomy, extra limbs, style breaks, a camera that shakes when it shouldn't. NEVER list anything the sequence is supposed to DO: if the subject transforms, words like "morphing", "character changes", "shape changes" or "transformation" must not appear here, and neither must a change of vehicle, wardrobe or scenery that the story requires. Read the brief and work out what genuinely must not happen.

Write in English, plainly, no markdown emphasis, no commentary."""

SCRIPT_SYSTEM = """You are a director breaking a sequence into shots, for a text-to-video model that generates one shot at a time and never sees the others.

Your output is the DIRECTION each shot's writer will work from, so each line has to stand completely on its own: someone reading only that line must know what to film. Concrete and visual, never a label — "the change spreads" is useless, "the obsidian plates lock across his shoulders and the frame beneath him thickens into a chassis" is the job.

Rules:
- One numbered line per shot, in order, two or three sentences each. Present tense.
- Each line says what HAPPENS in that shot and, at the end, the state it must ARRIVE at.
- Divide the arc in proportion to the shot lengths you are given. A ten-second shot carries roughly twice the change of a five-second one.
- Spend the whole sequence: the last shot lands on the brief's endpoint, and no earlier shot may get there first. If a transformation completes in shot 2 of 5, the plan is wrong.
- Never write "then", "next", "finally", "meanwhile" or "in this part": those words belong to a story told in one go, and each of these lines is read alone.
- Never summarise the whole arc inside a line, and never mention what happened before that shot — only what is true in it.
- Keep one subject, one location and one continuous action unless the brief asks for a cut.

Whatever language the brief is in, answer in ENGLISH — except for spoken words: if the brief quotes a line of dialogue, carry it into the shot that says it verbatim, in its own language and alphabet, and never translate it. The video model can speak those languages.

Answer with EXACTLY one labelled line per shot and nothing else — no preamble, no headings, no commentary:

[SHOT 1]: ...
[SHOT 2]: ..."""

# The per-shot instructions come in three flavours, picked by how many keyframes the shot actually
# gets. They are genuinely three different jobs — "describe the morph between these two given
# states", "carry on from this one state", "invent it from a line of text" — and folding them into
# one prompt with conditionals is what makes a small local model hedge and pad. The author tested
# the split by hand before it was built here.
_SHOT_ROLE = """You are an expert AI video prompt engineer, a director of photography and a voice director, writing for the multimodal MiniMax H3 model, which generates picture and sound together.

Whatever language the input is in, your output MUST be entirely in ENGLISH — the model works best that way. Answer with the labelled blocks and nothing else: no preamble, no explanation of what you did, no closing remarks.

THE ONE EXCEPTION IS SPOKEN WORDS. The model speaks other languages, so a line of dialogue keeps the language it was given in, verbatim, character for character: if the direction quotes what someone says, copy it exactly, in its own alphabet, and never translate, transliterate or "correct" it. When a line is not in English, name its language in the voice specification where the accent goes — (Female, 30s, quiet and unsteady, stunned, Russian, slow). If the direction only says that someone speaks without quoting the words, write the line yourself in the language the direction itself is written in.

A spoken line belongs to EXACTLY ONE place: the beat in which it is said. Written in two blocks, it gets spoken twice."""

_SHOT_TAIL = """Answer with EXACTLY these five labelled blocks, in this order, and nothing else. (Style, look and the negative list are added downstream — do not write them.)

[SITUATION]: one sentence — what is ALREADY true as this shot opens, naming who or what is on screen and where. Present tense. No history, no outcome beyond this shot.
[STORYBOARD]: the beat lines, each formatted `[<from>s-<to>s] Beat <n>: <what happens>`, together spanning 0s to the shot length you are given. These are beats of ONE continuous take, not separate shots: no cuts, no scene changes. SPEECH LIVES HERE: if anyone speaks, put the line inside the beat where it is spoken, once, as `(gender, age, tone, emotion, accent, pacing) "the exact words"` — e.g. `[2.5s-5.17s] Beat 2: he leans into the wind and says (Male, 30s, deep and gravelly, breathless, American, hurried) "They found me."` — and a line the direction gave in another language stays in that language, with the language in the accent slot: `… says (Female, 30s, flat, resigned, Russian, slow) "Он не придёт."`
[CAMERA]: one sentence — angle, framing and movement, in professional terms (low-angle tracking, slow push in, locked-off medium).
[AUDIO]: one or two sentences — ambience, Foley and score ONLY. Never repeat a spoken line here and never quote dialogue here: it already has its place and its moment in the storyboard, and a line written twice gets performed twice. Write "No dialogue." only when nobody speaks in this shot.
[END STATE]: one sentence describing this shot's LAST frame as a still photograph, precise enough that the next shot can start from it — and end it by naming the MOTION at that instant: what is moving, how fast and in which direction (or that everything is at rest).

Write plainly, present tense, no markdown emphasis."""

_SHOT_RULES = """- The model reads timings as PACING AND ORDER, not as a clock. Use beat lines, but never depend on an exact second.
- Two or three beats for a shot of about five seconds, covering its full length in order, with the LAST beat landing on the shot's final image.
- Describe what is SEEN and what CHANGES, in professional filmmaking terms. No story voice, no "we then see", no shot numbers other than the beat labels.
- This shot is a SLICE, not a summary: never describe the sequence's overall arc, never restate where it began, never re-do a change that has already happened. The model acts out anything you mention, so naming an earlier stage makes it play that stage again.
- The shot opens MID-MOVEMENT. A still first frame tells the model where things are but not how fast they are going, so it will ease the motion in from rest unless you say otherwise: state in the first beat that the subject is already travelling at speed, and keep that speed steady. Never write "begins to", "starts to move", "picks up speed" or "accelerates from" in the opening beat, and give the camera the same treatment — it is already moving with the subject."""

# two images: the bridge between them
SHOT_SYSTEM_MORPH = f"""{_SHOT_ROLE}

You are given TWO images: this shot's FIRST frame and its LAST frame, both fixed and non-negotiable. Your whole job is the bridge between them over the length you are told.

Read both images before writing: subject and wardrobe, materials, environment, light and colour, lens and angle. Then write what visibly changes from one to the other — pose, form, surface, position in frame — as one continuous movement. Invent no new characters and no new location: nothing may contradict the frames.

{_SHOT_RULES}
- The first beat opens on the state that is ALREADY in the first image — never show it being arrived at.
- The last beat lands exactly on the second image, neither short of it nor past it.

{_SHOT_TAIL}

### EXAMPLE (a 5.17 s slice, first image: a rider mid-ford; last image: the same rider on the gravel bank)
[SITUATION]: A rider in a soaked leather coat is already halfway across the flooded ford, water breaking white around the horse's chest.
[STORYBOARD]:
[0s-2.5s] Beat 1: The horse drives forward against the current, the rider low over its neck, spray rising into the low sun.
[2.5s-5.17s] Beat 2: The horse finds the shingle and climbs out, water sheeting off its flanks, until the rider straightens on the gravel bank.
[CAMERA]: Low tracking shot from the bank, holding a medium profile as the horse rises out of the water.
[AUDIO]: Churning water and hooves striking shingle, a low string bed swelling as they reach the bank. No dialogue.
[END STATE]: The rider sits upright on the dripping horse on the gravel bank, the ford behind them, morning sun full on their face; the horse is still walking forward at a steady pace, away from the water."""

# one image: carry on from it
SHOT_SYSTEM_START = f"""{_SHOT_ROLE}

You are given ONE image: this shot's FIRST frame, fixed. Describe what happens over the length you are told, starting from exactly that state. There is no target image — the shot ends wherever the action naturally reaches.

Read the image before writing: subject and wardrobe, materials, environment, light and colour, lens and angle. Keep every one of them consistent for the whole shot; invent no new characters and no new location.

{_SHOT_RULES}
- The state in the image is already reached: never show it being arrived at, and never revert to anything before it.

{_SHOT_TAIL}

### EXAMPLE (a 5.17 s slice, image: a rider on a gravel bank beside a ford)
[SITUATION]: A rider in a soaked leather coat sits upright on a dripping horse on the gravel bank, the ford behind them.
[STORYBOARD]:
[0s-2.5s] Beat 1: The horse shifts its weight and turns towards the treeline, the rider gathering the reins as water runs off the stirrups.
[2.5s-5.17s] Beat 2: They move off at a walk that lengthens into a trot, the low sun throwing their shadow long across the shingle.
[CAMERA]: Locked-off medium wide that eases into a slow pan as the horse leaves frame right.
[AUDIO]: Dripping water, gravel shifting under hooves, the river steady behind. A single low cello note as they move off. No dialogue.
[END STATE]: The horse is on the shingle track towards the trees, seen three-quarters from behind, the ford small in the background; it is moving away from camera at a steady trot."""

# no images: invent it from the direction (plus the previous shot's end state, when there is one)
SHOT_SYSTEM_TEXT = f"""{_SHOT_ROLE}

There are no images. Write the shot from the direction you are given, and — if a starting state is described to you in words — begin from exactly that state and move forward from it.

{_SHOT_RULES}
- Name the subject, the wardrobe and the setting explicitly in [SITUATION]: with no image, nothing else is holding this shot's identity together.

{_SHOT_TAIL}

### EXAMPLE (a 5.17 s slice, direction: "she reads the telegram and understands")
[SITUATION]: A woman in a grey travelling coat stands alone on an empty platform, a telegram open in both hands, gaslight above her.
[STORYBOARD]:
[0s-2s] Beat 1: Her eyes move down the page and stop; her hands lower a fraction as the paper goes slack.
[2s-5.17s] Beat 2: She lifts her head slowly, breath clouding, and says (Female, 30s, quiet and unsteady, stunned, English, slow) "He isn't coming.", then stares down the empty track while steam drifts across the lamps behind her.
[CAMERA]: Slow push in from a medium to a tight medium, settling as she raises her head.
[AUDIO]: Distant steam venting, a station clock ticking, one restrained piano figure entering as she looks up.
[END STATE]: A tight medium of the woman looking off down the track, telegram hanging at her side, gaslight catching the side of her face; she is motionless apart from her breath, the steam still drifting behind her."""

# "SCENE" stays accepted so a bible pasted from an older run still parses.
_LABELS_STYLE = ["STYLE", "SUBJECT", "SCENE", "AUDIO BED", "NEGATIVE"]
_LABELS_SHOT = ["SITUATION", "STORYBOARD", "CAMERA", "AUDIO", "END STATE"]

# Appended to a shot's negative list when its first frame is fixed. The model will otherwise happily
# rewind and replay the transformation it has just been handed the middle of.
ANCHOR_NEGATIVE = ("restarting the sequence from the beginning, reverting to an earlier stage, "
                   "repeating a change already visible in the first frame, cutting away to a "
                   "different subject or location")
# Short on purpose: the model reads this as content, not as instruction, so every extra word is
# something it tries to put on screen. The heavy lifting is done by ANCHOR_NEGATIVE.
ANCHOR_CLAUSE = "Continues from the first frame; the action moves forward only."


# ------------------------------------------------------------------------------------- text utils
def _blocks(text, labels):
    """{LABEL: body} out of a labelled answer; a body runs until the next known label.

    Tolerant on purpose: local models like to decorate labels (`**[Storyboard]**:`, `3. STORYBOARD -`),
    and losing a block to a stray asterisk would cost a whole shot."""
    if not text:
        return {}
    alt = "|".join(re.escape(x).replace(r"\ ", r"[\s_]+") for x in sorted(labels, key=len, reverse=True))
    # leading decoration, then the label, then trailing decoration: `**[Storyboard]**:` and
    # `3. CAMERA -` have to land on the same footing, or a stray asterisk ends up in the prompt
    pat = re.compile(r"^[\s>*#\-\d.)]*\[?\s*(" + alt + r")\s*\]?\s*\**\s*[:\-–]?[ \t]*", re.I | re.M)
    hits = list(pat.finditer(text))
    out = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        key = re.sub(r"[\s_]+", " ", m.group(1)).upper()
        body = text[m.end():end].strip()
        if body and key not in out:
            out[key] = body
    return out


_SHOT_LABEL = re.compile(r"^[\s>*#\-]*\[?\s*(?:SHOT|BEAT|PART)\s*(\d+)\s*\]?\s*[:.\-–)]\s*",
                         re.I | re.M)


def _parse_script(text, n):
    """`[SHOT 1]: …` lines → one direction per shot, whitespace collapsed to a single line each.

    Falls back to plain non-empty lines in order, because a model that drops the labels has usually
    still written the right N lines."""
    hits = list(_SHOT_LABEL.finditer(text or ""))
    out = [""] * n
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        idx = int(m.group(1)) - 1
        body = " ".join(text[m.end():end].split())
        if 0 <= idx < n and body and not out[idx]:
            out[idx] = body
    if not any(out):
        for i, ln in enumerate([x for x in (text or "").splitlines() if x.strip()][:n]):
            out[i] = " ".join(ln.split())
    return out


_QUOTED = re.compile(r"[\"“”«]([^\"“”«»]{3,300})[\"“”»]")


def dedupe_dialogue(storyboard, audio):
    """Strip from the audio block any sentence that repeats a line already spoken in a beat.

    MiniMax's own template puts voice-over in [Audio & Voice], but measured on real renders the line
    lands far better inside the beat that speaks it — with its timing — and a line present in BOTH
    blocks is sometimes performed TWICE. The prompts say so; this is the net for when the model does
    it anyway. Only whole sentences are dropped, so what is left still reads."""
    sb, au = storyboard or "", (audio or "").strip()
    if not sb or not au:
        return au
    lines = [m.group(1).strip() for m in _QUOTED.finditer(sb)]
    lines = [ln for ln in lines if len(ln) >= 3]
    if not lines:
        return au
    # Sentence boundaries have to allow a terminator INSIDE a quote (`… me." Next sentence`) or a
    # quoted line swallows what follows it, and must NOT fire mid-sentence (`shouts "Run!" here.`) or
    # the drop leaves a fragment behind — so a boundary also needs the next word to start a sentence.
    kept = []
    for sentence in re.split(r'(?<=[.!?]["”»\')\]])\s+(?=["“«(]?[A-ZА-ЯЁ])'
                             r'|(?<=[.!?])\s+(?=["“«(]?[A-ZА-ЯЁ])', au):
        low = sentence.lower()
        if any(ln.lower().rstrip(" .!?,") in low for ln in lines):
            continue
        kept.append(sentence)
    return " ".join(s.strip() for s in kept if s.strip()).strip()


def _beat_lines(text):
    """One line per shot, BLANKS KEPT — they are what "leave this shot to the LLM" looks like.
    Dropping empty lines would silently shift every note onto the wrong shot."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _split_overrides(text):
    """Shot prompts separated by a line of only dashes. Blank blocks mean "keep the LLM's"."""
    if not (text or "").strip():
        return []
    return [b.strip() for b in re.split(r"(?m)^\s*-{3,}\s*$", text)]


def _durations(text, n):
    """"5.2" → the same for every shot; "5.2, 6, 5.2" → per shot, last value repeating."""
    vals = []
    for tok in re.split(r"[,;\s]+", (text or "").strip()):
        if not tok:
            continue
        try:
            vals.append(max(0.2, float(tok)))
        except ValueError:
            raise ValueError(f"durations: '{tok}' is not a number (use '5.2' or '5.2, 6, 5.2')")
    if not vals:
        vals = [5.2]
    return [vals[min(i, len(vals) - 1)] for i in range(n)]


def _frame_list(images):
    if images is None or not hasattr(images, "ndim"):
        return []
    if images.ndim == 3:
        return [images[None, ...]]
    return [images[i:i + 1] for i in range(int(images.shape[0]))]


def _cfg_fingerprint(cfg):
    try:
        return shot_cache.key(json.dumps(cfg, sort_keys=True, default=str))
    except Exception:
        return shot_cache.key(str(sorted((cfg or {}).items(), key=lambda kv: str(kv[0]))))


# ============================================================================================ node
class KinburgMorpheusStoryboard:
    """Idea + optional keyframes → a full chain of Morpheus Dreams, written by a local LLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (LLM_CONFIG, {"tooltip": "A 'Local LLM Settings (GGUF)' bundle. Attach a 'Vision Settings (GGUF)' (mmproj) too if you wire keyframes — without it the images are ignored and the shots are written from text alone."}),
                "brief": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                     "tooltip": "The idea, in your own words: what happens across the whole sequence. One or two sentences is enough — the style bible and the per-shot prompts are written from it."}),
                "shot_count": ("INT", {"default": 0, "min": 0, "max": MAX_SHOTS,
                                       "tooltip": "How many shots to write. 0 = derive it from the keyframes (K frames → K-1 bridging shots). Set it higher than K-1 to keep going on text alone after the keyframes run out, or use it on its own with no keyframes at all."}),
                "durations": ("STRING", {"default": "5.17",
                                         "tooltip": "Seconds per shot: one value for all of them, or a comma list ('5.2, 6, 5.2') where the last value repeats. Each is snapped to H3's 0.71 s grid by the Dream node, and the LLM is told the snapped length so its beats cover the real duration."}),
                "anchor": (ANCHOR_MODES, {"default": "continuous", "advanced": True,
                                          "tooltip": "What a shot's FIRST frame is wired to when a keyframe exists for it:\n\n• continuous — only the END keyframe is wired; the shot starts from the previous shot's generated tail. Seams are exactly continuous, and the shot is still pulled to its planned keyframe by its end. Best default.\n\n• plan — both keyframes are wired, like doing it by hand. Tighter adherence to the storyboard, at the price of a small jump at each seam (a generated tail is never bit-identical to the re-encoded keyframe).\n\nEither way the LLM SEES both frames — this only changes what the sampler is conditioned on."}),
                "link": (LINK_MODES, {"default": "continue",
                                      "tooltip": "What shots with no start keyframe do: 'continue' inherits the previous shot's last frame (one flowing take), 'cut' starts fresh from text (a montage of hard cuts)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "LLM seed. Deliberately NOT 'control after generate': the prompts are the sampler's cache key, so a seed that moved every run would re-sample every shot at minutes apiece. Change it to re-roll all the writing."}),
                "cache": (CACHE_MODES, {"default": "disk", "advanced": True,
                                        "tooltip": "Cache the LLM's answers on disk (keyed causally, like the sampler's latents), so re-running the graph doesn't rewrite prompts and invalidate finished shots. Editing one shot's beat re-rolls that shot and the ones after it, because the end-state carries forward."}),
                "unload_after_run": (UNLOAD_MODES, {"default": "unload after run", "advanced": True,
                                                    "tooltip": "Free the LLM from VRAM when this node is done. Defaults to unloading because what runs next is H3 plus a 30B text encoder."}),
                "script": (["auto", "off"], {"default": "auto",
                                            "tooltip": "auto = before writing any shot, break the brief into ONE DIRECTION PER SHOT with a text-only call, and use those as the shots' direction. This is what stops the brief being handed to every shot whole — a shot told the entire story replays the entire story. Skipped automatically when you fill 'beats' yourself (your lines win).\n\noff = no plan. Shots then take their direction only from 'beats', their keyframes and the previous shot's end state — fine for a chain where every shot is bounded by two keyframes, thin for a text-driven one.\n\nThe plan comes out on the 'script' output in exactly the format 'beats' takes: edit a line, paste it back, and only that shot onwards is rewritten."}),
                "live_preview": ("BOOLEAN", {"default": True,
                                             "tooltip": "Stream every call to a 'Kinburg Live Log' node as it's written, one labelled block per call ('style bible', 'shot 2/4'), with the frames it was shown. Drop a Kinburg Live Log anywhere on the canvas — no wiring. On by default: writing a storyboard is minutes of otherwise silent work."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "keyframes": ("IMAGE", {"tooltip": "Keyframes in order, as one IMAGE batch — used as SHOT BOUNDARIES left to right: frame 1 starts shot 1, frame 2 ends shot 1 and starts shot 2, and so on. Give as many or as few as you have: none (all text), one (a hard opening then free fantasy), K-1 shots' worth, or all of them."}),
                "beats": ("STRING", {"multiline": True, "dynamicPrompts": False, "advanced": True,
                                     "tooltip": "Optional direction, ONE LINE PER SHOT ('shot 3: he crashes through a billboard'). Lines are matched to shots by position; blank or missing lines leave that shot to the LLM. This is the main handle on a text-only chain."}),
                "prompt_overrides": ("STRING", {"multiline": True, "dynamicPrompts": False, "advanced": True,
                                                "tooltip": "Finished prompts that BYPASS the LLM, separated by a line of three dashes (---), in shot order; an empty block keeps the LLM's version. The 'prompts' output uses the same format, so the loop is: run once, read it, fix the one shot that came out wrong, paste it back."}),
                "style": ("STRING", {"multiline": True, "dynamicPrompts": False, "advanced": True,
                                     "tooltip": "Skip the style-bible call and use these blocks verbatim: paste back a previous run's 'style' output (or hand-write [STYLE]: / [SCENE]: / [AUDIO BED]: / [NEGATIVE]: blocks). This is how you keep one look across several runs."}),
                "shots": (MORPHEUS_SHOT, {"tooltip": "An existing chain to append to — hand-build the opening shot with 'Morpheus Dream' and let this node write the rest."}),
                "style_system": ("STRING", {"multiline": True, "dynamicPrompts": False, "forceInput": True,
                                            "tooltip": "Override the built-in system prompt for the style-bible call. Blank = the default, which asks for [STYLE] / [SCENE] / [AUDIO BED] / [NEGATIVE]."}),
                "shot_system_morph": ("STRING", {"multiline": True, "dynamicPrompts": False, "forceInput": True,
                                                 "tooltip": "Override the system prompt used when a shot has BOTH keyframes ('describe the change that carries the first image into the second'). Blank = built-in default. All three shot prompts must ask for the same five labelled blocks — [SITUATION] / [STORYBOARD] / [CAMERA] / [AUDIO] / [END STATE] — or the answer can't be assembled."}),
                "shot_system_start": ("STRING", {"multiline": True, "dynamicPrompts": False, "forceInput": True,
                                                 "tooltip": "Override the system prompt used when a shot has only its FIRST keyframe ('carry on from this state; there is no target'). Blank = built-in default."}),
                "shot_system_text": ("STRING", {"multiline": True, "dynamicPrompts": False, "forceInput": True,
                                                "tooltip": "Override the system prompt used when a shot has NO images ('invent it from the direction, and from the previous shot's end state if there is one'). Blank = built-in default."}),
                "script_system": ("STRING", {"multiline": True, "dynamicPrompts": False, "forceInput": True,
                                             "tooltip": "Override the system prompt for the planning call. Blank = the default, which asks for one numbered line per shot, two or three concrete sentences each, ending on the state that shot must reach."}),
            },
        }

    RETURN_TYPES = (MORPHEUS_SHOT, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("shots", "prompts", "style", "report", "script")
    OUTPUT_TOOLTIPS = ("The finished chain — wire it straight into 'Morpheus (Video Sampler)'.",
                       "Every prompt, in the same ---separated format 'prompt_overrides' takes.",
                       "The style bible, to paste into 'style' on a later run for the same look.",
                       "Per-shot table: duration, which keyframes were used, where the text came from.",
                       "The plan: one direction per shot, one line each — exactly the format 'beats' takes. Edit a line and paste it back and only that shot onwards is rewritten; paste it back unchanged and nothing is.")
    FUNCTION = "write"
    CATEGORY = "Kinburg-Nodes/sampling"
    DESCRIPTION = ("Writes a whole Morpheus storyboard with a local LLM: one MiniMax-format prompt "
                   "per shot, keyframes consumed as shot boundaries, continuity carried between "
                   "shots by an end-state line. Outputs the chain the video sampler takes.")

    # -------------------------------------------------------------------------------- LLM plumbing
    @staticmethod
    def _ask(cfg, system, user_prompt, images, unload_comfy, tag, emit=None):
        """One call through the same path the Local LLM node uses. Returns (text, error).

        `emit`, when given, streams to a `Kinburg Live Log` node over the same `kinburg.llm` channel
        the Local LLM node uses — one labelled block per call, so a four-shot run reads as
        "style bible", "shot 1/4", … instead of two silent minutes."""
        call_cfg = dict(cfg)
        call_cfg["system_prompt"] = system
        call_cfg["context"] = ""
        call_cfg["output_format"] = "text"
        call_cfg["grammar"] = ""
        call_cfg["strip_think"] = True
        call_cfg["max_tokens"] = max(int(cfg.get("max_tokens", 512) or 512), 900)
        call_cfg["seed"] = cfg.get("seed", 0)

        image = None
        if images:
            image = images[0] if len(images) == 1 else torch.cat(images, dim=0)
        err, ctx = build_llm_request(call_cfg, user_prompt, image=image)
        if err:
            if emit:
                emit({"event": "done", "text": err, "label": tag})
            return "", err

        token_cb, stats = None, None
        if emit:
            ctx["req"]["stream_text"] = True
            stats = {}

            def token_cb(delta):
                emit({"event": "delta", "delta": delta})

            # the frames this call was actually shown, so the log answers "what did it look at?"
            emit({"event": "start", "label": tag, "max_tokens": int(ctx["max_tokens"]),
                  "n_ctx": int(ctx["req"].get("n_ctx", 0) or 0),
                  "answer_marker": ctx["answer_marker"] or "",
                  "images": _log_uris(images)})
        out = _generate_and_format(ctx["req"], ctx["load_sig"], ctx["max_tokens"], unload_comfy,
                                   False, ctx["directive"], ctx["strip_think"],
                                   ctx["answer_marker"], ctx["help"], token_cb=token_cb,
                                   show_progress=False, stats=stats)
        text = out[0] if isinstance(out, (list, tuple)) else out
        if emit:
            emit({"event": "done", "text": text, "label": tag,
                  "finish_reason": out[2] if len(out) > 2 else "",
                  "gen_seconds": out[6] if len(out) > 6 else 0,
                  "max_tokens": int(ctx["max_tokens"]),
                  "output_tokens": int(out[5]) if len(out) > 5 else 0,
                  "prompt_tokens": int((stats or {}).get("prompt_tokens", 0) or 0),
                  "context_used": int((stats or {}).get("context_used", 0) or 0),
                  "n_ctx": int((stats or {}).get("n_ctx", 0) or ctx["req"].get("n_ctx", 0) or 0)})
        if isinstance(text, str) and text.startswith("[ERROR]"):
            return "", text
        logging.info(f"[Morpheus] {tag}: {len((text or '').split())} words")
        return (text or "").strip(), None

    @staticmethod
    def _bible(text):
        """Parse a style bible, filling anything the model forgot so assembly can't break."""
        b = _blocks(text, _LABELS_STYLE)
        return {
            "style": b.get("STYLE", "").strip(),
            # SUBJECT is the invariant description; SCENE is the older, story-shaped block name
            "subject": (b.get("SUBJECT") or b.get("SCENE") or "").strip(),
            "audio_bed": b.get("AUDIO BED", "").strip(),
            "negative": b.get("NEGATIVE", "").strip()
                        or "text, subtitles, logos, watermarks, cartoon or CG look",
        }

    @staticmethod
    def _assemble(bible, body, anchored):
        """The six numbered sections, in MiniMax's own order.

        Only sections 1 and 6 come from the bible; everything else belongs to the shot. Stamping the
        sequence's ARC into section 2 (the first version did) makes H3 replay that arc inside every
        5-second shot — it honours the first-frame keyframe for a couple of frames, then acts out
        the text."""
        parts = []
        if bible["style"]:
            parts.append(f"1. [Style and Aesthetic]: {bible['style']}")
        # Section 2 is the SHOT's situation and nothing else. The bible's subject line is not
        # stamped: in a morph sequence the subject IS what changes, so "a cyclist on a road bike"
        # ends up contradicting a shot that shows a demon on a motorcycle. It stays context for the
        # writer and material for the `style` output.
        scene = body.get("SITUATION", "").strip()
        if anchored:
            scene = (scene + " " + ANCHOR_CLAUSE).strip()
        if scene:
            parts.append(f"2. [Scene Overview]: {scene}")
        sb = body.get("STORYBOARD", "").strip()
        if sb:
            parts.append("3. [Storyboard]:\n" + sb)
        if body.get("CAMERA"):
            parts.append(f"4. [Camera]: {body['CAMERA'].strip()}")
        # the bed is a FALLBACK, never an append: the shot's own audio is already written to match
        # it, so appending printed the same sentence twice in every single prompt
        audio = dedupe_dialogue(sb, body.get("AUDIO", "")) or bible["audio_bed"]
        if audio:
            parts.append(f"5. [Audio & Voice]: {audio}")
        negative = bible["negative"]
        if anchored:
            negative = negative.rstrip(" .,") + ", " + ANCHOR_NEGATIVE
        parts.append(f"6. [Negative Prompt/Constraints]: {negative}")
        return "\n\n".join(parts)

    # -------------------------------------------------------------------------------------- writer
    def write(self, config, brief, shot_count, durations, anchor, link, seed, cache,
              unload_after_run, script="auto", live_preview=True, keyframes=None, beats="",
              prompt_overrides="", style="", shots=None, style_system="", shot_system_morph="",
              shot_system_start="", shot_system_text="", script_system="", unique_id=None):
        _require_h3()
        t_run = time.time()
        cfg = config if isinstance(config, dict) else {}
        if not cfg:
            raise ValueError("config is not a 'Local LLM Settings (GGUF)' bundle.")
        model = _resolve_path(cfg.get("model", PLACEHOLDER), cfg.get("model_path", ""))
        if not model or not os.path.isfile(model):
            raise ValueError(f"LLM model file not found: {model or '(none selected)'}")

        warn = []
        frames = _frame_list(keyframes)
        K = len(frames)
        mmproj = _resolve_path(cfg.get("mmproj", PLACEHOLDER), cfg.get("mmproj_path", ""))
        can_see = bool(mmproj and os.path.isfile(mmproj))
        if K and not can_see:
            warn.append(f"{K} keyframe(s) are wired but the config has no mmproj — attach a "
                        f"'Vision Settings (GGUF)' or the LLM writes blind (the frames are still "
                        f"used as keyframes by the sampler)")

        n = int(shot_count) or max(1, K - 1)
        if n > MAX_SHOTS:
            n = MAX_SHOTS
            warn.append(f"shot_count clamped to {MAX_SHOTS}")
        if not shot_count and K < 2:
            warn.append("with fewer than 2 keyframes there is nothing to derive a shot count from, "
                        "so one shot was written — set shot_count for a longer sequence")
        if K > n + 1:
            warn.append(f"{K} keyframes but only {n} shots — frames {n + 2}..{K} are unused")

        durs = _durations(durations, n)
        beat_lines = _beat_lines(beats)
        if len(beat_lines) > n:
            warn.append(f"{len(beat_lines)} beat line(s) for {n} shots — lines are matched to shots "
                        f"by position, so the extra ones were ignored")
        overrides = _split_overrides(prompt_overrides)
        if len(overrides) > n:
            warn.append(f"{len(overrides)} prompt override block(s) for {n} shots — blocks are "
                        f"matched by position (separate them with a line of ---)")

        sys_style = (style_system or "").strip() or STYLE_SYSTEM
        sys_morph = (shot_system_morph or "").strip() or SHOT_SYSTEM_MORPH
        sys_start = (shot_system_start or "").strip() or SHOT_SYSTEM_START
        sys_text = (shot_system_text or "").strip() or SHOT_SYSTEM_TEXT
        sys_script = (script_system or "").strip() or SCRIPT_SYSTEM
        brief = (brief or "").strip()
        if not brief and not beat_lines:
            raise ValueError("Nothing to write from: give a brief, or per-shot beats, or both.")
        want_script = script == "auto" and not any(beat_lines) and bool(brief) and n > 1

        # Deliberately WITHOUT the beat lines: they don't reach the style bible (see the bible call
        # below), which keeps the bible — and therefore shot 1 — cached when a later shot's note is
        # re-worded. The division of labour is: `brief` describes the whole sequence, `beats` direct
        # individual shots.
        env = shot_cache.key(_cfg_fingerprint(cfg), brief, sys_style, sys_morph, sys_start,
                             sys_text, sys_script, int(seed), n, anchor,
                             link, ",".join(f"{d:g}" for d in durs))
        use_cache = cache == "disk"
        unload_comfy_once = bool(cfg.get("unload_comfy_models", True))
        stages = n + 1 + (1 if want_script else 0)
        pbar = comfy.utils.ProgressBar(stages)
        did_call = [False]  # only the FIRST real call should free ComfyUI's VRAM

        emit = None
        if live_preview:
            try:
                from server import PromptServer
                nid = str(unique_id[0] if isinstance(unique_id, list) else unique_id)

                def emit(payload):
                    try:
                        PromptServer.instance.send_sync("kinburg.llm", {"id": nid, **payload})
                    except Exception:
                        pass
            except Exception:
                emit = None

        def ask(system, prompt, images, tag):
            first = not did_call[0]
            did_call[0] = True
            return self._ask(cfg, system, prompt, images, unload_comfy_once and first, tag, emit)

        # --- the style bible: written once, stamped on every shot -------------------------------
        style_src = "wired"
        if (style or "").strip():
            bible = self._bible(style)
        else:
            bkey = shot_cache.key(env, "bible")
            hit = shot_cache.load_json(bkey) if use_cache else None
            if hit:
                bible, style_src = self._bible(hit.get("text", "")), "cached"
            else:
                # only the brief and the shape of the sequence — no per-shot beats, so re-directing
                # one shot later doesn't invalidate the look of the whole thing
                prompt = (f"Sequence brief:\n{brief}\n\n"
                          f"It will be told in {n} shot(s) of about "
                          f"{', '.join(f'{d:.1f}' for d in durs)} seconds."
                          + ("\n\nThe look must match the images provided." if K and can_see else ""))
                text, err = ask(sys_style, prompt, frames[:2] if can_see else [], "style bible")
                if err:
                    warn.append(f"style bible call failed ({err}) — falling back to the brief")
                    bible, style_src = self._bible(""), "failed"
                    bible["subject"] = brief
                else:
                    bible, style_src = self._bible(text), "written"
                    if use_cache:
                        shot_cache.save_json(bkey, {"text": text})
        # --- the script: one direction per shot ---------------------------------------------------
        # This exists because of one measured failure. Without it, a shot with no beat line of its own
        # was handed the ENTIRE brief as its direction — so the third shot of a text chain, given the
        # same whole-story instruction as the second, replayed the whole story compressed and then
        # added its own part. A shot can only be told about its own stretch.
        script_src = "hand-written beats" if any(beat_lines) else "off"
        if want_script:
            skey = shot_cache.key(env, "script")
            hit = shot_cache.load_json(skey) if use_cache else None
            if hit and hit.get("lines"):
                beat_lines, script_src = list(hit["lines"]), "cached"
            else:
                lens = ", ".join(f"shot {i + 1}: {_seconds(_frames_for(d)):.2f}s"
                                 for i, d in enumerate(durs))
                prompt = (f"Sequence brief:\n{brief}\n\n"
                          f"Break it into {n} shots. Their lengths: {lens}."
                          + ("\n\nThe images are the sequence's first and last keyframes — the plan "
                             "must arrive there." if K and can_see else ""))
                # only the two ENDS of the sequence: the plan needs to know where it starts and
                # where it lands, the boundaries in between are each shot's own business
                shown = ([frames[0]] + ([frames[-1]] if K > 1 else [])) if (can_see and K) else []
                text, err = ask(sys_script, prompt, shown, "script")
                if err:
                    warn.append(f"the planning call failed ({err}) — shots were written without a "
                                f"plan, so their pacing is improvised")
                    script_src = "failed"
                else:
                    lines = _parse_script(text, n)
                    missing = [i + 1 for i, ln in enumerate(lines) if not ln]
                    if missing:
                        warn.append(f"the plan had no line for shot(s) {missing} — those shots were "
                                    f"written without direction")
                    beat_lines, script_src = lines, "written"
                    if use_cache:
                        shot_cache.save_json(skey, {"lines": lines})
            pbar.update_absolute(1, stages)

        bible_text = "\n\n".join(f"[{k.upper()}]: {v}"
                                 for k, v in (("style", bible["style"]), ("subject", bible["subject"]),
                                              ("audio bed", bible["audio_bed"]),
                                              ("negative", bible["negative"])) if v)
        pbar.update_absolute(stages - n, stages)

        # --- one call per shot -------------------------------------------------------------------
        chain = shots
        rows, prompts, end_state, prev = [], [], "", shot_cache.key(env, "bible")
        for i in range(n):
            # boundary i opens shot i, boundary i+1 closes it — while the frames last
            vis_start = frames[i] if i < K else None
            vis_end = frames[i + 1] if i + 1 < K else None
            # what the LLM sees is not what the sampler is conditioned on: in 'continuous' the
            # start comes from the previous shot's generated tail, so only the end is wired
            cond_start = vis_start
            if anchor == "continuous" and i > 0 and link == "continue":
                cond_start = None
            beat = beat_lines[i] if i < len(beat_lines) else ""
            dur = durs[i]
            snapped = _seconds(_frames_for(dur))

            key_i = shot_cache.key(prev, i, beat, f"{snapped:.3f}",
                                   shot_cache.tensor_key(vis_start if can_see else None),
                                   shot_cache.tensor_key(vis_end if can_see else None))
            prev = key_i

            # Is this shot's first frame fixed at all? Either an image is conditioned on, or the
            # previous shot's tail is inherited. Only shot 1 of a chain (or a `cut`) is truly free,
            # and only a free shot may open the story.
            anchored = cond_start is not None or (i > 0 and link == "continue")

            override = overrides[i].strip() if i < len(overrides) else ""
            if override:
                prompt, src = override, "override"
            else:
                hit = shot_cache.load_json(key_i) if use_cache else None
                if hit:
                    body = {k: v for k, v in hit.get("body", {}).items()}
                    end_state = hit.get("end_state", "")
                    prompt, src = self._assemble(bible, body, anchored), "cached"
                else:
                    # The context per call is deliberately thin — length, the director's note, and
                    # a starting state only when there is no image to show it. Feeding the brief and
                    # the whole style bible in here made the model weave all of it into its answer.
                    imgs = [im for im in (vis_start, vis_end) if im is not None] if can_see else []
                    if len(imgs) == 2:
                        sys_shot, mode = sys_morph, "2 keyframes"
                    elif len(imgs) == 1 and vis_start is not None:
                        sys_shot, mode = sys_start, "start keyframe"
                    elif len(imgs) == 1:
                        # only an end frame: closest job is the morph one, minus a start image
                        sys_shot, mode = sys_morph, "end keyframe"
                    else:
                        sys_shot, mode = sys_text, "text"
                    parts = [f"Shot length: {snapped:.2f} seconds. The beats must span 0s to "
                             f"{snapped:.2f}s.",
                             # ONLY this shot's own direction. The brief used to land here whole
                             # whenever a shot had no line of its own, and a shot told the whole
                             # story acts out the whole story — that is what the script prevents.
                             ("Direction for this shot: " + beat) if beat else "",
                             # continuity: what the previous shot said its last frame looks like.
                             # This is the only thing holding a keyframe-less chain together.
                             ("The first frame is fixed and is not shown to you. It looks like this: "
                              + end_state + " Begin from exactly that state, already at that speed "
                              "and in that direction.")
                             if (i > 0 and end_state and vis_start is None) else "",
                             ("The single image given is this shot's LAST frame — the shot must "
                              "arrive exactly there.") if (len(imgs) == 1 and vis_start is None) else "",
                             ]
                    text, err = ask(sys_shot, "\n\n".join(p for p in parts if p), imgs,
                                    f"shot {i + 1}/{n} ({mode})")
                    if err:
                        warn.append(f"shot {i + 1}: LLM call failed ({err}) — used the director's "
                                    f"note or the brief as the prompt")
                        body = {"STORYBOARD": beat or brief}
                        src = "failed"
                    else:
                        body = _blocks(text, _LABELS_SHOT)
                        if not body.get("STORYBOARD"):
                            # unlabelled answer: keep it whole rather than throw the shot away
                            body = {"STORYBOARD": text.strip()}
                            warn.append(f"shot {i + 1}: the answer had no [STORYBOARD] label, so "
                                        f"the whole reply was used as the shot description")
                        end_state = body.get("END STATE", "").strip()
                        src = "written"
                        if use_cache:
                            shot_cache.save_json(key_i, {"shot": i + 1, "body": body,
                                                         "end_state": end_state})
                    prompt = self._assemble(bible, body, anchored)

            # Stamp what the writer could SEE, not who wrote it: a shot whose conditioned first
            # frame is an inherited tail was written against a forecast, and that is exactly the
            # shot the sampler's in-loop LLM should correct once the real frame exists.
            saw_start = cond_start is not None and can_see
            if src == "override":
                refine = "off"                      # a hand-pasted prompt is nobody else's business
            elif src == "failed":
                refine = "full"                     # there is no real prompt to preserve
            elif anchored and not saw_start:
                refine = "opening"
            else:
                refine = "off"

            chain, _info = KinburgMorpheusDream().build(
                prompt=prompt, duration=dur, link=link, seed_offset=0, shots=chain,
                start_frame=cond_start, end_frame=vis_end, refine=refine)
            prompts.append(prompt)
            rows.append((i + 1, snapped, "frame" if cond_start is not None else
                         ("inherited" if (i > 0 and link == "continue") else "text"),
                         "frame" if vis_end is not None else "open", src, len(prompt)))
            pbar.update_absolute(stages - n + i + 1, stages)

        if resolve_unload(unload_after_run, cfg):
            _shutdown_worker()
        if use_cache:
            shot_cache.prune()

        total = sum(_frames_for(d) for d in durs)
        report = ["Morpheus Storyboard — {} shot(s), {} frames = {:.2f} s of video".format(
                      n, total, _seconds(total)),
                  f"keyframes: {K} wired → {min(K, n + 1)} boundary/ies used"
                  + ("" if can_see else " (LLM is blind: no mmproj)"),
                  f"anchor: {anchor} · open shots: {link}",
                  f"style bible: {style_src}",
                  f"script: {script_src}",
                  f"LLM: {os.path.basename(model)} · seed {int(seed)} · cache {cache}",
                  f"total: {_format_elapsed(time.time() - t_run, 'human')}",
                  "",
                  "  #    dur   start        end     source     chars",
                  "  " + "-" * 52]
        for idx, dur, st, en, src, ln in rows:
            report.append(f"  {idx:<2} {dur:>6.2f}s {st:<12} {en:<7} {src:<10} {ln}")
        if warn:
            report += ["", "warnings:"] + [f"  · {w}" for w in warn]

        logging.info(f"[Morpheus] storyboard written: {n} shot(s) in "
                     f"{_format_elapsed(time.time() - t_run, 'human')}")
        return (chain, "\n\n---\n\n".join(prompts), bible_text, "\n".join(report),
                "\n".join(beat_lines[:n]))


NODE_CLASS_MAPPINGS = {"KinburgMorpheusStoryboard": KinburgMorpheusStoryboard}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgMorpheusStoryboard": "Morpheus Storyboard 🌙"}
