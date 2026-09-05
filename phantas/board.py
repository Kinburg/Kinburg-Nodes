"""Phantas Storyboard 🎞 — a brief becomes a chain of keyframe prompts.

The writing half of Phantas, and the director of the whole pipeline. Morpheus writes *shots*;
Phantas writes **states**, because a keyframe sits between two shots and is therefore a still moment
rather than an action. One board is:

    frame 1 ──beat 1──> frame 2 ──beat 2──> frame 3 ──beat 3──> frame 4
     a still            a still            a still            a still
        └──── shot 1 ─────┴──── shot 2 ─────┴──── shot 3 ────┘

so N frame prompts and N-1 beats come out of one plan. The beats are emitted in exactly the format
`Morpheus Storyboard`'s own `beats` field takes, and that node skips its planning call when `beats`
is filled — which is the point: **the arc is planned once.** Planning it again downstream would give
two LLM calls independent authority over the same story, and they would disagree.

`write_beats` turns that half off. A board whose keyframes go to `Save Clip` as slides never reaches
a video model, so the directions between them would be written for nobody — and it is the GRAMMAR
that stops them rather than an instruction: with the `transitions` array gone from the shape, the
model cannot spend a token deciding whether to obey. The board still carries one blank shot per gap,
so it stays a chain; the switch is the only thing between a slideshow and a film.

It also swaps the planner's system prompt, and that is the half that changes the pictures rather
than the clock. The continuous-take rule exists because a video model has to *travel* from one
framing to the next, so a crop change must be a camera move. Nothing travels between two slides:
the cut is free, and a slideshow planned under the video rules comes out as thirty variations on one
camera position. `PLAN_SYSTEM_STILLS` therefore asks for an edit instead — scale alternating, no two
neighbours framed alike, and some frames with nobody in them at all, because a slideshow needs air
the way a take does not. Both shipped prompts are treated as "the default", so the `system_plan`
widget picks up whichever the mode needs; text somebody typed there is theirs and is used in both.

Three things shape the prompts, all of them learned the hard way elsewhere in this pack:

  * **The bible is stamped, not rewritten.** Look, subject invariants and negatives are written once
    by a text-only call and pasted into every frame byte-for-byte. Anything regenerated per frame
    drifts, and here drift is structural rather than cosmetic: frame k is simultaneously the end of
    shot k-1 and the start of shot k, so two neighbours that disagree describe a shot that morphs
    halfway through.
  * **A frame prompt is a frozen moment.** It may not narrate. "He begins to transform" asks an
    image model for a motion smear; the plan's beat carries the change, the frame carries the state
    the change arrives at.
  * **The chain is one continuous take.** With no cuts, framing cannot jump — a crop change has to
    be a camera *move*, and the plan assigns framings to the boundaries so each shot describes the
    move between two of them.

The LLM never names seconds. It gives each transition a *weight* — how much visible change it
carries — and `timing.py` lays those onto H3's frame grid. See that module for why.

**The cast is the part that decides whether a character survives the sequence.** An image model has
no memory between calls, so "the singer" or "the man from the previous shot" is a different human
being in every picture — which is exactly what a first version of this node produced. Identity has
to be re-stated in full, in the same literal words, in every frame the person appears in. So the
bible carries a `[CAST]` block of `Name — full description` lines, the plan says who is `present` in
each keyframe, and those people's lines are stamped verbatim into that keyframe's prompt. Only the
people present: pasting the whole cast into a shot of an empty room grows a person in it.

The config's `context` is deliberately kept on every call, because a Context Collector or a Card
Preset wired into the LLM Settings node is the other way a cast arrives here.

Everything the LLM writes is cached on disk under a causal key: the prompts ARE the sampler's cache
key, so a prompt that changed on every run would re-render every frame.

Every call streams into a `Kinburg Live Log` node, the plan included — the worker's `use_stream` is
unconditional, and the branch that used to skip streaming for GBNF was removed precisely because it
cost those runs their live log.
"""
import json
import logging
import re

from . import timing
from ..local_llm.llm_node import (LLM_CONFIG, UNLOAD_MODES, _generate_and_format, _shutdown_worker,
                                  build_llm_request, resolve_unload)
from ..util import diskcache
from ..util.images import log_uris
from ..categories import CAT_PHANTAS

PHANTAS_BOARD = "KINBURG_PHANTAS_BOARD"
CACHE_MODES = ["disk", "off"]
MAX_FRAMES = 32                  # a board this long is already 3 minutes of video
PROMPT_SEP = "\n---\n"           # how the `prompts` output separates frames, and how an override is read

_store = diskcache.Store("phantas_boards", "Phantas")


# =========================================================================================== prompts
STYLE_SYSTEM = """You are a director writing the STYLE BIBLE for a sequence of still keyframes that will be generated one at a time, by an image model, in separate calls that never see each other.

These blocks are pasted into EVERY frame's prompt unchanged, so they may contain only what is true in every single frame.

CRITICAL: never describe the sequence's story, its beginning, its ending, or the stages of any change. A frame prompt that mentions the whole arc makes the image model try to show the arc in one picture.

Answer with EXACTLY these three labelled blocks, in this order, and nothing else:

[STYLE]: one paragraph, look and craft only — genre or reference, lens and focal length, depth of field, lighting, colour grade, grain and texture, atmosphere. No story, no camera moves, no shot list.
[CAST]: every person who appears, ONE PER LINE, as `Name — a full physical description`.
[SUBJECT]: one or two sentences of non-human INVARIANTS — the location, the time of day, the vehicle or the props. Never the story, never a start or an end state, and not the people (they are the cast).
[NEGATIVE]: a comma-separated list of faults to avoid. Always include text, subtitles, logos and watermarks; add only faults — blur, artifacts, distorted anatomy, extra limbs, extra fingers, style breaks. NEVER list anything the sequence is supposed to DO: if the subject transforms, words like "morphing", "transformation" or "shape change" must not appear here.

THE CAST BLOCK IS THE MOST IMPORTANT THING YOU WRITE. Each line is pasted verbatim into the prompt of every frame that person appears in, and the image model has no memory of the other frames: someone described in five words is a different human being in every picture. Write each of them the way a casting note does — apparent age, build and height, face shape and its distinguishing features, skin tone, hair colour and the exact cut, eye colour, facial hair, wardrobe from head to foot with colours and materials, and anything they always carry. Two or three sentences each, minimum. If people are described to you in the material you were given, use THOSE people and keep their given names, details and wording; invent nobody. If nobody appears, write `[CAST]: none`.

Write in English, plainly, no markdown emphasis, no commentary."""

PLAN_SYSTEM = """You are a director breaking a brief into KEYFRAMES for a continuous video sequence.

A keyframe is a frozen moment. Between two consecutive keyframes runs one shot, which the video model will generate as the movement from the first to the second. So N keyframes describe N-1 shots.

THE SEQUENCE IS ONE CONTINUOUS TAKE. There are no cuts anywhere in it. The camera may travel, push in, pull back, crane, orbit or follow, but it never jumps: a change of framing between two keyframes is a camera MOVE that the shot between them performs. Never plan a montage.

For each keyframe give:
- "framing": the shot size and camera angle at that instant (e.g. "wide low-angle three-quarter", "medium tracking profile", "close-up over the shoulder"). Consecutive framings must be reachable by a camera move.
- "present": the names of the cast members visible in that frame, exactly as the cast block spells them. An empty list if the frame shows nobody. Never name anyone who is not in the cast, and never leave someone out who is on screen — this list decides whose description gets attached to the picture.
- "state": what is frozen on screen at that instant — position, pose, form, what the light is doing. A description of a STILL. Never write a change, never write "begins to", "starts to" or "is about to".

For each transition (there is exactly one fewer than the keyframes) give:
- "beat": two or three sentences, present tense, saying what visibly HAPPENS between those two keyframes and what the camera does. This is read alone by another writer who cannot see the other beats, so it must stand completely on its own and never say "then", "next", "finally" or "meanwhile".
- "weight": an integer from 1 to 5 for how much visible change this transition carries. 1 is a held moment with a slow drift, 5 is the largest change in the sequence. Weights set how long each shot runs, so spend them honestly.

Spend the whole brief across the sequence: the last keyframe lands on the brief's endpoint and no earlier one may get there first. If a transformation completes at keyframe 2 of 6, the plan is wrong.

Answer with JSON only."""

PLAN_SYSTEM_STILLS = """You are a director breaking a brief into a sequence of STILL IMAGES — a slideshow. There is no video anywhere in it: each keyframe is shown on its own, one after another, and the audience never sees anything between two of them.

Because nothing is generated between two pictures, THE CUT IS FREE. From one frame to the next you may change shot size, angle, lens and distance as sharply as you like — and you should. A run of near-identical framings reads as a contact sheet rather than as an edited sequence. What may NOT change is the world: the same place, the same time of day, the same weather and light, the same people in the same wardrobe, unless the brief itself changes them.

For each keyframe give:
- "framing": the shot size and camera angle of that picture (e.g. "extreme wide high-angle", "medium two-shot at eye level", "macro insert on the hands"). Vary it deliberately down the sequence, and never give a frame the same framing as the one before it.
- "present": the names of the cast members visible in that frame, exactly as the cast block spells them. An empty list if the frame shows nobody. Never name anyone who is not in the cast, and never leave someone out who is on screen — this list decides whose description gets attached to the picture.
- "state": what is on screen in that picture — position, pose, form, what the light is doing. A description of a STILL. Never write a change, never write "begins to", "starts to" or "is about to".

Cut the sequence the way an editor would:
- Open on a frame wide enough to say where we are, and close on the brief's endpoint.
- Alternate scale. A close-up lands because a wide came before it.
- Not every picture needs a person in it. Empty rooms, landscapes, hands, objects and weather are what give a slideshow air — give some frames an empty "present" list on purpose.
- Every picture has to be worth stopping on. A frame whose only job is to get from one picture to the next has nothing to do here: in a slideshow there is no "between".

Spend the whole brief across the sequence: the last keyframe lands on the brief's endpoint and no earlier one may get there first. If a transformation completes at keyframe 2 of 6, the plan is wrong.

Answer with JSON only, and with the "frames" array alone — a slideshow has no transitions."""

FRAME_SYSTEM = """You are writing the prompt for ONE still image: a single keyframe of a video sequence.

You are given the style bible, this frame's framing and state, and — when there is one — the prompt of the frame immediately before it, so the two pictures can be of the same world.

Write ONE paragraph describing what is in THIS frame, as a photograph of a frozen instant:
- the subject, its exact pose and position in the frame, its form and its surfaces
- the framing you were given: shot size, camera angle, lens behaviour
- the environment and what the light is doing at this instant

Rules:
- **Everyone on screen is NAMED and described in full.** The image model never sees the other frames, so "the man from the previous shot" or "the singer" produces a different person every time. Give each person present their name and their face, hair, build and wardrobe again, in this frame, using the cast block's own words rather than a summary of them.
- A still has no time in it. Never write a change, a movement in progress, "begins to", "starts to", "is about to", or anything that happens before or after this instant.
- Never mention the sequence, the other frames, the shot, the story or its ending.
- If a previous frame's prompt is given, keep everything the brief did not change: the same wardrobe, the same location, the same time of day, the same light, the same lens.
- Plain descriptive English, one paragraph, no headings, no markdown, no commentary."""


def _plan_system(text, with_beats):
    """The plan call's system prompt: the mode's default, unless something was typed over it.

    The widget ships PRE-FILLED with `PLAN_SYSTEM`, so blank is not the only way of saying "use the
    default" — a node nobody has touched hands back that exact text. Both of the shipped prompts are
    therefore read as "default" and answered with whichever one the mode needs, which is also what
    stops the incoherent combination: the stills prompt asks for frames alone while the beats
    grammar demands transitions. Anything else is the user's, and the user's wins in both modes."""
    txt = (text or "").strip()
    if not txt or txt in (PLAN_SYSTEM.strip(), PLAN_SYSTEM_STILLS.strip()):
        return PLAN_SYSTEM if with_beats else PLAN_SYSTEM_STILLS
    return txt


def _plan_grammar(n_frames, with_transitions=True):
    """A GBNF grammar forcing exactly `n_frames` keyframes and `n_frames - 1` transitions.

    The counts are baked in by repeating the productions rather than using a repetition operator:
    the shape is then certain on any llama.cpp build, and a model that miscounts cannot even emit
    the wrong number of entries. String/char/int/ws productions are the ones the Vision Judge
    grammar has been using in production, escapes and all.

    `with_transitions=False` is the slideshow board: the whole `transitions` array goes, and the
    `trans`/`int`/`digit` productions go with it rather than being left defined and unreachable. A
    grammar is the only reliable way to stop that writing — an instruction not to write beats still
    costs the tokens the model spends deciding to obey it, while a rule it cannot reach costs none.
    """
    n_frames = max(2, int(n_frames))
    frames = " ws \",\" ws ".join(["frame"] * n_frames)
    root = 'root ::= ws "{" ws "\\"frames\\"" ws ":" ws "[" ws ' + frames + ' ws "]"'
    if with_transitions:
        trans = " ws \",\" ws ".join(["trans"] * (n_frames - 1))
        root += ' ws "," ws "\\"transitions\\"" ws ":" ws "[" ws ' + trans + ' ws "]"'
    out = (
        root + ' ws "}" ws\n'
        'frame ::= "{" ws "\\"framing\\"" ws ":" ws string ws "," ws "\\"present\\"" ws ":" ws names '
        'ws "," ws "\\"state\\"" ws ":" ws string ws "}"\n')
    if with_transitions:
        out += ('trans ::= "{" ws "\\"beat\\"" ws ":" ws string ws "," ws "\\"weight\\"" ws ":" ws '
                'int ws "}"\n')
    out += (
        'names ::= "[" ws (string (ws "," ws string)*)? ws "]"\n'
        'string ::= "\\"" char* "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt/] | "u" hex hex hex hex)\n'
        'hex ::= [0-9a-fA-F]\n')
    if with_transitions:
        out += ('int ::= digit digit?\n'
                'digit ::= [0-9]\n')
    return out + 'ws ::= [ \\t\\n]*\n'


_LABELS = ("STYLE", "CAST", "SUBJECT", "NEGATIVE")


def _blocks(text):
    """`[LABEL]: body` blocks out of the bible reply, tolerant of the model's formatting habits.

    CAST is the one block whose LINE BREAKS matter — one person per line — so it is joined with
    newlines while the prose blocks are joined with spaces."""
    out, cur = {}, None
    for raw in (text or "").split("\n"):
        line = raw.strip()
        m = re.match(r"^\**\s*\[?\s*(" + "|".join(_LABELS) + r")\s*\]?\s*\**\s*:\s*(.*)$", line, re.I)
        if m:
            cur = m.group(1).upper()
            out[cur] = m.group(2).strip()
        elif cur and line:
            out[cur] = (out[cur] + ("\n" if cur == "CAST" else " ") + line).strip()
    return out


def parse_cast(text):
    """`Name — description` lines → `[(name, whole line)]`.

    The line is kept WHOLE and verbatim, because that is the entire mechanism: the same literal
    words in every frame are what make a face come out the same, and a paraphrase is a different
    person. `—`, `-` and `:` all separate; a line with no separator is a name with no description
    and is kept anyway, so the plan can still refer to it."""
    out = []
    for raw in (text or "").split("\n"):
        line = raw.strip().lstrip("-•*").strip()
        if not line or line.lower() in ("none", "none.", "n/a"):
            continue
        name = line
        for sep in ("—", " - ", ":", " – "):
            if sep in line:
                name = line.split(sep, 1)[0]
                break
        name = name.strip().strip("*").strip()
        if name:
            out.append((name, line))
    return out


def cast_for(names, cast):
    """The verbatim cast lines for `names`, in CAST order.

    Cast order, not the order the plan happened to list them in: the stamped text has to be
    identical between two frames holding the same people, or the cache splits and the pictures
    drift for no reason. Matching is case-insensitive, and falls back to "the name appears in the
    line" so a plan that writes "Mira Vale" still finds "Mira"."""
    want = [str(n).strip().lower() for n in (names or []) if str(n).strip()]
    if not want:
        return []
    out = []
    for name, line in cast:
        low = name.lower()
        if any(w == low or w in low or low in w for w in want):
            out.append(line)
    return out


def parse_bible(text, cast_override=""):
    """A style bible, with anything the model forgot filled in so assembly can never break.

    A non-empty `cast_override` REPLACES whatever the model wrote: descriptions typed on the node
    (or arriving from Card Presets) are the authority on who these people are."""
    b = _blocks(text)
    cast_text = (cast_override or "").strip() or b.get("CAST", "").strip()
    return {
        "style": b.get("STYLE", "").strip(),
        "cast": cast_text,
        "subject": b.get("SUBJECT", "").strip(),
        "negative": b.get("NEGATIVE", "").strip()
                    or "text, subtitles, logos, watermarks, blur, artifacts, distorted anatomy",
    }


def parse_plan(text, n_frames, with_transitions=True):
    """`(frames, transitions, notes)` from the plan reply.

    The grammar makes the shape certain when it is honoured, but a worker that fell back, a cached
    answer from an older version or a hand-edited plan must not take the run down — anything missing
    is filled and named in `notes`, and the board still renders.

    With `with_transitions=False` the reply carries no transitions and none are looked for, but
    `n_frames - 1` BLANK ones still come back: a board keeps one shot per gap whatever mode wrote
    it, so it stays a chain `Phantas` can pack and `Morpheus Storyboard` can fill in later. A
    slideshow that turns out to want video is then one checkbox away rather than a re-plan."""
    notes = []
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        obj = {}
        notes.append("the plan did not parse as JSON — frames and beats were filled in blank")

    raw_f = obj.get("frames") if isinstance(obj.get("frames"), list) else []
    raw_t = obj.get("transitions") if isinstance(obj.get("transitions"), list) else []
    if len(raw_f) != n_frames:
        notes.append(f"the plan gave {len(raw_f)} keyframes, not {n_frames}")
    if with_transitions and len(raw_t) != n_frames - 1:
        notes.append(f"the plan gave {len(raw_t)} transitions, not {n_frames - 1}")

    frames = []
    for i in range(n_frames):
        d = raw_f[i] if i < len(raw_f) and isinstance(raw_f[i], dict) else {}
        present = d.get("present")
        present = [str(p).strip() for p in present if str(p).strip()] if isinstance(present, list) else []
        frames.append({"framing": str(d.get("framing", "") or "").strip(),
                       "present": present,
                       "state": str(d.get("state", "") or "").strip()})
    trans = []
    for i in range(n_frames - 1):
        d = (raw_t[i] if with_transitions and i < len(raw_t) and isinstance(raw_t[i], dict)
             else {})
        try:
            w = float(d.get("weight", 1))
        except (TypeError, ValueError):
            w = 1.0
        trans.append({"beat": str(d.get("beat", "") or "").strip(),
                      "weight": w if w > 0 else 1.0})
    return frames, trans, notes


def assemble(bible, body, framing, cast_lines=()):
    """One frame's finished prompt: the look, who is in it, this frame's body, the framing.

    `cast_lines` are the verbatim descriptions of the people the plan says are in THIS frame, and
    stamping them is the whole answer to "the same character came out different every frame": an
    image model has no memory between calls, so identity has to be re-stated in full every time, in
    the same words. Only the people present are stamped — pasting the whole cast into a shot of an
    empty room is how a picture grows a person who is not supposed to be in it.

    The bible's SUBJECT is context for the writer but is NOT stamped, for the reason Morpheus found
    the hard way: in a transformation sequence the subject is exactly what changes, so a standing
    "a cyclist on a road bike" ends up contradicting the frame that shows a demon on a motorcycle.
    """
    parts = []
    if bible.get("style"):
        parts.append(bible["style"].strip())
    lines = [ln.strip() for ln in (cast_lines or []) if str(ln).strip()]
    if lines:
        parts.append("\n".join(lines))
    if body:
        parts.append(body.strip())
    if framing:
        parts.append(framing.strip())
    return "\n\n".join(p for p in parts if p)


def split_override(text, n_frames):
    """A hand-edited `prompts` output read back in. `None` when nothing was typed; otherwise exactly
    `n_frames` entries, short input padded with "" (meaning "write this one")."""
    if not (text or "").strip():
        return None
    parts = [p.strip() for p in re.split(r"^\s*-{3,}\s*$", text, flags=re.M)]
    parts = [p for p in parts if p] or [text.strip()]
    return [parts[i] if i < len(parts) else "" for i in range(n_frames)]


# ============================================================================================= LLM
def _ask(cfg, system, user_prompt, unload_comfy, tag, emit=None, grammar=""):
    """One call through the same path the Local LLM node uses. Returns `(text, error)`.

    `grammar`, when given, constrains the answer — and takes llama.cpp's non-stream branch, so a
    grammar call arrives whole rather than token by token. Text calls stream into the live log."""
    call_cfg = dict(cfg)
    call_cfg["system_prompt"] = system
    # The config's `context` is deliberately KEPT: a Context Collector or a Card Preset wired into
    # the settings node is how the cast gets here, and wiping it (as the Morpheus writer does, where
    # the brief carries everything) is what made the first version write about strangers.
    call_cfg["output_format"] = "text"
    call_cfg["grammar"] = ""
    call_cfg["strip_think"] = True
    call_cfg["max_tokens"] = max(int(cfg.get("max_tokens", 512) or 512), 900)

    err, ctx = build_llm_request(call_cfg, user_prompt, grammar_override=(grammar or None))
    if err:
        if emit:
            emit({"event": "done", "text": err, "label": tag})
        return "", err

    token_cb, stats = None, None
    if emit:
        stats = {}
        # Grammar runs stream too — the worker's `use_stream` is unconditional, and the branch that
        # used to skip streaming for GBNF was removed precisely because it cost those runs their log.
        ctx["req"]["stream_text"] = True

        def token_cb(delta):
            emit({"event": "delta", "delta": delta})

        emit({"event": "start", "label": tag, "max_tokens": int(ctx["max_tokens"]),
              "n_ctx": int(ctx["req"].get("n_ctx", 0) or 0),
              "answer_marker": ctx["answer_marker"] or "", "images": []})
    out = _generate_and_format(ctx["req"], ctx["load_sig"], ctx["max_tokens"], unload_comfy,
                               False, ctx["directive"], ctx["strip_think"], ctx["answer_marker"],
                               ctx["help"], token_cb=token_cb, show_progress=False, stats=stats)
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
    logging.info(f"[Phantas] {tag}: {len((text or '').split())} words")
    return (text or "").strip(), None


def _cfg_fingerprint(cfg):
    """Cache identity of an LLM config bundle — everything that could change what it writes."""
    try:
        return diskcache.key(json.dumps(cfg, sort_keys=True, default=str))
    except Exception:
        return diskcache.key(str(sorted((cfg or {}).items(), key=lambda kv: str(kv[0]))))


# =========================================================================================== node
class KinburgPhantasStoryboard:
    """Writes the keyframe prompts, the beats between them, and how long each shot runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (LLM_CONFIG, {"tooltip": "A 'Local LLM Settings (GGUF)' bundle. Text only — this node writes, it never looks at pictures, so a light text model is the right one here and leaves the VRAM for the sampler."}),
                "brief": ("STRING", {"multiline": True, "default": "", "tooltip": "What the clip is. One or several sentences: who or what is on screen, where, and what happens over the whole sequence from its first moment to its last."}),
                "count_mode": (timing.COUNT_MODES, {"default": "scenes", "tooltip": "Which unit you are counting in. A keyframe sits BETWEEN shots, so the numbers are always one apart:\n\n• frames — 'count' is how many KEYFRAMES to draw (N pictures = N-1 shots).\n• scenes — 'count' is how many SHOTS to make (S shots = S+1 pictures).\n• duration — neither; the shot count is worked out from 'target_length'."}),
                "count": ("INT", {"default": 3, "min": 1, "max": MAX_FRAMES, "tooltip": "Keyframes or scenes, depending on 'count_mode'. Ignored when the mode is 'duration'."}),
                "target_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1, "tooltip": "Target length of the finished clip, in seconds. 0 = no target.\n\nIn 'duration' mode this decides the shot count. In the other two it only shapes the shot LENGTHS, and it must be achievable: n shots can add up to between n×5.17 s and n×15.08 s, and a target outside that band is an error rather than something quietly clamped."}),
                "durations": ("STRING", {"default": "", "tooltip": "Seconds per shot: one value for all of them, or a comma list ('5.17, 8, 5.17') where the last value repeats. LEAVE EMPTY to let the planner decide — it weights every transition by how much change it carries, and the weights are laid onto H3's 0.71 s frame grid here."}),
            },
            "optional": {
                "cast": ("STRING", {"multiline": True, "default": "", "tooltip": "Who is in this clip, ONE PER LINE, as 'Name — a full physical description'. Paste the character cards you already use for cover art: these lines are stamped VERBATIM into every frame that person appears in, and that is what makes the same face come out of every frame.\n\nDescribe them at cover-art length — age, build, face, hair, eyes, wardrobe head to foot. A person described in five words is a different human being in each picture, because the image model never sees the other frames.\n\nLeave empty and the style-bible call writes the cast itself from the brief and from whatever the LLM Settings' context holds — usable, but its own wording rather than yours. Leave empty ALSO when the subject is supposed to change (a transformation), since a fixed description would contradict it."}),
                "style_notes": ("STRING", {"multiline": True, "default": "", "tooltip": "Extra instructions for the look only — reference films, lens, grade, era. Goes to the style-bible call, not to the plan."}),
                "prompts_override": ("STRING", {"multiline": True, "default": "", "tooltip": "Paste the 'prompts' output back here after editing it, and those frames are used verbatim instead of being written again. Frames are separated by a line of '---'; an empty entry means 'write this one'."}),
                "preferred_length": ("FLOAT", {"default": timing.DEFAULT_SECONDS, "min": timing.MIN_SECONDS, "max": timing.MAX_SECONDS, "step": 0.1, "tooltip": "How long an average shot should run. Sets the shot count in 'duration' mode, and the average length when no target is given."}),
                "write_beats": ("BOOLEAN", {"default": True, "tooltip": "Write the beats — what visibly HAPPENS between each pair of keyframes — and the weights that give each shot its length.\n\nON (the default) is the video path. The beats come out on the 'beats' output in exactly the format 'Morpheus Storyboard' takes, so the arc is planned once instead of twice.\n\nOFF is the slideshow path: use it when the keyframes are going to 'Save Clip' as slides and no video is being rendered. The planning grammar drops the transitions array altogether, so the model cannot spend tokens on directions nobody will read.\n\nOff also swaps the planner's own system prompt for the slideshow one, which is the half that changes what the pictures look like: with no video between them the CUT IS FREE, so the plan stops being one continuous take and starts being an edit — shot sizes jump, scale alternates, and some frames are given nobody at all. Type your own text into 'system_plan' and yours is used in both modes instead.\n\nThe board still carries one shot per gap (blank beat, even length), so it stays a valid chain: turn this back on later, or wire the board to Morpheus anyway and its Storyboard writes the beats itself from the brief."}),
                "cache": (CACHE_MODES, {"default": "disk", "tooltip": "Cache the LLM's answers on disk, keyed causally, so re-running the graph does not rewrite the prompts and invalidate finished frames. Editing the brief re-rolls everything; editing one frame re-rolls that frame and the ones after it."}),
                "live_preview": ("BOOLEAN", {"default": True, "tooltip": "Stream every call to a 'Kinburg Live Log' node as it is written, one labelled block per call ('style bible', 'plan', 'frame 2/7'). Drop a Kinburg Live Log anywhere on the canvas — no wiring. The plan streams too, grammar and all."}),
                "unload_after_run": (UNLOAD_MODES, {"default": "config default", "tooltip": "Whether to free the LLM's VRAM when this node finishes. On a small card set this to 'unload after run': the sampler needs the room, and the writer has nothing left to do."}),
                "system_style": ("STRING", {"multiline": True, "default": STYLE_SYSTEM, "tooltip": "System prompt for the style-bible call. Blank = the built-in default."}),
                "system_plan": ("STRING", {"multiline": True, "default": PLAN_SYSTEM, "tooltip": "System prompt for the planning call. The JSON shape is forced by a grammar built from the frame count, so editing this can change the writing but can never break parsing.\n\nThere are TWO shipped defaults and 'write_beats' picks between them: the continuous-take prompt (shown here) when beats are on, and a slideshow prompt that cuts freely between shot sizes when they are off. Leave this field as it came — or blank — and the right one is used. Type anything of your own and yours is used in both modes."}),
                "system_frame": ("STRING", {"multiline": True, "default": FRAME_SYSTEM, "tooltip": "System prompt for the per-frame prompt calls. Blank = the built-in default."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (PHANTAS_BOARD, "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("board", "prompts", "beats", "durations", "style", "report")
    OUTPUT_TOOLTIPS = ("The board — wire it into 'Phantas'.",
                       "One prompt per keyframe, separated by '---'. Edit a frame and paste the whole thing back into 'prompts_override' to keep it.",
                       "One direction line per shot, in exactly the format 'Morpheus Storyboard' takes for its own `beats` — wire it there and the arc is planned once, not twice. Empty when 'write_beats' is off.",
                       "The shot lengths this board settled on, in the format Morpheus takes.",
                       "The style bible, as written.",
                       "What was written, what came from cache, and what the clock worked out.")
    FUNCTION = "write"
    CATEGORY = CAT_PHANTAS
    DESCRIPTION = ("Turns a brief into a chain of keyframe prompts plus the beats between them. "
                   "Feeds 'Phantas' (which renders the keyframes) and, through it, "
                   "'Morpheus Storyboard' (which writes the video prompts).")

    def write(self, config, brief, count_mode, count, target_length, durations, cast="",
              style_notes="", prompts_override="", preferred_length=timing.DEFAULT_SECONDS,
              write_beats=True, cache="disk", live_preview=True,
              unload_after_run="config default", system_style="", system_plan="",
              system_frame="", unique_id=None):
        cfg = dict(config or {})
        with_beats = bool(write_beats)
        sys_style = (system_style or "").strip() or STYLE_SYSTEM
        sys_plan = _plan_system(system_plan, with_beats)
        sys_frame = (system_frame or "").strip() or FRAME_SYSTEM
        use_cache = cache == "disk"
        # What the SHARED key folds in, as it always has — deliberately the video prompt rather
        # than the one this run uses. The bible does not depend on the plan prompt at all, so
        # toggling write_beats must not re-roll it: a re-rolled bible is a new [CAST] block, and a
        # new cast block is new faces in both versions of the same board. The prompt that was
        # actually used goes on the plan's own key below, where it belongs.
        plan_stamp = (system_plan or "").strip() or PLAN_SYSTEM
        unload_comfy = bool(cfg.get("unload_comfy_models"))
        unload_llm = resolve_unload(unload_after_run, cfg)

        n_frames, n_shots = timing.resolve_count(count_mode, count, target_length, preferred_length)
        if n_frames > MAX_FRAMES:
            raise ValueError(f"a board of {n_frames} keyframes is past the {MAX_FRAMES} limit — "
                             f"split the clip.")
        typed = timing.parse_durations(durations, n_shots)
        if typed is None and target_length:
            timing.check_target(n_shots, target_length)   # fail before the LLM runs, not after

        emit = None
        if live_preview:
            nid = str(unique_id) if unique_id is not None else "phantas"

            def emit(payload):
                try:
                    from server import PromptServer
                    PromptServer.instance.send_sync("kinburg.llm", {"id": nid, **payload})
                except Exception:  # pragma: no cover - a headless run has no server
                    pass

        report = [f"Phantas board: {n_frames} keyframes → {n_shots} shot(s)"]
        first = [True]   # the comfy unload happens once, before the first call of the run

        def ask(system, prompt, tag, grammar=""):
            text, err = _ask(cfg, system, prompt, unload_comfy and first[0], tag, emit, grammar)
            first[0] = False
            if err:
                raise RuntimeError(f"Phantas Storyboard ({tag}): {err}")
            return text

        try:
            env = diskcache.key(_cfg_fingerprint(cfg), brief, cast, style_notes, sys_style,
                                plan_stamp, sys_frame, n_frames)

            # ------------------------------------------------------------------------- the bible
            bkey = diskcache.key(env, "bible")
            hit = _store.load_json(bkey) if use_cache else None
            if hit:
                bible_text = hit.get("text", "")
                report.append("style bible: from cache")
            else:
                user = brief.strip()
                if cast.strip():
                    user += ("\n\nTHE CAST — these people are FIXED. Reproduce this block under "
                             "[CAST] exactly as it is written, word for word, and write the rest of "
                             "the bible around them:\n" + cast.strip())
                if style_notes.strip():
                    user += "\n\nLOOK NOTES (these govern the [STYLE] block):\n" + style_notes.strip()
                bible_text = ask(sys_style, user, "style bible")
                if use_cache:
                    _store.save_json(bkey, {"text": bible_text})
                report.append("style bible: written")
            bible = parse_bible(bible_text, cast)
            roster = parse_cast(bible["cast"])
            report.append(f"cast: {len(roster)} named "
                          + ("(yours, verbatim)" if cast.strip() else "(written by the bible call)")
                          + (" — " + ", ".join(n for n, _ in roster) if roster else ""))

            # -------------------------------------------------------------------------- the plan
            pkey = diskcache.key(env, "plan", n_frames, with_beats, sys_plan)
            hit = _store.load_json(pkey) if use_cache else None
            if hit:
                plan_text = hit.get("text", "")
                report.append("plan: from cache")
            else:
                plan_user = (
                    f"BRIEF:\n{brief.strip()}\n\n"
                    f"STYLE BIBLE (context — do not repeat it):\n{bible_text}\n\n"
                    + (f"THE CAST — refer to these people by exactly these names in 'present':\n"
                       + "\n".join(n for n, _ in roster) + "\n\n" if roster else "")
                    + (f"Plan exactly {n_frames} keyframes and exactly {n_shots} transitions "
                       f"between them. " if with_beats else
                       f"Plan exactly {n_frames} keyframes and NO transitions: these pictures are "
                       f"shown one after another as stills, with no video running between them, so "
                       f"answer with the 'frames' array alone. The whole brief is still spent "
                       f"across the keyframes — the change lives in their states. ")
                    + f"Keyframe 1 is the sequence's first image and keyframe {n_frames} is its "
                    f"last.")
                plan_text = ask(sys_plan, plan_user, "plan",
                                grammar=_plan_grammar(n_frames, with_beats))
                if use_cache:
                    _store.save_json(pkey, {"text": plan_text})
                report.append("plan: written")
            frames_plan, trans, notes = parse_plan(plan_text, n_frames, with_beats)
            report.extend("⚠ " + n for n in notes)
            if not with_beats:
                report.append("beats: off — stills only. The shots are still there, blank, so the "
                              "board stays a chain Morpheus can be given later.")

            # ------------------------------------------------------------------------- the clock
            if typed is not None:
                shot_frames = [timing.frames_for(s) for s in typed]
                report.append("lengths: typed on the node")
            else:
                shot_frames = timing.plan_frames(n_shots, [t["weight"] for t in trans],
                                                 target_length, preferred_length)
                report.append("lengths: even — no beats to weight them" if not with_beats else
                              "lengths: weighted by the planner "
                              f"({', '.join(str(int(t['weight'])) for t in trans)})")
            report.append(timing.describe(shot_frames))
            report.extend("⚠ " + w for w in timing.range_warnings(shot_frames))

            # ------------------------------------------------------------------------ the frames
            override = split_override(prompts_override, n_frames)
            prompts, prev_key, prev_prompt = [], diskcache.key(env, "bible"), ""
            for i, fr in enumerate(frames_plan):
                tag = f"frame {i + 1}/{n_frames}"
                if override and override[i]:
                    prompts.append(override[i])
                    prev_prompt = override[i]
                    prev_key = diskcache.key(prev_key, i, "override", override[i])
                    continue
                # the verbatim descriptions of whoever the plan says is in THIS frame
                here = cast_for(fr["present"], roster)
                if fr["present"] and not here:
                    report.append(f"⚠ {tag}: the plan named {', '.join(fr['present'])}, who is not "
                                  f"in the cast — nothing was stamped for them")
                # causal: this frame's key folds in the previous one, so editing frame 2 re-rolls
                # 2..N and leaves frame 1 on the disk
                key_i = diskcache.key(prev_key, i, fr["framing"], fr["state"], here)
                hit = _store.load_json(key_i) if use_cache else None
                if hit:
                    body = hit.get("prompt", "")
                    report.append(f"{tag}: from cache")
                else:
                    user = (f"STYLE BIBLE:\n{bible_text}\n\n"
                            f"THIS FRAME — framing: {fr['framing'] or 'as the look implies'}\n"
                            f"THIS FRAME — state: {fr['state']}\n")
                    if here:
                        user += ("\nIN THIS FRAME — describe each of them by name, in full, using "
                                 "these words:\n" + "\n".join(here) + "\n")
                    elif roster:
                        user += "\nNobody from the cast is in this frame. Do not put a person in it.\n"
                    if prev_prompt:
                        user += ("\nTHE FRAME BEFORE THIS ONE (same world, keep everything the "
                                 f"state above does not change):\n{prev_prompt}\n")
                    user += "\nWrite this frame's prompt."
                    body = ask(sys_frame, user, tag)
                    if use_cache:
                        _store.save_json(key_i, {"prompt": body})
                prompts.append(assemble(bible, body, fr["framing"], here))
                prev_prompt = body
                prev_key = key_i

            board = {
                "frames": [{"framing": f["framing"], "present": f["present"], "prompt": p}
                           for f, p in zip(frames_plan, prompts)],
                "shots": [{"beat": t["beat"], "weight": t["weight"], "frames": n}
                          for t, n in zip(trans, shot_frames)],
                "bible": bible,
                "negative": bible["negative"],
                "links": ["continue"] * n_shots,
            }
        finally:
            if unload_llm:
                _shutdown_worker()
        if use_cache:
            _store.prune()

        beats = ("\n".join(f"{i + 1}. {t['beat']}" for i, t in enumerate(trans))
                 if with_beats else "")
        durs = ", ".join(f"{timing.seconds_for(n):.2f}" for n in shot_frames)
        style = "\n\n".join(f"[{k.upper()}]: {v}" for k, v in bible.items() if v)
        return (board, PROMPT_SEP.join(prompts), beats, durs, style, "\n".join(report))


NODE_CLASS_MAPPINGS = {"KinburgPhantasStoryboard": KinburgPhantasStoryboard}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgPhantasStoryboard": "Phantas Storyboard 🎞"}
