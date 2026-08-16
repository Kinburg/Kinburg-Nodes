# 🎬 Video Generation & Storyboarding

<!-- index-order: 4 -->

[← back to the node index](../README.md#-node-index)

---

## 🌙 `morpheus/` — Morpheus Suite 🌙

> **System Purpose & Overview**  
> Video generation sampler, LLM-driven storyboard chain planner, and conversation-driven animation suite.

MiniMax H3 generates **5–15 seconds** per run and ComfyUI ships no extend/continue node for it, so the
only route to a minute of video is to run it several times and hand **the last frame of each shot to the
next one as its first keyframe**. That loop is what this suite is — named for the god who *shapes*
dreams, because dreams flow into one another instead of starting and stopping, and because the seam
between two shots is called a **morph** in video production anyway.

**`Morpheus Dream`** is one link: a prompt, a duration, and optionally a `start_frame` / `end_frame`.
Chain them exactly like `Sampler Settings` — wire the `shots` output into the next node's `shots` input;
left→right is shot order. How a shot's first frame is decided is the whole design:

1. `start_frame` wired → **that image** (always wins),
2. else `link = continue` → the last **generated** frame of the previous shot,
3. else (`link = cut`, or shot 1) → **no keyframe**: pure text-to-video.

So three keyframes and two shots ("1→2", "2→3") work; so does a chain with no images at all, where
shot 1 is text-only and every later shot inherits the previous tail; so does any mix — and `cut` is
there to put a real montage cut in the middle of one. `duration` is in **seconds** and snaps up to the
model's grid (17k+5 frames at 24 fps → steps of 0.71 s); the `info` output prints the length you
actually get. `seed_offset` re-rolls one shot without disturbing the ones before it.

**`Morpheus (Video Sampler)`** resolves the chain, samples every shot, decodes it, feeds the handoff
frame forward and returns **one IMAGE batch + one AUDIO track + fps + the final frame + a report**. It
exists because six things have to be right that a hand-wired graph gets wrong:

- **fps is not a parameter.** 24 is baked into the model — both the frame grid and the audio latent
  length are derived from it — so duration is given in seconds and **24.0** comes back out as a `FLOAT`,
  the type `Create Video` takes, so `images` / `audio` / `fps` drop straight in with no conversion node.
  An editable "fps" widget on a node like this would be a lie.
- **one progress bar.** Total = shots × steps, advanced monotonically, so it never fills up and resets
  per shot (which is exactly how a first real run reads as "nearly done" four times over). A cached shot
  jumps its slice at once.
- **loudness.** `VAEDecodeAudio` normalises **per decode** (`std × 5`), so decoding shots one at a time
  steps the level at every seam. Here the normalisation is computed once over the finished track, and
  seams get equal-length fade-out/fade-in ramps (`seam_fade_ms`) — deliberately **not** an overlapping
  crossfade, which would shift the sound against the picture.
- **the duplicated seam frame — and the re-acceleration behind it.** A continuing shot's first frame
  *is* the previous shot's last frame, so `seam_trim` drops it (default 1) and 1/24 s comes off that
  shot's audio to match. Raise it to 3-6 and it also swallows the **ease-in**: a keyframe carries
  position but no *velocity*, so the model starts every shot's motion from rest and the subject visibly
  speeds up again at each seam. Trimming happens at decode time, so tuning it re-uses cached shots and
  costs nothing. The prompts fight the same problem from the other side — every `[END STATE]` names the
  motion at that instant, and the shot rules forbid "begins to" / "picks up speed" in the opening beat.
  Each shot's audio slot is
  measured from the storyboard's **cumulative** frame position, not its own length — rounding each shot
  independently drifts up to half a sample per seam, which walks the sound off the picture over a long
  chain.
- **RAM.** 1344×768 float32 is 12.4 MB **per frame** — ~1.5 GB per 5-second shot. The output batch is
  pre-allocated once and filled shot by shot, so the peak is the final tensor plus one shot instead of
  double.
- **iterating.** Sampled **latents** are cached on disk (~7 MB per shot, versus ~1.5 GB of pixels) under
  `user/kinburg-nodes/morpheus_dreams`, keyed **causally**: a shot's key folds in the previous shot's,
  because its first frame *is* the previous shot's output. Editing shot 5 of 8 re-samples 5–8 and
  replays 1–4 off the disk. `shots_range` ("", `3`, `2-4`) narrows the output further — shots after the
  range are skipped outright, shots before it are needed only for the handoff frame and are free when
  cached. A cache write that fails goes in the **report**, not only in the log: the first real run wrote
  nothing for 25 minutes because safetensors rejects non-contiguous tensors (a video VAE hands frames
  back as a permuted view and `.to()` keeps those strides) and the only trace was one log line.
- **the canvas is exactly what you typed.** `width` / `height` are rounded to 32 and otherwise obeyed —
  no auto-sizing. An earlier version derived the canvas from the first keyframe, which is *technically*
  the better default (H3 **stretches** the first frame onto the canvas and never crops it, so a
  mismatched aspect ratio distorts the whole shot) but cost a run at 1344×768 that was meant to be at
  960×544 — three times the time, silently. So the aspect check stayed and became a **report warning**,
  and the size stayed where everyone expects it.

- **`lora_triggers`** — wire the **`triggers`** output of `Lora Unlim Accumulator` here and the
  trigger words go into **every** shot's prompt. Not simply appended: a Morpheus prompt is MiniMax's
  six numbered sections and the last one is `[Negative Prompt/Constraints]`, so text on the end
  reads as one more thing to *avoid* — they are inserted just before that section instead. They are
  also re-applied after the in-loop writer has reworked a shot, since a rewrite can drop them, and a
  trigger already present (case-insensitively) is never repeated. Because they go in *before* each
  shot's cache key is taken, changing them re-samples rather than replaying latents made without them.

### The in-loop writer — the forecast, removed

`Morpheus Storyboard` writes every shot before anything is sampled, so a shot whose first frame is
*inherited* is written against a **forecast**: the previous shot's `[END STATE]` sentence, the writer's
own guess about a frame that does not exist yet. When the guess and the rendered frame disagree, the
shot is conditioned on one state and told about another — the same prompt-versus-conditioning conflict
that caused the arc-replay bug, arriving by a different road.

Wire a vision `llm_config` into the sampler and the guess disappears. By the time shot N is about to be
sampled, shot N-1 has been decoded, so the **real** first frame is in hand, together with the planned
last keyframe — which turns every continuing shot from "text mode, blind" into the two-image job, the
mode that works best. Per-shot control is the `refine` widget on `Morpheus Dream`; `auto` reworks
exactly the shots whose first frame the writer never saw, and the Storyboard node stamps that flag
itself, because it knows what it could see. Two scopes:

- **opening** (what `auto` picks) — only `[Scene Overview]` and the first beat are rewritten. The
  forecast was the *only* wrong thing; the pacing, camera, audio and target were planned against real
  information, and the style/negative sections must stay byte-identical anyway. ~80 tokens instead of
  ~350.
- **full** — the shot's own five blocks are written from scratch, keeping sections 1 and 6. For shots
  that never had a real prompt.

The written text is cached under a causal key, because **the prompt is the sampler's cache key**: text
that came back one character different on a re-run would re-sample a five-minute shot. With nothing
wired, the sampler behaves exactly as it always did, and the `prompts` output always shows the text that
actually made the video — reworked or not.

Memory, on purpose: the default kills the LLM worker after every call and frees ComfyUI's models before
it, because on 12 GB a 26B vision model and H3 cannot coexist. That costs one H3 reload per seam;
`llm_keep_loaded` trades the safety for the speed if you have the headroom. The report breaks out
**write** and **sample** time per shot so the trade is visible rather than guessed at, and the writing
streams to `Kinburg Live Log` like everything else (`refine 2/4 (opening)`) — with the frames it was
shown, plus each shot's last frame as it is decoded, so the log doubles as a live storyboard of the
run. That works with `live_preview` on whether or not an LLM is wired.

The `MiniMax H3 Sigma Shift` patch (video 12 / audio 3) is applied inside the node unless the wired model
already carries it, and wiring your own `sigmas` into a model that carried no shift gets a warning,
because that schedule was almost certainly built against an unshifted model. `sigmas`, `sampler` and
`noise` are all optional overrides; without them the node builds the schedule from `steps` / `scheduler` /
`sampler_name` and seeds each shot with `seed + shot index + seed_offset`.

Known limits, all from the model rather than the node:

- **motion resets slightly at every seam**, because the model is conditioned on a still frame and never
  on a velocity — free for a hard cut, a small hitch in a continuous take.
- **there is no time addressing inside a shot.** Timestamps exist only for *reference* videos; the
  generated shot gets a prompt and its two end keyframes, and nothing binds "at 2 s" to a moment. If a
  transformation has to land at a particular beat, split it into two shots with a keyframe between them —
  more keyframes, not a more elaborate prompt.
- **soundtracks do not continue across shots.** Each shot's audio is generated from scratch, so the
  ramps hide the click but not the change of music. Until the reference path lands, the clean answer is
  `audio = mute` plus a single continuous track from `Siren` under the whole thing.
- **references are not wired up yet** (`MiniMax H3 Reference to Video`): `comfy/model_base.py` lets
  `minimax_refs` overwrite the keyframes' `cond_video_latents`, so a shot can be fl2va **or** ref2va,
  never both — carrying a character's identity across a long chain is the handoff frame's job for now.

Measured on the author's hardware, for scale: **20 s at 960×544 (4 shots) ≈ 25 min**, and the same shot
at 1344×768 takes roughly three times as long as at 960×544.

### `Morpheus Storyboard 🌙` — the LLM writes the whole chain

Doing it by hand goes: ask an LLM to expand the idea → generate a keyframe sheet → cut it up → ask a
vision LLM about each consecutive pair → paste the answers into N `Morpheus Dream` nodes. This node is
steps three-through-five: **idea (+ whatever keyframes you have) in, a finished chain out.**

**Keyframes are consumed as shot boundaries, left to right** — one rule, no modes, and it covers every
mix. With `K` frames and `N` shots, boundary *i* is known while *i < K*:

| K | what you get |
|---|---|
| 0 | every shot is text-only; shot 1 is pure text-to-video |
| 1 | a hard opening frame, then free fantasy on text alone |
| N+1 | every shot bounded by two hard frames |
| between | the first K−1 shots are bounded, then it runs on text |

`shot_count = 0` derives the count from the frames (K → K−1 shots); set it higher to keep going after
the frames run out. Up to 64 shots.

The prompts come out in **MiniMax's own recommended format** — the numbered `[Style and Aesthetic]` /
`[Scene Overview]` / `[Storyboard]` / `[Camera]` / `[Audio & Voice]` / `[Negative]` sections with
`[0s-1.5s] Beat n:` lines inside a shot. Three facts about that format drive the whole design:

- H3 reads those timings as **pacing and order, not as a clock** — nothing binds "at 2.0 s" to a frame.
  So beats get written, exact seconds are never promised, and real timing control comes from where the
  **shot boundaries** fall. If a transformation has to land on a beat, split the shot.
- **a shot must never be told the whole story.** H3 acts out whatever the prompt describes: hand it a
  first-frame keyframe *and* a scene overview that says "a cyclist becomes a demon", and it holds the
  keyframe for two frames, then rewinds and replays the entire arc inside those five seconds. (Measured:
  the seam latents proved the keyframe was applied — cosine 0.83-0.89 against the previous shot's tail
  versus 0.56 against a non-adjacent one — while the picture went its own way.) So section 2 is
  **per shot**: the bible contributes only the invariant `[SUBJECT]`, the shot adds its own situation,
  and an anchored shot gets an explicit forward-only clause plus anti-rewind negatives.
- the `[Style]` and `[Negative]` blocks have to be **identical in every shot** or the look drifts
  mid-sequence. So they are written **once** by a text-only "style bible" call and stamped onto every
  shot. Those two are all that gets stamped: the bible's `[Subject]` line stays behind as context for
  the writer, because in a morph sequence **the subject is what changes** — stamping "a cyclist on a
  road bike" into a shot that shows a demon on a motorcycle is a contradiction the model has to
  resolve, and it resolves it by turning the demon back into a cyclist. The bible comes out on the
  `style` output; paste it into the `style` input of a later run to keep one look across sessions.

**A spoken line lives in exactly one place: the beat that speaks it.** MiniMax's own template puts
voice-over in `[Audio & Voice]`, but measured on real renders the line lands far better inside its beat,
with its timing — `(Male, 30s, gravelly, urgent, American) "They found me."` — and a line written in
*both* blocks is sometimes performed **twice**, or lands at the wrong moment. So the shot prompts put
speech in the storyboard and forbid quoting it in the audio block, and the assembler drops from the
audio any sentence that repeats a line already spoken in a beat (whole sentences only, so what is left
still reads; if that empties the block, the bible's sound bed fills in).

**The prose is English; the dialogue is not.** H3 speaks other languages, so a quoted line keeps the
language and alphabet it was written in, carried through the planner and the shot writer verbatim —
`(Female, 30s, flat, resigned, Russian, slow) "Он не придёт."` — with the language filling the voice
spec's accent slot so the model knows how to say it. Only if the direction *describes* speech without
quoting it does the writer invent the words, in the language the direction itself is written in.
Everything else — description, camera, sound, negatives — stays English, which is what the model wants.

Each shot is written by **one of three system prompts**, picked by how many keyframes that shot got —
two images ("describe the change that carries the first into the second"), one ("carry on from this
state, there is no target"), none ("invent it from the direction and the previous end state"). They are
three genuinely different jobs, and one prompt with conditionals makes a small local model hedge. Each
has its own override widget, and the mode shows up in the live-log label (`shot 2/4 (2 keyframes)`).
Each carries a **worked example** of the five blocks — which is what does most of the work on a local
model — and each example is deliberately shaped like a *slice*: one continuous take, two or three
beats, no cuts. (Examples written as trailers, which is the natural way to write them, teach the model
to pack a whole story with hard cuts into five seconds.) Output is forced to English regardless of the
language you write the brief in, and the beat labels say "Beat", not "Shot", for the same
anti-cutting reason.
The per-call context is deliberately thin — length, the director's note, and a starting state only when
no image shows it. An earlier version passed the whole brief and bible into every call, and the model
dutifully wove all of it into its answer.

Continuity without keyframes comes from the shots themselves: every call also returns an `[END STATE]`
sentence describing its own last frame, which is handed to the next shot as its starting situation. That
is what stops a text-only chain from wandering off.

Before a single shot is written, a text-only **planning** call breaks the brief into **one direction per
shot** (`script = auto`). This is not a nicety: without it, a shot with no line of its own was handed the
*entire brief* as its direction, and a shot told the whole story tells the whole story — the third shot
of a text chain, given the same instruction as the second, replayed the arc compressed and then added its
own part. Each line is two or three concrete sentences saying what happens in that shot and the state it
must arrive at, sized to that shot's duration (a 10-second shot carries twice the change of a 5-second
one). Hand-written `beats` win and skip the call; the plan comes out on the `script` output in exactly
the format `beats` takes, so editing one line and pasting it back rewrites that shot onwards and nothing
else. The brief now never reaches a per-shot call by any route.

Two text fields do the steering, both matched to shots **by position**:

- **`beats`** — one line per shot of direction ("shot 3: he crashes through a billboard"). Blank lines
  are legal and mean "leave this one to the LLM", so they are *not* stripped.
- **`prompt_overrides`** — finished prompts that bypass the LLM entirely, separated by a line of `---`.
  The `prompts` output uses the same format, so the loop is: run once, read it, fix the one shot that
  came out wrong, paste it back. An overridden shot costs no LLM call.

`anchor` decides what the sampler is conditioned on when a keyframe exists: `continuous` (default) wires
only the **end** keyframe and lets the shot start from the previous shot's generated tail — seams are
exactly continuous and the shot is still pulled to its planned frame by its end; `plan` wires both, for
tighter storyboard adherence at the price of a small jump at each seam. Either way **the LLM sees both
frames** — this is about conditioning, not about what gets described.

Everything the LLM writes is cached on disk beside the latents, under the same kind of **causal** key,
because the prompts *are* the sampler's cache key: a prompt that changed on every run would re-sample
every shot at minutes apiece. Which also makes iterating cheap — re-word shot 2's beat and shots 2..N
get rewritten while shot 1 (and the style bible) replay untouched. The `seed` deliberately has no
"control after generate" for the same reason.

The LLM plumbing is the `Vision Judge` one (`build_llm_request` + `_generate_and_format`), so it takes
the same `Local LLM Settings (GGUF)` bundle; attach a `Vision Settings (GGUF)` mmproj if you wire
keyframes, or it writes blind (and says so in the report). `unload_after_run` defaults to **unloading**,
because what runs next is H3 plus a 30B text encoder. `live_preview` is **on** by default and streams
every call to a `Kinburg Live Log` node — one labelled block per call (`style bible`, `shot 2/4`), over the
same `kinburg.llm` channel the Local LLM node uses, so no new node and no wiring: writing a storyboard is
otherwise minutes of silence.

**`links`** (advanced) overrides `link` per shot, the same comma-list shape `durations` takes
(`continue, cut, continue`, last value repeating). Only shots with no start keyframe listen to it — a
wired frame always wins — so it is how you put one hard cut into an otherwise flowing sequence. Leave
it blank and every shot follows the single `link` widget; that case keeps the cache key byte-identical
to before the option existed, so adding it re-wrote nobody's prompts and re-sampled nobody's video.

### `Morpheus Dream Board 🌙` — a conversation becomes a storyboard

The bridge from `Local LLM Chat (GGUF)` to Storyboard. You chat with a character, pictures pile up in
the conversation, and this node turns the part you pick into a storyboard.

**Wire one output: `shots` → Storyboard's `shots`.** Every shot goes across carrying its own
keyframes, length, link mode and direction, with an **empty prompt** — which is what tells Storyboard
"this one is yours to write". Leave its `keyframes` / `durations` / `links` / `shot_count` alone.

That is worth more than the four wires it replaces. Parallel lists line up only by position, so
adding a line to `beats` in a Show Text silently shifts every later shot onto the wrong keyframe. And
Storyboard's `keyframes` batch is consumed left to right with no gaps, whereas a shot in a chain
names its own start and end — so a **text-only shot can sit between two keyframed ones**, which the
batch could never express.

The four separate outputs are still there if you prefer the explicit wiring: **`keyframes`**,
**`beats`**, **`durations`**, **`links`** and **`shot_count`** (wire that one too in that case —
Storyboard's own default stops at the keyframes and would skip the text-only shots at the end).
`beats` is worth taking either way: route it through a `Show Text` to read or hand-edit the
directions, and feed it into Storyboard's `beats`, which overrides the chain line by line.

**Pictures define the shots, not the other way round.** A Morpheus keyframe physically sits *between*
shots — frame 2 ends shot 1 and starts shot 2 — so K picked pictures give K−1 bounded shots and the
messages between two pictures are that shot's direction:

```
picture 1 ····· messages ····· picture 2 ····· messages ····· picture 3
          └──────  shot 1  ──────┘        └──────  shot 2  ──────┘
```

Nothing has to be forbidden or auto-dropped: a shot can't hold three keyframes, because a shot *is*
the span between two. Messages before the first picture join shot 1 (that's where a scene gets set
up); messages after the last one become text-only shots, which you split with **+ break** markers.
The cost — and it is this version's real limit — is that a shot boundary must land on a picture: a
long stretch of story with no picture in it can't be cut into several shots, because the interior
boundary would have no keyframe. If a span runs past H3's ~15 s, put another picture in the chat.

**Press ⟳ Update History** to pull the conversation in; **`→ chat`** picks which chat node when there
is more than one. There is no graph link on purpose — a link would sit below the chat's blocked
output, so rendering a video would need the chat to run first, and this way the node also works when
you run it on its own. Each message has a tick (does its text become direction) and each picture has
one (is it a boundary), and they are **independent**: a persona's bubble is usually text you don't
want and a picture you do. Re-pulling after more conversation keeps your ticks.

Turns of a **private** persona start unticked — the "camera" pattern, whose whole job is writing
image prompts rather than story. That goes **by persona, not by role**: your own messages carry
whichever persona was selected, so "tighter, more bokeh" typed at a camera is dropped along with its
reply, while your dialogue with the character keeps the character's name and stays. It has to — you
may well be playing a second character who is in the video, and dropping your half would leave the
first one talking to itself. (Same reasoning as the chat's own privacy rule, which withholds a
private reply *and* the instruction that produced it.)

The shot list under the chat is **derived live**, so there is nothing to "create" and no state that
can contradict itself: tick a picture and a shot appears. Per row you get `duration` (snapped to
H3's 0.71 s grid when it runs) and, for the text-only shots, `link`.

**`beats` are verbatim for now** — the picked messages, whitespace-collapsed and speaker-labelled.
Storyboard skips its own planning call when `beats` is filled ("your lines win"), so the shot writer
reads exactly what you selected. Route `beats` through a `Show Text` node first if you want to read
or hand-edit it before it is written up.

---

[← back to the node index](../README.md#-node-index)
