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

- **`trims`** (advanced) — frames dropped from the **tail** of each shot, a comma list with the last
  value repeating, blank by default and byte-identical to before when blank. This is where Orpheus'
  output goes: H3's lengths move in a 0.708 s quantum and bar lines do not, so a cut that must land
  on a downbeat is reached by generating the first legal length *longer* than the music needs and
  dropping the overshoot. `seam_trim` takes frames off the head for a different reason; these two
  share the shot, and a shot is never trimmed below 5 frames — the report says when it was clamped.
  The next shot's first frame moves to the cut point with it and the audio slot shrinks to match, so
  picture and sound stay together. Free to change, like `seam_trim`: it happens at decode, so cached
  latents replay.

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

## 🎞 `phantas/` — Phantas Suite 🎞

> **System Purpose & Overview**  
> Keyframe storyboard writer and multi-frame sampler: a brief becomes a chain of consistent stills,
> and that chain becomes the boundaries Morpheus dreams between.

Phantasos is Morpheus' brother, and in the myth he is the one who appears as still things. That is
the division here too: Morpheus makes motion, Phantas makes the pictures the motion runs between.

A Morpheus keyframe sits **between** shots — frame *k* ends shot *k−1* and starts shot *k* — so a
board of N pictures is a chain of N−1 shots, and the two numbers are always one apart. Everything
below follows from that one fact.

The pipeline is two nodes plus the two you already have:

```
Phantas Storyboard 🎞  →  Phantas 🎞  →  Morpheus Storyboard 🌙  →  Morpheus 🌙
   writes the states      renders them      writes the video prompts    renders the video
```

**The arc is planned once.** `Phantas Storyboard` already has to work out what happens between each
pair of pictures, so it emits those lines as `beats` in exactly the format `Morpheus Storyboard`
takes — and that node skips its own planning call when `beats` is filled. Wire them together and one
plan governs both halves; leave them apart and two LLM calls get independent authority over the same
story, which is a good way to watch them disagree.

### `Phantas Storyboard 🎞` — the brief becomes states and beats

Morpheus writes *shots*; this writes *states*. Three LLM calls, all of them streaming into a
`Kinburg Live Log` node:

1. a **style bible** — `[STYLE]`, `[CAST]`, `[SUBJECT]`, `[NEGATIVE]` — written once and stamped on
   every frame byte-for-byte;
2. a **plan**, GBNF-constrained to exactly N keyframes and N−1 transitions — or to the keyframes
   alone, see `write_beats` below — so a model that miscounts cannot even emit the wrong number of
   entries;
3. one call **per frame**, shown the bible, its own framing and state, whoever is in it, and the
   previous frame's prompt, so consecutive pictures are of the same world.

**The `cast` is what decides whether a character survives the sequence.** An image model has no
memory between calls: "the singer", or "the man from the previous shot", is a different human being
in every picture. Identity has to be re-stated in full, *in the same literal words*, in every frame
the person appears in — which is exactly why cover art comes out consistent across seeds and songs
while a storyboard written from the same context does not. So:

- Type (or wire) `cast` as one `Name — full description` line per person, at the length you would
  describe them for cover art. Those lines are the authority: they replace whatever the bible call
  wrote, and they are stamped **verbatim**.
- The plan says who is `present` in each keyframe, and only those people's lines are stamped into
  that keyframe's prompt. Pasting the whole cast into a shot of an empty room is how a picture grows
  a person who should not be in it.
- Leave `cast` empty and the bible call writes one itself — from the brief and from whatever the LLM
  Settings' `context` holds, which is the other way a cast gets here: a `Context Collector` or a
  `Card Presets` bundle on the settings node reaches every call. Leave it empty **also** when the
  subject is supposed to change, since a fixed description would contradict the transformation.

Two more rules are baked into the prompts and are the reason the output holds together:

- **A frame prompt is a frozen moment.** "He begins to transform" asks an image model for a motion
  smear. The change lives in the beat; the frame is the state it arrives at.
- **The chain is one continuous take.** With no cuts, framing cannot jump, so a crop change has to
  be a camera *move* — the plan assigns framings to the boundaries and each shot performs the move
  between two of them.

**Slideshow boards.** `write_beats` (on by default) is the video half of the plan: the beats and
the weights that set the shot lengths. Turn it off when the keyframes are going to `Save Clip` as
slides and nothing downstream renders video — those directions would be written for nobody. What
saves the time is the grammar, not an instruction: the transitions array is simply gone from the
shape the model may emit, so it cannot spend a token on it. On a thirty-keyframe board that is
twenty-nine paragraphs the planner never writes. The `beats` output comes out empty and the shot
lengths come out even, but the board still carries one blank shot per gap, so it stays a valid chain
— switch it back on later, or hand the board to Morpheus anyway and `Morpheus Storyboard` writes
the beats itself from the brief.

The switch also **swaps the planner's system prompt**, and that is the half that changes what the
pictures look like. The continuous-take rule exists because a video model has to *travel* from one
framing to the next, so a crop change has to be a camera move. Nothing travels between two slides:
the cut is free. Planned under the video rules, a slideshow comes out as thirty variations on one
camera position, so the stills prompt asks for an edit instead — scale alternating, no two
neighbours framed alike, and some frames deliberately given nobody at all, because a slideshow needs
air the way a continuous take does not. Both shipped prompts count as "the default", so an untouched
`system_plan` follows the mode; type your own text there and yours is used in both.

**Counting.** `count_mode` picks the unit, and the units are the same variable:

| `count_mode` | you type | you get |
|---|---|---|
| `frames` | how many pictures | N pictures → N−1 shots |
| `scenes` | how many shots | S shots → S+1 pictures |
| `duration` | `target_length` in seconds | the shot count, from the clock |

**The clock is integer arithmetic, and the LLM is kept out of it.** H3 runs `17k+5` frames at 24 fps
and is trained on 124–362 of them, so a shot may be exactly one of **fifteen** lengths, 0.708 s
apart, from 5.17 s to 15.08 s. The planner therefore never names seconds — it gives each transition
a *weight* for how much visible change it carries, and those are laid onto the grid here. A model
that writes "5.2 s" gets 5.88 s from the grid and the total drifts silently; a model that writes
weights cannot be wrong about time.

Two consequences worth knowing before you type a number:

- Fill `durations` (`"5.17, 8"`, last value repeating) and your numbers win outright. Leave it empty
  and the planner's weights decide, with `preferred_length` as the average shot.
- `target_length` and the shot count **over-determine each other**: n shots can only add up to
  between n×5.17 s and n×15.08 s. Twelve seconds of four scenes is not a preference to be clamped,
  it is a contradiction, and it is raised as one — before a single token is generated. Inside the
  band, any target is reachable to within ±0.35 s, because every shot moves in the same quantum.

Everything written is cached on disk under a causal key, so re-running the graph does not rewrite
the prompts and invalidate finished frames. Edit the `prompts` output, paste it back into
`prompts_override` (frames separated by a line of `---`), and those frames are used verbatim without
an LLM call — an empty entry means "write this one". The system prompts are editable fields
(`system_style`, `system_plan`, `system_frame`); the plan's JSON shape comes from the generated
grammar, so editing them can change the writing but never break parsing. `system_plan` is the one
with two defaults, picked by `write_beats` as described above.

Give this node a **light text-only** model. It never looks at a picture, and on a small card the
VRAM it does not take is VRAM the sampler gets.

### `Phantas 🎞` — rendering the keyframes

Walks the board once, one still per keyframe, and returns the `shots` chain with the prompts left
empty — which is what tells `Morpheus Storyboard` "these are yours to write". Neighbouring shots get
the **same tensor object** for the frame between them, because it is one picture playing two parts.

**Consistency here is structural, not cosmetic.** A shot is generated as the movement from its first
keyframe to its last, so two neighbours that disagree do not look "slightly different" — they
describe a shot that morphs halfway through. A shared seed is the *weakest* lever (same starting
noise, but the trajectories diverge within a few steps); it is the floor, not the mechanism. What
actually holds a board together, in order: the stamped style block, then an **anchor**.

`reference` picks the anchoring mechanism, and both are stock ComfyUI because they are what
different model families actually offer:

| `reference` | how | which models |
|---|---|---|
| `off` | text and seed only | anything |
| `redux` | CLIP-Vision → `StyleModelApply` | the Flux dev family, Krea included |
| `edit` | the anchor VAE-encoded as a `reference_latents` entry | Kontext, Qwen-Image-Edit |

`redux` is applied as an `attn_bias`, which is what `reference_strength` dials: at 1.0 Redux tends to
redraw its reference and every frame comes out the same picture, so below 1.0 keeps the identity
while leaving the prompt in charge of the shot. It needs `style_model` and `clip_vision` wired.
`edit` needs no extra inputs but does mean running an edit model.

`anchor` picks *which* picture: `first frame` holds global identity but lets neighbours drift,
`previous frame` holds the seam but accumulates drift down a long chain, `first + previous` does
both for one more encode. A wired `reference_image` is added on top of all of them and is the only
anchor frame 1 can have. Wire `first_frame` to use a picture you already have *as* keyframe 1 — a
photo, an earlier render, something out of a chat — and everything after it is anchored to that.

Rendered frames are cached on disk under a causal key, and that is not only a speed feature:
**ComfyUI's Cancel raises inside the sampler and discards the run**, so without a cache a stopped
board loses every frame it had finished. With one, Cancel *is* the stop button — cancel, re-run, and
the finished frames come straight back. It is also what makes `redo` cheap: name the frames you want
re-rolled (`"3"`, `"2-4"`, `"1,4-6"`) and they are generated again with their seed shifted by
`redo_seed_offset`, since the same seed and the same prompt would only hand back the same picture.
The frames *after* a re-rolled one render too — they were anchored to the picture it used to be —
while the ones before it stay cached.

`sampler_settings` is the shared `Sampler Settings` bundle, so chaining two of them gives every frame
a draft-then-polish pass, exactly as in Chimera and Ouroboros. `width` and `height` live on this node
rather than on the bundle: match the aspect ratio you will render the video at, since these pictures
are also what the video model's writer looks at. `captions` and `settings_data` go straight into
`Image Compare`, and each keyframe is pushed to a `Kinburg Live Log` node the moment it decodes.

---

## 🎶 `orpheus/` — Orpheus Suite 🎶

> **System Purpose & Overview**
> Reads a finished track and decides where the cuts fall, so a music video is edited to its own
> music instead of to a shot length somebody typed.

Orpheus is the one whose music moved everything else, and that is exactly the join this suite makes:
Siren writes the song, Phantas makes the pictures, Morpheus the motion — and this decides *when*.
It sits in front of the other two:

```
Siren 🧜  →  Orpheus 🎶  →  Phantas Storyboard 🎞  →  Phantas 🎞  →  Morpheus 🌙
 the song    where to cut      what each picture is     renders them    renders the video
```

Wire its `durations` output into `Phantas Storyboard`'s `durations` (or Morpheus') and the shot
boundaries stop being a guess.

### The one thing that decides the whole design

**H3's grid and the musical grid do not fit together.** A shot may be exactly one of fifteen
lengths, 0.708 s apart; a bar at 128 BPM is 1.875 s. No whole number of bars is a legal shot length
at that tempo — none of the six that fit the band. So if all you may choose is a shot *length*, a
cut that has to land on a downbeat arrives up to **±0.354 s** away, which at 128 BPM is most of a
beat: the difference between "cut to the music" and "the editor missed".

So the cut is never chosen from the grid. It is chosen **on the music**, H3 is then asked for the
smallest legal length that is *at least* that long, and the overshoot comes back as a per-shot trim
in frames, to be dropped at assembly. Generate long, cut on the beat. Measured across three hundred
random songs — random tempo, time signature, cut unit and downbeat phase — every musical cut lands
within **20.8 ms** of its line, which is half a frame at 24 fps and therefore the floor.

The trim is not free, and how expensive it is moves with the tempo. At 120 BPM a 4-bar phrase is
8.00 s and sits exactly on H3's grid: nothing is wasted. At 128 BPM the same phrase is 7.50 s and
snaps up to 8.00 — 6.2% of everything generated thrown away — while *two* phrases is 15.00 s against
a 15.083 s ceiling and wastes 0.6%. Ten times the difference, hiding inside one dropdown, so the
report prints the whole table for the tempo you actually have.

### `Orpheus (Audio → Shots) 🎶`

**With a Siren `plan` wired, section boundaries are not detected at all — they are read.** The table
you wrote says where the chorus starts, exactly, with its name; detection then only fills in the
accents inside each section. That is the normal case for your own songs and no detector can beat it.
Without one, everything is measured: onsets from spectral flux, tempo by autocorrelation, the
downbeat by the phase that carries the most onset energy, and the structural seams — the drop, the
moment the drums enter — from how unlike the next few seconds are to the last few.

Every estimate comes back with a **confidence**, and the report prints it. A ballad with no rhythm
section has no beat grid to find, and saying so is the only honest answer; type the `bpm` when you
know it, which for a song you generated you always do.

Two inputs are not what they look like:

- **`pace` is a bias, not a length.** It decides how many shots the rest of the track should become;
  the cut then goes to the best *available* moment, which is never exactly that number. On a 7.5 s
  phrase grid, 7 and 9 give byte-identical plans — there is nothing between the candidates to
  choose. Move it in whole seconds and watch the shot count.
- **`sensitivity`** is how far above its own neighbourhood a moment must stand to be a candidate.
  Low leaves the planner spoilt for choice, so shots land near `pace`; high leaves it reaching for
  the few big moments, so shots stretch. It does not touch section boundaries that came from a plan.

**`cues` is deliberately not `beats`.** `Morpheus Storyboard` skips its own planning call when
`beats` is filled — your lines win — so wiring a list of musical facts into it would hand the shot
writer "0:48 — drop" as an entire director's note, and it would film exactly that. Read `cues`
through a `Show Text` and merge the lines that earn it into `beats` yourself.

The **`scope`** output is the check that costs half a second instead of a render: the spectrogram
with the plan drawn on it, amber where a cut landed on a real moment and dim red where the planner
had nothing to cut on and fell back to length alone. If the amber lines do not sit on the transients,
nothing downstream is worth rendering. A row of red ones is a `sensitivity` or `cut_on` problem, and
is otherwise invisible until the video looks wrong.

`start_sec` / `end_sec` cut a window out of the track — to skip a long intro, or to make a teaser —
while the *whole* track is still analysed, because tempo and downbeat are measured far more reliably
over three minutes than over twenty seconds. `trims` is the per-shot frame count to drop; wire it
into Morpheus, or read it and trim at assembly.

---

## 🎬 `save_video/` — Save Clip

> **System Purpose & Overview**  
> Turn a picture (or a slideshow) plus a song into an mp4, with the pictures changing where the
> music does.

**`Save Clip`** is the video sibling of **Save Song**: same counter-based naming under
`ComfyUI/output`, the same "quality in words, not codec flags" dropdown, the same `Song Tags` going
into the file itself, and a player on the node when it is done. Required are an **`image`** (one
frame, or a batch of slides) and an **`audio`**; it writes an **h264 + AAC mp4** and outputs the
`video` object (so it also chains into ComfyUI's own video nodes), the saved `path` and a `report`.
Category `Kinburg-Nodes/video`.

**The batch is the list of pictures, not the list of frames.** A 4-image batch over a 3-minute song
is four shots. This is also why the node exists at all rather than being `Create Video` + `Save
Video`: those encode one frame per batch item, so the same song at 30 fps means a 5 400-frame
batch — about 17 GB of tensor for a 1024×1024 picture, before the encoder has seen anything. Here
at most a couple of prepared slides are alive at once and the encoder is fed frame by frame, so the
memory needed is set by the frame *size* and never by the song's length.

**Timing comes from the audio; the plan only supplies proportions.** A Siren plan is written in
*bars*, and bars are only seconds once you know the tempo — which lives on other nodes, and the
moment the two disagree the pictures slide off the music. The true length of the track, meanwhile, is
right there in the `audio` input. So the plan is read for its **ratios** and scaled so the last slide
ends on the last sample: no bpm to wire, no drift, and the stretch factor is printed in the `report`
where a plan that came out 5% short is visible instead of silent. Two shapes are read — **Siren
Score**'s / **Siren Cast**'s `plan` (`label | voice | 16 bars` rows, which is also where the section
labels come from) and **Orpheus**'s `durations` (a plain comma list of seconds, already cut to the
music). With nothing wired the song is split evenly.

**`layout`** is how the slides are handed to the sections:

* **one slide per section** — the picture changes on the section boundary. More slides than sections
  and the extra ones subdivide the *longest* sections; fewer and they cycle.
* **by section label** — every row called `Chorus` gets the **same** slide, so the chorus shot comes
  back the way it does in a cut music video. It costs nothing, because the labels are already in the
  plan.
* **even** — the plan is ignored and the song is split equally.

Neighbouring segments that end up on the same picture are **fused**, so a dissolve is never asked to
blend a picture into a copy of itself.

**`crossfade`** is seconds of dissolve at each change, centred **on** the cut — half before, half
after, so the moment the two pictures are equal is the moment the music turns. It is clamped to 40%
of the shorter neighbour, because a 1-second fade across a 1.5-second section is not a transition,
it is the whole shot. **`ken_burns`** (off by default) is a slow zoom and drift over each slide, in
and out alternately: a still held for three minutes reads as a broken video, and this is what makes
it read as a shot. It is not free — every frame becomes unique, so file size and encode time grow
several times over, and it wants 24 fps or more.

**`fps`** defaults to **12** on purpose. Nothing moves in a slideshow, so this is almost entirely
file size and encode time; raise it only when `crossfade` or `ken_burns` is on, since at 12 fps a
dissolve steps rather than flows. **`frame_size`** takes the picture's own size or one of the
platform shapes (16:9, 9:16, 1:1), always with even sides — `yuv420p` requires them, and an odd one
fails inside libx264 rather than in the node. **`fit`** decides what happens when the picture is a
different shape: padded onto a blurred enlargement of itself, padded onto black, or cropped to fill.

Wire the **`lyrics`** (the same text that went to Siren) and each section's own lines are written to
a **`.srt`** beside the video, timed to **the plan's section**, not to whatever picture happens to be
on screen — one still over a whole song is one segment, and the song still has six sections with
words in them. So subtitles need a `plan` wired, and they work under every `layout`, including
`even`. Sections are matched to plan rows by **label**, not by position, because a plan carries rows
the lyrics never had — the instrumental blocks Siren Score adds to reach the target length — and a
section with no sung lines simply gets no cue rather than an empty one.

---

[← back to the node index](../README.md#-node-index)
