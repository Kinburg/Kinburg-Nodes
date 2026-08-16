# 🎵 Audio & Music Suite

<!-- index-order: 3 -->

[← back to the node index](../README.md#-node-index)

---

## 🧜 `siren/` — Siren Suite 🧜

> **System Purpose & Overview**  
> Comprehensive music generation suite covering voice allocation (Siren Cast), lyrics structuring (Siren Score), music sampling & section windowing, audio spectrum analysis (Siren Scope), and AB comparison (Siren Compare).

### `Siren Cast (Voice Plan) 🧜` — who sings where

Replaces `TextEncodeAceStepAudio1.5`. It exists because of two things that node cannot express.

**Tags have no time axis; the plan does.** With `generate_audio_codes` on, the text encoder runs a
Qwen LM that emits `ceil(duration) * 5` tokens — one every **200 ms**, a 5 Hz plan of the whole song.
Each is a single FSQ index (`levels [8,8,8,5,5,5]` = 64000, one quantizer), so the tokens are
independent; the detokenizer expands each into 5 latent frames with attention running **only inside
its own 5-frame window**; and the result is concatenated onto the latent **channel-wise, frame for
frame**. That makes the plan the strongest — and the only *time-aligned* — conditioning in the model.
`tags` and `lyrics` reach the DiT globally through cross-attention, which is exactly why *"male
vocal, female vocal"* in the tags reads as a wish about the average track rather than an instruction
about the second verse.

So Cast builds the plan **section by section**, each with its own voice — not by generating clips and
gluing them: before decoding section *k*, the codes already written are put back into the prompt as
the assistant's output so far and decoding **continues**, with that section's caption in front of it
and the whole song's metas in the `<think>` block. The sampling loop is comfy's own
(`ace15.sample_manual_loop_no_classes`, handed ready-made `ids`), so nothing about how a token is
drawn is reimplemented. Total decode cost is one full-length pass; the extra is one prefill per
section. **One sampling run, no seams, no audio editor** — the in-graph version of the "generate each
part with a fixed seed and splice it" advice, done in the plan instead of in the waveform.

**`cfg_scale` on the core node barely touches the caption.** Its negative prompt repeats the *same*
caption and the *same* lyrics — only the `<think>` metas block is emptied. So `cfg_scale 2.0`
amplifies "with bpm/duration/key" against "without them", and the caption gets no guidance at all.
The tokenizer already accepts `caption_negative` / `lyrics_negative` / `*_negative` metas; the core
node just never passes them. Cast does, and **mirrors the metas into the negative** so the two
prompts differ in the caption alone. `guidance`:
- **`voice delta`** (default) — the negative is that section's caption **minus its voice line**, so
  `cfg_scale` amplifies precisely the difference between *"someone sings this"* and *"**she** sings
  this"*. A section with no voice has no delta and falls back to core behaviour.
- **`negative tags`** — the negative is the `negative_tags` text. General prompt adherence.
- **`metas only (core behaviour)`** — what the core node does, for honest A/B.

The `plan` is one section per line, paste-able from whatever wrote it (blank lines, `#` comments, a
markdown rule and a pasted header are ignored):

```
Intro    | -           | 8
Verse 1  | Alex        | 24
Chorus   | Nina        | 8 bars
Bridge   | Nina + Alex | 0:12 | drums drop out
Outro    | -           | 4
```

Column 2 is a wired **`voice`** name (several joined by `+`), or free text used verbatim, or `-` for
no vocal; column 3 takes seconds / `24s` / `m:ss` / `N bars`; an optional column 4 is appended to that
section's caption. Voices come from **Character Card**'s `voice` output (or **Card Presets**), so a
band member is described once for the lyrics LLM, the cover art and the song. Lengths are rounded to
whole codes (0.2 s) and add up to the **`seconds`** output — wire that into `Empty Ace Step 1.5 Latent
Audio` and the plan and the latent can never disagree, which is the classic AceStep mistake. An
**empty plan** is one caption for the whole song, i.e. the core node plus the guidance fix — start
there, it is one variable. Also out: `timeline` (m:ss per section, so a section can be typed straight
into Siren Section for a retake), `report`, and `gen_extra_info` recording who sang what. Each section
draws from `seed + its index`, so editing one section's voice leaves every earlier one bit-identical.
`cast_in_caption` appends the distinct voices to the **global** caption, telling the DiT which
timbres exist at all — the first thing to A/B if sections start bleeding into each other. The LM
sampling defaults are left exactly at the core node's on purpose (`cfg_scale 2.0 · temperature 0.85 ·
top_p 0.9 · top_k 0 · min_p 0.0`); the guidance is the one change to judge first. `temperature`
0.6–0.7 or `min_p` 0.02–0.05 tighten the plan after that — but **never `temperature 0`**: an
autoregressive audio LM decoded greedily falls into repeating the same bar.

### `Siren Score (Lyrics → Plan) 🧜`

Writes the plan from the lyrics, with no LLM pass in the way. Everything the plan needs is already in
the text: the section list and its order are the `[Verse 1 - …]` markers, who sings each one is in the
marker, and how long a section should be is a function of how many lines it has.

- **Sections** — a bracketed line whose text *starts with* a section word opens one (a prefix test, so
  `[Chorus - wall of guitars]` is a section and `[wall of guitars, no chorus pad]` is not). Synonyms
  map onto the canonical labels, including `Bridge/Chaos` → Bridge and `Hook`/`Refrain` → Chorus.
  Other bracketed lines are annotations of the current section — which is where the voice often hides,
  under a header that describes only the drums.
- **Voice**, most certain first: a member's name in the marker (`Keen Burg`, or a duet with `+`); a
  name buried in its prose; `MALE`/`FEMALE` matched against the wired cards' `voice_tags` and
  `gender`; failing all that, the marker's own vocal wording used verbatim, which is why this works on
  lyrics written before any of it existed. An ambiguous gender resolves *and* is flagged.
- **Two singers at once is the one thing the model cannot do**, structurally: the plan is one audio
  code per 200 ms and the caption is one description, so two timbres over the same frames come back as
  their average (measured once as "two female vocals" where a man and a woman were asked for). A
  bracketed line that **names** a member therefore *splits* the section — one sub-section per voice,
  so they **alternate**, which the model does well. Sub-sections are floored at 2 bars rather than
  `min_bars`, because an exchange of single shouted lines is meant to be short. A header duet with
  nothing to split on becomes the first-named singer plus a short `with male backing harmonies` note,
  and the report says so and shows what to write instead.
- **Lengths run backwards from the target, not forwards from a rate.** `pad_to_seconds` says how long
  the song is; `tail_bars` takes its slice off the end; a section with no sung lines takes
  `instrumental_bars`; everything left is shared among the sung sections **in proportion to their
  syllables** (Hamilton's method on 2-bar units, floors applied afterwards so a longer line can never
  come out shorter than a shorter one). The **singing rate is an output**, printed in the report.
  That way round is the point: with a rate as the input, the slack between the words and the wanted
  length hid *inside* the vocal sections — and a section with more room than its words need does not
  get sung slower, the model FILLS it. That is what turned a 6-second intro into a 40-second one that
  ate the first verse. With the length as the input, slack can only land where it was asked for.
  When the resulting rate falls outside 2.5–8 syllables a second the node says so, names the roomiest
  (or tightest) section, and gives the length this lyric would actually suit. Syllables are vowel
  groups with English's silent final `e` dropped; a line wholly in round brackets is an annotation, not
  a lyric — production notes don't get sung, and a backing echo like `(Живий!)` is sung *over* the line
  above rather than after it, so neither adds duration.
- **`tail_bars`** is a small explicit choice (0–32) rather than a residual. Measured with
  `lyrics_in_negative` off: a **short** tail (2–8 bars) buys a last chorus, a proper outro and a clean
  ending instead of the track stopping dead on the final word, while a **long** one is actively bad —
  the model repeats the last phrase over and over to fill it and starts eating the ends of held notes.
  (An earlier measurement had total length as the strongest quality driver; that was taken with
  `lyrics_in_negative` **on**, before the words were guided. Both are real; this is the one that holds.)
- **`pad_placement`** decides where those blocks go, and the trade-off is not taste — it is where a
  lyric can be interrupted. AceStep gets the words with **no timing in them**, matched against the
  plan, so a gap in the middle asks the model to hold the line until the singing resumes; if it
  doesn't, everything after shifts. `after the vocals` (default) and `intro + outro` sit entirely
  outside the lyric and carry none of that risk; the default is the blunter of the two only because it
  is the one with a take behind it. `between sections` is the most song-shaped — one block opens, the
  rest go after choruses and bridges, never inside a verse running into its own pre-chorus, and never
  inside a Bridge exchange — and the only one that can make the lyric drift.
- **A roughly even split of sung and instrumental bars is a working shape, not a smell.** The take that
  worked was 44 sung against 42 padded and came back harmonious, with enough instrumental breaks and
  no dragged words — the model fills that space musically, given a caption that says what the record
  is. The node only mentions the ratio when padding runs past twice the sung length, and then as an
  alternative (ask the lyrics pass for more sections) rather than a correction.
- **The 4th column** carries the marker's arrangement and delivery notes onto that section's caption,
  trimmed fragment-by-fragment against the card's own tags — so `[Intro - deep hypnotic spoken word -
  MALE vocal]` keeps "deep hypnotic spoken word" and drops "MALE vocal", which the card already said.
  Band members' **names never reach a caption** (AceStep's caption is a music description; "Keen Burg"
  is not one), and the column is capped at 4 clauses with the rest reported: everything there sits
  inside Siren Cast's cfg delta and is guided as hard as the voice, so a Bridge that accumulated 8
  contradictory clauses came back sung by one indistinct voice. If a take is muddy, `arrangement_notes`
  off is the cheapest thing to try.

Both Siren nodes are kept to the inputs you actually touch, with everything settled folded behind
*Show advanced inputs* — Score shows `lyrics` (input only; nobody types a lyric sheet), `bpm`,
`timesignature`, `pad_to_seconds`, `tail_bars`, and Cast shows `clip`, `tags`, `lyrics`,
`plan` (all three wired), `seed`, `bpm`, `timesignature`, `language`, `keyscale`.

`plan` is **optional**: unwired, the node encodes one caption for the whole song — the core node's
behaviour plus the guidance fix, which is where to start and how to A/B the guidance on its own. And
the plan's decode reports **one** progress bar across all its sections rather than one per section:
comfy's sampling loop builds its own bar on every call, so a twelve-section plan used to show twelve
bars each restarting at zero, which reads as a stuck node.

**The values that come from the song-config pass are plain fields, not dropdowns**, because a combo
input cannot accept the STRING a text parser hands it — which is what stopped the config being wired
straight in. They are parsed leniently and every correction is reported: `timesignature` takes
anything with a digit (`4`, `"4/4"`); `language` fixes the slips a model asked for a two-letter code
actually makes (Ukrainian is `uk` not `ua`, Chinese `zh` not `cn`, Japanese `ja` not `jp`);
`keyscale` reads `C major`, `c# minor`, `C sharp minor`, `d flat major` and `Am`, keeping whichever
spelling of a black key was written since AceStep's list carries both. Anything the list cannot
express falls back with a line in the report rather than quietly poisoning the metas.

### The sampler and the section window

An AceStep audio latent is a **one-dimensional strip of time**: `[B, 64, T]`, one frame per **40 ms**
(the 1.5 VAE turns a frame into 1920 samples at 48 kHz — **25 frames per second**). That single fact is
what a generic latent sampler can't exploit and what these two nodes exist for: name a stretch of the
strip **in seconds** and you can regenerate only that stretch, or append to it, and leave the rest of
the take alone. Chimera is still the node for splitting a schedule; Siren is the node for splitting
the **track**.

**`Siren Section (Audio Window) 🧜`** turns *"from 0:47.5 to 1:02, snapped to bars, with a 0.35 s
crossfade"* into a denoise mask on the latent:
- **`retake`** marks `start_sec`→`end_sec` as free to regenerate and freezes everything else; the
  latent's length doesn't change. `end_sec = 0` means "to the end of the track".
- **`extend`** **grows** the latent by `extend_sec` (at the `end` or the `start`) and marks only the new
  part as free. The existing take is frozen but still **visible to the model** — attention runs over
  the whole strip — so the new part is written to follow on from it. Extending at the `start` shifts
  every section marked earlier in the chain later by the same amount, so their timings stay on the
  music.
- **`snap`** (`bar` / `beat` / `off`) quantizes the edges to a grid built from `bpm` /
  `beats_per_bar` — use the values you gave `TextEncodeAceStepAudio1.5`. One bar at 120 bpm in 4/4 is
  2 s = exactly 50 latent frames. Replacing a section that starts mid-bar is the usual reason a retake
  refuses to sit in the groove. `grid_origin_sec` moves bar 1 for a track that opens with a pickup.
- **`fade_sec`** ramps the mask **outside** the window, so the range you named is rewritten in full and
  the join is spread into the neighbouring audio. In an image a hard mask edge is a visible line; in
  audio it is an audible **click**, so this is not cosmetic. 0.2–0.5 s is a good range.
- Wire the **`vae`** input and the frame rate is read from the model itself (sample rate ÷ samples per
  frame) instead of trusting the `latent_fps` widget — recommended, and it covers AceStep 1.0, which
  runs at a different rate.

The mask goes into the latent's own standard **`noise_mask`**, so the `latent` output also works with
the **stock** samplers — Siren is not required to use it. Chain several Section nodes (`section` input)
to mark several windows at once; overlapping windows merge, and the mask is rebuilt over all of them.

**`Siren (Music Sampler) 🧜`** samples it, with the dials **on the node** — defaults already set to
what ships for `acestep_v1.5_xl_base`: `steps 50 · cfg 6.0 · euler · simple`. The dials that do nothing
here are simply absent (`seed_mode` / `seed_step` are Ouroboros loop controls; a later stage's
`denoise` and `scheduler` can't matter when one schedule is shared), and the rarely-touched ones
(`eta`, `s_noise`, `s_churn`, `solver_type`, `stage_b_sampler`, `verbose`) are flagged **advanced** so
they collapse out of the way.

**`steps` is the whole schedule and `stage_b_steps` carves the tail off it** — 50 with `stage_b_steps`
10 means 40 + 10, so you never add the stages up yourself. Two stages exist for one reason: **high cfg
locks the lyrics and the structure but squeezes the sound, low cfg lets the timbre breathe but slurs
the words** — so `cfg 6.0` early and `stage_b_cfg 4.5` late buys both. AceStep's schedule is heavily
top-loaded (at the default shift of 3.0 the halfway step is still at **sigma 0.750**), so the boundary
has to be **late** to land in polishing territory: out of 50 steps, 10 puts it at sigma 0.429 and 15 at
0.562. Much earlier and the second stage starts rewriting the arrangement instead.

Under **advanced**, the second stage can also take its own **sampler**, **scheduler** and **seed**.
The scheduler can't simply replace the shared curve — the latent is sitting at a particular noise
level — so the alternate is rebuilt over the same length, its tail sliced out and its first sigma
**pinned to the level actually carried**, then forced monotonic: the same splice Chimera does, and the
report says when it happened. The seed only bites when stage B's sampler is ancestral or SDE, because
a continuing stage adds no fresh noise and the seed reaches nothing but the stochastic sampler's own
generator; `-1` means "use stage A's".

**`resume_from_sigma` is the retake dial**, and it replaces reasoning about `denoise`. Above 0 it
resumes an existing take from that noise level — the whole track, or just the marked stretch when a
Section is wired — and **derives the step count itself**, so the run walks exactly the tail of the
native schedule and costs proportionally less time. Measured on a 50-step shift-3 curve:

| `resume_from_sigma` | what you get | steps actually run |
|---|---|---|
| 0.43 | tidy up the performance, groove intact | 10 |
| 0.51 | same musical idea, different performance | 12 |
| 0.56 | noticeably different take | 14 |
| 0.71 | almost a new section | 22 |
| 1.00 (or 0 = off) | completely new | 50 |

It needs something to resume *from*: on an empty latent it only lowers the starting noise level and
weakens the result, which the node warns about rather than letting you wonder.

A **`Sampler Settings`** bundle can still be wired into the optional `sampler` input, and while it is
there it **replaces** the widgets outright (reported) — that is how a stored per-model recipe from the
model library's **Settings Select** / **Model Select** drives this node. A half-and-half rule would be
unreadable off the node face, so it's all or nothing.

**One rule follows from the masked math**, and the node enforces it: *with a section, exactly one stage
can run.* A masked run has to finish at sigma 0, because only there is the frozen region's reference
the **clean** latent; handing an in-flight latent to a second stage would re-pin the frozen audio to a
partially-denoised reference with no noise term — right at the handoff step, drifting after it. So a
retake gives the whole remaining tail to the first live stage and reports that the others were skipped.
Set `stage_b_steps` to 0 for retakes. Without a section, stages run continuous exactly like Chimera.

Why the rest of the take survives at all: ComfyUI's masked path re-pins the frozen frames every step
with `sigma·noise + (1−sigma)·original`, which for a flow model like AceStep is the exact forward
interpolation — so the untouched audio stays consistent with the noise level the sampler is working at,
and `reshape_mask` already handles the 1-D case. No custom sampling code is involved.

The report prints the curve's **quarter, half and three-quarter sigmas**, which is the only place
`shift` is visible: a flow schedule's endpoints are always 1→0 whatever shift is set to, so `50%=0.750`
vs `50%=0.500` is how you tell at a glance whether a `ModelSamplingAuraFlow` node is actually in the
model path. (Worth knowing: **bypassing that node is not "no shift"** — `ACEStep15.sampling_settings`
already carries `shift: 3.0`, so bypass and shift 3.0 are the same run.)

Outputs mirror Chimera's so the node drops into the same pipeline: **`latent`**, **`report`** (schedule,
per-stage times, section coverage as a percentage of the track, every warning), **`gen_extra_info`**
(`GEN_INFO`, for Generation Info's `extra` input) and **`time`** / **`seconds`**, measured around the
sampling calls inside the node. Category `Kinburg-Nodes/sampling`.

### `Siren Scope (Audio → Image) 🧜`

Audio can't go into **Image Compare** — a picture of it can. This node renders `AUDIO` to `IMAGE`, and
every decision in it exists to make two takes **comparable** rather than to look nice:

- **The dB scale is absolute, never auto-fitted.** `db_floor` / `db_ceiling` are dBFS, so a quiet take
  renders *dark* instead of being silently boosted to fill the frame. Auto-normalising each image is
  exactly what makes two spectrograms meaningless side by side.
- **The pixel grid is fixed.** The clip always spans `width_px`, so column *x* is the same moment in
  every render of a same-length track and an A/B flip doesn't jitter. There are no margins — every
  pixel is signal, plus the optional time ruler.
- **The bar grid lands where `Siren Section` would cut.** Give it the same `bpm` / `beats_per_bar` /
  `grid_origin_sec` and the bright lines are bar boundaries, the dim ones beats — so you pick a retake
  window off the picture and type those seconds straight into the Section node.

Modes: `mel spectrogram` (structure, drop-outs, a band-limited top end), `linear spectrogram` (hard
low-pass and resampling artifacts), `waveform` (level, silence, clipping) and `mel + waveform` stacked
on one time axis. `channels` can draw a mono mix, one side, or both stacked to catch a stereo collapse.

**Wire a second clip into `audio_b` and it becomes a difference view** — **black where the two are
identical**, warm where A is louder and cool where B is. Two things make that view mean something,
and without them it is worse than useless:

- **It compares energy over musical tiles, not raw bins.** Two independent takes never agree
  bin-for-bin — their fine detail and noise floor are uncorrelated — so a raw difference is a field of
  speckle that says nothing about whether they *sound* different. `diff_detail` averages energy over a
  tile first (`musical` ≈ ⅛ of the mel bands × 120 ms). Measured on synthetic takes: inaudible noise
  50 dB down goes from painting the frame to **0% of tiles**, while a real 10 dB shift in the top end
  still comes through at **9.4 dB**. Both takes are also floored at the same level *before* pooling —
  gate afterwards instead and the near-empty bins, where two takes disagree by tens of dB about
  essentially nothing, take over the picture.
- **The bottom panel compares short-term RMS, not samples.** A sample-by-sample subtraction of two
  takes that merely differ in phase comes out nearly as loud as the music itself — shift a track by
  3 ms and `A − B` peaks at 0.41 while nothing about the sound changed at all. The loudness panel
  answers the question a listener actually has: *where is one of these louder than the other.*

`diff_span_db` (default 6) sets how many dB reach full colour — the dial that decides whether nuance is
visible at all. `fine (raw bins)` turns the averaging off; it is the right choice only when the two
clips share actual samples, i.e. checking a **Siren retake**, where the frozen part should come out
pure black and only the marked section should light up.

**`gen_extra_info` is usually the clearer answer.** It carries duration, peak, RMS, crest, brightness
(spectral centroid), the low/mid/high energy split, near-silence, clipped samples and — in stereo —
correlation and side energy, in the `GEN_INFO` shape **Generation Info** merges. Feed it in alongside
the sampler settings and **Image Compare**'s `differences` mode tables exactly which of those numbers
moved between two runs. If you don't read spectrograms, that table answers *"how do these two differ"*
in a way no picture will.

`time_labels` off removes the ruler **and** its lines, so the panel is pure signal for a pixel-exact
overlay; the `bpm` grid is musical structure and stays. Rendering is plain torch (a colour ramp and a
5×7 bitmap font), so there is no plotting library in the graph and no font on disk to go missing.
Category `Kinburg-Nodes/audio`.

### `Siren Compare (Audio) 🧜`

**Image Compare** for music. It is a separate node rather than a mode of that one because almost
nothing carries over — there is no SSIM for a song, and the whole interaction is a *transport* — but
the delivery is identical: a portable folder (audio + scopes + `index.html` with relative links)
registered under a token and served by the existing `/image_compare_dir/` route, which already
streams with Range support. Same "🔗 Open comparison" link widget on the node, same offline bundle.

Collect the takes the way Image Compare does: a **`Set Accumulator (audio)`** on each branch and one
**`Get Accumulator (audio)`** into this node's single `audios` input. It carries the same
**`auto_collect`** toggle and **🔌 Collect All** button, so a Set you just added or bypassed is
re-wired right before the workflow is queued. `labels` takes `Get Accumulator (captions)` (one line
per take) and `notes` takes either that or `Get Accumulator (prompts)` (`---`-separated blocks);
`times` takes one line per take the same way Image Compare does — wire **Siren (Music Sampler)**'s
`time` output through a Set/Get Accumulator (texts) — and shows up next to each take's name plus as
two rows in Measurements: the raw time and **`time vs fastest`**, which turns it into the question
you actually have when comparing sampler settings (*what did the extra stage cost?*).
`settings_data` is Generation Info Filter's, the same output Image Compare uses. Lyrics go to a
resizable side panel — one block is shared by every take, `---` splits them per take — with
AceStep's `[section markers]` and `(asides)` coloured apart from the sung words.

**The spectrograms ship as data, so the view is live.** Rather than pre-rendering a picture per
combination of mode × colour map × dB floor × channel (measured at 288 renders, ~1 minute and 119 MB
*per track*, and still no continuous sliders or arbitrary A/B pairs), the node writes each take's mel
spectrogram as a 16-bit dB matrix packed into a PNG's R/G channels — about 580 kB, less than two
finished colour renders. The page decodes it once and everything after that is array arithmetic:
colour map, dB floor/ceiling, mode, channel, **zoom**, **diff against any take you pick**, and the
diff span, all instant and none of them needing a re-run. Only `scope_columns`, `n_fft` and `n_mels`
stay on the node, because those change the matrix itself — and `scope_columns` defaults to **auto**,
which ships 25 columns per second (one per AceStep latent frame, 40 ms), finer than any screen so
there is real detail under the magnifier rather than interpolation. The FFT therefore lives in
exactly one place — Python — with no JavaScript reimplementation free to drift away from it.

**Zoom** is a window into that matrix: the wheel zooms about the cursor, dragging pans (a click
without movement still seeks), `to loop` frames the marked bar, and while playing zoomed in the view
follows the playhead. The scrub bar keeps showing the whole track with the visible window bracketed.

Picking a **reference** puts every *other* take into the diverging diff ramp while the reference keeps
its own picture, so you have something to read the deltas against.

The page builds it into one canvas per take, so:

- **Every take plays off one Web Audio clock.** Each is decoded into an `AudioBuffer` and all sources
  are started against the same `AudioContext` time, so they stay sample-accurate for the whole run.
  Soloing is a **gain change**, not a stop-and-restart, ramped over 12 ms — so switching between takes
  mid-phrase is instant and silent. That matters more than it sounds: a 30 ms hiccup at the switch is
  exactly the artifact you'd mistake for a difference between the takes.
- **Scrubbing moves everything at once**, because there is only ever one position. Click any scope to
  seek there; the playhead is exact because the scope's pixel grid spans the whole clip by
  construction.
- **Shift-drag a scope to mark a loop** and hear one bar over and over — set on the buffer sources
  themselves, so the wrap is sample-accurate too.
- **`match level`** trims every take down to the quietest one's RMS. Without it the louder take simply
  wins, every time. (Down, never up: several unmuted at once would otherwise clip.)
- **`blind`** hides the names **and shuffles the takes**, calling them *Take A*, *Take B*… by
  position — hiding names alone was not a blind test, because with no `labels` wired every take is
  already called "Take 1", "Take 2", and even with labels the takes stayed in the order you wired
  them, which is the order you remember. The shuffle is a plain uniform one, so it sometimes leaves
  everything where it was: "never the arrangement you just saw" would be information, and with two
  takes it would be the whole answer. Switch blind **off** and the shuffled order *stays* while the
  names appear in place — the take you soloed is the one that lights up, which is the question you
  actually had. **`↺ original order`** puts them back; it shows up whenever blind has shuffled,
  never depending on how the draw came out. The **Measurements / Settings / Notes** tabs follow the
  same order and share a `differences only` switch that collapses each table to the rows that
  actually differ between takes.
- Keyboard: <kbd>space</kbd>, <kbd>1</kbd>…<kbd>9</kbd> to solo, <kbd>0</kbd> for all,
  <kbd>←</kbd>/<kbd>→</kbd> to seek, <kbd>L</kbd> to loop.

**Sending it to someone: turn `self_contained` on.** The default folder bundle opens fine *in-app*
and over HTTP, but a browser given a `file://` page treats it as having no origin and refuses to load
its own siblings — the audio is blocked by CORS and the spectrogram data can't be read back out of a
canvas ("tainted by cross-origin data"). So a zipped folder, double-clicked, plays nothing and draws
nothing. `self_contained` writes ONE .html with every take and matrix inlined as a `data:` URI, which
is same-origin and dodges both rules. Pick MP3 or Opus first unless lossless is the point: base64
adds a third, and a 3-minute FLAC take inlines to roughly 40 MB. A page opened off disk that *can't*
load says so on the page rather than sitting there empty.

Every take is measured on **one shared time base** (the longest one), so column *x* is the same
instant on all of them — which is what lets the diff line up and the playhead be right; a shorter take
simply stops early instead of being stretched. `audio_format` defaults to **FLAC and should stay
there** — you are comparing fine detail, and a lossy codec would add differences of its own on top of
the ones you're listening for.

**The one thing to remember about `extend`:** the `duration` you gave `TextEncodeAceStepAudio1.5`
describes the **whole** track and is baked into the tokens, so after extending you must raise it to the
new total and re-encode — the model is otherwise being told the song is shorter than the strip it is
writing on. It can't be read back out of the conditioning, so both nodes print the new length in
seconds and leave the check to you.

---

## 🔊 `audio_sr/` — Audio SR (48 kHz Upscale) 🔊

> **System Purpose & Overview**  
> Audio super-resolution and upscaling to pristine 48 kHz output.

**`Audio SR`** is bandwidth extension for a finished mix: AudioSR is a latent-diffusion model that
*invents* the top end rather than filtering it, so a track that dies at 11 kHz comes back with
plausible 11-24 kHz content. Mono, 48 kHz out. The model is vendored under `audio_sr/vendor/audiosr`
(MIT; see `vendor/NOTICE.md` for attribution and the three edits made to it) so the node does not depend
on another pack being installed.

**A stereo mix keeps its image.** AudioSR is a mono model, so the wrapper this replaces summed L and
R — and measured on a real 3-minute take that took an L/R correlation of **+0.45** and a side/mid RMS
of **0.61** down to **1.00** and **0.00**. The whole image, for good. `stereo = mid/side` sends only
the mid channel through the model and carries side through untouched, so only what the model invents
above the roll-off is centred. (`sum to mono` is still there to A/B against. Never run L and R
separately: two independent diffusion passes decorrelate and the invented top comes out phasey.)

**`match_level`** puts the output's energy *below 10 kHz* back where the input's was — not an overall
match, since the model genuinely adds energy up top and matching totals would turn the track down to
pay for it. Below the roll-off the model measured transparent (-0.4 dB at 8-12 kHz), so drift there is
drift: on the same take it was -1.2 dB at 0-4 kHz and -1.7 dB at 4-8 kHz, which reads as the mix
losing body.

What the model actually does, measured on that take (Raw → SR, energy per band):
`8-12 kHz -0.4 dB` · `12-16 kHz +3.3 dB` · `16-20 kHz **+30.6 dB**` · `20-24 kHz **+55.9 dB**`. So it
is transparent below about 12 kHz and writes the octave above from nothing — which is the job, since
AceStep's own output rolls off around 12 kHz however full-band its 48 kHz container is.

Three more things this fixes over the wrapper it grew out of:

- **The progress bar works.** The old one called `model_management.get_progress_state()` and
  `comfy.model_management.update_progress()` — *neither exists in ComfyUI* — inside a bare
  `except Exception: pass`, so it silently did nothing and the node looked hung for minutes. The time
  goes in the DDIM loop, so that is where it is driven from: the vendored `ddim.py` carries one
  `STEP_HOOK` (`None` by default, i.e. upstream behaviour) and the node fills it in. The bar counts
  **chunks × steps**, the real unit of work.
- **Cancel lands inside a chunk**, for the same reason — interruption used to be checked only between
  chunks, so a stop could sit unhonoured for fifteen seconds of audio.
- **Chunk geometry.** The plan is computed up front, so every window is a full chunk and the tail one
  is pulled *back* to end at the last sample instead of being padded with silence the model would
  denoise at full price. The crossfade uses a **periodic** Hann pair, whose halves sum to exactly 1;
  upstream's symmetric one dips about 1.2% at each join. `chunk_seconds` defaults to 15.36 = 3 × 5.12
  because the batch builder pads every chunk up to a multiple of 5.12 s.

**`audiosr/clap/` was cut** — 56 files, 3.25 MB, three quarters of the vendored source. Its only
construction site was a `self.clap = …` in `ddpm.py` that nothing in the package ever read, a leftover
of the AudioLDM lineage: super-resolution conditions on `VAEFeatureExtract`, not on CLAP. It was
costing 0.80 GB of the checkpoint's 6.18 GB and a HuggingFace round-trip *at import time*
(`BertModel.from_pretrained("bert-base-uncased")` fired while the module was merely being imported).
Checked rather than assumed: the trimmed `LatentDiffusion` has **0 missing** parameters against both
real checkpoints and 507 unexpected ones, all `clap.*` — so the model is fully satisfied by the
weights it gets, and only never-used tensors are now ignored. 1085.8 M params, 4.34 GB fp32, down
from 5.14.

`checkpoint` lists `ComfyUI/models/AudioSR` and the variant (`basic` for music, `speech` for voice) is
read from the file name, since the two need different configs. `keep_loaded` holds ~6 GB in VRAM
between runs. Category `Kinburg-Nodes/audio`.

**On the speechbrain landmine.** `util/imports.py` exists because of a bug that has nothing to do with
this node but killed it: speechbrain 1.1 puts `LazyModule` objects in `sys.modules`, `inspect.getmodule`
walks every entry doing `hasattr(m, "__file__")`, and for the ones whose optional dependency is absent
that raises `ImportError` — so once *any* pack has imported speechbrain, any node calling into
`inspect` dies with `Lazy import of LazyModule(target=speechbrain.integrations.k2_fsa) failed`.
speechbrain guards against exactly this, but the guard tests `filename.endswith("/inspect.py")` and on
Windows the frame reads `…\Lib\inspect.py`, so it never fires. `defuse_lazy_modules()` replaces the
unimportable entries with stubs — nothing is lost, they could not be imported anyway — and is called
at the top of the node's `run()`, not at import, because the mine is armed whenever the *other* pack
loads.

---

## 💾 `save_song/` — Save Song

> **System Purpose & Overview**  
> Save generated audio tracks with metadata and artwork integration.

**`Save Song`** saves an **`audio`** clip (required) as a song, with an optional **`image`**
cover and optional **`lyrics`** text (an input socket — wire a STRING in). The **`quality`** dropdown picks the audio format and
bitrate — **FLAC** (lossless) or **MP3 / Opus** at a chosen bitrate — encoded with PyAV exactly
like ComfyUI's own Save Audio (Opus is resampled to a supported rate automatically). The cover is
written as a **JPEG** (its `image_quality` is adjustable), and the lyrics as a **`.txt`** — all
three share one counter-based base name under `ComfyUI/output` (e.g. `songs/song_00001.flac`,
`…_00001.jpg`, `…_00001.txt`). It returns the standard `audio` / `images` UI results, so ComfyUI
shows a **native `<audio>` player and the cover preview** on the node **and** lists the saved
files in **Media Assets** (just like the core Save Audio node). Outputs the `audio` passthrough
plus the saved `path`. Category `Kinburg-Nodes/audio`.

---

[← back to the node index](../README.md#-node-index)
