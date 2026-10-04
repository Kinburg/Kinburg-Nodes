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
- **`voice delta`** (default) — the negative is the **shared** caption, so `cfg_scale` amplifies
  whatever that section *adds* to it: the difference between *"someone sings this"* and *"**she**
  sings this"*. Note that a section's column 4 rides the same delta — right for a duet's backing
  note, worth knowing for an arrangement note. A section that adds nothing has no delta and falls
  back to core behaviour.
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
section's caption.

**A row naming two singers cannot be sung by two singers**, for the reason in Siren Score below: one
timbre per 200 ms code. `duet_mode` decides which single thing gets said instead, and it applies
however the plan was written — Score protects a table it wrote, but a `Nina + Alex` typed straight
into the widget used to go to the failing case with no warning at all.
- **`lead + backing note`** (default) — the first name leads, the rest become a short phrase carrying
  their own timbre. The same collapse Score performs, now done here too.
- **`one unison duet`** — names the pair as a **single sound** ("two voices singing together in close
  unison harmony, airy alto female and raspy baritone male") rather than as a lead with backing.
  Worth an A/B on a fixed seed: the "two timbres average out" measurement was taken on two full
  descriptions pasted side by side, which is a *contradictory caption* rather than a description of a
  duet, and AceStep has certainly heard duets. Untested — that is what the mode is for.
- **`both descriptions`** — the old behaviour, to A/B against.

Whichever is chosen, the singers who lose their place in the *section* caption are still listed in
the **global** one, so a member who appears only in duets is not invisible to the model — that list
is, per `cast_in_caption`, the only route by which a second timbre can colour a section's frames.
Per-line markers in the lyrics remain the only way to make voices genuinely **alternate**. Voices come from **Character Card**'s `voice` output (or **Card Presets**), so a
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
- **The section name must be the FIRST word in the marker.** The test is a prefix, not a search, and
  it has to be: only "starts with" tells `[Chorus - massive wall of guitars]` from
  `[wall of guitars, no chorus pad]`. The cost is that a qualifier in front of the name folds a whole
  section into the previous one's notes — and the natural-sounding forms are exactly the ones that do
  it. ACE-Step's own shipped templates are full of them: `[Final Chorus - …]`, `[Final Verse - …]` and
  `[Guitar Solo]` are none of them sections here. That loss used to be silent; now the report catches
  the near miss and hands back the corrected line, with the qualifier moved behind the name where it
  belongs (`[Chorus - final, Layered harmonies]`, `[Solo - guitar]`).
- **Voice**, most certain first: a member's name in the marker (`Keen Burg`, or a duet with `+`); a
  name buried in its prose; `MALE`/`FEMALE` matched against the wired cards' `voice_tags` and
  `gender`; failing all that, the marker's own vocal wording used verbatim, which is why this works on
  lyrics written before any of it existed. An ambiguous gender resolves *and* is flagged.
- **Two singers at once is the one thing the model cannot do**, structurally: the plan is one audio
  code per 200 ms and the caption is one description, so two timbres over the same frames come back as
  their average (measured once as "two female vocals" where a man and a woman were asked for). A
  bracketed line that **names** a member therefore *splits* the section — one sub-section per voice,
  so they **alternate**, which the model does well. It does so in **every** mode: a marker naming a
  member is an explicit instruction, and `duets` has authority only over a header naming several
  singers with nothing under it to split on. (The two used to be one switch, so reaching the unison
  path also turned alternation off for the whole song — an exchange in the bridge or a unison chorus,
  never both, and the loss was silent because an inner marker that stops splitting degrades into an
  ordinary annotation rather than disappearing.) Sub-sections are floored at 2 bars rather than
  `min_bars`, because an exchange of single shouted lines is meant to be short. A header duet with
  nothing to split on becomes the first-named singer plus a short backing note in words — and that
  note carries the other singer's **own timbre**, two words of it, not merely their gender
  ("with aggressive deep male backing harmonies"). Gender alone is the least identifying thing a card
  knows and picks out neither man in a band with two; the budget is the same either way, and short is
  what mattered. Two or more backing singers fall back to the gender, and a mixed group to nothing —
  past that the words are describing a crowd. The report says what happened and what to write instead.
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
sampling calls inside the node. Category `Kinburg-Nodes/Bestiary/Siren`.

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
Category `Kinburg-Nodes/Bestiary/Siren`.

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

## 🐐 `satyr/` — Satyr Suite 🐐

> **System Purpose & Overview**
> Reads a YuE2 plan against the lyrics and says what it will and will not sing, before a render is
> spent on it — and corrects the one thing an edit can safely correct. Reading a plan (Satyr Read),
> checking one against the words (Satyr Score), and changing one by hand in a piano roll (Satyr Edit).

YuE2 composes before it performs: it writes an ABC score — a vocal melody, an instrumental melody,
chord symbols, section labels — and then renders that score into audio. The score is an ordinary
string input on `YuE2 Generate Music`, tokenised verbatim and never validated, which makes it the one
piece of YuE2's conditioning that can be edited by hand. It is also the only conditioning it has with
a **time axis**: `M:` and `L:` and `Q:` make a bar's length arithmetic, so a plan states the song's
duration rather than wishing for it. `style` and `lyrics` are one flat blob each for the whole track.

**How many singers a song has is decided in `style`; where they sing, with a LoRA on the text
encoder, by a plan whose registers agree with the lyrics' markers.** It took a long and mostly
negative investigation to get there, and the base model still gives the plan little say.

There is no field for a voice — no reference singer, no per-section timbre — and the ABC has exactly
two hardcoded voices, `Vocal` and `Ins`. Asked in `style` for two singers, the model writes both into
the one monophonic `Vocal` line as two **pitch bands**, and which band a phrase sits in does track
which singer took it, in every song measured. Moving one chorus down an octave flipped it from the
woman to the man once, at a fixed seed, with the untouched chorus staying female — a clean result.

On the base model it never reproduced as a lever. A style naming no voices is sung by one singer from
beginning to end however the plan is written; a style naming two produces two, placed where the model
likes. Two plans of the same song measured five semitones apart: one came back with two voices, one
with a single voice, and what differed was the style string. An earlier version of this page claimed a
gap below about eight semitones meant one singer. A third measurement killed that, and it is retracted.

**With a LoRA on the text encoder it works** — a YuE2 genre LoRA at strength 2.0 on the CLIP, with
`cfg_scale` 2.0 — measured at one seed with nothing else changed. YuE2
had placed its registers by habit, verses low and choruses high, against the markers in five sections
of nine, and that plan came back in one voice. With those five moved an octave, the same song was
sung by exactly the singers marked, its melody, rhythm and arrangement unchanged. Without the LoRA,
of the four sections marked for the woman only the one `recast` had moved came back in her voice,
and it sat no higher than the three that did not — so on the base model the register is still a
correlate, not a control.

### `Satyr Read (Plan → Map) 🐐`

Reads a plan and changes nothing. Prints the meter, tempo, bar length and total duration; whether the
plan holds one voice or two and where the two registers sit; and then every phrase with its timing,
its bar count, its syllable budget and the voice that takes it. `seconds` is the plan's own duration,
which the model performs a few per cent faster than.

A syllable budget counts note onsets, so a held note is one syllable however it is tied — including a
note held from one group into the next, which the exporter writes as `d4-|` at the end of one line
and `d4` at the start of the next. That is the earlier phrase's syllable. Read line by line it was
counted again at the start of the later phrase, one too many at every such boundary, and `Satyr
Score` matched the lyrics against those inflated counts — with `refit` on, it took a note out of a
section that already held exactly its words.

Note the distinction it draws between a band's notes and its seats. The note ranges of the two bands
**overlap**: a phrase belonging to the man as a whole still reaches up into the woman's notes for a
syllable or two. What separates cleanly is the phrase *median*, because a phrase goes to one singer
whole — so that is what classifies a phrase and what a move aims at.

### `Satyr Score (Plan → Plan) 🐐`

Takes a plan and the marked-up lyrics, and reports what the plan will actually do with those words.
It reads much more than it writes, and that is a finding rather than a design: an earlier version
rewrote every phrase, and the songs came back sparse and sung by a single voice.

**Matching the words to the sections is the whole foundation, and the obvious way is wrong.** YuE2
does not write one section per lyric block. It merges neighbours — a measured plan gave one `% verse`
112 notes, which is a 63-syllable verse plus the 47-syllable pre-chorus after it — inserts sections
that carry no words at all, and renames what is left by its own reading (a first `[Chorus]` became
`% pre-chorus`). Pairing them in order therefore puts somebody else's lyrics under nearly every
section, and everything downstream is then applied to the wrong text. `align` matches them by their
counts instead, which works because the model writes about one note per syllable: 1.03 measured over
a whole song, and exactly 47/47 and 39/39 on individual sections.

**A section's words start at its pickup, not its barline.** YuE2 writes an upbeat the way any song has
one: a chorus opening «Ми не-втом-ні!» was planned as three quick notes in the last beat of the
section before, «ні» on the downbeat, then a breath. Counted by barline, that song's two choruses and
its outro each came out three notes short and the sections before them three over; counted from the
pickup, all three were exact. So a run of notes in the last bar of a section, after at least a beat of
rest, that carries straight on into the next section's first note counts for the next section — in
the matching, the report, and both edits: `refit` leaves a pickup exactly as written, and `recast`
moves one with the section it opens.

What the report then tells you, before a render is spent:

- **which block landed where**, so a merge or a dropped section is visible rather than surprising;
- **a block the plan cannot sing in full** — a bridge given 17 notes for 39 syllables loses well over
  half its words;
- **a block the plan will not sing at all** — a measured plan gave an 8-syllable outro no notes;
- **which register each section uses** — with the LoRA, what decides who sings it once it agrees
  with the marker;
- **a section whose register contradicts its marker**, which is the case `recast` can act on.

**`recast` is on by default.** Only a section that disagrees with its marker moves, and it moves
whole, by octaves, with its pickup; a move that narrows the gap between the registers is taken back.
Moving phrases individually toward a band centre, which the first version did, smeared the two
clusters together and made things worse. With a LoRA on the text encoder this is the edit that put
every section with its singer above; on the base model expect little from it. The report's first
line gives the registers as sent, after the move; its finding gives the gap the plan came with.

**`refit` is off by default, and the default is the recommendation.** The idea is sound — the plan's
note count really is the syllable budget — but the edit is not. Removing a note conserves the bar it
lives in, so it does not free time, it stretches the notes that remain: a real run lost 96 notes,
pushed the vocal line from 50% silence to 57%, and sounded like the singer labouring through it. What
the model itself does is sing a line at a natural rate and rest the remainder of the bar.

The **`lyrics` output passes the text through as written**, and that default is a reversal. The
markers were being stripped, on the reasoning that YuE2 has no field for a singer and its own guidance
says to keep instructions out of the lyrics. Listening says otherwise: the model reads them and takes
the **performance** from them — `powerful belts`, `deep growl`, `vocal duel, intense emotional peak`
come back as strain and intensity, and a stripped lyric gives a flatter take. On their own the markers
never decided who sings; with the LoRA, a plan that agrees with them does.

`keep_markers` off still strips them, for the one case that needs it: a stage direction in ROUND
brackets — `(distorted bass, atmospheric guitar)` — which YuE2 reads as a backing vocal and sings
aloud. Square-bracket markers are safe either way.

### `Satyr Trim (Plan → Plan) 🐐`

Shortens a plan by cutting the stretches where nobody sings. This is the lever that was confirmed by
ear first and reasoned about second.

A measured plan ran **4:33 for a lyric AceStep had sung in 3:00**, and 61 of its 171 bars — 98
seconds, 36% of the song — carried no sung note at all: a 40-second introduction, a 29-second outro,
an interlude. Dropping only those bars brought it to 3:45 with every one of its 326 sung notes
untouched, and the take came back shorter and livelier. **The model obeys a bar count**, which is
what makes the whole suite worth having: the plan's time axis is real.

`intro_seconds`, `outro_seconds` and `between_seconds` set how long each kind of silent stretch may
run. 0 removes one outright; a number larger than the stretch leaves it alone. Wire `seconds` into
`Empty YuE2 Latent Audio`.

- **Nothing sung is ever removed.** The unit of the cut is a whole group — one span carried by both
  voices at once — and only groups whose Vocal line holds no note are candidates. The report prints
  the sung-note count before and after, and the node refuses to return a plan where they differ.
  Melody, words, registers and both singers come through as the model wrote them.
- **Which end survives depends on where the stretch sits.** An introduction leads *into* the singing,
  so its last bars are kept — they are the approach. Everything else leads *out* of what it follows,
  so the first bars are kept and the trailing repetition goes.
- **A stretch is a run of sections, not a section.** An introduction is sometimes one `%` label and
  sometimes four in a row; a limit applied per label would cut a four-label intro to four times the
  length asked for.
- **Sections are classified by what is in them, not by their labels.** A `% verse` the model wrote
  with no vocal is an instrumental passage whatever it is called, and an `% interlude` that sings is
  not a gap. A song that opens on a voice has no introduction to cut, however long its first
  instrumental section is.
- **Cuts land on group boundaries**, so a plan written in four-bar groups trims four bars at a time
  and a limit is met from below rather than exactly. One exception to "at most": a limit above zero
  always keeps one group, so a six-second interlude asked for five seconds is shortened to itself
  rather than deleted. Zero is how to say delete.

What it does **not** do is make the singing itself denser. The vocal sections of that same plan ran
1.2–2.0 syllables a second, which is slow, and the bars carrying those words are not touched here —
only the ones carrying none.

### `Satyr Import (MIDI → Plan) 🐐`

A MIDI file becomes the plan. This is the **only real control over the tune** the suite has: who
sings is settled in the style string and the plan gets no say, but what they sing is written right
here, so a melody composed in a DAW can be the one YuE2 performs.

Two parts become the `Vocal` and `Ins` voices. Tempo, meter and key come from the file's own meta
events, and **MIDI markers become `% section` labels** — the one place a DAW's arrangement markers
carry straight through. Point `midi_path` at a file (a full path, or a name in ComfyUI's input
folder) and name the parts, or leave them empty and let the guess run: a name containing *vocal*,
*lead* or *melody* wins, otherwise it is the order they appear in. The report lists every part with
its note count and pitch range, so one run tells you what to type.

**A part is a channel, not a track**, and the difference is the whole ball game on a downloaded MIDI.
A type-0 file — which most exports and nearly every file found on the internet is — keeps the entire
arrangement in one track and tells the parts apart by their MIDI channel. Reading a track as one line
therefore hands the melody, the bass, the guitar and the drum kit to the singer all at once, and YuE2
sings every note of it, at length and with conviction. So channels are split out, named by their
General MIDI family (*Strings*, *Bass*, *Guitar*), and **channel 10 is marked `[drums]` and never
chosen automatically** — its note numbers are drum sounds, not pitches, and asking for them by name
gets a warning rather than a performance.

Two more things the report says about the part it was given. **A Cyrillic name is decoded**, because
MIDI declares no encoding and a Russian sequencer's track name arrives as mojibake otherwise. And a
part that **sounds for almost the whole song without a break is flagged**: a sung line breathes, an
arrangement does not, so a line with no rests in it is a warning that the wrong part was picked.

**Nothing lossy happens quietly**, and there are four lossy steps:

- **Quantising.** Notes are snapped to the `L:` grid, and `grid: auto` picks the coarsest one the
  music actually fits — a 1/32 grid can write any rhythm and produces a line nobody can check. The
  report gives the average distance notes had to move: a MIDI quantised in a DAW converts exactly, a
  live take does not, and 0% versus 6% is the difference between a transcription and an
  approximation.
- **Flattening to one line.** ABC's voices are monophonic, so a chord keeps its top note and the
  report counts what was dropped and what was cut short. A melody under a held pedal tone comes out
  as the pedal — worth knowing before blaming the model.
- **One tempo.** A plan carries a single `Q:`, so a tempo map is reduced to its first value and the
  count of what was ignored is reported. Same for a meter that changes partway.
- **No chords.** MIDI does not carry chord symbols. Use `mode: melody` in YuE2, which is what its own
  documentation recommends for anything built on an existing melody.

Two things that look like details and are not:

**A note crossing a barline is one syllable.** ABC has to break it at the bar and tie the halves, and
a missing tie reads as two ordinary notes — the singer is handed a syllable that does not exist and
every word after it shifts. Nothing about the text looks wrong. The first draft made this mistake.

**An accidental holds to the end of its bar, across octaves.** After `^F`, every F in that measure
sounds sharp, so a later natural one has to be written `=F`. Spelling each note against the key
signature alone is right until a bar contains two spellings of a letter, and then it is a silent
wrong note. The converter tracks the bar as it writes it.

The check that all of this worked is in the test suite and is one line: convert, parse the result
with the pack's own reader, and every MIDI pitch must come back. That covers the key signature, the
accidentals, the octave marks and the letter spelling at once.

**The file also knows half the style.** YuE2 asks for *Language + Genre + Vocal Character + Tempo +
Instruments*, and a MIDI answers the last two exactly rather than by guesswork: the tempo is written
in it, and the General MIDI programs name every instrument in the arrangement. Put the half it cannot
know — language, genre, who is singing — in `style_prefix`, and the `style` output comes back whole:

    Ukrainian, Progressive Alternative Rock, Post-Grunge, dual vocals, powerful melodic female
    vocal, building tension gritty male vocal, 145 BPM, electric bass, violin, drums

That matters more than it looks, because the style is what YuE2 builds the accompaniment from, and
*distortion guitar* is a different song from *electric guitar*. Instruments are ordered by how much
each part actually plays, so a four-note triangle does not displace the guitar, and the list stops at
six: a style naming a dozen things has stopped describing anything. A meter is mentioned only when it
is not 4/4.

**What to do with it.** `abc` goes to `YuE2 Generate Music` (or through `Satyr Trim` first) and
`style` to the same node's style input. Better still, run the plan through **`Satyr Score`** with your
lyrics: a tune written without the words in mind will not have the right number of notes for them,
and that report says exactly which section is short. For a cover of something that already exists,
core's `SheetSage2 Audio to ABC` does the same job from audio — the two are complementary, one for
what you wrote and one for what you have.

### `Satyr Edit (Plan → Plan) 🐐`

A plan, edited by hand in a piano roll — no notation, blocks on a grid. Run the graph up to the node
once, press **✏ Open the editor** on it, and the plan opens full-screen: both voices (`Vocal` orange
and pink, `Ins` teal), the section lane, the bar ruler with the time and any key change, and the
chord lane. **Space** plays from the cursor with plain tones — a triangle for the voice, a filtered
saw for `Ins`, a quiet pad for the chords; nothing is downloaded — and a click in the ruler moves the
cursor. The wheel scrolls, Shift+wheel scrolls sideways, Ctrl+wheel zooms, and the strip at the bottom
is the whole song with the visible part framed; click it to jump.

**Editing** — every edit is one step of **undo** (Ctrl+Z / Ctrl+Y).

**🔒 Vocal rhythm** is on when the editor opens: sung notes go up and down, nothing else. YuE2 wrote
that rhythm for the words' stresses, and on a real render every sung note added, removed or split
moved the pauses inside the lines — the words came out phrased as if they were written differently.
So an edit that would change the Vocal rhythm is refused, and the status line says why; whole bars may
still come and go, and `Ins` moves freely. Unlock it to change the rhythm anyway.

- **Notes.** Click to select (Shift adds), drag across empty space to select a box, Ctrl+A for all.
  Drag a note up or down by semitones; the arrows nudge (Shift+↑/↓ an octave). With the rhythm
  unlocked a note also drags sideways on the grid or stretches by its end, ←/→ nudge it, Del or a
  right-click deletes, a double-click on empty space draws a note into the voice set in **Draw into**,
  on the **Grid** (1/16 … a bar), and a double-click on a note **joins** it to the note right before it
  — one syllable carried onto a new pitch — or splits it off again. A voice sings one note at a time,
  so a note dropped on others cuts them back rather than overlapping them. **Ctrl+C / Ctrl+V** copy
  notes: pick a phrase, copy it, click the ruler where it should go and paste — voices and pitches as
  they were, cutting back what it lands on.
- **Bars.** Drag along the ruler to pick bars, then **Delete** them (both voices, all chords — the
  way to shorten an interlude by hand), **Duplicate** them, or **Insert** as many empty ones before.
- **Sections.** Click one in the section lane to rename it (YuE2's own vocabulary), duplicate it,
  delete it or move it one place earlier or later.
- **Transpose** ±1 / ±12 moves whatever is selected: picked notes alone; picked bars or a section
  with their chords and key — a modulation, with the old key back right after it; or, with nothing
  picked, the whole song, its key and every chord, respelled the way SheetSage2 spells that key.
- **Chords.** Double-click the chord lane to type one from that beat (empty: no chord from there);
  right-click one to remove it. **♩=** sets the tempo.
- Hide a voice with its toggle and it can neither be played nor touched.

**The words.** Wire the same `lyrics` — and `voice_1` / `voice_2` — that go into Satyr Score, and the
editor shows which syllable each `Vocal` note sings: on the note itself where it fits, and in the
lyrics lane under the chords (zoomed out, each line is written from its first note).

**The lyric goes onto the whole song, and the sections follow the words.** That is how YuE2 sang on a
real render, where the plan's own labels would have misled: it had written the intro's lines into the
first bars it labelled `verse`, sung the first pre-chorus over a long break with no notes, a bridge
over an interlude with no notes at all, and a chorus of 52 syllables whole before that interlude. So
the editor lays the lyric line after line onto the song's phrases — and onto its silences of two
bars or more, where the model sings over the music — in the runs whose counts agree best: a line
broken by a breath spans two phrases, two short lines share one. Inside a run a syllable goes on each
note, the last held over any notes left (drawn **~**), or several on the longer notes when there are
too few. Measured against that render, the first lines landed within half a second of where the
model sang them. A word with no vowel, like «в», is sung with the next word and shown on its first
syllable.

The section lane then has two rows. On top, the song's sections **by its words** — each lyric block
from where its words begin, with who it is marked for and its notes against its syllables (amber when
far apart, red when its notes sit in the other singer's band, ⚠); click one to pick its notes. Under
it, thin, the plan's own sections as YuE2 labelled them — click one to rename, move, duplicate or
delete it, or to pin what it sings. **Sings:** is for where the counts cannot tell: the notes before a
last chorus that could carry the end of a bridge, or the chorus's first words — pin the interlude to
the bridge and they are the chorus's. A section pinned to a block sings that block and no other;
pinned to nothing, it sings nothing. Pinned sections show 📌. The pins are the editor's: the saved
plan carries none.

**Who sings where.** The singers lane, over the lyrics lane, lays the lyric's runs — the lines one
voice sings in a row, or one bracketed backing line — where they are sung, in the colour of the band
the lyrics' singer belongs in, and outlines in red a run whose notes sit in the other singer's band.
Click a run to pick its notes; **+12 / −12** then hands them to the other voice, and the rhythm stays
exactly as the model wrote it.

**The lyric sheet.** **Lyrics** shows the whole lyric beside the plan, as written, markers and all; drag
its edge to make it wider or narrower (the width is kept) — it never covers the plan. The lines whose
notes are on screen are lit, the line at the cursor brighter; click a line to go to it and pick its
notes.

**Why the rhythm is locked, measured.** An earlier version read the plan as one syllable per note
straight through the song, saw the words drift, and added notes until every section had a note per
syllable. The render was worse for it: the notes invented for the intro came out as fragments, and
the added and split notes moved the pauses inside the lines. The model had sung its words where
they fit all along; it writes a plan's rhythm for its lines, and notes added by hand only get in the
way.

The band line names the singer each band belongs to. Hovering a note says what it sings, in which
line; **dragging across the lyrics lane picks the notes that sing those words** — then Transpose
moves exactly those. **ⓘ** lists what Satyr Score would report. The
words follow the notes while you drag them and are laid on afresh after every edit; the plan going
out is not changed by them. The lyrics and voices are kept on the node, so a frozen run still shows
them.

**💾 Save** writes the plan back to the node and turns `use_edited` on. What the save kept and what it
re-rendered is shown in the editor, on the node, in its `report` and in the console. From then on the node sends the
edited plan and does not evaluate `abc` at all, so whatever feeds it — `YuE2 Generate ABC`, an LLM —
does not run again: the same freeze as Show Text's saved text. Turn `use_edited` off and the upstream
plan passes through again; the edit stays on the node until **discard the edit**. The editor always
opens on what the node sends: the edit while `use_edited` is on, the upstream plan otherwise. The
upstream plan is kept in the workflow (`plan_state`), so a reload does not force a re-run to open it.

**What you did not touch comes back as the text it was.** A group — one span both voices carry at
once — whose bars, notes, chords, key and meter are all unchanged, in the same context, is written
back as its original lines, byte for byte, so a diff of a save shows the bars you changed and nothing
else. Saved without an edit, a plan comes back identical, line ends included. **What you did touch is
written the way ComfyUI's own SheetSage2 exporter writes plans** — at most four bars a group, the chord
restated at every barline, key-relative spelling (`^^F` rather than `=G` in D# minor), long notes split
into lengths the format has and tied, empty bars folded to `Z` — because that is the text YuE2 learned
from. Forcing every bar of an exporter-written plan through this writer reproduces the exporter's
text byte for byte; the suite checks exactly that, key changes inside a bar included.

**A note is one sung event, however it is spelled**, which is what the syllable budget needs. A note
held across a barline, or from one group into the next, is one note and one syllable; the editor shows
it as one block, and the writer splits and ties it again. A tie onto a different pitch — one syllable
carried onto a new note — is kept as such and marked on the block.

**The band line.** When a plan has two voice bands the line between them is drawn dashed and every
`Vocal` phrase is coloured by the side its median sits on — pink above, orange below. It shows who
the model wrote each phrase for, which is the useful reading; it is not a switch. Moving a phrase
across it flipped the singer once at a fixed seed and has not worked as a lever since — how many
singers a song has is settled by the style string.

`report` says which plan went out and its length, key and tempo, with anything structurally wrong;
`seconds` is its duration, for `Empty YuE2 Latent Audio`.

### `Satyr Music (Guided YuE2) 🐐`

A drop-in replacement for `YuE2 Generate Music` that exposes **`cfg_scale`**, and that is the whole
of it. YuE2's own reference implementation takes a guidance scale — `protocol.py` validates 0–20 —
and ComfyUI's node never passes one, so guidance is switched off and the second branch is never even
built. The scale is reachable anyway, because the tokenizer reads it out of its kwargs.

**What the scale amplifies is not the obvious thing, and this is the reason to care.** The negative
branch is built from the instruction alone, with `[Tags]` and `[Lyrics]` removed, and the ABC is then
appended to *both* branches. Read against the real tokenizer:

    positive:  Generate a chord-annotated ABC transcription … \n[Tags]\nenglish, female vocal\n[Lyrics]…
    negative:  Generate a chord-annotated ABC transcription, then generate music with codec tokens…

So the plan cancels out of the difference entirely. Raising the scale does **not** weaken the plan —
it pushes the style and the words harder against a plan whose authority is unchanged.

**What that buys, measured, is diction.** Words the model tends to swallow come through, and the
clearest case is a line in round brackets, which it otherwise sings too quietly or drops altogether.
Raise it when a take is right but a phrase is mumbled. What it does **not** do is change who sings —
that was the hope this node was built on, and it was tested, and it does not. The result fits the
mechanism: amplifying the words makes the model articulate what it was given, not rearrange it.

`1.0` is off and is the default, so the node behaves exactly like the core one until it is turned up.
Anything else runs two branches: roughly twice the VRAM and twice the time.

The node refuses to run if the tokenizer gives the argument back unchanged, rather than silently
generating with guidance off while reporting otherwise.

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

**The input is kept below a crossover, the model is used only above it.** `crossover = auto` measures
where the input's spectrum ends (the highest frequency within 60 dB of its 1-4 kHz level, as a median
over frames so a hard start out of silence cannot fake a wide band) and splits 1 kHz below that:
`LP(input) + (model − LP(model))` with one linear-phase Kaiser filter and its exact complement. The
join has no hole, no bump and no delay, and everything under it is the input's own audio — the model
cannot touch the body of the mix. On an AceStep mix that puts the split at about 11 kHz. `manual` takes
`crossover_khz`; `off` is the old behaviour, the model's output everywhere. The report prints where
the input and the output end. The idea of a complementary crossover comes from *Refine* in Johannes
Plenio's [Plenio Music Production System](https://github.com/jplenio/Plenio-Music-Production-System);
the implementation here is our own.

**`match_level`** (only with the crossover off) puts the output's energy *below 10 kHz* back where the
input's was. That is what the crossover replaced: on the same take the model drifted -1.2 dB at
0-4 kHz and -1.7 dB at 4-8 kHz, which reads as the mix losing body — a level fix corrects the dB but
not the content the model rewrote down there.

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

## 💾 `save_song/` — Save Song, Song Tags & Remaster

> **System Purpose & Overview**  
> Bring a finished track to a loudness target, then save it with metadata and artwork integration.

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

**The cover and the lyrics also go *inside* the audio file**, because the `.jpg` and `.txt` beside
it are gone the moment the song is copied onto a phone, and every player on earth reads its title,
artwork and lyrics out of the file itself. MP3 gets a real **ID3v2.4** tag (`TIT2`/`TPE1`/…, the
lyrics as **USLT**, the cover as **APIC**); FLAC and Opus get **Vorbis comments** with the cover as a
`METADATA_BLOCK_PICTURE`. The **title is always filled** — from the file's own name when nothing
else supplies one, so a player never shows a blank where the song should be.

**`Song Tags`** carries the rest: `title`, `artist`, `album`, `album_artist`, `track`, `year`,
`genre`, `comment`, and an `extra` field taking one `name: value` per line. It is a separate node
because it is filled in *once* while Save Song is the one you keep re-running — and because eight
more widgets on Save Song would be eight more rows of a node you look at constantly. Wire its
`tags` output into Save Song's `tags` input; empty fields are simply not written.

In `extra`, names a player actually understands (`bpm`, `composer`, `publisher`, `copyright`,
`isrc`, `disc`, `language`, `mood`, `key`) land in the real field, and anything else becomes a
custom tag — `TXXX` in MP3, a named comment in FLAC/Opus — which is where `seed: 998877` belongs.
The optional `settings` input takes **Generation Info Filter**'s `settings_data`, so every setting
it selected rides along inside the song and the file remembers how it was made. Only the *first*
dump is used: this node describes one song, not a batch.

**No new dependency for any of it.** PyAV is already here — it is what ComfyUI's own Save Audio
encodes with — and it carries text metadata for all three formats. It cannot carry the cover: that
needs an `attached_pic` stream, which in PyAV 17 fails both ways (setting `stream.disposition`
raises, and muxing one tries to open an *mjpeg encoder* and errors out). So the two picture
containers are written as plain bytes in `tagging.py` instead of reaching for `mutagen`: an ID3 tag
is a header, a synchsafe length and a run of frames, and a Vorbis picture is a base64 block. Both
were verified by writing files and reading them back with ffmpeg — cover byte-identical, UTF-8 text
(Cyrillic included) intact, duration untouched.

One trap that is invisible until the tags go missing: **FLAC keeps its comments on the container,
Opus on the stream**. Set Opus's on the container and they vanish with no error at all.

**`Remaster (Loudness) 🎚️`** goes before Save Song (after Audio SR, if you use it) and brings a finished
take to a loudness target — **-14 LUFS** by default, what Spotify and YouTube normalise to — with its
**true peaks** held under a ceiling of **-1 dBTP**. Neither music model masters what it writes, so
without it every song is saved at whatever level it came out at. There is no compressor, on purpose:
the takes arrive already mixed. Outputs the `audio` and a `report`. Category `Kinburg-Nodes/audio`.

- **Loudness** is BS.1770-4 integrated loudness, torchaudio's implementation. The report also gives
  the **loudness range** (EBU Tech 3342, from 3 s short-term loudness on the same K-weighting) before
  and after, which shows how much of the song's dynamics the limiter took.
- **True peak** is read from the signal oversampled four times (twice from 96 kHz up). A waveform
  swings higher between samples than at them, and an MP3 or Opus encoder reproduces that swing, so the
  ceiling holds the true peak rather than the sample values; -1 leaves room for the encoder.
- **The limiter sees the whole song at once.** What every sample needs is known in advance, so the
  gain comes down over the 5 ms before a peak instead of clipping its front edge, recovers at 40 dB a
  second after it, and is one gain for both channels, so the stereo image stays put. Limiting takes a
  little loudness off, so the gain is nudged up and the limiter rerun until the target is met.
- **`max_limiting`** (6 dB) caps how much the limiter may take off a peak. A dynamic take can need far
  more to reach -14; then the node stops at the loudest level the cap allows, and the report says how
  far short it fell and what the target would have needed. `0` makes it a plain true-peak normaliser.

A loud take is turned down the same way. Silence, or a clip shorter than one 0.4 s gating block,
passes through unchanged. Five minutes of stereo at 48 kHz takes about five seconds on the CPU. The
idea of a mastering step that stops short instead of squashing comes from Johannes Plenio's
[Plenio Music Production System](https://github.com/jplenio/Plenio-Music-Production-System); the
implementation here is our own.

---

[← back to the node index](../README.md#-node-index)
