# Kinburg-Nodes

A personal collection of custom ComfyUI nodes. One folder = one package: ComfyUI reads
the node mappings from the root `__init__.py`, and the sets are split into subpackages.

## What's inside

### `local_llm/` — Local LLM (GGUF)
> **All the settings below live on the `Local LLM Settings (GGUF)` node** and reach the LLM node
> through a single **`config`** link. There's now **one** node — **`Local LLM (GGUF)`** — with
> `config` + `user_prompt`, plus an optional **`image`** input (connect it, with a `Vision Settings`
> node on the config, for vision) and two optional connect-only overrides — **`system_override`**
> (replaces the config's system prompt) and **`grammar_override`** (a GBNF grammar that forces
> `gbnf_grammar` output). A **`live_preview`** toggle streams the output to a **`Kinburg Live Log`**
> node as it's written (see below). The chat node uses the same bundle. (See the Settings/chat section below.)

Run a GGUF LLM right inside ComfyUI **with guaranteed VRAM unloading**: inference
runs in a separate worker process, so when it exits the OS reclaims all of its VRAM —
ideal right before image generation. Features: streaming progress bar, token counters
(including an estimated `thoughts_tokens` / `answer_tokens` split of the output),
`finish_reason`, a separate `thoughts` output, reasoning control (Qwen3 `/no_think` and a
configurable `answer_marker` for models that reason without `<think>` tags), `min_p` /
`stop`, flash attention, KV-cache quantization, and structured output (JSON / GBNF / a
built-in Ideogram prompt grammar). An **`extra_load_args`** field passes advanced keyword
args straight to llama-cpp-python's `Llama()` loader — `key=value` per line (e.g.
`n_threads=8`, `main_gpu=0`, `tensor_split=[1,1]`, `rope_freq_base=1000000`) or a JSON object;
unknown keys are ignored and changing it reloads the model. Note these are **Python-binding
args, not llama.cpp CLI flags** — CLI-only flags like `--spec-type draft-mtp` have no effect
here. These options live on the **`Local LLM Settings (GGUF)`** node (see below).

**`Kinburg Live Log 📜`** — turn on a source node's **`live_preview`** toggle and drop a
`Kinburg Live Log` node anywhere (no connections): it shows the output as it streams, token by token,
a block per generation labelled by the source node — and, when the source labels its individual
calls, by the call (`Morpheus Storyboard · shot 2/4 (2 keyframes)`). One log shows every LLM node in
the pack at once: `Local LLM (GGUF)`, `Chat`, `Morpheus Storyboard` and the Morpheus sampler's
in-loop writer. Blocks can carry **thumbnails** — the frames a vision call was actually shown, and
Morpheus posts each shot's last frame as it is decoded, so the log doubles as a live storyboard of a
render; thumbnails have a hover copy-to-clipboard button. (The old `LLM Live Log` id still loads and
behaves identically, hidden from the picker. The Ouroboros loop keeps its own
`Ouroboros Live Log 🐍📜`, whose iteration/score/stage layout is a different shape.) Each block's header counts the tokens generated so far
against the run's `max_tokens` ceiling (`142/512 tok`) plus the live `tok/s`, with a thin **budget
bar** under it, and on finish adds the context fill (`ctx 1780/4096 (43%) · prompt 1268 + gen 512`)
— amber when the output hit `max_tokens` or the context is nearly full. **Reasoning is separated
from the answer**: `<think>` blocks (or everything before your `answer_marker` line) render dim and
italic in a collapsible section headed `thinking… 4.2s`, which folds itself away the moment the
answer starts — click it to keep it open. The split mirrors the node's own, so what the log shows
as the answer is exactly the `text` output. Hover a block for a **copy** button (copies the answer,
without the reasoning); **`clear`** in the header wipes the log. The view follows
the newest text only while you're parked at the bottom: **scroll up and it stays put** while
generation continues, with a **`↓ latest`** pill to jump back. **Text runs only** — a grammar/JSON
run (e.g. a card via `grammar_override`) takes the worker's non-stream path, so its result shows up
in the log once it finishes rather than token by token. Same websocket mechanism as the Chat node's
live reply.

A **`chat_template_path`** field (also advanced) points to a `chat_template.jinja` file that
**overrides the model's built-in chat template**. Leave it empty (the default) and llama.cpp uses
the template embedded in the GGUF — correct for almost every model, so you normally need nothing
here. Set it only when a GGUF ships a **broken or missing** embedded template (answers come out
malformed / the system prompt is ignored) or when you want a **specific template variant** (e.g. a
community template that adds tool-calling or thinking toggles). It applies to **text models only** —
it is ignored while an `mmproj` (vision) is active, since the vision path does its own multimodal
formatting. Surrounding quotes are stripped and changing it reloads the model. Tip: if you're not
sure whether a downloaded `chat_template.jinja` differs from the one already inside the GGUF, it
usually doesn't — well-packaged models embed the right one.

The single **`Local LLM (GGUF)`** node runs it: wire a `config`, type a `user_prompt`, and read
the outputs. **Vision** is built in — connect the optional **`image`** input and add a **`Vision
Settings (GGUF)`** node (`mmproj` + `vision_handler = auto (MTMD)`, which handles most modern vision
GGUFs — LLaVA, Qwen2-VL, MiniCPM-V, Gemma 3, SmolVLM, …) onto the config; without an mmproj a
connected image raises a clear error. `image_max_side` (on Vision Settings) downscales before
sending; pair vision with `output_format = json_object` for a structured image description. Two
optional connect-only inputs override the config for that node: **`system_override`** (replaces the
system prompt) and **`grammar_override`** (a GBNF grammar that replaces the config's grammar and
forces `gbnf_grammar` output). Category `Kinburg-Nodes/LLM`.

The **`model`** (Local LLM Settings) and **`mmproj`** (Vision Settings) pickers scan
`ComfyUI/models/llm` **recursively**, so you can group a model with its `mmproj` — or organize
families — into **subfolders**; they show as `folder/model.gguf` in the dropdown (and the newer
ComfyUI frontend renders `/` as nested submenus). As before, pick the placeholder to type any
absolute path in `model_path` / `mmproj_path` instead.

**`Local LLM (server client, text)`** is a different beast: instead of the Python binding it
talks HTTP to an OpenAI-compatible LLM **server**, so you get the server's **full command
line**. A **`backend`** selector picks:
- **llama-server (launch)** — launches llama.cpp's `llama-server`; `extra_args` reaches any
  llama.cpp flag, e.g. `--spec-type draft-mtp` (MTP speculative decoding), `--flash-attn`,
  `--model-draft path/to/draft.gguf`.
- **koboldcpp (launch)** — launches `koboldcpp` with its own flag names (`--model`,
  `--contextsize`, `--gpulayers`) plus Kobold extras via `extra_args` (readiness via
  `/v1/models`; usual port 5001).
- **connect to running server** — launches nothing; set **`base_url`** (e.g.
  `http://localhost:5001`) and it calls a server you already run — koboldcpp's GUI, LM Studio,
  Ollama, vLLM, a remote box…

For the launch backends point **`server_binary`** at the executable (download it yourself —
neither is bundled). A launched server starts on demand, is reused while
backend/binary/model/`n_ctx`/`n_gpu_layers`/`host`/`port`/`extra_args` stay the same, and is
shut down on exit — or after each run when **`keep_alive`** is off, to free VRAM. Sampling, the
reasoning split (separate **`thoughts`** output, `strip_think`, `answer_marker`), reasoning
directives and structured output (`output_format` / `grammar`, incl. the Ideogram preset) match
the llama-cpp-python text node, so its full output set is here too (`text` / `thoughts` /
`finish_reason` / token counts / `gen_seconds` / `help`) plus a `server_log` tail for
troubleshooting. `unload_comfy_models` frees image models first; `ready_path` overrides the
health probe. Text only. Node: **`Local LLM (server client, text)`** (category
`Kinburg-Nodes/LLM`).

### `context/` — Character Card, Entity Card, Context Collector
Feed an LLM node reference material so it weaves named subjects into an expanded image prompt.
**`Character Card`** has fields (name, gender, age, eyes, hair, build, outfit, distinctive
features, free-form notes…) and outputs one tidy Markdown block, **skipping every empty
field**. **`Entity Card`** is its free-form sibling for non-people (an object, place, faction…) —
a name + a description. Both cards carry a **`save_preset_as`** field (+ optional comma-separated
**`tags`**): type a name and run to save the card to a reusable library — it saves at run time, so
it captures whatever's filled in, whether **typed or wired in** (e.g. from a photo description);
reuse it via **Card Presets** (below). To save an LLM's JSON card in one step instead, use **Card
Save**.
**`Context Collector`** gathers any number of cards / text chunks (auto-growing
`item_N` inputs, empties skipped) under a `title` and wraps them in a delimited block —
`<context>…</context>`, a custom tag, a Markdown heading, or none — so the model can tell
reference data from the instruction. Wire its `context` output into an LLM node's new
**`context`** input (present on all three LLM nodes; it's appended to the system prompt). Then a
prompt like *"Vasya and Kolya drink tea in a cafe"* comes back expanded with each character's
looks. Note: the diffusion model still has its own limits binding attributes across multiple
people. Category `Kinburg-Nodes/LLM`.

### `vision_judge/` — Vision LLM Judge
**`Vision LLM Judge`** scores a batch (or list) of images with a **vision GGUF** against a rubric
you write, returning a structured verdict per image — a **GBNF grammar** forces clean
`{score, tags, comment}` JSON on any model (the same guaranteed-structure trick as the Ideogram
preset). Wire a **`config`** (a `Local LLM Settings (GGUF)` with a `Vision Settings` mmproj), the
**`images`**, and a **`rubric`** (optionally the per-image **`prompts`**, `---`-separated, so it
can judge prompt adherence); the scale is set by `score_min` / `score_max`. It reuses the LLM
nodes' worker and keeps the model loaded across every image. **Multi-criteria mode** (optional):
fill **`criteria`** — one per line, `name` or `name: description` (e.g. `anatomy: hands, limbs` /
`sharpness` / `prompt_adherence`) — and the judge scores **each** criterion on the scale, with the
overall `score` being their average; the GBNF grammar is generated on the fly to force
`{scores:{…}, tags, comment}`. The field is **pre-filled with a sensible example** (overall_quality
/ anatomy / prompt_compliance / camera / text); edit it, or **clear it** for a single overall score
(the original behaviour). The two built-in prompts are **editable**: **`system_prompt`** (the judge persona —
the default is pre-filled) and **`comment_style`** (how the comment reads, default *one concise
sentence*; set e.g. *two to four sentences covering strengths and weaknesses* for a detailed
review); both fall back to the built-in default when blank, and the JSON shape stays managed (so
editing them only affects quality, never parsing). Outputs: **`results_json`** —
`[{index, score, score_max, tags, comment}]` (plus a
per-criterion **`scores`** object in multi-criteria mode), wire it into **Image Compare**'s
`judge_data` input for a read-only judge section per image (stars / tags / comment) alongside your
own review; **`summary`** (a readable per-image report, with the per-criterion breakdown when in
multi-criteria mode); and **`best_index`** (the top-scoring image). Closes the generate →
auto-evaluate → pick-the-best loop entirely locally. Category `Kinburg-Nodes/LLM`.

**`Criteria Builder 📋`** builds that `criteria` string for you — tick criteria instead of typing
them. It carries a catalog of curated, model-guiding descriptions (overall_quality, anatomy, hands,
face, composition, lighting, color, sharpness, background, camera, realism, text, …), one **toggle**
each; toggled-on ones are emitted as `name: description` lines. An **`extra`** box appends custom
lines, and an optional **`criteria_in`** input merges an upstream string first (chain builders, or
start from an existing set), with duplicates removed by name. The single **`criteria`** output feeds
both **Vision LLM Judge** and **Critic Settings (GGUF)** — right-click their `criteria` field →
**Convert widget to input** and wire it in (empty output = single overall score). The catalog is
`criteria_presets/catalog.json`; drop a **`catalog.user.json`** next to it (same shape) to add your
own criteria without editing the shipped file (it survives a git pull). Category `Kinburg-Nodes/LLM`.

### `ouroboros/` — Ouroboros (Self-Correcting Sampler) 🐍
**`Ouroboros (Self-Correcting Sampler) 🐍`** is a **closed-loop text→image optimizer** in one node.
Each iteration: an LLM **expands/rewrites the prompt** → an image is **sampled** → a vision
**critic** scores it and returns concrete **advice** (how to change the prompt) plus **negative-prompt
terms** for the flaws it sees → the advice feeds the next revision. It repeats until the score hits a
**target** or **`max_iterations`**, keeping the **best image across all iterations**. The loop is a
plain Python `while` inside the node (no graph-expansion loop), and generation is owned internally
via ComfyUI's own **custom-sampler API** (`comfy.sample.sample_custom`) / CLIP encode / VAE decode —
so the node is effectively a KSampler variant (the custom-sampler path is what lets Sampler Settings
expose per-sampler knobs like `eta`; sigmas come from a stock `KSampler` object, so single-stage
results match the classic path). It uses a **fixed seed** and a fixed start latent so only the prompt varies (a
clean optimization signal); the critic's advice targets the **weakest criterion** (see Vision Judge's
multi-criteria scoring), and negative terms **accumulate** (deduped) across iterations.

Settings follow the repo's bundle idiom, in two small companion nodes:
- **`Sampler Settings`** → `sampler_settings` (`SAMPLER_CFG`): seed, steps, cfg, sampler, scheduler,
  denoise, plus a **`seed_mode`** (`fixed` / `random` / `increment` / `decrement`) and **`seed_step`**
  — how the seed changes each iteration (`fixed` keeps the signal clean; the others explore, at the
  cost of mixing "prompt improved" with "seed got lucky"; `random` is reproducible from the seed).
  Advanced sampler knobs (reached via the custom-sampler API): **`eta`** (stochasticity for
  ancestral / SDE samplers — `0` = deterministic, higher = more variation; no-op on deterministic
  samplers), **`s_noise`** (extra-noise multiplier on stochastic steps; >1 = more grain/detail),
  **`s_churn`** (adds stochasticity to otherwise-deterministic euler / heun / dpm_2), and
  **`solver_type`** — which has **two separate vocabularies** in k-diffusion: `midpoint` / `heun` for
  the `dpmpp_2m_sde` family, and `phi_1` / `phi_2` for the SEEDS and exp_heun ones (`seeds_2`,
  `exp_heun_2_x0`, `exp_heun_2_x0_sde`). All four are offered; a value from the wrong family is dropped
  (the sampler keeps its own default) and noted once in the console, so no combination can break a run.
  Each knob is only passed to samplers
  that actually accept it, so unsupported ones are silently ignored (never an error). Defaults keep
  a single stage bit-identical to the classic path. (Image size comes from the latent — see below.)
  Category `Kinburg-Nodes/sampling`.
  <br>**Multi-stage refine (chaining):** Sampler Settings has an optional **`sampler_settings`** input —
  wire one node's output into the next to build a **chain** (left→right = stage order). Feed the chain
  into Ouroboros and it runs in **refine mode**: each iteration samples through every stage in order,
  the previous stage's latent feeding the next (stage 1 drafts at full denoise; give later stages
  their own sampler + `denoise < 1` to polish). The **critic always judges only the FINAL image** of
  the chain — there is no intermediate scoring. One Sampler Settings = the classic single pass
  (fully backward-compatible).
Optional **`model_negative`**: a second model for the **unconditional** pass, for checkpoints shipped
as a model + uncond-model pair (Ideogram). The loop builds the dual-model guider itself on **every
iteration**, so the freshly rewritten prompt is still what gets encoded — feeding a ready-made
`Dual Model CFG Guider` in instead would freeze the conditioning and defeat the loop, which is why the
input is a MODEL and not a GUIDER. ⚠ At **cfg 1.0** the guider skips the unconditional pass and the
second model does nothing; the console says so at the start of the run.

- **`Critic Settings (GGUF)`** → `critic_settings` (`CRITIC`): embeds a **vision `LLM_CONFIG`** (a
  Local LLM Settings with an mmproj) plus the evaluation rules: `criteria` (pre-filled), `rubric`,
  `score_min/max`, editable `system_prompt` and `advice_style`, `samples`
  (self-consistency: judge N times, take the median), and **`image_downscale`** — shrink the image
  the critic sees by this factor (`2` = half-size, `4` = quarter) for **fewer image tokens → faster
  judging and less VRAM**, at the cost of fine detail; `1.0` = full resolution (it never upscales
  past the Vision Settings `image_max_side` cap). Category `Kinburg-Nodes/LLM`.

Wire into Ouroboros: `model` / `clip` / `vae`, a **`latent`** (required — an Empty Latent matching
your model sets the image size; feed a real latent + denoise<1 for img2img/hires-feedback),
`user_prompt` (your intent), `negative`, `enhancer_settings` (a Local LLM Settings for the prompt
LLM — **its own `system_prompt` defines how to expand**), `critic_settings`, `sampler_settings`,
`target_score`, `max_iterations`, `low_vram`, and optional **`trigger_words`** (comma-separated words — e.g. **LoRA triggers**
from `Lora Unlim Accumulator`'s new `triggers` output — that are **always appended** to the enhanced
prompt so the LLM rewrite can't drop them). The two stop conditions — **`target_score`** (the loop
stops once the overall score reaches it) and **`max_iterations`** — now live on the Ouroboros node
itself, alongside **`enhancer_history`** (how many recent iterations — prompt + score — are recapped
to the enhancer LLM as "already tried, do not repeat"; `0` = none; the latest critic advice is always
sent). **Outputs** feed **Image Compare** directly: `images`
(all iterations) + `prompts` (`---`-separated) + `judge_data` (per-image scores/advice) + `captions`
(iteration labels) + `times` (per-iteration generation time) + `settings_data` (per-iteration
iteration#/score/seed/steps/cfg/sampler as `GEN_SETTINGS` — a per-stage breakdown in refine mode);
plus `best_image`, `best_prompt`,
`best_score`, a `report`, and `iterations`.

The node has a **`⏹ Stop loop`** button: press it mid-run and the loop finishes the current
iteration, then **stops and returns everything generated so far** (images, best, etc.) — a graceful
stop, unlike ComfyUI's Cancel which aborts and discards the run. (It works via a small backend flag
the node polls between iterations, so it takes effect at the next iteration boundary.)

**VRAM — the `low_vram` toggle (default on):** Ouroboros runs a diffusion model *and* one or two
GGUF LLMs, so it's the single control for how VRAM is shared. **On** keeps strictly one model
resident at a time — ComfyUI models are freed before each LLM call, the LLM worker is unloaded after
it, and, because every LLM load is then fresh, each call's **`n_ctx` is auto-sized to what the request
actually needs** (measured per role from the previous iteration's prompt tokens, rounded up with a
generation-budget cushion). This means the **`n_ctx` you set on the LLM Settings nodes becomes the
*ceiling*, not the amount always allocated** — so you can set it generously without paying the
KV-cache VRAM every step (the critic's figure includes image tokens; a rare under-shot reloads once
at the ceiling, so it never breaks the loop). **Off** keeps everything resident at the full `n_ctx`
— fastest, no reloads — for GPUs with VRAM to spare. Either way the live log's context line shows
`used/n_ctx`, so you can watch the auto-sizing at work. This toggle **overrides the `unload_*` flags
on the LLM Settings nodes** for the loop (those still apply to the standalone LLM nodes).

**Logging:** a **`full_console_log`** toggle (default on) controls **console/terminal** verbosity —
on, each iteration also prints the enhanced prompt, advice and negative additions (off = just a
one-line score); the full trace is always in the **`report`** output (wire it to a Show Text node)
regardless. For a **live in-canvas
view**, drop an **`Ouroboros Live Log 🐍📜`** node anywhere (no connections needed): it listens for
the loop's websocket events and shows a **thumbnail** of the image plus seed · score (+ per-criterion)
· time · the **full prompt** · advice · negative additions (failed/stopped iterations are flagged
too). Each LLM call also shows a **context-fill line** — `ⓘ enhancer/critic ctx used/n_ctx (%) ·
prompt + gen` — so you can see how close each request runs to the model's `n_ctx` window and whether
it was clipped: it turns amber and warns to **raise `max_tokens`** (output hit the ceiling) or
**raise `n_ctx`** (context ≥90% full, so the prompt itself is being truncated). The critic figure
includes the image tokens the vision model consumes. This side-steps the console's line truncation
entirely. Entries are **timestamped** (`HH:MM:SS`),
and a **`log_mode`** toggle on Ouroboros controls granularity: **`streaming`** (default) is like
`per step` but the enhancer's prompt **types out token by token** as it's written, its header
counting the tokens against the enhancer's `max_tokens` (`142/512 tok`) plus the live `tok/s`
(text only — the critic is grammar-constrained and can't stream, so its verdict still lands
whole); **`per step`**
posts each stage the moment it finishes — enhanced prompt → generated image → critic verdict, as
three live entries; **`per iteration`** posts one combined entry after the whole iteration. The log **survives
ComfyUI Desktop tab switches** — it's replayed from an in-memory history when the node is recreated
(kept in memory, not serialized into the workflow, so thumbnails don't bloat the file; cleared on a
new run and on full app restart). Hovering an image in the log shows a small **📋 copy** button in
its top-right corner that copies the picture to the clipboard (as PNG); hovering any entry shows a
**copy** button in its header that copies that entry's text (the prompt, or the advice + negative
additions), and **`clear`** in the log header wipes it. The log follows new rows
only while you're parked at the bottom — **scroll up and it stays put** while the loop keeps
running, with a **`↓ latest`** pill to jump back.

**VRAM discipline (built for small cards):** when a config's **`unload_comfy_models`** is on,
Ouroboros frees all ComfyUI models before each LLM call **and** frees the LLM worker before each
diffusion — so only **one model is resident at any moment** (diffusion *or* one GGUF). The cost is
model reloads around each phase; that is the accepted trade for running at all on low VRAM. On big
GPUs/farms, turn `unload_comfy_models` off to keep everything resident (no reloads); pointing the
enhancer and critic at the **same model file** further minimizes reloads. Category
`Kinburg-Nodes/sampling`.

### `chimera/` — Chimera (Multi-Sampler) 🦁
**`Chimera (Multi-Sampler) 🦁`** runs **two (or more) `Sampler Settings` bundles as consecutive stages
of ONE image**, the way the Ouroboros refine chain does — but with the **step budget under explicit
control** and a **physically correct handoff** between stages. It reuses the very same
**`Sampler Settings`** node (`SAMPLER_CFG`) documented above, so there is nothing new to configure:
wire one bundle into `stage_a`, another into `stage_b` (a chain also works and is flattened into
stages, left→right). `seed_mode` / `seed_step` are Ouroboros-loop dials and are ignored here — only
`seed` is used.

**The `handoff` mode is the point of the node:**
- **`continuous`** (default) — **ONE sigma schedule** is built once and each stage runs a **slice** of
  it. Stage A goes `0→a` and leaves the latent sitting at the sigma of step `a` (partially denoised);
  stage B resumes **at** `a` with **no fresh noise** and finishes `a→end`. The run is exactly one
  traversal of one noise trajectory, just with a different sampler / cfg / eta / model per segment.
  This is the equivalent of chaining two `KSampler (Advanced)` nodes through
  `start_at_step` / `end_at_step` — except the schedule **can't get out of sync**, because this node
  owns it. The
  shared curve takes its **scheduler and `denoise` from stage A**; a later stage's own `denoise` is
  reported as ignored, and a **different scheduler** on a later stage is rebuilt over the same length,
  sliced, and **pinned to the handoff sigma** (a mismatch there is what produces classic "refiner
  seam" artifacts).
- **`restart`** — the classic **two-KSamplers** pattern: every stage builds its **own** schedule and
  adds its **own** noise, so stage B **needs `denoise < 1`** or it re-noises from scratch and discards
  stage A (the node says so in the report). A genuinely different effect (img2img refine), not a worse
  one.

Because the node owns the schedule, a stage's **`steps` simply means "how many steps THIS stage
runs"**. **`step_split`** decides **where the boundary falls** — the last three modes are the same
decision in different units, so reach for whichever unit you're thinking in:
- **`stage steps`** — each stage runs exactly the `steps` it declares; the curve is their **sum**.
- **`at step`** — cut **`total_steps`** at **`handoff_step`**: stage A gets that many steps, the rest
  the remainder (`0` = halfway). Same unit as above, but the boundary lives on *this* node, so you can
  sweep it without editing the Sampler Settings — handy when hunting for the right split.
- **`at percent`** — the same cut as a percentage **of the steps**: 60% of 30 → 18 / 12.
- **`at sigma`** — the cut lands where the **noise level** crosses **`handoff_sigma`**. The mode for
  stages on **different schedulers**, where step indices aren't comparable but noise levels are — and
  the only one whose boundary stays put when you change the step count. The scale is model-dependent:
  flow-matching models (Flux / SD3 / Krea) run **1.0 → 0**, so the useful range is 0..1 and the `0.5`
  default sits mid-trajectory; SD / SDXL start around 14.6, where you'd want single digits. The report
  prints the curve's range and the sigma actually hit, so one run tells you the scale.

The last three ignore the `steps` set inside the Sampler Settings nodes.

**`total_steps`** (`0` = auto = the declared sum) doubles as a leftover-noise escape hatch: in the
`stage steps` split, setting it **longer** than the sum means the tail of the curve is never walked, so
the result deliberately **keeps residual noise** (reported). Shorter → the last stages are clipped.

**Two things inherent to splitting any sampler run** (they apply equally to a chain of two
`KSampler (Advanced)` nodes, and the node reports them rather than hiding them): **multistep samplers**
(`dpmpp_2m` / `3m`, `deis`, `ipndm`, `lms`, `res_multistep`, `sa_solver`, …) keep a history of previous
steps, which **resets at the boundary** — so the first step of stage B runs at reduced order (one step
out of N; single-step samplers like `euler` / `heun` / `res_2s` are unaffected and the split is exact
there). And with a **noise mask** on the input latent, a continued stage runs on zero noise, which is
also the tensor ComfyUI uses to re-noise the frozen region each step — so the masked-out area stops
being re-noised from stage 2 on (identical to stock `SamplerCustom` with `add_noise` disabled). For a
masked refine use the **`restart`** handoff, where every stage makes its own noise. Note also that even
an exact split is not *bit*-identical to a single run: the latent round-trips through
`inverse_noise_scaling` / `noise_scaling` between stages, which is mathematically exact but leaves
float32 rounding (≈±1 LSB after VAE decode — PSNR around 50 dB against a single pass).

**Models shipped as a pair, and custom guidance.** Some checkpoints (Ideogram) come as a model **plus
a separate unconditional model** for the negative pass. Wire the second one into **`model_negative`**:
Chimera builds the dual-model guider **per stage**, from that stage's own conditioning and cfg, so
`positive_b` and per-stage cfg keep working. ⚠ The guider **skips the unconditional pass entirely at
cfg 1.0** — the second model then does nothing at all, and the report says so, naming the stages at
fault. For anything else there's **`guider`**, which accepts any `GUIDER`: it carries its own model,
conditioning and cfg, so the node's `positive` / `negative` / `positive_b` and every stage's cfg are
ignored while it's wired (reported), and the sigma curve is built from the guider's own model. It
takes precedence over `model_negative`. **`sigmas`** accepts an external schedule and replaces the one
the node builds — `total_steps` and every stage's scheduler and denoise are then ignored, while all
four `step_split` modes keep working exactly as before, just slicing the curve you supplied.

Optional **per-stage overrides for everything after stage A**: **`model_b`** (a different LoRA stack or
a refiner — it must share the main model's noise schedule, since in `continuous` mode the curve is
built from the main model, the space the in-flight latent's noise level lives in), **`positive_b`** /
**`negative_b`** (polish on a different prompt — e.g. without the style tokens).

**Outputs:** `latent` (the result), **`handoff_latent`** (the raw latent between stages — feed it to
another sampler if you want), **`handoff_denoised`** (the **x0 estimate** at the handoff — VAE-decode
this to actually *see* what stage A produced, since the raw handoff latent is still noisy),
**`report`** (the exact split: curve, per-stage step ranges, sigma boundaries, every warning — also
printed to the console unless `verbose` is off), **`gen_extra_info`**, and **`time`** / **`seconds`**.
Category `Kinburg-Nodes/sampling`.

**`gen_extra_info` (`GEN_INFO`) is an ADDITION to a settings dump, not a dump on its own.** It holds
only what the node resolved at run time — one `[Chimera]` entry (handoff, split, `steps` walked vs
scheduled, the curve and its sigma range, total time) plus a `[Chimera stage]` entry per stage
(sampler, scheduler, its step range, cfg, eta, seed, its sigma boundaries, its own time, and any
`model_b` / `positive_b` overrides) — because the graph-walking **Generation Info** node can only read
widget literals: it can't know the resolved split (`18 / 30`), the sigma the handoff landed on, or
which stage a given `Sampler Settings` fed. Wire it into Generation Info's optional **`extra`** input:

```
Chimera ──latent────→ Generation Info ──data──→ Set Accumulator (gen info) ──→ Get ──→ Filter ──→ Image Compare
        └─gen_extra_info─→ (extra)
```

The entries are merged in front of the walked ones into **one** dump, so the branch stays a single
accumulator entry (one image). They're separate **fields**, not one blob, so the Filter's
`differences` mode shows exactly what changed between the compared runs. Wiring `gen_extra_info`
straight into the accumulator instead leaves that branch without any of the shared settings
(checkpoint, resolution, seed) — and since the Filter counts *present vs absent* as a difference,
`differences` mode then keeps every field the other branches have and nothing collapses.

**It times itself.** `time` (a string like `12.34 s`, ready for Image Compare's `times` input) and
`seconds` are measured **around the sampling calls inside the node**, with a per-stage breakdown and a
`s/step` figure in the report. That matters because `Start Timer` / `Stop Timer` measure the gap
between two *node executions*, and ComfyUI is free to schedule those relative to other branches — put
several sampler branches in one graph and the intervals can overlap, so a 6-step run can come out
"slower" than a 12-step one. Timing from inside can't be skewed that way. It covers **sampling only**
(CLIP encode and VAE decode happen in other nodes) and is a true wall-clock figure — `sample_custom`
moves the result off the GPU before returning, so it waits for the work to finish. One honest caveat:
the **first stage absorbs the model load** when the checkpoint isn't resident yet, because ComfyUI
loads it inside that first sampling call.

### `siren/` — Siren (Music Sampler) 🧜, Siren Section 🧜, Siren Scope 🧜, Siren Compare 🧜

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

#### `Siren Scope (Audio → Image) 🧜`

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

#### `Siren Compare (Audio) 🧜`

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
- **`blind`** hides the labels; the **Measurements / Settings / Notes** tabs share a `differences
  only` switch that collapses each table to the rows that actually differ between takes.
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

### `morpheus/` — Morpheus (Video Sampler) 🌙, Morpheus Dream 🌙, Morpheus Storyboard 🌙

MiniMax H3 generates **5–15 seconds** per run and ComfyUI ships no extend/continue node for it, so the
only route to a minute of video is to run it several times and hand **the last frame of each shot to the
next one as its first keyframe**. That loop is what these two nodes are — named for the god who *shapes*
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

#### The in-loop writer — the forecast, removed

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

#### `Morpheus Storyboard 🌙` — the LLM writes the whole chain

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

### `model_presets/` — Model Capture 📥, Model Select 🎛, Settings Select ⚙, Settings Save 💾

A **model library that lives outside the graph**: pick a model in one dropdown, its known-good
sampler settings in a second, and the workflow holds a single node where it used to hold a loader
stack plus a pile of `Sampler Settings` "for the other model" and switch logic to choose between
them.

The problem it solves is that a model is rarely just a file. `krea2` is a UNET + a CLIP + a VAE;
`z_image_turbo` adds `ModelSamplingAuraFlow` with its own shift; Ideogram adds a *second*
(unconditional) model and a `CFG Override` on top; old checkpoints carry CLIP and VAE inside;
GGUF builds need their own loaders. Swapping between them means rewiring several links and
remembering which settings belonged to which — every time.

**Model Capture 📥** registers an assembly by *reading it out of the graph*. Wire the outputs of a
loader stack you already have working — however many nodes, whatever they are — give it a `model_id`
and run. It walks its own inputs upstream through the queued prompt and stores that slice of graph
as the model's **recipe**. Then those loaders can be deleted from the workflow.

* **It hardcodes no node types.** Classes are looked up in `nodes.NODE_CLASS_MAPPINGS` — the one
  registry that holds V1 and V3, builtin and third-party — so a model that ships tomorrow with its
  own custom loader and its own patch node works with no code change, as long as it was captured.
* **It loads nothing.** The inputs are declared `lazy` and nothing is ever requested, so ComfyUI
  never executes the upstream loaders. Capturing a 40 GB assembly costs no VRAM and no time.
* **`mode` starts at `preview`** — the first run reports what it found and writes nothing, so a
  mis-wired capture can't overwrite a good bundle.
* **It refuses what it can't rebuild**, by node name and reason: anything needing execution context
  (hidden inputs), list in/out, subgraph expansion, or an output node. Those stay in the graph and
  go into Model Select's override inputs instead (below).

**Model Select 🎛** rebuilds the chosen model from its recipe and emits its preset. Outputs:
`model`, `model_negative` (present when the bundle has an unconditional-pass model — wire it into
Chimera's `model_negative`), `clip`, `vae`, `sampler_settings`, `width`, `height`, `info`,
`gen_extra_info` and `model_id`. Because `sampler_settings` is a `SAMPLER_CFG` **chain**, a saved
multi-stage preset drops straight into Chimera's `stage_a` (it flattens chains into stages) or into
Ouroboros.

* A **`🏷 family` filter** sits above the model dropdown and narrows it to one family — with a dozen
  community finetunes per base model, a flat list of everything is unusable. Changing it **clears the
  model and preset**, since switching family means you're on your way to a different model anyway;
  nothing is auto-selected, even when the family holds exactly one model, so loading a model is never
  a side effect of filtering. Only your own click on the filter clears anything — a workflow load or a
  🔄 Refresh always keeps what was selected. It is deliberately *not* a node input: widget values are
  part of ComfyUI's cache signature, so a cosmetic filter as a real widget would invalidate this node
  and everything downstream (i.e. re-sample) on every flip. It lives in `node.properties` instead —
  saved with the workflow, never sent to the backend, zero cache impact.
* The `preset` list narrows to what's valid for the picked model, and picking a model lands on that
  model's **default** preset. Ordering puts the default first, then the best measured score.
* **Only one model is ever loaded**, no matter how many are registered — the others are data, not
  nodes. `unload_others` (on by default) frees ComfyUI's resident models and the library's own
  cache first, keeping the one-heavy-thing-at-a-time discipline the LLM nodes use.
* `seed_override` `-1` keeps the seed the preset was measured with; anything else replaces it on
  every stage. `width` / `height` are fallbacks — a preset saved with a latent carries the real size
  and wins.
* **`model_override` / `clip_override` / `vae_override`** win over the library. That's the escape
  hatch for an assembly Capture refused: keep those loaders in the graph, wire them in, and presets
  keep working. There is no assembly this node can't serve, only ones it can't store.

**Settings Select ⚙** is Model Select minus the loading: pick a preset, get its `sampler_settings`,
`width`, `height`, `label`, `info` and `gen_extra_info` — and no model. That's the investigating
case: **one model, several runs at different settings and seeds**, compared side by side, where two
Model Selects would each want to own the model. `seed_override` is the field to sweep, and `label`
("`anc4+euler4 · seed 999`") is a ready-made caption for Image Compare via a Set/Get Accumulator.

It still has to know whose presets to offer, and the wired way is the good one: connect **Model
Select's `model_id`** into its `model_id` input and the picker follows whatever that node has
selected — so the model is chosen in exactly one place and every Settings Select hanging off it
re-narrows when you switch. Its own `model` dropdown is the fallback when there's no Model Select to
follow; the widget's label shows which model is actually in effect. Unlike Model Select it does *not*
jump to the model's default preset — choosing presets is the point of the node — it only clears a
choice that isn't valid for the current model. If the preset carries bundle `overrides`, `info` says
they don't apply here: overrides retune the model, and this node never builds one.

Settings Select carries the same `🏷 family` filter. **Settings Save** doesn't need one: wire Model
Select's `model_id` into it and the preset is filed under the model that actually ran — no second pick
to keep in sync, and no way to file a preset under the wrong model by mistake.

**Settings Save 💾** puts a settings chain into a model's library — pass-through, so wire it between
`Sampler Settings` and Chimera / Ouroboros and the run is unchanged. It **saves while `preset_name`
is non-empty** (clear the field to stop), the same rule as `save_preset_as` on the card nodes and for
the same reason: the values worth recording arrive over wires, and a frontend button can't see those.
Wire `score` (Ouroboros' `best_score`, or a Vision Judge score via JSON Extract), `seconds`
(Chimera's own measured time) and `latent` (for the size), and the library stops being a notebook —
Model Select shows `★4.40 · 31.2s` and the picker sorts by it.

**Presets can be shared across models.** A model declares `families` (e.g. `flow-1024`); a preset
saved with `shared` + matching `families` shows up for every model in them, so one good generic
recipe isn't copied per model — handy when a base model has a dozen community finetunes that all
want the same settings.

Families are the one place where a typo has **no symptom**: `flow-1204` raises nothing anywhere, the
shared preset simply never appears in any picker. So you shouldn't have to type one. Model Capture
and Settings Save both carry a **`＋ family`** dropdown of families that already exist, which appends
to the (still comma-separated, still multi-value) `families` field; the Library dialog shows every
known family as a **click-to-toggle chip** per model, with a text box only for creating a new one.
And if a family really is new, the report says so and suggests the closest existing name — creating
one stays possible, mistaking one for it doesn't stay silent.

**Editing a bundle without re-capturing it.** The 🗂 Library dialog's **🔧 Recipe** button opens an
editor for the captured assembly's settings — every literal input of every node in it. Controls are
rendered from each node class's *own* schema, fetched from ComfyUI's `/object_info`, so a FLOAT gets
its declared min/max/step and a combo gets its real option list. For a loader that means the **live
list of files on disk**, i.e. you can point a bundle at a different checkpoint from the dialog. A
node type that isn't installed right now is flagged and edited as plain text rather than dropped, and
a stored value that's no longer among a combo's options is highlighted instead of silently replaced.
Wired inputs are shown but not editable: a link is the *shape* of the assembly, and changing that
means re-capturing. Only fields you actually changed are sent.

**Per-preset overrides.** A preset can retune the bundle instead of cloning it — the **🎚** button on
a preset row lists every `class_type.input` in the bundle with its bundle value, and ticking one
stores an override. So "same model, shift 3 vs shift 5" is two presets over one bundle rather than
two near-identical bundles. Keys are class-scoped, not node-scoped, so an override survives
re-capturing the bundle (which renumbers its nodes); an override that matches nothing is reported as
a warning by Model Select rather than silently ignored.

Two details that make it behave: results are cached by a merkle key over the recipe, so **changing a
preset re-runs the cheap patches but never re-reads weights from disk**; and `IS_CHANGED` hashes what
was *resolved*, so editing a recipe or a preset in the library invalidates the node instead of
handing back the previous run's model.

Everything else is managed from the same **🗂 Library** dialog: model families, preset tags,
defaults, rename (presets follow), delete, and a raw view of the stored recipe. Persisted to
`model_presets/data/store.json`; routes under `/kinburg/models/…`. Category `Kinburg-Nodes/model`.

### Token Counter (GGUF)
**`Token Counter (GGUF)`** (in the `local_llm/` package) counts how many tokens a text is under a
model's tokenizer, **without generating** — via the same worker but a **vocab-only** load (just
the tokenizer, no weights, no VRAM; it also reuses an already-loaded model when the path matches,
so counting never disturbs generation). Wire a **`config`** (only its model is used) and a
**`text`**; outputs `token_count`, `char_count`, and an `info` line (e.g. `42 tokens · 180 chars`).
Handy for budgeting a prompt against a model's `n_ctx`. Category `Kinburg-Nodes/LLM`.

### Context Sizer (GGUF)
**`Context Sizer (GGUF)`** (in `local_llm/`) measures how many tokens a request actually needs and
suggests an `n_ctx`. Wire a **`config`**, your prompt text into the auto-growing **`text_*`** inputs,
and optionally an **`image`** (a batch is fine — it sizes for the largest). Counting is **lean** — no
full-model load: text via the vocab-only tokenizer, images via `mtmd_tokenize` on the clip/mmproj
only (no LLM weights, no forward pass), so image tokens — which depend on the model + resolution —
are counted for real (with a safe fallback to a full-model prefill if a build can't tokenize
weight-free). Outputs `text_tokens` / `image_tokens` / `total_tokens` / **`suggested_n_ctx`** (=
input + the config's `max_tokens` + a `margin`, rounded up) / `info`. Sizing `n_ctx` close to what a
request uses saves VRAM (the KV cache shrinks with it). Category `Kinburg-Nodes/LLM`.

All four LLM nodes above (**Local LLM (GGUF)**, **Vision LLM Judge**, **Token Counter**, **Context
Sizer**) carry a per-node **`unload_after_run`** selector — *config default* follows the Settings
node's `unload_llm_after_run`, *unload after run* frees VRAM after just this node (e.g. a different
model runs next), *keep loaded* stays warm — so one node can free VRAM without touching the others
on the same config.

### `grammar_presets/` — Grammar Presets
**`Grammar Presets`** is a dropdown of **GBNF grammars** → one `grammar` STRING output. It ships
with templates that force an LLM's output into a fixed shape — **Character Card (JSON)** and
**Entity Card (JSON)** — and you can add / edit / delete your own (**➕ Add grammar**, persisted on
disk). Wire the `grammar` output into a **Local LLM (GGUF)** node's `grammar_override` input, feed
that node a **photo** + a short prompt ("fill the card from this image"), and the vision model
returns exactly that structure straight from the picture — no multi-image-context gymnastics.
Pair it with **Card Save** (below) to drop that generated card straight into your library.
Category `Kinburg-Nodes/LLM`.

### `card_presets/` — Card Save, Card Presets
**`Card Save`** closes the loop Grammar Presets opens: wire an LLM's JSON card output (constrained
by a Grammar Presets grammar) into its `json_string` input and the parsed character / entity lands
in the **Card Presets** library — no JSON Extract → 12-wire dance into a Character Card (the
grammar's keys already mirror the card fields 1:1). `card_type` is `auto` (detects character vs
entity from the keys) / `character` / `entity`; **`save_as`** names the preset (empty → uses the
JSON's own `name`; since a photo often yields an empty name, `save_as` also becomes the card's
heading); **`tags`** are comma-separated labels for filtering the library. It outputs the rendered
`card` block (feed Context Collector in the **same** run), `saved_as`, and a `report` line — and
never breaks the graph on bad / empty JSON.

**`Card Presets`** is the reader: pick a character / entity from a dropdown and its ready Markdown
block comes out the `card` output (feed it into Context Collector). The optional **`filter`**
dropdown narrows the list to one tag. Build the library with **Card Save** (photo → card) or the
**Character Card** / **Entity Card** nodes' `save_preset_as` (+ `tags`) field (see above); presets
are rendered back through the card nodes' own logic — so the format always matches — and persisted
on disk. **🗑 Manage** edits tags / deletes entries, **🔄 Refresh** re-reads the list. Build a
character once (by hand or by photo), then reuse it from the dropdown instead of re-describing the
same photo every time. Category `Kinburg-Nodes/LLM`.

### `gguf_convert/` — Safetensors → GGUF converters
Turn `.safetensors` weights into `.gguf` from inside ComfyUI. Two nodes (category
`Kinburg-Nodes/GGUF`), one per model family, each with the same three outputs — **`gguf_path`**
(the finished file, ready to wire into a loader), a **`log`** tail, and a **`help`**
cheat-sheet. Conversion runs in ComfyUI's own Python; both stream progress to the console and
the `log` output, and `force = off` returns an already-built output instead of redoing the work.

**`Safetensors → GGUF (llama.cpp)`** converts a **language / multimodal LLM** with llama.cpp's
`convert_hf_to_gguf.py`. **`source`** takes a HuggingFace repo id (`Qwen/Qwen2.5-0.5B-Instruct`),
a `https://huggingface.co/owner/name` URL (downloaded for you via `huggingface_hub` — set
`hf_token` for gated repos), a **local HF model folder** (`config.json` + tokenizer +
`*.safetensors`), or a single `*.safetensors` file (its folder is used). `outtype` is the
precision written (`f16` is the usual base), and an optional `quantize` pass shrinks it to a
K-/I-quant (`Q4_K_M`, …) with **`llama-quantize`** — a compiled binary you provide (auto-listed
from `ComfyUI/models/llm`, or set `quantize_binary_path`); `quantize = none` needs no binary at
all. The llama.cpp scripts themselves are found via `llama_cpp_dir`, or **auto-cloned** into
`ComfyUI/models/llm/llama.cpp` when `auto_clone` is on. The output drops straight into the Local
LLM (GGUF) nodes. LLMs only — it can't read diffusion weights.

**`Diffusion Safetensors → GGUF (city96)`** is the diffusion counterpart (Flux, SD3, SDXL, SD1,
Aura, HiDream, Cosmos, LTXV, HunyuanVideo, Wan, Lumina2). It drives city96's **ComfyUI-GGUF**
`tools/convert.py`: pick the diffusion checkpoint from `ComfyUI/models/diffusion_models` or
`/unet` in the **`model`** dropdown, or point `model_path` at a local `.safetensors`, a HF file
URL (`…/resolve/main/model.safetensors`), or an `owner/name::file.safetensors` spec. Step 1
always writes an F16/BF16 gguf; an optional `quantize` pass (`Q4_K_S`, `Q8_0`, …) needs the
**patched** `llama-quantize` from city96's llama.cpp fork (the stock one won't handle diffusion
tensor shapes — build it per ComfyUI-GGUF/tools/README.md and set `quantize_binary_path`). For
Wan 2.1 / HunyuanVideo, **`fix_5d_tensors = auto`** reads the model's architecture from the gguf
and runs the required 5-D-tensor fix pass after quantization. The `tools/convert.py` is located
via `tool_dir`, or from an installed `custom_nodes/ComfyUI-GGUF` (auto-cloned there when
`auto_clone` is on — you'll want that node installed anyway, since its **Unet Loader (GGUF)** is
what loads the result). Diffusion models only — for LLMs use the node above.

### `image_compare/` — Image Compare (HTML)
Takes a batch **or an image list** (different sizes are fine — e.g. from Get Accumulator (images
list)) + short labels (`captions`) + full generation prompts (`prompts`)
and produces an interactive HTML comparison page — grid (columns + max row height),
before/after slider, opacity overlay, A/B flip, pixel difference, **synced zoom &amp; pan**, synced loupe, lightbox,
drag-to-reorder, and per-result **review** controls — a **hide**/reject button (with a
toolbar toggle to show/hide hidden results), a **star rating** (1–5), **tags**, and a
**comment** box — plus a "Save page" button and an `images_captioned` output (a *list*, so
mixed-size inputs stay separate) with the captions drawn on each image. The review state (hide / rating / tags / comment) persists across reloads of the served
page (browser localStorage). Save
options: `output_dir` (any folder; the page is served straight from there, no copies),
`save_captioned_images`, `save_prompts_txt`. Only `images` is required; the text inputs are
all sockets (no inline fields): `captions` (one per line, e.g. from **Get Accumulator
(captions)**), `prompts` (full per-image blocks separated by a `---` line, e.g. from **Get
Accumulator (prompts)**), and `settings_data` (structured per-image settings from
**Generation Info Filter**) — rendered under each image as one `[Class] param: value` line
per field, with its own page toggle. The `url` output is a clickable http
link. Node: **`Image Compare (HTML)`** (category `Kinburg-Nodes/image/compare`).

The node carries no *view* settings: what's visible — captions, prompts, settings, times, metrics,
judge, hidden results — is toggled **on the page**, where you can see the effect. It does carry the
accumulator **🔌 Collect All** button and its **`auto_collect`** toggle (see the accumulators section),
since gathering the branches is what feeds this node in the first place.

**The page header is three fixed rows** instead of a toolbar that wrapped into five, and it's grouped
by purpose rather than by accident of order:

- the **title** on its own thin line, never shortened;
- the **tools** line, read left to right: the five **modes** pinned left (also keys **1…5**), the
  viewing controls centred on the page — **👁 Panels**, **Zoom**, **⤢ Fit**, **🔍 Loupe** — and
  **⬇ Save page** / **📊 Report** / **❔ Help** pinned right;
- a **mode line** that always belongs to what you're doing: **Columns / Rows / Sort** in Grid, and the
  **A** / **B** selectors in the pair modes, where they meet at the centre of the page — the same seam
  the wipe divider sits on — so each selector is on the side of the image it controls and long captions
  grow **outward** instead of being cropped (the full label is in the option list and the tooltip).

The two panels you touch least often stand behind one button each: **👁 Panels** (the six visibility
toggles + hidden results, with a badge counting them) and **📊 Report** (the DB path, *Save run*,
*Open report browser*). The **Opacity** slider isn't in the header at all — it appears right above the
image it fades, as `A ──○── B` with a live percentage. The Kinburg-Nodes credit sits in a thin footer.
Result: **146 px → 109 px** of header at 1266 px wide (11.5% → 8.6% of the window), with a layout that
no longer reshuffles itself as the window changes.

**Rows on screen, not pixels.** Grid's height control is **Rows** — how many rows of results you want
to see at once (default **1**) — instead of a **Max h** in pixels you had to guess and re-guess. One row
fills the window, scrolling **snaps row by row**, and the page becomes a flipper: one comparison at a
time, always the same size. **Max h** is still there for when you want a specific scale (set **Rows** to
**0**), and while Rows is on it just reports the height that was worked out.

The mechanism is deliberately dumb, because the obvious implementation isn't: rather than measure the
caption/prompt/settings/review chrome and subtract it — a two-pass layout that has to re-run every time
you toggle a panel, change **Columns**, type in a comment or load a lazy image — a row is simply given a
**fixed height of one N-th of the visible area**, and each card's own flex layout hands the picture
whatever its panels leave over. So a card with a long prompt shows a slightly smaller image than its
neighbour, with no arithmetic anywhere; when the panels would crowd the picture out altogether **they**
scroll inside the card instead, and the image never drops below a third of it. Exactly **one** number is
ever measured — the height the stage can show — and only when the window (or the header's own wrapping)
actually changes it. The page is an app shell now: fixed header and footer, one scrolling stage, which
is also what lets the pair modes fill the window instead of the old magic `78vh`.

**Synced zoom &amp; pan.** The **Zoom** slider (×1–×8) magnifies **every** image at once, and they all
show the **same region**: drag any one of them and the visible area moves in all of them together.
**Ctrl + wheel** over an image zooms **at the cursor** (the bit you point at stays put); **double-click**,
**0**, or **⤢ Fit** returns to fit. Nothing is re-laid-out — each image is scaled by a CSS transform
inside a clipping frame, so the grid never moves and only the visible region does. The region is held
in *picture-relative* coordinates, so it survives switching modes and changing **Columns** / **Rows**,
and lands on the same content even when the compared images differ in size or aspect. Made for judging
fine detail — texture, small faces, foliage — without hunting for the same spot in each picture.

The **Loupe stacks on top of it**: the lens shows **zoom × lens** while its surroundings stay at the
current zoom, so you can park every image on a region and then peek closer still (plain **wheel** =
lens zoom, **Alt/Shift + wheel** = lens diameter, **Ctrl + wheel** = image zoom, dragging still pans).
Reordering goes through a **⠿ grip** in each card's top-left corner rather than the card itself, so all
of a card's text stays selectable and reordering keeps working at any zoom. In **Slider** mode the
divider's round handle always moves the wipe — which is what makes it reachable when the pointer is
busy panning or driving the lens — and **the wipe stays where you put it**: switching A/B, toggling a
panel, hiding a result or stepping through other modes no longer snaps it back to the middle.

The **🔗 Open comparison** link is stored on the node, so it survives tab switches and workflow
reloads. Three things keep it honest. The token → folder map behind the served URL is **persisted to
disk** (`<output>/kinburg/compare_dirs.json`, last 300 runs), so a link minted before a ComfyUI
restart still opens instead of 404-ing. The link is refreshed from the API's `executed` event rather
than from the `onExecuted` prototype chain, which any other installed extension can break by patching
it without calling through. And a link that was **restored** rather than produced by a run you just
did says so — `🔗 Open comparison (previous run)` — and is quietly checked against the server; if that
run's folder is gone it turns into a dull `⚠ Comparison expired — run to rebuild` and stops being
clickable, instead of opening a blank 404 tab.

Two more controls. **`embed_images`** chooses how the comparison is saved: **off** (default) writes
a **portable folder** `<prefix>_<datetime>/` — a light `index.html` + an `images/` subfolder with
relative links — so you can open it offline, zip and share it, or open it in-app; **on** writes a
single self-contained `.html` with every image inlined as base64 (one bigger file). Both open from
the node's `url` output. And an optional **`reference`** image (or
**`reference_index`**, a 0-based index into the batch; `-1` = off) enables **similarity
metrics** — **SSIM** and **PSNR** of every image vs the reference — shown under each image (a
**Metrics** page toggle) with a **Similarity** sort in the grid. Great for upscale / img2img /
restoration, or for checking how far a quantized model's output drifts from its fp16 baseline.
Note these measure *closeness to the reference*, not aesthetic quality (that's the Vision LLM
Judge's job).

An optional **`judge_data`** input takes the **Vision LLM Judge**'s `results_json` and renders
each image's AI verdict — stars (proportional to its score), tags and a comment — as a
**read-only** section under it, with its own **🤖 Judge** page toggle, kept separate from your
own hide / rating / tags / comment review controls.

Also in this package: **`Color Caption`** — write a caption and type two colors, the
**text** color and the **band** color behind it (each as a HEX `#RRGGBB` value). It outputs a one-line JSON
`{"caption": "...", "color": "#RRGGBB", "band_color": "#RRGGBB"}`. Wire it into the compare
node's `captions` input (one caption per line) to style that label — both on the page and
on the drawn `images_captioned`. The defaults (white text on a black band) reproduce the
classic look. The `captions` input still accepts plain text lines exactly as before; each
line is treated independently, so you can mix plain and styled captions. Category
`Kinburg-Nodes/image/compare`.

### `image_batch/` — Unlim Image Batch
**`Unlim Image Batch`** concatenates an unlimited number of IMAGE inputs into a single
batch. The input list grows on its own (like Unlim Text Concat): `image_1` (required) +
`image_2`, and a new empty slot appears whenever you connect the last one. A single batch
tensor needs every frame at the same size, so when inputs differ `mode` reconciles them
**without resampling** (no quality loss): `as is` stacks pixels untouched and errors on a
size mismatch; `crop to smallest` center-crops every input down to the smallest size;
`pad to largest` center-pads every input up to the largest, filling the borders with
`pad_color` (HEX). Mismatched channel counts (RGB vs RGBA) are padded with opaque alpha so
everything stacks. Each input may itself be a batch. `skip_empty` (on by default) drops
empty / unconnected inputs so bypassing a branch doesn't break the batch. Category
`Kinburg-Nodes/image`.

**`Unlim Image List`** is the sibling for when the sizes *shouldn't* be reconciled: it
returns a ComfyUI **image list** (`OUTPUT_IS_LIST`) instead of a stacked tensor, so frames
of different sizes can travel together. The growing inputs work the same way; there are no
options. Every input is split into single frames, in slot order, so the list length is the
total image count and each index is exactly one image — convenient for loop / iterator
nodes (read the length, take an item by index, process it inside the loop). Note that
downstream nodes then run once per item rather than on a single batch. Category
`Kinburg-Nodes/image`.

### `collage/` — Collage
**`Collage`** arranges images into a grid on a single output canvas (e.g. an A4-ish
2480×3508). Source is the wired `input_images` batch, or — if nothing is connected —
every image in `folder_path` (natural-sorted, so `img2` < `img10`). `cols` sets the column
count and the rows follow from the image count; `gap` spaces the cells and `margin` frames
the grid, with the cell size derived so the whole grid fits the canvas. Each image is fit
into its cell with its aspect ratio preserved; the letterbox border is filled with the
image's own top-left pixel color so it blends in. The background is `background_color`
(HEX), or a connected `background_image` (stretched to the output size, first frame).
Category `Kinburg-Nodes/image`.

### `loops/` — flexible iteration
ComfyUI's execution graph is acyclic, so these cover the two practical loop shapes, both with
**auto-growing wildcard `*` state slots** (the node shows only the connected slots plus one
spare; drag in any type, a matching slot appears — on inputs *and* outputs):

**`For Each (Open)` / `For Each (Collect)`** — iterate a batch/list **one element at a time**,
on the same graph-expansion engine as Repeat. Feed Open any iterables (image/mask batches,
LATENT batches, lists…); each iteration it emits a single **`element_*`** (the idx-th item of
each input) plus `index` / `total`, iterating to the **shortest** input. Build your body off
the `element_*` outputs (e.g. a collage from `element_0` + `element_1`) and wire the result(s)
into Collect's **`result_*`** inputs; Collect accumulates each iteration's results and, when the
loop ends, emits each `collected_*` as the gathered Python list. Feeding that into a **`List
Output`** node turns it into a real per-item ComfyUI list — so different-sized images travel
separately (a Preview shows N images, one per item; a Save writes each). The **🔗 Add / link
Close** button on Open wires the whole chain: it creates Collect (linked by `flow`) plus a List
Output on `collected_0`. (A list output can't be emitted from inside the loop itself — ComfyUI
flattens list outputs during graph expansion — so the fan-out lives in this node just past it.)
Category `Kinburg-Nodes/loops`.

**`List Output`** fans a value that holds a Python list into a proper ComfyUI list (one item per
element). For Each (Collect) pairs with it, but it's a handy standalone converter too.

**`Repeat (Open)` / `Repeat (Close)`** — a real iterative loop with **carried state**, built on
ComfyUI's graph expansion: each pass, Close clones the loop body wired back into the next
iteration, all inside one queue run. Wire your starting values into `Open`, read `index` inside
the body, and feed the updated values into `Close`; after `count` iterations `Close` outputs the
final state. The **🔗 Add / link Close** button on `Open` spawns the matching Close and wires the
`flow` (and `index`) links for you, so you never hand-draw the feedback. `count` must stay a
widget value (it's read when the loop expands). Category `Kinburg-Nodes/loops`.

**`While (Open)` / `While (Close)`** — the same engine, condition-driven instead of counted.
Close has a `condition` (BOOLEAN) input computed in the loop body: the loop keeps going while
it's True and stops the moment it's False, with `max_iterations` on Open as a hard safety cap.
Same wildcard state slots and 🔗 auto-pairing as Repeat. Category `Kinburg-Nodes/loops`.

**`Get by Index`** takes the index-th element of anything indexable — an IMAGE/MASK batch (→ a
1-frame batch), a LATENT batch (→ one latent, other keys preserved), a list, a string, or any
other tensor — so inside a loop you can feed a whole batch in and pull out the current frame by
`index`. Negative indices count from the end; `out_of_range` is `clamp` / `wrap` (cycle) /
`error`. It also outputs `length` (the container size), handy for driving a loop's `count`.

**`Delay`** passes any value straight through after pausing `seconds` (with an optional console
`label`). Drop it into a wire inside a loop body to slow iterations down and watch the loop run.
Category `Kinburg-Nodes/loops`.

### `util/` — Date String, Unlim Text Concat, Color Picker, Any/Combo to String, Text Transform, Any Switch, JSON Extract
**`Date String`** appends the current **date** (and optionally **time**) to a string, with
selectable formats. Handy for building save paths: e.g. `project/2026-06-20` (a folder per
day) or `.../2026-06-20/17-05` (per minute). The `/` separator creates subfolders; `_`/`-`
make a flat name. Presets plus a `custom` strftime field. The node is re-evaluated on every
run so the date never freezes from caching. Category `Kinburg-Nodes/util`.

**`Unlim Text Concat`** joins an unlimited number of string inputs with a `separator`
(multi-line allowed; default is a newline). The input list grows on its own — it starts
with `text_1` (required) and `text_2`, and each time you connect the last slot a new empty
one appears (disconnect and trailing empties collapse). `skip_empty` drops empty/unconnected
inputs so they don't leave stray separators. Pairs naturally with **Color Caption** →
Concat (newline) → the compare node's `captions`. Category `Kinburg-Nodes/util`.

**`Color Picker`** is the handy color control from Color Caption (a 10-swatch palette + a
native color picker, or type a HEX) as a standalone node. Outputs the normalized `#RRGGBB`
string plus its `R` / `G` / `B` components. Category `Kinburg-Nodes/util`.

**`Any to String`** / **`Combo to String`** convert a value into a real `STRING` you can feed
into text/preview/prompt nodes. **`Any to String`** has a wildcard `*` input for values that
already connect loosely. **`Combo to String`** solves the specific case ComfyUI blocks by
design: a **COMBO** output (e.g. the core `Primitive` node in combo mode) refuses to wire into
a `STRING` input. Connect the Primitive's COMBO output into `Combo to String` (a small frontend
patch in `web/combo_to_string.js` allows the link and pushes the selected value in), and its
`string` output carries the value — e.g. `ru` — wherever you need it. Category
`Kinburg-Nodes/util`.

**`Text Transform`** does string find/replace, regex, trim and case in one node —
`replace`, `regex_replace`, `regex_extract`, `regex_findall`, `strip`, `collapse_whitespace`,
`lower`/`upper`/`title`, with `ignorecase` / `multiline` / `dotall` flags. The regex is
**validated**: a bad pattern (or replacement backreference) is reported on the **`error`** output
while the input text passes through unchanged, so a typo never crashes the run. Outputs `text` /
`count` / `error`. Category `Kinburg-Nodes/util`.

**`Any Switch`** forwards one of several inputs, chosen by `select` (1-based). The `input_*`
slots take any type and auto-grow (the node shows the connected ones plus a spare); outputs the
selected `output` and its `selected` slot number. It routes a value — ComfyUI still computes the
other branches (it's not a lazy gate). Category `Kinburg-Nodes/util`.

**`JSON Extract`** pulls fields out of a JSON string by path into separate `STRING` outputs. You
write **one path per line** in the `paths` field (each line either a bare path or `path -> alias`;
blank lines and `# comments` are ignored) and the node grows one **auto-labelled `value_*` output
per line** — named by the alias, or the path's last key — so the graph is self-documenting; no
fixed slot count and no more guessing which `value_3` is which. Outputs are rebuilt **when you
click away from the field**, not while you type, so they appear complete and ready to wire. Don't want to type paths? Hit
**🔍 Explore JSON** to paste a sample (or reuse the node's last run), then click ＋ on any field to
add its path — arrays also offer a `[*]` button that grabs every element. A **live preview** on the
node shows the extracted values after each run. Outputs are `found` (True when the JSON parsed and
every non-empty path resolved) and `report` (per-path hit/miss) first, then the labelled values.
Path syntax: dot keys, `[n]` / `.n` indices (negative allowed), **`[*]` / `*` wildcard** (grabs
every element/value and joins them with `array_join`, e.g. `elements[*].desc`), `$` (or empty) for
the whole document; an object/array value comes out as compact JSON, and prose-wrapped JSON is
tolerated (the first `{…}`/`[…]` is parsed). Missing paths return `default`. Up to 12 paths map to
outputs (extras are noted in `report`). Pairs with the structured-output LLM nodes (e.g.
`ideogram4_json`) to route sub-fields into different prompt inputs. Category `Kinburg-Nodes/util`.

### `timer/` — Start Timer, Stop Timer
**`Start Timer`** / **`Stop Timer`** measure the wall-clock time of a slice of a workflow.
Any value (MODEL, LATENT, IMAGE, …) passes through unchanged — wire Start at the beginning of
the slice and Stop at the end, and that data dependency forces ComfyUI to run Start → slice →
Stop in order. **Start Timer has unlimited inputs** (`any_1` passes through; the rest are
dependency taps that grow as you connect): it starts only after *all* connected inputs are
ready, so tap every branch feeding your sampler (noise / guider / sampler / sigmas / latent)
to start the clock right before the sampler runs, not as soon as one branch resolves. Start also outputs the start time as epoch
seconds (feed it into Stop) and as a formatted string; Stop outputs the `elapsed` time
(format: `auto` / `seconds` / `milliseconds` / `HH:MM:SS` / `human`) and the raw seconds —
wire `elapsed` to any text preview to see it. Both nodes always re-execute (so the timing is
real), which means the wrapped slice is recomputed every run while the timers are active —
mute/bypass them when you're not measuring. Category `Kinburg-Nodes/util`.

### `gen_info/` — Generation Info, Generation Info Filter
**`Generation Info`** lists the settings of the branch that produced an output. Pass your
**`LATENT`** through its `passthrough` slot (tap it downstream of where your branches
converge — e.g. the sampler's latent output — so the upstream walk reaches them all);
the node reads ComfyUI's hidden `PROMPT`, walks upstream, and lists the upstream nodes'
widget values (`[RandomNoise] noise_seed: …`, `[KSamplerSelect] sampler_name: …`, etc.). The
dump shows on the node, collapsed by default — click to expand. Outputs a human dump (`info`)
and machine-readable `data` (GEN_INFO). Category `Kinburg-Nodes/util`.
<br>An optional **`extra`** input takes `GEN_INFO` from a node that reports its **own runtime facts**
— e.g. **Chimera**'s `gen_extra_info`, which knows the resolved step split, the sigma the handoff
landed on and the per-stage times, none of which a walk over widget literals can see. It is merged
**in front of** the walked entries into a **single** dump (per-class `ord`s are renumbered so every
entry stays addressable), so the branch still feeds one `Set Accumulator (gen info)` — one image
downstream, not a phantom extra one. Such an input is an *addition* to a dump: send it to the
accumulator directly and that branch carries none of the shared settings, which makes the Filter's
`differences` mode keep everything (a field present in some branches and absent in others counts as
a difference).

**`Generation Info Filter`** takes a single `data` bundle (`GEN_INFO_LIST`) from a
**Get Accumulator (gen info)** — which collects one Generation Info `data` per branch — and
emits a per-image `settings` string (one block per dump, in accumulator `index` order).
`mode`: `all`, `differences` (only the
fields that vary across the inputs — each block then shows them all, so the images line
up), `custom` (fields named in `custom_fields`, one selector per line: `ClassType`,
`ClassType[n]`, `ClassType[n].param`, `ClassType.param`; `[n]` is the 1-based occurrence),
or `differences + custom`. A `help` output prints the selector syntax. `skip_empty` (on by
default) drops empty dumps so a bypassed branch keeps the blocks aligned.
Outputs: `settings_data`
(`GEN_SETTINGS`, structured per-image `{key, value}` with class-qualified keys) — wire it into
the compare node's **`settings_data`** input, which both renders the settings under each image
(one `[Class] param: value` line per field, toggleable) and stores them by field in a saved
report for filtering/sorting. A plain-text `settings` output is also provided for standalone
use (saving to a file, feeding a text node, etc.). Category `Kinburg-Nodes/util`.

### `report/` — investigate report DB (work in progress)
The Image Compare node carries a `report_db` input (default `<output>/kinburg/reports.db`,
editable on the page) and saves clean per-image PNGs to a run-scoped folder. The served
comparison page has a **"Save run to report"** button that POSTs the run — images, caption,
prompt, settings, and your per-result verdict/rating/comment — to a local **SQLite** DB
(`/kinburg/report/save`). Settings are stored both as text and expanded into a key/value
table so the report browser can filter/sort by any setting field, including ones that didn't
exist before. Re-saving the same run updates it in place (no duplicates). A **📊 Report**
button (also at `/kinburg/report`) opens a browser page over the whole DB — a
sortable/filterable table (thumbnail, run, caption, status, ★ rating, tags, settings, prompt,
comment) with free-text search and a by-setting-field filter; rows link back to their
comparison. You can **edit in place** (toggle status, set rating, add/remove tags, edit the
comment — saved straight to the DB), **export** the filtered view to CSV / Markdown / HTML,
and **delete a run** (removing its rows and image files).

### `lora/` — Lora Trigger Loader, Lora Unlim Accumulator
Stack any number of LoRAs (with their trigger words) onto a model in one node. **`Lora Trigger
Loader`** is pure config: a searchable LoRA dropdown, `strength_model` + `strength_clip`, and an
optional `trigger` word; its single `lora` output (type `KINBURG_LORA`) carries that spec —
nothing is loaded yet. **`Lora Unlim Accumulator`** takes a `model` (+ optional `clip`) and a
`prompt` (input only — wire it in), plus an auto-growing list of `lora_*` inputs fed by the
loaders (the `clip` input sits above the first LoRA). It loads and applies each LoRA in slot
order — to the model with `strength_model`, and to CLIP with `strength_clip` when a CLIP is
connected (otherwise model-only) — appends the non-empty trigger words to the prompt (in their
own paragraph after a blank line, comma-separated among themselves), and outputs the patched
`model` / `clip` / `prompt` plus a **`triggers`** output (just the comma-separated trigger words,
no prompt — wire it into **Ouroboros**'s `trigger_words` so the triggers survive the LLM prompt
rewrite). A LoRA with no effective strength (off) is skipped entirely — neither applied nor does
its trigger word get added. Loaded files are cached per run. Category `Kinburg-Nodes/lora`.

### `accumulators/` — Set / Get Results (name-based accumulators)
For collecting parallel branches without manual batch wiring. **`Set Results (image)`** is a
labelled pass-through: connect a flow's final image and give it an accumulator `name` (e.g.
`IMG_RESULTS`); copying the flow auto-increments its `index`. **`Get Results (image)`** has a
**dropdown** of the defined accumulator names — pick one (it auto-collects) — plus a
**Collect** button that physically wires every matching Set's output into it (real links, in
`index` order) and batches them (reusing Unlim Image Batch: `mode` /
`pad_color` / `skip_empty`). Plug `Get Results` where an image batch used to go (e.g. Image
Compare). A sibling **`Get Accumulator (images list)`** collects from the *same* Set nodes but
returns a **list** instead of a batch, so accumulated images of different sizes coexist (feed it
straight into Image Compare, which now takes a batch or a list). Press Collect again after adding/removing Set nodes — it rebuilds the links from
scratch and **only wires active Sets** (a Set in Bypass or Mute is skipped, so it drops out on
re-collect). A **text pair** —
**`Set Results (text)`** / **`Get Results (text)`** — works the same way and joins the
collected texts with a `separator` (reusing Unlim Text Concat) in index order. Two
compare-tuned twins of the text pair drop the separator field entirely and hardcode the
separator Image Compare expects, so the two ends can't mismatch: **`Set/Get Accumulator
(prompts)`** joins blocks with a `---` line (feed `Get` into the compare node's `prompts`),
and **`Set/Get Accumulator (captions)`** joins with a newline, one caption per line (feed into
`captions`). A **gen-info pair** — **`Set Accumulator (gen info)`** / **`Get Accumulator (gen info)`** — works the same
way for the `data` (GEN_INFO) of **Generation Info** nodes: Get collects every matching Set's
dump (in index order) and outputs a single `data` bundle (`GEN_INFO_LIST`) — wire that one
output straight into **Generation Info Filter**, so its settings no longer need wiring branch
by branch. The wiring is plain links, so execution and caching are completely standard.

**Collect All lives on `Image Compare (HTML)`** — collecting only ever serves a comparison, so the
button sits on the node that consumes the accumulators rather than on a helper node of its own. Its
**🔌 Collect All** button (re)wires *every* Get Accumulator in the graph at once (the label then shows
how many were wired), so you don't have to visit each one after scaling the workflow. Like the
per-node Collect it rebuilds links from scratch and wires only **active** Sets — bypass or mute a Set
and re-collect to drop it from every accumulator. The compare node's **`auto_collect`** toggle (on by
default) does it automatically right before the workflow is queued, so a run always reflects the
current, non-bypassed Sets; turn it off to collect only on the button press. *(This replaces the old
standalone `Collect All Accumulators` node — delete it from older workflows, where it will show up as
missing.)*

### `prompt_presets/` — Prompt Presets
**`Prompt Presets`** is five **flexible preset slots**, each emitting a `STRING` prompt fragment
(five outputs). Each slot has **two dropdowns** — a **category** selector and a **preset**
selector (the presets of the chosen category) — so any slot can draw from any category. Out of
the box the five slots default to the classic **Camera / Aesthetics / Light / Medium /
Background** categories (and the output label follows each slot's chosen category), but you can
point any slot at any other category, including your own. Every category ships with curated
built-in presets (e.g. *Camera → Cinematic Anamorphic*, *Light → Rembrandt*, *Aesthetics →
Cyberpunk*) plus a `🚫 None` option that resolves to an empty string. Selecting a preset resolves
to its fragment at run time; wire the outputs into your prompt builder / text concat.

You can **add your own presets** (**➕ Add preset** — pick a category, name it, type the fragment;
re-using an existing name edits it) and **manage categories** (**🗂 Categories** — add / rename /
delete your own categories; the five built-in categories are protected). **Save the current
slots as a setup** (**💾 Save setup** — names the whole combination of all five slots'
category+preset); the **⚙ setup** selector at the top re-applies any saved setup in one click,
and **🗑 Manage** lists your setups and custom presets for deletion (built-ins can't be removed).
Custom presets, categories and setups are persisted on disk (`prompt_presets/data/store.json`,
git-ignored) via `PromptServer` routes under `/kinburg/presets`, so they survive restarts and
appear across all `Prompt Presets` nodes without an object-info reload. Category
`Kinburg-Nodes/prompt`.

### `prompt_variations/` — Prompt Variations
**`Prompt Variations`** expands one template into many prompts (a per-item `STRING` **list**).
Write choices with `{a|b|c}` (nesting allowed, e.g. `{a {x|y}|b}`) and optional `__wildcard__`
refs (one option per line from `<wildcards_dir>/name.txt`, default `ComfyUI/wildcards`), and the
node emits the **cartesian product** — empty options like `{, dramatic|}` mean "nothing" (dangling
commas are cleaned up). `mode` = `all` (every combination, capped by `limit`) or `random`
(`limit` random combinations, reproducible via `seed`); `dedupe` drops duplicates. Outputs
**`prompts`** (a list — feed it into **For Each (Open)**), **`count`**, and a **`preview`**. It's
the input generator for a prompt-space sweep: Prompt Variations → For Each → your sampler → a
Set/Get Accumulator → **Image Compare**, to render and compare every variant at once. Category
`Kinburg-Nodes/prompt`.

### `list_ops/` — Insert / Remove for batches and lists
Edit collections by position without rewiring. Two families:

**Image batch** (a single `[B,H,W,C]` IMAGE tensor, frame-wise): **`Image Batch Insert`** puts an
`image` (itself possibly a batch) into `batch` at a chosen spot — `position` = *at start / at end
/ at index / after index* with a 0-based `index` (negative counts from the end, `-1` = last).
Frames of a different size are reconciled losslessly like Unlim Image Batch (`mode` /
`pad_color`). **`Image Batch Remove`** drops `count` frame(s) starting at `index` and returns both
the `batch` remainder and the `removed` frames (`index` may be negative). Category
`Kinburg-Nodes/image`.

**Generic list** (a ComfyUI list of ANY type — images, strings, latents, ints…): **`List
Insert`** inserts the `item` input into `list` at the chosen `position` / `index` (feed `item` a
multi-item list to insert several at once); **`List Remove`** drops `count` item(s) at `index`
and returns the `list` remainder plus the `removed` items. An item is one list element (a single
value counts as a one-item list). Use these when items aren't same-size images — for frame-level
edits of a same-size IMAGE batch, use the Image Batch nodes above. Category `Kinburg-Nodes/list`.

### `show_text/` — Show Text (Markdown)
**`Show Text (Markdown)`** displays whatever you wire into it as text. Its `value` input is the
wildcard `*` type, so **anything** connects — a STRING, a number, a COMBO, even a dict/list
(rendered as pretty JSON) — and the node converts it to text (a whole batch/list is gathered
into one view). A **markdown** toggle flips between a rendered markdown preview (headings,
**bold**/*italic*, `code`, code blocks, lists, links, quotes, `---`) and an **editable raw
textarea** you can tweak before saving — inside a fixed-size scroll box, so toggling never
resizes the node. HTML in the text is escaped (safe to display).

Unlike the core **Preview Text** node, the shown text is stored **in the workflow** (the node's
`properties`), so it **survives switching between workflow tabs** in the desktop app instead of
resetting. **💾 Save .md** writes the current text to disk via a `PromptServer` route
(`/kinburg/showtext/save`): relative `save_path`s land under ComfyUI's `output` folder, a `.md`
extension is added automatically, parent folders are created, and `{date}` / `{time}` /
`{datetime}` placeholders expand to the current date/time. **`autosave`** does the same
automatically on every run when a path is set. **📋 Copy** copies the text to the clipboard, and
a small header shows a char/line counter. The converted text is also a **`text` (STRING) output**,
so the node can sit inline in a wire and pass it downstream.

**Freeze / use the saved text (`use_saved_text`).** By default the node reads its `value` input
(running its upstream) and the shown text tracks it. Flip **`use_saved_text`** on to output the
text saved & edited *in the node* instead — and the input is **not evaluated**, so its upstream
never runs. Typical use: an LLM generates a prompt → Show Text → image generation; you eyeball
the prompt, tweak it in the node, flip the toggle on, and re-queue — the **edited** prompt goes
downstream and the **LLM doesn't re-run**. This uses ComfyUI's *lazy evaluation*
(`check_lazy_status`), not just output caching, so it holds even after a ComfyUI restart or with
a randomized seed. The edited text rides in a hidden, serialized `saved_text` widget (so it
reaches the backend and is stored in the workflow); edit it and re-queue to push changes without
regenerating. Leave the toggle off for the original behavior. Category `Kinburg-Nodes/util`.

### Chat — `Local LLM Chat (GGUF)` + `Local LLM Settings (GGUF)`
**`Local LLM Chat (GGUF)`** (in the `local_llm/` package) is a self-contained multi-turn chat node.
It's deliberately bare — just the **chat window** (User/LLM bubbles in a fixed-height, scrollable
box, so the node doesn't grow as the chat fills), the message field, a **context meter**, an
optional row of **persona chips**, and the **Send / Approve / Clear** buttons. Everything about
*how* to generate comes through the **`persona_1`** input.

**`Local LLM Settings (GGUF)`** is that config node: it holds the options — model, system prompt,
sampling, loader (`n_ctx` / `n_gpu_layers` / …), reasoning split, `output_format` / grammar,
`extra_load_args`, `chat_template_path` (optional chat-template override — see above), unload
toggles — plus two connect-only inputs: **`context`** (reference material
appended to the system prompt — e.g. Character Card / Context Collector) and **`vision`** (from a
**`Vision Settings (GGUF)`** node — `mmproj` / `vision_handler` / `image_max_side`; connect it only
for vision). It emits everything as one `KINBURG_LLM_CONFIG` bundle. Wire its `config` output into
any LLM node — the chat node's **`persona_1`** **and** the text/vision nodes take the same bundle,
so one Settings node can drive several.

- **📨 Send** runs the workflow up to the chat node: `run()` generates a reply from the stored
  history + your message (+ optional image), **streams it into the bubble live** (over a
  `kinburg.chatllm` websocket event), and **blocks the downstream branch** (`ExecutionBlocker`) so
  nothing past the node runs while you chat. Reasoning models: the `<think>…` stream shows in an
  open **💭 thinking** block during generation, then collapses into a **💭 reasoning** toggle with
  the answer as the main text (only the answer goes downstream).
- **✅ Approve** runs with the gate open: `run()` **skips generation** and emits the **last reply**
  on the `text` output, so it flows downstream immediately (no re-generation, any seed).

**Editing the conversation.** Hover any bubble — yours or the model's — for **⧉ copy · ✎ edit ·
↻ resend · 🗑 delete**. ✎ swaps the bubble for an inline textarea sized to the message it holds
(Esc cancels, Ctrl+Enter saves). ↻ replays exactly the turn that produced the message, dropping
everything below it — a normal question/answer pair goes back to your question, while a persona's
no-user-message turn replays just that one reply. All of it is plain surgery on the stored history, so the
next turn simply sees the conversation you left behind. Everything is disabled while a reply is in
flight; if a run dies before reaching the node, **✕** on the live bubble unsticks it.

**Personas.** Four inputs — **`persona_1..4`** — each take a *whole* `Local LLM Settings (GGUF)`
bundle, so a persona brings its own **model, sampling and system prompt**, not just a different
prompt. **`persona_1` is the node's config**: wire only that one and this is an ordinary chat node
with no chip row. Wire a second and a chip row appears, one chip per persona. Clicking a chip only
**selects** it — 📨 Send is the sole trigger — and the active persona's bundle becomes the config
for that turn. All personas share one history, so a prompt-writer sees the whole discussion (as far
back as its window reaches); when it speaks, the other personas' replies are prefixed with their
name (`[Order manager]: …`) so it doesn't mistake them for its own past turns. Bubbles are labelled
with their persona once there's more than one.

The **⚙** chip sets, per persona: the **chip label** (defaults to the title of the wired-in node,
but only if you renamed it — four stock Settings nodes all read the same), an optional **trigger**
message, and the two context controls below. With nothing else wired the chip row is hidden and the
node behaves exactly as before.

**Who sees what.** Two independent per-persona settings decide which turns reach the model. Both
are worked out per request, never frozen into the history, so changing one takes effect on messages
that are already there — and the chat dims exactly what the persona you have selected won't see.

- **`Keep in context`** — *how long*. A prompt-writer emits a 300-token draft every press, and after
  an iteration or two the newer one supersedes it. This is how many of that persona's **own** most
  recent turns survive: blank = all (the default), `0` = none, `2` = the last two. It counts its own
  turns, not every turn since — chatting with someone else for twenty messages shouldn't make the
  writer forget the draft you're revising. A turn is the reply *plus* the instruction that produced
  it, so nothing is left dangling.
- **`Private`** — *for whom*. The persona's turns go to nobody but itself. It still reads its own
  back (that's what makes revising a draft work), while the persona you're brainstorming with never
  wades through prompt sprawl. Combine them: private + keep 2 = "I see my own last two, nobody else
  sees any."

Withheld messages stay in the chat, dimmed with 🚫; hover for which rule caught them. A message you
hide by hand is a third, separate thing and comes back with the **👁** button on its bubble.
✅ Approve releases the last reply regardless of any of this.

**What 📨 Send does with an empty input box** depends on who spoke last:

| Situation | What happens |
|---|---|
| You typed something | A normal turn, as always. |
| Box empty, **you switched persona** | A turn with **no user message at all** — the persona works from the conversation and its own system prompt, so nothing prods it in a way everyone else would then see in the context. Set a **trigger** in ⚙ if you'd rather send a standing instruction. |
| Box empty, **the last reply is the active persona's own** | **Continues that reply** from where `max_tokens` cut it off, appending to the same bubble instead of starting a new one. |

Two caveats on those: a no-user-message turn asks the model for a second `assistant` block in a row,
which a few chat templates (mistral-family) reject outright — the trigger field is the way out. And
continuing a reply uses a raw prefill, so it needs a chat template and doesn't work on the vision
path. Both modes also skip `thinking_directive`, since it would have to *become* the user turn.

**Context meter.** A thin row under the chat shows the KV-cache fill after the last turn —
`ctx 3 412 / 8 192 · 42% · 120 out · 4.2s` — from the same worker numbers `Kinburg Live Log` reports.
The bar turns amber past 75% and red past 90%. When a reply stops because it hit `max_tokens` the
row says so, which is the cue to press Send with an empty box and let the persona finish it.

**Archiving (`⤵ Archive N`).** When the conversation starts crowding the window, this folds the
older turns into a single **🗂 archived summary** and the model reads *that* instead of them. The
originals stay right where they were, dimmed — nothing is deleted, and **🗑 on the summary puts them
all back**. The summary is an ordinary bubble, so **✎ edits it** when the model dropped something
it shouldn't have.

It is one one-shot request with an empty history, so it still works when the chat is *already* over
budget, and it never touches VRAM: by default the summariser borrows the active persona's
already-loaded model with a compression prompt swapped in. Pick a dedicated persona in ⚙ if you'd
rather have a small fast model do it in its own voice. Turns that are already withheld — hidden by
hand, aged out of a retention window, or belonging to a private persona — are never folded in, so a
private thread can't leak into a shared brief.

The summary is **cumulative**: each pass rewrites it from the previous one plus the next block, so
archiving repeatedly never loses the first pass. One press folds at most 30 messages, which keeps
the summariser's own prompt from overflowing on a very long chat — press again for the next block.
In ⚙: **`Keep verbatim`** is how many recent messages are never folded (default 8), and **`Nag at`**
is the fill % at which the button turns amber (default 70). It's manual on purpose — the button
tells you how many would go, and you decide when.

**VRAM.** Personas that share a model *and* the loader fields (`n_ctx`, `n_gpu_layers`, `n_batch`,
`flash_attn`, `kv_cache_type`, `extra_load_args`, mmproj) cost **no reload** when you switch —
the worker's load signature ignores the system prompt and sampling. Differ in any of them and the
worker process is killed and restarted, which does free the VRAM but costs a full load; a chip is
marked **⟳** when picking it would reload the model. **`unload_on_approve`** (on by default) frees
the LLM entirely when you press ✅ Approve, so the image model downstream has room.

The dialogue — plus your pending message, the picked persona and the turn descriptor — lives in a
single **`chat_state`** JSON input that persists in the workflow (**🗑 Clear** wipes it; personas are
untouched). It is one input rather than six on purpose: the Vue frontend draws a 24 px row for
*every* widget a node owns, hidden or not, so six little carriers left 168 px of dead grey space
under the chat. The frontend doesn't render this one as a widget at all — it removes the
auto-created widget and lets the chat window itself carry the value — so the node has no invisible
rows. Workflows saved in the older six-widget format are migrated on load.

**Vision is optional:** the `image` input is on the
**chat node** (attached to the current turn); set an `mmproj` on the Settings node to enable it —
connecting an image with no `mmproj` set shows an error in the chat. `unload_llm_after_run` defaults
**off** so the model stays in VRAM for fast back-and-forth. Chat outputs: `text` (the approved
reply, gated) and a `help` cheat-sheet. Category `Kinburg-Nodes/LLM`.

### `save_song/` — Save Song
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

### `group_control/` — Group Control
**`Group Control 🎚️`** is a client-side control panel for enabling/bypassing workflow **groups by
name**. It lists every *unique* group title in the graph, one row each, with a switch that flips
all groups carrying that name between **`ALWAYS`** (active) and **`BYPASS`** (skipped) at once —
so naming three groups `Upscale` and toggling one row turns the whole set off together (the row
shows a `×N` badge for how many groups share the name). "Toggling a group" rewrites the `mode` of
the nodes inside it, exactly like ComfyUI's own *Set Group Nodes to…* menu, so the state is baked
into the target nodes and travels with the workflow — no extra serialization, survives a reload.
**Nesting is supported**: membership is resolved by bounding box, so an outer group automatically
covers the nodes of any group nested inside it, and nested names are shown **indented** by depth.
The list **grows and rebuilds itself** as you add, rename or delete groups (it polls the graph),
and a mixed selection (some nodes in a group active, some not) shows a `MIXED` marker. Each row
also has a **`▶` button that runs *only that group*** — it queues the group's output nodes as
ComfyUI **partial-execution targets**, so just this group and the nodes it depends on run while
every unrelated branch is skipped (the same mechanism as the core *Queue Selected Output Nodes*).
It respects the current on/off state (won't run a bypassed group) and tells you if the group has
no output node to run. A **`⋯` button** (or **right-click** the row / switch) opens a small menu
with **Run · Focus · `ALWAYS` · `BYPASS` · `NEVER` · Solo** — the discoverable way to **mute a
group to `NEVER`**, and to **Solo** it (set this group `ALWAYS` and every other group `BYPASS` as a
*persistent* state, with an **Undo solo** entry that restores the previous modes). **Click a row's
name or colour dot to _focus_ it** — the canvas pans/zooms to that group. A **filter box** hides
non-matching rows by name (handy with many groups; view-only, never touches the graph), and while
a filter is active the header **`all on`** / **`all off`** / **`all ✕`** (set to Never) buttons act
only on the matching groups (otherwise on every group).
**Hide the groups you never touch.** A group that just loads the base models, VAE or LoRAs is
set-and-forget — it only clutters the panel. Pick **`🚫 Hide from this list`** in a row's **`⋯`**
menu and the row disappears; the header count shows how many are put away (`7 +2🚫`). Nothing is
lost and nothing is destructive: the group keeps its current mode, and hidden groups are
deliberately **skipped by `all on` / `all off` / `all ✕` and by Solo**, so a bulk switch-off can
never disable the loaders your workflow needs. To get one back, click the header **`👁`** button —
hidden rows reappear **dimmed** (marked `🚫`), fully usable, with **`👁 Unhide`** in their `⋯` menu;
click `👁` again to put them away, or **right-click `👁` to unhide everything at once**. The hidden
set is saved on the node (in `properties`), so it travels with the workflow and survives a reload,
while the reveal toggle itself is per-session. If you hide a group that has nested groups inside it,
the children stay listed and simply shift one level left.
**Reorder the rows** to taste: **`sort ⇅`** sorts by name (click toggles A–Z / Z–A; **right-click**
restores the original order), or **drag a row by its `⠿` handle**. Reordering is **nesting-aware**:
a group can only be moved **among its siblings** (groups sharing the same parent), and dragging a
parent **carries its whole subtree** — so a nested group can't be stranded under an unrelated parent
in the list. The chosen order is saved on the node (in `properties`), so it travels with the
workflow and survives a reload — it's purely cosmetic and never affects the prompt (or the actual
group nesting, which is defined by the groups' positions on the canvas). The node itself has no inputs or outputs and never runs on
the backend — it's excluded from the prompt and only manipulates other nodes' modes on the client
before the run. Category `Kinburg-Nodes/util`.

## Installation

1. Clone this repository into `ComfyUI/custom_nodes` (or install it through
   **ComfyUI-Manager**).
2. The Local LLM nodes need `llama-cpp-python` (CUDA build). It is installed
   **automatically** by `install.py` (which ComfyUI-Manager runs on install). To do it
   by hand, run with this ComfyUI's Python:
   ```
   <ComfyUI>/.venv/Scripts/python.exe <ComfyUI>/custom_nodes/Kinburg-Nodes/install.py
   ```
   The Image Compare and Date String nodes need nothing extra.
3. The **GGUF converters** use packages that already ship with ComfyUI (`huggingface_hub`,
   `gguf`, `torch`, `safetensors`) and fetch the conversion scripts themselves (`git` on PATH,
   `auto_clone` on). Only **quantization** needs an external binary you provide: `llama-quantize`
   for the LLM converter, or the **patched** `llama-quantize` from city96's llama.cpp fork for
   the diffusion converter (see ComfyUI-GGUF/tools/README.md). Leaving `quantize = none` needs
   nothing extra.

Each node's parameters are documented in their tooltips. The Local LLM node also exposes a
`help` output with a quick cheat-sheet — wire it to a "Preview as Text" node to read it.

## License

MIT — see [LICENSE](LICENSE).
