# 🐍 Iterative Samplers & Optimizers

[← back to the node index](../README.md#-node-index)

---

## 🐍 `ouroboros/` — Ouroboros (Self-Correcting Sampler) 🐍

> **System Purpose & Overview**  
> A closed-loop text-to-image optimizer in a single node. An LLM rewrites the prompt, a custom sampler renders an image, a vision critic scores it and returns concrete advice, and the loop auto-adjusts until target scores are reached.

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
(the critic's verdict still lands whole — it isn't wired to the log token by token, though nothing
stops it now that grammar runs stream); **`per step`**
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

---

## 🦁 `chimera/` — Chimera (Multi-Sampler) 🦁

> **System Purpose & Overview**  
> Multi-stage sampling pipeline node allowing sequential execution of multiple samplers, schedulers, and denoise ranges.

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

---

## 🔄 `loops/` — Flexible Iteration Loops

> **System Purpose & Overview**  
> Flexible iteration control flow graphs for ComfyUI workflows: For Each, Repeat, and While loops with index retrieval and delay.

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

---

[← back to the node index](../README.md#-node-index)
