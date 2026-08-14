# 🎛️ Presets & Asset Management

<!-- index-order: 5 -->

[← back to the node index](../README.md#-node-index)

---

## 🎛️ `model_presets/` — Model Capture & Settings Management

> **System Purpose & Overview**  
> Snapshot and recall complete model settings, CLIP/VAE configurations, and sampler parameter sets.

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

---

## 💬 `prompt_presets/` — Prompt Presets

> **System Purpose & Overview**  
> Save and manage reusable prompt snippet libraries.

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

---

## 🔀 `prompt_variations/` — Prompt Variations

> **System Purpose & Overview**  
> Generate combinatorial and matrix variations of prompt text strings.

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

---

## ⚡ `lora/` — Lora Trigger Loader & Accumulator

> **System Purpose & Overview**  
> LoRA trigger phrase auto-loading and dynamic unlimited LoRA accumulator.

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

---

[← back to the node index](../README.md#-node-index)
