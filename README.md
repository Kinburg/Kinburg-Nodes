# Kinburg-Nodes

A personal collection of custom ComfyUI nodes. One folder = one package: ComfyUI reads
the node mappings from the root `__init__.py`, and the sets are split into subpackages.

## What's inside

### `local_llm/` — Local LLM (GGUF, text)
Run a text GGUF LLM right inside ComfyUI **with guaranteed VRAM unloading**: inference
runs in a separate worker process, so when it exits the OS reclaims all of its VRAM —
ideal right before image generation. Features: streaming progress bar, token counters
(including an estimated `thoughts_tokens` / `answer_tokens` split of the output),
`finish_reason`, a separate `thoughts` output, reasoning control (Qwen3 `/no_think` and a
configurable `answer_marker` for models that reason without `<think>` tags), `min_p` /
`stop`, flash attention, KV-cache quantization, and structured output (JSON / GBNF / a
built-in Ideogram prompt grammar). Node: **`Local LLM (GGUF, text)`** (category
`Kinburg-Nodes/LLM`).

### `image_compare/` — Image Compare (HTML)
Takes a batch of images + short labels (`captions`) + full generation prompts (`prompts`)
and produces an interactive HTML comparison page — grid (columns + max row height),
before/after slider, opacity overlay, A/B flip, pixel difference, synced loupe, lightbox,
drag-to-reorder, and per-result **review** controls — a **hide**/reject button (with a
toolbar toggle to show/hide hidden results), a **star rating** (1–5), **tags**, and a
**comment** box — plus a "Save page" button and a batch of images with the captions drawn on
them. The review state (hide / rating / tags / comment) persists across reloads of the served
page (browser localStorage). Save
options: `output_dir` (any folder; the page is served straight from there, no copies),
`save_captioned_images`, `save_prompts_txt`. It also takes an optional **`settings`** input
(per-image blocks separated by `settings_separator`, e.g. from **Generation Info Filter**)
shown under each image with its own toggle on the page. The `url` output is a clickable http
link. Node: **`Image Compare (HTML)`** (category `Kinburg-Nodes/image/compare`).

Also in this package: **`Color Caption`** — write a caption and pick two colors, the
**text** color and the **band** color behind it (each via a 10-swatch palette plus a
native color picker for anything custom). It outputs a one-line JSON
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

### `util/` — Date String, Unlim Text Concat
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
sampler / latent / image output through its `passthrough` slot (tap it downstream of where
your branches converge — e.g. the sampler output — so the upstream walk reaches them all);
the node reads ComfyUI's hidden `PROMPT`, walks upstream, and lists the upstream nodes'
widget values (`[RandomNoise] noise_seed: …`, `[KSamplerSelect] sampler_name: …`, etc.). The
dump shows on the node, collapsed by default — click to expand. Outputs a human dump (`info`)
and machine-readable `data` (GEN_INFO). Category `Kinburg-Nodes/util`.

**`Generation Info Filter`** takes the `data` of one Generation Info per branch (growing
inputs) and emits a per-image `settings` string. `mode`: `all`, `differences` (only the
fields that vary across the inputs — each block then shows them all, so the images line
up), `custom` (fields named in `custom_fields`, one selector per line: `ClassType`,
`ClassType[n]`, `ClassType[n].param`, `ClassType.param`; `[n]` is the 1-based occurrence),
or `differences + custom`. A `help` output prints the selector syntax. `skip_empty` (on by
default) drops empty / unconnected inputs so a bypassed branch keeps the blocks aligned.
Outputs: `settings` (text — into the compare
node's **`settings`** input, shown under each image with its own page toggle) and
`settings_data` (`GEN_SETTINGS`, structured per-image `{key, value}` with class-qualified
keys — into the compare node's **`settings_data`** input, so a saved report stores settings
by field for filtering/sorting). Category `Kinburg-Nodes/util`.

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

Each node's parameters are documented in their tooltips. The Local LLM node also exposes a
`help` output with a quick cheat-sheet — wire it to a "Preview as Text" node to read it.

## License

MIT — see [LICENSE](LICENSE).
