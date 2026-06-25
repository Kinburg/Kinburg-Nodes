# Kinburg-Nodes

A personal collection of custom ComfyUI nodes. One folder = one package: ComfyUI reads
the node mappings from the root `__init__.py`, and the sets are split into subpackages.

## What's inside

### `local_llm/` — Local LLM (GGUF, text + vision)
Run a GGUF LLM right inside ComfyUI **with guaranteed VRAM unloading**: inference
runs in a separate worker process, so when it exits the OS reclaims all of its VRAM —
ideal right before image generation. Features: streaming progress bar, token counters
(including an estimated `thoughts_tokens` / `answer_tokens` split of the output),
`finish_reason`, a separate `thoughts` output, reasoning control (Qwen3 `/no_think` and a
configurable `answer_marker` for models that reason without `<think>` tags), `min_p` /
`stop`, flash attention, KV-cache quantization, and structured output (JSON / GBNF / a
built-in Ideogram prompt grammar). Node: **`Local LLM (GGUF, text)`** (category
`Kinburg-Nodes/LLM`).

**`Local LLM (GGUF, vision)`** is the multimodal twin — same engine, sampling, reasoning
split and output formats, plus an **`image`** input and a second model path for the
projector **`mmproj`** `.gguf` (both files ship together in a vision model's repo; put them
in `ComfyUI/models/llm`). `vision_handler = auto (MTMD)` uses llama.cpp's generic
multimodal loader, which handles most modern vision GGUFs (LLaVA, Qwen2-VL, MiniCPM-V,
Gemma 3, SmolVLM, …) straight from the mmproj — only switch to a specific family if auto
fails. `image_max_side` downscales images before they're sent to the worker. Combine it with
`output_format = json_object` to get a structured description of an image. Both LLM nodes
share one worker process. Node: **`Local LLM (GGUF, vision)`** (category `Kinburg-Nodes/LLM`).

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
`save_captioned_images`, `save_prompts_txt`. Only `images` is required; the text inputs are
all sockets (no inline fields): `captions` (one per line, e.g. from **Get Accumulator
(captions)**), `prompts` (full per-image blocks separated by a `---` line, e.g. from **Get
Accumulator (prompts)**), and `settings_data` (structured per-image settings from
**Generation Info Filter**) — rendered under each image as one `[Class] param: value` line
per field, with its own page toggle. The `url` output is a clickable http
link. Node: **`Image Compare (HTML)`** (category `Kinburg-Nodes/image/compare`).

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

### `util/` — Date String, Unlim Text Concat, Color Picker
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

### `accumulators/` — Set / Get Results (name-based accumulators)
For collecting parallel branches without manual batch wiring. **`Set Results (image)`** is a
labelled pass-through: connect a flow's final image and give it an accumulator `name` (e.g.
`IMG_RESULTS`); copying the flow auto-increments its `index`. **`Get Results (image)`** has a
**dropdown** of the defined accumulator names — pick one (it auto-collects) — plus a
**Collect** button that physically wires every matching Set's output into it (real links, in
`index` order) and batches them (reusing Unlim Image Batch: `mode` /
`pad_color` / `skip_empty`). Plug `Get Results` where an image batch used to go (e.g. Image
Compare). Press Collect again after adding/removing Set nodes. A **text pair** —
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

**`Collect All Accumulators`** is a one-button helper for big workflows: a standalone node
whose single **🔌 Collect All** button (re)wires *every* Get Accumulator in the graph at once,
so you don't have to visit each one after scaling the workflow. Category
`Kinburg-Nodes/accumulators`.

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
