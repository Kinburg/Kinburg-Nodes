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
> `gbnf_grammar` output). The chat node uses the same bundle. (See the Settings/chat section below.)

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
a name + a description. Both cards carry a **`save_preset_as`** field: type a name and run to save
the card to a reusable library — it saves at run time, so it captures whatever's filled in, whether
**typed or wired in** (e.g. from a photo description); reuse it via **Card Presets** (below).
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
nodes' worker and keeps the model loaded across every image. Outputs: **`results_json`** —
`[{index, score, score_max, tags, comment}]`, wire it into **Image Compare**'s `judge_data`
input for a read-only judge section per image (stars / tags / comment) alongside your own review;
**`summary`** (a readable per-image report); and **`best_index`** (the top-scoring image). Closes
the generate → auto-evaluate → pick-the-best loop entirely locally. Category `Kinburg-Nodes/LLM`.

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
Category `Kinburg-Nodes/LLM`.

### `card_presets/` — Card Presets
**`Card Presets`** is a saved library of filled cards: pick a character / entity from a dropdown and
its ready Markdown block comes out the `card` output (feed it into Context Collector). You build the
library from the **Character Card** / **Entity Card** nodes' `save_preset_as` field (see above);
presets are rendered back through the card nodes' own logic — so the format always matches — and
persisted on disk. **🗑 Manage** deletes entries, **🔄 Refresh** re-reads the list. Build a character
once (by hand or by photo via Grammar Presets), then reuse it from the dropdown instead of
re-describing the same photo every time. Category `Kinburg-Nodes/LLM`.

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
before/after slider, opacity overlay, A/B flip, pixel difference, synced loupe, lightbox,
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

**`JSON Extract`** pulls fields out of a JSON string by path into separate `STRING` outputs — up
to six independent paths (`path_1…6`) → six `value_*` outputs, plus a `found` flag (all non-empty
paths resolved) and a `report` (per-path hit/miss). Path syntax: dot keys, `[n]` / `.n` indices
(negative allowed), `$` (or empty) for the whole document; an object/array value comes out as
compact JSON, and prose-wrapped JSON is tolerated (the first `{…}`/`[…]` is parsed). Pairs with
the structured-output LLM nodes (e.g. `ideogram4_json`) to route sub-fields into different prompt
inputs. Category `Kinburg-Nodes/util`.

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
`model` / `clip` / `prompt`. A LoRA with no effective strength (off) is skipped entirely —
neither applied nor does its trigger word get added. Loaded files are cached per run. Category
`Kinburg-Nodes/lora`.

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

**`Collect All Accumulators`** is a one-button helper for big workflows: a standalone node
whose single **🔌 Collect All** button (re)wires *every* Get Accumulator in the graph at once,
so you don't have to visit each one after scaling the workflow. Like the per-node Collect, it
rebuilds links from scratch and wires only **active** Sets — bypass or mute a Set and re-collect
to drop it from every accumulator. Its **`auto_collect`** toggle (on by default) re-collects
automatically right before the workflow is queued, so the run always reflects the current
(non-bypassed) Sets; turn it off to collect only on the button press. Category
`Kinburg-Nodes/accumulators`.

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
box, so the node doesn't grow as the chat fills), the **`user_message`** field and the **Send /
Approve / Clear** buttons. Everything about *how* to generate comes through a single **`config`**
input.

**`Local LLM Settings (GGUF)`** is that config node: it holds the options — model, system prompt,
sampling, loader (`n_ctx` / `n_gpu_layers` / …), reasoning split, `output_format` / grammar,
`extra_load_args`, `chat_template_path` (optional chat-template override — see above), unload
toggles — plus two connect-only inputs: **`context`** (reference material
appended to the system prompt — e.g. Character Card / Context Collector) and **`vision`** (from a
**`Vision Settings (GGUF)`** node — `mmproj` / `vision_handler` / `image_max_side`; connect it only
for vision). It emits everything as one `KINBURG_LLM_CONFIG` bundle. Wire its `config` output into
any LLM node — the chat node **and** the text/vision nodes take the same bundle, so one Settings
node can drive several.

- **📨 Send** runs the workflow up to the chat node: `run()` generates a reply from the stored
  history + your message (+ optional image), **streams it into the bubble live** (over a
  `kinburg.chatllm` websocket event), and **blocks the downstream branch** (`ExecutionBlocker`) so
  nothing past the node runs while you chat. Reasoning models: the `<think>…` stream shows in an
  open **💭 thinking** block during generation, then collapses into a **💭 reasoning** toggle with
  the answer as the main text (only the answer goes downstream). Each message has a **⧉ copy** button.
- **✅ Approve** runs with the gate open: `run()` **skips generation** and emits the **last reply**
  on the `text` output, so it flows downstream immediately (no re-generation, any seed).

The dialogue lives in a hidden, serialized `history_json` widget (persists in the workflow;
**🗑 Clear** wipes it). **Vision is optional:** the `image` input is on the **chat node** (attached
to the current turn); set an `mmproj` on the Settings node to enable it — connecting an image with
no `mmproj` set shows an error in the chat. `unload_llm_after_run` defaults **off** so the model
stays in VRAM for fast back-and-forth. Chat outputs: `text` (the approved reply, gated) and a
`help` cheat-sheet. Category `Kinburg-Nodes/LLM`.

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
and a mixed selection (some nodes in a group active, some not) shows a `MIXED` marker. Extras:
**`all on`** / **`all off`** buttons, and a **right-click** on any switch sets **`MUTE`** (Never)
instead of Bypass. **Reorder the rows** to taste: **`sort ⇅`** sorts by name (click toggles A–Z /
Z–A; **right-click** restores the original order), or **drag a row by its `⠿` handle**. The chosen
order is saved on the node (in `properties`), so it travels with the workflow and survives a
reload — it's purely cosmetic and never affects the prompt. The node has no inputs or outputs and
never runs on the backend — it's excluded from the prompt and only manipulates other nodes' modes
on the client before the run. Category `Kinburg-Nodes/util`.

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
