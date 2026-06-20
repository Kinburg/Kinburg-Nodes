# Kinburg-Nodes

A personal collection of custom ComfyUI nodes. One folder = one package: ComfyUI reads
the node mappings from the root `__init__.py`, and the sets are split into subpackages.

## What's inside

### `local_llm/` — Local LLM (GGUF, text)
Run a text GGUF LLM right inside ComfyUI **with guaranteed VRAM unloading**: inference
runs in a separate worker process, so when it exits the OS reclaims all of its VRAM —
ideal right before image generation. Features: streaming progress bar, token counters,
`finish_reason`, a separate `thoughts` output, reasoning control (Qwen3 `/no_think` and a
configurable `answer_marker` for models that reason without `<think>` tags), `min_p` /
`stop`, flash attention, KV-cache quantization, and structured output (JSON / GBNF / a
built-in Ideogram prompt grammar). Node: **`Local LLM (GGUF, text)`** (category
`Kinburg-Nodes/LLM`).

### `image_compare/` — Image Compare (HTML)
Takes a batch of images + short labels (`captions`) + full generation prompts (`prompts`)
and produces an interactive HTML comparison page — grid (columns + max row height),
before/after slider, opacity overlay, A/B flip, pixel difference, synced loupe, lightbox,
and a "Save page" button — plus a batch of images with the captions drawn on them. Save
options: `output_dir` (any folder; the page is served straight from there, no copies),
`save_captioned_images`, `save_prompts_txt`. The `url` output is a clickable http link.
Node: **`Image Compare (HTML)`** (category `Kinburg-Nodes/image/compare`).

### `util/` — Date String
Appends the current **date** (and optionally **time**) to a string, with selectable
formats. Handy for building save paths: e.g. `project/2026-06-20` (a folder per day) or
`.../2026-06-20/17-05` (per minute). The `/` separator creates subfolders; `_`/`-` make a
flat name. Presets plus a `custom` strftime field. The node is re-evaluated on every run
so the date never freezes from caching. Node: **`Date String`** (category
`Kinburg-Nodes/util`).

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
