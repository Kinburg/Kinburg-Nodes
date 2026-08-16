# 🎨 Kinburg-Nodes

<!-- BEGIN GENERATED badges — tools/gen_readme_index.py -->
[![version](https://img.shields.io/badge/version-3.1.0-blue.svg)](pyproject.toml)
[![nodes](https://img.shields.io/badge/nodes-91-orange.svg)](#-node-index)
[![tests](https://img.shields.io/badge/tests-814%20checks-brightgreen.svg)](#-tests)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![ComfyUI Manager](https://img.shields.io/badge/ComfyUI--Manager-installable-8A2BE2.svg)](https://github.com/ltdrdata/ComfyUI-Manager)
<!-- END GENERATED badges -->

A personal collection of custom ComfyUI nodes. One folder = one package: ComfyUI reads the node mappings from the root `__init__.py`, and the sets are split into subpackages.

---

<!-- BEGIN GENERATED index — tools/gen_readme_index.py -->
## 📍 Node Index

All **91** nodes, grouped by the package they live in. Every package links to its full documentation under [`docs/`](docs).

### 🧠 LLM & Vision Systems

📖 **[🧠 LLM & Vision Systems](docs/llm.md)**

| Package | Nodes |
|---|---|
| [`local_llm/` — Local LLM (GGUF) & Live Logging](docs/llm.md#-local_llm--local-llm-gguf--live-logging) | `Context Sizer (GGUF)`, `Kinburg Live Log 📜`, `LLM Live Log (old id)`, `Local LLM (GGUF)`, `Local LLM Chat (GGUF)`, `Local LLM Settings (GGUF)`, `Send Image to Chat`, `Token Counter (GGUF)`, `Vision Settings (GGUF)` |
| [`llm_server/` — Local LLM (server client, text)](docs/llm.md#-llm_server--local-llm-server-client-text) | `Local LLM (server client, text)` |
| [`context/` — Character Card, Entity Card, Context Collector](docs/llm.md#-context--character-card-entity-card-context-collector) | `Character Card`, `Context Collector`, `Entity Card` |
| [`vision_judge/` — Vision LLM Judge & Criteria Builder](docs/llm.md#-vision_judge--vision-llm-judge--criteria-builder) | `Criteria Builder 📋`, `Vision LLM Judge` |
| [`card_presets/` — Card Save & Card Presets](docs/llm.md#-card_presets--card-save--card-presets) | `Card Presets`, `Card Save` |
| [`grammar_presets/` — Grammar Presets](docs/llm.md#-grammar_presets--grammar-presets) | `Grammar Presets` |
| [`show_text/` — Show Text (Markdown)](docs/llm.md#-show_text--show-text-markdown) | `Show Text (Markdown)` |
| [`gguf_convert/` — Safetensors → GGUF Converters](docs/llm.md#-gguf_convert--safetensors--gguf-converters) | `Diffusion Safetensors -> GGUF (city96)`, `Safetensors -> GGUF (llama.cpp)` |

### 🐍 Iterative Samplers & Optimizers

📖 **[🐍 Iterative Samplers & Optimizers](docs/samplers.md)**

| Package | Nodes |
|---|---|
| [`ouroboros/` — Ouroboros (Self-Correcting Sampler) 🐍](docs/samplers.md#-ouroboros--ouroboros-self-correcting-sampler-) | `Ouroboros (Self-Correcting Sampler) 🐍`, `Ouroboros Critic Settings 🐍`, `Ouroboros Live Log 🐍📜`, `Sampler Settings` |
| [`chimera/` — Chimera (Multi-Sampler) 🦁](docs/samplers.md#-chimera--chimera-multi-sampler-) | `Chimera (Multi-Sampler) 🦁` |

### 🎵 Audio & Music Suite

📖 **[🎵 Audio & Music Suite](docs/audio.md)**

| Package | Nodes |
|---|---|
| [`siren/` — Siren Suite 🧜](docs/audio.md#-siren--siren-suite-) | `Siren (Music Sampler) 🧜`, `Siren Cast (Voice Plan) 🧜`, `Siren Compare (Audio) 🧜`, `Siren Scope (Audio → Image) 🧜`, `Siren Score (Lyrics → Plan) 🧜`, `Siren Section (Audio Window) 🧜` |
| [`audio_sr/` — Audio SR (48 kHz Upscale) 🔊](docs/audio.md#-audio_sr--audio-sr-48-khz-upscale-) | `Audio SR (48 kHz Upscale) 🔊` |
| [`save_song/` — Save Song & Song Tags](docs/audio.md#-save_song--save-song--song-tags) | `Save Song`, `Song Tags` |

### 🎬 Video Generation & Storyboarding

📖 **[🎬 Video Generation & Storyboarding](docs/video.md)**

| Package | Nodes |
|---|---|
| [`morpheus/` — Morpheus Suite 🌙](docs/video.md#-morpheus--morpheus-suite-) | `Morpheus (Video Sampler) 🌙`, `Morpheus Dream Board 🌙`, `Morpheus Dream 🌙`, `Morpheus Storyboard 🌙` |

### 🎛️ Presets & Asset Management

📖 **[🎛️ Presets & Asset Management](docs/presets.md)**

| Package | Nodes |
|---|---|
| [`model_presets/` — Model Capture & Settings Management](docs/presets.md#-model_presets--model-capture--settings-management) | `Model Capture 📥`, `Model Select 🎛`, `Settings Save 💾`, `Settings Select ⚙` |
| [`prompt_presets/` — Prompt Presets](docs/presets.md#-prompt_presets--prompt-presets) | `Prompt Presets` |
| [`prompt_variations/` — Prompt Variations](docs/presets.md#-prompt_variations--prompt-variations) | `Prompt Variations` |
| [`lora/` — Lora Trigger Loader & Accumulator](docs/presets.md#-lora--lora-trigger-loader--accumulator) | `Lora Trigger Loader`, `Lora Unlim Accumulator` |

### 🖼️ Image Processing & Visualizers

📖 **[🖼️ Image Processing & Visualizers](docs/images.md)**

| Package | Nodes |
|---|---|
| [`image_compare/` — Image Compare (HTML) & Color Caption](docs/images.md#-image_compare--image-compare-html--color-caption) | `Color Caption`, `Image Compare (HTML)` |
| [`image_batch/` — Unlim Image Batch & List](docs/images.md#-image_batch--unlim-image-batch--list) | `Unlim Image Batch`, `Unlim Image List` |
| [`collage/` — Collage Layout Builder](docs/images.md#-collage--collage-layout-builder) | `Collage` |

### 🛠️ Workflow Control & Utilities

📖 **[🛠️ Workflow Control & Utilities](docs/utilities.md)**

| Package | Nodes |
|---|---|
| [`util/` — General Workflow Utilities](docs/utilities.md#-util--general-workflow-utilities) | `Any Switch`, `Any to String`, `Color Picker`, `Combo to String`, `Date String`, `JSON Extract`, `Text Transform`, `Unlim Text Concat` |
| [`timer/` — Execution Timer](docs/utilities.md#-timer--execution-timer) | `Start Timer`, `Stop Timer` |
| [`gen_info/` — Generation Info & Filter](docs/utilities.md#-gen_info--generation-info--filter) | `Generation Info`, `Generation Info Filter` |
| [`group_control/` — Group Control 🎚️](docs/utilities.md#-group_control--group-control-) | `Group Control 🎚️` |
| [`loops/` — Flexible Iteration Loops](docs/utilities.md#-loops--flexible-iteration-loops) | `Delay`, `For Each (Collect)`, `For Each (Open)`, `Get by Index`, `List Output`, `Repeat (Close)`, `Repeat (Open)`, `While (Close)`, `While (Open)` |
| [`accumulators/` — Name-Based Accumulators](docs/utilities.md#-accumulators--name-based-accumulators) | `Get Accumulator (audio)`, `Get Accumulator (captions)`, `Get Accumulator (gen info)`, `Get Accumulator (images list)`, `Get Accumulator (images)`, `Get Accumulator (prompts)`, `Get Accumulator (texts)`, `Set Accumulator (audio)`, `Set Accumulator (captions)`, `Set Accumulator (gen info)`, `Set Accumulator (images)`, `Set Accumulator (prompts)`, `Set Accumulator (texts)` |
| [`list_ops/` — List & Batch Operations](docs/utilities.md#-list_ops--list--batch-operations) | `Image Batch Insert`, `Image Batch Remove`, `List Insert`, `List Remove` |
| [`report/` — Investigation Report DB](docs/utilities.md#-report--investigation-report-db) | — *no nodes of its own* |

---

<details>
<summary><b>🔎 All node ids (for workflow JSON / bug reports)</b></summary>

| Display name | Node id | Category |
|---|---|---|
| Any Switch | `KinburgAnySwitch` | `Kinburg-Nodes/util` |
| Any to String | `AnyToString` | `Kinburg-Nodes/util` |
| Audio SR (48 kHz Upscale) 🔊 | `KinburgAudioSR` | `Kinburg-Nodes/audio` |
| Card Presets | `CardPresets` | `Kinburg-Nodes/LLM/presets` |
| Card Save | `CardSave` | `Kinburg-Nodes/LLM/presets` |
| Character Card | `CharacterCard` | `Kinburg-Nodes/LLM/context` |
| Chimera (Multi-Sampler) 🦁 | `KinburgChimeraSampler` | `Kinburg-Nodes/Bestiary/Chimera` |
| Collage | `CustomCollageNode` | `Kinburg-Nodes/image` |
| Color Caption | `ColorCaption` | `Kinburg-Nodes/image/compare` |
| Color Picker | `ColorPicker` | `Kinburg-Nodes/util` |
| Combo to String | `ComboToString` | `Kinburg-Nodes/util` |
| Context Collector | `ContextCollector` | `Kinburg-Nodes/LLM/context` |
| Context Sizer (GGUF) | `KinburgContextSizer` | `Kinburg-Nodes/LLM` |
| Criteria Builder 📋 | `CriteriaBuilder` | `Kinburg-Nodes/LLM/presets` |
| Date String | `KinburgDateString` | `Kinburg-Nodes/util` |
| Delay | `KinburgDelay` | `Kinburg-Nodes/flow/loops` |
| Diffusion Safetensors -> GGUF (city96) | `SafetensorsToGGUFDiffusion` | `Kinburg-Nodes/LLM/GGUF` |
| Entity Card | `EntityCard` | `Kinburg-Nodes/LLM/context` |
| For Each (Collect) | `KinburgForEachCollect` | `Kinburg-Nodes/flow/loops` |
| For Each (Open) | `KinburgForEachOpen` | `Kinburg-Nodes/flow/loops` |
| Generation Info | `GenerationInfo` | `Kinburg-Nodes/util` |
| Generation Info Filter | `GenerationInfoFilter` | `Kinburg-Nodes/util` |
| Get Accumulator (audio) | `GetAccumAudio` | `Kinburg-Nodes/flow/accumulators` |
| Get Accumulator (captions) | `GetAccumCaptions` | `Kinburg-Nodes/flow/accumulators` |
| Get Accumulator (gen info) | `GetAccumGenInfo` | `Kinburg-Nodes/flow/accumulators` |
| Get Accumulator (images list) | `GetAccumImagesList` | `Kinburg-Nodes/flow/accumulators` |
| Get Accumulator (images) | `GetAccumImages` | `Kinburg-Nodes/flow/accumulators` |
| Get Accumulator (prompts) | `GetAccumPrompts` | `Kinburg-Nodes/flow/accumulators` |
| Get Accumulator (texts) | `GetAccumTexts` | `Kinburg-Nodes/flow/accumulators` |
| Get by Index | `KinburgGetByIndex` | `Kinburg-Nodes/flow/loops` |
| Grammar Presets | `GrammarPresets` | `Kinburg-Nodes/LLM/presets` |
| Group Control 🎚️ | `KinburgGroupControl` | `Kinburg-Nodes/util` |
| Image Batch Insert | `ImageBatchInsert` | `Kinburg-Nodes/image` |
| Image Batch Remove | `ImageBatchRemove` | `Kinburg-Nodes/image` |
| Image Compare (HTML) | `ImageCompareHTML` | `Kinburg-Nodes/image/compare` |
| JSON Extract | `KinburgJSONExtract` | `Kinburg-Nodes/util` |
| Kinburg Live Log 📜 | `KinburgLiveLog` | `Kinburg-Nodes/LLM` |
| LLM Live Log (old id) | `KinburgLLMLog` | `Kinburg-Nodes/LLM` |
| List Insert | `ListInsert` | `Kinburg-Nodes/flow/list` |
| List Output | `KinburgListEmit` | `Kinburg-Nodes/flow/loops` |
| List Remove | `ListRemove` | `Kinburg-Nodes/flow/list` |
| Local LLM (GGUF) | `LocalLLMGGUF` | `Kinburg-Nodes/LLM` |
| Local LLM (server client, text) | `LocalLLMServerText` | `Kinburg-Nodes/LLM` |
| Local LLM Chat (GGUF) | `LocalLLMChatGGUF` | `Kinburg-Nodes/LLM` |
| Local LLM Settings (GGUF) | `LocalLLMSettingsGGUF` | `Kinburg-Nodes/LLM` |
| Lora Trigger Loader | `LoraTriggerLoader` | `Kinburg-Nodes/lora` |
| Lora Unlim Accumulator | `LoraUnlimAccumulator` | `Kinburg-Nodes/lora` |
| Model Capture 📥 | `KinburgModelCapture` | `Kinburg-Nodes/model` |
| Model Select 🎛 | `KinburgModelSelect` | `Kinburg-Nodes/model` |
| Morpheus (Video Sampler) 🌙 | `KinburgMorpheus` | `Kinburg-Nodes/Bestiary/Morpheus` |
| Morpheus Dream Board 🌙 | `KinburgDreamBoard` | `Kinburg-Nodes/Bestiary/Morpheus` |
| Morpheus Dream 🌙 | `KinburgMorpheusDream` | `Kinburg-Nodes/Bestiary/Morpheus` |
| Morpheus Storyboard 🌙 | `KinburgMorpheusStoryboard` | `Kinburg-Nodes/Bestiary/Morpheus` |
| Ouroboros (Self-Correcting Sampler) 🐍 | `KinburgOuroboros` | `Kinburg-Nodes/Bestiary/Ouroboros` |
| Ouroboros Critic Settings 🐍 | `KinburgCriticSettings` | `Kinburg-Nodes/Bestiary/Ouroboros` |
| Ouroboros Live Log 🐍📜 | `KinburgOuroborosLog` | `Kinburg-Nodes/Bestiary/Ouroboros` |
| Prompt Presets | `PromptPresets` | `Kinburg-Nodes/prompt` |
| Prompt Variations | `PromptVariations` | `Kinburg-Nodes/prompt` |
| Repeat (Close) | `KinburgRepeatClose` | `Kinburg-Nodes/flow/loops` |
| Repeat (Open) | `KinburgRepeatOpen` | `Kinburg-Nodes/flow/loops` |
| Safetensors -> GGUF (llama.cpp) | `SafetensorsToGGUF` | `Kinburg-Nodes/LLM/GGUF` |
| Sampler Settings | `KinburgSamplerSettings` | `Kinburg-Nodes/Bestiary` |
| Save Song | `KinburgSaveSong` | `Kinburg-Nodes/audio` |
| Send Image to Chat | `LocalLLMChatSendImage` | `Kinburg-Nodes/LLM` |
| Set Accumulator (audio) | `SetAccumAudio` | `Kinburg-Nodes/flow/accumulators` |
| Set Accumulator (captions) | `SetAccumCaptions` | `Kinburg-Nodes/flow/accumulators` |
| Set Accumulator (gen info) | `SetAccumGenInfo` | `Kinburg-Nodes/flow/accumulators` |
| Set Accumulator (images) | `SetAccumImages` | `Kinburg-Nodes/flow/accumulators` |
| Set Accumulator (prompts) | `SetAccumPrompts` | `Kinburg-Nodes/flow/accumulators` |
| Set Accumulator (texts) | `SetAccumTexts` | `Kinburg-Nodes/flow/accumulators` |
| Settings Save 💾 | `KinburgSettingsSave` | `Kinburg-Nodes/model` |
| Settings Select ⚙ | `KinburgSettingsSelect` | `Kinburg-Nodes/model` |
| Show Text (Markdown) | `KinburgShowText` | `Kinburg-Nodes/util` |
| Siren (Music Sampler) 🧜 | `KinburgSirenSampler` | `Kinburg-Nodes/Bestiary/Siren` |
| Siren Cast (Voice Plan) 🧜 | `KinburgSirenCast` | `Kinburg-Nodes/Bestiary/Siren` |
| Siren Compare (Audio) 🧜 | `KinburgSirenCompare` | `Kinburg-Nodes/Bestiary/Siren` |
| Siren Scope (Audio → Image) 🧜 | `KinburgSirenScope` | `Kinburg-Nodes/Bestiary/Siren` |
| Siren Score (Lyrics → Plan) 🧜 | `KinburgSirenScore` | `Kinburg-Nodes/Bestiary/Siren` |
| Siren Section (Audio Window) 🧜 | `KinburgSirenSection` | `Kinburg-Nodes/Bestiary/Siren` |
| Song Tags | `KinburgSongTags` | `Kinburg-Nodes/audio` |
| Start Timer | `StartTimer` | `Kinburg-Nodes/util` |
| Stop Timer | `StopTimer` | `Kinburg-Nodes/util` |
| Text Transform | `KinburgTextTransform` | `Kinburg-Nodes/util` |
| Token Counter (GGUF) | `KinburgTokenCounter` | `Kinburg-Nodes/LLM` |
| Unlim Image Batch | `UnlimImageBatch` | `Kinburg-Nodes/image` |
| Unlim Image List | `UnlimImageList` | `Kinburg-Nodes/image` |
| Unlim Text Concat | `UnlimTextConcat` | `Kinburg-Nodes/util` |
| Vision LLM Judge | `VisionLLMJudge` | `Kinburg-Nodes/LLM` |
| Vision Settings (GGUF) | `LocalLLMVisionSettingsGGUF` | `Kinburg-Nodes/LLM` |
| While (Close) | `KinburgWhileClose` | `Kinburg-Nodes/flow/loops` |
| While (Open) | `KinburgWhileOpen` | `Kinburg-Nodes/flow/loops` |

</details>
<!-- END GENERATED index -->

---

## 📦 Installation

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

---

## 🧪 Tests

```bash
python tests/run.py
```

with ComfyUI's own interpreter (`.venv/Scripts/python.exe`). **814 checks in 14 suites, about a
minute** — over `local_llm/` (chat, worker, send-image, lazy guard), `morpheus/` (storyboard, dream
board), `siren/` (cast, score), `audio_sr/`, `lora/`, three of the `web/*.js` extensions (chat,
dream board, group control), and the docs audit below. Nothing real is loaded (llama.cpp, H3 and the
browser are all stubbed), so it is a regression net for those paths and **not** a substitute for
trying a change in the app. `tests/README.md` says exactly what is and is not covered.

### 📑 Keeping the docs honest

The node index in this README is **generated**, so it cannot drift from the nodes the pack
registers:

```bash
python tools/gen_readme_index.py
```

It rewrites only the regions between the `<!-- BEGIN/END GENERATED -->` markers — every other line
here is hand-written. `--check` verifies instead of writing (that is the `docs` test suite, so a
stale index fails the tests), and `--run-tests` also refreshes the tests badge from a real run.

Adding a node needs no edit there at all: it appears in the index on the next run. Adding a
**package** needs only its `##` heading in the right `docs/*.md` file — the tool reads the folder
name out of the heading, and fails with `undocumented: <folder>` if a node's folder has no section
to live in. A group's position comes from its own `<!-- index-order: N -->` line.

Alongside the index it audits the prose: every node must be mentioned, every local link and anchor
must resolve, and every backticked `snake_case` term must be a real node input or output — anything
else has to be listed in `tools/known_terms.txt` (sampler names, enum values, comfy internals).
That last check is the one that catches a renamed input still being documented under its old name.



## 📄 License

MIT — see [LICENSE](LICENSE).