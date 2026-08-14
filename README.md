# 🎨 Kinburg-Nodes

[![version](https://img.shields.io/badge/version-3.1.0-blue.svg)](pyproject.toml)
[![nodes](https://img.shields.io/badge/nodes-90-orange.svg)](#-node-index)
[![tests](https://img.shields.io/badge/tests-807%20checks-brightgreen.svg)](#-tests)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![ComfyUI Manager](https://img.shields.io/badge/ComfyUI--Manager-installable-8A2BE2.svg)](https://github.com/ltdrdata/ComfyUI-Manager)

A personal collection of custom ComfyUI nodes. One folder = one package: ComfyUI reads the node mappings from the root `__init__.py`, and the sets are split into subpackages.

---

## 📍 Node Index

All **90** nodes, grouped by the package they live in. Every package links to its full documentation under [`docs/`](docs).

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
| [`ouroboros/` — Ouroboros (Self-Correcting Sampler) 🐍](docs/samplers.md#-ouroboros--ouroboros-self-correcting-sampler-) | `Critic Settings (GGUF)`, `Ouroboros (Self-Correcting Sampler) 🐍`, `Ouroboros Live Log 🐍📜`, `Sampler Settings` |
| [`chimera/` — Chimera (Multi-Sampler) 🦁](docs/samplers.md#-chimera--chimera-multi-sampler-) | `Chimera (Multi-Sampler) 🦁` |
| [`loops/` — Flexible Iteration Loops](docs/samplers.md#-loops--flexible-iteration-loops) | `Delay`, `For Each (Collect)`, `For Each (Open)`, `Get by Index`, `List Output`, `Repeat (Close)`, `Repeat (Open)`, `While (Close)`, `While (Open)` |

### 🎵 Audio & Music Suite

📖 **[🎵 Audio & Music Suite](docs/audio.md)**

| Package | Nodes |
|---|---|
| [`siren/` — Siren Suite 🧜](docs/audio.md#-siren--siren-suite-) | `Siren (Music Sampler) 🧜`, `Siren Cast (Voice Plan) 🧜`, `Siren Compare (Audio) 🧜`, `Siren Scope (Audio → Image) 🧜`, `Siren Score (Lyrics → Plan) 🧜`, `Siren Section (Audio Window) 🧜` |
| [`audio_sr/` — Audio SR (48 kHz Upscale) 🔊](docs/audio.md#-audio_sr--audio-sr-48-khz-upscale-) | `Audio SR (48 kHz Upscale) 🔊` |
| [`save_song/` — Save Song](docs/audio.md#-save_song--save-song) | `Save Song` |

### 🎬 Video Generation & Storyboarding

📖 **[🎬 Video Generation & Storyboarding](docs/video.md)**

| Package | Nodes |
|---|---|
| [`morpheus/` — Morpheus Suite 🌙](docs/video.md#-morpheus--morpheus-suite-) | `Dream Board 🎬`, `Morpheus (Video Sampler) 🌙`, `Morpheus Dream 🌙`, `Morpheus Storyboard 🌙` |

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
| Card Presets | `CardPresets` | `Kinburg-Nodes/LLM` |
| Card Save | `CardSave` | `Kinburg-Nodes/LLM` |
| Character Card | `CharacterCard` | `Kinburg-Nodes/LLM` |
| Chimera (Multi-Sampler) 🦁 | `KinburgChimeraSampler` | `Kinburg-Nodes/sampling` |
| Collage | `CustomCollageNode` | `Kinburg-Nodes/image` |
| Color Caption | `ColorCaption` | `Kinburg-Nodes/image/compare` |
| Color Picker | `ColorPicker` | `Kinburg-Nodes/util` |
| Combo to String | `ComboToString` | `Kinburg-Nodes/util` |
| Context Collector | `ContextCollector` | `Kinburg-Nodes/LLM` |
| Context Sizer (GGUF) | `KinburgContextSizer` | `Kinburg-Nodes/LLM` |
| Criteria Builder 📋 | `CriteriaBuilder` | `Kinburg-Nodes/LLM` |
| Critic Settings (GGUF) | `KinburgCriticSettings` | `Kinburg-Nodes/LLM` |
| Date String | `KinburgDateString` | `Kinburg-Nodes/util` |
| Delay | `KinburgDelay` | `Kinburg-Nodes/loops` |
| Diffusion Safetensors -> GGUF (city96) | `SafetensorsToGGUFDiffusion` | `Kinburg-Nodes/GGUF` |
| Dream Board 🎬 | `KinburgDreamBoard` | `Kinburg-Nodes/sampling` |
| Entity Card | `EntityCard` | `Kinburg-Nodes/LLM` |
| For Each (Collect) | `KinburgForEachCollect` | `Kinburg-Nodes/loops` |
| For Each (Open) | `KinburgForEachOpen` | `Kinburg-Nodes/loops` |
| Generation Info | `GenerationInfo` | `Kinburg-Nodes/util` |
| Generation Info Filter | `GenerationInfoFilter` | `Kinburg-Nodes/util` |
| Get Accumulator (audio) | `GetAccumAudio` | `Kinburg-Nodes/accumulators` |
| Get Accumulator (captions) | `GetAccumCaptions` | `Kinburg-Nodes/accumulators` |
| Get Accumulator (gen info) | `GetAccumGenInfo` | `Kinburg-Nodes/accumulators` |
| Get Accumulator (images list) | `GetAccumImagesList` | `Kinburg-Nodes/accumulators` |
| Get Accumulator (images) | `GetAccumImages` | `Kinburg-Nodes/accumulators` |
| Get Accumulator (prompts) | `GetAccumPrompts` | `Kinburg-Nodes/accumulators` |
| Get Accumulator (texts) | `GetAccumTexts` | `Kinburg-Nodes/accumulators` |
| Get by Index | `KinburgGetByIndex` | `Kinburg-Nodes/loops` |
| Grammar Presets | `GrammarPresets` | `Kinburg-Nodes/LLM` |
| Group Control 🎚️ | `KinburgGroupControl` | `Kinburg-Nodes/util` |
| Image Batch Insert | `ImageBatchInsert` | `Kinburg-Nodes/image` |
| Image Batch Remove | `ImageBatchRemove` | `Kinburg-Nodes/image` |
| Image Compare (HTML) | `ImageCompareHTML` | `Kinburg-Nodes/image/compare` |
| JSON Extract | `KinburgJSONExtract` | `Kinburg-Nodes/util` |
| Kinburg Live Log 📜 | `KinburgLiveLog` | `Kinburg-Nodes/LLM` |
| LLM Live Log (old id) | `KinburgLLMLog` | `Kinburg-Nodes/LLM` |
| List Insert | `ListInsert` | `Kinburg-Nodes/list` |
| List Output | `KinburgListEmit` | `Kinburg-Nodes/loops` |
| List Remove | `ListRemove` | `Kinburg-Nodes/list` |
| Local LLM (GGUF) | `LocalLLMGGUF` | `Kinburg-Nodes/LLM` |
| Local LLM (server client, text) | `LocalLLMServerText` | `Kinburg-Nodes/LLM` |
| Local LLM Chat (GGUF) | `LocalLLMChatGGUF` | `Kinburg-Nodes/LLM` |
| Local LLM Settings (GGUF) | `LocalLLMSettingsGGUF` | `Kinburg-Nodes/LLM` |
| Lora Trigger Loader | `LoraTriggerLoader` | `Kinburg-Nodes/lora` |
| Lora Unlim Accumulator | `LoraUnlimAccumulator` | `Kinburg-Nodes/lora` |
| Model Capture 📥 | `KinburgModelCapture` | `Kinburg-Nodes/model` |
| Model Select 🎛 | `KinburgModelSelect` | `Kinburg-Nodes/model` |
| Morpheus (Video Sampler) 🌙 | `KinburgMorpheus` | `Kinburg-Nodes/sampling` |
| Morpheus Dream 🌙 | `KinburgMorpheusDream` | `Kinburg-Nodes/sampling` |
| Morpheus Storyboard 🌙 | `KinburgMorpheusStoryboard` | `Kinburg-Nodes/sampling` |
| Ouroboros (Self-Correcting Sampler) 🐍 | `KinburgOuroboros` | `Kinburg-Nodes/sampling` |
| Ouroboros Live Log 🐍📜 | `KinburgOuroborosLog` | `Kinburg-Nodes/sampling` |
| Prompt Presets | `PromptPresets` | `Kinburg-Nodes/prompt` |
| Prompt Variations | `PromptVariations` | `Kinburg-Nodes/prompt` |
| Repeat (Close) | `KinburgRepeatClose` | `Kinburg-Nodes/loops` |
| Repeat (Open) | `KinburgRepeatOpen` | `Kinburg-Nodes/loops` |
| Safetensors -> GGUF (llama.cpp) | `SafetensorsToGGUF` | `Kinburg-Nodes/GGUF` |
| Sampler Settings | `KinburgSamplerSettings` | `Kinburg-Nodes/sampling` |
| Save Song | `KinburgSaveSong` | `Kinburg-Nodes/audio` |
| Send Image to Chat | `LocalLLMChatSendImage` | `Kinburg-Nodes/LLM` |
| Set Accumulator (audio) | `SetAccumAudio` | `Kinburg-Nodes/accumulators` |
| Set Accumulator (captions) | `SetAccumCaptions` | `Kinburg-Nodes/accumulators` |
| Set Accumulator (gen info) | `SetAccumGenInfo` | `Kinburg-Nodes/accumulators` |
| Set Accumulator (images) | `SetAccumImages` | `Kinburg-Nodes/accumulators` |
| Set Accumulator (prompts) | `SetAccumPrompts` | `Kinburg-Nodes/accumulators` |
| Set Accumulator (texts) | `SetAccumTexts` | `Kinburg-Nodes/accumulators` |
| Settings Save 💾 | `KinburgSettingsSave` | `Kinburg-Nodes/model` |
| Settings Select ⚙ | `KinburgSettingsSelect` | `Kinburg-Nodes/model` |
| Show Text (Markdown) | `KinburgShowText` | `Kinburg-Nodes/util` |
| Siren (Music Sampler) 🧜 | `KinburgSirenSampler` | `Kinburg-Nodes/sampling` |
| Siren Cast (Voice Plan) 🧜 | `KinburgSirenCast` | `Kinburg-Nodes/sampling` |
| Siren Compare (Audio) 🧜 | `KinburgSirenCompare` | `Kinburg-Nodes/audio` |
| Siren Scope (Audio → Image) 🧜 | `KinburgSirenScope` | `Kinburg-Nodes/audio` |
| Siren Score (Lyrics → Plan) 🧜 | `KinburgSirenScore` | `Kinburg-Nodes/sampling` |
| Siren Section (Audio Window) 🧜 | `KinburgSirenSection` | `Kinburg-Nodes/sampling` |
| Start Timer | `StartTimer` | `Kinburg-Nodes/util` |
| Stop Timer | `StopTimer` | `Kinburg-Nodes/util` |
| Text Transform | `KinburgTextTransform` | `Kinburg-Nodes/util` |
| Token Counter (GGUF) | `KinburgTokenCounter` | `Kinburg-Nodes/LLM` |
| Unlim Image Batch | `UnlimImageBatch` | `Kinburg-Nodes/image` |
| Unlim Image List | `UnlimImageList` | `Kinburg-Nodes/image` |
| Unlim Text Concat | `UnlimTextConcat` | `Kinburg-Nodes/util` |
| Vision LLM Judge | `VisionLLMJudge` | `Kinburg-Nodes/LLM` |
| Vision Settings (GGUF) | `LocalLLMVisionSettingsGGUF` | `Kinburg-Nodes/LLM` |
| While (Close) | `KinburgWhileClose` | `Kinburg-Nodes/loops` |
| While (Open) | `KinburgWhileOpen` | `Kinburg-Nodes/loops` |

</details>

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

with ComfyUI's own interpreter (`.venv/Scripts/python.exe`). **807 checks in 13 suites, about a
minute** — over `local_llm/` (chat, worker, send-image, lazy guard), `morpheus/` (storyboard, dream
board), `siren/` (cast, score), `audio_sr/`, `lora/`, and three of the `web/*.js` extensions (chat,
dream board, group control). Nothing real is loaded (llama.cpp, H3 and the browser are all stubbed),
so it is a regression net for those paths and **not** a substitute for trying a change in the app.
`tests/README.md` says exactly what is and is not covered.



## 📄 License

MIT — see [LICENSE](LICENSE).