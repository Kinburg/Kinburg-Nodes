# 🧠 LLM & Vision Systems

<!-- index-order: 1 -->

[← back to the node index](../README.md#-node-index)

---

## 🤖 `local_llm/` — Local LLM (GGUF) & Live Logging

> **System Purpose & Overview**  
> Run GGUF LLMs directly inside ComfyUI with **guaranteed VRAM unloading** via a separate worker process. Features real-time streaming progress, token counts, reasoning/answer splitting (`<think>` tags or custom markers), vision model support (mmproj), structured output (JSON/GBNF/Ideogram), and recursive GGUF scanning.

### `Local LLM (GGUF)` — *Main GGUF inference & vision node*

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

For an OpenAI-compatible **server** instead of the Python binding, see
[`llm_server/`](#-llm_server--local-llm-server-client-text) below.

### `Local LLM Chat (GGUF) & Send Image` — *Interactive multi-turn LLM chat interface inside ComfyUI*

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

**Sending a picture.** **Ctrl+V**, drop a file on the chat window, or hit the **📎** in the corner
of the input box. Thumbnails queue in a tray above the input (**✕** takes one off); they go out with
your next message and stay in its bubble for good — click one to open it full size. Works with an
empty message: a picture on its own is a perfectly good "look at this".

Two things make this different from the `image` input. First, it is **not a graph link** — the
picture is uploaded to `input/kinburg_chat/` and referenced by name, so 📨 Send never re-runs an
image branch to fetch it (an `image` wired in from a sampler would regenerate on *every* message
unless its seed is fixed). Second, the pixels reach the model on **that turn only**. From the next
turn the picture is a text marker — `[image]`, or `[image: …]` once something fills the caption —
so a long conversation with pictures in it costs no more context than a conversation without. The
chat still shows every one of them; only the model's copy degrades to a line of text.

Needs an `mmproj` on the active persona's Settings node, same as any vision run — and mixing
picture turns with plain ones is cheap: the projector is attached per request rather than at load
time, so the model stays put and only the (much smaller) clip is loaded and released around it.

**`Send Image to Chat`** does the same thing for a picture your workflow just generated, so you
never have to copy one out of a preview by hand. Wire an `IMAGE` in (it passes straight through, so
the node sits inline), and:

- **`send_as`** — *me (user)* drops it in the chat's tray, exactly as if you had pasted it: pixels
  go to the model with your next message. A **persona** instead hangs it on that persona's most
  recent bubble, so it reads as though they sent it. The model is not shown those pixels — llama.cpp
  only takes images on user turns, and a persona has no need to study a photo it supposedly took —
  and by default the picture leaves nothing in the context either. That is usually right: the
  picture came from something the persona had just described, so a marker would say it twice. Turn
  on **`note_in_context`** (with a `caption`) when you want the conversation to record it.
- **`when`** — *on button press* saves the picture and waits for the **📌** on the node, so you can
  look at the result and re-roll before committing. *every run* pushes as soon as the node executes.
  Either way the filename is a hash of the pixels, so re-running a branch that produced the same
  picture pushes the same reference and the chat recognises it instead of stacking duplicates.
  📌 reads `send_as`, `caption`, `shot` and `note_in_context` **when you press it** — deciding who a
  picture comes from is something you do after looking at it, so changing them post-generation
  works without re-running anything. Only `megapixels` needs a re-run; it changes the saved file.
- **`caption`** — one or two sentences of plain prose: this is what the model reads about the
  picture once the pixels are gone. Not the generation prompt — a paragraph of comma-separated tags
  sitting in the conversation teaches the persona to write in comma-separated tags. If a "camera"
  persona writes both, split them: one line for the sampler, one for here.
- **`megapixels`** downscales the copy that goes to the chat (0 = full size); **`shot`** is an
  optional keyframe label kept with the picture, so a chat can be read back as a storyboard.
- **`→ chat`** picks the target chat window; leave it on auto when there is only one.

Like pasting, none of this is a graph link, so it never causes 📨 Send to re-run a sampler.

**Taking one back.** Hover a picture in a bubble for a **✕** — sent by the wrong persona, or by
accident, and it comes straight back out without touching the reply it was hanging on. A bubble
that existed only to carry it disappears with it. 🗑 on a whole message, and 🗑 Clear on the whole
chat, take their pictures too; Clear counts them before asking.

Removing a picture deletes its file, unless something in the graph is still showing it — another
bubble, another chat node, or a `Morpheus Dream Board 🌙` whose snapshot still names it (a picture in two
places is one file, since the name is a hash of its pixels).
Deletion is limited to `input/kinburg_chat/` — your own `LoadImage` sources in the input folder are
never touched. And a file removed by mistake comes back by re-running the branch that made it: the
hash, and therefore the name, is the same.

**Stopping a reply.** While one is being written its bubble carries **⏹** and **✕**, and they mean
different things. **⏹** stops the model between tokens and **keeps what it has written**: the
partial text lands in the history like any other reply, the meter says `⏹ stopped`, and Send with an
empty box carries on from it — the same continuation a reply cut off by `max_tokens` gets. **✕**
walks away from the turn and discards it. Stopping does not kill the worker or unload the model, so
the next turn starts immediately.

**Editing the conversation.** Hover any bubble — yours or the model's — for **⧉ copy · ✎ edit ·
↻ resend · 🗑 delete**. ✎ swaps the bubble for an inline textarea sized to the message it holds
(Esc cancels, Ctrl+Enter saves). ↻ replays exactly the turn that produced the message, dropping
everything below it — a normal question/answer pair goes back to your question, while a persona's
no-user-message turn replays just that one reply. All of it is plain surgery on the stored history, so the
next turn simply sees the conversation you left behind. Everything is disabled while a reply is in
flight; if a run dies before reaching the node, **✕** on the live bubble unsticks it.

**Personas.** Six inputs — **`persona_1..6`** — each take a *whole* `Local LLM Settings (GGUF)`
bundle, so a persona brings its own **model, sampling and system prompt**, not just a different
prompt. **`persona_1` is the node's config**: wire only that one and this is an ordinary chat node
with no chip row. Wire a second and a chip row appears, one chip per persona. Clicking a chip only
**selects** it — 📨 Send is the sole trigger — and the active persona's bundle becomes the config
for that turn. All personas share one history, so a prompt-writer sees the whole discussion (as far
back as its window reaches); when it speaks, the other personas' replies are prefixed with their
name (`[Order manager]: …`) so it doesn't mistake them for its own past turns. Bubbles are labelled
with their persona once there's more than one.

The **⚙** chip sets, per persona: the **chip label** (defaults to the title of the wired-in node,
but only if you renamed it — stock Settings nodes all read the same), an optional **trigger**
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

### `Token Counter (GGUF)` — *Token counting & context estimation utility*

**`Token Counter (GGUF)`** (in the `local_llm/` package) counts how many tokens a text is under a
model's tokenizer, **without generating** — via the same worker but a **vocab-only** load (just
the tokenizer, no weights, no VRAM; it also reuses an already-loaded model when the path matches,
so counting never disturbs generation). Wire a **`config`** (only its model is used) and a
**`text`**; outputs `token_count`, `char_count`, and an `info` line (e.g. `42 tokens · 180 chars`).
Handy for budgeting a prompt against a model's `n_ctx`. Category `Kinburg-Nodes/LLM`.

### `Context Sizer (GGUF)` — *Context size calculator for GGUF models*

**`Context Sizer (GGUF)`** (in `local_llm/`) measures how many tokens a request actually needs and
suggests an `n_ctx`. Wire a **`config`**, your prompt text into the auto-growing **`text_*`** inputs,
and optionally an **`image`** (a batch is fine — it sizes for the largest). Counting is **lean** — no
full-model load: text via the vocab-only tokenizer, images via `mtmd_tokenize` on the clip/mmproj
only (no LLM weights, no forward pass), so image tokens — which depend on the model + resolution —
are counted for real (with a safe fallback to a full-model prefill if a build can't tokenize
weight-free). Outputs `text_tokens` / `image_tokens` / `total_tokens` / **`suggested_n_ctx`** (=
input + the config's `max_tokens` + a `margin`, rounded up) / `info`. Sizing `n_ctx` close to what a
request uses saves VRAM (the KV cache shrinks with it). Category `Kinburg-Nodes/LLM`.

All four LLM nodes that take a config bundle (**Local LLM (GGUF)**,
[**Vision LLM Judge**](#-vision_judge--vision-llm-judge--criteria-builder), **Token Counter**,
**Context Sizer**) carry a per-node **`unload_after_run`** selector — *config default* follows the Settings
node's `unload_llm_after_run`, *unload after run* frees VRAM after just this node (e.g. a different
model runs next), *keep loaded* stays warm — so one node can free VRAM without touching the others
on the same config.

---

## 🌐 `llm_server/` — Local LLM (server client, text)

> **System Purpose & Overview**  
> Drive an OpenAI-compatible LLM **server** (llama-server, koboldcpp, or any already-running one) instead of the in-process Python binding, so the server's own full command line is available.

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

---

## 📜 `context/` — Character Card, Entity Card, Context Collector

> **System Purpose & Overview**  
> Feed LLM nodes structured reference material (Character & Entity Cards) so they weave named subjects into expanded image prompts, with built-in voice tag integration for music generation.

Feed an LLM node reference material so it weaves named subjects into an expanded image prompt.
**`Character Card`** has fields (name, gender, age, eyes, hair, build, outfit, distinctive
features, free-form notes…) and outputs one tidy Markdown block, **skipping every empty
field**. **`Entity Card`** is its free-form sibling for non-people (an object, place, faction…) —
a name + a description. Both cards carry a **`save_preset_as`** field (+ optional comma-separated
**`tags`**): type a name and run to save the card to a reusable library — it saves at run time, so
it captures whatever's filled in, whether **typed or wired in** (e.g. from a photo description);
reuse it via **Card Presets** (below). To save an LLM's JSON card in one step instead, use **Card
Save**.
Character Card also describes a **singing** voice — **`voice_tags`** (a music caption fragment:
*"male lead vocal, raspy baritone, close-mic"*) and **`voice_notes`** (prose for the LLM: range,
habits, what they never do). Both go into the Markdown block, and a second **`voice`** output
carries `voice_tags` alone to **Siren Cast**, which is what makes a band member one card: their
looks reach the cover-art prompt, their voice reaches the song, and the same card tells the lyrics
LLM who it's writing for. Only `voice_tags` reaches AceStep — pouring "brown eyes, navy dress" into
a music caption just dilutes it. **Card Presets** emits the same `voice`, so a saved member drives
the music too (a character saved before these fields existed simply comes back with empty tags).
**`Context Collector`** gathers any number of cards / text chunks (auto-growing
`item_N` inputs, empties skipped) under a `title` and wraps them in a delimited block —
`<context>…</context>`, a custom tag, a Markdown heading, or none — so the model can tell
reference data from the instruction. Wire its `context` output into an LLM node's new
**`context`** input (present on all three LLM nodes; it's appended to the system prompt). Then a
prompt like *"Vasya and Kolya drink tea in a cafe"* comes back expanded with each character's
looks. Note: the diffusion model still has its own limits binding attributes across multiple
people. Category `Kinburg-Nodes/LLM/context`.

---

## 👁️ `vision_judge/` — Vision LLM Judge & Criteria Builder

<!-- packages: vision_judge, criteria_presets -->

> **System Purpose & Overview**  
> Score a batch or list of images using a Vision GGUF against custom rubrics or multi-criteria definitions, returning structured JSON verdicts with guaranteed formatting via GBNF grammars.

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
both **Vision LLM Judge** and **Ouroboros Critic Settings 🐍** — right-click their `criteria` field →
**Convert widget to input** and wire it in (empty output = single overall score). The catalog is
`criteria_presets/catalog.json`; drop a **`catalog.user.json`** next to it (same shape) to add your
own criteria without editing the shipped file (it survives a git pull). Category `Kinburg-Nodes/LLM/presets`.

---

## 🎴 `card_presets/` — Card Save & Card Presets

> **System Purpose & Overview**  
> Save and recall Character and Entity card presets dynamically from disk.

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
same photo every time. Category `Kinburg-Nodes/LLM/presets`.

---

## 📝 `grammar_presets/` — Grammar Presets

> **System Purpose & Overview**  
> Pre-built GBNF grammar selection library for forcing structured LLM outputs (JSON, markdown, prompt schemas).

**`Grammar Presets`** is a dropdown of **GBNF grammars** → one `grammar` STRING output. It ships
with templates that force an LLM's output into a fixed shape — **Character Card (JSON)**,
**Entity Card (JSON)**, **Siren Song Config (text)** and **Siren Voice Plan (table)** — and you can
add / edit / delete your own (**➕ Add grammar**, persisted on disk). Wire the `grammar` output into
a **Local LLM (GGUF)** node's `grammar_override` input, feed that node a **photo** + a short prompt
("fill the card from this image"), and the vision model returns exactly that structure straight from
the picture — no multi-image-context gymnastics. Pair it with **Card Save** (below) to drop that
generated card straight into your library.

**Song Config** is the LLM pass that turns a brief into Siren Cast's inputs: a caption block (`*Genre:*` / `*Instruments:*`), a blank
line, then the metas as `key: value`. `keyscale` / `language` / `timesignature` are pinned to the
exact values Siren Cast's combos accept — a free character class cheerfully writes `C sharp minor`
or `ua` (Ukrainian is `uk`), and neither is in the list — and `bpm` is held to 60–249. It has **no
vocals line on purpose**: per-section voices are the plan's job, and naming the timbres in the
caption as well drags every section towards their average. **Voice Plan** runs *after* the lyrics,
because sizing a section needs the finished text; lengths are constrained to musical **bar** counts
and labels to the usual section names, so they line up with the `[Verse 1 - …]` markers. Give that
pass the same Context Collector block with the band in it — the names in the table have to match the
`name` on the Character Cards.

Voice Plan bounds its rows (`row{3,16}`) and **requires a closing `END` line**, and both are there
for the same reason: a grammar whose repetition is open-ended never *forces* the model to finish, it
only *permits* EOS — and EOS is a low-probability option that `top_p` / `min_p` / `top_k` prune away,
after which the sole legal continuation is another row and the model writes rows until `max_tokens`.
So for a grammar-constrained pass, **turn the truncation samplers off** (`top_p 1.0`, `top_k 0`,
`min_p 0`) and `repeat_penalty` with them — a table format *mandates* repeated tokens, so penalising
them fights the grammar. Low `temperature` is enough to keep it disciplined. Putting `END` in the
node's `stop` field as well makes stopping independent of EOS entirely. `Siren Cast` treats a bare
`END` line as end-of-table and silently drops anything after it. Category `Kinburg-Nodes/LLM/presets`.

---

## 📖 `show_text/` — Show Text (Markdown)

> **System Purpose & Overview**  
> Display Markdown and raw text formatted outputs cleanly on ComfyUI node surfaces.

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

---

## ⚙️ `gguf_convert/` — Safetensors → GGUF Converters

> **System Purpose & Overview**  
> Convert safetensors model weights directly to GGUF format inside ComfyUI for LLMs and Diffusion models.

Turn `.safetensors` weights into `.gguf` from inside ComfyUI. Two nodes (category
`Kinburg-Nodes/LLM/GGUF`), one per model family, each with the same three outputs — **`gguf_path`**
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

---

[← back to the node index](../README.md#-node-index)
