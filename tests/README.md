# tests

Run from the pack root:

```bash
python tests/run.py
```

Use ComfyUI's own interpreter — `.venv/Scripts/python.exe` on Windows, `.venv/bin/python`
elsewhere. `run.py` finds it by itself when you invoke it with something else, but the suites
import torch, PIL and comfy, so a bare system `python` will not do. Node.js is needed for the two
JS suites; without it they are skipped with a note rather than failing.

```bash
python tests/run.py chat            # only the suites whose name contains "chat"
python tests/run.py board -v        # -v prints every check, not just the failures
```

Non-zero exit on any failure, so this drops into a pre-commit hook as-is.

## What is here

| suite | covers |
|---|---|
| `test_chat.py` | `LocalLLMChatGGUF.run()` — personas 1..6, the Approve gate, attachment refs and the `[image]` markers that stand in for them, path resolution and its traversal guard |
| `test_worker.py` | the real `gguf_worker` main loop over a **stubbed `llama_cpp`** — the projector attach/release swap, load-signature stability, chat-template handling (incl. the reasoning variables bound onto the handler on the text, override, vision and resume paths), grammar streaming, and ⏹ stopping a reply mid-stream (the stdin reader thread, the partial text it keeps, no bleed into the next request) |
| `test_gguf_info.py` | reading a GGUF's own answers out of its header — the hand-rolled KV parser against **synthesised GGUF bytes** (a real key placed after a skipped 9000-entry array, so an off-by-one in the skip fails loudly), and the chat-template probe against templates that behave like the real families: Qwen-style effort validation (only what it accepts survives, its default is found), Gemma-style `enable_thinking` with the opposite polarity and no effort at all, an mmproj, a template that refuses a system turn, no template, and a file that is not a GGUF |
| `test_llm_reasoning.py` | the Settings node's `enable_thinking` / `reasoning_effort` — what each option puts in the request (and that `model default` puts nothing), that neither touches the load signature, that they are appended last so saved workflows keep their widget values, and the reasoning split for templates that prefill the opening `<think>` |
| `test_send_image.py` | `Send Image to Chat` — megapixel downscale, the content-hash filename, the payload, and `attachments.discard()` including everything it must refuse to delete |
| `test_send_image_log.py` | `Send Image to Live Log 📜` — which frames of a batch go over the websocket (and that `all` is capped), the block label it builds, the encoder-refused warning, and the channel + node id the event carries |
| `test_dream_board.py` | `Dream Board` — the whole "pictures define the shots" rule, the outputs, the `MORPHEUS_SHOT` chain |
| `test_storyboard.py` | `Morpheus Storyboard.write()` over a **faked LLM** — filling a wired chain in place vs appending, per-shot keyframes/durations/links, the beats override |
| `test_phantas_timing.py` | Phantas' clock — the fifteen legal shot lengths (checked against ComfyUI's own `align_frame_count`), the three counting units, and laying a target length onto shots by weight to within half a grid step |
| `test_phantas_board.py` | `Phantas Storyboard.write()` over a **faked LLM** — the counting, the generated plan grammar, the bible stamped byte-for-byte, causal keys, the prompt override, and the over-determined-target error |
| `test_phantas.py` | `Phantas.render()` over a **stubbed sampler** with a real disk cache — the shot chain and its shared boundary frames, the anchor rules, both reference mechanisms, `redo`, and that a cached frame is bit-identical to the fresh one |
| `test_diskcache.py` | `util/diskcache.py` and the Morpheus adapter over it — causal keys, content hashes, the fp16 store round-trip, LRU pruning |
| `test_lora_triggers.py` | `_with_triggers` — where trigger words land in a MiniMax prompt, and that they never land in `[Negative]` |
| `test_model_triggers.py` | the Model Library's LoRA trigger words end to end — harvesting them out of a captured recipe (order, dedupe, a LoRA turned off), the store's keep-on-None rule, Model Select's prompt join / encode / zeroed negative, that no prompt means no encode at all, that a resident bundle is not unloaded on a prompt edit, and Capture's starting preset. Also the store's **write safety** (an unreadable file refuses every write and leaves the file alone; each write rolls a `.bak`) and **family** rename / delete across every holder at once. Runs against a temp store, never the real library |
| `test_save_clip.py` | `Save Clip`'s timeline — the plan read for its proportions and scaled onto the audio (so a slideshow cannot drift off its song), the three slide layouts (subdividing, cycling, the chorus shot coming back, neighbours on the same picture fused), the crossfade centred on the cut and clamped to its shorter neighbour, the Ken Burns moves, and the .srt matched to sections by label. No torch, no PyAV, no PIL — the encoder is ComfyUI's own recipe and fails loudly on the first frame |
| `test_card_presets.py` | the Card Presets store as the card library's editor uses it — the field schema it reads off the card nodes themselves (order, labels, the multiline flag, the save-only fields left out), renaming in one write (no ghost left behind, tags inherited, a stale `old_name` degrading to a create), and the preview renderer over unsaved values. Runs against a temp store, never the real library |
| `test_siren_cast.py` | `Siren Cast` — plan parsing (lengths in seconds / bars / `m:ss`, voices, the 4th column), the per-section caption and its negative under each `guidance` mode, and that `seconds` matches the codes actually written |
| `test_siren_score.py` | `Siren Score` — section and voice detection off a lyric sheet, syllable counting, the backwards length split (Hamilton on 2-bar units, floors, `tail_bars`, `pad_placement`) and the rate it reports |
| `test_lazy_guard.py` | `Show Text`'s `use_saved_text` — `check_lazy_status` must not evaluate the input when the toggle is on, so the upstream never runs |
| `test_audio_sr.py` | `Audio SR` — the chunk plan and its pulled-back tail, the periodic-Hann crossfade summing to 1, mid/side vs sum-to-mono, `match_level` below the roll-off, and `defuse_lazy_modules()` |
| `test_docs.py` | `tools/gen_readme_index.py --check` — the README index matches the registered nodes, every node folder has a `docs/` section, every local link and anchor resolves, and every backticked term is a real input/output |
| `js/build_chat.mjs` | `web/chat_llm.js` + `web/chat_send.js` — the attachment tray, paste/upload, 🗑 Clear and its file cleanup, `sendToChat` |
| `js/build_dream_board.mjs` | `web/dream_board.js` — the JS shot-derivation port (same fixtures as the Python one), the snapshot pull, ref holding |
| `js/build_model_presets.mjs` | `web/model_presets.js` — the saved-workflow migration (Model Select's and Settings Select's outputs re-ordered by NAME so any older layout works, width/height wires dropped from the link table, the target inputs and the slots alike, `widgets_values` anchored from the END per node, and idempotence) plus the Library dialog's two reachability sections: shared presets listed with no models in the library at all, and a family's armed delete posting once for every holder |
| `js/build_card_presets.mjs` | `web/card_presets.js` — the **card library** dialog against a stubbed backend that behaves like the real routes: the rename following every node that picked the card (and no node that merely owns a `preset` widget), the values merge that keeps the fields the editor never drew, the editor end to end (opens on the card, previews the block, a new card taking its library name from Name until you type one), search / tag narrowing / focus-first ordering, and the armed delete |
| `js/build_group_control.mjs` | `web/group_control.js` — the **group links** engine: the pure resolver (polarity, one-of, chains, contradictory cycles) and the live side against a hand-built graph — toggling, changes made outside the panel, pause, bulk overrides, and the nested-group snapshot/restore |

A JS suite is *assembled*: `build_*.mjs` concatenates `js/stubs.mjs`, the extension file with its
`import` lines stripped, and the assertions, then writes `run_*.mjs` and `run.py` executes it. That
is how a test reaches module-private functions and can call the real `setup()`, `onExecuted` and
widget callbacks. The generated `run_*.mjs` are build artifacts and are git-ignored.

## What is NOT here — read this before trusting a green run

- **Most of the pack.** These suites cover `local_llm/`, `morpheus/`, the pure logic of `siren/`
  (Cast's plan parsing and Score's length maths, not a sampling run), `audio_sr/`, `lora/` and three
  `web/*.js` files. Chimera, Image Compare, Ouroboros, Vision Judge, the model/prompt presets nodes,
  `util/`, `loops/`, `accumulators/` and the rest — well over half of the 90 nodes — have no tests at
  all. The `docs` suite is not coverage either: it checks what the README *says* about a node, never
  what the node does.
- **Nothing real is loaded.** No llama.cpp, no H3, no diffusion model, no VAE. `llama_cpp` is a
  stub, the LLM call in the storyboard suite is a stub, and the H3 frame grid is faked in the Dream
  Board suite. These prove control flow and data shape, never that a model behaves.
- **No browser.** The JS suites run against a hand-written DOM stub, so they verify logic and
  wiring, not rendering, layout or ComfyUI integration. Two live bugs this year — a stale widget
  read and a missing delete affordance — were invisible here and only turned up in the real app.
- **No graph execution.** Nothing checks that a node's declared inputs match how ComfyUI actually
  calls it, beyond the signature checks in the trigger suite.

So: a regression net for the LLM / chat / Morpheus text path. Still test in the real app.

## Adding to a suite

Fixtures are synthetic and must stay that way — no real conversation content, ever. `_env.py` has
the shared plumbing: `load_pack()` for the whole pack, `fake_package()` + `load_module()` to load
one module without dragging its neighbours in, and `Checker` for the `check(label, cond, extra)`
lines every suite prints.
