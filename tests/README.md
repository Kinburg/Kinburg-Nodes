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
| `test_worker.py` | the real `gguf_worker` main loop over a **stubbed `llama_cpp`** — the projector attach/release swap, load-signature stability, chat-template handling, grammar streaming |
| `test_send_image.py` | `Send Image to Chat` — megapixel downscale, the content-hash filename, the payload, and `attachments.discard()` including everything it must refuse to delete |
| `test_dream_board.py` | `Dream Board` — the whole "pictures define the shots" rule, the outputs, the `MORPHEUS_SHOT` chain |
| `test_storyboard.py` | `Morpheus Storyboard.write()` over a **faked LLM** — filling a wired chain in place vs appending, per-shot keyframes/durations/links, the beats override |
| `test_lora_triggers.py` | `_with_triggers` — where trigger words land in a MiniMax prompt, and that they never land in `[Negative]` |
| `js/build_chat.mjs` | `web/chat_llm.js` + `web/chat_send.js` — the attachment tray, paste/upload, 🗑 Clear and its file cleanup, `sendToChat` |
| `js/build_dream_board.mjs` | `web/dream_board.js` — the JS shot-derivation port (same fixtures as the Python one), the snapshot pull, ref holding |

A JS suite is *assembled*: `build_*.mjs` concatenates `js/stubs.mjs`, the extension file with its
`import` lines stripped, and the assertions, then writes `run_*.mjs` and `run.py` executes it. That
is how a test reaches module-private functions and can call the real `setup()`, `onExecuted` and
widget callbacks. The generated `run_*.mjs` are build artifacts and are git-ignored.

## What is NOT here — read this before trusting a green run

- **Most of the pack.** These suites cover `local_llm/`, `morpheus/` and three `web/*.js` files.
  Chimera, Siren, Image Compare, Ouroboros, Vision Judge, the presets nodes, `util/`, `loops/` and
  the rest — the majority of the 87 nodes — have no tests at all.
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
