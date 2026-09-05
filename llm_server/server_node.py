"""Local LLM Server — run a llama.cpp server under ComfyUI, so its VRAM can leave and come back.

The pair of nodes here is the ComfyUI face of :mod:`gateway`; read that module's docstring for how
the address SillyTavern talks to is kept alive while the model behind it is killed and reloaded.

* **Local LLM Server** — the configuration and the on switch. Put it in a workflow of its own and
  queue it once; the gateway then lives as long as ComfyUI does.
* **LLM Server Control** — the off switch, small enough to drop into an image workflow. Wire
  anything through its `passthrough` slot and that data dependency puts the unload *before* the
  sampler that needs the VRAM.

Text only, and no generation of its own: this node used to send its own `/v1/chat/completions` and
hand back the answer, which is a different job. Use *Local LLM (GGUF)* for prompts inside a graph.
"""
import os

from ..categories import CAT_LLM
from ..util.anytype import ANY
from . import gateway
from .gateway import (CONTROL_ACTIONS, CONTROL_FREE, CONTROL_LOAD, CONTROL_NODE_ID,
                      CONTROL_STOP, KOBOLDCPP, LLAMA_SERVER, SERVER_NODE_ID)

try:
    from ..local_llm.llm_node import _gguf_dir, _list_models, _resolve_path, PLACEHOLDER
except Exception:  # pragma: no cover — registry scan without ComfyUI on the path
    PLACEHOLDER = "(use model_path field)"

    def _gguf_dir():
        return None

    def _list_models():
        return [PLACEHOLDER]

    def _resolve_path(choice, manual):
        return (manual or "").strip().strip('"').strip("'").strip()


#: The model dropdown's placeholder says "model_path"; the binary picker needs its own wording.
BIN_PLACEHOLDER = "(use server_binary_path field)"

SERVE = "serve"
SERVE_LOAD = "serve + load the model now"
RELOAD = "reload the model"
STOP = "stop everything"
ACTIONS = [SERVE, SERVE_LOAD, RELOAD, STOP]


HELP_TEXT = """# Local LLM Server — quick help

Runs `llama-server` (or `koboldcpp`) **under ComfyUI**, behind a small gateway of its own, so the
model can be thrown out of VRAM for an image and pulled back for the next message without the chat
client ever noticing.

## The two halves
- The **gateway** listens on `listen_host`:`listen_port` for as long as ComfyUI runs. That is the
  address you configure once in SillyTavern and never touch again.
- The **model** is a real server process on a private port. It loads on the first request that
  needs it and is killed whenever the VRAM is wanted elsewhere.

## Wiring SillyTavern
Chat Completion → **Custom (OpenAI-compatible)**, endpoint = this node's `base_url` output
(`http://127.0.0.1:5001/v1`), API key blank. **Connect** works with nothing loaded — the model
list and the health probe are answered by the gateway itself.

## Getting the VRAM back
- Drop **LLM Server Control** into the image workflow with `action` = free vram, and pass any wire
  through it so it runs before the sampler.
- Or leave `free_on_prompt` on: any queued prompt that is not itself about the server unloads
  the model as it is queued — before a single node of it runs. That covers the workflows you
  forgot to edit. Only a graph that wants the model UP (this node, or a Control node set to
  load/status) is spared; a Control node set to free vram does not switch the net off.
- `idle_unload_minutes` frees it after a quiet spell. 0 = never.

## Parameter repair (`fix_params`)
SillyTavern stores list-valued options as JSON *strings*; llama.cpp rejects the whole request when
one is not a real array — most famously `dry_sequence_breakers`. The gateway rewrites those on the
way through. `auto_retry` goes further: a 400 naming a field is retried once without it, and that
field is stripped from then on. `drop_fields` (one name per line) removes fields outright.

## Notes
- `server_binary` must point at the executable — download or build it yourself, nothing is bundled.
- `extra_args` is the server's own command line: `--flash-attn`, `--jinja`, `--model-draft …`.
  Changing it (or the model, or `n_ctx`) reloads the model on the next request.
- `http://127.0.0.1:5001/kinburg/status` reports the state as JSON, `/kinburg/unload` and
  `/kinburg/load` do what they say — handy from a script or a browser tab.
"""


def _server_binaries():
    """Auto-list llama-server / koboldcpp executables under models/llm, RECURSIVELY.

    A llama.cpp release unpacks into its own versioned folder (`llama/llama-b9862-bin-win-cuda…/`)
    and koboldcpp is usually one loose .exe, so a flat listing of models/llm finds neither. Listed
    relative to models/llm with forward slashes, the way the model dropdown does it, so the choice
    stays short on screen and serialises the same on any OS.
    """
    found = []
    d = _gguf_dir()
    if d and os.path.isdir(d):
        for root, _dirs, files in os.walk(d):
            for f in files:
                lo = f.lower()
                if (("llama-server" in lo or "kobold" in lo)
                        and (lo.endswith(".exe") or "." not in lo)):
                    rel = os.path.relpath(os.path.join(root, f), d).replace(os.sep, "/")
                    found.append(rel)
    return [BIN_PLACEHOLDER] + sorted(found, key=str.lower)


def _resolve_binary(choice, manual):
    """The dropdown wins when it is not the placeholder; its value is relative to models/llm."""
    if choice and choice != BIN_PLACEHOLDER:
        d = _gguf_dir()
        picked = os.path.join(d, *choice.split("/")) if d else choice
        if os.path.isfile(picked):
            return picked
    return (manual or "").strip().strip('"').strip("'").strip()


def _split_args(text):
    """`extra_args` as a list. shlex would eat Windows backslashes, so quotes are handled by hand
    and everything else splits on whitespace — which is what a llama.cpp command line needs."""
    out, cur, quote = [], "", ""
    for ch in (text or "").replace("\n", " "):
        if quote:
            if ch == quote:
                quote = ""
            else:
                cur += ch
        elif ch in "\"'":
            quote = ch
        elif ch.isspace():
            if cur:
                out.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


class LocalLLMServer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (ACTIONS, {"default": SERVE, "tooltip": "serve = start the gateway and wait for the first request. 'load now' also loads the model straight away. 'reload' kills a loaded model so the next request starts it fresh. 'stop everything' shuts the gateway down too."}),
                "flavour": ([LLAMA_SERVER, KOBOLDCPP], {"default": LLAMA_SERVER, "tooltip": "Which server is being launched — it only picks the flag names (-m / --model, --ctx-size / --contextsize …)."}),
                "server_binary": (_server_binaries(), {"tooltip": "Executable to launch. Auto-lists every llama-server / koboldcpp found anywhere under ComfyUI/models/llm, subfolders included; otherwise pick the placeholder and type a path below. Download or build it yourself — not bundled."}),
                "server_binary_path": ("STRING", {"default": "", "tooltip": "Full path to the server executable when the dropdown is the placeholder."}),
                "model": (_list_models(), {"tooltip": "The .gguf from ComfyUI/models/llm to serve. Placeholder = type a path in model_path."}),
                "model_path": ("STRING", {"default": "", "tooltip": "Full path to the .gguf when 'model' is the placeholder."}),
                "n_ctx": ("INT", {"default": 8192, "min": 256, "max": 1048576, "step": 256, "tooltip": "Context size (--ctx-size / --contextsize). A long chat plus a character card eats this fast."}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "GPU layers (-ngl / --gpulayers). -1 = all."}),
                "extra_args": ("STRING", {"multiline": True, "default": "", "tooltip": "Raw flags appended to the server's command line, e.g. --flash-attn --jinja --model-draft path. Changing this reloads the model."}),
                "listen_host": ("STRING", {"default": "127.0.0.1", "tooltip": "Where the gateway listens. 127.0.0.1 = this machine only; 0.0.0.0 to reach it from another box on the LAN."}),
                "listen_port": ("INT", {"default": 5001, "min": 1, "max": 65535, "tooltip": "The port SillyTavern is pointed at. It stays open while ComfyUI runs, model or no model."}),
                "autoload": ("BOOLEAN", {"default": True, "tooltip": "Load the model on the first request that needs one. Off = a request with nothing loaded gets a clear error instead."}),
                "free_on_prompt": ("BOOLEAN", {"default": True, "tooltip": "Unload the model whenever a ComfyUI prompt is queued that contains neither of these two nodes — the safety net for image workflows you did not edit."}),
                "idle_unload_minutes": ("INT", {"default": 0, "min": 0, "max": 1440, "tooltip": "Unload after this many minutes with no request. 0 = never."}),
                "unload_comfy_models": ("BOOLEAN", {"default": True, "tooltip": "Free ComfyUI's own models from VRAM before loading the LLM."}),
                "fix_params": ("BOOLEAN", {"default": True, "tooltip": "Repair the request bodies llama.cpp refuses — SillyTavern sends dry_sequence_breakers and friends as JSON strings where an array is required."}),
                "auto_retry": ("BOOLEAN", {"default": True, "tooltip": "On a 400 that names a field, drop that field and retry once, then strip it from later requests. Keeps one unknown option from breaking every message."}),
                "drop_fields": ("STRING", {"multiline": True, "default": "", "tooltip": "Field names to strip from every request, one per line. The manual version of auto_retry."}),
                "backend_port": ("INT", {"default": 0, "min": 0, "max": 65535, "tooltip": "Private port for the model server. 0 = pick a free one."}),
                "ready_path": ("STRING", {"default": "", "tooltip": "Path polled until the model is ready. Blank = per-flavour default (llama-server /health, koboldcpp /v1/models)."}),
                "startup_timeout": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 10, "tooltip": "How long a load may take before it is called a failure."}),
                "request_timeout": ("INT", {"default": 900, "min": 10, "max": 7200, "step": 10, "tooltip": "How long one forwarded request may take."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "base_url", "server_log", "help")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT_LLM

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute: queueing this workflow is how you (re)start a gateway that died, and a
        # cached hit would do nothing at all.
        return float("NaN")

    def run(self, action, flavour, server_binary, server_binary_path, model, model_path,
            n_ctx, n_gpu_layers, extra_args, listen_host, listen_port, autoload, free_on_prompt,
            idle_unload_minutes, unload_comfy_models, fix_params, auto_retry, drop_fields,
            backend_port, ready_path, startup_timeout, request_timeout):

        if action == STOP:
            gateway.shutdown("asked by the Local LLM Server node")
            return ("gateway stopped", "", gateway.log_tail(), HELP_TEXT)

        binary = _resolve_binary(server_binary, server_binary_path)
        if not binary or not os.path.isfile(binary):
            return (f"[ERROR] server binary not found: {binary or '(none selected)'}",
                    "", gateway.log_tail(), HELP_TEXT)
        resolved = _resolve_path(model, model_path)
        if not resolved or not os.path.isfile(resolved):
            return (f"[ERROR] model file not found: {resolved or '(none selected)'}",
                    "", gateway.log_tail(), HELP_TEXT)

        gateway.configure(
            flavour=flavour, binary=binary, model=resolved,
            n_ctx=int(n_ctx), n_gpu_layers=int(n_gpu_layers), extra=tuple(_split_args(extra_args)),
            backend_port=int(backend_port), autoload=bool(autoload),
            free_on_prompt=bool(free_on_prompt), idle_minutes=int(idle_unload_minutes),
            unload_comfy=bool(unload_comfy_models), fix_params=bool(fix_params),
            auto_retry=bool(auto_retry),
            drop_fields=tuple(s.strip() for s in (drop_fields or "").splitlines() if s.strip()),
            ready_path=(ready_path or "").strip(),
            startup_timeout=int(startup_timeout), request_timeout=int(request_timeout),
        )

        host = (listen_host or "127.0.0.1").strip() or "127.0.0.1"
        ok, err = gateway.serve(host, int(listen_port))
        if not ok:
            return (f"[ERROR] {err}", "", gateway.log_tail(), HELP_TEXT)

        reach = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        base_url = f"http://{reach}:{int(listen_port)}/v1"

        if action == RELOAD:
            gateway.stop_model("reload asked by the Local LLM Server node")
        if action in (SERVE_LOAD, RELOAD):
            # Inside a running prompt, so it must not wait for ComfyUI to go idle — that is us.
            ok, err = gateway.ensure_model(force=True, wait_for_comfy=False)
            if not ok:
                return (f"[ERROR] {err}", base_url, gateway.log_tail(), HELP_TEXT)

        return (gateway.status_text(), base_url, gateway.log_tail(), HELP_TEXT)


class LLMServerControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (CONTROL_ACTIONS, {"default": CONTROL_FREE, "tooltip": "free vram = kill the model, keep the gateway listening (the next chat message loads it again). load = load it now. stop the gateway = close the port too."}),
            },
            "optional": {
                "passthrough": (ANY, {"tooltip": "Any value — it comes back out unchanged. Wiring it through here is what makes ComfyUI run this node BEFORE whatever the wire feeds."}),
            },
        }

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("passthrough", "status")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT_LLM

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def run(self, action, passthrough=None):
        if action == CONTROL_FREE:
            gateway.stop_model("asked by the LLM Server Control node")
        elif action == CONTROL_LOAD:
            ok, err = gateway.ensure_model(force=True, wait_for_comfy=False)
            if not ok:
                return (passthrough, f"[ERROR] {err}")
        elif action == CONTROL_STOP:
            gateway.shutdown("asked by the LLM Server Control node")
        return (passthrough, gateway.status_text())


NODE_CLASS_MAPPINGS = {SERVER_NODE_ID: LocalLLMServer, CONTROL_NODE_ID: LLMServerControl}
NODE_DISPLAY_NAME_MAPPINGS = {SERVER_NODE_ID: "Local LLM Server",
                              CONTROL_NODE_ID: "LLM Server Control"}
