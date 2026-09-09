"""Local LLM Server — run a llama.cpp server under ComfyUI, so its VRAM can leave and come back.

The nodes here are the ComfyUI face of :mod:`gateway`; read that module's docstring for how the
address SillyTavern talks to is kept alive while the models behind it are killed and reloaded, and
:mod:`options` for how each setting is spelled by each backend.

* **Local LLM Server** — the configuration and the on switch. Put it in a workflow of its own and
  queue it once; the gateway then lives as long as ComfyUI does.
* **LLM Server Control** — the off switch, small enough to drop into an image workflow. Wire
  anything through its `passthrough` slot and that data dependency puts the unload *before* the
  sampler that needs the VRAM.
* **LLM Server Draft / Vision / Embeddings** — optional side models. Each is a small node that
  plugs into the server node, so the second model's settings only exist when you use one.

No generation of its own: this node used to send its own `/v1/chat/completions` and hand back the
answer, which is a different job. Use *Local LLM (GGUF)* for prompts inside a graph.
"""
import os

from ..categories import CAT_LLM
from ..util.anytype import ANY
from . import gateway, options
from .gateway import (CONTROL_ACTIONS, CONTROL_FREE, CONTROL_LOAD, CONTROL_NODE_ID,
                      CONTROL_STOP, KOBOLDCPP, LLAMA_SERVER, SERVER_NODE_ID)

try:
    from ..local_llm.llm_node import (
        _gguf_dir, _list_models, _list_mmproj, _resolve_path, PLACEHOLDER,
        REASONING_DEFAULT, REASONING_EFFORTS, THINKING_MODES,
    )
except Exception:  # pragma: no cover — registry scan without ComfyUI on the path
    PLACEHOLDER = "(use model_path field)"
    REASONING_DEFAULT = "model default"
    THINKING_MODES = [REASONING_DEFAULT, "on", "off"]
    REASONING_EFFORTS = [REASONING_DEFAULT, "xhigh", "medium", "low", "custom"]

    def _gguf_dir():
        return None

    def _list_models():
        return [PLACEHOLDER]

    def _list_mmproj():
        return [PLACEHOLDER]

    def _resolve_path(choice, manual):
        return (manual or "").strip().strip('"').strip("'").strip()


#: The model dropdown's placeholder says "model_path"; the binary picker needs its own wording.
BIN_PLACEHOLDER = "(use server_binary_path field)"

#: The optional side-model bundles. Named types so a Draft node cannot be wired into the Vision
#: slot by accident.
DRAFT_CONFIG = "KINBURG_LLM_SERVER_DRAFT"
VISION_CONFIG = "KINBURG_LLM_SERVER_VISION"
EMBED_CONFIG = "KINBURG_LLM_SERVER_EMBED"

SERVE = "serve"
SERVE_LOAD = "serve + load the model now"
RELOAD = "reload the model"
STOP = "stop everything"
ACTIONS = [SERVE, SERVE_LOAD, RELOAD, STOP]

SPEC_NONE = "off"
SPEC_TYPES = [SPEC_NONE] + options.SPEC_TYPES

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

## VRAM levers, in the order worth trying
1. `cpu_moe_layers` — on a Mixture-of-Experts model this moves the experts of the first N layers
   to system RAM and is by far the biggest saving per point of speed lost. Useless on a dense one.
2. `kv_cache_type` = q8_0 — a long chat is mostly KV cache. q4_0 if you are still short.
3. `n_ctx` — the cache is proportional to it.
4. `n_gpu_layers` — the blunt one; every layer left on the CPU costs real speed.
`flash_attn` is `auto` on llama-server and already on in koboldcpp, so leave it alone unless a
model misbehaves.

## Thinking
- `reasoning` on/off is the switch (`-rea`); `reasoning_budget` caps how many tokens it may spend.
- `reasoning_format` decides where the thoughts go. The default (`auto`) already hands them back
  as `reasoning_content`, which SillyTavern shows as its own collapsible block — set `none` if you
  would rather see the raw think tags in the reply.
- `enable_thinking` / `reasoning_effort` are chat-template variables (`--chat-template-kwargs`),
  the same pair the Local LLM Settings node has: they reach families a prompt directive cannot.

## Side models
Optional inputs, each its own small node: **draft** (speculative decoding), **vision** (an mmproj,
so the client can send images) and **embeddings**. On llama-server the embedding model runs as a
SECOND process — `--embeddings` restricts a llama-server to embedding work — and the gateway routes
`/v1/embeddings` and `/rerank` to it, so SillyTavern's Vector Storage can use the same address.
koboldcpp serves embeddings from the one process.

## Parameter repair (`fix_params`)
SillyTavern stores list-valued options as JSON *strings*; llama.cpp rejects the whole request when
one is not a real array — most famously `dry_sequence_breakers`. The gateway rewrites those on the
way through. `auto_retry` goes further: a 400 naming a field is retried once without it, and that
field is stripped from then on. `drop_fields` (one name per line) removes fields outright.

## Notes
- `server_binary` must point at the executable — download or build it yourself, nothing is bundled.
- `extra_args` is for what has no widget: rope/yarn, `--tensor-split`, `--override-tensor`,
  `--lora`, `--fit`… Changing it (or any other launch setting) reloads the model.
- The two backends are not the same program: settings koboldcpp has no word for are listed in the
  `status` output rather than silently dropped.
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


def _side_model(bundle, choice_key, path_key):
    """Resolve a side bundle's model choice to a real path, or "" when there is none."""
    if not isinstance(bundle, dict):
        return ""
    p = _resolve_path(bundle.get(choice_key, PLACEHOLDER), bundle.get(path_key, ""))
    return p if p and os.path.isfile(p) else ""


class LocalLLMServer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (ACTIONS, {"default": SERVE, "tooltip": "serve = start the gateway and wait for the first request. 'load now' also loads the model straight away. 'reload' kills a loaded model so the next request starts it fresh. 'stop everything' shuts the gateway down too."}),
                "flavour": ([LLAMA_SERVER, KOBOLDCPP], {"default": LLAMA_SERVER, "tooltip": "Which server is being launched. It picks the flag names for every setting below — and a few settings only one of them has (the status output names those)."}),
                "server_binary": (_server_binaries(), {"tooltip": "Executable to launch. Auto-lists every llama-server / koboldcpp found anywhere under ComfyUI/models/llm, subfolders included; otherwise pick the placeholder and type a path below. Download or build it yourself — not bundled."}),
                "server_binary_path": ("STRING", {"default": "", "tooltip": "Full path to the server executable when the dropdown is the placeholder."}),
                "model": (_list_models(), {"tooltip": "The .gguf from ComfyUI/models/llm to serve. Placeholder = type a path in model_path."}),
                "model_path": ("STRING", {"default": "", "tooltip": "Full path to the .gguf when 'model' is the placeholder."}),
                "n_ctx": ("INT", {"default": 8192, "min": 256, "max": 1048576, "step": 256, "tooltip": "Context size (--ctx-size / --contextsize). A long chat plus a character card eats this fast — and the KV cache grows with it."}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "GPU layers (-ngl / --gpulayers). -1 = all."}),
                "extra_args": ("STRING", {"multiline": True, "default": "", "tooltip": "Raw flags for what has no widget: rope/yarn, --tensor-split, --override-tensor, --lora, --fit… Appended last, so they win. Changing this reloads the model."}),
                "listen_host": ("STRING", {"default": "127.0.0.1", "tooltip": "Where the gateway listens. 127.0.0.1 = this machine only; 0.0.0.0 to reach it from another box on the LAN."}),
                "listen_port": ("INT", {"default": 5001, "min": 1, "max": 65535, "tooltip": "The port SillyTavern is pointed at. It stays open while ComfyUI runs, model or no model."}),
                "autoload": ("BOOLEAN", {"default": True, "tooltip": "Load the model on the first request that needs one. Off = a request with nothing loaded gets a clear error instead."}),
                "free_on_prompt": ("BOOLEAN", {"default": True, "tooltip": "Unload whenever a ComfyUI prompt is queued that is not itself about the server — the safety net for image workflows you did not edit."}),
                "idle_unload_minutes": ("INT", {"default": 0, "min": 0, "max": 1440, "tooltip": "Unload after this many minutes with no request. 0 = never."}),
                "unload_comfy_models": ("BOOLEAN", {"default": True, "tooltip": "Free ComfyUI's own models from VRAM before loading the LLM."}),
                "fix_params": ("BOOLEAN", {"default": True, "tooltip": "Repair the request bodies llama.cpp refuses — SillyTavern sends dry_sequence_breakers and friends as JSON strings where an array is required."}),
                "auto_retry": ("BOOLEAN", {"default": True, "tooltip": "On a 400 that names a field, drop that field and retry once, then strip it from later requests. Keeps one unknown option from breaking every message."}),
                "drop_fields": ("STRING", {"multiline": True, "default": "", "tooltip": "Field names to strip from every request, one per line. The manual version of auto_retry."}),
                "backend_port": ("INT", {"default": 0, "min": 0, "max": 65535, "tooltip": "Private port for the model server. 0 = pick a free one."}),
                "ready_path": ("STRING", {"default": "", "tooltip": "Path polled until the model is ready. Blank = per-flavour default (llama-server /health, koboldcpp /v1/models)."}),
                "startup_timeout": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 10, "tooltip": "How long a load may take before it is called a failure."}),
                "request_timeout": ("INT", {"default": 900, "min": 10, "max": 7200, "step": 10, "tooltip": "How long one forwarded request may take."}),
                # Everything below is appended rather than filed next to its relatives: ComfyUI
                # stores widgets_values POSITIONALLY, so inserting mid-list would shift every value
                # in every saved workflow from that point on.
                "flash_attn": (options.FLASH_ATTN_MODES, {"default": "auto", "tooltip": "Flash Attention: faster, smaller KV cache. llama-server decides for itself on 'auto' and koboldcpp has it on already, so this is for forcing it off on a model that misbehaves."}),
                "kv_cache_type": (options.KV_CACHE_TYPES, {"default": "f16", "tooltip": "Quantize the KV cache to fit a longer chat in VRAM. q8_0 is nearly free; q4_0 is noticeable. Full effect needs Flash Attention — without it only K is quantized."}),
                "n_batch": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 64, "tooltip": "Logical batch size (-b / --blasbatchsize). 0 = the server's default. Bigger reads a long prompt faster and costs VRAM."}),
                "n_ubatch": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 64, "tooltip": "Physical batch size (-ub). 0 = default. llama-server only."}),
                "threads": ("INT", {"default": 0, "min": 0, "max": 256, "tooltip": "CPU threads (-t). 0 = the server's own choice. Matters only for what is not on the GPU."}),
                "cpu_moe_layers": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "Keep the Mixture-of-Experts weights of the first N layers in system RAM (-ncmoe / --moecpu). 0 = off. On a MoE model this frees more VRAM per point of speed than anything else here; on a dense model it does nothing."}),
                "parallel_slots": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Concurrent request slots (-np / --multiuser). 0 = the server's default. Each slot costs its own share of the KV cache, so 1 is right for one chat."}),
                "context_shift": ("BOOLEAN", {"default": True, "tooltip": "Let the server drop the oldest tokens and keep going when the chat outgrows n_ctx. Off = it stops instead (--no-context-shift / --noshift)."}),
                "alias": ("STRING", {"default": "", "tooltip": "The model name the API reports, and what the gateway lists in /v1/models. Blank = the .gguf filename. llama-server only."}),
                "api_key": ("STRING", {"default": "", "tooltip": "Require this key on every request (--api-key / --password). Blank = no authentication. The gateway forwards whatever the client sends."}),
                "reasoning": (options.REASONING_MODES, {"default": REASONING_DEFAULT, "tooltip": "Thinking on or off (-rea). 'model default' leaves it to the model's template. koboldcpp can only express 'off'."}),
                "reasoning_budget": ("INT", {"default": -1, "min": -1, "max": 1048576, "step": 64, "tooltip": "How many tokens thinking may spend before it is cut off (--reasoning-budget). -1 = unrestricted, 0 = end it immediately. llama-server only."}),
                "reasoning_format": (options.REASONING_FORMATS, {"default": "auto", "tooltip": "Where the thoughts go. 'auto' (the default) hands them back separately as reasoning_content, which SillyTavern renders as its own collapsible block; 'none' leaves the raw think tags in the reply. llama-server only."}),
                "enable_thinking": (THINKING_MODES, {"default": REASONING_DEFAULT, "tooltip": "Chat-template variable (--chat-template-kwargs), not prompt text — so the model cannot ignore it. 'model default' leaves it undefined, which each family reads its own way (Qwen3.5/3.8: ON, Gemma-4: OFF)."}),
                "reasoning_effort": (REASONING_EFFORTS, {"default": REASONING_DEFAULT, "tooltip": "How hard to think, as a chat-template variable. Qwen3.5/3.8 accept xhigh / medium / low and ERROR on anything else; use 'custom' for other families (gpt-oss: high)."}),
                "reasoning_effort_custom": ("STRING", {"default": "", "tooltip": "Effort value sent when reasoning_effort = custom (e.g. 'high' for gpt-oss). Empty = send nothing."}),
                "chat_template_file": ("STRING", {"default": "", "tooltip": "A .jinja chat template that OVERRIDES the one in the GGUF (--chat-template-file). Empty = the model's own, which is right for almost every model. llama-server only."}),
            },
            "optional": {
                "draft": (DRAFT_CONFIG, {"tooltip": "Optional — wire an 'LLM Server Draft' node here for speculative decoding."}),
                "vision": (VISION_CONFIG, {"tooltip": "Optional — wire an 'LLM Server Vision' node here to serve an mmproj so the chat client can send images."}),
                "embeddings": (EMBED_CONFIG, {"tooltip": "Optional — wire an 'LLM Server Embeddings' node here to serve /v1/embeddings (SillyTavern's Vector Storage) from the same address."}),
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
            backend_port, ready_path, startup_timeout, request_timeout,
            flash_attn="auto", kv_cache_type="f16", n_batch=0, n_ubatch=0, threads=0,
            cpu_moe_layers=0, parallel_slots=0, context_shift=True, alias="", api_key="",
            reasoning=REASONING_DEFAULT, reasoning_budget=-1, reasoning_format="auto",
            enable_thinking=REASONING_DEFAULT, reasoning_effort=REASONING_DEFAULT,
            reasoning_effort_custom="", chat_template_file="",
            draft=None, vision=None, embeddings=None):

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

        cfg = dict(
            flavour=flavour, binary=binary, model=resolved,
            n_ctx=int(n_ctx), n_gpu_layers=int(n_gpu_layers), extra=tuple(_split_args(extra_args)),
            backend_port=int(backend_port), autoload=bool(autoload),
            free_on_prompt=bool(free_on_prompt), idle_minutes=int(idle_unload_minutes),
            unload_comfy=bool(unload_comfy_models), fix_params=bool(fix_params),
            auto_retry=bool(auto_retry),
            drop_fields=tuple(s.strip() for s in (drop_fields or "").splitlines() if s.strip()),
            ready_path=(ready_path or "").strip(),
            startup_timeout=int(startup_timeout), request_timeout=int(request_timeout),
            flash_attn=flash_attn, kv_cache_type=kv_cache_type, n_batch=int(n_batch),
            n_ubatch=int(n_ubatch), threads=int(threads), cpu_moe_layers=int(cpu_moe_layers),
            parallel_slots=int(parallel_slots), context_shift=bool(context_shift),
            alias=alias, api_key=api_key, reasoning=reasoning,
            reasoning_budget=int(reasoning_budget), reasoning_format=reasoning_format,
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort,
            reasoning_effort_custom=reasoning_effort_custom,
            chat_template_file=chat_template_file,
        )

        problems = []
        if isinstance(vision, dict):
            mm = _side_model(vision, "mmproj", "mmproj_path")
            if not mm:
                problems.append("vision: the mmproj was not found — vision is off")
            cfg.update(mmproj_resolved=mm, mmproj_offload=vision.get("mmproj_offload", True),
                       image_max_tokens=vision.get("image_max_tokens", 0))
        if isinstance(draft, dict):
            dm = _side_model(draft, "draft_model", "draft_model_path")
            spec = draft.get("spec_type", SPEC_NONE)
            if not dm and not str(spec).startswith("ngram"):
                problems.append("draft: the draft model was not found — speculative decoding is off")
            cfg.update(draft_model_resolved=dm,
                       spec_type="" if spec == SPEC_NONE else spec,
                       draft_n_gpu_layers=draft.get("draft_n_gpu_layers", -1),
                       draft_n_max=draft.get("draft_n_max", 0),
                       draft_n_min=draft.get("draft_n_min", 0),
                       draft_p_min=draft.get("draft_p_min", 0.0))
        if isinstance(embeddings, dict):
            em = _side_model(embeddings, "embed_model", "embed_model_path")
            if not em:
                problems.append("embeddings: the model was not found — embeddings are off")
            cfg.update(embed_model_resolved=em,
                       embed_n_ctx=embeddings.get("embed_n_ctx", 0),
                       embed_n_gpu_layers=embeddings.get("embed_n_gpu_layers", -1),
                       pooling=embeddings.get("pooling", REASONING_DEFAULT),
                       rerank=embeddings.get("rerank", False),
                       embed_extra=tuple(_split_args(embeddings.get("embed_extra_args", ""))))

        problems += options.unsupported(cfg, flavour)
        gateway.configure(**cfg)

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

        status = gateway.status_text()
        if problems:
            status += "\n\nignored by this backend:\n- " + "\n- ".join(problems)
        return (status, base_url, gateway.log_tail(), HELP_TEXT)


class LLMServerDraft:
    """Speculative decoding: a small model guesses ahead and the big one checks the guess."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spec_type": (SPEC_TYPES, {"default": "draft-simple", "tooltip": "How the guessing is done (--spec-type). draft-simple = a small model of the same family. draft-mtp = a Multi-Token-Prediction head shipped alongside the model (an 'mtp-' .gguf). The ngram-* methods need NO draft model — they guess from what the text already contains, which is free and works well on repetitive writing. llama-server only; koboldcpp infers the method from the draft model."}),
                "draft_model": (_list_models(), {"tooltip": "The draft .gguf. Must share the main model's vocabulary — a different family will be rejected. Leave on the placeholder for the ngram-* methods, which need no model at all."}),
                "draft_model_path": ("STRING", {"default": "", "tooltip": "Full path to the draft .gguf when the dropdown is the placeholder."}),
                "draft_n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "GPU layers for the draft model (-ngld). -1 = all. A draft model is small; keeping it on the GPU is the point."}),
                "draft_n_max": ("INT", {"default": 0, "min": 0, "max": 512, "tooltip": "Tokens to draft per round (--spec-draft-n-max / --draftamount). 0 = the server's default (3). Higher pays off only when the draft is usually right."}),
                "draft_n_min": ("INT", {"default": 0, "min": 0, "max": 512, "tooltip": "Fewest tokens worth drafting (--spec-draft-n-min). 0 = default. llama-server only."}),
                "draft_p_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Stop drafting once the draft model's confidence drops below this (--spec-draft-p-min). 0 = the server's default. llama-server only."}),
            },
        }

    RETURN_TYPES = (DRAFT_CONFIG,)
    RETURN_NAMES = ("draft",)
    FUNCTION = "run"
    CATEGORY = CAT_LLM

    def run(self, **kw):
        return (dict(kw),)


class LLMServerVision:
    """An mmproj served alongside the model, so the chat client can send images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mmproj": (_list_mmproj(), {"tooltip": "Projector mmproj .gguf from ComfyUI/models/llm (subfolders included, mmproj-named files first). It must be the one built for this model."}),
                "mmproj_path": ("STRING", {"default": "", "tooltip": "Full path to the mmproj .gguf when the dropdown is the placeholder."}),
                "mmproj_offload": ("BOOLEAN", {"default": True, "tooltip": "Keep the projector on the GPU. Off (--no-mmproj-offload / --mmprojcpu) puts it on the CPU: a little VRAM back, slower image encoding."}),
                "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 32768, "step": 64, "tooltip": "Cap how many context tokens one image may take (--image-max-tokens). 0 = the model's own limit. Lower = cheaper and blurrier. llama-server only."}),
            },
        }

    RETURN_TYPES = (VISION_CONFIG,)
    RETURN_NAMES = ("vision",)
    FUNCTION = "run"
    CATEGORY = CAT_LLM

    def run(self, **kw):
        return (dict(kw),)


class LLMServerEmbeddings:
    """An embedding model on the same address — SillyTavern's Vector Storage talks to this."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embed_model": (_list_models(), {"tooltip": "The embedding .gguf (EmbeddingGemma, nomic-embed, bge…). NOT a chat model: on llama-server this runs as a second, embedding-only process."}),
                "embed_model_path": ("STRING", {"default": "", "tooltip": "Full path to the embedding .gguf when the dropdown is the placeholder."}),
                "embed_n_ctx": ("INT", {"default": 0, "min": 0, "max": 131072, "step": 256, "tooltip": "Context for the embedding model. 0 = its trained default, which is what you want unless it complains."}),
                "embed_n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "GPU layers for the embedding model. -1 = all; these models are small. 0 keeps it on the CPU (koboldcpp reads only 0 vs not-0)."}),
                "pooling": (options.POOLING_TYPES, {"default": "model default", "tooltip": "How token vectors become one vector (--pooling). Leave on the model's default unless its card says otherwise. llama-server only."}),
                "rerank": ("BOOLEAN", {"default": False, "tooltip": "Also expose /rerank. Only meaningful with a reranker model. llama-server only."}),
                "embed_extra_args": ("STRING", {"multiline": True, "default": "", "tooltip": "Raw flags for the embedding process only, e.g. --embd-normalize 2."}),
            },
        }

    RETURN_TYPES = (EMBED_CONFIG,)
    RETURN_NAMES = ("embeddings",)
    FUNCTION = "run"
    CATEGORY = CAT_LLM

    def run(self, **kw):
        return (dict(kw),)


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


NODE_CLASS_MAPPINGS = {
    SERVER_NODE_ID: LocalLLMServer,
    CONTROL_NODE_ID: LLMServerControl,
    "LLMServerDraft": LLMServerDraft,
    "LLMServerVision": LLMServerVision,
    "LLMServerEmbeddings": LLMServerEmbeddings,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    SERVER_NODE_ID: "Local LLM Server",
    CONTROL_NODE_ID: "LLM Server Control",
    "LLMServerDraft": "LLM Server Draft (GGUF)",
    "LLMServerVision": "LLM Server Vision (GGUF)",
    "LLMServerEmbeddings": "LLM Server Embeddings (GGUF)",
}
