"""Local LLM (server client, text) — drive an OpenAI-compatible LLM server.

Instead of the llama-cpp-python binding, this talks HTTP to a real server's
`/v1/chat/completions`, so you get the server's **full command line**. Three backends:

* **llama-server (launch)** — launches llama.cpp's `llama-server`. `extra_args` reaches any
  llama.cpp flag, e.g. `--spec-type draft-mtp`, `--flash-attn`, `--model-draft ...`.
* **koboldcpp (launch)** — launches `koboldcpp`; same idea with Kobold's flag names
  (`--model`, `--contextsize`, `--gpulayers`) and its own extras via `extra_args`.
* **connect to running server** — launches nothing; just points at a `base_url` you already
  run (koboldcpp GUI, LM Studio, Ollama, vLLM, a remote box…).

Sampling, the reasoning split (`thoughts` output / `strip_think` / `answer_marker`), reasoning
directives and structured output (`output_format` / `grammar`) mirror the llama-cpp-python
text node — the shared helpers are imported from it. Launch backends are reused across runs
while their config is unchanged and shut down on exit (or after each run when `keep_alive` is
off). Text only.
"""
import os
import re
import json
import time
import shlex
import atexit
import threading
import subprocess
from collections import deque
from urllib import request as _urlreq, error as _urlerr
from ..categories import CAT_LLM

try:
    from ..local_llm.llm_node import (
        _gguf_dir, _list_models, _resolve_path, PLACEHOLDER,
        _split_reasoning, _apply_directive, _apply_output_format, _with_context,
    )
except Exception:  # pragma: no cover — registry scan without ComfyUI
    PLACEHOLDER = "(use model_path field)"

    def _gguf_dir():
        return None

    def _list_models():
        return [PLACEHOLDER]

    def _resolve_path(choice, manual):
        return (manual or "").strip().strip('"').strip("'").strip()

    def _split_reasoning(raw, marker=""):
        return raw, ""

    def _apply_directive(user_prompt, thinking_directive, custom_directive):
        return user_prompt, ""

    def _apply_output_format(output_format, grammar, system_prompt):
        return output_format, grammar, system_prompt

    def _with_context(system_prompt, context):
        ctx = (context or "").strip()
        if not ctx:
            return system_prompt
        base = (system_prompt or "").rstrip()
        return (base + "\n\n" + ctx) if base else ctx


# Per-backend launch flag names + readiness probe. "connect" launches nothing.
LLAMA_SERVER = "llama-server (launch)"
KOBOLDCPP = "koboldcpp (launch)"
CONNECT = "connect to running server"

_BACKENDS = {
    LLAMA_SERVER: {"launch": True, "model": "-m", "ctx": "--ctx-size", "ngl": "-ngl",
                   "host": "--host", "port": "--port", "ready": "/health"},
    KOBOLDCPP: {"launch": True, "model": "--model", "ctx": "--contextsize", "ngl": "--gpulayers",
                "host": "--host", "port": "--port", "ready": "/v1/models"},
    CONNECT: {"launch": False, "ready": "/v1/models"},
}
_BACKEND_LABELS = list(_BACKENDS.keys())

HELP_TEXT = """# Local LLM (server client, text) — quick help

Talks to an OpenAI-compatible LLM server over HTTP, so you get the server's full command line.

## Backends
- `llama-server (launch)` — put any llama.cpp flag in `extra_args`, e.g. `--spec-type draft-mtp`,
  `--flash-attn`, `--model-draft path/to/draft.gguf`. Readiness probed via `/health`.
- `koboldcpp (launch)` — Kobold flag names; extras via `extra_args` (e.g. `--draftmodel ...`).
  Readiness `/v1/models`; usual port 5001.
- `connect to running server` — launches nothing; set `base_url` (e.g. http://localhost:5001)
  and it calls that server. Works with koboldcpp's GUI, LM Studio, Ollama, vLLM, remote…

## Outputs
- **text** — the answer (reasoning removed when `strip_think` is on)
- **thoughts** — `<think>…</think>` content, or text before `answer_marker` (empty if none)
- **finish_reason** — `stop` = done · `length` = hit the token/context limit (truncated)
- **sys/user_tokens** — prompt split (estimated by length; server gives only the total) ·
  **output_tokens** — generated · **thoughts/answer_tokens** — the output split by length
- **gen_seconds** — request round-trip · **server_log** — tail of a launched server's output

## If the answer is cut off (`finish_reason = length`)
Raise **max_tokens** and, for launch backends, **n_ctx** (a long `<think>` phase eats context).

## Structured output
- **json_object** forces valid JSON · **gbnf_grammar** forces a shape from the `grammar` field ·
  **ideogram4_json** is the built-in nested Ideogram prompt grammar (use max_tokens ≥ 384).

## Notes
- `server_binary` must point at the executable for the launch backends (download it yourself).
- `keep_alive` ON keeps a launched server loaded (holds VRAM); OFF stops it after each run.
- `ready_path` overrides the health probe (blank = per-backend default).
"""

# One shared launched process across all node instances.
_proc = None
_sig = None
_log = deque(maxlen=400)
_lock = threading.Lock()


def _server_binaries():
    """Auto-list llama-server / koboldcpp executables found in models/llm (and its parent)."""
    found = []
    d = _gguf_dir()
    for base in filter(None, [d, os.path.dirname(d) if d else None]):
        if not os.path.isdir(base):
            continue
        for f in sorted(os.listdir(base)):
            lo = f.lower()
            if (("llama-server" in lo or "koboldcpp" in lo or "kobold" in lo)
                    and (lo.endswith(".exe") or "." not in lo)):
                p = os.path.join(base, f)
                if os.path.isfile(p) and p not in found:
                    found.append(p)
    return found + [PLACEHOLDER]


def _resolve_binary(choice, manual):
    if choice and choice != PLACEHOLDER and os.path.isfile(choice):
        return choice
    return (manual or "").strip().strip('"').strip("'").strip()


def _drain(proc):
    try:
        for line in iter(proc.stdout.readline, ""):
            if line == "" and proc.poll() is not None:
                break
            _log.append(line.rstrip("\n"))
    except Exception:
        pass


def _log_tail(n=40):
    return "\n".join(list(_log)[-n:])


def _stop_server():
    global _proc, _sig
    proc = _proc
    _proc = None
    _sig = None
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


atexit.register(_stop_server)


def _http_get(url, timeout=2.0):
    with _urlreq.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def _http_post_json(url, obj, timeout=600.0):
    data = json.dumps(obj).encode("utf-8")
    req = _urlreq.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with _urlreq.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def _wait_ready(base_url, ready_path, proc, timeout_s):
    """Poll until ready_path returns 200, the process dies, or we time out."""
    deadline = time.time() + max(1, int(timeout_s))
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False, f"server exited during startup (code {proc.returncode})."
        try:
            status, _ = _http_get(base_url + ready_path, timeout=2.0)
            if status == 200:
                return True, ""
        except (_urlerr.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.4)
    return False, f"server did not become ready within {int(timeout_s)}s (probed {ready_path})."


def _ensure_server(sig, argv, base_url, ready_path, timeout_s):
    """Reuse a launched server matching sig, else (re)launch. Returns (ok, error_message)."""
    global _proc, _sig
    if _proc is not None and _proc.poll() is None and _sig == sig:
        return True, ""

    _stop_server()
    _log.clear()
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except Exception as e:
        return False, f"Failed to launch server: {e}"

    _proc = proc
    _sig = sig
    threading.Thread(target=_drain, args=(proc,), daemon=True).start()

    ok, err = _wait_ready(base_url, ready_path, proc, timeout_s)
    if not ok:
        tail = _log_tail()
        _stop_server()
        return False, err + ("\n--- server log ---\n" + tail if tail else "")
    return True, ""


def _split_count(total, a, b):
    """Split an exact token total between two strings, proportional to their length."""
    denom = len(a) + len(b)
    first = round(total * len(a) / denom) if (denom and total) else 0
    return first, max(0, total - first)


def _err(msg):
    """Keep the 11-output signature on every error path."""
    return (f"[ERROR] {msg}", "", "error", 0, 0, 0, 0.0, HELP_TEXT, 0, 0, _log_tail())


class LocalLLMServerText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (_BACKEND_LABELS, {"default": LLAMA_SERVER, "tooltip": "llama-server / koboldcpp are launched for you; 'connect to running server' just calls base_url of a server you already run."}),
                "server_binary": (_server_binaries(), {"tooltip": "Executable for the launch backends. Auto-lists ones in ComfyUI/models/llm; else pick the placeholder and set server_binary_path. Download it yourself — not bundled."}),
                "server_binary_path": ("STRING", {"default": "", "tooltip": "Full path to the server executable when the dropdown is the placeholder."}),
                "base_url": ("STRING", {"default": "", "tooltip": "For 'connect to running server': the server root, e.g. http://localhost:5001. Blank = use host:port below. Ignored by the launch backends."}),
                "model": (_list_models(), {"tooltip": "Main .gguf from ComfyUI/models/llm (launch backends). Placeholder = type a path in model_path."}),
                "model_path": ("STRING", {"default": "", "tooltip": "Full path to the .gguf when 'model' is the placeholder."}),
                "model_name": ("STRING", {"default": "", "tooltip": "Value sent as the request's 'model' field. Usually optional; some servers route by it. Blank = 'local'."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "context": ("STRING", {"multiline": True, "default": "", "tooltip": "Reference material appended to the system prompt (e.g. character cards from Context Collector). Empty = ignored."}),
                "max_tokens": ("INT", {"default": 512, "min": 16, "max": 32768, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 32768}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Min-p sampling. 0 = off. Try ~0.05 (often paired with top_p=1.0, top_k=0)"}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01}),
                "stop": ("STRING", {"multiline": True, "default": "", "tooltip": "Stop strings, one per line."}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "n_ctx": ("INT", {"default": 4096, "min": 256, "max": 1048576, "step": 256, "tooltip": "Context size for the launch backends (llama-server --ctx-size / koboldcpp --contextsize). Raise it if answers get cut off (finish_reason = length)."}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "GPU layers for the launch backends (llama-server -ngl / koboldcpp --gpulayers). -1 = all."}),
                "host": ("STRING", {"default": "127.0.0.1", "tooltip": "Host for launch backends (also the base_url fallback for connect mode)."}),
                "port": ("INT", {"default": 8080, "min": 1, "max": 65535, "tooltip": "Port for launch backends. koboldcpp's usual default is 5001."}),
                "extra_args": ("STRING", {"multiline": True, "default": "", "tooltip": "Raw CLI flags appended to the launch command, e.g. `--spec-type draft-mtp` (llama-server) or `--draftmodel path` (koboldcpp). Changing this relaunches the server."}),
                "strip_think": ("BOOLEAN", {"default": True, "tooltip": "Keep reasoning out of 'text' (it still goes to 'thoughts'). Off = leave raw reasoning in 'text'."}),
                "answer_marker": ("STRING", {"default": "", "tooltip": "For models that reason WITHOUT <think> tags: answer = text after the LAST line equal to this marker; before it -> thoughts. Empty = use <think> tags."}),
                "thinking_directive": (["model default", "/no_think (Qwen3)", "/think (Qwen3)", "custom"], {"default": "model default", "tooltip": "Append a reasoning-control directive to the prompt. /no_think makes Qwen3-style models skip the <think> phase; 'custom' uses the field below."}),
                "custom_directive": ("STRING", {"default": "", "tooltip": "Directive appended when thinking_directive = custom (e.g. /no_think)."}),
                "output_format": (["text", "json_object", "gbnf_grammar", "ideogram4_json"], {"default": "text", "tooltip": "Output: free text · valid JSON (json_object) · custom GBNF grammar (field below) · ideogram4_json = built-in nested Ideogram JSON."}),
                "grammar": ("STRING", {"multiline": True, "default": "", "tooltip": "GBNF grammar text, used when output_format = gbnf_grammar. Sent to the server as its 'grammar' field."}),
                "ready_path": ("STRING", {"default": "", "tooltip": "Override the readiness probe path. Blank = per-backend default (llama-server /health, koboldcpp & connect /v1/models)."}),
                "startup_timeout": ("INT", {"default": 180, "min": 10, "max": 3600, "step": 10, "tooltip": "How long to wait for the server/model to be ready."}),
                "request_timeout": ("INT", {"default": 600, "min": 10, "max": 7200, "step": 10}),
                "unload_comfy_models": ("BOOLEAN", {"default": True, "tooltip": "Unload ComfyUI (image) models from VRAM before generating."}),
                "keep_alive": ("BOOLEAN", {"default": True, "tooltip": "Keep a launched server running after generation (holds VRAM). Off = stop it after each run. No effect in connect mode."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("text", "thoughts", "finish_reason", "sys_tokens", "user_tokens", "output_tokens", "gen_seconds", "help", "thoughts_tokens", "answer_tokens", "server_log")
    FUNCTION = "run"
    CATEGORY = CAT_LLM

    def run(self, backend, server_binary, server_binary_path, base_url, model, model_path,
            model_name, system_prompt, user_prompt, max_tokens, temperature, top_p, top_k,
            min_p, repeat_penalty, stop, seed, n_ctx, n_gpu_layers, host, port, extra_args,
            ready_path, startup_timeout, request_timeout, unload_comfy_models, keep_alive,
            context="", strip_think=True, answer_marker="", thinking_directive="model default",
            custom_directive="", output_format="text", grammar=""):

        cfg = _BACKENDS.get(backend, _BACKENDS[LLAMA_SERVER])
        ready = (ready_path or "").strip() or cfg["ready"]
        host = (host or "127.0.0.1").strip() or "127.0.0.1"

        system_prompt = _with_context(system_prompt, context)
        user_prompt, directive = _apply_directive(user_prompt, thinking_directive, custom_directive)
        eff_format, eff_grammar, eff_system = _apply_output_format(output_format, grammar, system_prompt)

        if unload_comfy_models:
            try:
                import comfy.model_management as mm
                mm.unload_all_models()
                mm.soft_empty_cache(True)
            except Exception as e:
                print(f"[LocalLLMServer] unload_all_models failed: {e}")

        t0 = time.perf_counter()
        with _lock:
            if cfg["launch"]:
                binary = _resolve_binary(server_binary, server_binary_path)
                if not binary or not os.path.isfile(binary):
                    return _err(f"Server binary not found: {binary or '(none selected)'}")
                resolved = _resolve_path(model, model_path)
                if not resolved or not os.path.isfile(resolved):
                    return _err(f"Model file not found: {resolved or '(none selected)'}")
                try:
                    extra = shlex.split(extra_args or "", posix=(os.name != "nt"))
                except ValueError as e:
                    return _err(f"Could not parse extra_args: {e}")

                url = f"http://{host}:{int(port)}"
                argv = [binary, cfg["model"], resolved, cfg["host"], host, cfg["port"], str(int(port)),
                        cfg["ctx"], str(int(n_ctx)), cfg["ngl"], str(int(n_gpu_layers))] + extra
                sig = (backend, binary, resolved, int(n_ctx), int(n_gpu_layers), host, int(port), tuple(extra))
                ok, err = _ensure_server(sig, argv, url, ready, startup_timeout)
                if not ok:
                    return _err(err)
            else:
                url = (base_url or "").strip().rstrip("/") or f"http://{host}:{int(port)}"
                try:
                    _http_get(url + ready, timeout=5.0)
                except Exception as e:
                    return _err(f"Can't reach server at {url} ({e}). Is it running?")

            messages = []
            if (eff_system or "").strip():
                messages.append({"role": "system", "content": eff_system.strip()})
            messages.append({"role": "user", "content": user_prompt or ""})

            body = {
                "model": (model_name or "").strip() or "local",
                "messages": messages,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "min_p": float(min_p),
                "repeat_penalty": float(repeat_penalty),
                "seed": int(seed),
                "stream": False,
            }
            stops = [s for s in (stop or "").splitlines() if s.strip()]
            if stops:
                body["stop"] = stops
            if eff_format == "json_object":
                body["response_format"] = {"type": "json_object"}
            elif eff_format == "gbnf_grammar" and (eff_grammar or "").strip():
                body["grammar"] = eff_grammar  # llama.cpp/koboldcpp GBNF extension

            try:
                status, raw = _http_post_json(url + "/v1/chat/completions", body, timeout=request_timeout)
            except _urlerr.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")
                except Exception:
                    pass
                return _err(f"Server returned HTTP {e.code}: {detail or e.reason}")
            except Exception as e:
                return _err(f"Request failed: {e}")
            finally:
                if cfg["launch"] and not keep_alive:
                    _stop_server()

        gen_seconds = round(time.perf_counter() - t0, 2)
        try:
            data = json.loads(raw)
        except Exception as e:
            return _err(f"Bad JSON from server: {e}")

        choice = (data.get("choices") or [{}])[0]
        raw_text = ((choice.get("message") or {}).get("content")) or ""
        finish_reason = choice.get("finish_reason", "") or ""
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)

        # Drop the reasoning directive echo, then split reasoning from the answer (same as the
        # llama-cpp-python node). Token splits are proportional-by-length estimates.
        if directive:
            raw_text = re.sub(r"\s*" + re.escape(directive) + r"(?=\W|$)", "", raw_text).strip()
        answer, thoughts = _split_reasoning(raw_text, answer_marker)
        text = answer if strip_think else raw_text
        thoughts_tokens, answer_tokens = _split_count(output_tokens, thoughts, answer)
        sys_tokens, user_tokens = _split_count(prompt_tokens, eff_system or "", user_prompt or "")

        return (text, thoughts, finish_reason, sys_tokens, user_tokens, output_tokens,
                gen_seconds, HELP_TEXT, thoughts_tokens, answer_tokens, _log_tail())


NODE_CLASS_MAPPINGS = {"LocalLLMServerText": LocalLLMServerText}
NODE_DISPLAY_NAME_MAPPINGS = {"LocalLLMServerText": "Local LLM (server client, text)"}
