import os
import sys
import json
import re
import time
import atexit
import threading
import subprocess

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(NODE_DIR, "gguf_worker.py")
RESP_PREFIX = "@@LLM_RESPONSE@@"
PROG_PREFIX = "@@LLM_PROGRESS@@"
PLACEHOLDER = "(use model_path field)"

HELP_TEXT = """# Local LLM (GGUF) — quick help

## Outputs
- **text** — the answer (reasoning removed when `strip_think` is on)
- **thoughts** — `<think>…</think>` content (reasoning models only, else empty)
- **finish_reason** — `stop` = finished on its own; `length` = hit `max_tokens` (truncated)
- **sys/user/output_tokens** — token counts · **gen_seconds** — load + generation time

## If the answer is cut off
`finish_reason = length`. Raise **max_tokens** (it is a ceiling, not a target) and
ask for brevity in the prompt by *words/sentences*, not tokens.

## Sampling presets
- **Creative / RP:** temp 0.7–0.9, min_p 0.05, top_p 0.95, top_k 40, repeat_penalty 1.1
- **Instruct / Qwen3:** temp 0.7, top_p 0.8, top_k 20, min_p 0 (+ thinking_directive = /no_think)
- **Deterministic (JSON / extraction):** temp 0.0, fixed seed
- Using min_p as the main filter? Set top_p 1.0, top_k 0 and you can raise temperature.

## VRAM
- Turn on **flash_attn** (faster, smaller KV cache).
- **kv_cache_type = q8_0** to fit a bigger context.
- All layers on GPU: `n_gpu_layers = -1`. To free VRAM for image gen keep both unload toggles ON.

## Reasoning models
- **thinking_directive = /no_think (Qwen3)** to skip the `<think>` phase (faster, fewer tokens).
- **strip_think** keeps reasoning out of `text` (it still goes to `thoughts`).
- **answer_marker** — for models that reason WITHOUT `<think>` tags (e.g. a "Thinking
  Process:" preamble): set a marker (say `===PROMPT===`) and tell the model to print it
  right before the final answer. Text after the last marker = answer, before = thoughts.

## Structured output (output_format)
- **json_object** — forces valid JSON.
- **gbnf_grammar** — forces an exact shape; write rules in the **grammar** field. Examples:

Comma-separated tag list:
```
root ::= tag (", " tag)*
tag  ::= [a-zA-Z0-9 ]+
```

Pick one option:
```
root ::= "photorealistic" | "anime" | "oil painting" | "3d render"
```
The grammar controls the *shape*, not the meaning — still prompt for the right content.
Note: grammar modes run without streaming, so the live token progress bar stays idle
until the result is ready.

## Ideogram JSON prompt (output_format = ideogram4_json)
A built-in GBNF grammar **forces** this nested JSON, so the structure is guaranteed on
any model: `high_level_description`, `style_description` {aesthetics, lighting, photo,
medium}, `compositional_deconstruction` {background, elements[] each with type / bbox /
desc / optional color_palette}. Describe the scene in your prompt and use
**max_tokens >= 384** so the JSON is not cut off. (Field *contents* still depend on the
model — a strong instruct model fills them best.)
"""

IDEOGRAM4_GRAMMAR = r"""root ::= "{" "\"high_level_description\":" string "," "\"style_description\":" styleobj "," "\"compositional_deconstruction\":" compobj "}"
styleobj ::= "{" "\"aesthetics\":" string "," "\"lighting\":" string "," "\"photo\":" string "," "\"medium\":" string "}"
compobj ::= "{" "\"background\":" string "," "\"elements\":" "[" element ("," element)? ("," element)? ("," element)? "]" "}"
element ::= "{" "\"type\":" eltype "," "\"bbox\":" bbox "," "\"desc\":" string optcolor "}"
optcolor ::= ("," "\"color_palette\":" palette)?
eltype ::= "\"obj\"" | "\"text\""
bbox ::= "[" int "," int "," int "," int "]"
palette ::= "[" hexcolor ("," hexcolor)? ("," hexcolor)? "]"
hexcolor ::= "\"#" hexd hexd hexd hexd hexd hexd "\""
hexd ::= [0-9A-Fa-f]
int ::= digit digit? digit? digit?
digit ::= [0-9]
string ::= "\"" (schar1 schar*)? "\""
schar1 ::= [a-zA-Z0-9 .,!?'():;&%#+-]
schar ::= [a-zA-Z0-9 .,!?'():;/&%#+-]
"""

IDEOGRAM4_INSTRUCTION = (
    "Describe the user's requested image as structured JSON data; be vivid and "
    "concrete. Field guide: high_level_description = the whole scene in 1-3 sentences; "
    "style_description = aesthetics, lighting, camera/photo (e.g. '85mm, f/1.8'), and "
    "medium; compositional_deconstruction = a background plus 2-4 elements, each a "
    "bounding box [x1, y1, x2, y2] (~0-1000) with type 'obj' (an object) or 'text' "
    "(text to render in the image) and a short desc."
)

_worker_proc = None
_worker_sig = None
_worker_lock = threading.Lock()


def _gguf_dir():
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "llm")
    except Exception:
        return None


def _list_models():
    d = _gguf_dir()
    files = []
    if d and os.path.isdir(d):
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".gguf"))
    return [PLACEHOLDER] + files


def _split_reasoning(raw, marker=""):
    """Separate reasoning from the answer, returning (answer, thoughts).

    1) If `marker` is set, the answer is everything after the LAST line whose whole
       content equals the marker (text before -> thoughts). Only a standalone marker
       line counts, so the model quoting the marker inline in its reasoning is ignored.
       Use this for models that reason WITHOUT <think> tags (e.g. a "Thinking Process:"
       preamble): tell the model to print the marker on its own line before the answer.
    2) Otherwise extract <think>...</think> blocks (also handles an unclosed one
       left by truncation)."""
    m = (marker or "").strip()
    if m:
        lines = raw.split("\n")
        last = -1
        for idx, ln in enumerate(lines):
            if ln.strip() == m:
                last = idx
        if last != -1:
            return "\n".join(lines[last + 1:]).strip(), "\n".join(lines[:last]).strip()
    parts = re.findall(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    if "<think>" in answer:                      # opened but never closed
        i = answer.index("<think>")
        parts.append(answer[i + len("<think>"):])
        answer = answer[:i]
    thoughts = "\n\n".join(p.strip() for p in parts).strip()
    return answer.strip(), thoughts


def _shutdown_worker():
    global _worker_proc, _worker_sig
    proc = _worker_proc
    _worker_proc = None
    _worker_sig = None
    if proc is None:
        return
    try:
        if proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


atexit.register(_shutdown_worker)


def _start_worker(load_sig):
    global _worker_proc, _worker_sig
    _shutdown_worker()
    proc = subprocess.Popen(
        [sys.executable, WORKER, "--serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        bufsize=1,
        cwd=NODE_DIR,
    )
    _worker_proc = proc
    _worker_sig = load_sig
    return proc


def _ensure_worker(load_sig):
    if (_worker_proc is not None and _worker_proc.poll() is None
            and _worker_sig == load_sig):
        return _worker_proc
    return _start_worker(load_sig)


def _request(proc, req, progress_cb=None):
    try:
        proc.stdin.write(json.dumps(req, ensure_ascii=True) + "\n")
        proc.stdin.flush()
    except Exception as e:
        return {"status": "error", "message": f"worker exited (write failed: {e})"}
    while True:
        line = proc.stdout.readline()
        if line == "":
            return {"status": "error", "message": "worker exited before responding"}
        if line.startswith(RESP_PREFIX):
            try:
                return json.loads(line[len(RESP_PREFIX):])
            except Exception as e:
                return {"status": "error", "message": f"bad response json: {e}"}
        if line.startswith(PROG_PREFIX):
            if progress_cb is not None:
                try:
                    progress_cb(int(line[len(PROG_PREFIX):].strip()))
                except Exception:
                    pass
            continue
        sys.stdout.write("[LocalLLM worker] " + line)


class LocalLLMTextGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_models(), {"tooltip": "Pick a .gguf from ComfyUI/models/llm. Choose the placeholder to type any path in model_path"}),
                "model_path": ("STRING", {"default": "", "tooltip": "Full path to a .gguf, used when 'model' is the placeholder. Lets you load models from anywhere on disk"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 512, "min": 16, "max": 32768, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 32768}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Min-p sampling. 0 = off. Try ~0.05 (often paired with top_p=1.0, top_k=0)"}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01}),
                "stop": ("STRING", {"multiline": True, "default": "", "tooltip": "Stop strings, one per line. Generation stops as soon as any is produced"}),
                "n_ctx": ("INT", {"default": 4096, "min": 256, "max": 1048576, "step": 256}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "-1 = all layers on GPU, 0 = all on CPU"}),
                "n_batch": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "flash_attn": ("BOOLEAN", {"default": False, "tooltip": "Flash Attention: faster and a smaller KV cache (less VRAM)"}),
                "kv_cache_type": (["f16", "q8_0", "q4_0"], {"default": "f16", "tooltip": "Quantize the KV cache to fit a bigger context in VRAM. q8_0/q4_0 auto-enable Flash Attention"}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "unload_comfy_models": ("BOOLEAN", {"default": True, "tooltip": "Unload ComfyUI (image) models from VRAM before running the LLM"}),
                "unload_llm_after_run": ("BOOLEAN", {"default": True, "tooltip": "Free the LLM from VRAM after generating. Turn OFF to keep it loaded for faster repeated runs"}),
                "strip_think": ("BOOLEAN", {"default": True, "tooltip": "Keep reasoning out of the 'text' output (it still goes to the 'thoughts' output). Off = leave raw reasoning in 'text'"}),
                "answer_marker": ("STRING", {"default": "", "tooltip": "For models that print reasoning WITHOUT <think> tags: the answer is taken after the LAST occurrence of this marker, everything before goes to 'thoughts'. Empty = use <think> tags. Tip: instruct the model to print this exact marker right before its final answer."}),
                "thinking_directive": (["model default", "/no_think (Qwen3)", "/think (Qwen3)", "custom"], {"default": "model default", "tooltip": "Append a reasoning-control directive to the prompt. /no_think makes Qwen3-style models skip the <think> phase; 'custom' uses the field below. No effect on models without such a switch"}),
                "custom_directive": ("STRING", {"default": "", "tooltip": "Directive text appended to the prompt when thinking_directive = custom (e.g. /no_think)"}),
                "output_format": (["text", "json_object", "gbnf_grammar", "ideogram4_json"], {"default": "text", "tooltip": "Output: free text · valid JSON (json_object) · custom GBNF grammar (field below) · ideogram4_json = built-in nested Ideogram JSON. Grammar modes run without the live progress bar"}),
                "grammar": ("STRING", {"multiline": True, "default": "", "tooltip": "GBNF grammar text, used when output_format = gbnf_grammar"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("text", "thoughts", "finish_reason", "sys_tokens", "user_tokens", "output_tokens", "gen_seconds", "help")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, model, model_path, system_prompt, user_prompt, max_tokens, temperature,
            top_p, top_k, min_p, repeat_penalty, stop, n_ctx, n_gpu_layers, n_batch,
            flash_attn, kv_cache_type, seed, unload_comfy_models, unload_llm_after_run,
            strip_think, thinking_directive="model default", custom_directive="",
            output_format="text", grammar="", answer_marker=""):

        def _err(msg):
            # Keep the output signature consistent on every error path.
            return (f"[ERROR] {msg}", "", "error", 0, 0, 0, 0.0, HELP_TEXT)

        # Resolve the model: a real dropdown selection wins; else the manual path.
        resolved = ""
        if model and model != PLACEHOLDER:
            d = _gguf_dir()
            resolved = os.path.join(d, model) if d else model
        elif model_path and model_path.strip():
            resolved = model_path.strip()
        if not resolved or not os.path.isfile(resolved):
            return _err(f"Model file not found: {resolved or '(none selected)'}")

        directive = {
            "/no_think (Qwen3)": "/no_think",
            "/think (Qwen3)": "/think",
            "custom": (custom_directive or "").strip(),
        }.get(thinking_directive, "")
        if directive:
            user_prompt = (user_prompt or "").rstrip() + "\n" + directive

        if unload_comfy_models:
            try:
                import comfy.model_management as mm
                mm.unload_all_models()
                mm.soft_empty_cache(True)
            except Exception as e:
                print(f"[LocalLLM] unload_all_models failed: {e}")

        stop_list = [s for s in (stop or "").splitlines() if s.strip()]

        eff_format, eff_grammar, eff_system = output_format, grammar, system_prompt
        if output_format == "ideogram4_json":
            eff_format, eff_grammar = "gbnf_grammar", IDEOGRAM4_GRAMMAR
            sep = (system_prompt.rstrip() + "\n\n") if system_prompt.strip() else ""
            eff_system = sep + IDEOGRAM4_INSTRUCTION

        req = {
            "model_path": resolved,
            "system_prompt": eff_system,
            "user_prompt": user_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "stop": stop_list,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
            "flash_attn": flash_attn,
            "kv_cache_type": kv_cache_type,
            "seed": seed,
            "output_format": eff_format,
            "grammar": eff_grammar,
            "verbose": False,
        }
        load_sig = (resolved, int(n_ctx), int(n_gpu_layers), int(n_batch),
                    bool(flash_attn), kv_cache_type)

        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(max_tokens)
        except Exception:
            pbar = None

        def _progress(n):
            if pbar is not None:
                try:
                    pbar.update_absolute(min(n, max_tokens))
                except Exception:
                    pass

        gen_seconds = 0.0
        with _worker_lock:
            try:
                t0 = time.perf_counter()
                proc = _ensure_worker(load_sig)
                data = _request(proc, req, _progress)
                if (data.get("status") == "error"
                        and "worker exited" in data.get("message", "")):
                    proc = _start_worker(load_sig)
                    data = _request(proc, req, _progress)
                gen_seconds = round(time.perf_counter() - t0, 2)
            except Exception as e:
                _shutdown_worker()
                return _err(f"Worker communication failed: {e}")
            finally:
                if unload_llm_after_run:
                    _shutdown_worker()

        _progress(max_tokens)

        if data.get("status") == "success":
            raw = data["output"]
            if directive:
                raw = re.sub(r"\s*" + re.escape(directive) + r"(?=\W|$)", "", raw).strip()
            answer, thoughts = _split_reasoning(raw, answer_marker)
            text = answer if strip_think else raw
            return (
                text,
                thoughts,
                data.get("finish_reason", ""),
                int(data.get("sys_tokens", 0)),
                int(data.get("user_tokens", 0)),
                int(data.get("output_tokens", 0)),
                gen_seconds,
                HELP_TEXT,
            )

        print("[LocalLLM] worker error:\n", data.get("traceback", ""))
        return _err(data.get("message", "unknown"))


NODE_CLASS_MAPPINGS = {"LocalLLMTextGGUF": LocalLLMTextGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"LocalLLMTextGGUF": "Local LLM (GGUF, text)"}
