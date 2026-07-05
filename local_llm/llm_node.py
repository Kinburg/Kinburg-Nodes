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
- **sys/user/output_tokens** — token counts · **thoughts/answer_tokens** — the output split
  by text length (estimate; sums to output_tokens) · **gen_seconds** — load + generation time

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


VISION_HELP_TEXT = """# Local LLM (GGUF, vision) — quick help

Needs TWO files: the vision model `.gguf` (**model**) AND its projector `mmproj` `.gguf`
(**mmproj**). They usually ship together in a model's repo — put both in ComfyUI/models/llm.

## Inputs
- **vision_handler = auto (MTMD)** works for most modern vision GGUFs (LLaVA, Qwen2-VL,
  MiniCPM-V, Gemma3, SmolVLM, …). If a model won't load or returns garbage, pick its family.
- **image** — the picture(s) to analyze; a batch sends every frame. **image_max_side**
  downscales before sending (the model resizes internally anyway); 0 = send full size.
- Sampling, max_tokens, the reasoning split, and output_format/grammar behave exactly like
  the text node — e.g. force a JSON description of the image with output_format = json_object.

## Notes
- Image tokens are counted inside the model and are NOT reflected in the token outputs here.
- VRAM: the projector loads alongside the model; keep both unload toggles ON to free it.
"""

# Vision chat-handler dropdown: friendly label -> key sent to the worker (_HANDLER_MAP there).
_VISION_HANDLERS = [
    ("auto (MTMD)", "auto"),
    ("LLaVA 1.5", "llava-1.5"),
    ("LLaVA 1.6", "llava-1.6"),
    ("Qwen2-VL / Qwen2.5-VL", "qwen2-vl"),
    ("MiniCPM-V 2.6", "minicpm-v-2.6"),
    ("Moondream", "moondream"),
    ("NanoLLaVA", "nanollava"),
    ("Llama 3 Vision", "llama3-vision"),
    ("Obsidian", "obsidian"),
    ("Gemma", "gemma"),
]
_VISION_HANDLER_LABELS = [lbl for lbl, _ in _VISION_HANDLERS]
_VISION_HANDLER_KEY = {lbl: key for lbl, key in _VISION_HANDLERS}


def _list_mmproj():
    """models/llm .gguf files for the mmproj dropdown, mmproj-named ones first."""
    d = _gguf_dir()
    files = []
    if d and os.path.isdir(d):
        allg = [f for f in os.listdir(d) if f.lower().endswith(".gguf")]
        files = sorted(f for f in allg if "mmproj" in f.lower()) + \
                sorted(f for f in allg if "mmproj" not in f.lower())
    return [PLACEHOLDER] + files


def _resolve_path(choice, manual):
    """A models/llm dropdown selection wins; else the manual path (quotes/space stripped)."""
    if choice and choice != PLACEHOLDER:
        d = _gguf_dir()
        return os.path.join(d, choice) if d else choice
    return (manual or "").strip().strip('"').strip("'").strip()


def _err(msg, help_text=HELP_TEXT):
    """Keep the 10-output signature consistent on every error path."""
    return (f"[ERROR] {msg}", "", "error", 0, 0, 0, 0.0, help_text, 0, 0)


def _apply_directive(user_prompt, thinking_directive, custom_directive):
    """Append a Qwen3-style reasoning directive to the prompt; return (prompt, directive)."""
    directive = {
        "/no_think (Qwen3)": "/no_think",
        "/think (Qwen3)": "/think",
        "custom": (custom_directive or "").strip(),
    }.get(thinking_directive, "")
    if directive:
        user_prompt = (user_prompt or "").rstrip() + "\n" + directive
    return user_prompt, directive


def _parse_extra_args(text):
    """Parse the extra_load_args field into a dict of Llama() kwargs.

    Accepts either a JSON object, or one `key=value` per line (blank / `#` lines skipped);
    values are parsed with ast.literal_eval (so 8, 1.0, True, [1,1], "str" keep their type),
    falling back to a bare string. Raises ValueError on malformed input.
    """
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("extra_load_args JSON must be an object, e.g. {\"n_threads\": 8}")
        return obj
    import ast
    out = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if "=" not in ln:
            raise ValueError(f"extra_load_args: expected key=value, got: {ln!r}")
        k, v = ln.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(f"extra_load_args: empty key in line: {ln!r}")
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v  # bare, unquoted string value
    return out


def _with_context(system_prompt, context):
    """Append a reference-context block (e.g. from Context Collector) to the system prompt."""
    ctx = (context or "").strip()
    if not ctx:
        return system_prompt
    base = (system_prompt or "").rstrip()
    return (base + "\n\n" + ctx) if base else ctx


def _apply_output_format(output_format, grammar, system_prompt):
    """Expand the built-in ideogram4_json preset; return (format, grammar, system_prompt)."""
    if output_format == "ideogram4_json":
        sep = (system_prompt.rstrip() + "\n\n") if system_prompt.strip() else ""
        return "gbnf_grammar", IDEOGRAM4_GRAMMAR, sep + IDEOGRAM4_INSTRUCTION
    return output_format, grammar, system_prompt


def _encode_images(image, max_side):
    """ComfyUI IMAGE [B,H,W,C] float 0..1 -> list of PNG `data:` URIs, downscaled to max_side
    (0 = no downscale). Encoded here (the node has PIL/torch) so the worker stays llama-only."""
    import io
    import base64
    import numpy as np
    from PIL import Image

    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None, ...]
    uris = []
    for im in arr:
        pil = Image.fromarray((np.clip(im, 0.0, 1.0) * 255.0).astype("uint8")).convert("RGB")
        if max_side and max(pil.size) > max_side:
            s = max_side / max(pil.size)
            pil = pil.resize((max(1, round(pil.width * s)), max(1, round(pil.height * s))),
                             Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        uris.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
    return uris


def _generate_and_format(req, load_sig, max_tokens, unload_comfy_models,
                         unload_llm_after_run, directive, strip_think, answer_marker, help_text):
    """Shared core for both LLM nodes: optionally free ComfyUI VRAM, talk to the worker with a
    live token progress bar, then split reasoning out and return the 10-output tuple."""
    if unload_comfy_models:
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache(True)
        except Exception as e:
            print(f"[LocalLLM] unload_all_models failed: {e}")

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
            return _err(f"Worker communication failed: {e}", help_text)
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
        out_tok = int(data.get("output_tokens", 0))
        # Split the exact total proportionally by text length — a cheap estimate (the
        # tokenizer lives only in the worker). thoughts_tokens + answer_tokens == out_tok.
        denom = len(answer) + len(thoughts)
        thoughts_tokens = round(out_tok * len(thoughts) / denom) if (denom and out_tok) else 0
        answer_tokens = max(0, out_tok - thoughts_tokens)
        return (text, thoughts, data.get("finish_reason", ""),
                int(data.get("sys_tokens", 0)), int(data.get("user_tokens", 0)),
                out_tok, gen_seconds, help_text, thoughts_tokens, answer_tokens)

    print("[LocalLLM] worker error:\n", data.get("traceback", ""))
    return _err(data.get("message", "unknown"), help_text)


class LocalLLMTextGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_models(), {"tooltip": "Pick a .gguf from ComfyUI/models/llm. Choose the placeholder to type any path in model_path"}),
                "model_path": ("STRING", {"default": "", "tooltip": "Full path to a .gguf, used when 'model' is the placeholder. Lets you load models from anywhere on disk. Surrounding quotes (e.g. from Windows 'Copy as path') are stripped automatically."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "context": ("STRING", {"multiline": True, "default": "", "tooltip": "Reference material appended to the system prompt — e.g. character cards from Context Collector. Type it here or wire in a STRING. The model uses it to expand named subjects in the prompt. Empty = ignored."}),
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
                "extra_load_args": ("STRING", {"multiline": True, "default": "", "tooltip": "Advanced: extra keyword args for llama-cpp-python's Llama() loader. One per line as key=value (e.g. n_threads=8, main_gpu=0, rope_freq_base=1000000, tensor_split=[1,1]) or a JSON object. These are Python-binding args, NOT llama.cpp CLI flags — e.g. '--spec-type draft-mtp' has no effect here. Unknown keys are ignored (logged to console). Changing this reloads the model."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "INT", "INT")
    RETURN_NAMES = ("text", "thoughts", "finish_reason", "sys_tokens", "user_tokens", "output_tokens", "gen_seconds", "help", "thoughts_tokens", "answer_tokens")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, model, model_path, system_prompt, user_prompt, max_tokens, temperature,
            top_p, top_k, min_p, repeat_penalty, stop, n_ctx, n_gpu_layers, n_batch,
            flash_attn, kv_cache_type, seed, unload_comfy_models, unload_llm_after_run,
            strip_think, context="", thinking_directive="model default", custom_directive="",
            output_format="text", grammar="", answer_marker="", extra_load_args=""):

        # Resolve the model: a real dropdown selection wins; else the manual path (quotes
        # stripped so a Windows "Copy as path" value works as-is).
        resolved = _resolve_path(model, model_path)
        if not resolved or not os.path.isfile(resolved):
            return _err(f"Model file not found: {resolved or '(none selected)'}")

        try:
            extra = _parse_extra_args(extra_load_args)
        except ValueError as e:
            return _err(str(e))

        system_prompt = _with_context(system_prompt, context)
        user_prompt, directive = _apply_directive(user_prompt, thinking_directive, custom_directive)
        eff_format, eff_grammar, eff_system = _apply_output_format(output_format, grammar, system_prompt)

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
            "stop": [s for s in (stop or "").splitlines() if s.strip()],
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
            "flash_attn": flash_attn,
            "kv_cache_type": kv_cache_type,
            "seed": seed,
            "output_format": eff_format,
            "grammar": eff_grammar,
            "extra_load_args": extra,
            "verbose": False,
        }
        load_sig = (resolved, int(n_ctx), int(n_gpu_layers), int(n_batch),
                    bool(flash_attn), kv_cache_type, None, None,
                    json.dumps(extra, sort_keys=True, default=str))

        return _generate_and_format(req, load_sig, max_tokens, unload_comfy_models,
                                    unload_llm_after_run, directive, strip_think,
                                    answer_marker, HELP_TEXT)


class LocalLLMVisionGGUF:
    """Multimodal twin of the text node: same engine + sampling, plus an mmproj projector and
    an image input. Shares the worker and the generation core; only the loader inputs differ."""

    @classmethod
    def INPUT_TYPES(cls):
        # Reuse the text node's widgets verbatim (single source of truth) and inject the
        # vision-only inputs right after model_path so the shared params can't drift.
        req = {}
        for k, v in LocalLLMTextGGUF.INPUT_TYPES()["required"].items():
            req[k] = v
            if k == "model_path":
                req["mmproj"] = (_list_mmproj(), {"tooltip": "Projector mmproj .gguf from ComfyUI/models/llm (mmproj-named files first). Choose the placeholder to type any path in mmproj_path."})
                req["mmproj_path"] = ("STRING", {"default": "", "tooltip": "Full path to the mmproj .gguf, used when 'mmproj' is the placeholder. Surrounding quotes are stripped."})
                req["vision_handler"] = (_VISION_HANDLER_LABELS, {"default": "auto (MTMD)", "tooltip": "auto (MTMD) is llama.cpp's generic multimodal loader and handles most modern vision GGUFs. Switch to the model's family only if auto fails to load it or returns garbage."})
                req["image"] = ("IMAGE", {"tooltip": "Image(s) to analyze. A batch sends every frame to the model."})
                req["image_max_side"] = ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 64, "tooltip": "Downscale each image so its longest side is at most this many px before sending (the model resizes internally anyway). 0 = full size."})
        return {"required": req}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "INT", "INT")
    RETURN_NAMES = ("text", "thoughts", "finish_reason", "sys_tokens", "user_tokens", "output_tokens", "gen_seconds", "help", "thoughts_tokens", "answer_tokens")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, model, model_path, mmproj, mmproj_path, vision_handler, image, image_max_side,
            system_prompt, user_prompt, max_tokens, temperature, top_p, top_k, min_p,
            repeat_penalty, stop, n_ctx, n_gpu_layers, n_batch, flash_attn, kv_cache_type,
            seed, unload_comfy_models, unload_llm_after_run, strip_think, context="",
            thinking_directive="model default", custom_directive="", output_format="text",
            grammar="", answer_marker="", extra_load_args=""):

        resolved = _resolve_path(model, model_path)
        if not resolved or not os.path.isfile(resolved):
            return _err(f"Model file not found: {resolved or '(none selected)'}", VISION_HELP_TEXT)
        mmproj_resolved = _resolve_path(mmproj, mmproj_path)
        if not mmproj_resolved or not os.path.isfile(mmproj_resolved):
            return _err(f"mmproj file not found: {mmproj_resolved or '(none selected)'} — vision needs the model's mmproj .gguf.", VISION_HELP_TEXT)
        try:
            extra = _parse_extra_args(extra_load_args)
        except ValueError as e:
            return _err(str(e), VISION_HELP_TEXT)
        try:
            images = _encode_images(image, int(image_max_side))
        except Exception as e:
            return _err(f"Failed to encode image(s): {e}", VISION_HELP_TEXT)

        handler_key = _VISION_HANDLER_KEY.get(vision_handler, "auto")
        system_prompt = _with_context(system_prompt, context)
        user_prompt, directive = _apply_directive(user_prompt, thinking_directive, custom_directive)
        eff_format, eff_grammar, eff_system = _apply_output_format(output_format, grammar, system_prompt)

        req = {
            "model_path": resolved,
            "mmproj_path": mmproj_resolved,
            "vision_handler": handler_key,
            "images": images,
            "system_prompt": eff_system,
            "user_prompt": user_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "stop": [s for s in (stop or "").splitlines() if s.strip()],
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
            "flash_attn": flash_attn,
            "kv_cache_type": kv_cache_type,
            "seed": seed,
            "output_format": eff_format,
            "grammar": eff_grammar,
            "extra_load_args": extra,
            "verbose": False,
        }
        load_sig = (resolved, int(n_ctx), int(n_gpu_layers), int(n_batch),
                    bool(flash_attn), kv_cache_type, mmproj_resolved, handler_key,
                    json.dumps(extra, sort_keys=True, default=str))

        return _generate_and_format(req, load_sig, max_tokens, unload_comfy_models,
                                    unload_llm_after_run, directive, strip_think,
                                    answer_marker, VISION_HELP_TEXT)


NODE_CLASS_MAPPINGS = {"LocalLLMTextGGUF": LocalLLMTextGGUF, "LocalLLMVisionGGUF": LocalLLMVisionGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"LocalLLMTextGGUF": "Local LLM (GGUF, text)", "LocalLLMVisionGGUF": "Local LLM (GGUF, vision)"}
