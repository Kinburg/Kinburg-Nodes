import os
import sys
import json
import re
import time
import hashlib
import atexit
import threading
import subprocess

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(NODE_DIR, "gguf_worker.py")
RESP_PREFIX = "@@LLM_RESPONSE@@"
PROG_PREFIX = "@@LLM_PROGRESS@@"
TOK_PREFIX = "@@LLM_TOKEN@@"

# Custom type of the config bundle emitted by Local LLM Settings and consumed by the LLM nodes.
LLM_CONFIG = "KINBURG_LLM_CONFIG"
# Vision sub-bundle (mmproj / handler / max side) that plugs optionally into Local LLM Settings.
VISION_CONFIG = "KINBURG_VISION_CONFIG"
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
Grammar runs stream like any other, so the token progress bar and the live log follow
them token by token.

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


def _list_gguf_rel(d):
    """Every .gguf under `d`, RECURSIVELY, as paths relative to `d` with forward-slash separators —
    so models organized into subfolders show up as 'family/model.gguf' in the dropdown (the newer
    ComfyUI frontend even renders '/' as nested submenus), and the choice serializes the same on any
    OS. Sorted case-insensitively so files in the same subfolder group together."""
    out = []
    if d and os.path.isdir(d):
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".gguf"):
                    out.append(os.path.relpath(os.path.join(root, f), d).replace(os.sep, "/"))
    return sorted(out, key=str.lower)


def _list_models():
    return [PLACEHOLDER] + _list_gguf_rel(_gguf_dir())


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
    # Force the worker's stdio to UTF-8 too (belt-and-suspenders with its own reconfigure), so
    # the ensure_ascii=False protocol round-trips non-ASCII cleanly on Windows.
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, WORKER, "--serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        bufsize=1,
        cwd=NODE_DIR,
        env=env,
    )
    _worker_proc = proc
    _worker_sig = load_sig
    return proc


def _ensure_worker(load_sig):
    if (_worker_proc is not None and _worker_proc.poll() is None
            and _worker_sig == load_sig):
        return _worker_proc
    return _start_worker(load_sig)


def count_tokens(cfg, text):
    """Count tokens of `text` under the config's model tokenizer, via the worker (vocab-only —
    no weights/VRAM). Returns (token_count, error_str); token_count is -1 on error. Reuses a
    running worker without disturbing any loaded generation model."""
    g = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: d)
    resolved = _resolve_path(g("model", PLACEHOLDER), g("model_path", ""))
    if not resolved or not os.path.isfile(resolved):
        return (-1, f"Model file not found: {resolved or '(none selected)'} — check the Settings node.")
    req = {"cmd": "count", "model_path": resolved, "text": text or ""}
    with _worker_lock:
        try:
            proc = (_worker_proc if (_worker_proc is not None and _worker_proc.poll() is None)
                    else _start_worker(("__count__",)))
            data = _request(proc, req)
            if data.get("status") == "error" and "worker exited" in data.get("message", ""):
                data = _request(_start_worker(("__count__",)), req)
        except Exception as e:
            _shutdown_worker()
            return (-1, f"Worker communication failed: {e}")
    if data.get("status") == "success":
        return (int(data.get("token_count", 0)), "")
    print("[LocalLLM] token count error:\n", data.get("traceback", ""))
    return (-1, data.get("message", "unknown error"))


def count_prompt(cfg, user_prompt, image=None):
    """Exact prompt token count (text + chat template + image tokens) via a 1-token prefill on
    the full model — loads the vision projector when an `image` is given (Path B). The measuring
    context is forced large enough that the prefill can't overflow n_ctx. Returns
    (prompt_tokens, error); prompt_tokens is -1 on error."""
    base = cfg if isinstance(cfg, dict) else {}
    cfg2 = dict(base)
    cfg2["n_ctx"] = max(int(base.get("n_ctx", 4096) or 4096), 8192)  # headroom for the probe
    err, ctx = build_llm_request(cfg2, user_prompt, image=image)
    if err:
        return (-1, err)
    req = ctx["req"]
    req["count_only"] = True
    with _worker_lock:
        try:
            proc = _ensure_worker(ctx["load_sig"])
            data = _request(proc, req)
            if data.get("status") == "error" and "worker exited" in data.get("message", ""):
                data = _request(_start_worker(ctx["load_sig"]), req)
        except Exception as e:
            _shutdown_worker()
            return (-1, f"Worker communication failed: {e}")
    if data.get("status") == "success":
        return (int(data.get("prompt_tokens", 0)), "")
    print("[LocalLLM] count_prompt error:\n", data.get("traceback", ""))
    return (-1, data.get("message", "unknown error"))


def count_image_tokens(cfg, image):
    """Lean per-image token count via mtmd (clip only — no LLM weights, no decode). `image` is an
    IMAGE tensor/batch. Returns (list[int] per frame | None, error). None/error → caller should
    fall back to the full-model prefill (this path needs the clip to tokenize weight-free, which
    a given build may not support)."""
    g = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: d)
    model = _resolve_path(g("model", PLACEHOLDER), g("model_path", ""))
    if not model or not os.path.isfile(model):
        return (None, f"model file not found: {model or '(none selected)'}")
    mmproj = _resolve_path(g("mmproj", PLACEHOLDER), g("mmproj_path", ""))
    if not mmproj or not os.path.isfile(mmproj):
        return (None, "no mmproj (vision) in the config — connect a Vision Settings node")
    try:
        uris = _encode_images(image, int(g("image_max_side", 1024)))
    except Exception as e:
        return (None, f"image encode failed: {e}")
    req = {"cmd": "count_mtmd", "model_path": model, "mmproj_path": mmproj,
           "n_gpu_layers": int(g("n_gpu_layers", -1)), "images": uris}
    with _worker_lock:
        try:
            proc = (_worker_proc if (_worker_proc is not None and _worker_proc.poll() is None)
                    else _start_worker(("__count__",)))
            data = _request(proc, req)
            if data.get("status") == "error" and "worker exited" in data.get("message", ""):
                data = _request(_start_worker(("__count__",)), req)
        except Exception as e:
            _shutdown_worker()
            return (None, f"Worker communication failed: {e}")
    if data.get("status") == "success":
        return ([int(x) for x in data.get("image_tokens", [])], "")
    print("[LocalLLM] count_image_tokens error:\n", data.get("traceback", ""))
    return (None, data.get("message", "unknown error"))


def _request(proc, req, progress_cb=None, token_cb=None):
    try:
        proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
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
        if line.startswith(TOK_PREFIX):
            if token_cb is not None:
                try:
                    token_cb(json.loads(line[len(TOK_PREFIX):]))
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
- VRAM: the projector is loaded on the first run that actually has an image and released again
  on the first run without one, so it only occupies VRAM while vision is in use. The model
  itself stays put through that — a run with a picture and a run without share one loaded
  model, which is what makes a chat that mixes the two cheap. Keep both unload toggles ON to
  free the model as well.
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
    """models/llm .gguf files (recursive, subfolders included) for the mmproj dropdown, with
    mmproj-named ones (or ones under an mmproj folder) first."""
    allg = _list_gguf_rel(_gguf_dir())
    files = [f for f in allg if "mmproj" in f.lower()] + [f for f in allg if "mmproj" not in f.lower()]
    return [PLACEHOLDER] + files


def _resolve_path(choice, manual):
    """A models/llm dropdown selection wins; else the manual path (quotes/space stripped). The
    dropdown value may be a subfolder-relative path ('family/model.gguf'); split on '/' so it
    rejoins with the OS separator (a plain root filename has no '/', so this is a no-op for it)."""
    if choice and choice != PLACEHOLDER:
        d = _gguf_dir()
        return os.path.join(d, *choice.split("/")) if d else choice
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
    (0 = no downscale). Encoded here (the node has PIL/torch) so the worker stays llama-only.

    `image` may also be a LIST of such tensors, which is how the chat node sends a turn's
    attachments: they come off disk one file at a time and need not share a resolution, so they
    cannot be stacked into one batch."""
    import io
    import base64
    import numpy as np
    from PIL import Image

    frames = []
    for t in (image if isinstance(image, (list, tuple)) else [image]):
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        if arr.ndim == 3:
            arr = arr[None, ...]
        frames.extend(arr)
    uris = []
    for im in frames:
        pil = Image.fromarray((np.clip(im, 0.0, 1.0) * 255.0).astype("uint8")).convert("RGB")
        if max_side and max(pil.size) > max_side:
            s = max_side / max(pil.size)
            pil = pil.resize((max(1, round(pil.width * s)), max(1, round(pil.height * s))),
                             Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        # JPEG (q92) instead of PNG: far smaller base64 payload over the pipe and faster to
        # encode; the vision model resizes/re-encodes internally, so q92 is visually lossless
        # for image understanding. (pil is already RGB above, so no alpha to worry about.)
        pil.save(buf, format="JPEG", quality=92)
        uris.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
    return uris


def _generate_and_format(req, load_sig, max_tokens, unload_comfy_models,
                         unload_llm_after_run, directive, strip_think, answer_marker, help_text,
                         token_cb=None, show_progress=True, stats=None):
    """Shared core for both LLM nodes: optionally free ComfyUI VRAM, talk to the worker with a
    live token progress bar, then split reasoning out and return the 10-output tuple.
    `show_progress=False` suppresses the per-token progress bar — used by callers (e.g. the
    Vision Judge) that drive their own image-level progress bar for the whole node.
    `stats` (a dict, when passed) is filled on success with this call's token / context-fill
    figures (prompt_tokens, output_tokens, context_used, n_ctx, finish_reason) — the 10-tuple
    return is unchanged, so existing callers are unaffected."""
    if unload_comfy_models:
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache(True)
        except Exception as e:
            print(f"[LocalLLM] unload_all_models failed: {e}")

    pbar = None
    if show_progress:
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
            data = _request(proc, req, _progress, token_cb)
            if (data.get("status") == "error"
                    and "worker exited" in data.get("message", "")):
                proc = _start_worker(load_sig)
                data = _request(proc, req, _progress, token_cb)
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
        if isinstance(stats, dict):
            stats.update({
                "prompt_tokens": int(data.get("prompt_tokens", 0)),
                "output_tokens": out_tok,
                "sys_tokens": int(data.get("sys_tokens", 0)),
                "user_tokens": int(data.get("user_tokens", 0)),
                "context_used": int(data.get("context_used", 0)),
                "n_ctx": int(data.get("n_ctx", 0)),
                "finish_reason": data.get("finish_reason", ""),
            })
        return (text, thoughts, data.get("finish_reason", ""),
                int(data.get("sys_tokens", 0)), int(data.get("user_tokens", 0)),
                out_tok, gen_seconds, help_text, thoughts_tokens, answer_tokens)

    print("[LocalLLM] worker error:\n", data.get("traceback", ""))
    return _err(data.get("message", "unknown"), help_text)


def _base_config_widgets():
    """The model / sampling / loader / reasoning / output widgets carried by the Local LLM
    Settings node. The LLM nodes read these from the config bundle instead of hosting them."""
    return {
        "model": (_list_models(), {"tooltip": "Pick a .gguf from ComfyUI/models/llm (subfolders included — organize models into folders and they show as 'folder/model.gguf'). Choose the placeholder to type any path in model_path"}),
        "model_path": ("STRING", {"default": "", "tooltip": "Full path to a .gguf, used when 'model' is the placeholder. Surrounding quotes (e.g. from Windows 'Copy as path') are stripped automatically."}),
        "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
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
        "unload_llm_after_run": ("BOOLEAN", {"default": False, "tooltip": "Free the LLM from VRAM after each run. Off (default) keeps it loaded for fast repeated runs / chat; turn ON in image workflows to free VRAM."}),
        "strip_think": ("BOOLEAN", {"default": True, "tooltip": "Keep reasoning out of the 'text' output (it still goes to the 'thoughts' output). Off = leave raw reasoning in 'text'"}),
        "answer_marker": ("STRING", {"default": "", "tooltip": "For models that print reasoning WITHOUT <think> tags: the answer is taken after the LAST occurrence of this marker, everything before goes to 'thoughts'. Empty = use <think> tags."}),
        "thinking_directive": (["model default", "/no_think (Qwen3)", "/think (Qwen3)", "custom"], {"default": "model default", "tooltip": "Append a reasoning-control directive to the prompt. /no_think makes Qwen3-style models skip the <think> phase; 'custom' uses the field below."}),
        "custom_directive": ("STRING", {"default": "", "tooltip": "Directive text appended to the prompt when thinking_directive = custom (e.g. /no_think)"}),
        "output_format": (["text", "json_object", "gbnf_grammar", "ideogram4_json"], {"default": "text", "tooltip": "Output: free text · valid JSON · custom GBNF grammar (field below) · ideogram4_json. Grammar modes run without the live progress bar"}),
        "grammar": ("STRING", {"multiline": True, "default": "", "tooltip": "GBNF grammar text, used when output_format = gbnf_grammar"}),
        "extra_load_args": ("STRING", {"multiline": True, "default": "", "tooltip": "Advanced: extra keyword args for llama-cpp-python's Llama() loader. One per line as key=value or a JSON object. These are Python-binding args, NOT llama.cpp CLI flags. Unknown keys are ignored. Changing this reloads the model."}),
        "chat_template_path": ("STRING", {"default": "", "tooltip": "Advanced: path to a chat_template.jinja file that OVERRIDES the model's built-in chat template. Empty (default) = use the template embedded in the GGUF, which is correct for almost every model. Only needed when a model ships a broken/missing embedded template, or you want a specific template variant. TEXT models only — ignored when an mmproj (vision) is active, since vision uses its own formatting. Surrounding quotes are stripped. Changing this reloads the model."}),
    }


# Per-node override for freeing the model after a run, independent of the shared config.
UNLOAD_MODES = ["config default", "unload after run", "keep loaded"]


def resolve_unload(mode, cfg):
    """Resolve a node's `unload_after_run` selector to a bool. 'config default' follows the
    Settings node's `unload_llm_after_run`; the other two force the choice for THIS node only."""
    if mode == "unload after run":
        return True
    if mode == "keep loaded":
        return False
    g = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: d)
    return bool(g("unload_llm_after_run", False))


def build_llm_request(cfg, user_prompt, image=None, history=None,
                      system_override=None, grammar_override=None):
    """From a KINBURG_LLM_CONFIG bundle + this call's prompt, build the worker request and its
    load signature. `system_override` (a non-empty string) replaces the config's system_prompt;
    `grammar_override` (a non-empty string) replaces the grammar and forces gbnf_grammar output.
    Returns (error_str, None) on failure, else (None, ctx) where ctx holds the `req`, `load_sig`,
    `help`, `directive`, and the `_generate_and_format` knobs."""
    g = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: d)

    resolved = _resolve_path(g("model", PLACEHOLDER), g("model_path", ""))
    if not resolved or not os.path.isfile(resolved):
        return (f"Model file not found: {resolved or '(none selected)'} — check the Settings node.", None)

    mmproj_resolved = _resolve_path(g("mmproj", PLACEHOLDER), g("mmproj_path", ""))
    use_vision = False
    if image is not None:
        if mmproj_resolved and os.path.isfile(mmproj_resolved):
            use_vision = True
        else:
            return ("An image is connected, but the Settings node has no mmproj set. "
                    "Add an mmproj (vision projector .gguf) to use vision.", None)

    try:
        extra = _parse_extra_args(g("extra_load_args", ""))
    except ValueError as e:
        return (str(e), None)

    images = []
    if use_vision:
        try:
            images = _encode_images(image, int(g("image_max_side", 1024)))
        except Exception as e:
            return (f"Failed to encode image(s): {e}", None)

    # Optional chat-template override: a chat_template.jinja that REPLACES the model's embedded
    # template. Read here (not in the worker) so a bad path fails with a clean node error. It only
    # ever applies to TEXT turns — the vision path uses its own mtmd handler — but it is read
    # regardless of `use_vision` so that the load signature below is the same either way. Make it
    # conditional and one config would produce two signatures, which is exactly the reload this
    # per-request handler swap exists to avoid.
    chat_template = ""
    ct_path = (g("chat_template_path", "") or "").strip().strip('"').strip("'").strip()
    if ct_path:
        if not os.path.isfile(ct_path):
            return (f"chat_template_path not found: {ct_path}", None)
        try:
            with open(ct_path, "r", encoding="utf-8") as f:
                chat_template = f.read()
        except Exception as e:
            return (f"chat_template_path: cannot read '{ct_path}': {e}", None)
        if not chat_template.strip():
            return (f"chat_template_path is empty: {ct_path}", None)

    up, directive = _apply_directive(user_prompt, g("thinking_directive", "model default"), g("custom_directive", ""))
    # A connected system_override replaces the config's persona; context is still appended.
    sys_base = (system_override if (isinstance(system_override, str) and system_override.strip())
                else g("system_prompt", "You are a helpful assistant."))
    system = _with_context(sys_base, g("context", ""))
    eff_format, eff_grammar, eff_system = _apply_output_format(g("output_format", "text"), g("grammar", ""), system)
    # A connected grammar_override wins: use it and force GBNF-constrained output.
    if isinstance(grammar_override, str) and grammar_override.strip():
        eff_format, eff_grammar = "gbnf_grammar", grammar_override

    req = {
        "model_path": resolved, "system_prompt": eff_system, "user_prompt": up,
        "max_tokens": int(g("max_tokens", 512)), "temperature": float(g("temperature", 0.7)),
        "top_p": float(g("top_p", 0.95)), "top_k": int(g("top_k", 40)),
        "min_p": float(g("min_p", 0.0)), "repeat_penalty": float(g("repeat_penalty", 1.1)),
        "stop": [s for s in (g("stop", "") or "").splitlines() if s.strip()],
        "n_ctx": int(g("n_ctx", 4096)), "n_gpu_layers": int(g("n_gpu_layers", -1)),
        "n_batch": int(g("n_batch", 512)), "flash_attn": bool(g("flash_attn", False)),
        "kv_cache_type": g("kv_cache_type", "f16"), "seed": int(g("seed", 0)),
        "output_format": eff_format, "grammar": eff_grammar,
        "extra_load_args": extra, "verbose": False,
        "chat_template": chat_template,
    }
    if history:
        req["history"] = history
    handler_key = _VISION_HANDLER_KEY.get(g("vision_handler", "auto (MTMD)"), "auto")
    if use_vision:
        req["mmproj_path"] = mmproj_resolved
        req["vision_handler"] = handler_key
        req["images"] = images

    # What a change here costs: the whole worker PROCESS is killed and restarted (_ensure_worker).
    # The projector is deliberately absent — the worker attaches and releases it per request, so a
    # picture in the middle of a text chat no longer restarts anything. Mirrors _load_sig there.
    load_sig = (resolved, req["n_ctx"], req["n_gpu_layers"], req["n_batch"], req["flash_attn"],
                req["kv_cache_type"],
                json.dumps(extra, sort_keys=True, default=str),
                hashlib.sha1(chat_template.encode("utf-8")).hexdigest() if chat_template else "")

    ctx = {
        "req": req, "load_sig": load_sig,
        "help": VISION_HELP_TEXT if use_vision else HELP_TEXT, "directive": directive,
        "max_tokens": req["max_tokens"],
        "unload_comfy": bool(g("unload_comfy_models", True)),
        "unload_llm": bool(g("unload_llm_after_run", False)),
        "strip_think": bool(g("strip_think", True)),
        "answer_marker": g("answer_marker", ""),
    }
    return (None, ctx)


class LocalLLMGGUF:
    """One-shot GGUF LLM. All settings come from a Local LLM Settings (GGUF) `config`. Text by
    default; connect an `image` (with an mmproj on the Settings node, via Vision Settings) for
    vision. An optional `system_override` replaces the config's system prompt for this node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (LLM_CONFIG, {"tooltip": "Wire a 'Local LLM Settings (GGUF)' node here — it carries the model, system prompt, sampling, reasoning, output format, etc."}),
                "user_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The prompt / question for the model."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image(s) for vision — needs an mmproj on the Settings node (Vision Settings). Omit for a text-only run."}),
                "system_override": ("STRING", {"forceInput": True, "tooltip": "Optional: replaces the config's system_prompt for this node (connect-only). Context still applies."}),
                "grammar_override": ("STRING", {"forceInput": True, "tooltip": "Optional: a GBNF grammar (connect-only) that replaces the config's grammar and forces gbnf_grammar output for this node."}),
                "unload_after_run": (UNLOAD_MODES, {"default": "config default", "tooltip": "Free the model from VRAM after THIS node runs, without touching the shared config. 'config default' follows the Settings node; 'unload after run' frees VRAM (a different model runs next); 'keep loaded' stays warm (the same model runs next)."}),
                "live_preview": ("BOOLEAN", {"default": False, "tooltip": "Stream the generated text to a 'Kinburg Live Log' node as it's written, token by token. Grammar / JSON runs stream too, so a card built via grammar_override types itself out as the model constrains it."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "INT", "INT")
    RETURN_NAMES = ("text", "thoughts", "finish_reason", "sys_tokens", "user_tokens", "output_tokens", "gen_seconds", "help", "thoughts_tokens", "answer_tokens")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, config, user_prompt, image=None, system_override=None, grammar_override=None,
            unload_after_run="config default", live_preview=False, unique_id=None):
        err, ctx = build_llm_request(config, user_prompt, image=image,
                                     system_override=system_override, grammar_override=grammar_override)
        if err:
            return _err(err, VISION_HELP_TEXT if image is not None else HELP_TEXT)
        unload_llm = resolve_unload(unload_after_run, config)

        # Optional live text streaming to a Kinburg Live Log node over ComfyUI's websocket — same
        # mechanism the Chat node uses. Text runs only; a grammar run takes the worker's non-stream
        # path, so token_cb never fires and the log just gets the final text on 'done'.
        token_cb, emit = None, None
        if live_preview:
            try:
                from server import PromptServer
                nid = str(unique_id[0] if isinstance(unique_id, list) else unique_id)

                def emit(payload):
                    try:
                        PromptServer.instance.send_sync("kinburg.llm", {"id": nid, **payload})
                    except Exception:
                        pass

                ctx["req"]["stream_text"] = True

                def token_cb(delta):
                    emit({"event": "delta", "delta": delta})

                # The log counts deltas live against this ceiling ("142/512 tok"); n_ctx lets it
                # show the context fill once the exact figures land on 'done', and answer_marker
                # lets it split reasoning from the answer exactly the way _split_reasoning does.
                emit({"event": "start", "max_tokens": int(ctx["max_tokens"]),
                      "n_ctx": int(ctx["req"].get("n_ctx", 0) or 0),
                      "answer_marker": ctx["answer_marker"] or ""})
            except Exception:
                token_cb, emit = None, None

        stats = {} if emit else None
        out = _generate_and_format(ctx["req"], ctx["load_sig"], ctx["max_tokens"], ctx["unload_comfy"],
                                   unload_llm, ctx["directive"], ctx["strip_think"],
                                   ctx["answer_marker"], ctx["help"], token_cb=token_cb, stats=stats)
        if emit:
            emit({"event": "done", "text": out[0], "finish_reason": out[2], "gen_seconds": out[6],
                  "max_tokens": int(ctx["max_tokens"]), "output_tokens": int(out[5]),
                  "prompt_tokens": int((stats or {}).get("prompt_tokens", 0) or 0),
                  "context_used": int((stats or {}).get("context_used", 0) or 0),
                  "n_ctx": int((stats or {}).get("n_ctx", 0) or ctx["req"].get("n_ctx", 0) or 0)})
        return out


NODE_CLASS_MAPPINGS = {"LocalLLMGGUF": LocalLLMGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"LocalLLMGGUF": "Local LLM (GGUF)"}
