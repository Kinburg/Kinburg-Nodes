import sys
import os
import re
import json
import gc
import glob
import ctypes
import traceback
import importlib.util
import importlib.metadata

RESP_PREFIX = "@@LLM_RESPONSE@@"
PROG_PREFIX = "@@LLM_PROGRESS@@"
TOK_PREFIX = "@@LLM_TOKEN@@"


def _prepare_cuda():
    """The prebuilt llama-cpp-python wheel does NOT bundle the CUDA runtime, and the
    llama_cpp loader opens llama.dll with a flag (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
    under which directories added via add_dll_directory are ignored when resolving
    dependencies. So we PRELOAD the required CUDA DLLs by full path from torch\\lib.
    Once they are loaded into the process, ggml-cuda.dll resolves them by name.
    This only works when the wheel's CUDA *major* matches torch's (install.py picks
    the matching wheel); ggml-cuda.dll links cudart64_<major>.dll by exact name, so a
    cu124 wheel on a CUDA-13 torch can't be satisfied by cudart64_13.dll. torch itself
    is NOT imported (find_spec only locates the module, it does not execute it), so no
    second CUDA context is created."""
    try:
        spec = importlib.util.find_spec("torch")
        if not (spec and spec.origin):
            return
        tl = os.path.join(os.path.dirname(spec.origin), "lib")
        if not os.path.isdir(tl):
            return
        os.add_dll_directory(tl)
        for pattern in ("cudart64_*.dll", "cublasLt64_*.dll", "cublas64_*.dll"):
            for dll in sorted(glob.glob(os.path.join(tl, pattern))):
                try:
                    ctypes.CDLL(dll)
                except Exception:
                    pass
    except Exception:
        pass


def _import_hint():
    """Actionable hint appended to an llama_cpp import failure -- almost always a
    CUDA-major mismatch between the installed wheel and torch."""
    cu = None
    try:
        v = importlib.metadata.version("torch")
        m = re.search(r"\+cu(\d+)", v)
        cu = m.group(1) if m else None
    except Exception:
        pass
    hint = ("\n[Kinburg-Nodes] This usually means the installed llama-cpp-python was "
            "built for a different CUDA version than torch. Re-run the installer with "
            "THIS ComfyUI's python:\n"
            "  python custom_nodes/Kinburg-Nodes/install.py")
    if cu:
        hint += f"\n(torch reports CUDA build cu{cu}.)"
    return hint


def _send(obj):
    sys.stdout.write(RESP_PREFIX + json.dumps(obj, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _progress(n):
    sys.stdout.write(PROG_PREFIX + str(n) + "\n")
    sys.stdout.flush()


def _token(s):
    # JSON-encode so newlines in the delta don't break the one-line protocol.
    sys.stdout.write(TOK_PREFIX + json.dumps(s, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _load_sig(req):
    return (
        req.get("model_path"),
        int(req.get("n_ctx", 4096)),
        int(req.get("n_gpu_layers", -1)),
        int(req.get("n_batch", 512)),
        bool(req.get("flash_attn", False)),
        req.get("kv_cache_type", "f16"),
        req.get("mmproj_path") or None,
        (req.get("vision_handler") or "auto") if (req.get("mmproj_path") or "").strip() else None,
        json.dumps(req.get("extra_load_args") or {}, sort_keys=True, default=str),
    )


def _merge_extra_load_args(base, extra):
    """Merge user-supplied Llama() kwargs into `base`, keeping only names Llama.__init__
    accepts (unless it takes **kwargs). Unknown keys are dropped with a console note so a
    typo or a CLI-only flag can't crash the loader."""
    if not extra:
        return base
    import inspect
    from llama_cpp import Llama
    try:
        params = inspect.signature(Llama.__init__).parameters
        accepted = set(params)
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    except (ValueError, TypeError):
        accepted, has_var_kw = set(), True  # can't introspect -> pass everything through
    for k, v in extra.items():
        if has_var_kw or k in accepted:
            base[k] = v          # user args override the defaults above
        else:
            sys.stdout.write(f"[LocalLLM worker] ignoring unknown Llama() arg: {k}\n")
            sys.stdout.flush()
    return base


# Vision chat-handler key (sent by the node) -> class name in llama_cpp.llama_chat_format.
# "auto" uses MTMD, llama.cpp's generic multimodal loader, which handles most modern vision
# GGUFs from the mmproj metadata; the rest are family-specific fallbacks.
_HANDLER_MAP = {
    "auto": "MTMDChatHandler",
    "llava-1.5": "Llava15ChatHandler",
    "llava-1.6": "Llava16ChatHandler",
    "qwen2-vl": "Qwen25VLChatHandler",
    "minicpm-v-2.6": "MiniCPMv26ChatHandler",
    "moondream": "MoondreamChatHandler",
    "nanollava": "NanoLlavaChatHandler",
    "llama3-vision": "Llama3VisionAlphaChatHandler",
    "obsidian": "ObsidianChatHandler",
    "gemma": "Gemma4ChatHandler",
}


def _make_chat_handler(req):
    """Build a vision chat handler for the request's mmproj, or None for a text-only run."""
    mmproj = (req.get("mmproj_path") or "").strip()
    if not mmproj:
        return None
    import inspect
    from llama_cpp import llama_chat_format as cf
    key = (req.get("vision_handler") or "auto").strip().lower()
    cls = getattr(cf, _HANDLER_MAP.get(key, "MTMDChatHandler"), None) or cf.MTMDChatHandler
    kwargs = {"clip_model_path": mmproj, "verbose": bool(req.get("verbose", False))}
    try:  # MTMD takes use_gpu; the legacy handlers don't
        if "use_gpu" in inspect.signature(cls.__init__).parameters:
            kwargs["use_gpu"] = int(req.get("n_gpu_layers", -1)) != 0
    except Exception:
        pass
    return cls(**kwargs)


def _user_content(req):
    """User message content: text-only string, or a list of image_url parts + the text when
    the request carries `images` (base64 data: URIs encoded by the node)."""
    prompt = req.get("user_prompt", "")
    images = req.get("images") or []
    if not images:
        return prompt
    content = [{"type": "image_url", "image_url": {"url": u}} for u in images]
    content.append({"type": "text", "text": prompt})
    return content


def _kv_type(llama_cpp, name):
    return {
        "f16": getattr(llama_cpp, "GGML_TYPE_F16", 1),
        "q8_0": getattr(llama_cpp, "GGML_TYPE_Q8_0", 8),
        "q4_0": getattr(llama_cpp, "GGML_TYPE_Q4_0", 2),
    }.get(name, getattr(llama_cpp, "GGML_TYPE_F16", 1))


def main():
    _prepare_cuda()
    try:
        import llama_cpp
        from llama_cpp import Llama
    except Exception as e:
        _send({"status": "error", "message": f"import llama_cpp failed: {e}{_import_hint()}",
               "traceback": traceback.format_exc()})
        return

    llm = None
    chat_handler = None
    current_sig = None

    while True:
        line = sys.stdin.readline()
        if line == "":          # EOF: parent closed the pipe
            break
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception as e:
            _send({"status": "error", "message": f"bad request json: {e}"})
            continue

        if req.get("cmd") == "exit":
            break

        try:
            sig = _load_sig(req)
            if llm is None or sig != current_sig:
                if llm is not None:
                    del llm
                    llm = None
                    chat_handler = None  # drop the projector/clip context too
                    gc.collect()
                kv = req.get("kv_cache_type", "f16")
                kv_type = _kv_type(llama_cpp, kv)
                flash = bool(req.get("flash_attn", False)) or kv != "f16"
                chat_handler = _make_chat_handler(req)  # None for a text-only model
                load_kwargs = dict(
                    model_path=req["model_path"],
                    n_ctx=int(req.get("n_ctx", 4096)),
                    n_gpu_layers=int(req.get("n_gpu_layers", -1)),
                    n_batch=int(req.get("n_batch", 512)),
                    seed=int(req.get("seed", -1)),
                    flash_attn=flash,
                    type_k=kv_type,
                    type_v=kv_type,
                    chat_handler=chat_handler,
                    verbose=bool(req.get("verbose", False)),
                )
                _merge_extra_load_args(load_kwargs, req.get("extra_load_args") or {})
                llm = Llama(**load_kwargs)
                current_sig = sig

            messages = []
            sys_prompt = (req.get("system_prompt") or "").strip()
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            # Prior chat turns (empty for the single-shot text/vision nodes). Images ride on the
            # final user message only (added via _user_content below).
            for m in (req.get("history") or []):
                r, c = m.get("role"), m.get("content", "")
                if r in ("user", "assistant") and isinstance(c, str) and c:
                    messages.append({"role": r, "content": c})
            messages.append({"role": "user", "content": _user_content(req)})

            gen_kwargs = dict(
                messages=messages,
                max_tokens=int(req.get("max_tokens", 512)),
                temperature=float(req.get("temperature", 0.7)),
                top_p=float(req.get("top_p", 0.95)),
                top_k=int(req.get("top_k", 40)),
                min_p=float(req.get("min_p", 0.0)),
                repeat_penalty=float(req.get("repeat_penalty", 1.1)),
                seed=int(req.get("seed", -1)),
            )
            stop = req.get("stop") or []
            if stop:
                gen_kwargs["stop"] = stop

            out_fmt = req.get("output_format", "text")
            if out_fmt == "json_object":
                gen_kwargs["response_format"] = {"type": "json_object"}
            elif out_fmt == "gbnf_grammar":
                gtext = (req.get("grammar") or "").strip()
                if gtext:
                    gen_kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(gtext)

            use_stream = "grammar" not in gen_kwargs
            gen_kwargs["stream"] = use_stream

            parts = []
            n = 0
            finish_reason = ""
            if use_stream:
                stream_text = bool(req.get("stream_text"))
                for chunk in llm.create_chat_completion(**gen_kwargs):
                    ch = chunk["choices"][0]
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        parts.append(piece)
                        n += 1
                        _progress(n)
                        if stream_text:
                            _token(piece)
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                text = "".join(parts)
            else:
                out = llm.create_chat_completion(**gen_kwargs)
                ch = out["choices"][0]
                text = ch["message"]["content"]
                finish_reason = ch.get("finish_reason", "")
                n = int((out.get("usage") or {}).get("completion_tokens", 0))
                _progress(n)

            def _count(s):
                s = s or ""
                if not s:
                    return 0
                try:
                    return len(llm.tokenize(s.encode("utf-8"), add_bos=False, special=False))
                except Exception:
                    return 0

            _send({
                "status": "success",
                "output": text,
                "finish_reason": finish_reason,
                "sys_tokens": _count(req.get("system_prompt")),
                "user_tokens": _count(req.get("user_prompt")),
                "output_tokens": n,
            })
        except Exception as e:
            _send({"status": "error", "message": str(e),
                   "traceback": traceback.format_exc()})

    try:
        del llm
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    main()
