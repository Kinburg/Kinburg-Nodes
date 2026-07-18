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
    # ensure_ascii=False keeps non-ASCII (e.g. Cyrillic) compact over the pipe; the stream is
    # forced to UTF-8 in main() so the parent (also UTF-8) decodes it correctly.
    sys.stdout.write(RESP_PREFIX + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _progress(n):
    sys.stdout.write(PROG_PREFIX + str(n) + "\n")
    sys.stdout.flush()


def _token(s):
    # JSON-encode so newlines in the delta don't break the one-line protocol.
    sys.stdout.write(TOK_PREFIX + json.dumps(s, ensure_ascii=False) + "\n")
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
        req.get("chat_template") or "",
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


def _apply_chat_template_override(llm, req, chat_handler):
    """Replace the model's embedded chat template with the user-supplied one (read by the node
    from chat_template_path and sent as `chat_template`). TEXT models only: when a vision
    `chat_handler` is active it does its own multimodal formatting, so we leave it alone.
    On any failure we keep the embedded template and just log it — never break generation."""
    ct = req.get("chat_template") or ""
    if not ct.strip() or chat_handler is not None:
        return
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        eos_id, bos_id = llm.token_eos(), llm.token_bos()
        eos = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
        bos = llm._model.token_get_text(bos_id) if bos_id != -1 else ""
        llm.chat_handler = Jinja2ChatFormatter(
            template=ct, eos_token=eos, bos_token=bos,
            stop_token_ids=[eos_id] if eos_id != -1 else None,
        ).to_chat_handler()
        llm.chat_format = None  # create_chat_completion prefers chat_handler when set
        sys.stdout.write("[LocalLLM worker] custom chat template override applied\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"[LocalLLM worker] chat template override failed ({e}); using embedded\n")
        sys.stdout.flush()


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
    # Force the stdio pipes to UTF-8 (Windows subprocesses default to the locale codepage, e.g.
    # cp1251, which would garble/crash on non-ASCII once ensure_ascii=False is used). The parent
    # opens the pipe as UTF-8 too, so both ends agree.
    for _stream in (sys.stdout, sys.stdin):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
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
    # Lightweight vocab-only models kept for the token-counter command, keyed by model path.
    # They load just the tokenizer/vocab (no weights, ~no VRAM), so counting never disturbs a
    # loaded generation model.
    vocab_cache = {}
    # (vocab-only text model, MTMD handler) per (model, mmproj, use_gpu) for lean image-token
    # counting via mtmd — clip is loaded, LLM weights are NOT, and no decode is run.
    mtmd_cache = {}

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

        if req.get("cmd") == "count":
            # Count tokens only — reuse an already-loaded model for the same path, else a cheap
            # vocab-only load. No generation, no VRAM footprint for the counting model.
            try:
                text = req.get("text", "") or ""
                mp = req.get("model_path")
                if llm is not None and current_sig and current_sig[0] == mp:
                    tok = llm
                else:
                    tok = vocab_cache.get(mp)
                    if tok is None:
                        tok = Llama(model_path=mp, vocab_only=True, verbose=False)
                        vocab_cache[mp] = tok
                n = len(tok.tokenize(text.encode("utf-8"), add_bos=False, special=False))
                _send({"status": "success", "token_count": n})
            except Exception as e:
                _send({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
            continue

        if req.get("cmd") == "count_mtmd":
            # Lean image-token count: load a vocab-only text model + the clip/mmproj, run
            # mtmd_tokenize, and sum the IMAGE chunks' token counts. No LLM weights, no decode —
            # mirrors MTMDChatHandler's tokenize loop but stops before eval. Returns per-image
            # token counts. (If mtmd can't run weight-free on this build, this raises and the
            # caller falls back to the full-model prefill.)
            try:
                import llama_cpp.mtmd_cpp as mtmd_cpp
                from llama_cpp.llama_chat_format import MTMDChatHandler
                mp = req.get("model_path")
                mmproj = req.get("mmproj_path")
                use_gpu = int(req.get("n_gpu_layers", -1)) != 0
                key = (mp, mmproj, use_gpu)
                entry = mtmd_cache.get(key)
                if entry is None:
                    vllm = Llama(model_path=mp, vocab_only=True, verbose=False)
                    handler = MTMDChatHandler(clip_model_path=mmproj, verbose=False, use_gpu=use_gpu)
                    handler._init_mtmd_context(vllm)  # loads clip; may need weights → then raises
                    entry = (vllm, handler)
                    mtmd_cache[key] = entry
                handler = entry[1]
                marker = mtmd_cpp.mtmd_default_marker().decode("utf-8")
                counts = []
                for uri in (req.get("images") or []):
                    bitmap = handler._create_bitmap_from_bytes(handler.load_image(uri))
                    chunks = mtmd_cpp.mtmd_input_chunks_init()
                    try:
                        itext = mtmd_cpp.mtmd_input_text()
                        itext.text = marker.encode("utf-8")
                        itext.add_special = False
                        itext.parse_special = True
                        barr = (mtmd_cpp.mtmd_bitmap_p_ctypes * 1)(bitmap)
                        rc = mtmd_cpp.mtmd_tokenize(handler.mtmd_ctx, chunks, ctypes.byref(itext), barr, 1)
                        if rc != 0:
                            raise ValueError(f"mtmd_tokenize failed: {rc}")
                        img = 0
                        for i in range(mtmd_cpp.mtmd_input_chunks_size(chunks)):
                            ch = mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                            if ch is not None and mtmd_cpp.mtmd_input_chunk_get_type(ch) == mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE:
                                img += int(mtmd_cpp.mtmd_input_chunk_get_n_tokens(ch))
                        counts.append(img)
                    finally:
                        mtmd_cpp.mtmd_input_chunks_free(chunks)
                        mtmd_cpp.mtmd_bitmap_free(bitmap)
                _send({"status": "success", "image_tokens": counts})
            except Exception as e:
                _send({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
            continue

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
                _apply_chat_template_override(llm, req, chat_handler)
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

            # count_only: prefill just to read the prompt token count (text + chat template +
            # image tokens, exactly as the model sees them) — used by the Context Sizer. A
            # 1-token generation is enough to populate usage.prompt_tokens.
            if req.get("count_only"):
                out = llm.create_chat_completion(messages=messages, max_tokens=1, temperature=0.0)
                _send({"status": "success",
                       "prompt_tokens": int((out.get("usage") or {}).get("prompt_tokens", 0))})
                continue

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
