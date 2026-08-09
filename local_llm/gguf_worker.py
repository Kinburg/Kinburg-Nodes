import sys
import os
import re
import json
import gc
import glob
import ctypes
import queue
import threading
import traceback
import importlib.util
import importlib.metadata

RESP_PREFIX = "@@LLM_RESPONSE@@"
PROG_PREFIX = "@@LLM_PROGRESS@@"
TOK_PREFIX = "@@LLM_TOKEN@@"
# A bare line, not a request: it has to be recognisable by the reader thread without parsing JSON,
# because it arrives WHILE a generation is running.
ABORT_LINE = "@@LLM_ABORT@@"

# Set by the reader thread, read between tokens. Cleared before every request, so a press that
# lands after a generation already finished cannot cut the next one short.
_ABORT = threading.Event()


def _stdin_reader(inbox):
    """Own stdin in a thread, so a stop can arrive mid-generation.

    The main loop used to read stdin itself, which meant nothing was read while tokens were being
    pumped — the abort would have sat in the pipe until the reply it was meant to interrupt had
    already finished. Requests go on the queue; the abort sentinel just sets a flag.
    """
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            line = ""
        if line == "":                      # EOF: the parent closed the pipe
            inbox.put(None)
            return
        line = line.strip()
        if not line:
            continue
        if line == ABORT_LINE:
            _ABORT.set()
        else:
            inbox.put(line)


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
    """What forces the LLM weights to be re-read. Deliberately NOT the vision projector: that is
    attached per request (see `_vision_sig`), so a chat that alternates between a turn with a
    picture and a turn without keeps one model resident instead of reloading it every time."""
    return (
        req.get("model_path"),
        int(req.get("n_ctx", 4096)),
        int(req.get("n_gpu_layers", -1)),
        int(req.get("n_batch", 512)),
        bool(req.get("flash_attn", False)),
        req.get("kv_cache_type", "f16"),
        json.dumps(req.get("extra_load_args") or {}, sort_keys=True, default=str),
        req.get("chat_template") or "",
    )


def _vision_sig(req):
    """Which projector this request wants, or None for a text-only one."""
    mmproj = (req.get("mmproj_path") or "").strip()
    if not mmproj:
        return None
    return (mmproj, (req.get("vision_handler") or "auto").strip().lower(),
            int(req.get("n_gpu_layers", -1)) != 0)


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


def _build_template_handler(llm, req):
    """A chat handler for the user-supplied chat template (read by the node from
    chat_template_path and sent as `chat_template`), or None to keep the model's embedded one.

    Returned rather than assigned to `llm.chat_handler`, because that one slot is shared with the
    vision projector and the main loop picks between them per request. `llm.chat_format` is left
    alone on purpose: it is the fallback `create_chat_completion` uses whenever `chat_handler` is
    None, so clearing it would break every plain text turn once the projector detaches.
    TEXT turns only — a vision handler does its own multimodal formatting and wins when active.
    On any failure we keep the embedded template and just log it — never break generation."""
    ct = req.get("chat_template") or ""
    if not ct.strip():
        return None
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        eos_id, bos_id = llm.token_eos(), llm.token_bos()
        eos = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
        bos = llm._model.token_get_text(bos_id) if bos_id != -1 else ""
        handler = Jinja2ChatFormatter(
            template=ct, eos_token=eos, bos_token=bos,
            stop_token_ids=[eos_id] if eos_id != -1 else None,
        ).to_chat_handler()
        sys.stdout.write("[LocalLLM worker] custom chat template override applied\n")
        sys.stdout.flush()
        return handler
    except Exception as e:
        sys.stdout.write(f"[LocalLLM worker] chat template override failed ({e}); using embedded\n")
        sys.stdout.flush()
        return None


def _free_handler(handler):
    """Let a vision projector's clip go, and hand back None so callers can just assign the result.

    The mtmd context is built against one specific llama model handle and registered on the
    handler's ExitStack, so closing that stack is what actually releases the clip — and its VRAM.
    Which also means ORDER MATTERS at the call sites: free this before dropping the model it was
    created against, never after."""
    if handler is None:
        return None
    try:
        handler._exit_stack.close()
    except Exception as e:
        sys.stdout.write(f"[LocalLLM worker] releasing the vision projector failed: {e}\n")
        sys.stdout.flush()
    return None


def _continue_prompt(llm, req, messages, cont_text):
    """Raw prompt tokens for RESUMING a truncated assistant reply.

    `messages` end just before that reply. We render them through the model's chat template with
    the generation prompt on (so the string ends with an opened assistant turn), glue the reply's
    own text onto it, and tokenize ourselves — `create_completion` would otherwise add a second
    BOS on top of the one most templates already emit."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    eos_id, bos_id = llm.token_eos(), llm.token_bos()
    eos = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
    bos = llm._model.token_get_text(bos_id) if bos_id != -1 else ""

    tmpl = (req.get("chat_template") or "").strip()
    if not tmpl:
        tmpl = ((getattr(llm, "metadata", None) or {}).get("tokenizer.chat_template") or "").strip()
    if not tmpl:
        raise ValueError("this model ships no chat template, so a truncated reply can't be "
                         "resumed — set chat_template_path on the Settings node")

    fmt = Jinja2ChatFormatter(template=tmpl, eos_token=eos, bos_token=bos,
                              stop_token_ids=[eos_id] if eos_id != -1 else None)
    try:
        rendered = fmt(messages=messages).prompt
    except TypeError:  # older signatures want the llama handle too
        rendered = fmt(llama=llm, messages=messages).prompt

    text = rendered + cont_text
    add_bos = not (bos and text.startswith(bos))
    return llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)


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
    # One loaded model, two formatters that take turns in llm.chat_handler: `text_handler` is the
    # optional chat-template override (built once with the model), `vision_handler` is the mmproj
    # projector, built on the first request that carries images and released again on the first
    # one that doesn't. Neither is part of the load signature, so switching costs no reload.
    text_handler = None
    vision_handler = None
    vision_sig = None
    current_sig = None
    # Lightweight vocab-only models kept for the token-counter command, keyed by model path.
    # They load just the tokenizer/vocab (no weights, ~no VRAM), so counting never disturbs a
    # loaded generation model.
    vocab_cache = {}
    # (vocab-only text model, MTMD handler) per (model, mmproj, use_gpu) for lean image-token
    # counting via mtmd — clip is loaded, LLM weights are NOT, and no decode is run.
    mtmd_cache = {}

    inbox = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(inbox,), daemon=True).start()

    while True:
        line = inbox.get()
        if line is None:        # EOF: parent closed the pipe
            break
        # Whatever happened during the last request is over; only a press that lands from here on
        # belongs to this one.
        _ABORT.clear()

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
                    # The projector goes FIRST — its mtmd context was created against the model
                    # handle we are about to drop, so freeing it afterwards would be a use-after-
                    # free on llama.cpp's side.
                    vision_handler = _free_handler(vision_handler)
                    vision_sig = None
                    text_handler = None
                    del llm
                    llm = None
                    gc.collect()
                kv = req.get("kv_cache_type", "f16")
                kv_type = _kv_type(llama_cpp, kv)
                flash = bool(req.get("flash_attn", False)) or kv != "f16"
                load_kwargs = dict(
                    model_path=req["model_path"],
                    n_ctx=int(req.get("n_ctx", 4096)),
                    n_gpu_layers=int(req.get("n_gpu_layers", -1)),
                    n_batch=int(req.get("n_batch", 512)),
                    seed=int(req.get("seed", -1)),
                    flash_attn=flash,
                    type_k=kv_type,
                    type_v=kv_type,
                    # Attached per request below, never at load time. Left None here so
                    # Llama.__init__ still derives chat_format from the model's own metadata —
                    # that is what formats a plain text turn once the projector detaches.
                    chat_handler=None,
                    verbose=bool(req.get("verbose", False)),
                )
                _merge_extra_load_args(load_kwargs, req.get("extra_load_args") or {})
                llm = Llama(**load_kwargs)
                text_handler = _build_template_handler(llm, req)
                current_sig = sig

            # Pick this request's formatter. create_chat_completion reads llm.chat_handler at call
            # time, so swapping it is free — and it is the whole reason a picture no longer costs
            # two model reloads (one to attach the projector, one to drop it again).
            vsig = _vision_sig(req)
            if vsig is None:
                # Text turn: release the clip rather than let it sit on VRAM unused. Rebuilding it
                # for the next picture is a fraction of a second, against tens for the weights.
                vision_handler = _free_handler(vision_handler)
                vision_sig = None
            elif vsig != vision_sig:
                vision_handler = _free_handler(vision_handler)
                vision_handler = _make_chat_handler(req)
                vision_sig = vsig
            llm.chat_handler = vision_handler if vision_handler is not None else text_handler

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
            # The chat node can ask for a turn with NO user message — "answer from the
            # conversation, your instructions are already in the system prompt" — so a persona
            # switch doesn't have to inject a prodding message into everyone's context. Then we
            # stop at the history and let the template open an assistant turn. Opt-in via
            # allow_no_user so an empty prompt still behaves as before for every other node, and
            # never send an empty list. Note some templates (mistral) demand alternating roles
            # and will reject two assistant turns in a row.
            skip_user = (req.get("allow_no_user") and messages
                         and not (req.get("user_prompt") or "").strip()
                         and not req.get("images"))
            if not skip_user:
                messages.append({"role": "user", "content": _user_content(req)})

            # count_only: prefill just to read the prompt token count (text + chat template +
            # image tokens, exactly as the model sees them) — used by the Context Sizer. A
            # 1-token generation is enough to populate usage.prompt_tokens.
            if req.get("count_only"):
                out = llm.create_chat_completion(messages=messages, max_tokens=1, temperature=0.0)
                _send({"status": "success",
                       "prompt_tokens": int((out.get("usage") or {}).get("prompt_tokens", 0))})
                continue

            # Resuming a reply that ran out of max_tokens: prefill the model with the partial
            # text so it writes the REST of that message instead of starting a new one. Needs the
            # raw completion API, so it can't go through the multimodal handlers.
            cont = req.get("continue_text") or ""
            # Only actual images block a resume. It used to be enough for a projector to be
            # loaded at all, which meant a chat with vision configured could never continue a
            # truncated reply; now the projector is detached on text turns, so this is exact.
            if cont and req.get("images"):
                raise ValueError("a truncated reply can't be resumed on the vision path — "
                                 "type a message instead")

            gen_kwargs = dict(
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
            if out_fmt == "json_object" and not cont:
                # create_completion has no response_format; a resumed reply is mid-document
                # anyway, so re-imposing "must be a whole JSON object" would be wrong.
                gen_kwargs["response_format"] = {"type": "json_object"}
            elif out_fmt == "gbnf_grammar":
                gtext = (req.get("grammar") or "").strip()
                if gtext:
                    gen_kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(gtext)

            if cont:
                gen_kwargs["prompt"] = _continue_prompt(llm, req, messages, cont)
                run = llm.create_completion
            else:
                gen_kwargs["messages"] = messages
                run = llm.create_chat_completion

            # Grammar-constrained runs stream too. In llama-cpp `stream` decides only the SHAPE of
            # what create_completion returns — `stream=False` is literally `next(<the very same
            # generator>)` (llama.py:1886) — so the token loop, the sampler chain and the grammar
            # are identical either way, and `json_object` (a grammar under the hood) has always
            # streamed here. This used to read `"grammar" not in gen_kwargs`, which cost every GBNF
            # run its live token bar and its live log for no reason at all.
            #
            # The only thing the aggregated path offers is `usage`; the exact prompt count is
            # recovered from n_tokens below regardless, which is what every text run already does.
            # Left as a variable, and the branch below left standing, so flipping this one line
            # back is the whole revert if a build ever disagrees.
            use_stream = True
            gen_kwargs["stream"] = use_stream

            parts = []
            n = 0
            finish_reason = ""
            prompt_tokens = 0  # exact prefill count, when the (non-stream) path exposes usage
            if use_stream:
                stream_text = bool(req.get("stream_text"))
                for chunk in run(**gen_kwargs):
                    # ⏹ between tokens. Abandoning the generator is enough — llama.cpp simply stops
                    # being pumped, and the model is left usable for the next request. What has been
                    # written so far is kept and returned: stopping a reply should hand you the good
                    # part, not throw the turn away.
                    if _ABORT.is_set():
                        finish_reason = "aborted"
                        break
                    ch = chunk["choices"][0]
                    piece = ch.get("text") if cont else (ch.get("delta") or {}).get("content")
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
                out = run(**gen_kwargs)
                ch = out["choices"][0]
                text = ch["text"] if cont else ch["message"]["content"]
                finish_reason = ch.get("finish_reason", "")
                usage = out.get("usage") or {}
                n = int(usage.get("completion_tokens", 0))
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                _progress(n)

            def _count(s):
                s = s or ""
                if not s:
                    return 0
                try:
                    return len(llm.tokenize(s.encode("utf-8"), add_bos=False, special=False))
                except Exception:
                    return 0

            # Context accounting for the caller (used by the Ouroboros live log to show how close
            # each call runs to the limit). n_ctx() is the authoritative loaded window; n_tokens is
            # the actual KV-cache fill after this call (prompt prefill + generated, INCLUDING the
            # chat-template and image tokens on the vision path). Both best-effort.
            try:
                ctx_limit = int(llm.n_ctx())
            except Exception:
                ctx_limit = int(req.get("n_ctx", 0) or 0)
            try:
                ctx_used = int(getattr(llm, "n_tokens", 0) or 0)
            except Exception:
                ctx_used = 0
            if not prompt_tokens and ctx_used:
                prompt_tokens = max(0, ctx_used - n)

            _send({
                "status": "success",
                "output": text,
                "finish_reason": finish_reason,
                "sys_tokens": _count(req.get("system_prompt")),
                "user_tokens": _count(req.get("user_prompt")),
                "output_tokens": n,
                "prompt_tokens": prompt_tokens,
                "context_used": ctx_used if ctx_used else (prompt_tokens + n),
                "n_ctx": ctx_limit,
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
