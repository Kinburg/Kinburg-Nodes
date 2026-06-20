import sys
import os
import json
import gc
import glob
import ctypes
import traceback
import importlib.util

RESP_PREFIX = "@@LLM_RESPONSE@@"
PROG_PREFIX = "@@LLM_PROGRESS@@"


def _prepare_cuda():
    """The cu124 llama-cpp-python wheel does NOT bundle the CUDA runtime, and the
    llama_cpp loader opens llama.dll with a flag (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
    under which directories added via add_dll_directory are ignored when resolving
    dependencies. So we PRELOAD the required CUDA DLLs by full path from torch\\lib
    (cu129 -- backward compatible within CUDA 12.x). Once they are loaded into the
    process, ggml-cuda.dll resolves them by name. torch itself is NOT imported
    (find_spec only locates the module, it does not execute it), so no second CUDA
    context is created."""
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


def _send(obj):
    sys.stdout.write(RESP_PREFIX + json.dumps(obj, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _progress(n):
    sys.stdout.write(PROG_PREFIX + str(n) + "\n")
    sys.stdout.flush()


def _load_sig(req):
    return (
        req.get("model_path"),
        int(req.get("n_ctx", 4096)),
        int(req.get("n_gpu_layers", -1)),
        int(req.get("n_batch", 512)),
        bool(req.get("flash_attn", False)),
        req.get("kv_cache_type", "f16"),
    )


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
        _send({"status": "error", "message": f"import llama_cpp failed: {e}",
               "traceback": traceback.format_exc()})
        return

    llm = None
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
                    gc.collect()
                kv = req.get("kv_cache_type", "f16")
                kv_type = _kv_type(llama_cpp, kv)
                flash = bool(req.get("flash_attn", False)) or kv != "f16"
                llm = Llama(
                    model_path=req["model_path"],
                    n_ctx=int(req.get("n_ctx", 4096)),
                    n_gpu_layers=int(req.get("n_gpu_layers", -1)),
                    n_batch=int(req.get("n_batch", 512)),
                    seed=int(req.get("seed", -1)),
                    flash_attn=flash,
                    type_k=kv_type,
                    type_v=kv_type,
                    verbose=bool(req.get("verbose", False)),
                )
                current_sig = sig

            messages = []
            sys_prompt = (req.get("system_prompt") or "").strip()
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": req.get("user_prompt", "")})

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
                for chunk in llm.create_chat_completion(**gen_kwargs):
                    ch = chunk["choices"][0]
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        parts.append(piece)
                        n += 1
                        _progress(n)
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
