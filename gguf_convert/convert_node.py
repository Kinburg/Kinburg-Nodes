"""Safetensors -> GGUF converter — turn a HuggingFace LLM into a .gguf using llama.cpp's
`convert_hf_to_gguf.py`, then (optionally) quantize it with `llama-quantize`.

The node takes a model as EITHER a HuggingFace repo id / URL (downloaded with
`huggingface_hub`) OR a local path (a HF model folder, or a single `.safetensors` file whose
folder holds the config + tokenizer). It runs llama.cpp's conversion script to produce an
f16/bf16 `.gguf`, and if a quant is chosen it runs `llama-quantize` to shrink it to a
K-/I-quant. The output is the final `.gguf` path — feed it straight into the Local LLM (GGUF)
nodes.

Scope: llama.cpp only converts LANGUAGE / multimodal-LLM weights. It cannot convert diffusion
models (SD / Flux) — for those you need city96's ComfyUI-GGUF `convert.py` or
stable-diffusion.cpp, not this node.

Everything heavy (folder_paths, huggingface_hub, subprocess work) is imported/called inside the
methods so the package still imports for a registry scan without ComfyUI present.
"""
import os
import re
import sys
import shutil
import subprocess
from collections import deque
from ..categories import CAT_LLM_GGUF

# Reuse the Local LLM helper that locates ComfyUI/models/llm; fall back to None on a bare
# registry scan (no ComfyUI on the path).
try:
    from ..local_llm.llm_node import _gguf_dir
except Exception:  # pragma: no cover — registry scan without ComfyUI
    def _gguf_dir():
        return None

PLACEHOLDER = "(auto-detect / use path field)"
CLONE_URL = "https://github.com/ggml-org/llama.cpp"

# convert_hf_to_gguf.py --outtype values (single-step; no external binary needed).
OUTTYPES = ["f16", "bf16", "q8_0", "f32", "auto"]

# Post-quantization presets handed to `llama-quantize` (needs a compiled llama.cpp binary).
QUANTS = [
    "none", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0",
    "Q3_K_M", "Q3_K_S", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ2_M",
]

# Conversion-script filenames across llama.cpp versions (newest first).
_CONVERT_SCRIPTS = ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"]

HELP_TEXT = """# Safetensors -> GGUF — quick help

Converts a HuggingFace LLM to `.gguf` with llama.cpp's `convert_hf_to_gguf.py`, then
optionally quantizes it with `llama-quantize`. LLMs only — NOT diffusion models (SD/Flux).

## source
- A HF repo id (`Qwen/Qwen2.5-0.5B-Instruct`) or URL (`https://huggingface.co/...`) —
  downloaded for you via huggingface_hub. Set `hf_token` for gated/private repos.
- A local folder that holds `config.json` + `*.safetensors` + tokenizer files.
- A single `*.safetensors` file — its containing folder is used (must hold config/tokenizer).

## Output type & quantization
- `outtype` is the precision the conversion script writes (`f16` is the usual base).
- `quantize` runs a 2nd pass to shrink the model (e.g. `Q4_K_M`). This needs a compiled
  `llama-quantize` binary — auto-detected in ComfyUI/models/llm, or set `quantize_binary_path`.
  Grab a prebuilt one from llama.cpp's GitHub Releases if you don't build it yourself.
- `quantize = none` -> the `outtype` gguf is the final output.

## llama.cpp scripts
- `llama_cpp_dir` = a llama.cpp checkout. Leave blank and keep `auto_clone` on to have it
  `git clone`d into ComfyUI/models/llm/llama.cpp automatically (source only — no compiled
  binaries, so `llama-quantize` still has to be provided for the quantize pass).

## Notes
- Conversion runs in ComfyUI's own Python. If it complains about a missing package
  (e.g. `gguf`, `sentencepiece`), install it into that environment.
- `force = false` returns an already-existing output without redoing the work.
- Progress and any errors are streamed to the ComfyUI console and the `log` output.
"""


def _list_quantize_binaries():
    """Auto-list llama-quantize executables in models/llm and its parent (+ any cloned repo)."""
    found = []
    d = _gguf_dir()
    roots = []
    for base in filter(None, [d, os.path.dirname(d) if d else None]):
        roots.append(base)
    # A repo cloned by auto_clone keeps build outputs under build/bin — scan a couple of levels.
    if d:
        roots.append(os.path.join(d, "llama.cpp"))
    for base in roots:
        if not base or not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            depth = dirpath[len(base):].count(os.sep)
            if depth > 3:
                _dirs[:] = []
                continue
            for f in files:
                lo = f.lower()
                if ("llama-quantize" in lo or "quantize" == lo.split(".")[0]) and \
                        (lo.endswith(".exe") or "." not in lo):
                    p = os.path.join(dirpath, f)
                    if os.path.isfile(p) and p not in found:
                        found.append(p)
    return found + [PLACEHOLDER]


def _resolve_binary(choice, manual):
    if choice and choice != PLACEHOLDER and os.path.isfile(choice):
        return choice
    return (manual or "").strip().strip('"').strip("'").strip()


def _clean(s):
    return (s or "").strip().strip('"').strip("'").strip()


def _parse_source(source):
    """Classify `source` -> ('local', abspath) | ('repo', 'owner/name') | (None, error_msg).

    A path that exists on disk is local. Otherwise we try to read a HF repo id, either from a
    huggingface.co URL or a bare `owner/name` slug.
    """
    s = _clean(source)
    if not s:
        return None, "source is empty — give a HF repo id / URL or a local model path."
    if os.path.exists(s):
        return "local", os.path.abspath(s)
    # URL form: https://huggingface.co/<owner>/<name>[/tree/<rev>][/...]
    m = re.search(r"huggingface\.co/([^/\s]+/[^/\s#?]+)", s)
    if m:
        repo = m.group(1)
        repo = re.sub(r"\.git$", "", repo)
        return "repo", repo
    # Bare slug: owner/name (allow the datasets/models prefixes people sometimes paste)
    m = re.fullmatch(r"(?:models/)?([\w.\-]+/[\w.\-]+)", s)
    if m:
        return "repo", m.group(1)
    return None, (f"Couldn't read a model from source={s!r}. Use a local path, a HF repo id "
                  f"like 'owner/name', or a https://huggingface.co/owner/name URL.")


def _model_dir_from_local(path):
    """A local source -> the HF model directory to feed the script. A .safetensors file maps to
    its parent folder. Returns (dir, error_msg)."""
    if os.path.isfile(path):
        if not path.lower().endswith(".safetensors"):
            return None, f"Local file is not a .safetensors: {path}"
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        return None, f"Model folder not found: {path}"
    if not os.path.isfile(os.path.join(path, "config.json")):
        return None, (f"No config.json in {path} — the conversion script needs the full HF model "
                      f"folder (config.json + tokenizer + *.safetensors), not just the weights.")
    return path, ""


def _download_hf(repo_id, token, download_dir, log):
    """snapshot_download the repo (weights + config + tokenizer) and return its local path."""
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        raise RuntimeError("huggingface_hub is not installed in ComfyUI's Python. "
                           "Install it (pip install huggingface_hub) or pass a local path.")
    kwargs = {
        "repo_id": repo_id,
        # Skip files the converter never reads to save bandwidth/disk.
        "ignore_patterns": ["*.gguf", "*.bin.index.json.lock", "*.msgpack", "*.h5", "*.onnx"],
    }
    tok = _clean(token)
    if tok:
        kwargs["token"] = tok
    if _clean(download_dir):
        kwargs["local_dir"] = _clean(download_dir)
    log.append(f"Downloading {repo_id} from HuggingFace ...")
    print(f"[GGUFConvert] downloading {repo_id} ...")
    path = snapshot_download(**kwargs)
    log.append(f"Downloaded to {path}")
    return path


def _find_convert_script(llama_dir):
    for name in _CONVERT_SCRIPTS:
        p = os.path.join(llama_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _locate_llama_cpp(user_dir, auto_clone, log):
    """Return the path to convert_hf_to_gguf.py, cloning llama.cpp if allowed. Raises on failure."""
    # 1) Explicit directory from the user.
    ud = _clean(user_dir)
    if ud:
        if not os.path.isdir(ud):
            raise RuntimeError(f"llama_cpp_dir does not exist: {ud}")
        script = _find_convert_script(ud)
        if script:
            return script
        raise RuntimeError(f"No {_CONVERT_SCRIPTS[0]} found in {ud}. Point llama_cpp_dir at a "
                           f"llama.cpp checkout root.")

    # 2) A previously cloned copy under models/llm/llama.cpp.
    base = _gguf_dir()
    cached = os.path.join(base, "llama.cpp") if base else None
    if cached:
        script = _find_convert_script(cached)
        if script:
            return script

    # 3) Clone it.
    if not auto_clone:
        raise RuntimeError("No llama.cpp found. Set llama_cpp_dir, or enable auto_clone to fetch it.")
    if not base:
        raise RuntimeError("Can't determine ComfyUI/models/llm for the auto-clone. "
                           "Set llama_cpp_dir to a llama.cpp checkout instead.")
    if shutil.which("git") is None:
        raise RuntimeError("git is not on PATH, so auto_clone can't run. Install git, or clone "
                           "llama.cpp yourself and set llama_cpp_dir.")
    os.makedirs(base, exist_ok=True)
    log.append(f"Cloning llama.cpp into {cached} ...")
    print(f"[GGUFConvert] git clone {CLONE_URL} -> {cached}")
    rc = _run(["git", "clone", "--depth", "1", CLONE_URL, cached], None, log)
    if rc != 0:
        raise RuntimeError(f"git clone failed (exit {rc}). See the log/console.")
    script = _find_convert_script(cached)
    if not script:
        raise RuntimeError(f"Cloned llama.cpp but no {_CONVERT_SCRIPTS[0]} in {cached}.")
    return script


def _run(cmd, cwd, log):
    """Run a command, teeing its combined output to the console and the log list. Returns exit code."""
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.pop("NO_LOCAL_GGUF", None)  # let the script use its bundled gguf-py
    print(f"[GGUFConvert] $ {' '.join(cmd)}")
    log.append("$ " + " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
    except FileNotFoundError as e:
        log.append(f"failed to start: {e}")
        print(f"[GGUFConvert] failed to start: {e}")
        return 127
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip("\n")
        log.append(line)
        print(f"[GGUFConvert] {line}")
    proc.wait()
    return proc.returncode


def _diagnose_convert_failure(log_text):
    """Turn a known convert_hf_to_gguf.py failure into an actionable hint ('' if none)."""
    t = (log_text or "").lower()
    if "bitsandbytes" in t or "quant method is not yet supported" in t:
        return ("This model is ALREADY quantized (e.g. bitsandbytes NF4 / INT8 — note the 'NF4' "
                "in its name). llama.cpp can't convert pre-quantized weights; download the "
                "ORIGINAL fp16/bf16 version of the model and convert that, then use the "
                "'quantize' field for a GGUF K-quant. Quantizing an NF4 model to GGUF would be "
                "double-quantization anyway (poor quality).")
    if "gptq" in t or "awq" in t:
        return ("This looks like a GPTQ/AWQ pre-quantized model, which llama.cpp can't convert. "
                "Convert the original fp16/bf16 model instead.")
    if "not supported" in t and "architecture" in t:
        return ("This model's architecture may be unsupported by your llama.cpp build. Update it: "
                "delete ComfyUI/models/llm/llama.cpp to re-clone the latest, or `git pull` your "
                "llama_cpp_dir.")
    return ""


# general.file_type (LlamaFileType) -> the short tag we put in the filename.
_FTYPE_TAG = {"ALL_F32": "f32", "MOSTLY_F16": "f16", "MOSTLY_BF16": "bf16", "MOSTLY_Q8_0": "q8_0"}


def _read_gguf_ftype(path):
    """Real precision tag ('f16'/'bf16'/'f32'/'q8_0'/…) from a gguf's general.file_type; '' if unknown.

    Used to replace the nominal 'auto' tag in the filename with what the converter actually wrote."""
    try:
        import gguf
        field = gguf.GGUFReader(path).fields.get("general.file_type")
        if field is None:
            return ""
        raw = field.parts[field.data[0]]
        try:
            val = int(raw[0])
        except (TypeError, IndexError):
            val = int(raw)
        name = gguf.LlamaFileType(val).name
    except Exception:
        return ""
    return _FTYPE_TAG.get(name, name.lower().replace("mostly_", "").replace("all_", ""))


def _find_existing_precision(out_dir, base):
    """An already-converted base-precision gguf for `base` (any known tag), or '' — so an
    outtype=auto rerun is still idempotent even though the real tag isn't known up front."""
    for tag in ("f16", "bf16", "f32", "q8_0"):
        p = os.path.join(out_dir, f"{base}.{tag}.gguf")
        if os.path.isfile(p):
            return p
    return ""


def _derive_base_name(kind, value, model_dir):
    """A clean base filename for the output: repo name, folder name, or file stem."""
    if kind == "repo":
        return value.split("/")[-1]
    name = os.path.basename(os.path.normpath(model_dir))
    return name or "model"


class SafetensorsToGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("STRING", {"default": "", "tooltip": "HF repo id ('owner/name'), a https://huggingface.co/owner/name URL, a local HF model folder, or a single .safetensors file. Windows 'Copy as path' quotes are stripped."}),
                "output_dir": ("STRING", {"default": "", "tooltip": "Folder to write the .gguf into. Created if missing. Blank -> ComfyUI/models/llm."}),
                "output_name": ("STRING", {"default": "", "tooltip": "Base filename without extension. Blank -> derived from the model name. The precision/quant tag and .gguf are added automatically."}),
                "outtype": (OUTTYPES, {"default": "f16", "tooltip": "Precision written by convert_hf_to_gguf.py. f16 is the usual base to quantize from. When quantize != none this is the intermediate."}),
                "quantize": (QUANTS, {"default": "none", "tooltip": "Optional 2nd pass with llama-quantize to shrink the model (e.g. Q4_K_M). none = keep the outtype gguf. Needs a compiled llama-quantize binary."}),
                "quantize_binary": (_list_quantize_binaries(), {"tooltip": "llama-quantize executable (auto-listed from ComfyUI/models/llm). Choose the placeholder to type a path in quantize_binary_path. Only used when quantize != none."}),
                "quantize_binary_path": ("STRING", {"default": "", "tooltip": "Full path to llama-quantize(.exe), used when the dropdown is the placeholder."}),
            },
            "optional": {
                "llama_cpp_dir": ("STRING", {"default": "", "tooltip": "Path to a llama.cpp checkout (holds convert_hf_to_gguf.py). Blank -> use/auto-clone one under ComfyUI/models/llm/llama.cpp."}),
                "auto_clone": ("BOOLEAN", {"default": True, "tooltip": "If no llama.cpp is found, git clone it automatically. Needs git on PATH. Clones source only (no compiled llama-quantize)."}),
                "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace token for gated/private repos. Leave blank for public models."}),
                "hf_download_dir": ("STRING", {"default": "", "tooltip": "Where to download HF repos. Blank -> the default HuggingFace cache. Ignored for local sources."}),
                "keep_intermediate": ("BOOLEAN", {"default": False, "tooltip": "When quantizing, keep the intermediate outtype gguf as well as the quantized one. Off -> delete it after quantizing."}),
                "force": ("BOOLEAN", {"default": False, "tooltip": "Re-run even if the target .gguf already exists. Off -> return the existing file without redoing the work."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("gguf_path", "log", "help")
    FUNCTION = "run"
    CATEGORY = CAT_LLM_GGUF
    OUTPUT_NODE = True

    def run(self, source, output_dir, output_name, outtype, quantize, quantize_binary,
            quantize_binary_path, llama_cpp_dir="", auto_clone=True, hf_token="",
            hf_download_dir="", keep_intermediate=False, force=False):
        log = deque(maxlen=800)

        def _tail():
            return "\n".join(log)

        def _err(msg):
            print(f"[GGUFConvert] ERROR: {msg}")
            log.append(f"ERROR: {msg}")
            return (f"[ERROR] {msg}", _tail(), HELP_TEXT)

        # --- resolve the model directory (download HF repos on the fly) --------------------
        kind, value = _parse_source(source)
        if kind is None:
            return _err(value)
        try:
            if kind == "repo":
                model_dir = _download_hf(value, hf_token, hf_download_dir, log)
            else:
                model_dir, e = _model_dir_from_local(value)
                if not model_dir:
                    return _err(e)
        except Exception as e:
            return _err(str(e))

        # --- output paths ------------------------------------------------------------------
        out_dir = _clean(output_dir) or _gguf_dir()
        if not out_dir:
            return _err("output_dir is empty and ComfyUI/models/llm is unavailable. Set output_dir.")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            return _err(f"Can't create output_dir {out_dir}: {e}")

        base = _clean(output_name) or _derive_base_name(kind, value, model_dir)
        base = os.path.basename(base.replace("\\", "/"))  # tolerate a pasted full path
        base = re.sub(r'[<>:"/\\|?*]', "_", base) or "model"  # keep it a safe filename
        do_quant = quantize != "none"
        # The convert step writes this; for outtype=auto it's a temp '.auto.gguf' that gets
        # renamed to the real precision tag once we can read it from the finished file.
        intermediate = os.path.join(out_dir, f"{base}.{outtype}.gguf")
        final = os.path.join(out_dir, f"{base}.{quantize}.gguf") if do_quant else None

        def _existing(path):
            msg = f"Output already exists (force=off): {path}"
            print(f"[GGUFConvert] {msg}")
            log.append(msg)
            return (path, _tail(), HELP_TEXT)

        # Idempotency: skip the work if the target already exists (unless force).
        if not force:
            if do_quant:
                if os.path.isfile(final):
                    return _existing(final)
            elif outtype == "auto":
                done = _find_existing_precision(out_dir, base)
                if done:
                    return _existing(done)
            elif os.path.isfile(intermediate):
                return _existing(intermediate)

        # --- locate llama.cpp's conversion script -----------------------------------------
        try:
            convert_script = _locate_llama_cpp(llama_cpp_dir, auto_clone, log)
        except Exception as e:
            return _err(str(e))
        llama_root = os.path.dirname(convert_script)

        # --- step 1: convert safetensors -> gguf ------------------------------------------
        cmd = [sys.executable, convert_script, model_dir,
               "--outfile", intermediate, "--outtype", outtype]
        log.append(f"Converting {model_dir} -> {intermediate} ({outtype}) ...")
        rc = _run(cmd, llama_root, log)
        if rc != 0:
            hint = _diagnose_convert_failure("\n".join(log))
            msg = f"convert_hf_to_gguf.py failed (exit {rc}). See the log above."
            if hint:
                msg += "\n\nHint: " + hint
            return _err(msg)
        if not os.path.isfile(intermediate):
            return _err(f"Conversion reported success but {intermediate} is missing.")

        # outtype=auto -> rename the '.auto.gguf' to the precision the converter actually chose.
        if outtype == "auto":
            tag = _read_gguf_ftype(intermediate)
            if tag and tag != "auto":
                renamed = os.path.join(out_dir, f"{base}.{tag}.gguf")
                if os.path.normcase(os.path.abspath(renamed)) != os.path.normcase(os.path.abspath(intermediate)):
                    try:
                        os.replace(intermediate, renamed)
                        log.append(f"Detected precision {tag}; renamed -> {os.path.basename(renamed)}")
                        intermediate = renamed
                    except Exception as e:
                        log.append(f"Could not rename {intermediate} -> {renamed}: {e}")

        if not do_quant:
            print(f"[GGUFConvert] done -> {intermediate}")
            log.append(f"Done -> {intermediate}")
            return (intermediate, _tail(), HELP_TEXT)

        # --- step 2: quantize --------------------------------------------------------------
        qbin = _resolve_binary(quantize_binary, quantize_binary_path)
        if not qbin or not os.path.isfile(qbin):
            return _err(f"quantize = {quantize} needs a llama-quantize binary, but none was found "
                        f"({qbin or 'nothing selected'}). Set quantize_binary_path, or use "
                        f"quantize = none. Prebuilt binaries are on llama.cpp's GitHub Releases.")
        qcmd = [qbin, intermediate, final, quantize]
        log.append(f"Quantizing -> {final} ({quantize}) ...")
        rc = _run(qcmd, llama_root, log)
        if rc != 0:
            return _err(f"llama-quantize failed (exit {rc}). The intermediate {intermediate} was kept.")
        if not os.path.isfile(final):
            return _err(f"Quantize reported success but {final} is missing.")

        if not keep_intermediate:
            try:
                os.remove(intermediate)
                log.append(f"Removed intermediate {intermediate}")
            except Exception as e:
                log.append(f"Could not remove intermediate {intermediate}: {e}")

        print(f"[GGUFConvert] done -> {final}")
        log.append(f"Done -> {final}")
        return (final, _tail(), HELP_TEXT)


NODE_CLASS_MAPPINGS = {"SafetensorsToGGUF": SafetensorsToGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"SafetensorsToGGUF": "Safetensors -> GGUF (llama.cpp)"}
