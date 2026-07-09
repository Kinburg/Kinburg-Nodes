"""Diffusion Safetensors -> GGUF — convert a diffusion model (Flux / SD3 / SDXL / SD1 / Aura /
HiDream / Cosmos / LTXV / HunyuanVideo / Wan / Lumina2) to `.gguf` using city96's
**ComfyUI-GGUF** `tools/convert.py`, then optionally quantize it.

This is the diffusion counterpart to the LLM converter. llama.cpp's own scripts can't read
diffusion weights, so this drives city96's converter instead:

1. `tools/convert.py --src <model.safetensors> --dst <out-F16.gguf>` — writes an F16/BF16 gguf.
   Standalone: needs only `gguf>=0.13.0`, torch, safetensors, tqdm (all ship with ComfyUI).
2. (optional) `llama-quantize <F16.gguf> <out-Q.gguf> <Qtype>` — shrinks it. This needs the
   **patched** `llama-quantize` from city96's llama.cpp fork; the stock llama.cpp one can't
   handle diffusion tensor shapes. Not bundled — build it or grab a prebuilt one.
3. (Wan 2.1 / HunyuanVideo only) `fix_5d_tensors.py` — re-injects 5-D tensors that quantization
   drops. Run automatically in `fix_5d_tensors = auto` when the model's arch is wan/hyvid.

The output `.gguf` loads with ComfyUI-GGUF's Unet Loader (GGUF). For diffusion models, NOT LLMs
— for language models use **Safetensors -> GGUF (llama.cpp)** instead.

Heavy work is imported/called inside the methods so the package still imports for a registry
scan without ComfyUI present.
"""
import os
import re
import sys
import shutil
from collections import deque

# Reuse the generic helpers from the LLM converter. Fall back to local copies when this file is
# loaded directly (registry scan) rather than as part of the package.
try:
    from .convert_node import (_clean, _resolve_binary, _run, _list_quantize_binaries,
                               _read_gguf_ftype, PLACEHOLDER)
except Exception:  # pragma: no cover — direct-file load without package context
    PLACEHOLDER = "(auto-detect / use path field)"

    def _clean(s):
        return (s or "").strip().strip('"').strip("'").strip()

    def _resolve_binary(choice, manual):
        if choice and choice != PLACEHOLDER and os.path.isfile(choice):
            return choice
        return (manual or "").strip().strip('"').strip("'").strip()

    def _run(*a, **k):
        raise RuntimeError("_run unavailable in registry-scan fallback")

    def _list_quantize_binaries():
        return [PLACEHOLDER]

    def _read_gguf_ftype(path):
        return ""

CITY96_URL = "https://github.com/city96/ComfyUI-GGUF"

# Post-quantization presets for diffusion models (handed to the patched llama-quantize).
QUANTS = [
    "none", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K",
]

# Architectures whose quantized files need the 5-D tensor fix pass.
_FIX_5D_ARCHS = {"wan", "hyvid"}

HELP_TEXT = """# Diffusion Safetensors -> GGUF — quick help

Converts a DIFFUSION model (Flux, SD3, SDXL, SD1, Aura, HiDream, Cosmos, LTXV, HunyuanVideo,
Wan, Lumina2) to `.gguf` via city96's ComfyUI-GGUF `tools/convert.py`, then optionally
quantizes it. For LLMs use 'Safetensors -> GGUF (llama.cpp)' instead.

## model / model_path
- Pick a file from ComfyUI/models/diffusion_models or /unet in the `model` dropdown, or
- set `model_path` to a local `.safetensors`/`.ckpt`, a HF file URL
  (`https://huggingface.co/owner/name/resolve/main/model.safetensors`), or `owner/name::file.safetensors`.
- Give the DIFFUSION model file (the single unet/DiT checkpoint), NOT a full HF pipeline
  folder and NOT the diffusers UNET layout.

## Output type & quantization
- Step 1 always writes an F16/BF16 gguf (this alone is a valid, if large, model).
- `quantize` runs step 2 (e.g. `Q4_K_S`, `Q8_0`). It needs the **patched** `llama-quantize`
  from city96's llama.cpp fork — build it per ComfyUI-GGUF/tools/README.md and point
  `quantize_binary_path` at it (or drop it in ComfyUI/models/llm to auto-detect). The stock
  llama.cpp binary will NOT work for diffusion models.
- `fix_5d_tensors` (auto/on/off): after quantizing Wan 2.1 / HunyuanVideo models, a fix pass
  re-adds 5-D tensors. `auto` reads the arch from the gguf and runs it only for wan/hyvid.

## Tooling
- `tool_dir` = a ComfyUI-GGUF checkout (holds `tools/convert.py`). Blank -> use an installed
  `custom_nodes/ComfyUI-GGUF`, else `git clone` it there when `auto_clone` is on. You'll want
  ComfyUI-GGUF installed anyway — its Unet Loader (GGUF) is what loads the output.

## Notes
- Conversion runs in ComfyUI's own Python; if it reports a missing package, install it there.
- `force = false` returns an already-existing output instead of redoing the work.
- Output/errors stream to the ComfyUI console and the `log` output.
"""


def _custom_nodes_dir():
    """.../custom_nodes (this file is .../custom_nodes/kinburg-nodes/gguf_convert/diffusion_node.py)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _diffusion_model_list():
    """Filenames from models/diffusion_models + models/unet for the dropdown (deduped)."""
    try:
        import folder_paths
    except Exception:
        return [PLACEHOLDER]
    names, seen = [], set()
    for key in ("diffusion_models", "unet"):
        try:
            for n in folder_paths.get_filename_list(key):
                if n not in seen:
                    seen.add(n)
                    names.append(n)
        except Exception:
            pass
    return [PLACEHOLDER] + names


def _resolve_model_file(name):
    import folder_paths
    for key in ("diffusion_models", "unet"):
        try:
            p = folder_paths.get_full_path(key, name)
            if p and os.path.isfile(p):
                return p
        except Exception:
            pass
    return None


def _default_out_dir():
    try:
        import folder_paths
        for key in ("diffusion_models", "unet"):
            try:
                dirs = folder_paths.get_folder_paths(key)
                if dirs:
                    return dirs[0]
            except Exception:
                pass
        return os.path.join(folder_paths.models_dir, "diffusion_models")
    except Exception:
        return None


def _download_hf_file(model_path, token, log):
    """Download a single file referenced by a HF file URL or an `owner/name::file` spec."""
    from huggingface_hub import hf_hub_download
    s = _clean(model_path)
    tok = _clean(token) or None
    m = re.search(r"huggingface\.co/([^/\s]+/[^/\s]+)/(?:resolve|blob)/([^/\s]+)/(.+?)(?:[?#].*)?$", s)
    if m:
        repo, rev, fname = m.group(1), m.group(2), m.group(3)
        log.append(f"Downloading {fname} from {repo}@{rev} ...")
        print(f"[GGUFConvert-Diff] downloading {repo}/{fname} ...")
        return hf_hub_download(repo_id=repo, filename=fname, revision=rev, token=tok)
    if "::" in s:
        repo, fname = (x.strip() for x in s.split("::", 1))
        log.append(f"Downloading {fname} from {repo} ...")
        print(f"[GGUFConvert-Diff] downloading {repo}/{fname} ...")
        return hf_hub_download(repo_id=repo, filename=fname, token=tok)
    return None


def _resolve_source(model, model_path, token, log):
    """Resolve the source model to a local file path. Returns (path, error_msg)."""
    if model and model != PLACEHOLDER:
        p = _resolve_model_file(model)
        return (p, "") if p else (None, f"Selected model not found on disk: {model}")

    s = _clean(model_path)
    if not s:
        return None, "No model selected and model_path is empty."
    if os.path.isfile(s):
        if not s.lower().endswith((".safetensors", ".sft", ".ckpt")):
            return None, f"model_path is not a .safetensors/.ckpt file: {s}"
        return s, ""
    if os.path.isdir(s):
        return None, ("model_path is a folder. Diffusion conversion needs the single diffusion "
                      "model .safetensors file, not a HF pipeline folder.")
    if "huggingface.co" in s or "::" in s:
        try:
            p = _download_hf_file(s, token, log)
        except Exception as e:
            return None, f"HuggingFace download failed: {e}"
        if p:
            return p, ""
    return None, ("Couldn't resolve model_path. Use a local .safetensors path, a HF file URL "
                  "(https://huggingface.co/owner/name/resolve/main/model.safetensors), or "
                  "'owner/name::model.safetensors'.")


def _find_tool(root, name):
    for cand in (os.path.join(root, "tools", name), os.path.join(root, name)):
        if os.path.isfile(cand):
            return cand
    return None


def _locate_convert(tool_dir, auto_clone, log):
    """Return the path to ComfyUI-GGUF's convert.py, cloning the repo if allowed. Raises on failure."""
    ud = _clean(tool_dir)
    if ud:
        p = _find_tool(ud, "convert.py")
        if p:
            return p
        raise RuntimeError(f"No tools/convert.py under {ud}. Point tool_dir at a ComfyUI-GGUF checkout.")

    installed = os.path.join(_custom_nodes_dir(), "ComfyUI-GGUF")
    p = _find_tool(installed, "convert.py")
    if p:
        return p

    if not auto_clone:
        raise RuntimeError("ComfyUI-GGUF not found. Set tool_dir, or enable auto_clone to fetch it.")
    if shutil.which("git") is None:
        raise RuntimeError("git is not on PATH, so auto_clone can't run. Install git, or clone "
                           "city96/ComfyUI-GGUF yourself and set tool_dir.")
    log.append(f"Cloning ComfyUI-GGUF into {installed} ...")
    print(f"[GGUFConvert-Diff] git clone {CITY96_URL} -> {installed}")
    rc = _run(["git", "clone", "--depth", "1", CITY96_URL, installed], None, log)
    if rc != 0:
        raise RuntimeError(f"git clone failed (exit {rc}). See the log/console.")
    p = _find_tool(installed, "convert.py")
    if not p:
        raise RuntimeError(f"Cloned ComfyUI-GGUF but no tools/convert.py in {installed}.")
    return p


def _read_gguf_arch(path):
    """Best-effort read of general.architecture from a gguf file; '' if unavailable."""
    try:
        import gguf
        reader = gguf.GGUFReader(path)
        field = reader.fields.get("general.architecture")
        if field is None:
            return ""
        raw = field.parts[field.data[0]]
        b = bytes(raw.tolist()) if hasattr(raw, "tolist") else bytes(raw)
        return b.decode("utf-8", "replace").strip().lower()
    except Exception:
        return ""


class SafetensorsToGGUFDiffusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_diffusion_model_list(), {"tooltip": "Diffusion model from ComfyUI/models/diffusion_models or /unet. Choose the placeholder to type a path/URL in model_path. The dropdown is searchable."}),
                "model_path": ("STRING", {"default": "", "tooltip": "Used when 'model' is the placeholder: a local .safetensors/.ckpt path, a HF file URL (…/resolve/main/model.safetensors), or 'owner/name::model.safetensors'. Give the single diffusion model file, not a folder."}),
                "output_dir": ("STRING", {"default": "", "tooltip": "Folder for the .gguf. Created if missing. Blank -> ComfyUI/models/diffusion_models."}),
                "output_name": ("STRING", {"default": "", "tooltip": "Base filename without extension. Blank -> derived from the source file. The precision/quant tag and .gguf are added automatically."}),
                "quantize": (QUANTS, {"default": "none", "tooltip": "Optional 2nd pass to shrink the model (e.g. Q4_K_S, Q8_0). none = keep the F16 gguf. Needs the PATCHED llama-quantize from city96's llama.cpp fork."}),
                "quantize_binary": (_list_quantize_binaries(), {"tooltip": "Patched llama-quantize executable (auto-listed from ComfyUI/models/llm). Choose the placeholder to type a path in quantize_binary_path. Only used when quantize != none."}),
                "quantize_binary_path": ("STRING", {"default": "", "tooltip": "Full path to the patched llama-quantize(.exe), used when the dropdown is the placeholder."}),
            },
            "optional": {
                "fix_5d_tensors": (["auto", "on", "off"], {"default": "auto", "tooltip": "Wan 2.1 / HunyuanVideo need a post-quantize pass to re-add 5-D tensors. auto = run it only when the gguf's arch is wan/hyvid. Ignored when quantize = none."}),
                "tool_dir": ("STRING", {"default": "", "tooltip": "Path to a ComfyUI-GGUF checkout (holds tools/convert.py). Blank -> use an installed custom_nodes/ComfyUI-GGUF or auto-clone it there."}),
                "auto_clone": ("BOOLEAN", {"default": True, "tooltip": "If ComfyUI-GGUF isn't found, git clone it into custom_nodes. Needs git on PATH. (You'll want it installed anyway — its Unet Loader (GGUF) loads the output.)"}),
                "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace token for gated/private files. Leave blank for public models."}),
                "keep_intermediate": ("BOOLEAN", {"default": False, "tooltip": "When quantizing, keep the intermediate F16 gguf (and the pre-fix file) as well. Off -> delete them after."}),
                "force": ("BOOLEAN", {"default": False, "tooltip": "Re-run even if the target .gguf exists. Off -> return the existing file without redoing the work."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("gguf_path", "log", "help")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/GGUF"
    OUTPUT_NODE = True

    def run(self, model, model_path, output_dir, output_name, quantize, quantize_binary,
            quantize_binary_path, fix_5d_tensors="auto", tool_dir="", auto_clone=True,
            hf_token="", keep_intermediate=False, force=False):
        log = deque(maxlen=800)

        def _tail():
            return "\n".join(log)

        def _err(msg):
            print(f"[GGUFConvert-Diff] ERROR: {msg}")
            log.append(f"ERROR: {msg}")
            return (f"[ERROR] {msg}", _tail(), HELP_TEXT)

        # --- resolve the source diffusion model -------------------------------------------
        src, e = _resolve_source(model, model_path, hf_token, log)
        if not src:
            return _err(e)

        # --- output paths ------------------------------------------------------------------
        out_dir = _clean(output_dir) or _default_out_dir()
        if not out_dir:
            return _err("output_dir is empty and ComfyUI/models/diffusion_models is unavailable. Set output_dir.")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as ex:
            return _err(f"Can't create output_dir {out_dir}: {ex}")

        base = _clean(output_name) or os.path.splitext(os.path.basename(src))[0]
        base = os.path.basename(base.replace("\\", "/"))  # tolerate a pasted full path
        base = re.sub(r'[<>:"/\\|?*]', "_", base) or "model"
        f16_path = os.path.join(out_dir, f"{base}-F16.gguf")
        do_quant = quantize != "none"
        final = os.path.join(out_dir, f"{base}-{quantize}.gguf") if do_quant else f16_path

        if os.path.isfile(final) and not force:
            msg = f"Output already exists (force=off): {final}"
            print(f"[GGUFConvert-Diff] {msg}")
            log.append(msg)
            return (final, _tail(), HELP_TEXT)

        # --- locate ComfyUI-GGUF's convert.py ---------------------------------------------
        try:
            convert_py = _locate_convert(tool_dir, auto_clone, log)
        except Exception as ex:
            return _err(str(ex))
        tools_dir = os.path.dirname(convert_py)

        # --- step 1: convert safetensors -> F16 gguf --------------------------------------
        log.append(f"Converting {src} -> {f16_path} ...")
        rc = _run([sys.executable, convert_py, "--src", src, "--dst", f16_path], tools_dir, log)
        if rc != 0:
            return _err(f"convert.py failed (exit {rc}). See the log above.")
        if not os.path.isfile(f16_path):
            return _err(f"Conversion reported success but {f16_path} is missing.")

        # convert.py writes F16 or BF16 depending on the source; rename to the real one so the
        # filename doesn't claim -F16 for a BF16 model.
        real = _read_gguf_ftype(f16_path)
        if real:
            want = os.path.join(out_dir, f"{base}-{real.upper()}.gguf")
            if os.path.normcase(os.path.abspath(want)) != os.path.normcase(os.path.abspath(f16_path)):
                try:
                    os.replace(f16_path, want)
                    log.append(f"Detected precision {real.upper()}; renamed -> {os.path.basename(want)}")
                    f16_path = want
                except Exception as ex:
                    log.append(f"Could not rename {f16_path} -> {want}: {ex}")

        if not do_quant:
            print(f"[GGUFConvert-Diff] done -> {f16_path}")
            log.append(f"Done -> {f16_path}")
            return (f16_path, _tail(), HELP_TEXT)

        # --- step 2: quantize (patched llama-quantize) ------------------------------------
        qbin = _resolve_binary(quantize_binary, quantize_binary_path)
        if not qbin or not os.path.isfile(qbin):
            return _err(f"quantize = {quantize} needs the PATCHED llama-quantize (city96 fork), but "
                        f"none was found ({qbin or 'nothing selected'}). Build it per "
                        f"ComfyUI-GGUF/tools/README.md and set quantize_binary_path, or use "
                        f"quantize = none. The stock llama.cpp binary won't work for diffusion.")

        # Decide whether the 5-D tensor fix is needed.
        apply_fix = False
        if fix_5d_tensors == "on":
            apply_fix = True
        elif fix_5d_tensors == "auto":
            arch = _read_gguf_arch(f16_path)
            apply_fix = arch in _FIX_5D_ARCHS
            log.append(f"arch = {arch or '?'} -> fix_5d {'ON' if apply_fix else 'off'}")

        if apply_fix:
            fix_script = _find_tool(os.path.dirname(tools_dir), "fix_5d_tensors.py") \
                or os.path.join(tools_dir, "fix_5d_tensors.py")
            if not os.path.isfile(fix_script):
                return _err(f"fix_5d_tensors needed but fix_5d_tensors.py not found next to convert.py "
                            f"({tools_dir}). Update ComfyUI-GGUF, or set fix_5d_tensors = off.")
            raw = os.path.join(out_dir, f"{base}-{quantize}.pre5d.gguf")
            log.append(f"Quantizing -> {raw} ({quantize}) ...")
            rc = _run([qbin, f16_path, raw, quantize], tools_dir, log)
            if rc != 0:
                return _err(f"llama-quantize failed (exit {rc}). The F16 gguf was kept.")
            log.append(f"Fixing 5-D tensors -> {final} ...")
            rc = _run([sys.executable, fix_script, "--src", raw, "--dst", final], tools_dir, log)
            if rc != 0:
                return _err(f"fix_5d_tensors.py failed (exit {rc}). Kept {raw}.")
            if not keep_intermediate:
                for p in (raw,):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        else:
            log.append(f"Quantizing -> {final} ({quantize}) ...")
            rc = _run([qbin, f16_path, final, quantize], tools_dir, log)
            if rc != 0:
                return _err(f"llama-quantize failed (exit {rc}). The F16 gguf was kept.")

        if not os.path.isfile(final):
            return _err(f"Quantize reported success but {final} is missing.")

        if not keep_intermediate:
            try:
                os.remove(f16_path)
                log.append(f"Removed intermediate {f16_path}")
            except Exception as ex:
                log.append(f"Could not remove intermediate {f16_path}: {ex}")

        print(f"[GGUFConvert-Diff] done -> {final}")
        log.append(f"Done -> {final}")
        return (final, _tail(), HELP_TEXT)


NODE_CLASS_MAPPINGS = {"SafetensorsToGGUFDiffusion": SafetensorsToGGUFDiffusion}
NODE_DISPLAY_NAME_MAPPINGS = {"SafetensorsToGGUFDiffusion": "Diffusion Safetensors -> GGUF (city96)"}
