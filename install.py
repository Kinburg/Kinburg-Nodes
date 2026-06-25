# install.py
# Installs the GPU build of llama-cpp-python that the Local LLM nodes need,
# matched to THIS ComfyUI's CUDA version.
#
# ComfyUI-Manager runs this automatically right after cloning the repo, so a
# normal "install via Manager / git URL" needs no extra steps. You can also run
# it by hand with THIS ComfyUI's python:
#   <ComfyUI>\.venv\Scripts\python.exe custom_nodes\Kinburg-Nodes\install.py
import re
import sys
import subprocess
import importlib.util
import importlib.metadata

# Pinned prebuilt wheel version (no source build needed). The same version is
# published under several CUDA-tagged indexes; we pick the one whose CUDA *major*
# matches the installed torch.
VERSION = "0.3.30"
BASE = "https://abetlen.github.io/llama-cpp-python/whl"

# llama-cpp-python's ggml-cuda.dll is linked against a specific CUDA *major* and
# loads cudart64_<major>.dll / cublas64_<major>.dll BY NAME at runtime. Those DLLs
# ship inside torch\lib, so the wheel's CUDA major MUST match torch's CUDA major:
#   torch 2.x+cu124/cu126/cu128 -> CUDA 12 -> cu124 wheel (needs cudart64_12.dll)
#   torch 2.x+cu130            -> CUDA 13 -> cu130 wheel (needs cudart64_13.dll)
# A mismatch is exactly what breaks new installs: a cu124 wheel on a CUDA-13 torch
# can't find cudart64_12.dll and llama.dll fails with "or one of its dependencies".
CUDA_MAJOR_TO_INDEX = {"12": "cu124", "13": "cu130"}

# Snippet that mirrors gguf_worker._prepare_cuda(): preload the CUDA runtime from
# torch\lib, then import llama_cpp. Used to test whether the current build loads.
_IMPORT_TEST = (
    "import os, glob, ctypes, importlib.util\n"
    "spec = importlib.util.find_spec('torch')\n"
    "if spec and spec.origin:\n"
    "    tl = os.path.join(os.path.dirname(spec.origin), 'lib')\n"
    "    if os.path.isdir(tl):\n"
    "        os.add_dll_directory(tl)\n"
    "        for pat in ('cudart64_*.dll','cublasLt64_*.dll','cublas64_*.dll'):\n"
    "            for d in sorted(glob.glob(os.path.join(tl, pat))):\n"
    "                try: ctypes.CDLL(d)\n"
    "                except Exception: pass\n"
    "import llama_cpp\n"
)


def _torch_cuda_major():
    """CUDA major torch was built for ('12', '13', ...), or None for a CPU torch.
    Parsed from torch's dist version (e.g. '2.10.0+cu130') so torch is not imported."""
    try:
        v = importlib.metadata.version("torch")
    except Exception:
        return None
    m = re.search(r"\+cu(\d+)", v)
    return m.group(1)[:-1] if m else None        # 'cu130' -> '13', 'cu124' -> '12'


def _wheel_index(cuda_major):
    """(index_url, human_label) for the wheel matching this torch. Falls back to
    the prebuilt CPU wheel when there is no (recognised) CUDA torch."""
    sub = CUDA_MAJOR_TO_INDEX.get(cuda_major, "cpu")
    label = f"CUDA {cuda_major}.x ({sub})" if sub != "cpu" else "CPU"
    return f"{BASE}/{sub}", label


def _llama_loads_ok():
    """True only if llama_cpp is installed AND actually loads its shared library
    in a fresh subprocess (a wrong-CUDA build imports as a module but fails here)."""
    if importlib.util.find_spec("llama_cpp") is None:
        return False
    r = subprocess.run([sys.executable, "-c", _IMPORT_TEST],
                       capture_output=True, text=True)
    return r.returncode == 0


def main():
    if sys.platform != "win32":
        # Non-Windows: keep the previous lightweight behaviour. The CUDA-DLL
        # matching above is Windows-specific; on Linux/macOS see the
        # llama-cpp-python docs for a GPU build.
        if importlib.util.find_spec("llama_cpp") is not None:
            print("[Kinburg-Nodes] llama-cpp-python already installed - skipping.")
            return
        print("[Kinburg-Nodes] Non-Windows platform: installing the default wheel. "
              "For a GPU build see the llama-cpp-python docs.")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               f"llama-cpp-python=={VERSION}", "--prefer-binary"])
        return

    index_url, label = _wheel_index(_torch_cuda_major())

    installed = importlib.util.find_spec("llama_cpp") is not None
    if installed and _llama_loads_ok():
        print("[Kinburg-Nodes] llama-cpp-python already installed and loads OK - skipping.")
        return

    cmd = [sys.executable, "-m", "pip", "install",
           f"llama-cpp-python=={VERSION}", "--prefer-binary",
           "--extra-index-url", index_url]
    if installed:
        # A build is present but won't load (almost always a CUDA-major mismatch).
        # The version string is identical across CUDA builds, so pip won't swap it
        # without a forced reinstall. --no-deps leaves numpy/jinja2/etc. untouched.
        cmd += ["--force-reinstall", "--no-deps", "--no-cache-dir"]
        print(f"[Kinburg-Nodes] installed llama-cpp-python fails to load; "
              f"reinstalling the {label} build.")
    else:
        print(f"[Kinburg-Nodes] installing llama-cpp-python ({label} build).")

    print("[Kinburg-Nodes] running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("[Kinburg-Nodes] done. Verify CUDA with: python local_llm/selftest.py")


if __name__ == "__main__":
    main()
