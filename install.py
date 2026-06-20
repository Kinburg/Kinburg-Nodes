# install.py
# Installs the GPU build of llama-cpp-python that the Local LLM nodes need.
#
# ComfyUI-Manager runs this automatically right after cloning the repo, so a
# normal "install via Manager / git URL" needs no extra steps. You can also run
# it by hand with THIS ComfyUI's python:
#   <ComfyUI>\.venv\Scripts\python.exe custom_nodes\Kinburg-Nodes\install.py
import sys
import importlib.util
import subprocess

# Pinned prebuilt wheel: no source build needed. Bump these together if you move
# to a newer CUDA-major torch (e.g. CUDA 13 -> use a cu13x wheel index).
PKG = "llama-cpp-python==0.3.30"
CUDA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"


def main():
    if importlib.util.find_spec("llama_cpp") is not None:
        print("[Kinburg-Nodes] llama-cpp-python already installed - skipping.")
        return

    cmd = [sys.executable, "-m", "pip", "install", PKG, "--prefer-binary"]
    if sys.platform == "win32":
        # Prebuilt CUDA 12.4 wheel. It does NOT bundle the CUDA runtime; the node
        # loads cudart/cublas from torch\lib at runtime, so a CUDA-12.x torch must
        # be present in the same environment (it is, in a standard ComfyUI install).
        cmd += ["--extra-index-url", CUDA_INDEX]
    else:
        print("[Kinburg-Nodes] Non-Windows platform detected: installing the "
              "default wheel. For a GPU build see the llama-cpp-python docs.")

    print("[Kinburg-Nodes] running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("[Kinburg-Nodes] done. Verify CUDA with: python local_llm/selftest.py")


if __name__ == "__main__":
    main()
