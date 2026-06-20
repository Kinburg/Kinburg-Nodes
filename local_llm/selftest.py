# selftest.py -- check that llama-cpp-python sees CUDA in this venv.
# Run:  <venv>\Scripts\python.exe custom_nodes\Kinburg-Nodes\local_llm\selftest.py
# It mirrors the same CUDA-DLL preparation that the real worker (gguf_worker.py) does.
import os, glob, ctypes, importlib.util

spec = importlib.util.find_spec("torch")
if spec and spec.origin:
    tl = os.path.join(os.path.dirname(spec.origin), "lib")
    os.add_dll_directory(tl)
    for pattern in ("cudart64_*.dll", "cublasLt64_*.dll", "cublas64_*.dll"):
        for dll in sorted(glob.glob(os.path.join(tl, pattern))):
            try:
                ctypes.CDLL(dll)
                print("preloaded:", os.path.basename(dll))
            except Exception as e:
                print("preload FAIL:", os.path.basename(dll), e)

import llama_cpp
print("llama_cpp version:", llama_cpp.__version__)
from llama_cpp import llama_print_system_info
info = llama_print_system_info()
print(info.decode("utf-8") if isinstance(info, bytes) else info)
print("IMPORT_OK")
