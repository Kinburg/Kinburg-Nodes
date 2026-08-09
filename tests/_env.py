"""Where everything is, worked out from this file's own location.

The suites used to carry absolute paths, which made them run on exactly one machine. Nothing here
may hardcode a path: the pack sits at ``<ComfyUI>/custom_nodes/<whatever it was cloned as>/`` and
that is all any of this needs to know.
"""
import importlib.util
import sys
import types
from pathlib import Path

TESTS = Path(__file__).resolve().parent
PACK = TESTS.parent                      # …/custom_nodes/kinburg-nodes
COMFY = PACK.parents[1]                  # …/ComfyUI  (the one with folder_paths.py in it)


def comfy_on_path():
    """Make ComfyUI's own modules importable. Needed by any suite that loads the real pack."""
    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))
    return COMFY


def load_pack(name="kn"):
    """Import the WHOLE pack under `name`. Pulls in torch and comfy, so it is slow — use it only
    where a suite genuinely needs the real nodes (the folder has a hyphen in it, so a plain import
    can never work)."""
    comfy_on_path()
    spec = importlib.util.spec_from_file_location(
        name, str(PACK / "__init__.py"), submodule_search_locations=[str(PACK)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fake_package(root_name, *subpackages):
    """Stand-in parent packages, so ONE module can be loaded without its neighbours.

    `from ..local_llm.attachments import …` only resolves if the parents exist, and importing the
    real ones would drag torch, comfy and the H3 extras into a suite meant to stay small.
    """
    root = types.ModuleType(root_name)
    root.__path__ = [str(PACK)]
    sys.modules[root_name] = root
    for sub in subpackages:
        m = types.ModuleType(f"{root_name}.{sub}")
        m.__path__ = [str(PACK / sub)]
        sys.modules[f"{root_name}.{sub}"] = m
    return root


def load_module(dotted, relpath):
    """One module of the pack, by path, under a name of your choosing."""
    spec = importlib.util.spec_from_file_location(dotted, str(PACK / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


class Checker:
    """The tiny assertion helper every suite shares: print a line, remember the failures."""

    def __init__(self):
        self.fails = []

    def __call__(self, label, cond, extra=""):
        print(("  ok   " if cond else "  FAIL ") + label + (("  " + str(extra)) if extra else ""))
        if not cond:
            self.fails.append(label)

    def done(self):
        print("\n" + ("ALL PASS" if not self.fails else "FAILED: " + ", ".join(self.fails)))
        sys.exit(1 if self.fails else 0)
