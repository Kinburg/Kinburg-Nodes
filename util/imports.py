"""Defuse third-party lazy modules that make Python's own introspection throw.

speechbrain 1.1 registers its optional submodules as `LazyModule` objects and puts them **in
`sys.modules`**. Touching an attribute on one triggers the real import, and for the ones whose
dependency is absent (k2, spacy, flair, …) that raises `ImportError`. `inspect.getmodule()` walks
every value in `sys.modules` doing `hasattr(module, "__file__")` — so once anything anywhere in the
process has imported speechbrain, **any** call to `inspect.getmodule` blows up:

    ImportError: Lazy import of LazyModule(package=None,
                 target=speechbrain.integrations.k2_fsa, loaded=False) failed

speechbrain guards against exactly this (`utils/importutils.py`, `ensure_module`) — but the guard is

    if importer_frame.filename.endswith("/inspect.py"):

with a forward slash, and on Windows the frame's filename is `…\\Lib\\inspect.py`. So the guard never
fires and the failure is **Windows-only**. It is also nothing to do with whichever node reports it:
one node importing speechbrain arms it, and an unrelated node calling into `inspect` — directly, or
through `warnings`, `torch`, or the traceback machinery — is where it goes off. In this house the
importer is `ComfyUI_MusicTools` and the casualty was `ComfyUI-AudioSR`.

The fix is to replace the unresolvable entries with empty stand-in modules, after which `hasattr` on
them answers instead of raising. Nothing is taken away: they could not be imported in the first place.

Call `defuse_lazy_modules()` from a node's `run()` rather than at import time — the mine is armed
whenever the *other* pack loads, which may be after this one.
"""
import sys
import types

# Only these packages are touched. A lazy module is a legitimate pattern; the bug being worked
# around is specific, so the blast radius is kept specific too.
KNOWN = ("speechbrain",)

_done = set()


def _lazy_type(package):
    """The `LazyModule` class of an already-imported package, or None. Never imports anything: if the
    package is absent there is nothing to defuse."""
    mod = sys.modules.get(package)
    if mod is None:
        return None
    utils = sys.modules.get(f"{package}.utils.importutils")
    return getattr(utils, "LazyModule", None) if utils else None


def defuse_lazy_modules(packages=KNOWN, verbose=True):
    """Replace unimportable lazy `sys.modules` entries with stubs. Returns the names stubbed.

    Cheap and idempotent — one pass over `sys.modules` per package, and each package is only ever
    processed once per process, so this can sit at the top of a node's `run()`."""
    stubbed = []
    for package in packages:
        if package in _done:
            continue
        lazy_type = _lazy_type(package)
        if lazy_type is None:
            continue
        for name, mod in list(sys.modules.items()):
            if not isinstance(mod, lazy_type) or not name.startswith(package):
                continue
            try:
                mod.ensure_module(1)      # resolves? then `hasattr` on it is already safe
            except Exception:
                stub = types.ModuleType(name)
                stub.__doc__ = (f"Stand-in for {name}, which cannot be imported here (its optional "
                                f"dependency is missing). Replaced by kinburg-nodes so that "
                                f"inspect.getmodule() does not raise while walking sys.modules.")
                stub.__file__ = None
                stub.__spec__ = None
                sys.modules[name] = stub
                stubbed.append(name)
        _done.add(package)
    if stubbed and verbose:
        print(f"[Kinburg] defused {len(stubbed)} unimportable lazy module(s) that would break "
              f"inspect.getmodule(): {', '.join(stubbed)}")
    return stubbed


def reset_for_tests():
    """Forget which packages have been processed. For the suite only."""
    _done.clear()
