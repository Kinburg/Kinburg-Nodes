"""The speechbrain landmine: one node importing it makes `inspect.getmodule` throw for everyone.

Run against a faithful stand-in rather than the real speechbrain, so the suite pins the *mechanism*
and stays fast — then, at the end, against the real thing if it happens to be installed.

The mechanism, from speechbrain 1.1's `utils/importutils.py`: optional submodules are `LazyModule`
objects placed in `sys.modules`; touching an attribute imports the target for real; and the guard
that is supposed to make `inspect` safe tests `filename.endswith("/inspect.py")`, which never matches
on Windows. So `inspect.getmodule()` — which does `hasattr(m, "__file__")` over every value in
`sys.modules` — raises ImportError, and the node that reports it is rarely the node that caused it.
"""
import inspect
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "util")
guard = load_module("kn.util.imports", "util/imports.py")

check = Checker()


# --------------------------------------------------- a stand-in with the real failure mode in it
class LazyModule(types.ModuleType):
    """As speechbrain's: resolving is deferred, and `__getattr__` is where it goes off."""

    def __init__(self, name, importable):
        super().__init__(name)
        self.target = name
        self.importable = importable
        self.tries = 0

    def ensure_module(self, stacklevel=1):
        self.tries += 1
        if not self.importable:
            raise ImportError(f"Lazy import of LazyModule(target={self.target}) failed")
        return types.ModuleType(self.target)

    def __getattr__(self, attr):
        return getattr(self.ensure_module(1), attr)


def _arm():
    """Install a fake package whose lazy submodules are in sys.modules — one resolvable, two not."""
    pkg = types.ModuleType("fakebrain")
    pkg.__path__ = []
    utils = types.ModuleType("fakebrain.utils.importutils")
    utils.LazyModule = LazyModule
    sys.modules["fakebrain"] = pkg
    sys.modules["fakebrain.utils.importutils"] = utils
    mines = {"fakebrain.k2_fsa": False, "fakebrain.spacy": False, "fakebrain.nnet": True}
    for name, ok in mines.items():
        sys.modules[name] = LazyModule(name, ok)
    return mines


def _sweep():
    """What inspect.getmodule does: `hasattr(m, '__file__')` over every value in sys.modules."""
    for mod in list(sys.modules.values()):
        hasattr(mod, "__file__")


def _explodes():
    try:
        _sweep()
    except ImportError:
        return True
    return False


_arm()
guard.reset_for_tests()
check("the mine is real: walking sys.modules raises before the fix", _explodes())
check("...and inspect.getmodule is exactly that walk",
      "sys.modules" in inspect.getsource(inspect.getmodule))

stubbed = guard.defuse_lazy_modules(packages=("fakebrain",), verbose=False)
check("only the unimportable ones are stubbed",
      sorted(stubbed) == ["fakebrain.k2_fsa", "fakebrain.spacy"], stubbed)
check("the walk is safe afterwards", not _explodes())
check("...and stays safe on a second sweep", not _explodes() and not _explodes())
check("a lazy module that CAN resolve is left alone, not replaced",
      isinstance(sys.modules["fakebrain.nnet"], LazyModule))
check("the stubs are real modules, so hasattr answers instead of importing",
      isinstance(sys.modules["fakebrain.k2_fsa"], types.ModuleType)
      and not isinstance(sys.modules["fakebrain.k2_fsa"], LazyModule)
      and sys.modules["fakebrain.k2_fsa"].__file__ is None)
check("...and they say what they are, for whoever finds one in a traceback",
      "cannot be imported" in (sys.modules["fakebrain.k2_fsa"].__doc__ or ""))

before = sys.modules["fakebrain.nnet"].tries
check("a second call does nothing — one pass per package per process",
      guard.defuse_lazy_modules(packages=("fakebrain",), verbose=False) == []
      and sys.modules["fakebrain.nnet"].tries == before)

check("an absent package is not imported just to check it",
      guard.defuse_lazy_modules(packages=("no_such_package_at_all",), verbose=False) == []
      and "no_such_package_at_all" not in sys.modules)
guard.reset_for_tests()
check("a package present but with no lazy machinery is left alone",
      guard.defuse_lazy_modules(packages=("sys",), verbose=False) == [])
check("only the named package's entries are touched — a lazy module elsewhere is not ours to fix",
      guard.KNOWN == ("speechbrain",))

# ------------------------------------------------------- and against the real thing, if it is here
guard.reset_for_tests()
try:
    import speechbrain  # noqa: F401
    real = True
except Exception:
    real = False

if real:
    from speechbrain.utils.importutils import LazyModule as SBLazy
    lazies = [n for n, m in sys.modules.items() if isinstance(m, SBLazy)]
    check(f"real speechbrain {speechbrain.__version__} does put LazyModules in sys.modules",
          len(lazies) > 0, len(lazies))
    check("...and its inspect guard is the forward-slash one, so Windows is unprotected",
          '"/inspect.py"' in inspect.getsource(SBLazy.ensure_module)
          or "'/inspect.py'" in inspect.getsource(SBLazy.ensure_module))
    broke = _explodes()
    fixed = guard.defuse_lazy_modules(verbose=False)
    check("the real package breaks the walk, and the fix repairs it",
          (not broke) or (fixed and not _explodes()), (broke, fixed))
    check("speechbrain itself still works afterwards", bool(speechbrain.__version__))
else:
    check("speechbrain is not installed here, so only the stand-in was exercised", True)

check.done()
