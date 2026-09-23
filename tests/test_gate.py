"""Any Gate: a closed gate has to stop BOTH directions, and neither half is its own code.

The node is four lines long, and each is a contract with ComfyUI's executor rather than logic of
its own — which is exactly why it is worth pinning:

  * the blocker it hands back must be the very class ``execution.py`` isinstance-checks. Import it
    from a module that doesn't have it (or let the guarded import quietly fail) and a closed gate
    forwards ``None`` instead — the run sails on and saves a black image;
  * ``check_lazy_status`` returning ``[]`` while closed — not the blocker — is what keeps the
    sampler UPSTREAM from running. The blocker only ever stops what comes after;
  * ...and it must return ``["input"]`` while open, or the value never gets evaluated and the gate
    passes a hole through in the one mode where it is supposed to be a plain wire;
  * the input must be DECLARED lazy, since ComfyUI ignores ``check_lazy_status`` without it — the
    upstream would then run on every queue with nothing to show for it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, comfy_on_path, fake_package, load_module  # noqa: E402

comfy_on_path()
fake_package("kn", "util")
gate = load_module("kn.util.gate", "util/gate.py")
node = gate.AnyGate()

check = Checker()

# --------------------------------------------------------------- the blocker ComfyUI knows about
# execution.py takes it from comfy_execution.graph; the node takes it from graph_utils. Same class
# or the isinstance check that does the blocking never fires.
from comfy_execution.graph import ExecutionBlocker as CORE_BLOCKER  # noqa: E402

check("the guarded import found ComfyUI's blocker", gate.ExecutionBlocker is not None)
check("...and it is the class the executor isinstance-checks",
      gate.ExecutionBlocker is CORE_BLOCKER, gate.ExecutionBlocker)

# ------------------------------------------------------------------------------ the declarations
spec = gate.AnyGate.INPUT_TYPES()
check("the input is required, not optional",
      "input" in spec.get("required", {}) and "input" not in spec.get("optional", {}),
      sorted(spec))
check("the input is declared lazy (without this check_lazy_status is never called)",
      spec["required"]["input"][1].get("lazy") is True, spec["required"]["input"][1])
check("the toggle defaults to open, so a fresh node is a plain wire",
      spec["required"]["enabled"][1]["default"] is True)
check("both slots are the wildcard type", spec["required"]["input"][0] == "IMAGE"
      and gate.AnyGate.RETURN_TYPES[0] == "LATENT", gate.AnyGate.RETURN_TYPES)

# ------------------------------------------------------------------- open: a wire, and only a wire
SENTINEL = object()
check("open: the upstream is requested", node.check_lazy_status(input=None, enabled=True) == ["input"],
      node.check_lazy_status(input=None, enabled=True))
out = node.run(input=SENTINEL, enabled=True)
check("open: the value passes through untouched", out == (SENTINEL,) and out[0] is SENTINEL, out)

# ------------------------------------------------------------------------ closed: stop both ways
check("closed: nothing is requested, so the upstream never runs",
      node.check_lazy_status(input=None, enabled=False) == [],
      node.check_lazy_status(input=None, enabled=False))
blocked = node.run(input=None, enabled=False)[0]
check("closed: the output is a blocker", isinstance(blocked, CORE_BLOCKER), type(blocked))
check("...with message None, so downstream is skipped SILENTLY rather than erroring",
      getattr(blocked, "message", "missing") is None, getattr(blocked, "message", "missing"))
check("closed: even with a value in hand it is not forwarded",
      isinstance(node.run(input=SENTINEL, enabled=False)[0], CORE_BLOCKER))

# A BOOLEAN widget can arrive as 0/1 rather than False/True depending on what feeds it.
check("a falsy 0 closes the gate too", isinstance(node.run(input=SENTINEL, enabled=0)[0], CORE_BLOCKER)
      and node.check_lazy_status(input=None, enabled=0) == [])
check("a truthy 1 opens it", node.run(input=SENTINEL, enabled=1) == (SENTINEL,)
      and node.check_lazy_status(input=None, enabled=1) == ["input"])

check.done()
