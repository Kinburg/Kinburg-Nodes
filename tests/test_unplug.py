"""Any Unplug: the node downstream must still RUN, and must see what an empty socket looks like.

Any Gate's blocker stops the run; this one must not. So what is pinned here is the handoff — what
arrives in the slot while the link is off, and that the pack's own consumer treats it as absence:

  * `_with_context` in the LLM node reads the wired-in context as ``(context or "").strip()``, so
    the default `nothing` really does give the same system prompt as an unconnected socket. That
    is the whole promise of the node for the context → Settings → LLM case, and it lives in a file
    this node knows nothing about — exactly the kind of pairing that breaks quietly later;
  * `empty text` exists for consumers that strip or join first and would raise on None, so it has
    to hand over a real ``str``, not None;
  * every value the `when_off` combo advertises has to be one `run` actually understands. A typo
    in either list would silently fall through to None and look fine until an LLM node erred.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, comfy_on_path, fake_package, load_module  # noqa: E402

comfy_on_path()
fake_package("kn", "util", "local_llm")
unplug = load_module("kn.util.unplug", "util/unplug.py")
llm = load_module("kn.local_llm.llm_node", "local_llm/llm_node.py")
node = unplug.AnyUnplug()

check = Checker()

# ------------------------------------------------------------------------------ the declarations
spec = unplug.AnyUnplug.INPUT_TYPES()
check("the input is declared lazy", spec["required"]["input"][1].get("lazy") is True)
check("the toggle defaults to connected", spec["required"]["enabled"][1]["default"] is True)
check("the combo defaults to 'nothing'", spec["required"]["when_off"][1]["default"] == "nothing"
      and spec["required"]["when_off"][0] == unplug.OFF_VALUES, spec["required"]["when_off"][0])

# --------------------------------------------------------------- connected: a wire, and only a wire
SENTINEL = object()
check("connected: the upstream is requested",
      node.check_lazy_status(input=None, enabled=True) == ["input"])
check("connected: the value passes through untouched",
      node.run(input=SENTINEL, enabled=True)[0] is SENTINEL)
check("connected: 'when_off' is ignored while the link is on",
      node.run(input=SENTINEL, enabled=True, when_off="empty text")[0] is SENTINEL)

# ------------------------------------------------------- unplugged: nothing arrives, but it arrives
check("unplugged: nothing is requested, so the branch feeding it never runs",
      node.check_lazy_status(input=None, enabled=False) == [])
check("unplugged/nothing: the slot gets None, what an unconnected optional input holds",
      node.run(input=SENTINEL, enabled=False, when_off="nothing") == (None,))
empty = node.run(input=SENTINEL, enabled=False, when_off="empty text")[0]
check("unplugged/empty text: a real str, so .strip()/join on the far side is a no-op",
      isinstance(empty, str) and empty == "", repr(empty))
check("every advertised off value is one run() understands — none falls through by accident",
      all(node.run(input=SENTINEL, enabled=False, when_off=v)[0] in (None, "")
          for v in unplug.OFF_VALUES))
check("a falsy 0 unplugs the link too",
      node.run(input=SENTINEL, enabled=0)[0] is None
      and node.check_lazy_status(input=None, enabled=0) == [])

# --------------------------------------------- the actual promise, against the consumer in the pack
SYS = "You are a careful assistant."
CTX = "## Characters\nAda — an engineer."
check("the fixture is a context that DOES change the prompt when it is connected",
      llm._with_context(SYS, CTX) != SYS and CTX in llm._with_context(SYS, CTX))
off = node.run(input=CTX, enabled=False, when_off="nothing")[0]
check("unplugged: the LLM's system prompt is byte-for-byte the one it has with no context wired",
      llm._with_context(SYS, off) == SYS, llm._with_context(SYS, off))
check("...and 'empty text' lands in the same place",
      llm._with_context(SYS, node.run(input=CTX, enabled=False, when_off="empty text")[0]) == SYS)
check("connected: the context is back in the prompt, unchanged",
      llm._with_context(SYS, node.run(input=CTX, enabled=True)[0]) == llm._with_context(SYS, CTX))

check.done()
