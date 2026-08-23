"""The Settings node's reasoning controls, and the reasoning split they produce.

`enable_thinking` / `reasoning_effort` are chat-template VARIABLES, so what matters on this side is
only what lands in the request: nothing at all on `model default` (an undefined variable is not the
same as a false one — Qwen3.5/3.8 read it as thinking ON, Gemma-4 as OFF), and no change to the load
signature either way. The worker end of the same feature is in `test_worker.py`.

`_split_reasoning` is here too because those templates end the PROMPT with an open `<think>`: the
model writes only the closing tag, and without putting the opener back the whole reasoning phase
would come out as the answer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import comfy_on_path, fake_package, load_module  # noqa: E402

comfy_on_path()
fake_package("kn", "local_llm")
ln = load_module("kn.local_llm.llm_node", "local_llm/llm_node.py")
load_module("kn.local_llm.settings_node", "local_llm/settings_node.py")

fails = []


def check(label, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + label + (("  " + str(extra)) if extra else ""))
    if not cond:
        fails.append(label)


def tkw(**cfg):
    return ln._template_kwargs(cfg.get)


# ── what reaches the template ───────────────────────────────────────────────────────────────
check("an untouched config asks for nothing", tkw() == {}, tkw())
check("'model default' is not False — it defines no variable at all",
      tkw(enable_thinking="model default", reasoning_effort="model default") == {},
      tkw(enable_thinking="model default", reasoning_effort="model default"))
check("off sends the boolean False", tkw(enable_thinking="off") == {"enable_thinking": False})
check("on sends the boolean True", tkw(enable_thinking="on") == {"enable_thinking": True})
check("a named effort goes through as-is",
      tkw(reasoning_effort="xhigh") == {"reasoning_effort": "xhigh"})
check("custom takes the typed value (gpt-oss's 'high' has no place in Qwen's list)",
      tkw(reasoning_effort="custom", reasoning_effort_custom=" high ")
      == {"reasoning_effort": "high"})
check("custom with an empty field sends nothing",
      tkw(reasoning_effort="custom", reasoning_effort_custom="  ") == {})
check("both knobs travel together",
      tkw(enable_thinking="off", reasoning_effort="low")
      == {"enable_thinking": False, "reasoning_effort": "low"})

# ── the widgets themselves ──────────────────────────────────────────────────────────────────
w = ln._base_config_widgets()
for name in ("enable_thinking", "reasoning_effort", "reasoning_effort_custom"):
    check(f"the Settings node offers {name}", name in w)
check("both selectors default to the neutral option",
      w["enable_thinking"][1]["default"] == "model default"
      and w["reasoning_effort"][1]["default"] == "model default")
check("the effort list is Qwen3.5/3.8's own (their template errors on anything else)",
      w["reasoning_effort"][0] == ["model default", "xhigh", "medium", "low", "custom"],
      w["reasoning_effort"][0])
# ComfyUI stores widgets_values positionally: anything inserted mid-list shifts every value after
# it in every saved workflow, so the new knobs have to be at the very end.
check("...and they are appended last, so saved workflows keep their values",
      list(w)[-3:] == ["enable_thinking", "reasoning_effort", "reasoning_effort_custom"],
      list(w)[-4:])

# ── the request ─────────────────────────────────────────────────────────────────────────────
BASE = {"model": ln.PLACEHOLDER, "model_path": str(Path(__file__).resolve()),  # any real file
        "system_prompt": "sys", "n_ctx": 4096}


def build(**extra):
    err, ctx = ln.build_llm_request({**BASE, **extra}, "hi")
    assert err is None, err
    return ctx


plain, off = build(), build(enable_thinking="off")
check("no reasoning knobs -> the request carries no template_kwargs at all",
      "template_kwargs" not in plain["req"], plain["req"].get("template_kwargs"))
check("setting one puts it in the request",
      off["req"]["template_kwargs"] == {"enable_thinking": False},
      off["req"].get("template_kwargs"))
check("...but NOT in the load signature — flipping it must not restart the worker",
      off["load_sig"] == plain["load_sig"])

# ── the split, with the opening tag prefilled by the template ───────────────────────────────
check("a normal <think> block still splits",
      ln._split_reasoning("<think>why</think>answer") == ("answer", "why"))
check("a prefilled opener is put back (Qwen3.5/3.8 end the prompt with '<think>')",
      ln._split_reasoning("why\n</think>\n\nanswer") == ("answer", "why"),
      ln._split_reasoning("why\n</think>\n\nanswer"))
check("an unclosed block left by truncation still reads as reasoning",
      ln._split_reasoning("<think>ran out") == ("", "ran out"))
check("text with no tags at all is all answer",
      ln._split_reasoning("just an answer") == ("just an answer", ""))
check("a closing tag AFTER a real opener is not touched twice",
      ln._split_reasoning("a<think>b</think>c</think>d")
      == ("ac</think>d", "b"), ln._split_reasoning("a<think>b</think>c</think>d"))
check("answer_marker still wins over both",
      ln._split_reasoning("why\n</think>\n===GO===\nanswer", "===GO===")
      == ("answer", "why\n</think>"))

print("\n" + ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
