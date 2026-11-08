"""Show Text (Markdown) and Show Text (Lite): the freeze has to keep the upstream from running.

The author runs single nodes and freezes an LLM's text with ``use_saved_text``, so the contracts
worth pinning are ComfyUI's, not display logic:

  * ``value`` must be DECLARED lazy, or ComfyUI never calls ``check_lazy_status`` and the LLM
    upstream runs on every queue whatever the toggle says;
  * frozen → ``[]``: nothing requested, the upstream never runs, ``saved_text`` is the output;
  * not frozen and connected → ``["value"]``; not connected → ``[]``, because requesting an input
    with no link behind it is an error, not a no-op;
  * every input arrives wrapped in a 1-element list (INPUT_IS_LIST) — the toggle included, so a
    bare ``bool(use_saved_text)`` would read ``[False]`` as frozen.

Lite is the same node minus the file side, and it has one contract of its own: no STRING input at
all. A link dropped onto a fresh node takes the first free input of its exact type before a ``*``
one, so a STRING socket on the hidden ``saved_text`` would swallow the text link meant for
``value`` where nobody can see it to reconnect.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "util", "show_text")
st = load_module("kn.show_text.text_node", "show_text/text_node.py")
Full, Lite = st.KinburgShowText, st.KinburgShowTextLite

check = Checker()


def ins(cls):
    return cls.INPUT_TYPES()["optional"]


# ------------------------------------------------------------------------------ the declarations
for cls in (Full, Lite):
    tag = cls.__name__
    check(f"{tag}: value is declared lazy (without this check_lazy_status is never called)",
          ins(cls)["value"][1].get("lazy") is True)
    check(f"{tag}: value is optional, so the node also runs with nothing wired",
          "value" in ins(cls) and not cls.INPUT_TYPES()["required"])
    check(f"{tag}: one STRING output, gathered over the whole list",
          cls.RETURN_TYPES == ("STRING",) and cls.INPUT_IS_LIST is True and cls.OUTPUT_NODE is True)

check("Lite has no save_path / autosave", not {"save_path", "autosave"} & set(ins(Lite)), sorted(ins(Lite)))
check("...and deleting them did not reach into the full node's spec",
      {"save_path", "autosave"} <= set(ins(Full)), sorted(ins(Full)))
# str() first: ANY compares equal to everything, "STRING" included.
lite_strings = [k for k, v in ins(Lite).items() if str(v[0]) == "STRING"]
check("Lite has no STRING input, so a dropped text link lands on value", not lite_strings, lite_strings)
check("...while saved_text still gets a text widget to carry the edit",
      ins(Lite)["saved_text"][1].get("widgetType") == "STRING", ins(Lite)["saved_text"])

# -------------------------------------------------------------------------------- the lazy gate
for cls in (Full, Lite):
    node, tag = cls(), cls.__name__
    check(f"{tag}: frozen requests nothing, so the upstream never runs",
          node.check_lazy_status(value=[None], use_saved_text=[True], saved_text=["x"], markdown=[True]) == [])
    check(f"{tag}: connected and not frozen requests value",
          node.check_lazy_status(value=[None], use_saved_text=[False], saved_text=[""], markdown=[True]) == ["value"])
    check(f"{tag}: unconnected (kwarg omitted) requests nothing",
          node.check_lazy_status(use_saved_text=[False], saved_text=[""], markdown=[True]) == [])
    check(f"{tag}: [False] is not frozen — the toggle is list-wrapped too",
          node.check_lazy_status(value=[None], use_saved_text=[False]) == ["value"])

# ------------------------------------------------------------------------------------- the output
for cls in (Full, Lite):
    node, tag = cls(), cls.__name__
    out = node.run(value=["hello"], markdown=[True], use_saved_text=[False], saved_text=["old"])
    check(f"{tag}: not frozen passes the input through, to both the output and the node",
          out["result"] == ("hello",) and out["ui"]["kinburg_showtext"] == ["hello"], out)
    out = node.run(value=["hello"], markdown=[True], use_saved_text=[True], saved_text=["edited"])
    check(f"{tag}: frozen outputs the saved text, not the input", out["result"] == ("edited",), out)
    check(f"{tag}: frozen with nothing wired still outputs the saved text",
          node.run(markdown=[True], use_saved_text=[True], saved_text=["edited"])["result"] == ("edited",))
    check(f"{tag}: a batch is gathered into one text",
          node.run(value=["a", "b"], use_saved_text=[False], saved_text=[""])["result"] == ("a\n\nb",))
    check(f"{tag}: a dict is shown as JSON",
          node.run(value=[{"k": 1}], use_saved_text=[False], saved_text=[""])["result"] == ('{\n  "k": 1\n}',))

# ------------------------------------------------------------------------------- the file side
with tempfile.TemporaryDirectory() as d:
    target = os.path.join(d, "notes", "report")
    Full().run(value=["# hi"], save_path=[target], autosave=[True], use_saved_text=[False], saved_text=[""])
    written = target + ".md"
    check("full: autosave writes <save_path>.md, creating the folder",
          os.path.isfile(written) and open(written, encoding="utf-8").read() == "# hi")
    Full().run(value=["x"], save_path=[os.path.join(d, "off")], autosave=[False], use_saved_text=[False], saved_text=[""])
    check("full: autosave off writes nothing", not os.path.exists(os.path.join(d, "off.md")))

    cwd = os.getcwd()
    os.chdir(d)
    try:
        before = sorted(os.listdir(d))
        Lite().run(value=["x"], markdown=[True], use_saved_text=[False], saved_text=[""])
        check("Lite writes no file", sorted(os.listdir(d)) == before)
    finally:
        os.chdir(cwd)

check.done()
