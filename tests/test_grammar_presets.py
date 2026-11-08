"""Prompt Presets (class id GrammarPresets) — the store and the node.

A preset became a prompt + grammar pair, so the store had to learn four things: read the file
Grammar Presets left behind, keep the built-ins read-only while letting an edit sit on top of them,
rename in one write, and refuse an entry with nothing in it. The node had to stop answering a
missing preset with an empty string — a workflow from another machine would otherwise run without
its grammar and say nothing. Everything runs against a store.json in a temp dir, never the real
library.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "grammar_presets")
store = load_module("kn.grammar_presets.store", "grammar_presets/store.py")
node_mod = load_module("kn.grammar_presets.presets_node", "grammar_presets/presets_node.py")

check = Checker()

TMP = Path(tempfile.mkdtemp(prefix="kinburg_prompts_")) / "data" / "store.json"
store._store_path = lambda: str(TMP)          # the real library must not be touched


def seed(data):
    TMP.parent.mkdir(parents=True, exist_ok=True)
    TMP.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def raises(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return None


CARD = "Character Card (JSON)"
G = 'root ::= "===PROMPT===\\n" body "===END==="'
SYS = "Output exactly matching this format:\n===PROMPT===\n…\n===END==="

# ---------------------------------------------------------- the file Grammar Presets wrote
seed({"grammars": {"Prompt and notes": G}})
check("an old grammars file still loads, as grammar-only presets",
      store.get("Prompt and notes") == {"prompt": "", "grammar": G}, store.get("Prompt and notes"))
check("...and its entries are in the dropdown", "Prompt and notes" in store.list_names())
store.upsert("Prompt and notes", SYS, G)
on_disk = json.loads(TMP.read_text(encoding="utf-8"))
check("the next save rewrites it in the new shape",
      list(on_disk) == ["presets"] and on_disk["presets"]["Prompt and notes"] == {"prompt": SYS, "grammar": G},
      on_disk)

# --------------------------------------------------------------------------------- built-ins
seed({})
check("a built-in is a grammar with no prompt",
      store.get(CARD) == {"prompt": "", "grammar": store.CHARACTER_CARD})
check("the dropdown is None and then the built-ins", store.list_names() == [store.NONE, *store.DEFAULTS])
store.upsert("beta", SYS)
store.upsert("Alpha", SYS)
check("...then yours A→Z regardless of case",
      store.list_names() == [store.NONE, *store.DEFAULTS, "Alpha", "beta"], store.list_names())

seed({})
store.upsert(CARD, "Fill the card from the photo.", store.CHARACTER_CARD)
check("saving under a built-in's name writes an edit over it",
      store.get(CARD)["prompt"] == "Fill the card from the photo.")
check("...which the list marks as an edited built-in",
      store.full_data()["presets"][CARD] == {"builtin": True, "edited": True, "prompt": True, "grammar": True},
      store.full_data()["presets"][CARD])
check("...without listing the name twice", store.list_names().count(CARD) == 1)
store.upsert(CARD, delete=True)
check("deleting the edit brings the original back", store.get(CARD) == {"prompt": "", "grammar": store.CHARACTER_CARD})
check("...unmarked", store.full_data()["presets"][CARD]["edited"] is False)
check("a built-in with no edit over it can't be deleted",
      raises(lambda: store.upsert(CARD, delete=True)) == "cannot delete a built-in preset")

# ------------------------------------------------------------------------- what may be saved
seed({})
check("a preset with neither field is refused",
      raises(lambda: store.upsert("Empty", "  ", "\n")) == "a preset needs a prompt, a grammar, or both")
check("...and nothing was written", store.get("Empty") is None)
store.upsert("Prompt only", SYS, "")
store.upsert("Grammar only", "", G)
flags = store.full_data()["presets"]
check("a prompt alone is a preset", flags["Prompt only"] == {"builtin": False, "edited": False, "prompt": True, "grammar": False})
check("...and so is a grammar alone", flags["Grammar only"]["grammar"] and not flags["Grammar only"]["prompt"])
check("the None option is not a name you can save under", raises(lambda: store.upsert(store.NONE, SYS)) is not None)
check("the list carries no texts — the editor fetches the one it opens",
      all(set(v) == {"builtin", "edited", "prompt", "grammar"} for v in store.full_data()["presets"].values()))

# ------------------------------------------------------------------------------------- rename
seed({"presets": {"Writer": {"prompt": SYS, "grammar": G}, "Critic": {"prompt": "be harsh", "grammar": ""}}})
store.upsert("Novelist", SYS, G, old_name="Writer")
names = store.list_names()
check("a rename lands under the new name", store.get("Novelist") == {"prompt": SYS, "grammar": G})
check("...and leaves no copy behind", "Writer" not in names, names)
check("...without touching anything else", store.get("Critic") == {"prompt": "be harsh", "grammar": ""})
seed({})
store.upsert("Nobody", SYS, old_name="never-existed")
check("a stale old_name degrades to a plain create", store.get("Nobody") is not None)

# ---------------------------------------------------------------------------------------- node
seed({"presets": {"Writer": {"prompt": SYS, "grammar": G}}})
node = node_mod.GrammarPresets()
check("the node emits grammar, prompt and name", node.run("Writer") == (G, SYS, "Writer"))
check("...and the empty string for each under None", node.run(store.NONE) == ("", "", ""))
check("an empty field comes out empty, which the LLM node reads as not set",
      node.run(CARD)[1] == "" and node.run(CARD)[0] == store.CHARACTER_CARD)
err = raises(lambda: node.run("Deleted yesterday"))
check("a preset the library doesn't have stops the run instead of emitting nothing",
      err is not None and "'Deleted yesterday'" in err, err)
check("the outputs keep grammar first, so old links into it don't move",
      node_mod.GrammarPresets.RETURN_NAMES == ("grammar", "prompt", "name"))
check("the class id stays GrammarPresets under its new name",
      node_mod.NODE_DISPLAY_NAME_MAPPINGS == {"GrammarPresets": "Prompt Presets"})

check.done()
