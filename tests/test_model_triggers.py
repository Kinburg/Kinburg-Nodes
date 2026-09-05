"""LoRA trigger words through the Model Library: capture -> store -> Model Select's prompt.

The words are read out of the captured RECIPE, never off a wire — Model Capture's inputs are lazy,
so a STRING coming from `Lora Unlim Accumulator.triggers` would arrive as None, and making it eager
would execute the loader stack the node promises never to touch. So the harvest is what this suite
leans on hardest.

Also covers the other half of that promise on the Select side: a wired `prompt` re-runs the node on
every edit of the text, and `_free()` must NOT throw the resident bundle out of VRAM when replay is
about to build nothing.

Plus the store's write safety, which earned its place the hard way: a library holding 22 captured
bundles was replaced by an empty one, and the shape of the code made that a single unreadable
moment away at any time.

`store` is redirected at a temp file — the real library lives in model_presets/data/store.json and
this must not go near it.
"""
import asyncio
import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import comfy_on_path, fake_package, load_module  # noqa: E402

comfy_on_path()
fake_package("kn", "model_presets", "ouroboros")
# The real ouroboros.nodes drags in torch, the LLM nodes and the judge for one string constant.
_ouro = types.ModuleType("kn.ouroboros.nodes")
_ouro.SAMPLER_CFG = "SAMPLER_CFG"
sys.modules["kn.ouroboros.nodes"] = _ouro


def load(name):
    return load_module("kn.model_presets." + name, "model_presets/" + name + ".py")


store = load("store")
replay = load("replay")
load("save_node")
cap = load("capture_node")
sel = load("select_node")

_TMP = Path(tempfile.mkdtemp(prefix="kinburg-lib-")) / "store.json"
store._store_path = lambda: str(_TMP)

fails = []


def check(label, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + label + (("  " + str(extra)) if extra else ""))
    if not cond:
        fails.append(label)


def lora(nid, name, trigger, sm=1.0, sc=1.0):
    return {nid: {"class_type": "LoraTriggerLoader",
                  "inputs": {"lora_name": name, "strength_model": sm, "strength_clip": sc,
                             "trigger": trigger}}}


GRAPH = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "base.safetensors"}},
         **lora("2", "face.safetensors", "ohwx man, portrait"),
         **lora("3", "off.safetensors", "should-not-appear", sm=0.0, sc=0.0),
         **lora("10", "style.safetensors", "portrait, neon glow", sc=0.0),
         "11": {"class_type": "LoraUnlimAccumulator",
                "inputs": {"model": ["1", 0], "lora_1": ["2", 0], "lora_2": ["3", 0],
                           "lora_3": ["10", 0]}}}

# -- the harvest ----------------------------------------------------------------------------
got = cap.harvest_triggers(GRAPH)
check("words are read straight out of the captured recipe",
      got == "ohwx man, portrait, neon glow", repr(got))
check("...in canonical node order, not string order ('10' after '3')",
      got.index("ohwx man") < got.index("neon glow"), repr(got))
check("a LoRA turned all the way down contributes nothing",
      "should-not-appear" not in got, repr(got))
check("strength_clip 0 alone does NOT disable a LoRA", "neon glow" in got, repr(got))
check("a word two LoRAs share appears once", got.count("portrait") == 1, repr(got))
check("an assembly with no LoRAs harvests nothing",
      cap.harvest_triggers({"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "x"}}}) == "")
check("trigger_words / triggers are honoured too, for other packs' loaders",
      cap.harvest_triggers({"1": {"class_type": "X", "inputs": {"trigger_words": "a"}},
                            "2": {"class_type": "Y", "inputs": {"triggers": "b"}}}) == "a, b")
check("a linked (non-literal) trigger is ignored, not crashed on",
      cap.harvest_triggers({"1": {"class_type": "X", "inputs": {"trigger": ["9", 0]}}}) == "")
check("lora_names lists what a bundle carries",
      cap.lora_names(GRAPH) == ["face.safetensors", "off.safetensors", "style.safetensors"],
      cap.lora_names(GRAPH))

# -- the store ------------------------------------------------------------------------------
RECIPE = {"nodes": {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "base.safetensors"}}},
          "outputs": {"model": ["1", 0]}}
STAGES_EARLY = [{"sampler_name": "euler", "scheduler": "simple", "steps": 4, "cfg": 1.0, "seed": 1}]
store.upsert_model("m1", recipe=RECIPE, triggers="ohwx man")
check("triggers round-trip through the store", store.model_triggers("m1") == "ohwx man")
store.upsert_model("m1", recipe=RECIPE)
check("...and None keeps them — a re-capture must not wipe hand-typed words",
      store.model_triggers("m1") == "ohwx man")
store.upsert_model("m1", triggers="")
check("...while an empty string clears them (that's what the dialog sends)",
      store.model_triggers("m1") == "")
store.upsert_model("m1", triggers="  spaced  ")
check("stored words are trimmed", store.model_triggers("m1") == "spaced")
check("the frontend sees them in full_data",
      store.full_data()["models"]["m1"]["triggers"] == "spaced")
check("a model that never had any reports '' rather than KeyError",
      store.model_triggers("nope") == "")

# -- write safety ------------------------------------------------------------------------------
# A read that fails must never become a write that erases. The old `_load()` returned an empty
# library on ANY error and the caller wrote that over the real thing; recipes are captured from
# loader stacks that get deleted afterwards, so what is lost that way exists nowhere else.
store.upsert_model("keepme", recipe=RECIPE, triggers="do not lose me")
_TMP.write_text("{ not json at all", encoding="utf-8")
try:
    store.upsert_preset("keepme", "x", {"stages": STAGES_EARLY})
    check("an unreadable library refuses the write", False, "it went through")
except RuntimeError as e:
    check("an unreadable library refuses the write", "refusing to write" in str(e), str(e)[:60])
check("...and leaves the file exactly as it found it",
      _TMP.read_text(encoding="utf-8") == "{ not json at all")
check("reads stay lenient, so the nodes still load", store.model_names() == [store.NONE],
      store.model_names())
bak = _TMP.with_suffix(".json.bak1")
check("every write left a backup behind", bak.exists(), bak)
bak.replace(_TMP)
# One write behind by definition — .bak1 is the file as it stood BEFORE the last save. That is the
# point: something readable to fall back to, not a second copy of whatever just went wrong.
check("...and copying it back gives a working library again",
      store.model_names() == [store.NONE, "m1"], store.model_names())

# -- families ----------------------------------------------------------------------------------
# A family is stored nowhere on its own: all_families() derives the list from whoever declares one.
# So fixing a typo used to mean visiting every model and every shared preset that carried it, and
# from the nodes there was no way at all.
store.upsert_model("fam_a", recipe=RECIPE, families="krea2_real, flow")
store.upsert_model("fam_b", recipe=RECIPE, families="Krea2_Real")
store.upsert_preset("", "sh_one", {"stages": STAGES_EARLY, "families": "krea2_real"}, shared=True)
store.upsert_preset("", "sh_none", {"stages": STAGES_EARLY, "families": ""}, shared=True)
check("the family list is derived from its holders",
      "krea2_real" in store.all_families() and "flow" in store.all_families(),
      store.all_families())
use = store.family_usage()
check("usage names both the models and the shared presets",
      use["krea2_real"]["models"] == ["fam_a", "fam_b"]
      and use["krea2_real"]["shared"] == ["sh_one"], use.get("krea2_real"))
check("a family only one model declares is reported that way",
      use["flow"] == {"models": ["fam_a"], "shared": []}, use.get("flow"))

touched, _ = store.rename_family("KREA2_REAL", "krea2_realism")
check("a rename matches case-insensitively and hits every holder", touched == 3, touched)
check("...models included, whatever case they wrote it in",
      store.get_model("fam_b")["families"] == ["krea2_realism"],
      store.get_model("fam_b")["families"])
check("...shared presets included",
      store.full_data()["shared"]["sh_one"]["families"] == ["krea2_realism"])
check("...and the old name is gone from the derived list",
      "krea2_real" not in store.all_families(), store.all_families())
check("...while the family it was not about is untouched", "flow" in store.all_families())

touched, orphaned, _ = store.delete_family("krea2_realism")
check("a delete also hits every holder", touched == 3, touched)
check("...and reports the shared preset it left with no family", orphaned == ["sh_one"], orphaned)
check("...but does NOT delete that preset — that is a separate decision",
      "sh_one" in store.full_data()["shared"], sorted(store.full_data()["shared"]))
check("...and the models simply lose the tag",
      store.get_model("fam_a")["families"] == ["flow"], store.get_model("fam_a")["families"])
for bad, why in ((("", "x"), "no name"), (("flow", ""), "no new name")):
    try:
        store.rename_family(*bad)
        check(f"a rename with {why} is refused", False, "it went through")
    except ValueError:
        check(f"a rename with {why} is refused", True)
store.upsert_preset("", "sh_one", None, shared=True, delete=True)
store.upsert_preset("", "sh_none", None, shared=True, delete=True)

# -- the join -------------------------------------------------------------------------------
check("words go in their own paragraph after the prompt",
      sel.join_prompt("a cat", "ohwx man") == "a cat\n\nohwx man")
check("no prompt -> just the words", sel.join_prompt(None, "ohwx man") == "ohwx man")
check("no words -> just the prompt", sel.join_prompt("a cat", "") == "a cat")
check("neither -> empty", sel.join_prompt(None, None) == "")

# -- Model Select ---------------------------------------------------------------------------
import torch  # noqa: E402  (after the light checks — this is the slow import)


class FakeClip:
    def __init__(self):
        self.seen = []

    def tokenize(self, text):
        self.seen.append(text)
        return {"t": text}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.ones(1, 2, 4), {"pooled_output": torch.ones(1, 4), "text": tokens["t"]}]]


CLIP = FakeClip()
FREED = []
CACHED = {"v": False}


async def fake_replay(recipe, purge_others=False):
    return ({"model": "MODEL", "clip": CLIP, "vae": "VAE"}, ["UNETLoader (built)"])


replay.replay = fake_replay
replay.is_cached = lambda r: CACHED["v"]
sel.ModelSelect._free = lambda self: FREED.append(1)

store.upsert_model("m1", recipe=RECIPE, triggers="ohwx man, neon glow")
NAMES = list(sel.ModelSelect.RETURN_NAMES)


def run(**kw):
    CLIP.seen.clear()
    out = asyncio.run(sel.ModelSelect().run(model="m1", preset=store.NONE, **kw))
    return dict(zip(NAMES, out))


check("the output tuple still matches RETURN_TYPES",
      len(sel.ModelSelect.RETURN_TYPES) == len(NAMES) == len(run()), len(run()))

r = run(prompt="a cat on a roof")
check("the bundle's words are appended to the prompt",
      r["prompt"] == "a cat on a roof\n\nohwx man, neon glow", repr(r["prompt"]))
check("...and come out on their own for Ouroboros' trigger_words",
      r["triggers"] == "ohwx man, neon glow", repr(r["triggers"]))
check("the combined prompt is what got encoded", CLIP.seen[0] == r["prompt"], CLIP.seen)
check("no negative prompt -> the negative is the ZEROED positive, not an encode of ''",
      len(CLIP.seen) == 1 and float(r["negative"][0][0].abs().sum()) == 0.0, CLIP.seen)
check("...pooled_output is zeroed too",
      float(r["negative"][0][1]["pooled_output"].abs().sum()) == 0.0)
check("...and the positive is untouched by that", float(r["positive"][0][0].abs().sum()) > 0)

r = run(prompt="a cat", negative_prompt="blurry, watermark")
check("a wired negative prompt is encoded instead of zeroed",
      CLIP.seen == ["a cat\n\nohwx man, neon glow", "blurry, watermark"], CLIP.seen)
check("...and is a real conditioning", float(r["negative"][0][0].abs().sum()) > 0)

r = run()
check("with no prompt wired nothing is encoded — CLIP is never even touched",
      CLIP.seen == [] and r["positive"] is None and r["negative"] is None, CLIP.seen)
check("...but the words still come out, for a graph that assembles its own prompt",
      r["triggers"] == "ohwx man, neon glow" and r["prompt"] == "ohwx man, neon glow")
check("the info line names the words", "ohwx man, neon glow" in r["info"])
check("gen_extra_info records them, so a saved image says which words ran",
      json.loads(r["gen_extra_info"])[0]["params"]["triggers"] == "ohwx man, neon glow")

# -- _free() must not evict a bundle nothing is about to rebuild ------------------------------
FREED.clear()
CACHED["v"] = False
run(prompt="first")
check("a bundle that is NOT resident is still made room for", len(FREED) == 1, FREED)
FREED.clear()
CACHED["v"] = True
run(prompt="second")
run(prompt="third")
check("editing the prompt of a resident bundle unloads nothing", FREED == [], FREED)
FREED.clear()
CACHED["v"] = False
run(prompt="fourth", unload_others=False)
check("unload_others off frees nothing either way", FREED == [], FREED)

# -- the starting preset Capture can save -----------------------------------------------------
STAGES = [{"sampler_name": "euler", "scheduler": "simple", "steps": 8, "cfg": 1.0, "seed": 7}]
store.upsert_model("m2", recipe=RECIPE)
out = cap.ModelCapture._save_preset("m2", STAGES, "first try")
check("Capture can store a starting preset in the registering run",
      "first try" in store.preset_names("m2"), store.preset_names("m2"))
check("...and it becomes the default, since it is the only one",
      (store.get_preset("m2", "first try") or {}).get("default") is True, out)
cap.ModelCapture._save_preset("m2", STAGES, "second")
check("a later one does not steal the default",
      (store.get_preset("m2", "second") or {}).get("default") is False)
check("an empty name saves nothing and says so",
      not cap.ModelCapture._save_preset("m2", None, ""))
check("a name with no stages is an error, not a silent no-op",
      "NOT saved" in " ".join(cap.ModelCapture._save_preset("m2", None, "empty")))

print("\n" + ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
