"""Morpheus sampler: LoRA triggers must land where they help, and never in the negative block."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import COMFY, PACK, comfy_on_path, fake_package, load_module, load_pack  # noqa: E402

comfy_on_path()

spec = importlib.util.spec_from_file_location("kn", str(PACK / "__init__.py"),
                                              submodule_search_locations=[str(PACK)])
kn = importlib.util.module_from_spec(spec)
sys.modules["kn"] = kn
spec.loader.exec_module(kn)
mn = sys.modules["kn.morpheus.nodes"]
sb = sys.modules["kn.morpheus.storyboard"]
wt = mn._with_triggers

fails = []


def check(label, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + label + (("  " + str(extra)) if extra else ""))
    if not cond:
        fails.append(label)


# A real assembled prompt, straight from the node that writes them, so the section layout is not
# something this test invented.
BIBLE = {"style": "grainy 35mm", "subject": "a woman in a navy dress",
         "audio_bed": "room tone", "negative": "text, logos, blur"}
BODY = {"SITUATION": "she stands at the window", "STORYBOARD": "[0s-2s] Beat 1: she turns",
        "CAMERA": "slow push in", "AUDIO": "footsteps"}
REAL = sb.KinburgMorpheusStoryboard._assemble(BIBLE, BODY, True)
TRIG = "kinburgstyle, ohwx woman"

check("the fixture really ends on the negative section",
      REAL.rstrip().splitlines()[-1].lower().startswith("6. [negative"),
      REAL.rstrip().splitlines()[-1][:40])

out = wt(REAL, TRIG)
neg_at = out.lower().index("[negative")
trg_at = out.index("kinburgstyle")
check("triggers land BEFORE the negative section", trg_at < neg_at, (trg_at, neg_at))
check("...and after the positive sections",
      out.index("slow push in") < trg_at, (out.index("slow push in"), trg_at))
check("both triggers are in", "kinburgstyle" in out and "ohwx woman" in out)
check("the negative list is untouched and still last",
      "text, logos, blur" in out.split("[Negative")[1] and out.rstrip().endswith(
          REAL.rstrip().rsplit(": ", 1)[-1]), out.rstrip()[-40:])
check("no trigger slipped in after the negative label",
      "kinburgstyle" not in out.split("[Negative")[1].lower(), out.split("[Negative")[1][:60])
# only sections 1 and 6 come from the bible; the rest is the shot's own body
check("nothing else was lost",
      all(k in out for k in ("grainy 35mm", "she turns", "slow push in", "footsteps")))
check("the six sections are still all there",
      all(f"{i}." in out for i in range(1, 7)))

# idempotence — this is what lets it run again after the in-loop writer
check("applying twice changes nothing", wt(out, TRIG) == out)
check("a trigger already present is not repeated",
      wt("a shot with kinburgstyle in it already. 6. [Negative]: x", "kinburgstyle").count("kinburgstyle") == 1)
check("case is ignored when checking",
      wt("KinburgStyle here. 6. [Negative]: x", "kinburgstyle").lower().count("kinburgstyle") == 1,
      wt("KinburgStyle here. 6. [Negative]: x", "kinburgstyle"))
check("a partly-new list adds only the new one",
      wt(REAL + " kinburgstyle", TRIG).count("ohwx woman") == 1
      and wt(REAL + " kinburgstyle", TRIG).count("kinburgstyle") == 1)

# no triggers, no change
check("empty triggers leave the prompt alone", wt(REAL, "") == REAL)
check("whitespace-only triggers too", wt(REAL, "   ,  , ") == REAL)
check("None is safe", wt(REAL, None) == REAL)
check("an empty prompt still gets the triggers", wt("", TRIG) == "kinburgstyle, ohwx woman")
check("a None prompt is safe", wt(None, TRIG) == "kinburgstyle, ohwx woman")

# a hand-written prompt with no sections at all: fall back to appending
plain = "a woman crosses a sunlit kitchen"
check("a prompt with no negative section gets them appended",
      wt(plain, TRIG) == plain + "\n\nkinburgstyle, ohwx woman", repr(wt(plain, TRIG)))
check("...with a trailing comma tidied",
      wt("a woman, ", TRIG) == "a woman\n\nkinburgstyle, ohwx woman", repr(wt("a woman, ", TRIG)))

# the label is matched loosely, so a hand-edited prompt still works
for variant in ("6. [Negative Prompt/Constraints]: x", "[NEGATIVE]: x", "  6 . [negative]: x",
                "6.[Negative Prompt]: x"):
    got = wt("positive text\n\n" + variant, "trg")
    check(f"label variant {variant[:24]!r:28} is respected",
          got.index("trg") < got.lower().index("[negative"), repr(got[-60:]))

# more than one negative-looking line: the LAST one wins, so nothing positive ends up after it
two = "pos\n\n6. [Negative]: a\n\nmore pos\n\n6. [Negative]: b"
got = wt(two, "trg")
check("with two negative sections the triggers go before the last",
      got.index("trg") > got.index("more pos")
      and got.index("trg") < got.rindex("[Negative]"), repr(got))

# ── the input is declared and reaches render() ───────────────────────────────────────────────
opt = mn.KinburgMorpheus.INPUT_TYPES()["optional"]
check("lora_triggers is an optional connect-only input",
      opt["lora_triggers"][0] == "STRING" and opt["lora_triggers"][1].get("forceInput") is True,
      opt["lora_triggers"])
import inspect
sig = inspect.signature(mn.KinburgMorpheus.render).parameters
check("render takes it, defaulting to empty", sig["lora_triggers"].default == "",
      sig["lora_triggers"].default)
check("...and it is optional, so old workflows still call render fine",
      all(p.default is not inspect.Parameter.empty
          for name, p in sig.items() if name in ("lora_triggers", "llm_config")))

# the accumulator really does emit triggers to wire in
la = sys.modules["kn.lora.lora_nodes"]
acc = [c for c in la.NODE_CLASS_MAPPINGS.values() if "triggers" in getattr(c, "RETURN_NAMES", ())]
check("Lora Unlim Accumulator has a triggers output to wire here", bool(acc),
      [getattr(c, "RETURN_NAMES", None) for c in la.NODE_CLASS_MAPPINGS.values()])

print("\n" + ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
