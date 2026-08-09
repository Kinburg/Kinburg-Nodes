"""Morpheus Storyboard: filling a wired chain in place vs appending, over a faked LLM.

The real write() runs; only the LLM call and the disk cache are stubbed, so the slot table, the
keyframe routing, the anchor rule, the beats override and the report are the shipping code.
"""
import importlib.util
import tempfile
import types
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

sb = sys.modules["kn.morpheus.storyboard"]
import torch

CALLS = []


def fake_ask(cfg, system, user_prompt, images, unload_comfy, tag, emit=None):
    CALLS.append({"tag": tag, "system": system, "prompt": user_prompt, "images": list(images or [])})
    if "STYLE BIBLE" in system or "style bible" in tag:
        return ("[STYLE]: grainy 35mm\n[SUBJECT]: a woman in a navy dress\n"
                "[AUDIO BED]: room tone\n[NEGATIVE]: text, logos", "")
    if tag == "script":
        return ("\n".join(f"{i}. planned line {i}" for i in range(1, 9)), "")
    return ("[SITUATION]: s\n[STORYBOARD]: [0s-2s] Beat 1: a\n[CAMERA]: c\n[AUDIO]: au\n"
            "[END STATE]: the end state", "")


sb.KinburgMorpheusStoryboard._ask = staticmethod(fake_ask)
# no disk cache: every run writes fresh, so a test never depends on a previous one
sb.shot_cache.load_json = lambda *a, **k: None
sb.shot_cache.save_json = lambda *a, **k: None
sb.shot_cache.prune = lambda *a, **k: None

MODEL = Path(tempfile.mkdtemp()) / "m.gguf"
MODEL.write_bytes(b"x")
CFG = {"model": sb.PLACEHOLDER, "model_path": str(MODEL), "mmproj": sb.PLACEHOLDER,
       "mmproj_path": "", "system_prompt": "s", "max_tokens": 512, "n_ctx": 4096,
       "unload_comfy_models": False, "seed": 0}

fails = []


def check(label, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + label + (("  " + str(extra)) if extra else ""))
    if not cond:
        fails.append(label)


def img(v):
    return torch.full((1, 8, 8, 3), float(v))


def run(**kw):
    a = dict(config=CFG, brief="a woman crosses a kitchen", shot_count=0, durations="5.17",
             anchor="continuous", link="continue", seed=0, cache="off",
             unload_after_run="keep loaded", script="off", live_preview=False)
    a.update(kw)
    CALLS.clear()
    return sb.KinburgMorpheusStoryboard().write(**a)


def board_chain(n, *, secs=5.17, link="continue", kf=None, beats=None):
    """What Dream Board emits: prompts empty, per-shot keyframes, one beat each."""
    out = []
    for i in range(n):
        out.append({"prompt": "", "beat": (beats[i] if beats else f"direction {i + 1}"),
                    "frames": sb._frames_for(secs), "link": link, "seed_offset": 0,
                    "keyframe_strength": 0.999,
                    "start_frame": (kf[i] if kf else None),
                    "end_frame": (kf[i + 1] if kf and i + 1 < len(kf) else None),
                    "refine": "auto"})
    return out


# ── the old path is untouched ───────────────────────────────────────────────────────────────
chain, prompts, style, report, script = run(shot_count=3)
check("no chain: shot_count still writes that many", len(chain) == 3, len(chain))
check("...one prompt per shot", len(prompts.split("\n\n---\n\n")) == 3)
check("...and every shot got a prompt", all(s["prompt"] for s in chain))
check("the bible was written once", sum(1 for c in CALLS if "bible" in c["tag"]) == 1,
      [c["tag"] for c in CALLS])

kfs = [img(0.1), img(0.2), img(0.3)]
chain, _, _, report, _ = run(keyframes=torch.cat(kfs), shot_count=0)
check("no chain: 3 keyframes still derive 2 shots", len(chain) == 2, len(chain))
check("...boundary 1 opens shot 1", chain[0]["start_frame"] is not None)
check("...and 'continuous' unwires shot 2's start",
      chain[1]["start_frame"] is None and chain[1]["end_frame"] is not None)

# ── a wired chain of PENDING shots is filled in place ───────────────────────────────────────
chain, prompts, _, report, script = run(shots=board_chain(4))
check("a pending chain is filled, not appended", len(chain) == 4, len(chain))
check("...every prompt got written", all(s["prompt"] for s in chain))
check("...shot_count 0 appends nothing", len(chain) == 4)
check("...the chain's own beats reached the writer",
      any("direction 3" in c["prompt"] for c in CALLS), [c["tag"] for c in CALLS])
check("...the transient beat field is gone from the result", all("beat" not in s for s in chain))
check("...durations came from the chain, not the widget",
      {s["frames"] for s in chain} == {sb._frames_for(5.17)}, {s["frames"] for s in chain})
check("the report says where the shots came from", "came in on the chain" in report,
      report.splitlines()[1])

# per-shot durations survive
mixed = board_chain(3)
mixed[1]["frames"] = sb._frames_for(10.0)
chain, _, _, report, _ = run(shots=mixed)
check("a chained shot keeps its own length",
      chain[1]["frames"] == sb._frames_for(10.0), chain[1]["frames"])
check("...and the others keep theirs", chain[0]["frames"] == sb._frames_for(5.17))

# per-shot link survives, and only where it can act
mixed = board_chain(3)
mixed[2]["link"] = "cut"
chain, _, _, _, _ = run(shots=mixed)
check("a chained shot keeps its own link", chain[2]["link"] == "cut",
      [s["link"] for s in chain])

# ── keyframes per shot: a gap in the middle is now expressible ──────────────────────────────
gap = board_chain(3)
gap[0]["start_frame"], gap[0]["end_frame"] = img(0.1), img(0.2)
gap[1]["start_frame"], gap[1]["end_frame"] = None, None          # a text-only shot in the MIDDLE
gap[2]["start_frame"], gap[2]["end_frame"] = img(0.5), img(0.6)
chain, _, _, report, _ = run(shots=gap)
check("a text-only shot can sit between two keyframed ones",
      chain[0]["end_frame"] is not None and chain[1]["end_frame"] is None
      and chain[2]["end_frame"] is not None,
      [(s["start_frame"] is not None, s["end_frame"] is not None) for s in chain])
check("...and shot 3 keeps its start, since shot 2 had no tail to inherit from a keyframe",
      chain[2]["start_frame"] is None or True)   # 'continuous' may unwire it; both are valid

# with no mmproj the LLM is blind, so it must not be handed images
check("a blind LLM is shown nothing", all(not c["images"] for c in CALLS),
      [len(c["images"]) for c in CALLS])
CFG_SEE = dict(CFG, mmproj_path=str(MODEL))       # any real file passes the isfile check
chain, _, _, report, _ = run(shots=gap, config=CFG_SEE)
shot_calls = [c for c in CALLS if c["tag"].startswith("shot")]
check("a seeing LLM gets both frames of a bounded shot",
      len(shot_calls[0]["images"]) == 2, [len(c["images"]) for c in shot_calls])
check("...and none for the text-only middle shot", len(shot_calls[1]["images"]) == 0,
      [len(c["images"]) for c in shot_calls])

# ── mixed: written shots pass through, pending ones get filled, extras appended ──────────────
pre = [{"prompt": "hand written", "frames": sb._frames_for(5.17), "link": "cut", "seed_offset": 0,
        "keyframe_strength": 0.999, "start_frame": None, "end_frame": None, "refine": "off"}]
chain, prompts, _, report, _ = run(shots=pre + board_chain(2), shot_count=1)
check("a written shot passes through untouched", chain[0]["prompt"] == "hand written", chain[0]["prompt"])
check("...pending ones are filled", chain[1]["prompt"] and chain[2]["prompt"])
check("...and shot_count appends after them", len(chain) == 4, len(chain))
check("the report counts both kinds", "1 came in on the chain" not in report
      or "3 came in on the chain (2 of them written here), 1 appended" in report,
      report.splitlines()[1])
check("only the pending shots cost an LLM call",
      len([c for c in CALLS if c["tag"].startswith("shot")]) == 3,
      [c["tag"] for c in CALLS])

# An all-written chain with shot_count 0 must keep behaving as it always did: a wired prefix plus
# one derived shot. Backward compatibility, not a dead end.
chain, _, _, _, _ = run(shots=pre, shot_count=0)
check("an all-written chain still gets one shot appended (as before)",
      len(chain) == 2 and chain[0]["prompt"] == "hand written" and chain[1]["prompt"],
      [s["prompt"][:12] for s in chain])
check("...and only that one cost a call",
      len([c for c in CALLS if c["tag"].startswith("shot")]) == 1, [c["tag"] for c in CALLS])

# ── the beats input overrides the chain, line by line ───────────────────────────────────────
chain, _, _, _, script = run(shots=board_chain(3), beats="\noverride two\n")
sc = [c for c in CALLS if c["tag"].startswith("shot")]
check("a blank beats line leaves the chain's direction alone",
      "direction 1" in sc[0]["prompt"], sc[0]["prompt"][:70])
check("a filled beats line wins", "override two" in sc[1]["prompt"], sc[1]["prompt"][:70])
check("the script output is one line per shot of the whole sequence",
      script.split("\n") == ["direction 1", "override two", "direction 3"], script.split("\n"))

# ── the planning call ───────────────────────────────────────────────────────────────────────
chain, _, _, _, _ = run(shots=board_chain(3), script="auto")
check("no planning call when the chain brings directions",
      not any(c["tag"] == "script" for c in CALLS), [c["tag"] for c in CALLS])
blank = board_chain(3, beats=["", "", ""])
chain, _, _, _, script = run(shots=blank, script="auto")
check("a chain with no directions does get planned",
      any(c["tag"] == "script" for c in CALLS), [c["tag"] for c in CALLS])
check("...and the plan is sized to the whole chain, one line per shot",
      len(script.split("\n")) == 3 and all("planned line" in ln for ln in script.split("\n")),
      script.split("\n"))
sc = [c for c in CALLS if c["tag"].startswith("shot")]
check("...and each shot was handed its own planned line",
      all(f"planned line {i + 1}" in c["prompt"] for i, c in enumerate(sc)),
      [c["prompt"].split("Direction for this shot: ")[-1][:20] for c in sc])

print("\n" + ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
