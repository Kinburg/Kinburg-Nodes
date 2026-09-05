"""Phantas Storyboard: the real write() over a faked LLM.

Only the model call, the disk cache and the VRAM unload are stubbed — the counting, the grammar,
the bible stamping, the causal keys, the clock and every output are the shipping code.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, load_pack  # noqa: E402

load_pack()
bd = sys.modules["kn.phantas.board"]
T = sys.modules["kn.phantas.timing"]

check = Checker()

CALLS = []
SAVED = {}          # cache key -> payload, so the causal chain can be inspected
SHUTDOWNS = []

BIBLE = ("[STYLE]: grainy 35mm anamorphic, low sun\n"
         "[CAST]: Mira — a woman of 30, shaved head, jade eyes, silver jacket\n"
         "Dov — a man of 45, heavy beard, round glasses, brown corduroy\n"
         "[SUBJECT]: a coastal highway at dawn\n"
         "[NEGATIVE]: text, logos, extra limbs")

CAST_TEXT = ("Mira — a woman of 30, shaved head, jade eyes, silver jacket\n"
             "Dov — a man of 45, heavy beard, round glasses, brown corduroy")
PRESENT = ["Mira"]


def plan_json(n_frames, weights=None, present=None, with_trans=True):
    w = weights or [1] * (n_frames - 1)
    pres = PRESENT if present is None else present
    obj = {"frames": [{"framing": f"framing {i + 1}", "present": list(pres),
                       "state": f"state {i + 1}"} for i in range(n_frames)]}
    if with_trans:
        obj["transitions"] = [{"beat": f"beat {i + 1}", "weight": w[i]}
                              for i in range(n_frames - 1)]
    return json.dumps(obj)


PLAN = {"n": 4, "weights": None}


def fake_ask(cfg, system, user_prompt, unload_comfy, tag, emit=None, grammar=""):
    CALLS.append({"tag": tag, "system": system, "prompt": user_prompt, "grammar": grammar,
                  "unload_comfy": unload_comfy})
    if tag == "style bible":
        return BIBLE, None
    if tag == "plan":
        # answer the shape the grammar allows, so every beats-off check below is also a check
        # that the node handed the planner the right grammar
        return plan_json(PLAN["n"], PLAN["weights"], with_trans="transitions" in grammar), None
    return f"body of {tag}", None


REAL_ASK = bd._ask          # kept for the call-building checks at the end
bd._ask = fake_ask
bd._shutdown_worker = lambda *a, **k: SHUTDOWNS.append(1)
bd._store.load_json = lambda *a, **k: None          # every run writes fresh
bd._store.save_json = lambda k, obj: SAVED.setdefault(k, obj)
bd._store.prune = lambda *a, **k: None

CFG = {"model": "m.gguf", "model_path": "", "max_tokens": 512, "unload_comfy_models": False}
Node = bd.KinburgPhantasStoryboard()


def run(**kw):
    CALLS.clear()
    SAVED.clear()
    SHUTDOWNS.clear()
    args = dict(config=CFG, brief="a cyclist becomes a demon rider", count_mode="scenes",
                count=3, target_length=0.0, durations="", live_preview=False)
    args.update(kw)
    return Node.write(**args)


# ------------------------------------------------------------------------------- counting units
PLAN["n"] = 4
board, prompts, beats, durs, style, report = run(count_mode="scenes", count=3)
check("3 scenes → 4 keyframes", len(board["frames"]) == 4)
check("3 scenes → 3 shots", len(board["shots"]) == 3)
check("one call per frame, plus bible and plan", len(CALLS) == 6, [c["tag"] for c in CALLS])
check("the calls are labelled for the log",
      [c["tag"] for c in CALLS][:3] == ["style bible", "plan", "frame 1/4"])
check("prompts output has one entry per keyframe", len(prompts.split(bd.PROMPT_SEP)) == 4)
check("beats output has one line per shot", len(beats.strip().split("\n")) == 3)
check("beats are numbered the way Morpheus reads them", beats.startswith("1. beat 1"), beats[:40])
check("durations output has one value per shot", len(durs.split(",")) == 3, durs)
check("links are all continue — v1 has no cuts", board["links"] == ["continue"] * 3)
check("report names the shape", "4 keyframes → 3 shot(s)" in report, report.split("\n")[0])

b2 = run(count_mode="frames", count=4)[0]
check("frames mode with 4 pictures matches scenes mode with 3",
      len(b2["frames"]) == 4 and len(b2["shots"]) == 3)

PLAN["n"] = 7
b3 = run(count_mode="duration", count=0, target_length=31.0)[0]
check("duration mode: 31 s becomes 6 shots / 7 keyframes",
      len(b3["frames"]) == 7 and len(b3["shots"]) == 6)
check("duration mode hits its target",
      abs(T.seconds_for(sum(s["frames"] for s in b3["shots"])) - 31.0) <= 0.36,
      T.describe([s["frames"] for s in b3["shots"]]))

# ------------------------------------------------------------------------------- the style bible
PLAN["n"] = 4
board, prompts, beats, durs, style, report = run()
frames = prompts.split(bd.PROMPT_SEP)
check("the bible's STYLE is stamped on every frame",
      all("grainy 35mm anamorphic" in f for f in frames))
check("the stamp is byte-identical everywhere",
      len({f[:f.index("body of")] for f in frames}) == 1)
check("the SUBJECT is NOT stamped (it is what changes)",
      not any("coastal highway at dawn" in f for f in frames))
check("style output carries the parsed bible", "[STYLE]" in style and "[NEGATIVE]" in style)
check("the negative is on the board for the sampler", "extra limbs" in board["negative"])
check("a bible with no NEGATIVE still gets one",
      "text" in bd.parse_bible("[STYLE]: x")["negative"])
check("bible parsing survives markdown bolding",
      bd.parse_bible("**[STYLE]**: neon")["style"] == "neon")

# ------------------------------------------------------------------------------------- the cast
check("the cast keeps its line breaks — one person per line",
      len(bd.parse_cast(bd.parse_bible(BIBLE)["cast"])) == 2)
roster = bd.parse_cast(CAST_TEXT)
check("a cast line yields the name", [n for n, _ in roster] == ["Mira", "Dov"])
check("…and keeps the whole line verbatim", roster[0][1] == CAST_TEXT.split("\n")[0])
check("a colon separates too", bd.parse_cast("Ana: tall")[0][0] == "Ana")
check("bullets are stripped", bd.parse_cast("- Ana — tall")[0][0] == "Ana")
check("'none' is not a person", bd.parse_cast("none") == [])
check("a name with no description is still a name", bd.parse_cast("Ana")[0] == ("Ana", "Ana"))
check("cast_for picks only who is present", bd.cast_for(["Dov"], roster) == [roster[1][1]])
check("cast_for takes nobody when nobody is named", bd.cast_for([], roster) == [])
check("cast_for is case-insensitive", bd.cast_for(["mira"], roster) == [roster[0][1]])
check("cast_for matches a fuller name", bd.cast_for(["Mira Vale"], roster) == [roster[0][1]])
check("cast_for returns CAST order, not the plan's",
      bd.cast_for(["Dov", "Mira"], roster) == [roster[0][1], roster[1][1]])

check("the present cast member is stamped into every frame",
      all("shaved head, jade eyes" in f for f in frames))
check("…and the absent one is NOT (or the picture grows a person)",
      not any("heavy beard" in f for f in frames))
check("the board records who the plan put in each frame",
      all(f["present"] == ["Mira"] for f in board["frames"]))
check("the frame call is told to describe them by name",
      all("IN THIS FRAME" in c["prompt"] for c in CALLS if c["tag"].startswith("frame ")))

_orig_present = PRESENT[:]
PRESENT[:] = []
_b, _p, _, _, _, _rep = run()
check("an empty 'present' stamps nobody", "shaved head" not in _p)
check("…and the writer is told not to add one",
      any("Do not put a person in it" in c["prompt"] for c in CALLS if c["tag"].startswith("frame ")))
PRESENT[:] = ["Ghost"]
_b, _p, _, _, _, _rep = run()
check("a name that is not in the cast is reported, not stamped",
      "not in the cast" in _rep and "shaved head" not in _p, _rep.split("\n")[-1])
PRESENT[:] = _orig_present

# a cast typed on the node overrides whatever the model wrote, verbatim
_b, _p, _, _, _style, _rep = run(cast="Zoya — 22, red braid, leather flight jacket")
check("a typed cast replaces the model's", "Zoya" in _style and "shaved head" not in _style)
check("…and is what the bible call is shown",
      "these people are FIXED" in [c for c in CALLS if c["tag"] == "style bible"][0]["prompt"])
check("…and the report says whose it is", "(yours, verbatim)" in _rep)
check("the report names the cast", "cast: 2 named (written by the bible call) — Mira, Dov" in run()[5])

# ----------------------------------------------------------------------------------- the grammar
g = bd._plan_grammar(4)
check("the grammar names 4 frames", g.count("frame ws") + g.count("ws frame") >= 4)
check("the grammar names 3 transitions", g.split("transitions")[1].count("trans") >= 3)
check("the grammar defines every production it uses",
      all(f"\n{p} ::=" in "\n" + g for p in ("root", "frame", "trans", "string", "char", "hex",
                                             "int", "digit", "ws")))
check("frame and trans have the fields the parser reads",
      '\\"framing\\"' in g and '\\"state\\"' in g and '\\"beat\\"' in g and '\\"weight\\"' in g)
check("a two-frame board is the smallest legal grammar", "trans" in bd._plan_grammar(2))
check("the plan call is the only grammar call",
      [c["tag"] for c in CALLS if c["grammar"]] == ["plan"])

# ------------------------------------------------------------------------ beats off (a slideshow)
g_off = bd._plan_grammar(4, False)
check("beats off drops the transitions array", "transitions" not in g_off)
check("…and every production only it reached",
      not any(f"\n{r} ::=" in "\n" + g_off
              for r in ("trans", "int", "digit")))
check("…while the keyframes are still forced",
      g_off.count("frame ws") + g_off.count("ws frame") >= 4)
check("beats on is byte-for-byte the grammar it always was",
      bd._plan_grammar(4, True) == bd._plan_grammar(4))

check("an untouched node gets the mode's own prompt",
      bd._plan_system("", True) == bd.PLAN_SYSTEM
      and bd._plan_system("", False) == bd.PLAN_SYSTEM_STILLS)
check("…and so does one still holding the widget's pre-filled default",
      bd._plan_system(bd.PLAN_SYSTEM, False) == bd.PLAN_SYSTEM_STILLS
      and bd._plan_system(bd.PLAN_SYSTEM_STILLS, True) == bd.PLAN_SYSTEM)
check("a prompt somebody typed wins in both modes",
      bd._plan_system("mine", True) == "mine" and bd._plan_system("mine", False) == "mine")
check("the stills prompt asks for the frames array alone",
      "no transitions" in bd.PLAN_SYSTEM_STILLS and "CUT IS FREE" in bd.PLAN_SYSTEM_STILLS)
check("…and drops the continuous-take rule the video prompt needs",
      "ONE CONTINUOUS TAKE" in bd.PLAN_SYSTEM and "CONTINUOUS TAKE" not in bd.PLAN_SYSTEM_STILLS)
check("both prompts still describe the same three keyframe fields",
      all(f'"{k}"' in bd.PLAN_SYSTEM_STILLS for k in ("framing", "present", "state")))

f, t, notes = bd.parse_plan(plan_json(4, with_trans=False), 4, with_transitions=False)
check("a frames-only plan parses clean", len(f) == 4 and len(t) == 3 and not notes)
check("…and still yields one blank, evenly weighted shot per gap",
      all(x == {"beat": "", "weight": 1.0} for x in t), t)

PLAN["n"] = 4
board, prompts, beats, durs, style, report = run(write_beats=False)
plan_call = [c for c in CALLS if c["tag"] == "plan"][0]
check("the planner is told there is no video between the stills",
      "NO transitions" in plan_call["prompt"])
check("…and is given the slideshow system prompt", plan_call["system"] == bd.PLAN_SYSTEM_STILLS)
check("…and is given a grammar that cannot emit any",
      "transitions" not in plan_call["grammar"])
check("the beats output is empty, so Morpheus Storyboard would plan its own", beats == "", repr(beats))
check("the board still has a shot per gap", len(board["shots"]) == 3)
check("…blank and evenly weighted",
      all(s["beat"] == "" and s["weight"] == 1.0 for s in board["shots"]))
check("…at legal lengths, so it is still a chain Morpheus can take",
      all(s["frames"] in T.legal_frames() for s in board["shots"]))
check("the keyframes are still written, one call each",
      len([c for c in CALLS if c["tag"].startswith("frame ")]) == 4)
check("the frames still carry the bible and the cast",
      all("grainy 35mm anamorphic" in f and "shaved head" in f
          for f in prompts.split(bd.PROMPT_SEP)))
check("the report says the beats are off", "beats: off" in report, report)
check("…and that nothing weighted the lengths", "lengths: even" in report, report)
check("a plan with no transitions is not reported as a miscount",
      "transitions, not" not in report, report)

run(write_beats=True)
keys_on = list(SAVED)
run(write_beats=False)
keys_off = list(SAVED)
check("toggling beats re-rolls the plan", keys_off[1] != keys_on[1])
check("…but not the bible, and not the frames it did not change",
      keys_off[0] == keys_on[0] and keys_off[2:] == keys_on[2:])

run(write_beats=True)
check("beats on still gets the continuous-take prompt",
      [c for c in CALLS if c["tag"] == "plan"][0]["system"] == bd.PLAN_SYSTEM)

# ------------------------------------------------------------------------------- plan robustness
f, t, notes = bd.parse_plan(plan_json(4), 4)
check("a good plan parses clean", len(f) == 4 and len(t) == 3 and not notes)
f, t, notes = bd.parse_plan("not json at all", 4)
check("garbage still yields a full board", len(f) == 4 and len(t) == 3)
check("…and says so", any("did not parse" in n for n in notes))
f, t, notes = bd.parse_plan(plan_json(2), 4)
check("a short plan is padded", len(f) == 4 and f[3]["state"] == "")
check("…and the miscount is reported", any("2 keyframes, not 4" in n for n in notes))
f, t, _ = bd.parse_plan(json.dumps({"frames": [], "transitions": [{"beat": "b", "weight": 0}]}), 3)
check("a zero weight becomes a usable one", t[0]["weight"] == 1.0)
check("prose-wrapped JSON is recovered",
      len(bd.parse_plan("here you go:\n" + plan_json(3) + "\nhope that helps", 3)[0]) == 3)

# --------------------------------------------------------------------------------------- the clock
board = run(durations="8")[0]
check("a typed duration wins over the planner",
      [s["frames"] for s in board["shots"]] == [192, 192, 192])
board, _, _, durs, _, report = run(durations="5.17, 8")
check("the last typed value repeats", [s["frames"] for s in board["shots"]] == [124, 192, 192])
check("report says the lengths were typed", "lengths: typed" in report)

PLAN["weights"] = [1, 5, 1]
board, _, _, _, _, report = run(target_length=30.0)
shots = [s["frames"] for s in board["shots"]]
check("planner weights shape the shots", shots[1] == max(shots) and shots[0] == shots[2], shots)
check("weighted shots are all legal lengths", all(s in T.legal_frames() for s in shots))
check("the weighted total hits the target", abs(T.seconds_for(sum(shots)) - 30.0) <= 0.36)
check("report shows the weights", "weighted by the planner (1, 5, 1)" in report, report)
check("the board keeps each shot's weight", [s["weight"] for s in board["shots"]] == [1, 5, 1])
PLAN["weights"] = None

def catches(fn):
    try:
        fn()
        return None
    except (ValueError, RuntimeError) as e:
        return str(e)


msg = catches(lambda: run(count=4, target_length=12.0))
check("an impossible target is refused", msg is not None and "impossible" in msg, msg)
check("…before a single token is generated", CALLS == [])

# ------------------------------------------------------------------------------------- overrides
edited = bd.PROMPT_SEP.join(["MY OWN FRAME ONE", "", "", ""])
board, prompts, _, _, _, _ = run(prompts_override=edited)
check("an overridden frame is used verbatim", prompts.split(bd.PROMPT_SEP)[0] == "MY OWN FRAME ONE")
check("…and costs no LLM call", "frame 1/4" not in [c["tag"] for c in CALLS])
check("…while the others are still written", "frame 2/4" in [c["tag"] for c in CALLS])
check("an empty override is ignored", bd.split_override("  \n ", 3) is None)
check("a short override is padded", bd.split_override("one", 3) == ["one", "", ""])
check("--- splits the frames", bd.split_override("a\n---\nb", 2) == ["a", "b"])

# ------------------------------------------------------------------- continuity and causal keys
run()
frame_calls = [c for c in CALLS if c["tag"].startswith("frame ")]
check("frame 1 has no previous frame", "THE FRAME BEFORE" not in frame_calls[0]["prompt"])
check("frame 2 is shown frame 1's prompt", "body of frame 1/4" in frame_calls[1]["prompt"])
check("every frame call carries the bible", all("STYLE BIBLE" in c["prompt"] for c in frame_calls))
check("every frame call carries its own state",
      all(f"state {i + 1}" in c["prompt"] for i, c in enumerate(frame_calls)))

run()
keys_a = list(SAVED)
PLAN["n"] = 4
first_a = keys_a[2]      # bible, plan, frame 1, frame 2 …
run()
check("the same input gives the same keys", list(SAVED)[2] == first_a)

_orig = fake_ask


def ask_changed(cfg, system, user_prompt, unload_comfy, tag, emit=None, grammar=""):
    if tag == "plan":
        CALLS.append({"tag": tag, "system": system, "prompt": user_prompt, "grammar": grammar,
                      "unload_comfy": unload_comfy})
        d = json.loads(plan_json(4))
        d["frames"][1]["state"] = "a different second state"
        return json.dumps(d), None
    return _orig(cfg, system, user_prompt, unload_comfy, tag, emit, grammar)


bd._ask = ask_changed
run()
keys_b = list(SAVED)
bd._ask = _orig
check("editing frame 2 leaves frame 1's key alone", keys_b[2] == keys_a[2])
check("…and re-rolls frames 2 onwards", keys_b[3:] != keys_a[3:] and len(keys_b) == len(keys_a))

# --------------------------------------------------------------------------------- housekeeping
run()
check("the LLM is left loaded by default", SHUTDOWNS == [])
run(unload_after_run="unload after run")
check("…and freed when the node says so", SHUTDOWNS == [1])

CFG2 = dict(CFG, unload_comfy_models=True)
run(config=CFG2)
check("comfy is unloaded once, before the first call",
      [c["unload_comfy"] for c in CALLS] == [True] + [False] * 5)


def boom(*a, **k):
    return "", "[ERROR] model file not found"


bd._ask = boom
msg = catches(run)
bd._ask = _orig
check("an LLM error stops the run with a readable message",
      msg is not None and "style bible" in msg and "model file not found" in msg, msg)

# ------------------------------------------------------------- what _ask actually asks the worker
BUILT = {}


def fake_build(cfg, user_prompt, image=None, history=None, system_override=None,
               grammar_override=None):
    BUILT["cfg"] = dict(cfg)
    BUILT["grammar_override"] = grammar_override
    return None, {"req": {}, "load_sig": "", "max_tokens": 900, "directive": "", "strip_think": True,
                  "answer_marker": "", "help": ""}


def fake_gen(req, *a, **kw):
    BUILT["req"] = dict(req)
    BUILT["token_cb"] = kw.get("token_cb")
    return ("answer", "", "stop", "", "", 3, 0.1)


bd.build_llm_request, bd._generate_and_format = fake_build, fake_gen
CFG_CTX = dict(CFG, context="THE BAND: two singers, described at length.")

REAL_ASK(CFG_CTX, "sys", "user", False, "plan", emit=lambda p: None,
         grammar="root ::= \"x\"")
check("the config's context is KEPT — it is how the cast arrives",
      BUILT["cfg"].get("context") == CFG_CTX["context"], BUILT["cfg"].get("context"))
check("a grammar call still streams — the worker's use_stream is unconditional",
      BUILT["req"].get("stream_text") is True)
check("…with a token callback attached", callable(BUILT["token_cb"]))
check("the grammar is passed as an override", BUILT["grammar_override"] == 'root ::= "x"')

REAL_ASK(CFG_CTX, "sys", "user", False, "frame 1/2", emit=lambda p: None)
check("a text call streams too", BUILT["req"].get("stream_text") is True)
check("no grammar means no override", BUILT["grammar_override"] is None)

REAL_ASK(CFG_CTX, "sys", "user", False, "frame 1/2", emit=None)
check("with no log node there is nothing to stream to", BUILT["token_cb"] is None)
check("…and the request says so", "stream_text" not in BUILT["req"])
check("output_format is forced to text so the grammar override wins",
      BUILT["cfg"]["output_format"] == "text" and BUILT["cfg"]["grammar"] == "")

check.done()
