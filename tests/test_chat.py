"""Exercise LocalLLMChatGGUF.run() for hold_open + 6 personas, without ComfyUI."""
import importlib.util
import json
import types
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import COMFY, PACK, comfy_on_path, fake_package, load_module, load_pack  # noqa: E402

comfy_on_path()

pkg = types.ModuleType("kn")
pkg.__path__ = [str(PACK / "local_llm")]
sys.modules["kn"] = pkg


def load(name):
    spec = importlib.util.spec_from_file_location(
        "kn." + name, str(PACK / "local_llm" / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kn." + name] = mod
    spec.loader.exec_module(mod)
    return mod


load("llm_node")
load("settings_node")
atts = load("attachments")
chat = load("chat_node")

calls = []
SENT = {}


def fake_build(cfg, prompt, image=None, history=None, system_override=None, grammar_override=None):
    calls.append({"cfg": cfg, "prompt": prompt, "system_override": system_override,
                  "history": history, "image": image})
    SENT.clear()
    SENT.update({"image": image, "prompt": prompt, "history": history})
    return (None, {"req": {}, "load_sig": (), "max_tokens": 8, "unload_comfy": False,
                   "unload_llm": False, "directive": "", "answer_marker": "", "help": "H"})


def fake_gen(*a, **kw):
    if isinstance(kw.get("stats"), dict):
        kw["stats"].update({"n_ctx": 4096, "context_used": 10})
    return ("REPLY", "", "stop", 0, 0, 3, 0.5, "H", 0, 3)


shut = []
chat.build_llm_request = fake_build
chat._generate_and_format = fake_gen
chat._shutdown_worker = lambda: shut.append(1)
chat.ExecutionBlocker = None          # so the gate is a plain string we can assert on

P = [{"model": "m%d" % i, "system_prompt": "s%d" % i} for i in range(1, 7)]
HIST = [{"role": "user", "content": "hi"},
        {"role": "assistant", "content": "LAST", "persona": "A"}]


def state(**kw):
    st = {"v": 1, "user": "", "history": HIST, "nonce": 1, "approved": False,
          "persona": 1, "turn": {"mode": "turn", "persona": "A", "keep": {}, "private": []}}
    st.update(kw)
    return json.dumps(st)


node = chat.LocalLLMChatGGUF()
fails = []


def check(label, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + "  " + label + (("  " + str(extra)) if extra else ""))
    if not cond:
        fails.append(label)


# ── declared inputs ─────────────────────────────────────────────────────────────────────────
opt = chat.LocalLLMChatGGUF.INPUT_TYPES()["optional"]
check("persona_5/6 declared", "persona_5" in opt and "persona_6" in opt)
check("chat_state is still the last optional (widget order unchanged)",
      list(opt)[-1] == "chat_state", list(opt)[-1])
check("no stray hold_open left behind", "hold_open" not in opt)

# ── the two gate paths are untouched ────────────────────────────────────────────────────────
calls.clear(); shut.clear()
out = node.run(P[0], chat_state=state())
check("normal turn generates", len(calls) == 1)
check("normal turn blocks downstream", isinstance(out, dict) and "ui" in out)

calls.clear(); shut.clear()
out = node.run(P[0], chat_state=state(approved=True))
check("approve emits the last reply, no generation",
      isinstance(out, tuple) and out[0] == "LAST" and not calls)
check("approve unloads (unload_on_approve default on)", shut == [1], shut)

calls.clear(); shut.clear()
node.run(P[0], chat_state=state(approved=True), unload_on_approve=False)
check("approve + keep loaded: no unload", not shut)

# ── six personas ────────────────────────────────────────────────────────────────────────────
for n in (5, 6):
    calls.clear()
    kw = {"persona_5": P[4], "persona_6": P[5]}
    node.run(P[0], chat_state=state(persona=n), **kw)
    check("persona_%d picked as the turn's config" % n,
          calls and calls[0]["cfg"]["model"] == "m%d" % n,
          calls[0]["cfg"]["model"] if calls else "-")

calls.clear()
node.run(P[0], chat_state=state(persona=7), persona_5=P[4], persona_6=P[5])
check("out-of-range persona falls back to the first wired one",
      calls and calls[0]["cfg"]["model"] == "m1", calls[0]["cfg"]["model"] if calls else "-")

# ── persona_6 as the summariser ─────────────────────────────────────────────────────────────
calls.clear()
turn = {"mode": "fold", "persona": "A", "keep": {}, "private": [], "upto": 2, "summarizer": 6}
node.run(P[0], chat_state=state(turn=turn), persona_5=P[4], persona_6=P[5])
check("summarizer=6 uses persona_6 in its own voice",
      calls and calls[0]["cfg"]["model"] == "m6" and calls[0]["system_override"] is None,
      (calls[0]["cfg"]["model"], calls[0]["system_override"]) if calls else "-")

calls.clear()
turn["summarizer"] = 0
node.run(P[0], chat_state=state(turn=turn))
check("summarizer=0 still borrows the active model + DIGEST_SYSTEM",
      calls and calls[0]["system_override"] == chat.DIGEST_SYSTEM)

# ── old call shape (a workflow saved before this change) ────────────────────────────────────
calls.clear()
out = node.run(P[0], None, P[1], None, None)
check("positional pre-change call still works", len(calls) == 1)

# ── attachments ─────────────────────────────────────────────────────────────────────────────
import tempfile

import numpy as np
from PIL import Image

INDIR = Path(tempfile.gettempdir()) / "kb_att_test"
INDIR.mkdir(parents=True, exist_ok=True)
sub = INDIR / "kinburg_chat"
sub.mkdir(exist_ok=True)
for nm, size in (("a.png", (7, 5)), ("b.png", (4, 9))):
    Image.fromarray(np.full((size[1], size[0], 3), 128, dtype="uint8")).save(sub / nm)

fake_fp = types.ModuleType("folder_paths")
fake_fp.get_input_directory = lambda: str(INDIR)
fake_fp.get_temp_directory = lambda: str(INDIR / "temp")
fake_fp.get_output_directory = lambda: str(INDIR / "out")
sys.modules["folder_paths"] = fake_fp

A = {"name": "a.png", "subfolder": "kinburg_chat", "type": "input"}
B = {"name": "b.png", "subfolder": "kinburg_chat", "type": "input"}

# _content_of — what the model reads once the pixels are gone
check("marker for a bare attachment",
      chat._content_of({"content": "look", "att": [A]}) == "look\n[image]",
      repr(chat._content_of({"content": "look", "att": [A]})))
check("marker carries the caption",
      chat._content_of({"content": "", "att": [dict(A, caption="a red dress")]})
      == "[image: a red dress]")
check("two attachments, two markers",
      chat._content_of({"content": "", "att": [A, B]}) == "[image] [image]")
check("no attachment changes nothing", chat._content_of({"content": "hi"}) == "hi")
check("ctx:false hides an attachment from the model",
      chat._content_of({"content": "hi", "att": [dict(A, ctx=False)]}) == "hi",
      repr(chat._content_of({"content": "hi", "att": [dict(A, ctx=False)]})))
check("a hidden picture doesn't hide its neighbour",
      chat._content_of({"content": "", "att": [dict(A, ctx=False), dict(B, caption="seen")]})
      == "[image: seen]")
# a persona bubble whose only content is a hidden picture must drop out of the context entirely,
# not arrive as an empty assistant turn
check("a bubble that is only a hidden picture is empty to the model",
      chat._content_of({"content": "", "att": [dict(A, ctx=False)]}) == "")
calls.clear()
hist5 = [{"role": "assistant", "content": "", "persona": "A", "att": [dict(A, ctx=False)]},
         {"role": "user", "content": "nice"}]
node.run(P[0], chat_state=json.dumps(
    {"v": 1, "user": "?", "history": hist5, "nonce": 1, "approved": False, "persona": 1,
     "att": [], "turn": {"mode": "turn", "persona": "A", "keep": {}, "private": []}}))
check("...and is skipped when the context is built",
      [m["content"] for m in calls[0]["history"]] == ["nice"], calls[0]["history"])

# path resolution refuses to leave the base directory
paths, missing = atts.resolve_refs([A, B])
check("both files resolve", len(paths) == 2 and not missing, (len(paths), missing))
_, esc = atts.resolve_refs([{"name": "x.png", "subfolder": "../..", "type": "input"}])
check("traversal is refused", esc == ["x.png"], esc)
_, gone = atts.resolve_refs([{"name": "nope.png", "subfolder": "kinburg_chat", "type": "input"}])
check("a missing file is reported, not skipped", gone == ["nope.png"], gone)

# the tray reaches build_llm_request as a list of [1,H,W,C] tensors
calls.clear()
node.run(P[0], chat_state=state(att=[A, B]))
check("tray images are passed to the request", isinstance(SENT["image"], list)
      and len(SENT["image"]) == 2, type(SENT.get("image")).__name__)
check("each is a [1,H,W,C] tensor",
      all(tuple(t.shape)[0] == 1 and tuple(t.shape)[3] == 3 for t in SENT["image"]),
      [tuple(t.shape) for t in SENT["image"]])

# no attachment -> image stays None, so the projector is never loaded for nothing
calls.clear()
node.run(P[0], chat_state=state())
check("a text-only turn sends image=None", SENT["image"] is None, SENT["image"])

# a continuation must not carry pixels (the worker refuses vision + prefill)
calls.clear()
hist2 = [{"role": "user", "content": "hi"},
         {"role": "assistant", "content": "LAST", "persona": "A"}]
node.run(P[0], chat_state=json.dumps(
    {"v": 1, "user": "", "history": hist2, "nonce": 1, "approved": False, "persona": 1,
     "att": [A], "turn": {"mode": "continue", "persona": "A", "keep": {}, "private": []}}))
check("a 'continue' turn sends no images", SENT["image"] is None, SENT["image"])

# a broken ref stops the turn with a readable error instead of a traceback
calls.clear()
out = node.run(P[0], chat_state=state(att=[{"name": "ghost.png", "subfolder": "kinburg_chat",
                                            "type": "input"}]))
payload = json.loads(out["ui"]["kinburg_chatllm"][0]) if isinstance(out, dict) else {}
check("a vanished attachment errors cleanly", "ghost.png" in payload.get("reply", ""),
      payload.get("reply", "")[:70])
check("...and nothing was generated", not calls)

# history: an older picture arrives as its marker, not as pixels
calls.clear()
hist3 = [{"role": "user", "content": "look", "att": [dict(A, caption="red dress")]},
         {"role": "assistant", "content": "nice", "persona": "A"}]
node.run(P[0], chat_state=json.dumps(
    {"v": 1, "user": "and now?", "history": hist3, "nonce": 1, "approved": False, "persona": 1,
     "att": [], "turn": {"mode": "turn", "persona": "A", "keep": {}, "private": []}}))
hsent = calls[0]["history"]
check("the old turn is remembered as text", hsent[0]["content"] == "look\n[image: red dress]",
      repr(hsent[0]["content"]))
check("...and carries no pixels", SENT["image"] is None)

# an image-only message survives the context build (it has a marker, so it isn't 'empty')
calls.clear()
hist4 = [{"role": "user", "content": "", "att": [A]},
         {"role": "assistant", "content": "I see", "persona": "A"}]
node.run(P[0], chat_state=json.dumps(
    {"v": 1, "user": "?", "history": hist4, "nonce": 1, "approved": False, "persona": 1,
     "att": [], "turn": {"mode": "turn", "persona": "A", "keep": {}, "private": []}}))
check("an image-only message is not dropped from the context",
      calls[0]["history"][0]["content"] == "[image]", calls[0]["history"])

print("\n" + ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
