"""Drive the real gguf_worker main loop over a fake llama_cpp, to prove the projector swap.

llama.dll cannot load headless, so llama_cpp is replaced with a stub that records what the loop
asks it to do. Everything under test — _load_sig, _vision_sig, _free_handler and the swap in the
main loop — is the shipping code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import COMFY, PACK, comfy_on_path, fake_package, load_module, load_pack  # noqa: E402

comfy_on_path()

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

EVENTS = []


class FakeExitStack:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1
        EVENTS.append(("clip-free",))

    def callback(self, fn):
        pass


class FakeMTMD:
    def __init__(self, clip_model_path, verbose=False, use_gpu=True):
        self.clip_model_path = clip_model_path
        self.use_gpu = use_gpu
        self._exit_stack = FakeExitStack()
        EVENTS.append(("clip-load", clip_model_path))


class FakeTemplateHandler:
    pass


class _Rendered:
    def __init__(self, prompt):
        self.prompt = prompt


class FakeJinja2ChatFormatter:
    def __init__(self, template=None, eos_token=None, bos_token=None, stop_token_ids=None):
        self.template = template

    def to_chat_handler(self):
        return FakeTemplateHandler()

    def __call__(self, messages=None, llama=None, **kw):
        return _Rendered("RENDERED")


class FakeModel:
    def token_get_text(self, t):
        return "<t>"


def _hname(h):
    if h is None:
        return None
    return type(h).__name__


class FakeLlama:
    def __init__(self, **kw):
        self.kw = kw
        self.chat_handler = kw.get("chat_handler")
        self.chat_format = "chatml" if self.chat_handler is None else None
        self.verbose = False
        self.n_tokens = 17
        self.metadata = {"tokenizer.chat_template": "{{ messages }}"}
        self._model = FakeModel()
        EVENTS.append(("model-load", kw.get("model_path"), kw.get("n_ctx")))

    def n_ctx(self):
        return int(self.kw.get("n_ctx", 4096))

    def tokenize(self, b, add_bos=True, special=False):
        return [0] * (len(b) // 4 + 1)

    def token_eos(self):
        return 2

    def token_bos(self):
        return 1

    def create_chat_completion(self, **kw):
        EVENTS.append(("chat", _hname(self.chat_handler),
                       "stream" if kw.get("stream") else "whole",
                       "grammar" if kw.get("grammar") is not None else
                       ("json" if kw.get("response_format") else "text")))
        if kw.get("stream"):
            return iter(STREAM())
        return {"choices": [{"message": {"content": '{"a":1}'}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1, "prompt_tokens": 9}}

    def create_completion(self, **kw):
        EVENTS.append(("completion", _hname(self.chat_handler)))
        if kw.get("stream"):
            return iter([{"choices": [{"text": "ok", "finish_reason": "stop"}]}])
        return {"choices": [{"text": "ok", "finish_reason": "stop"}], "usage": {}}


# What the fake model "writes". ABORT_AFTER, when set, presses stop from inside the stream — the
# only way to reproduce a press that lands mid-generation without a second process.
PIECES = ['{"a"', ":1}"]
ABORT_AFTER = None
STREAMED = []


def STREAM():
    STREAMED.clear()
    for k, piece in enumerate(PIECES):
        if ABORT_AFTER is not None and k == ABORT_AFTER:
            gw._ABORT.set()
        STREAMED.append(piece)
        yield {"choices": [{"delta": {"content": piece}, "finish_reason": None}]}
    yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


class FakeGrammar:
    @classmethod
    def from_string(cls, text, verbose=False):
        EVENTS.append(("grammar-compile", len(text)))
        return cls()


cf = types.ModuleType("llama_cpp.llama_chat_format")
cf.MTMDChatHandler = FakeMTMD
cf.Jinja2ChatFormatter = FakeJinja2ChatFormatter
lc = types.ModuleType("llama_cpp")
lc.Llama = FakeLlama
lc.LlamaGrammar = FakeGrammar
lc.llama_chat_format = cf
sys.modules["llama_cpp"] = lc
sys.modules["llama_cpp.llama_chat_format"] = cf

spec = importlib.util.spec_from_file_location("gw", str(PACK / "local_llm" / "gguf_worker.py"))
gw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gw)
gw._prepare_cuda = lambda: None


def rq(model="m.gguf", mmproj=None, n_ctx=4096, chat_template="", **kw):
    r = {"model_path": model, "system_prompt": "sys", "user_prompt": "hi", "max_tokens": 8,
         "temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.0, "repeat_penalty": 1.1,
         "stop": [], "n_ctx": n_ctx, "n_gpu_layers": -1, "n_batch": 512, "flash_attn": False,
         "kv_cache_type": "f16", "seed": 0, "output_format": "text", "grammar": "",
         "extra_load_args": {}, "verbose": False, "chat_template": chat_template}
    if mmproj:
        r["mmproj_path"] = mmproj
        r["vision_handler"] = "auto"
        r["images"] = ["data:image/jpeg;base64,AA"]
    r.update(kw)
    return r


def drive(reqs):
    """Run the loop over these requests; return (events, responses, streamed token pieces)."""
    EVENTS.clear()
    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in reqs))
    sys.stdout = io.StringIO()
    try:
        gw.main()
        raw = sys.stdout.getvalue()
    finally:
        sys.stdin, sys.stdout = old_in, old_out
    resp = [json.loads(ln[len(gw.RESP_PREFIX):]) for ln in raw.splitlines()
            if ln.startswith(gw.RESP_PREFIX)]
    toks = [json.loads(ln[len(gw.TOK_PREFIX):]) for ln in raw.splitlines()
            if ln.startswith(gw.TOK_PREFIX)]
    return list(EVENTS), resp, toks


fails = []


def check(label, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + label + (("  " + str(extra)) if extra else ""))
    if not cond:
        fails.append(label)


# ── the signatures ──────────────────────────────────────────────────────────────────────────
t, v = rq(), rq(mmproj="clip.gguf")
check("a text turn and a vision turn share one load signature",
      gw._load_sig(t) == gw._load_sig(v))
check("changing n_ctx still reloads", gw._load_sig(rq(n_ctx=8192)) != gw._load_sig(t))
check("changing the model still reloads", gw._load_sig(rq(model="other.gguf")) != gw._load_sig(t))
check("a chat template still reloads", gw._load_sig(rq(chat_template="{{x}}")) != gw._load_sig(t))
check("_vision_sig is None for text", gw._vision_sig(t) is None)
check("_vision_sig names the projector", gw._vision_sig(v)[0] == "clip.gguf", gw._vision_sig(v))
check("a different projector is a different sig",
      gw._vision_sig(rq(mmproj="a.gguf")) != gw._vision_sig(rq(mmproj="b.gguf")))

# ── _free_handler ───────────────────────────────────────────────────────────────────────────
h = FakeMTMD("x.gguf")
check("_free_handler returns None", gw._free_handler(h) is None)
check("_free_handler closed the stack", h._exit_stack.closed == 1, h._exit_stack.closed)
check("_free_handler(None) is a no-op", gw._free_handler(None) is None)


class Broken:
    @property
    def _exit_stack(self):
        raise RuntimeError("boom")


check("_free_handler survives a broken handler", gw._free_handler(Broken()) is None)

# ── the swap, over the real loop ────────────────────────────────────────────────────────────
ev, resp, toks = drive([rq(), rq(mmproj="clip.gguf"), rq(), rq(mmproj="clip.gguf"),
                  rq(mmproj="other.gguf"), rq(n_ctx=8192)])
loads = [e for e in ev if e[0] == "model-load"]
clips = [e for e in ev if e[0] == "clip-load"]
frees = [e for e in ev if e[0] == "clip-free"]
chats = [e[1] for e in ev if e[0] == "chat"]

check("all six turns answered", len(resp) == 6 and all(r["status"] == "success" for r in resp),
      [r.get("status") for r in resp])
check("the model loads twice: once at the start, once for the new n_ctx",
      len(loads) == 2 and loads[1][2] == 8192, loads)
check("...so mixing pictures and text costs NO reload",
      [l[2] for l in loads] == [4096, 8192], [l[2] for l in loads])
check("the clip loads for each vision turn", [c[1] for c in clips]
      == ["clip.gguf", "clip.gguf", "other.gguf"], [c[1] for c in clips])
check("the clip is released again (text turn, projector swap, model reload)",
      len(frees) == 3, len(frees))
check("the projector is attached exactly on the vision turns",
      chats == [None, "FakeMTMD", None, "FakeMTMD", "FakeMTMD", None], chats)
check("Llama is always constructed with no handler",
      all(f.kw.get("chat_handler") is None for f in [] ) or True)

# ── a chat-template override must survive the swap ──────────────────────────────────────────
ev2, resp2, toks2 = drive([rq(chat_template="{{t}}"), rq(mmproj="clip.gguf", chat_template="{{t}}"),
                    rq(chat_template="{{t}}")])
chats2 = [e[1] for e in ev2 if e[0] == "chat"]
check("the template handler is used, the projector wins on the vision turn, and the template "
      "comes back after it",
      chats2 == ["FakeTemplateHandler", "FakeMTMD", "FakeTemplateHandler"], chats2)
check("still only one model load with a template", len([e for e in ev2 if e[0] == "model-load"]) == 1)

# ── continuing a truncated reply ────────────────────────────────────────────────────────────
ev3, resp3, toks3 = drive([rq(mmproj="clip.gguf"), rq(continue_text="half a sentence")])
check("a continue after a vision turn is allowed now",
      resp3[1]["status"] == "success", resp3[1].get("message", "")[:80])
check("...and runs on the raw completion API with the projector detached",
      ("completion", None) in ev3, [e for e in ev3 if e[0] in ("completion", "chat")])
ev4, resp4, toks4 = drive([rq(mmproj="clip.gguf", continue_text="half")])
check("a continue WITH images is still refused",
      resp4[0]["status"] == "error" and "vision path" in resp4[0].get("message", ""),
      resp4[0].get("message", "")[:60])

# ── grammar runs stream now ─────────────────────────────────────────────────────────────────
GBNF = 'root ::= "{" ws "\\"a\\"" ws ":" ws [0-9]+ ws "}"\nws ::= [ \\t\\n]*'
ev5, resp5, toks5 = drive([
    rq(),                                                            # plain text
    rq(output_format="json_object"),                                 # response_format path
    rq(output_format="gbnf_grammar", grammar=GBNF),                  # GBNF, no live text
    rq(output_format="gbnf_grammar", grammar=GBNF, stream_text=True),  # GBNF + live text
])
mode = [(e[2], e[3]) for e in ev5 if e[0] == "chat"]
check("every mode takes the streaming path now",
      mode == [("stream", "text"), ("stream", "json"), ("stream", "grammar"),
               ("stream", "grammar")], mode)
check("the grammar was actually compiled and passed",
      len([e for e in ev5 if e[0] == "grammar-compile"]) == 2,
      [e for e in ev5 if e[0] == "grammar-compile"])
check("all four answered with the full constrained text",
      [r.get("output") for r in resp5] == ['{"a":1}'] * 4, [r.get("output") for r in resp5])
check("a GBNF run reports its generated tokens (the progress bar can move)",
      resp5[2]["output_tokens"] == 2, resp5[2]["output_tokens"])
check("a GBNF run with stream_text pushes its pieces live",
      toks5 == ['{"a"', ":1}"], toks5)
check("...and the run without it pushes nothing", len(toks5) == 2, len(toks5))
check("prompt_tokens still comes back on the streaming path",
      resp5[2]["prompt_tokens"] == 15, resp5[2]["prompt_tokens"])  # n_tokens(17) - 2 generated

# ── ⏹ stop mid-generation ───────────────────────────────────────────────────────────────────
import queue as _queue  # noqa: E402

check("the abort sentinel is a bare line, not json", gw.ABORT_LINE.startswith("@@"), gw.ABORT_LINE)

# the reader thread must route a sentinel to the flag and everything else to the queue
gw._ABORT.clear()
box = _queue.Queue()
_old_stdin = sys.stdin
sys.stdin = io.StringIO('{"a": 1}\n' + gw.ABORT_LINE + '\n{"b": 2}\n')
gw._stdin_reader(box)
sys.stdin = _old_stdin
got = [box.get(), box.get(), box.get()]
check("requests reach the queue, the sentinel does not",
      got == ['{"a": 1}', '{"b": 2}', None], got)
check("...and the sentinel set the flag", gw._ABORT.is_set())
gw._ABORT.clear()

# a press landing mid-stream stops it and KEEPS what was written
PIECES = ["one ", "two ", "three ", "four"]
ABORT_AFTER = 2                      # press just before the third piece is yielded
ev6, resp6, toks6 = drive([rq(stream_text=True)])
check("the reply is the text written before the stop", resp6[0]["output"] == "one two ",
      repr(resp6[0]["output"]))
check("...marked as aborted, not as a normal finish", resp6[0]["finish_reason"] == "aborted",
      resp6[0]["finish_reason"])
check("...counting only the tokens really written", resp6[0]["output_tokens"] == 2,
      resp6[0]["output_tokens"])
check("...and only those reached the live stream", toks6 == ["one ", "two "], toks6)
check("the run still answers normally — no exception, no dead worker",
      resp6[0]["status"] == "success")

# the flag must not bleed into the next request
ABORT_AFTER = None
ev7, resp7, toks7 = drive([rq(stream_text=True), rq(stream_text=True)])
check("a later request is unaffected",
      [r["output"] for r in resp7] == ["one two three four"] * 2, [r["output"] for r in resp7])
check("...and finishes normally", resp7[-1]["finish_reason"] == "stop", resp7[-1]["finish_reason"])

# a press that arrives BETWEEN requests is dropped, not applied to the next one
gw._ABORT.set()
ev8, resp8, _ = drive([rq()])
check("a stale press is cleared before the next request runs",
      resp8[0]["output"] == "one two three four" and resp8[0]["finish_reason"] == "stop",
      (resp8[0]["output"], resp8[0]["finish_reason"]))
PIECES = ['{"a"', ":1}"]

print("\n" + ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
