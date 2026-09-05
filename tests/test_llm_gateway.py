"""The LLM gateway end to end, against a stand-in backend that is a real subprocess.

Nothing here is mocked at the HTTP boundary: the gateway listens on a real port, spawns
`tests/fake_llm_backend.py` the same way it spawns `llama-server` (same argv shape, same `/health`
readiness probe), and every assertion goes through a real client socket. What is faked is only the
model — the stand-in validates `dry_sequence_breakers` the way llama.cpp's `server-schema.cpp` does
and can be told to reject any other field, which is the whole point of the exercise.

What it pins:

* connecting with **nothing loaded** — the model list and the health probe SillyTavern's Connect
  button uses are answered by the gateway, and no model is started for them;
* the first POST loading the model and then being forwarded, which is the trick the whole design
  exists for (SillyTavern must never see a closed port);
* the parameter repair — SillyTavern's `dry_sequence_breakers` JSON *string* reaching the model as
  a real array, and an unknown field named in a 400 being dropped and retried once;
* streaming passing through chunk by chunk;
* every way the VRAM goes back: the explicit unload, the queue hook that spares prompts containing
  the pack's own two nodes, and a reload on the next request.
"""
import http.client
import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, TESTS, fake_package, load_module  # noqa: E402

# `python -m fake_llm_backend` is how the gateway's own argv shape can launch our stand-in.
os.environ["PYTHONPATH"] = str(TESTS) + os.pathsep + os.environ.get("PYTHONPATH", "")

fake_package("kn", "llm_server")
compat = load_module("kn.llm_server.compat", "llm_server/compat.py")
gw = load_module("kn.llm_server.gateway", "llm_server/gateway.py")

check = Checker()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


PORT = free_port()


def call(method, path, body=None, timeout=60, port=None):
    conn = http.client.HTTPConnection("127.0.0.1", port or PORT, timeout=timeout)
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if payload else {}
    conn.request(method, path, body=payload, headers=headers)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    return r.status, raw


def jcall(*a, **kw):
    status, raw = call(*a, **kw)
    try:
        return status, json.loads(raw.decode("utf-8"))
    except Exception:
        return status, {"_raw": raw.decode("utf-8", "replace")}


def last_body():
    """What actually reached the model, straight from the stand-in's own record."""
    return jcall("GET", "/last")[1].get("body") or {}


def configure(**over):
    cfg = dict(flavour="llama-server", binary=sys.executable, model="fake_llm_backend",
               n_ctx=4096, n_gpu_layers=-1, extra=("--reject", "bogus_field"),
               backend_port=0, autoload=True, free_on_prompt=True, idle_minutes=0,
               unload_comfy=False, fix_params=True, auto_retry=True, drop_fields=(),
               startup_timeout=90, request_timeout=90)
    cfg.update(over)
    gw.configure(**cfg)


# --------------------------------------------------------------- the dialect fixes, on their own
ST_BREAKERS = json.dumps(["\n", ":", '"', "*"])   # byte-for-byte what SillyTavern stores
fixed, notes = compat.normalize({"dry_sequence_breakers": ST_BREAKERS, "temperature": 0.7})
check("ST's JSON-string breakers become a real array",
      fixed["dry_sequence_breakers"] == ["\n", ":", '"', "*"], fixed["dry_sequence_breakers"])
check("the rest of the body is untouched", fixed["temperature"] == 0.7 and len(notes) == 1)
check("an already-correct array is left alone",
      compat.normalize({"dry_sequence_breakers": ["\n"]}) == ({"dry_sequence_breakers": ["\n"]}, []))
check("an empty array is dropped, not passed on (llama.cpp refuses it)",
      "dry_sequence_breakers" not in compat.normalize({"dry_sequence_breakers": []})[0])
check("a 400 names the field it choked on",
      compat.field_from_error("Field 'dry_sequence_breakers': Error: must be a non-empty array")
      == "dry_sequence_breakers")
check("a 400 that names nothing yields nothing", compat.field_from_error("out of memory") == "")

# --------------------------------------------------------------- connecting with nothing loaded
configure()
ok, err = gw.serve("127.0.0.1", PORT)
check("the gateway listens", ok, err)
check("no model was started just by listening", not gw.running())

status, models = jcall("GET", "/v1/models")
check("GET /v1/models answers with the model unloaded", status == 200, status)
check("...and names the configured model",
      [m["id"] for m in models.get("data", [])] == ["fake_llm_backend"], models)
check("...without loading anything", not gw.running())

status, health = jcall("GET", "/health")
check("GET /health answers with the model unloaded", status == 200 and health["status"] == "ok")
check("...still without loading anything", not gw.running())

# --------------------------------------------------------------- the first message loads the model
t0 = time.time()
status, answer = jcall("POST", "/v1/chat/completions", {
    "model": "whatever", "messages": [{"role": "user", "content": "hi"}],
    "dry_sequence_breakers": ST_BREAKERS, "temperature": 0.8,
})
check("a POST with nothing loaded loads the model and is answered", status == 200, answer)
check("...the model really is up now", gw.running())
check("...and the answer is the model's own",
      answer.get("choices", [{}])[0].get("message", {}).get("content") == "ok", answer)

check("the model list keeps the same id once the model IS up (the client remembers it)",
      [m["id"] for m in jcall("GET", "/v1/models")[1].get("data", [])] == ["fake_llm_backend"])

sent = last_body()
check("the breakers reached the model as an array, not a string",
      sent.get("dry_sequence_breakers") == ["\n", ":", '"', "*"], sent.get("dry_sequence_breakers"))
check("everything else reached it unchanged", sent.get("temperature") == 0.8)

# --------------------------------------------------------------- a field llama.cpp refuses
status, answer = jcall("POST", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "hi"}], "bogus_field": 1, "temperature": 0.5,
})
check("a rejected field is dropped and the request retried, not failed", status == 200, answer)
check("...the retry reached the model without it", "bogus_field" not in last_body())
check("...and it is remembered for later requests", "bogus_field" in gw.STATE.learned_drops)
check("...the rest of that request survived the retry", last_body().get("temperature") == 0.5)

jcall("POST", "/v1/chat/completions", {"messages": [], "bogus_field": 2, "top_p": 0.9})
check("a later request is stripped up front (no second 400 needed)",
      "bogus_field" not in last_body() and last_body().get("top_p") == 0.9)

# --------------------------------------------------------------- streaming
status, raw = call("POST", "/v1/chat/completions",
                   {"messages": [{"role": "user", "content": "hi"}], "stream": True})
text = raw.decode("utf-8", "replace")
check("a streamed reply comes through", status == 200, status)
check("...with every token, in order",
      [p for p in ("one", "two", "three") if p in text] == ["one", "two", "three"], text[:120])
check("...and its terminator", "[DONE]" in text)

# --------------------------------------------------------------- giving the VRAM back
check("the explicit unload kills the model", gw.stop_model("test") and not gw.running())
status, _ = jcall("GET", "/v1/models")
check("the gateway is still listening with the model gone", status == 200)

status, answer = jcall("POST", "/v1/chat/completions",
                       {"messages": [{"role": "user", "content": "again"}]})
check("the next message loads it back", status == 200 and gw.running(), answer)

gw._on_prompt({"prompt": {"1": {"class_type": "KSampler"}, "2": {"class_type": "VAEDecode"}}})
check("a queued image prompt takes the VRAM back", not gw.running())

jcall("POST", "/v1/chat/completions", {"messages": []})
gw._on_prompt({"prompt": {"1": {"class_type": gw.SERVER_NODE_ID}}})
check("a prompt that IS the server node is left alone", gw.running())
gw._on_prompt({"prompt": {"1": {"class_type": gw.CONTROL_NODE_ID,
                                "inputs": {"action": gw.CONTROL_LOAD}}}})
check("...so is one whose control node asks for the model", gw.running())
gw._on_prompt({"prompt": {"1": {"class_type": gw.CONTROL_NODE_ID,
                                "inputs": {"action": gw.CONTROL_FREE}},
                          "2": {"class_type": "KSampler"}}})
check("but a control node set to FREE does not disarm the hook — it may sit after the sampler",
      not gw.running())

configure(free_on_prompt=False)
jcall("POST", "/v1/chat/completions", {"messages": []})
gw._on_prompt({"prompt": {"1": {"class_type": "KSampler"}}})
check("free_on_prompt off means a queued prompt leaves the model alone", gw.running())

# --------------------------------------------------------------- the switches
configure(drop_fields=("mirostat",))
jcall("POST", "/v1/chat/completions", {"messages": [], "mirostat": 2, "top_k": 40})
check("drop_fields strips a field by name",
      "mirostat" not in last_body() and last_body().get("top_k") == 40, last_body())

configure(fix_params=False, auto_retry=False)
status, answer = jcall("POST", "/v1/chat/completions",
                       {"messages": [], "dry_sequence_breakers": ST_BREAKERS})
check("with fix_params off the model's own 400 is passed straight back", status == 400, status)
check("...with llama.cpp's wording intact",
      "dry_sequence_breakers" in json.dumps(answer), answer)

gw.stop_model("test")
configure(autoload=False)
status, answer = jcall("POST", "/v1/chat/completions", {"messages": []})
check("autoload off refuses instead of loading", status == 503 and not gw.running(), status)
check("...and says why", "autoload" in json.dumps(answer), answer)

# --------------------------------------------- the other flavour, which has no /health at all
# Only the command line is checked live: the stand-in is launched as `python -m <module>`, which
# koboldcpp's own `--model` spelling cannot express. The probe path it needs is exercised by the
# ready_path override just below, which goes through the same `_probe` call.
configure(flavour="koboldcpp", autoload=True, fix_params=True, auto_retry=True, drop_fields=())
argv = gw._argv(gw.STATE.cfg, 1234)
check("koboldcpp gets its own flag names",
      argv[1:2] == ["--model"] and "--contextsize" in argv and "--gpulayers" in argv, argv)
check("...and is waited for on /v1/models, since it has no /health",
      gw.FLAVOURS["koboldcpp"]["ready"] == "/v1/models"
      and gw.FLAVOURS["llama-server"]["ready"] == "/health")

configure(ready_path="/props")
status, _ = jcall("POST", "/v1/chat/completions", {"messages": []})
check("ready_path overrides the probe", status == 200 and gw.running(), status)

# --------------------------------------------------------------- shutdown
gw.shutdown("end of suite")
check("shutdown stops the model", not gw.running())
try:
    call("GET", "/v1/models", timeout=2)
    listening = True
except Exception:
    listening = False
check("shutdown closes the port", not listening)

check.done()
