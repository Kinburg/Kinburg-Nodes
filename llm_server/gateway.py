"""One address SillyTavern can always reach, with llama.cpp servers behind it that come and go.

The problem this solves is VRAM. A chat model and an image model do not fit at once, so the model
has to leave when a picture is being made and come back afterwards — but SillyTavern has no idea
any of that is happening. Point it straight at `llama-server` and the moment ComfyUI kills that
process to free VRAM, SillyTavern's next message hits a closed port: nothing is listening, so
nothing can start the model back up.

So the listener and the models are split in two:

* the **gateway** — a small HTTP server owned by this ComfyUI process, started by the *Local LLM
  Server* node and kept up for as long as ComfyUI runs. This is the address SillyTavern is
  configured with, and it never goes away;
* the **backends** — real `llama-server` (or `koboldcpp`) subprocesses on private ports, holding
  the VRAM. Each is started on the first request that actually needs it, and killed whenever the
  VRAM is wanted elsewhere.

There are two backend roles. `chat` is the model you talk to. `embed` only exists on the
llama-server side, because `--embeddings` *restricts* a llama-server process to embedding work —
so an embedding model cannot share the chat process and gets one of its own, which the gateway
routes `/v1/embeddings` and `/rerank` to. koboldcpp takes `--embeddingsmodel` in the same process,
so there it stays a flag and this file never starts a second one.

A request arriving while its backend is down simply waits for it: the gateway loads the model,
then forwards. SillyTavern sees a slow first reply, never an error. Requests that do *not* need a
model — the model list and the health probe SillyTavern's **Connect** button uses — are answered by
the gateway itself, so connecting works with nothing loaded at all.

Freeing the VRAM happens two ways, and both are needed:

* the **LLM Server Control** node, dropped into an image workflow, so the unload is an explicit,
  ordered step of that graph;
* `free_on_prompt`, a hook on ComfyUI's prompt queue: any prompt that is not itself about the
  server unloads every backend as it is queued. That is the safety net for the workflows you
  forgot to edit.

Bodies on the way through are repaired by :mod:`compat` — see there for why SillyTavern's
`dry_sequence_breakers` makes llama.cpp reject the whole request. Command lines are built by
:mod:`options`, which owns the two backends' flag dialects.

Everything here is deliberately free of ComfyUI imports at module level: the pack is imported by a
registry scan with no `comfy` on the path, and this file has to survive that.
"""
import atexit
import http.client
import json
import os
import socket
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import options
from .compat import field_from_error, normalize
from .options import KOBOLDCPP, LLAMA_SERVER

#: The two node ids this module manages.
SERVER_NODE_ID = "LocalLLMServer"
CONTROL_NODE_ID = "LLMServerControl"

#: The Control node's actions. They live here rather than next to the node because the queue hook
#: has to read one out of a queued prompt's JSON, and the node imports from this module already.
CONTROL_FREE = "free vram (unload the model)"
CONTROL_LOAD = "load the model"
CONTROL_STOP = "stop the gateway"
CONTROL_STATUS = "status only"
CONTROL_ACTIONS = [CONTROL_FREE, CONTROL_LOAD, CONTROL_STOP, CONTROL_STATUS]

#: Control actions that mean "this graph wants the model UP" — the only ones that make the queue
#: hook stand down. A Control node set to *free vram* must NOT disarm it: the node may well sit
#: after the sampler in that graph, and then the safety net is the only thing freeing the card in
#: time. Unloading early just makes the node's own call a no-op.
CONTROL_KEEP = frozenset({CONTROL_LOAD, CONTROL_STATUS})

#: Backend roles.
CHAT = "chat"
EMBED = "embed"
ROLES = (CHAT, EMBED)

#: The path that answers 200 once a model is in. koboldcpp has no `/health`, but it only starts
#: listening after the model is loaded, so any endpoint answering at all means it is ready.
READY_PATHS = {LLAMA_SERVER: "/health", KOBOLDCPP: "/v1/models"}

#: Requests the embedding backend owns, when there is one.
EMBED_PATHS = frozenset({"/v1/embeddings", "/embeddings", "/embedding",
                         "/v1/rerank", "/rerank", "/reranking", "/v1/reranking"})

#: Headers that describe one hop and must not be copied to the next one.
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


class _Backend:
    """One managed server process."""

    def __init__(self, role):
        self.role = role
        self.proc = None
        self.sig = None
        self.port = 0
        self.loaded_at = 0.0

    def alive(self):
        return self.proc is not None and self.proc.poll() is None


class _State:
    def __init__(self):
        self.cfg = {}
        self.load_lock = threading.RLock()   # serialises spawn / kill, so two requests load once
        self.backends = {role: _Backend(role) for role in ROLES}
        self.httpd = None
        self.httpd_addr = ""
        self.httpd_thread = None
        self.log = deque(maxlen=600)
        self.last_request = 0.0
        self.requests = 0
        self.hook_installed = False
        self.idle_thread = None
        self.learned_drops = set()           # fields a 400 taught us to strip


STATE = _State()


# --------------------------------------------------------------------------------- logging
def log(msg):
    STATE.log.append(time.strftime("%H:%M:%S ") + str(msg))
    print("[LLM gateway] " + str(msg))


def log_tail(n=60):
    return "\n".join(list(STATE.log)[-n:])


# --------------------------------------------------------------------------------- backends
def running(role=CHAT):
    return STATE.backends[role].alive()


def _free_port():
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _argv(cfg, role, port):
    return options.embed_argv(cfg, port) if role == EMBED else options.chat_argv(cfg, port)


def _sig(cfg, role):
    """A backend reloads when its own command line changes — so the signature IS that command
    line, built with a placeholder port. Add an option to `options.py` and this covers it."""
    return (role, tuple(_argv(cfg, role, 0)), int(cfg.get("backend_port") or 0))


def _drain(proc, role):
    prefix = "" if role == CHAT else "[embed] "
    try:
        for line in iter(proc.stdout.readline, ""):
            if line == "" and proc.poll() is not None:
                break
            STATE.log.append(prefix + line.rstrip("\n"))
    except Exception:
        pass


def _probe(port, path="/health", timeout=2.0):
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", path)
        r = conn.getresponse()
        r.read()
        conn.close()
        return r.status
    except Exception:
        return 0


def _exit_hint(code):
    """Windows answers a missing DLL with a status code and not a word of explanation, and a CUDA
    build of llama-server run outside the shell it was built in is exactly that case."""
    if code in (-1073741515, 3221225781):        # 0xC0000135 STATUS_DLL_NOT_FOUND
        return (" A DLL it needs was not found - a CUDA build wants the CUDA runtime on PATH."
                " Try the binary by hand in a terminal first.")
    return ""


def _comfy_busy():
    """True while ComfyUI is executing a prompt. Loading a model on top of a running sampler is how
    you get an out-of-memory inside someone else's node, so the gateway waits instead."""
    try:
        from server import PromptServer
        jobs, _pending = PromptServer.instance.prompt_queue.get_current_queue_volatile()
        return bool(jobs)
    except Exception:
        return False


def _unload_comfy():
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache(True)
    except Exception as e:
        log("could not unload ComfyUI's models: " + str(e))


def ensure_model(role=CHAT, force=False, wait_for_comfy=True):
    """Load one backend if it is not up. Returns ``(ok, error)``.

    `force` overrides `autoload` (that is the Control node's explicit *load*). `wait_for_comfy` is
    off when the caller is itself a node inside a running prompt — otherwise it waits on itself.
    """
    cfg = STATE.cfg
    if not cfg:
        return False, "no model configured - run the Local LLM Server node once."
    if role == EMBED and not options.needs_embed_process(cfg):
        return False, "no separate embedding model is configured."
    if running(role):
        return True, ""
    if not force and not cfg.get("autoload", True):
        return False, "the model is unloaded and autoload is off - run the Local LLM Server node."

    with STATE.load_lock:
        if running(role):
            return True, ""
        if wait_for_comfy and _comfy_busy():
            log("ComfyUI is busy - waiting for it to finish before loading the model")
            deadline = time.time() + 900
            while _comfy_busy() and time.time() < deadline:
                time.sleep(0.5)
        if cfg.get("unload_comfy", True):
            _unload_comfy()

        back = STATE.backends[role]
        port = (int(cfg.get("backend_port") or 0) or 0) if role == CHAT else 0
        port = port or _free_port()
        argv = _argv(cfg, role, port)
        if not argv:
            return False, "nothing to launch for the %s backend." % role
        what = os.path.basename(argv[2] if len(argv) > 2 else "?")
        log("loading %s%s on port %d" % ("" if role == CHAT else "embedding model ", what, port))
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
        except Exception as e:
            return False, "could not launch " + str(cfg.get("binary")) + ": " + str(e)

        back.proc = proc
        back.sig = _sig(cfg, role)
        back.port = port
        threading.Thread(target=_drain, args=(proc, role), daemon=True).start()

        if role == EMBED:
            ready_path = "/health"               # the embedding process is always a llama-server
        else:
            ready_path = ((cfg.get("ready_path") or "").strip()
                          or READY_PATHS.get(cfg.get("flavour"), "/health"))
        t0 = time.time()
        timeout_s = int(cfg.get("startup_timeout", 300))
        while time.time() < t0 + timeout_s:
            if proc.poll() is not None:
                back.proc = None
                return False, ("the server exited while loading (code %s).%s\n--- log ---\n%s"
                               % (proc.returncode, _exit_hint(proc.returncode), log_tail(25)))
            if _probe(port, ready_path) == 200:
                back.loaded_at = time.time()
                log("%s ready in %.1fs" % (role, time.time() - t0))
                return True, ""
            time.sleep(0.4)
        stop_model("startup timed out", role)
        return False, "the model did not become ready within %ds." % timeout_s


def stop_model(reason="", role=None):
    """Kill a backend, or every backend when `role` is None. The VRAM is only actually free once
    the process is gone, so this waits for it."""
    stopped = False
    with STATE.load_lock:
        for r in (ROLES if role is None else (role,)):
            back = STATE.backends[r]
            proc = back.proc
            back.proc = None
            back.loaded_at = 0.0
            if proc is None:
                continue
            stopped = True
            log("unloading the %s model%s" % (r, (" - " + reason) if reason else ""))
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except Exception:
                        proc.kill()
                        proc.wait(timeout=10)
            except Exception as e:
                log("could not stop the server cleanly: " + str(e))
    return stopped


# --------------------------------------------------------------------------------- status
def _stem(path):
    return os.path.splitext(os.path.basename(path or ""))[0]


def _model_id():
    """What the gateway calls the chat model. `alias` wins, because that is also what llama-server
    reports for itself once it is up, and the client remembers whichever id it was shown."""
    cfg = STATE.cfg
    return (options.txt(cfg.get("alias")) or _stem(cfg.get("model")) or "local")


def _embed_id():
    return _stem(STATE.cfg.get("embed_model_resolved")) or "embedding"


def _role_status(role):
    back = STATE.backends[role]
    up = back.alive()
    return {"loaded": up, "port": back.port if up else 0,
            "pid": back.proc.pid if up else 0,
            "loaded_seconds": round(time.time() - back.loaded_at, 1) if back.loaded_at else 0.0}


def status():
    cfg = STATE.cfg
    chat = _role_status(CHAT)
    out = {
        "listening": STATE.httpd_addr,
        "model_loaded": chat["loaded"],
        "model": os.path.basename(cfg.get("model", "")) if cfg else "",
        "model_id": _model_id() if cfg else "",
        "backend_port": chat["port"],
        "pid": chat["pid"],
        "loaded_seconds": chat["loaded_seconds"],
        "idle_seconds": round(time.time() - STATE.last_request, 1) if STATE.last_request else 0.0,
        "requests": STATE.requests,
        "autoload": bool(cfg.get("autoload", True)) if cfg else False,
        "idle_unload_minutes": cfg.get("idle_minutes", 0) if cfg else 0,
        "free_on_prompt": bool(cfg.get("free_on_prompt", True)) if cfg else False,
        "dropped_fields": sorted(STATE.learned_drops),
    }
    if cfg and options.needs_embed_process(cfg):
        out["embed"] = dict(_role_status(EMBED), model_id=_embed_id())
    elif cfg and options.txt(cfg.get("embed_model_resolved")):
        out["embed"] = {"model_id": _embed_id(), "in_chat_process": True}
    return out


def status_text():
    s = status()
    if not s["listening"]:
        return "gateway: not listening"
    if s["model_loaded"]:
        model = "loaded: %s (pid %d, up %.0fs)" % (s["model"], s["pid"], s["loaded_seconds"])
    else:
        model = "unloaded" + (" - loads on the next request" if s["autoload"] else " - autoload off")
    lines = ["gateway: %s  (%d requests)" % (s["listening"], s["requests"]),
             "model: " + model,
             "model id for SillyTavern: " + s["model_id"]]
    emb = s.get("embed")
    if emb:
        if emb.get("in_chat_process"):
            lines.append("embeddings: %s (served by the chat process)" % emb["model_id"])
        else:
            lines.append("embeddings: %s (%s)"
                         % (emb["model_id"], "loaded" if emb["loaded"] else "unloaded"))
    if s["dropped_fields"]:
        lines.append("dropped fields: " + ", ".join(s["dropped_fields"]))
    return "\n".join(lines)


def _synthesise(bare):
    """What the gateway answers with no model loaded, so SillyTavern's Connect works anyway."""
    cfg = STATE.cfg
    if bare in ("/health", "/v1/health"):
        return {"status": "ok",
                "kinburg": "gateway up, model unloaded - it loads on the next request"}
    if bare in ("/models", "/v1/models"):
        ids = [_model_id()]
        if options.needs_embed_process(cfg):
            ids.append(_embed_id())
        return {"object": "list",
                "data": [{"id": i, "object": "model", "created": 0,
                          "owned_by": "kinburg-gateway"} for i in ids]}
    if bare == "/props":
        return {"default_generation_settings": {"n_ctx": int(cfg.get("n_ctx", 4096) or 4096)},
                "total_slots": 1, "model_path": cfg.get("model", ""),
                "chat_template": "", "build_info": "kinburg-gateway"}
    if bare == "/":
        return status()
    return None


def _role_for(bare):
    return EMBED if (bare in EMBED_PATHS and options.needs_embed_process(STATE.cfg)) else CHAT


# --------------------------------------------------------------------------------- HTTP side
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "kinburg-llm-gateway"

    # BaseHTTPRequestHandler writes a line to stderr per request; ours go to the ring buffer.
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_PUT(self):
        self._handle("PUT")

    def do_DELETE(self):
        self._handle("DELETE")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ------------------------------------------------------------------ small helpers
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")

    def _send_json(self, code, obj):
        blob = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self._cors()
        self.end_headers()
        self._write(blob)

    def _send_error_json(self, code, message):
        self._send_json(code, {"error": {"message": message, "type": "kinburg_gateway",
                                         "code": code}})

    def _write(self, blob):
        try:
            self.wfile.write(blob)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True
            return False

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n > 0 else b""

    # ------------------------------------------------------------------ routing
    def _handle(self, method):
        path = self.path
        bare = path.split("?")[0].rstrip("/") or "/"
        body = self._read_body()
        STATE.last_request = time.time()
        STATE.requests += 1

        if bare == "/kinburg/status":
            return self._send_json(200, status())
        if bare == "/kinburg/unload":
            return self._send_json(200, {"unloaded": stop_model("asked over /kinburg/unload")})
        if bare == "/kinburg/load":
            ok, err = ensure_model(CHAT, force=True)
            return self._send_json(200 if ok else 503, {"loaded": ok, "error": err})

        role = _role_for(bare)
        if method == "POST":
            ok, err = ensure_model(role)
            if not ok:
                return self._send_error_json(503, err)
            body = self._repair(body)
        elif bare in ("/models", "/v1/models"):
            # Always ours, loaded or not. The chat client remembers the id it was shown, and it
            # must not change under it when the model happens to be up (llama-server names the
            # model after its own path or `--alias`, which is a different string).
            return self._send_json(200, _synthesise(bare))
        elif not running(role):
            synth = _synthesise(bare)
            if synth is not None:
                return self._send_json(200, synth)
            return self._send_error_json(503, "the model is not loaded and this path needs it.")

        self._relay(method, path, body, role)

    def _repair(self, body):
        """Fix the request body llama.cpp would refuse. Non-JSON bodies pass through untouched."""
        cfg = STATE.cfg
        if not body or not cfg.get("fix_params", True):
            return body
        try:
            obj = json.loads(body.decode("utf-8"))
        except Exception:
            return body
        drop = tuple(cfg.get("drop_fields", ())) + tuple(STATE.learned_drops)
        fixed, notes = normalize(obj, drop=drop)
        for n in notes:
            log("request: " + n)
        return json.dumps(fixed).encode("utf-8") if notes else body

    # ------------------------------------------------------------------ forwarding
    def _relay(self, method, path, body, role=CHAT, attempt=0):
        timeout = int(STATE.cfg.get("request_timeout", 900))
        port = STATE.backends[role].port
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["Host"] = "127.0.0.1:" + str(port)
        if body:
            headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            conn.request(method, path, body=body or None, headers=headers)
            resp = conn.getresponse()

            # A 400 from llama.cpp names the field it choked on. Drop that one field and try again
            # - that is what keeps a new SillyTavern option from breaking every message.
            if resp.status == 400 and STATE.cfg.get("auto_retry", True) and body and attempt < 4:
                raw = resp.read()
                retry = self._retry_body(raw, body)
                conn.close()
                if retry is not None:
                    return self._relay(method, path, retry, role, attempt + 1)
                return self._send_raw(400, resp.getheader("Content-Type", "application/json"), raw)

            self._stream(resp)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        except Exception as e:
            log("forwarding %s %s failed: %s" % (method, path, e))
            try:
                self._send_error_json(502, "the model server did not answer: " + str(e))
            except Exception:
                self.close_connection = True
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _retry_body(self, raw, body):
        """The same body without the field llama.cpp rejected, or None if we cannot tell."""
        name = field_from_error(raw.decode("utf-8", "replace"))
        if not name:
            return None
        try:
            obj = json.loads(body.decode("utf-8"))
        except Exception:
            return None
        if name not in obj:
            return None
        obj.pop(name)
        STATE.learned_drops.add(name)
        log("llama.cpp rejected '%s' - dropped it and retried (stripped from now on)" % name)
        return json.dumps(obj).encode("utf-8")

    def _send_raw(self, code, ctype, blob):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(blob)))
        self._cors()
        self.end_headers()
        self._write(blob)

    def _stream(self, resp):
        """Copy the answer through as it arrives - token by token for a streamed reply."""
        clen = resp.getheader("Content-Length")
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        for h in ("Cache-Control", "X-Accel-Buffering"):
            if resp.getheader(h):
                self.send_header(h, resp.getheader(h))
        if clen is not None:
            self.send_header("Content-Length", clen)
            self._cors()
            self.end_headers()
            left = int(clen)
            while left > 0:
                chunk = resp.read(min(65536, left))
                if not chunk or not self._write(chunk):
                    break
                left -= len(chunk)
            return
        # No length: the upstream is streaming (SSE). Re-chunk it ourselves so the HTTP/1.1 framing
        # stays valid, and flush every chunk - buffering here would stall the live reply.
        self.send_header("Transfer-Encoding", "chunked")
        self._cors()
        self.end_headers()
        while True:
            try:
                chunk = resp.read1(65536)
            except Exception:
                break
            if not chunk:
                break
            if not self._write(b"%x\r\n" % len(chunk) + chunk + b"\r\n"):
                return
            try:
                self.wfile.flush()
            except Exception:
                return
        self._write(b"0\r\n\r\n")


# --------------------------------------------------------------------------------- lifecycle
def _idle_watch():
    while STATE.httpd is not None:
        time.sleep(15)
        try:
            minutes = float(STATE.cfg.get("idle_minutes", 0) or 0)
            if minutes > 0 and STATE.last_request and any(running(r) for r in ROLES):
                if time.time() - STATE.last_request > minutes * 60:
                    stop_model("idle for %g min" % minutes)
        except Exception:
            pass


def wants_the_model(nodes):
    """True when a queued prompt is one the model should survive: it starts the server, or it holds
    a Control node that asks for the model rather than for the memory."""
    for v in (nodes or {}).values():
        if not isinstance(v, dict):
            continue
        kind = v.get("class_type")
        if kind == SERVER_NODE_ID:
            return True
        if kind == CONTROL_NODE_ID:
            if (v.get("inputs") or {}).get("action") in CONTROL_KEEP:
                return True
    return False


def _on_prompt(json_data):
    """ComfyUI queue hook: a prompt that is not about the LLM server takes the VRAM back.

    This runs on the web server's own thread and blocks it until the process is gone, which is the
    point: the prompt is not queued until the VRAM is actually free, so the sampler cannot start
    while the model is still holding the card. It is a second or two.
    """
    try:
        if STATE.cfg.get("free_on_prompt", True) and any(running(r) for r in ROLES):
            if not wants_the_model((json_data or {}).get("prompt")):
                stop_model("a ComfyUI prompt was queued")
    except Exception as e:
        log("prompt hook failed: " + str(e))
    return json_data


def _install_hook():
    if STATE.hook_installed:
        return
    try:
        from server import PromptServer
        PromptServer.instance.add_on_prompt_handler(_on_prompt)
        STATE.hook_installed = True
    except Exception as e:
        log("could not hook ComfyUI's prompt queue: " + str(e))


def configure(**cfg):
    """Set (or replace) what the gateway serves. A backend whose command line changed is stopped,
    so the next request that needs it starts it with the new flags."""
    for role in ROLES:
        back = STATE.backends[role]
        if back.alive() and back.sig != _sig(cfg, role):
            stop_model("the %s model or its flags changed" % role, role)
    STATE.cfg = cfg


def serve(host, port):
    """Start the listener (or keep the running one). Returns ``(ok, error)``."""
    addr = "http://%s:%d" % (host, int(port))
    if STATE.httpd is not None and STATE.httpd_addr == addr:
        _install_hook()
        return True, ""
    stop_serving()
    try:
        httpd = ThreadingHTTPServer((host, int(port)), _Handler)
    except Exception as e:
        return False, "could not listen on %s:%s - %s" % (host, port, e)
    httpd.daemon_threads = True
    STATE.httpd = httpd
    STATE.httpd_addr = addr
    STATE.httpd_thread = threading.Thread(target=httpd.serve_forever,
                                          kwargs={"poll_interval": 0.4}, daemon=True)
    STATE.httpd_thread.start()
    log("listening on %s  (SillyTavern points at %s/v1)" % (addr, addr))
    _install_hook()
    if STATE.idle_thread is None or not STATE.idle_thread.is_alive():
        STATE.idle_thread = threading.Thread(target=_idle_watch, daemon=True)
        STATE.idle_thread.start()
    return True, ""


def stop_serving():
    httpd = STATE.httpd
    STATE.httpd = None
    STATE.httpd_addr = ""
    if httpd is not None:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass
        log("gateway stopped listening")


def shutdown(reason=""):
    stop_model(reason or "shutting down")
    stop_serving()


atexit.register(shutdown, "ComfyUI is exiting")
