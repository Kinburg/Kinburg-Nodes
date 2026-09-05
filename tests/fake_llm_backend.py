"""A stand-in `llama-server` for the gateway suite — the real one is 4 GB of model away.

It is launched exactly as the gateway launches the real thing (`<binary> -m <model> --host … --port
… --ctx-size … -ngl …`), answers `/health` once it is up, and copies the two behaviours the gateway
exists to work around:

* it validates `dry_sequence_breakers` the way llama.cpp's `server-schema.cpp` does — a JSON string
  or an empty array is a 400 whose message names the field;
* any field named in `--reject` is a 400 in the same shape, which is how the suite exercises the
  drop-the-field-and-retry path without pinning it to one option's name.

`GET /last` hands back the body of the last request it received, so a test can prove what actually
reached the model rather than what the gateway meant to send.
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"last": None, "reject": set(), "requests": 0}


def _validate(body):
    """The subset of llama.cpp's field validation this suite needs. Returns an error or ""."""
    for name in sorted(STATE["reject"]):
        if name in body:
            return "Field '%s': Error: this build does not accept %s" % (name, name)
    if "dry_sequence_breakers" in body:
        v = body["dry_sequence_breakers"]
        if not (isinstance(v, list) and v and all(isinstance(x, str) for x in v)):
            return ("Field 'dry_sequence_breakers': Error: dry_sequence_breakers must be a "
                    "non-empty array of strings")
    return ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        blob = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/health", "/v1/health"):
            return self._json(200, {"status": "ok"})
        if p == "/last":
            return self._json(200, {"body": STATE["last"], "requests": STATE["requests"]})
        if p in ("/models", "/v1/models"):
            return self._json(200, {"object": "list", "data": [{"id": "fake-backend"}]})
        if p == "/props":
            return self._json(200, {"build_info": "fake-backend"})
        return self._json(404, {"error": {"message": "no such path: " + p}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = {}
        STATE["last"] = body
        STATE["requests"] += 1

        bad = _validate(body)
        if bad:
            return self._json(400, {"error": {"message": bad, "type": "invalid_request_error"}})
        if body.get("stream"):
            return self._sse()
        return self._json(200, {
            "id": "fake", "object": "chat.completion", "model": body.get("model", "fake"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
        })

    def _sse(self):
        """Three tokens, chunked and flushed one at a time — the shape a live reply arrives in."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for piece in ("one", "two", "three"):
            frame = ("data: " + json.dumps(
                {"choices": [{"delta": {"content": piece}, "index": 0}]}) + "\n\n").encode("utf-8")
            self.wfile.write(b"%x\r\n" % len(frame) + frame + b"\r\n")
            self.wfile.flush()
            time.sleep(0.02)
        done = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n" % len(done) + done + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--reject", default="")
    ap.add_argument("--boot-seconds", type=float, default=0.0)
    # Whatever else the gateway puts on the command line is the real server's business, not ours.
    args, _rest = ap.parse_known_args(argv if argv is not None else sys.argv[1:])
    STATE["reject"] = {s for s in args.reject.split(",") if s}
    if args.boot_seconds:
        time.sleep(args.boot_seconds)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.serve_forever()


if __name__ == "__main__":
    main()
