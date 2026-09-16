"""The gateway's log, as structured events rather than lines of text.

The text ring in :mod:`gateway` is fine for a `server_log` output you read after the fact. What it
cannot answer is the question you actually have while a character is talking to you: *did that
message arrive, what did SillyTavern send with it, how long did it take, and how full is the
context now?* So every interesting moment is also an **event** — a dict with fields rather than a
sentence — kept in a ring here and pushed to the frontend over the websocket, where the
*LLM Server Live Log* node renders it.

Two things make this different from the pack's other live logs:

* the events happen **outside a ComfyUI run**. SillyTavern talks to the gateway whenever it likes,
  with no prompt queued and quite possibly no browser open, so the ring is kept on the backend and
  a log node fetches the backlog when it appears — rather than only ever seeing what arrives while
  it is watching;
* the requests carry **someone's conversation**. What is logged by default is shape and numbers
  only: how many messages, how many characters, which samplers, how many tokens, how long. The
  message text is opt-in (`log_text` on the server node) and trimmed even then, because a chat
  window rendered on a ComfyUI canvas is not always for the person sitting there.

The summarisers are pure and the ring is plain data, so all of it is testable without a server.
"""
import json
import threading
import time
from collections import deque

#: How many events are kept for a log node that appears late. ~400 is a long chat session.
MAX_EVENTS = 400

#: The websocket channel the log node listens on.
CHANNEL = "kinburg.llmserver"

#: How much of a message is kept when `log_text` is on. Enough to recognise the turn, not enough
#: to be a transcript.
PREVIEW_CHARS = 400

#: Sampler fields worth showing, in the order they read best. This is the answer to "what is my
#: SillyTavern preset ACTUALLY sending?", which is otherwise guesswork.
SAMPLER_KEYS = (
    "temperature", "top_p", "top_k", "min_p", "typical_p", "top_n_sigma", "nsigma",
    "repetition_penalty", "repeat_penalty", "frequency_penalty", "presence_penalty",
    "repeat_last_n", "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "xtc_probability", "xtc_threshold", "mirostat", "seed", "max_tokens", "n_predict",
)

_lock = threading.Lock()
_events = deque(maxlen=MAX_EVENTS)
_seq = 0


# ------------------------------------------------------------------------------- the ring
def emit(kind, **fields):
    """Stamp an event, keep it, and push it to any log node that is watching.

    Fire-and-forget on purpose: a frontend that is not there, or a websocket that is closed, must
    never be able to fail a request that SillyTavern is waiting on.
    """
    global _seq
    with _lock:
        _seq += 1
        ev = dict(fields, kind=kind, seq=_seq, t=time.strftime("%H:%M:%S"), ts=time.time())
        _events.append(ev)
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(CHANNEL, ev)
    except Exception:
        pass
    return ev


def recent(after=0, limit=MAX_EVENTS):
    """Events newer than `after` (a seq number), oldest first — the backlog a log node replays."""
    with _lock:
        out = [e for e in _events if e["seq"] > int(after or 0)]
    return out[-int(limit):] if limit else out


def clear():
    global _seq
    with _lock:
        _events.clear()
    return _seq


def last_seq():
    return _seq


# ------------------------------------------------------------------------------- summarisers
def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def sampler_of(body):
    """The sampler settings present in a request, in a fixed order. Absent keys stay absent — the
    point is to show what the client sent, not what it could have sent."""
    if not isinstance(body, dict):
        return {}
    out = {}
    for k in SAMPLER_KEYS:
        v = _num(body.get(k))
        if v is not None:
            out[k] = v
    # SillyTavern sends this as a JSON *string* on some paths (see compat.py) — len() of that is
    # a character count, and "×27 breakers" in the log would be a lie. Count only a real list; the
    # repair that turns the string into one shows up as its own note anyway.
    breakers = body.get("dry_sequence_breakers")
    if isinstance(breakers, (list, tuple)) and breakers:
        out["dry_breakers"] = len(breakers)
    if body.get("grammar") or body.get("json_schema"):
        out["constrained"] = 1
    stops = body.get("stop")
    if isinstance(stops, (list, tuple)) and stops:
        out["stops"] = len(stops)
    return out


def request_summary(body):
    """Shape and size of a chat request — no text. `chars` is the whole conversation as sent, which
    is the number that tells you the context is about to overflow."""
    if not isinstance(body, dict):
        return {}
    msgs = body.get("messages")
    out = {"stream": bool(body.get("stream")), "sampler": sampler_of(body)}
    if isinstance(msgs, list):
        out["messages"] = len(msgs)
        out["chars"] = sum(len(m.get("content") or "") for m in msgs if isinstance(m, dict))
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        out["system"] = roles.count("system")
        out["images"] = sum(1 for m in msgs if isinstance(m, dict)
                            and isinstance(m.get("content"), list))
    elif isinstance(body.get("prompt"), str):
        out["chars"] = len(body["prompt"])
    if isinstance(body.get("input"), (str, list)):
        inp = body["input"]
        out["inputs"] = 1 if isinstance(inp, str) else len(inp)
    return out


def _trim(text, limit=PREVIEW_CHARS):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _content_text(content):
    """A message's text, whether it is a plain string or OpenAI's list-of-parts form."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text")
    return ""


def prompt_preview(body, limit=PREVIEW_CHARS):
    """The last user turn — the thing you just typed — and nothing else of the conversation."""
    if not isinstance(body, dict):
        return ""
    msgs = body.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                return _trim(_content_text(m.get("content")), limit)
        return ""
    return _trim(body.get("prompt", ""), limit)


def reply_preview(obj, limit=PREVIEW_CHARS):
    if not isinstance(obj, dict):
        return ""
    choice = (obj.get("choices") or [{}])[0]
    if not isinstance(choice, dict):
        return ""
    msg = choice.get("message") or {}
    return _trim(msg.get("content") or choice.get("text") or "", limit)


def usage_of(obj):
    """Token counts out of a response body, under either spelling llama.cpp uses."""
    if not isinstance(obj, dict):
        return {}
    u = obj.get("usage")
    out = {}
    if isinstance(u, dict):
        for src, dst in (("prompt_tokens", "prompt_tokens"),
                         ("completion_tokens", "gen_tokens"),
                         ("total_tokens", "total_tokens")):
            v = _num(u.get(src))
            if v is not None:
                out[dst] = int(v)
    t = obj.get("timings")
    if isinstance(t, dict):
        for src, dst in (("prompt_n", "prompt_tokens"), ("predicted_n", "gen_tokens")):
            v = _num(t.get(src))
            if v is not None:
                out.setdefault(dst, int(v))
        v = _num(t.get("predicted_per_second"))
        if v is not None:
            out["tps"] = round(float(v), 1)
    return out


def stream_stats(frames, tail):
    """What a streamed answer can tell us once it is over.

    Each SSE frame llama.cpp sends carries one token, so counting frames IS the generated-token
    count — but the final frame may also carry real `usage`/`timings`, which wins when present.
    `tail` is the last couple of kilobytes; parsing more than that to learn a number is not worth
    holding a reply in memory for.
    """
    out = {"gen_tokens": max(0, int(frames) - 1), "estimated": True}
    for line in reversed(str(tail or "").splitlines()):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            got = usage_of(json.loads(payload))
        except Exception:
            continue
        if got:
            out.update(got)
            out["estimated"] = False
            break
    return out
