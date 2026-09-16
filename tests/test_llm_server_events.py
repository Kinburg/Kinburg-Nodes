"""What the gateway puts in its log — and, just as much, what it must not.

The live log is drawn on a ComfyUI canvas, and the requests going through the gateway are somebody's
conversation with a character. So the rule these checks exist to hold is: **the default summary
carries shape and numbers, never text**. `log_text` is what turns previews on, and even then they
are trimmed. A regression here is not a cosmetic one.

The rest is the reading the log is for: which sampler settings the client actually sent (the answer
to "what is my SillyTavern preset really doing?"), the token counts under either spelling llama.cpp
uses, and the fact that a STREAMED answer is measured by counting frames as they fly past rather
than by holding a reply in memory to count it afterwards.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "llm_server")
ev = load_module("kn.llm_server.events", "llm_server/events.py")

check = Checker()

SECRET = "the thing nobody else at this computer should read"
CHAT = {
    "messages": [
        {"role": "system", "content": "You are Dolores. " + SECRET},
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": SECRET},
    ],
    "stream": True, "temperature": 1.05, "top_p": 0.95, "top_k": 0, "min_p": 0.05,
    "repetition_penalty": 1.1, "dry_multiplier": 0.8, "dry_base": 1.75,
    "dry_sequence_breakers": ["\n", ":", '"', "*"], "stop": ["###", "User:"],
    "max_tokens": 300, "model": "whatever", "not_a_sampler": "ignored",
}

# ------------------------------------------------------------------ the rule: no text by default
summary = ev.request_summary(CHAT)
blob = json.dumps(summary)
check("the default summary carries NO message text", SECRET not in blob, blob[:120])
check("...not even the system prompt", "Dolores" not in blob)
check("it counts the conversation instead", summary["messages"] == 4 and summary["system"] == 1,
      summary)
check("...and its size, which is what says the context is filling up",
      summary["chars"] == sum(len(m["content"]) for m in CHAT["messages"]), summary["chars"])
check("streaming is flagged", summary["stream"] is True)

# ------------------------------------------------------------------ the sampler line
s = summary["sampler"]
check("every numeric sampler the client sent is picked up",
      s["temperature"] == 1.05 and s["top_p"] == 0.95 and s["min_p"] == 0.05
      and s["repetition_penalty"] == 1.1 and s["max_tokens"] == 300, s)
check("a zero is a setting, not an absence", s["top_k"] == 0)
check("what is not a sampler is not collected", "not_a_sampler" not in s and "model" not in s)
check("list-valued options are counted, not quoted",
      s["dry_breakers"] == 4 and s["stops"] == 2 and "\\n" not in json.dumps(s), s)
check("nothing absent is invented", "mirostat" not in s and "xtc_probability" not in s)
# SillyTavern sends the breakers as a JSON string on some paths; len() of THAT is a character
# count, and a log row reading "dry ×27" would be a confident lie.
as_string = {"dry_sequence_breakers": json.dumps(["\n", ":"])}
check("a breaker list that arrived as a string is not counted as characters",
      "dry_breakers" not in ev.sampler_of(as_string), ev.sampler_of(as_string))
check("a constrained request says so",
      ev.sampler_of({"grammar": "root ::= \"a\""})["constrained"] == 1)
check("a string where a number belongs is skipped, not crashed on",
      ev.sampler_of({"temperature": "hot"}) == {})
check("and a non-dict is simply empty", ev.sampler_of(None) == {} and ev.request_summary([]) == {})

# ------------------------------------------------------------------ previews, when asked for
check("the preview is the LAST user turn, not the whole conversation",
      ev.prompt_preview(CHAT) == SECRET, ev.prompt_preview(CHAT))
check("...trimmed to what fits a log row",
      len(ev.prompt_preview({"messages": [{"role": "user", "content": "x" * 5000}]}, 40)) <= 42)
check("an image turn's text is found inside OpenAI's list-of-parts form",
      ev.prompt_preview({"messages": [{"role": "user", "content": [
          {"type": "text", "text": "what is this"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]})
      == "what is this")
check("...and the base64 of the image never comes with it",
      "AAAA" not in ev.prompt_preview({"messages": [{"role": "user", "content": [
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}))
check("a conversation with no user turn previews nothing",
      ev.prompt_preview({"messages": [{"role": "system", "content": "x"}]}) == "")
check("a text-completion request previews its prompt",
      ev.prompt_preview({"prompt": "once upon a time"}) == "once upon a time")
check("newlines are folded so one row stays one row",
      ev.prompt_preview({"prompt": "a\n\n  b"}) == "a b")

reply = {"choices": [{"message": {"role": "assistant", "content": "  she smiled.\n\n"}}]}
check("the reply preview is the reply", ev.reply_preview(reply) == "she smiled.")
check("...also in the completion (non-chat) shape",
      ev.reply_preview({"choices": [{"text": "plain"}]}) == "plain")
check("an error body previews nothing", ev.reply_preview({"error": {"message": "no"}}) == "")

# ------------------------------------------------------------------ token counts
check("usage is read where OpenAI puts it",
      ev.usage_of({"usage": {"prompt_tokens": 1420, "completion_tokens": 180}})
      == {"prompt_tokens": 1420, "gen_tokens": 180}, ev.usage_of({"usage": {"prompt_tokens": 1}}))
t = ev.usage_of({"timings": {"prompt_n": 12, "predicted_n": 34, "predicted_per_second": 14.53}})
check("...and where llama.cpp puts it", t == {"prompt_tokens": 12, "gen_tokens": 34, "tps": 14.5}, t)
both = ev.usage_of({"usage": {"prompt_tokens": 1420, "completion_tokens": 180},
                    "timings": {"prompt_n": 1, "predicted_n": 2, "predicted_per_second": 9.0}})
check("usage wins when both are there, but the rate is still taken",
      both["prompt_tokens"] == 1420 and both["gen_tokens"] == 180 and both["tps"] == 9.0, both)
check("a body with neither yields neither", ev.usage_of({"choices": []}) == {})

# ------------------------------------------------------------------ a streamed answer
tail_plain = 'data: {"choices":[{"delta":{"content":"e"}}]}\n\ndata: [DONE]\n\n'
st = ev.stream_stats(41, tail_plain)
check("with no usage in the stream, the frames ARE the count",
      st["gen_tokens"] == 40 and st["estimated"] is True, st)
tail_usage = ('data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":1420,'
              '"completion_tokens":180}}\n\ndata: [DONE]\n\n')
st2 = ev.stream_stats(41, tail_usage)
check("a final chunk carrying real usage wins over the estimate",
      st2["gen_tokens"] == 180 and st2["prompt_tokens"] == 1420 and st2["estimated"] is False, st2)
check("the [DONE] frame is not a token", ev.stream_stats(1, "data: [DONE]\n\n")["gen_tokens"] == 0)
check("a truncated tail does not throw", ev.stream_stats(3, 'data: {"choi')["gen_tokens"] == 2)

# ------------------------------------------------------------------ the ring
ev.clear()
first = ev.emit("note", text="one")
ev.emit("note", text="two")
check("an event is stamped with a sequence and a clock",
      first["seq"] >= 1 and first["kind"] == "note" and ":" in first["t"], first)
check("the ring hands back what it has", [e["text"] for e in ev.recent()] == ["one", "two"])
check("...and only what a caller has not seen",
      [e["text"] for e in ev.recent(after=first["seq"])] == ["two"])
check("the sequence keeps climbing so a reconnect cannot replay old rows",
      ev.emit("note", text="three")["seq"] == first["seq"] + 2)
for i in range(ev.MAX_EVENTS + 50):
    ev.emit("note", text="flood %d" % i)
check("an endless chat does not grow an endless ring", len(ev.recent()) == ev.MAX_EVENTS,
      len(ev.recent()))
check("...and it is the oldest that go", ev.recent()[0]["text"].startswith("flood"))
ev.clear()
check("clear empties it", ev.recent() == [] and ev.last_seq() > 0)

check.done()
