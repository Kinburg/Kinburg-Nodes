"""Reading a GGUF's own answers out of its header — the parser, and the template probe.

Two things are worth pinning down here.

**The hand-rolled KV parser.** `gguf.GGUFReader` takes ~7 seconds on a 17 GB model because its cost
scales with the tensor count, which is useless for a dropdown, so `read_metadata` walks the KV block
itself and stops before the tensor index. The dangerous part is skipping the 250k-entry token array:
land one byte off and every key after it is garbage. So the fixtures here put a real key *after* a
huge array on purpose.

**The probe.** Which reasoning controls a model honours is read out of its chat template — the AST
says which variables it mentions, rendering says which values it accepts. Both are checked against
templates written to behave like the real families (Qwen3.5/3.8 validates the effort and raises;
Gemma-4 reads `enable_thinking` with the opposite polarity and no effort at all).

Fixtures are synthesised GGUF headers in a temp dir — a few hundred bytes each, no model needed.
"""
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import comfy_on_path, fake_package, load_module  # noqa: E402

comfy_on_path()
fake_package("kn", "local_llm")
load_module("kn.local_llm.llm_node", "local_llm/llm_node.py")
gi = load_module("kn.local_llm.gguf_info", "local_llm/gguf_info.py")

TMP = Path(tempfile.mkdtemp(prefix="kinburg-gguf-"))
fails = []


def check(label, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + label + (("  " + str(extra)) if extra else ""))
    if not cond:
        fails.append(label)


# ── a minimal GGUF writer, so the parser is tested against bytes and not against itself ────────
STR, U32, F32, ARRAY = 8, 4, 6, 9


def _s(text):
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key, type_id, payload):
    return _s(key) + struct.pack("<I", type_id) + payload


def kv_str(key, text):
    return _kv(key, STR, _s(text))


def kv_u32(key, n):
    return _kv(key, U32, struct.pack("<I", int(n)))


def kv_f32(key, x):
    return _kv(key, F32, struct.pack("<f", float(x)))


def kv_str_array(key, items):
    return _kv(key, ARRAY, struct.pack("<I", STR) + struct.pack("<Q", len(items))
               + b"".join(_s(x) for x in items))


def kv_u32_array(key, items):
    return _kv(key, ARRAY, struct.pack("<I", U32) + struct.pack("<Q", len(items))
               + b"".join(struct.pack("<I", int(x)) for x in items))


def write_gguf(name, entries, magic=b"GGUF", version=3):
    path = TMP / name
    body = b"".join(entries)
    path.write_bytes(magic + struct.pack("<I", version)
                     + struct.pack("<QQ", 866, len(entries)) + body)
    return str(path)


# Behaves like Qwen3.5/3.8: validates reasoning_effort and raises, defaults to xhigh, and prefills
# the think tag unless enable_thinking is explicitly false.
QWENISH = (
    "{%- set eff = reasoning_effort|default('xhigh') %}"
    "{%- if enable_thinking is undefined or enable_thinking is true %}"
    "{%- if eff not in ('xhigh', 'medium', 'low') %}"
    "{{- raise_exception('Unexpected reasoning effort ' ~ eff) }}{%- endif %}"
    "{%- if eff != 'medium' %}<|im_start|>system\nEffort {{ eff }}<|im_end|>\n{%- endif %}"
    "{%- endif %}"
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n"
    "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
    "{%- else %}<think>\n{%- endif %}{%- endif %}"
)
# Behaves like Gemma-4: enable_thinking OFF unless asked, and no notion of effort at all.
GEMMAISH = (
    "{{ bos_token }}"
    "{%- if enable_thinking is defined and enable_thinking %}<|turn>system\n<|think|><turn|>\n"
    "{%- endif %}"
    "{%- for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|turn>model\n{%- endif %}"
)
# A template that needs a user turn and refuses a bare system one, to exercise PROBE_SHAPES.
PICKY = (
    "{%- if messages[0].role == 'system' %}{{ raise_exception('no system messages') }}{%- endif %}"
    "{%- for m in messages %}[{{ m.role }}] {{ m.content }}\n{%- endfor %}"
)

BIG_TOKENS = [f"tok{i}" for i in range(9000)]          # past _ARRAY_MAX, so it must be SKIPPED

QWEN = write_gguf("qwenish.gguf", [
    kv_str("general.architecture", "qwen35"),
    kv_str("general.name", "Qwenish-27B"),
    kv_str("general.size_label", "27B"),
    kv_u32("qwen35.context_length", 262144),
    kv_u32("qwen35.block_count", 65),
    kv_f32("general.sampling.temp", 1.0),
    kv_f32("general.sampling.top_p", 0.949999988079071),
    kv_u32("general.sampling.top_k", 20),
    kv_f32("general.sampling.min_p", 0.0),
    kv_f32("general.sampling.penalty_repeat", 1.0),
    # The trap: a huge array in the middle, with real keys on both sides of it.
    kv_str_array("tokenizer.ggml.tokens", BIG_TOKENS),
    kv_u32_array("tokenizer.ggml.token_type", [1] * 9000),
    kv_u32("tokenizer.ggml.eos_token_id", 248046),
    kv_str("tokenizer.chat_template", QWENISH),
])
GEMMA = write_gguf("gemmaish.gguf", [
    kv_str("general.architecture", "gemma4"),
    kv_u32("gemma4.context_length", 131072),
    kv_u32("gemma4.block_count", 42),
    kv_u32_array("gemma4.attention.head_count_kv", [16, 16, 4] * 14),   # per-layer, must be READ
    kv_f32("general.sampling.temp", 1.0),
    kv_str("tokenizer.chat_template", GEMMAISH),
])
PROJ = write_gguf("mmproj.gguf", [
    kv_str("general.architecture", "clip"),
    kv_u32("clip.vision.block_count", 27),
])
PICKY_F = write_gguf("picky.gguf", [
    kv_str("general.architecture", "picky"),
    kv_str("tokenizer.chat_template", PICKY),
])
NOTMPL = write_gguf("notmpl.gguf", [kv_str("general.architecture", "bare")])
BADMAGIC = write_gguf("bad.gguf", [kv_str("general.architecture", "x")], magic=b"NOPE")

# ── the parser ─────────────────────────────────────────────────────────────────────────────────
md = gi.read_metadata(QWEN)
check("scalars and strings come back", md["general.architecture"] == "qwen35"
      and md["qwen35.context_length"] == 262144, md.get("general.architecture"))
check("a huge string array is skipped, not built",
      md["tokenizer.ggml.tokens"] is None, type(md["tokenizer.ggml.tokens"]).__name__)
check("...and the key AFTER it is still read correctly — the skip landed on the right byte",
      md["tokenizer.ggml.eos_token_id"] == 248046, md.get("tokenizer.ggml.eos_token_id"))
check("...as is the template, which sits after a second huge array",
      md["tokenizer.chat_template"] == QWENISH)
check("a float keeps its value", abs(md["general.sampling.top_p"] - 0.95) < 1e-6,
      md.get("general.sampling.top_p"))
check("a SHORT array is read, not skipped",
      gi.read_metadata(GEMMA)["gemma4.attention.head_count_kv"][:3] == [16, 16, 4],
      gi.read_metadata(GEMMA)["gemma4.attention.head_count_kv"][:3])
check("every key is accounted for", len(md) == 14, len(md))

# ── the whole inspection ───────────────────────────────────────────────────────────────────────
q = gi.inspect_gguf(QWEN)
check("a Qwen-shaped model reads clean", q["ok"] and q["arch"] == "qwen35", q.get("error"))
check("the trained context comes through", q["context_length"] == 262144, q.get("context_length"))
check("the file's recommended sampling is mapped onto the node's widget names",
      q["sampling"] == {"temperature": 1, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                        "repeat_penalty": 1}, q["sampling"])
eff = q["template"]["effort"]
check("reasoning_effort is recognised as read", eff["read"])
check("...only the values the template accepts survive the probe",
      eff["supported"] == ["low", "medium", "xhigh"], eff["supported"])
check("...and the one matching a bare render is reported as its default",
      eff["default"] == "xhigh", eff["default"])
think = q["template"]["thinking"]
check("enable_thinking is recognised as read", think["read"] and think["differs"])
check("...with unset meaning ON for this family", think["default"] == "on", think["default"])
check("the report names all of it",
      "low, medium, xhigh" in q["report"] and "262 144" in q["report"]
      and "unset means on" in q["report"], q["report"])

g = gi.inspect_gguf(GEMMA)
check("a template that never mentions reasoning_effort is not credited with supporting it",
      g["template"]["effort"]["read"] is False and g["template"]["effort"]["supported"] == [],
      g["template"]["effort"])
check("...and the report says the widget does nothing there",
      "NOT read by this template" in g["report"])
check("the opposite polarity is detected, not assumed",
      g["template"]["thinking"]["read"] and g["template"]["thinking"]["default"] == "off",
      g["template"]["thinking"])

p = gi.inspect_gguf(PROJ)
check("an mmproj is called out rather than treated as a model",
      p["ok"] and p["is_projector"] and "mmproj" in p["report"], p.get("report"))
check("...and its missing template is stated, not crashed on",
      p["template"]["present"] is False)

pk = gi.inspect_gguf(PICKY_F)
check("a template that refuses a system turn is still probed (PROBE_SHAPES falls through)",
      pk["template"]["renders"] is True, pk["template"].get("error"))

n = gi.inspect_gguf(NOTMPL)
check("no template at all is a clean report, not an error", n["ok"] and not n["template"]["present"])
check("...and no recommended sampling says so plainly",
      "no recommended sampling values" in n["report"])

bad = gi.inspect_gguf(BADMAGIC)
check("a file that is not a GGUF fails with a reason", not bad["ok"] and "magic" in bad["error"],
      bad.get("error"))
check("a missing path fails without an exception",
      gi.inspect_gguf(str(TMP / "nope.gguf"))["ok"] is False)
check("an empty path fails without an exception", gi.inspect_gguf("")["ok"] is False)

# ── the cache ──────────────────────────────────────────────────────────────────────────────────
check("the same file is served from cache", gi.inspect_gguf(QWEN) is q)
# Re-downloading a quant can land inside the same second, so the key has to be finer than that.
Path(QWEN).write_bytes(Path(QWEN).read_bytes())      # same size, same second, new mtime_ns
check("...but a file rewritten within the same second is read again",
      gi.inspect_gguf(QWEN) is not q)

# ── the node ───────────────────────────────────────────────────────────────────────────────────
out = gi.LocalLLMModelInfo().run(model=gi.PLACEHOLDER, model_path=QWEN)
check("the node returns the report plus parseable json",
      "qwen35" in out["result"][0] and '"arch": "qwen35"' in out["result"][1],
      out["result"][0].splitlines()[0])
check("...and hands the text to the frontend through ui.text",
      out["ui"]["text"][0] == out["result"][0])
bad_out = gi.LocalLLMModelInfo().run(model=gi.PLACEHOLDER, model_path=str(TMP / "nope.gguf"))
check("a bad pick reports instead of raising", "✕" in bad_out["result"][0], bad_out["result"][0])

print("\n" + ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
