"""What a .gguf says about itself — read from its header, without loading a single weight.

A GGUF carries more than the tensors: the architecture, the context length it was trained for,
often the sampling values its authors recommend, and the **chat template**. The template is the
interesting one, because the reasoning controls of a modern model are template *variables* — so the
file itself is the authority on which of them exist and what they accept, and nothing here has to
be hardcoded per family.

Two ways of asking, both cheap and both honest:

  * **the template's AST** — ``jinja2.meta.find_undeclared_variables`` gives the exact set of
    variables the template reads. If ``reasoning_effort`` is not in it, the model has no effort
    control and the widget for it is dead weight on that model. That is a fact, not a guess.
  * **rendering it** — with a candidate value substituted, and the render either raises or doesn't.
    Qwen3.5/3.8's template validates the value itself (``raise_exception`` on anything outside
    xhigh / medium / low), so the accepted set falls out of trying them. The candidate whose render
    matches the no-variable render is the model's own default.

Only the KV block at the very front of the file is read — a few hundred kilobytes, never the tensor
index and never a weight. That matters more than it sounds: `gguf.GGUFReader` takes **7 seconds** on
a 17 GB model, all of it spent building views over 866 tensors this has no use for, and a dropdown
that stalls for seven seconds is not a dropdown. Results are also cached per (path, mtime, size).
"""
import json
import os
import struct

from .llm_node import PLACEHOLDER, _list_models, _resolve_path
from ..categories import CAT_LLM

# Effort values worth trying. A superset on purpose: which ones survive is the answer, and a family
# nobody has shipped yet gets tested for free as long as it names its levels from this vocabulary.
EFFORT_CANDIDATES = ("minimal", "none", "low", "medium", "high", "xhigh", "max")

# Recommended sampling values, when the file carries them -> the widget they belong to on the
# Settings node. llama-cpp-python never reads these itself; the node always passes its own values,
# which is why applying them has to be an explicit action rather than a silent default.
SAMPLING_KEYS = {
    "general.sampling.temp": "temperature",
    "general.sampling.top_p": "top_p",
    "general.sampling.top_k": "top_k",
    "general.sampling.min_p": "min_p",
    "general.sampling.penalty_repeat": "repeat_penalty",
}

_CACHE = {}

GGUF_MAGIC = b"GGUF"
# GGUF value type id -> (struct format, byte width). Strings (8) and arrays (9) are their own shapes.
_SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
            6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_STRING, _ARRAY = 8, 9
# An array longer than this is walked past rather than built: `tokenizer.ggml.tokens` is a quarter
# of a million strings and nothing here wants it. Per-layer arrays (head_count_kv) are far shorter
# and do come through.
_ARRAY_MAX = 4096


def _rd(f, fmt, size):
    buf = f.read(size)
    if len(buf) < size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, buf)[0]


def _rd_str(f):
    return f.read(_rd(f, "<Q", 8)).decode("utf-8", "replace")


def _skip(f, elem_type, count):
    """Walk past an array without materialising it."""
    if elem_type in _SCALARS:
        f.seek(_SCALARS[elem_type][1] * count, os.SEEK_CUR)
    elif elem_type == _STRING:
        for _ in range(count):
            f.seek(_rd(f, "<Q", 8), os.SEEK_CUR)
    else:
        raise ValueError(f"cannot skip array of type {elem_type}")


def _rd_typed(f, t):
    if t == _STRING:
        return _rd_str(f)
    if t == _ARRAY:
        elem_type = _rd(f, "<I", 4)
        count = _rd(f, "<Q", 8)
        if count > _ARRAY_MAX:
            _skip(f, elem_type, count)
            return None                      # present, deliberately not read
        return [_rd_typed(f, elem_type) for _ in range(count)]
    if t not in _SCALARS:
        raise ValueError(f"unknown GGUF value type {t}")
    return _rd(f, *_SCALARS[t])


def read_metadata(path):
    """The GGUF key/value block, as ``{key: value}``. Stops before the tensor index.

    Hand-rolled rather than delegated because the KV block is all of 57 keys and sits in the first
    pages of the file, while the library reader's cost scales with the tensor count.
    """
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError("not a GGUF file (bad magic)")
        version = _rd(f, "<I", 4)
        if version not in (1, 2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        _rd(f, "<Q", 8)                      # tensor count — the index itself is never read
        out = {}
        for _ in range(_rd(f, "<Q", 8)):
            key = _rd_str(f)
            out[key] = _rd_typed(f, _rd(f, "<I", 4))
        return out


def _num(v):
    """GGUF numbers arrive as numpy scalars; round the floats so 0.949999988 reads as 0.95."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) and abs(f) >= 1 else round(f, 4)



def _environment():
    """The same Jinja setup llama-cpp-python renders chat templates in, so what is probed here is
    what would actually happen at generation time (llama_chat_format.py:222)."""
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(loader=jinja2.BaseLoader(), trim_blocks=True,
                                        lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    return env


# Templates disagree about what a minimal conversation looks like — Qwen refuses one with no user
# turn, some refuse a system message, some want alternating roles. Try the shapes in order and use
# the first that renders; a template none of them satisfies simply isn't probed.
PROBE_SHAPES = (
    [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
    [{"role": "user", "content": "U"}],
    [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"},
     {"role": "user", "content": "U2"}],
)


def _probe(template, eos, bos):
    """Render the template with and without the reasoning variables; report what it accepts.

    Returns a dict describing the template — never raises, because an exotic template failing to
    render is a thing to report, not a reason for the node to die.
    """
    from jinja2 import meta
    env = _environment()
    out = {"present": True, "vars": [], "renders": False, "error": "",
           "effort": {"read": False, "supported": [], "default": None},
           "thinking": {"read": False, "default": None, "differs": False}}
    try:
        ast = env.parse(template)
        out["vars"] = sorted(meta.find_undeclared_variables(ast))
        tmpl = env.from_string(template)
    except Exception as e:
        out["error"] = f"template will not parse: {e}"
        return out

    def raise_exception(msg):
        raise ValueError(msg)

    def render(messages, **kw):
        return tmpl.render(messages=messages, add_generation_prompt=True, eos_token=eos,
                           bos_token=bos, raise_exception=raise_exception,
                           strftime_now=lambda f: "", **kw)

    base, msgs = None, None
    for shape in PROBE_SHAPES:
        try:
            base = render(shape)
            msgs = shape
            break
        except Exception as e:
            out["error"] = str(e)
    if msgs is None:
        return out
    out["renders"], out["error"] = True, ""

    # The AST is what decides whether a variable means anything here. Probing a template that never
    # mentions `reasoning_effort` would "accept" every value, which reads as support but is silence.
    if "reasoning_effort" in out["vars"]:
        out["effort"]["read"] = True
        for cand in EFFORT_CANDIDATES:
            try:
                got = render(msgs, reasoning_effort=cand)
            except Exception:
                continue
            out["effort"]["supported"].append(cand)
            if got == base and out["effort"]["default"] is None:
                out["effort"]["default"] = cand
    if "enable_thinking" in out["vars"]:
        rendered = {}
        for flag in (True, False):
            try:
                rendered[flag] = render(msgs, enable_thinking=flag)
            except Exception:
                rendered[flag] = None
        out["thinking"]["read"] = True
        out["thinking"]["differs"] = rendered.get(True) != rendered.get(False)
        for flag in (True, False):
            if rendered.get(flag) == base:
                out["thinking"]["default"] = "on" if flag else "off"
                break
    return out


def inspect_gguf(path):
    """Everything this file says about itself, as a dict. ``{"ok": False, "error": …}`` on failure."""
    path = (path or "").strip()
    if not path or not os.path.isfile(path):
        return {"ok": False, "error": f"not a file: {path or '(none)'}"}
    try:
        st = os.stat(path)
        # st_mtime_ns, not int(st_mtime): whole seconds would serve a stale answer for a file
        # replaced within the same second — which is exactly what re-downloading a quant looks like.
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    except OSError as e:
        return {"ok": False, "error": str(e)}
    if key in _CACHE:
        return _CACHE[key]

    try:
        meta_fields = read_metadata(path)
    except Exception as e:
        return {"ok": False, "error": f"cannot read as GGUF ({e}) — a truncated or still-"
                                      f"downloading file lands here too."}
    try:
        arch = str(meta_fields.get("general.architecture") or "")
        template = meta_fields.get("tokenizer.chat_template")
        info = {
            "ok": True,
            "path": path,
            "file": os.path.basename(path),
            "name": str(meta_fields.get("general.name") or ""),
            "arch": arch,
            "size_label": str(meta_fields.get("general.size_label") or ""),
            "context_length": _num(meta_fields.get(f"{arch}.context_length")),
            "n_layers": _num(meta_fields.get(f"{arch}.block_count")),
            "rope_scaling": str(meta_fields.get(f"{arch}.rope.scaling.type") or ""),
            # An mmproj carries clip.* keys and no block_count of its own — worth saying out loud,
            # since picking one as the `model` is an easy and confusing mistake.
            "is_projector": any(k.startswith("clip.") for k in meta_fields),
            "sampling": {},
            "keys": len(meta_fields),
        }
        for key_name, widget in SAMPLING_KEYS.items():
            v = _num(meta_fields.get(key_name))
            if v is not None:
                info["sampling"][widget] = v
        if isinstance(template, str) and template.strip():
            # Placeholder eos/bos: the real strings live in a 250k-entry array this deliberately
            # walks past, and the probe only ever compares renders against each other — a template
            # that merely emits a token text cannot change which VARIABLES it reads.
            info["template"] = _probe(template, "<eos>", "<bos>")
        else:
            info["template"] = {"present": False, "vars": [], "renders": False,
                                "error": "this GGUF ships no chat template",
                                "effort": {"read": False, "supported": [], "default": None},
                                "thinking": {"read": False, "default": None, "differs": False}}
        info["report"] = build_report(info)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    _CACHE[key] = info
    return info


def build_report(info):
    """The human-readable rendering — the node's output and the console line."""
    t = info.get("template") or {}
    eff, think = t.get("effort") or {}, t.get("thinking") or {}
    lines = [f"Local LLM Model Info — {info.get('file') or '(none)'}"]
    head = [x for x in (info.get("arch"), info.get("size_label"),
                        f"{info['n_layers']} layers" if info.get("n_layers") else None) if x]
    if head:
        lines.append("  " + " · ".join(str(x) for x in head))
    if info.get("is_projector"):
        lines.append("  ⚠ this looks like an mmproj (vision projector), not a model — it belongs "
                     "in Vision Settings' 'mmproj', not in 'model'")
    if info.get("context_length"):
        lines.append(f"  trained context: {info['context_length']:,} tokens".replace(",", " ")
                     + (f" · rope scaling {info['rope_scaling']}" if info.get("rope_scaling") else "")
                     + "  (n_ctx costs VRAM — raise it deliberately, not to the maximum)")
    if info.get("sampling"):
        lines.append("  the file's own recommended sampling: "
                     + " · ".join(f"{k} {v}" for k, v in info["sampling"].items()))
    else:
        lines.append("  no recommended sampling values in this file (many GGUFs carry none)")

    if not t.get("present"):
        lines.append("  chat template: NONE — set chat_template_path on the Settings node")
        return "\n".join(lines)
    if not t.get("renders"):
        lines.append(f"  chat template: present but would not render here ({t.get('error')})")
        return "\n".join(lines)

    if eff.get("read"):
        sup = eff.get("supported") or []
        lines.append("  reasoning_effort: " + (", ".join(sup) if sup else "read but accepts nothing?")
                     + (f"  (its default: {eff['default']})" if eff.get("default") else ""))
    else:
        lines.append("  reasoning_effort: NOT read by this template — the widget does nothing here")
    if think.get("read"):
        note = f"leaving it unset means {think['default']}" if think.get("default") else \
            "unset matches neither on nor off"
        lines.append("  enable_thinking: read"
                     + ("" if think.get("differs") else " but on/off render the same")
                     + f" · {note}")
    else:
        lines.append("  enable_thinking: NOT read by this template — the widget does nothing here")
    extra = [v for v in (t.get("vars") or [])
             if v in ("tools", "preserve_thinking", "add_vision_id", "response_declaration",
                      "reasoning_content", "thinking_text")]
    if extra:
        lines.append("  other variables it reads: " + ", ".join(extra))
    return "\n".join(lines)


class LocalLLMModelInfo:
    """Read a .gguf's own metadata and chat template and report what it supports."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_models(), {"tooltip": "The .gguf to inspect, from ComfyUI/models/llm. Nothing is loaded — only the file's header is read, so this costs no VRAM and works on a model you have no room for."}),
                "model_path": ("STRING", {"default": "", "tooltip": "Full path to a .gguf, used when 'model' is the placeholder. Surrounding quotes are stripped."}),
            },
        }

    # Combos are baked when /object_info is served, so a model added since then must still validate.
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "json")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT_LLM
    DESCRIPTION = ("What a .gguf says about itself: architecture, trained context length, the "
                   "sampling values its authors recommend, and — read out of its embedded chat "
                   "template — which reasoning controls it actually honours and what "
                   "reasoning_effort accepts. Loads no weights and touches no VRAM.")

    def run(self, model=PLACEHOLDER, model_path=""):
        resolved = _resolve_path(model, model_path)
        info = inspect_gguf(resolved)
        if not info.get("ok"):
            text = f"Local LLM Model Info — ✕ {info.get('error')}"
        else:
            text = info["report"]
        print("[Model Info] " + text.replace("\n", "\n[Model Info] "))
        return {"ui": {"text": [text]},
                "result": (text, json.dumps(info, ensure_ascii=False, indent=2, default=str))}


# The route the Settings node's frontend calls to narrow its own dropdowns to what the picked model
# understands. Guarded like every other route in this pack, so the package still imports headless.
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/kinburg/llm/gguf_info")
    async def _gguf_info(request):
        resolved = _resolve_path(request.query.get("model") or PLACEHOLDER,
                                 request.query.get("model_path") or "")
        info = inspect_gguf(resolved)
        return web.json_response(info, status=200 if info.get("ok") else 404)
except Exception as e:  # pragma: no cover
    print(f"[LocalLLM] could not register the gguf-info route: {e}")


NODE_CLASS_MAPPINGS = {"LocalLLMModelInfo": LocalLLMModelInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"LocalLLMModelInfo": "Local LLM Model Info 🔍"}
