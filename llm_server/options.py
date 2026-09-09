"""Every setting the server nodes offer, and how each backend spells it.

One table, two dialects. `llama-server` and `koboldcpp` do the same things under different flag
names — and sometimes with the opposite polarity (llama.cpp's Flash Attention is `auto` and you
turn it *on*; koboldcpp's is on already and you turn it *off*) — so the mapping is written out once
here rather than sprinkled through the node.

The rule every builder follows: **the neutral value emits nothing**. A widget left alone must not
put a flag on the command line, because the flag llama.cpp does not receive is the one whose
default it is free to change. That is also what keeps a saved workflow from freezing today's
defaults into next year's build.

Options a backend simply does not have are dropped, and :func:`unsupported` names them so the node
can say so out loud instead of silently ignoring a widget the user just set.

Pure string handling: no subprocess, no HTTP, no ComfyUI.
"""
import json

LLAMA_SERVER = "llama-server"
KOBOLDCPP = "koboldcpp"

#: Values that mean "say nothing".
NEUTRAL = ("", "auto", "model default", "default", "none/auto")

#: KV cache types both servers accept (llama.cpp knows more; koboldcpp's list is the narrower one).
KV_CACHE_TYPES = ["f16", "bf16", "q8_0", "q5_1", "q4_0"]

#: `--spec-type` in llama.cpp b9862. koboldcpp has no equivalent — it infers from the draft model.
SPEC_TYPES = ["draft-simple", "draft-mtp", "draft-eagle3", "draft-dflash",
              "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"]

#: `--pooling` on llama-server's embedding process.
POOLING_TYPES = ["model default", "none", "mean", "cls", "last", "rank"]

FLASH_ATTN_MODES = ["auto", "on", "off"]
REASONING_MODES = ["model default", "on", "off"]
REASONING_FORMATS = ["auto", "none", "deepseek", "deepseek-legacy"]


def _num(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def txt(value):
    """A path or name as the user pasted it, quotes and stray spaces removed."""
    return str(value or "").strip().strip('"').strip("'").strip()


_txt = txt


# --------------------------------------------------------------------------- per-option builders
# Each takes (value, flavour) and returns the argv tokens it contributes.

def _flash_attn(v, flavour):
    if v in (None, "", "auto"):
        return []
    if flavour == KOBOLDCPP:
        # koboldcpp enables Flash Attention by default, so only "off" is expressible.
        return ["--noflashattention"] if v == "off" else []
    return ["-fa", str(v)]


def _kv_cache_type(v, flavour):
    if not v or v == "f16":                      # f16 is both servers' default
        return []
    if flavour == KOBOLDCPP:
        return ["--quantkv", str(v)]
    # V-cache quantisation needs Flash Attention; -fa auto turns it on when it can, and asking for
    # it here would override a deliberate `flash_attn = off`.
    return ["-ctk", str(v), "-ctv", str(v)]


def _n_batch(v, flavour):
    n = _num(v)
    if n <= 0:
        return []
    return ["--blasbatchsize", str(n)] if flavour == KOBOLDCPP else ["-b", str(n)]


def _n_ubatch(v, flavour):
    n = _num(v)
    if n <= 0 or flavour == KOBOLDCPP:
        return []
    return ["-ub", str(n)]


def _threads(v, flavour):
    n = _num(v)
    return ["--threads", str(n)] if n > 0 else []


def _cpu_moe_layers(v, flavour):
    n = _num(v)
    if n <= 0:
        return []
    return ["--moecpu", str(n)] if flavour == KOBOLDCPP else ["-ncmoe", str(n)]


def _parallel_slots(v, flavour):
    n = _num(v)
    if n <= 0:
        return []
    return ["--multiuser", str(n)] if flavour == KOBOLDCPP else ["-np", str(n)]


def _context_shift(v, flavour):
    if v is None or bool(v):                      # on is the default on both
        return []
    return ["--noshift"] if flavour == KOBOLDCPP else ["--no-context-shift"]


def _alias(v, flavour):
    v = _txt(v)
    if not v or flavour == KOBOLDCPP:
        return []
    return ["-a", v]


def _api_key(v, flavour):
    v = _txt(v)
    if not v:
        return []
    return ["--password", v] if flavour == KOBOLDCPP else ["--api-key", v]


def _reasoning(v, flavour):
    if v in (None, "", "model default"):
        return []
    if flavour == KOBOLDCPP:
        # Kobold has no on/off switch, only an effort level — and "none" is how you say off there.
        return ["--reasoningeffort", "none"] if v == "off" else []
    return ["-rea", str(v)]


def _reasoning_budget(v, flavour):
    n = _num(v, -1)
    if n < 0 or flavour == KOBOLDCPP:             # -1 = unrestricted, llama.cpp's own default
        return []
    return ["--reasoning-budget", str(n)]


def _reasoning_format(v, flavour):
    if v in (None, "", "auto") or flavour == KOBOLDCPP:
        return []
    return ["--reasoning-format", str(v)]


def _chat_template_file(v, flavour):
    v = _txt(v)
    if not v or flavour == KOBOLDCPP:             # koboldcpp has no jinja-template override
        return []
    return ["--chat-template-file", v]


def template_kwargs(cfg):
    """The `enable_thinking` / `reasoning_effort` pair as llama.cpp's `--chat-template-kwargs`.

    These are **chat-template variables**, not sampler settings: the template decides what to do
    with them, which is why they work on families a prompt directive cannot reach. Same three
    widgets, same meaning, as the Local LLM Settings node.
    """
    out = {}
    think = cfg.get("enable_thinking")
    if think in ("on", "off"):
        out["enable_thinking"] = (think == "on")
    effort = cfg.get("reasoning_effort")
    if effort == "custom":
        effort = _txt(cfg.get("reasoning_effort_custom"))
    if effort and effort not in NEUTRAL:
        out["reasoning_effort"] = effort
    return out


def _template_kwargs(cfg, flavour):
    kw = template_kwargs(cfg)
    if not kw:
        return []
    if flavour == KOBOLDCPP:
        # Only the effort survives, and only the levels Kobold names.
        eff = kw.get("reasoning_effort")
        if kw.get("enable_thinking") is False:
            return ["--reasoningeffort", "none"]
        return ["--reasoningeffort", eff] if eff in ("low", "medium", "high") else []
    return ["--chat-template-kwargs", json.dumps(kw, separators=(",", ":"))]


#: Order matters only for readability of the resulting command line.
_SIMPLE = [
    ("flash_attn", _flash_attn),
    ("kv_cache_type", _kv_cache_type),
    ("n_batch", _n_batch),
    ("n_ubatch", _n_ubatch),
    ("threads", _threads),
    ("cpu_moe_layers", _cpu_moe_layers),
    ("parallel_slots", _parallel_slots),
    ("context_shift", _context_shift),
    ("alias", _alias),
    ("api_key", _api_key),
    ("reasoning", _reasoning),
    ("reasoning_budget", _reasoning_budget),
    ("reasoning_format", _reasoning_format),
    ("chat_template_file", _chat_template_file),
]

#: What each backend cannot express, so the node can report it rather than pretend.
_UNSUPPORTED = {
    KOBOLDCPP: {
        "n_ubatch": "koboldcpp has one batch size (--blasbatchsize), not a separate ubatch",
        "alias": "koboldcpp does not let the API model name be renamed",
        "reasoning_budget": "koboldcpp has no thinking-token budget",
        "reasoning_format": "koboldcpp always leaves the think tags in the reply",
        "chat_template_file": "koboldcpp has no jinja chat-template override",
        "draft_n_min": "koboldcpp drafts a fixed amount (--draftamount)",
        "draft_p_min": "koboldcpp has no draft probability floor",
        "spec_type": "koboldcpp picks the speculative method from the draft model itself",
        "image_max_tokens": "koboldcpp caps vision by pixels (--visionmaxres), not tokens",
        "pooling": "koboldcpp uses the embedding model's own pooling",
    },
    LLAMA_SERVER: {},
}


#: The number that means "leave it alone" for options whose neutral value is not 0.
_NEUTRAL_NUM = {"reasoning_budget": -1, "draft_n_gpu_layers": -1}


def is_neutral(key, value):
    """True when a widget is still at the value that emits no flag — so reporting it as ignored
    would be noise. Numbers need the per-option table: -1 is 'unrestricted' for a budget and
    'all layers' for a layer count, not 'set to minus one'."""
    if value is None or value is False:
        return True
    if isinstance(value, bool):
        return False                              # True is a deliberate choice
    if isinstance(value, str):
        return not value.strip() or value in NEUTRAL
    if isinstance(value, (int, float)):
        return float(value) == float(_NEUTRAL_NUM.get(key, 0))
    return False


def unsupported(cfg, flavour):
    """The options this backend has no word for, among the ones actually set. The node prints
    these: a widget that silently does nothing is worse than one that says so."""
    return [f"{key}: {why}" for key, why in sorted(_UNSUPPORTED.get(flavour, {}).items())
            if key in cfg and not is_neutral(key, cfg[key])]


# --------------------------------------------------------------------------- argv builders
def _vision_args(cfg, flavour):
    mmproj = _txt(cfg.get("mmproj_resolved"))
    if not mmproj:
        return []
    out = ["--mmproj", mmproj]
    if not cfg.get("mmproj_offload", True):
        out += ["--mmprojcpu"] if flavour == KOBOLDCPP else ["--no-mmproj-offload"]
    n = _num(cfg.get("image_max_tokens"))
    if n > 0 and flavour == LLAMA_SERVER:
        out += ["--image-max-tokens", str(n)]
    return out


def _draft_args(cfg, flavour):
    model = _txt(cfg.get("draft_model_resolved"))
    spec = cfg.get("spec_type")
    ngram = isinstance(spec, str) and spec.startswith("ngram")
    if not model and not ngram:
        return []
    out = []
    if flavour == KOBOLDCPP:
        if model:
            out += ["--draftmodel", model]
        n = _num(cfg.get("draft_n_max"))
        if n > 0:
            out += ["--draftamount", str(n)]
        ngl = _num(cfg.get("draft_n_gpu_layers"), -1)
        if ngl >= 0:
            out += ["--draftgpulayers", str(ngl)]
        return out
    if spec and spec not in NEUTRAL:
        out += ["--spec-type", spec]
    if model:
        out += ["--spec-draft-model", model]
    ngl = _num(cfg.get("draft_n_gpu_layers"), -1)
    if ngl != -1:
        out += ["--spec-draft-ngl", str(ngl)]
    for key, flag in (("draft_n_max", "--spec-draft-n-max"), ("draft_n_min", "--spec-draft-n-min")):
        n = _num(cfg.get(key))
        if n > 0:
            out += [flag, str(n)]
    try:
        p = float(cfg.get("draft_p_min") or 0)
    except (TypeError, ValueError):
        p = 0.0
    if p > 0:
        out += ["--spec-draft-p-min", "%g" % p]
    return out


def _embed_inline_args(cfg, flavour):
    """koboldcpp serves embeddings from the SAME process, so they are flags on the chat command
    line. llama-server's `--embeddings` restricts a process to embedding-only, so there it needs a
    second process instead — see :func:`embed_argv`."""
    model = _txt(cfg.get("embed_model_resolved"))
    if not model or flavour != KOBOLDCPP:
        return []
    out = ["--embeddingsmodel", model]
    ctx = _num(cfg.get("embed_n_ctx"))
    if ctx > 0:
        out += ["--embeddingsmaxctx", str(ctx)]
    if _num(cfg.get("embed_n_gpu_layers"), -1) != 0:
        out += ["--embeddingsgpu"]
    return out


def needs_embed_process(cfg):
    """True when the embedding model has to run as a second server of its own."""
    return bool(_txt(cfg.get("embed_model_resolved")) and cfg.get("flavour") != KOBOLDCPP)


def chat_argv(cfg, port):
    """The full command line for the chat server, on its private `port`."""
    flavour = cfg.get("flavour") or LLAMA_SERVER
    kob = flavour == KOBOLDCPP
    argv = [cfg["binary"],
            "--model" if kob else "-m", cfg["model"],
            "--host", "127.0.0.1", "--port", str(int(port)),
            "--contextsize" if kob else "--ctx-size", str(_num(cfg.get("n_ctx"), 4096)),
            "--gpulayers" if kob else "-ngl", str(_num(cfg.get("n_gpu_layers"), -1))]
    for key, build in _SIMPLE:
        argv += build(cfg.get(key), flavour)
    argv += _template_kwargs(cfg, flavour)
    argv += _vision_args(cfg, flavour)
    argv += _draft_args(cfg, flavour)
    argv += _embed_inline_args(cfg, flavour)
    return argv + list(cfg.get("extra", ()))


def embed_argv(cfg, port):
    """The command line for the second, embedding-only llama-server. Empty when none is needed."""
    if not needs_embed_process(cfg):
        return []
    argv = [cfg["binary"], "-m", cfg["embed_model_resolved"],
            "--host", "127.0.0.1", "--port", str(int(port)), "--embeddings"]
    ctx = _num(cfg.get("embed_n_ctx"))
    if ctx > 0:
        argv += ["--ctx-size", str(ctx)]
    argv += ["-ngl", str(_num(cfg.get("embed_n_gpu_layers"), -1))]
    pooling = cfg.get("pooling")
    if pooling and pooling not in NEUTRAL:
        argv += ["--pooling", str(pooling)]
    if cfg.get("rerank"):
        argv += ["--rerank"]
    return argv + list(cfg.get("embed_extra", ()))
