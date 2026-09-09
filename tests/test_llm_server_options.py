"""The server settings against the two command lines they have to become.

Every flag here was read out of the real binaries' own `--help` (llama.cpp b9862 and koboldcpp
1.9x), so this suite is where a rename in either upstream shows up. What it pins:

* **the neutral rule** — a widget left at its default puts NOTHING on the command line, so the
  server keeps its own defaults and a saved workflow does not freeze today's into next year's;
* **the polarity traps** — llama.cpp's Flash Attention is off until asked and koboldcpp's is on
  until refused, and the same widget has to mean the same thing on both;
* **the dialect** — `-ncmoe` vs `--moecpu`, `-ctk/-ctv` vs `--quantkv`, `-b` vs `--blasbatchsize`;
* **what koboldcpp simply has not got**, which the node reports rather than dropping in silence;
* **the embedding split** — `--embeddings` restricts a llama-server process to embedding work, so
  there it is a second command line, while koboldcpp takes `--embeddingsmodel` inline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "llm_server")
o = load_module("kn.llm_server.options", "llm_server/options.py")

check = Checker()

BASE = {"binary": "srv.exe", "model": "chat.gguf", "n_ctx": 8192, "n_gpu_layers": -1}


def llama(**over):
    return o.chat_argv(dict(BASE, flavour="llama-server", **over), 1234)


def kobold(**over):
    return o.chat_argv(dict(BASE, flavour="koboldcpp", **over), 1234)


def pair(argv, flag):
    """The value that follows `flag`, or None when the flag is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


# ------------------------------------------------------------------ the skeleton both need
a = llama()
check("llama-server: model / host / port / ctx / ngl, in that order",
      a[:11] == ["srv.exe", "-m", "chat.gguf", "--host", "127.0.0.1", "--port", "1234",
                 "--ctx-size", "8192", "-ngl", "-1"], a[:11])
b = kobold()
check("koboldcpp: the same five under its own names",
      b[:11] == ["srv.exe", "--model", "chat.gguf", "--host", "127.0.0.1", "--port", "1234",
                 "--contextsize", "8192", "--gpulayers", "-1"], b[:11])
check("nothing else is added when nothing else is set", len(a) == 11 and len(b) == 11, (a, b))

# ------------------------------------------------------------------ the neutral rule
DEFAULTS = dict(flash_attn="auto", kv_cache_type="f16", n_batch=0, n_ubatch=0, threads=0,
                cpu_moe_layers=0, parallel_slots=0, context_shift=True, alias="", api_key="",
                reasoning="model default", reasoning_budget=-1, reasoning_format="auto",
                enable_thinking="model default", reasoning_effort="model default",
                reasoning_effort_custom="", chat_template_file="")
check("a node with every widget at its default emits the bare command line",
      llama(**DEFAULTS) == a, llama(**DEFAULTS)[11:])
check("...on koboldcpp too", kobold(**DEFAULTS) == b, kobold(**DEFAULTS)[11:])

# ------------------------------------------------------------------ flash attention polarity
check("llama-server: flash_attn on is asked for explicitly", pair(llama(flash_attn="on"), "-fa") == "on")
check("llama-server: flash_attn off is asked for too", pair(llama(flash_attn="off"), "-fa") == "off")
check("koboldcpp: 'on' says nothing, because it already is",
      "--noflashattention" not in kobold(flash_attn="on") and len(kobold(flash_attn="on")) == 11)
check("koboldcpp: only 'off' is expressible, as a refusal",
      "--noflashattention" in kobold(flash_attn="off"))

# ------------------------------------------------------------------ the VRAM levers
kv = llama(kv_cache_type="q8_0")
check("llama-server quantizes K and V together", pair(kv, "-ctk") == "q8_0" and pair(kv, "-ctv") == "q8_0")
check("koboldcpp has one knob for both", pair(kobold(kv_cache_type="q8_0"), "--quantkv") == "q8_0")
check("f16 is nobody's flag — it is already the default",
      "-ctk" not in llama(kv_cache_type="f16") and "--quantkv" not in kobold(kv_cache_type="f16"))
check("MoE offload: -ncmoe vs --moecpu",
      pair(llama(cpu_moe_layers=12), "-ncmoe") == "12"
      and pair(kobold(cpu_moe_layers=12), "--moecpu") == "12")
check("batch size: -b vs --blasbatchsize",
      pair(llama(n_batch=1024), "-b") == "1024"
      and pair(kobold(n_batch=1024), "--blasbatchsize") == "1024")
check("ubatch is llama-server's alone",
      pair(llama(n_ubatch=256), "-ub") == "256" and "-ub" not in kobold(n_ubatch=256))
check("slots: -np vs --multiuser",
      pair(llama(parallel_slots=2), "-np") == "2"
      and pair(kobold(parallel_slots=2), "--multiuser") == "2")
check("context shift is only mentioned when refused",
      "--no-context-shift" in llama(context_shift=False)
      and "--noshift" in kobold(context_shift=False)
      and "--no-context-shift" not in llama(context_shift=True))
check("threads share a spelling", pair(llama(threads=8), "--threads") == "8"
      and pair(kobold(threads=8), "--threads") == "8")
check("the api key is a key on one and a password on the other",
      pair(llama(api_key="s3cret"), "--api-key") == "s3cret"
      and pair(kobold(api_key="s3cret"), "--password") == "s3cret")
check("alias is llama-server's alone",
      pair(llama(alias="dolores"), "-a") == "dolores" and "-a" not in kobold(alias="dolores"))

# ------------------------------------------------------------------ thinking
check("reasoning on/off is a switch on llama-server",
      pair(llama(reasoning="on"), "-rea") == "on" and pair(llama(reasoning="off"), "-rea") == "off")
check("koboldcpp can only say 'off', and says it as effort=none",
      pair(kobold(reasoning="off"), "--reasoningeffort") == "none"
      and "--reasoningeffort" not in kobold(reasoning="on"))
check("the thinking budget is a number, and -1 means unrestricted",
      pair(llama(reasoning_budget=512), "--reasoning-budget") == "512"
      and "--reasoning-budget" not in llama(reasoning_budget=-1))
check("budget 0 IS a setting (end thinking at once), not a default",
      pair(llama(reasoning_budget=0), "--reasoning-budget") == "0")
check("reasoning_format is passed only when it is not auto",
      pair(llama(reasoning_format="none"), "--reasoning-format") == "none"
      and "--reasoning-format" not in llama(reasoning_format="auto"))

kw = o.template_kwargs({"enable_thinking": "off", "reasoning_effort": "low"})
check("template variables become one JSON object", kw == {"enable_thinking": False, "reasoning_effort": "low"}, kw)
check("'model default' means the variable is left undefined, which is not the same as false",
      o.template_kwargs({"enable_thinking": "model default"}) == {})
check("a custom effort replaces the dropdown's value",
      o.template_kwargs({"reasoning_effort": "custom", "reasoning_effort_custom": "high"})
      == {"reasoning_effort": "high"})
check("...and an empty custom field sends nothing",
      o.template_kwargs({"reasoning_effort": "custom", "reasoning_effort_custom": ""}) == {})
check("llama-server takes them as --chat-template-kwargs JSON",
      pair(llama(enable_thinking="off", reasoning_effort="low"), "--chat-template-kwargs")
      == '{"enable_thinking":false,"reasoning_effort":"low"}')
check("koboldcpp keeps only the effort level it knows",
      pair(kobold(reasoning_effort="medium"), "--reasoningeffort") == "medium")
check("...and reads 'thinking off' as effort none",
      pair(kobold(enable_thinking="off"), "--reasoningeffort") == "none")
check("...and drops an effort it has no name for (xhigh)",
      "--reasoningeffort" not in kobold(reasoning_effort="xhigh"))
check("the chat template override is llama-server's alone",
      pair(llama(chat_template_file="t.jinja"), "--chat-template-file") == "t.jinja"
      and "--chat-template-file" not in kobold(chat_template_file="t.jinja"))

# ------------------------------------------------------------------ side models
v = dict(mmproj_resolved="proj.gguf", mmproj_offload=True, image_max_tokens=0)
check("vision is one flag when the projector stays on the GPU",
      llama(**v)[11:] == ["--mmproj", "proj.gguf"], llama(**v)[11:])
check("...and two when it is pushed to the CPU",
      "--no-mmproj-offload" in llama(**dict(v, mmproj_offload=False))
      and "--mmprojcpu" in kobold(**dict(v, mmproj_offload=False)))
check("the image token cap is llama-server's alone",
      pair(llama(**dict(v, image_max_tokens=1024)), "--image-max-tokens") == "1024"
      and "--image-max-tokens" not in kobold(**dict(v, image_max_tokens=1024)))
check("no projector means no vision flags at all",
      "--mmproj" not in llama(mmproj_resolved="", mmproj_offload=False))

d = dict(draft_model_resolved="draft.gguf", spec_type="draft-mtp", draft_n_gpu_layers=-1,
         draft_n_max=0, draft_n_min=0, draft_p_min=0.0)
check("a draft model brings its method with it",
      llama(**d)[11:] == ["--spec-type", "draft-mtp", "--spec-draft-model", "draft.gguf"],
      llama(**d)[11:])
check("koboldcpp takes the model and forgets the method",
      kobold(**d)[11:] == ["--draftmodel", "draft.gguf"], kobold(**d)[11:])
full = llama(**dict(d, draft_n_gpu_layers=0, draft_n_max=8, draft_n_min=2, draft_p_min=0.4))
check("every draft knob has a flag",
      pair(full, "--spec-draft-ngl") == "0" and pair(full, "--spec-draft-n-max") == "8"
      and pair(full, "--spec-draft-n-min") == "2" and pair(full, "--spec-draft-p-min") == "0.4", full)
ngram = llama(spec_type="ngram-mod", draft_model_resolved="")
check("an ngram method needs no draft model — it guesses from the text",
      ngram[11:] == ["--spec-type", "ngram-mod"], ngram[11:])
check("but a draft-* method with no model is dropped entirely",
      llama(spec_type="draft-simple", draft_model_resolved="")[11:] == [])

# ------------------------------------------------------------------ embeddings: the split
e = dict(embed_model_resolved="embed.gguf", embed_n_ctx=0, embed_n_gpu_layers=-1,
         pooling="model default", rerank=False)
check("llama-server needs a SECOND process for embeddings",
      o.needs_embed_process(dict(BASE, flavour="llama-server", **e)))
check("...so nothing about them is on the chat command line", llama(**e)[11:] == [])
emb = o.embed_argv(dict(BASE, flavour="llama-server", binary="srv.exe", **e), 9999)
check("the embedding process is embedding-only, on its own port",
      emb == ["srv.exe", "-m", "embed.gguf", "--host", "127.0.0.1", "--port", "9999",
              "--embeddings", "-ngl", "-1"], emb)
emb2 = o.embed_argv(dict(BASE, flavour="llama-server", binary="srv.exe",
                         **dict(e, embed_n_ctx=2048, pooling="mean", rerank=True,
                                embed_extra=("--embd-normalize", "2"))), 9999)
check("...with its own ctx, pooling, rerank and extra flags",
      pair(emb2, "--ctx-size") == "2048" and pair(emb2, "--pooling") == "mean"
      and "--rerank" in emb2 and emb2[-2:] == ["--embd-normalize", "2"], emb2)

check("koboldcpp serves embeddings from the one process",
      not o.needs_embed_process(dict(BASE, flavour="koboldcpp", **e)))
kemb = kobold(**dict(e, embed_n_ctx=1024))
check("...as flags on the chat command line",
      kemb[11:] == ["--embeddingsmodel", "embed.gguf", "--embeddingsmaxctx", "1024",
                    "--embeddingsgpu"], kemb[11:])
check("...and no second command line is built for it",
      o.embed_argv(dict(BASE, flavour="koboldcpp", **e), 9999) == [])
check("embed_n_gpu_layers = 0 keeps it off the GPU there",
      "--embeddingsgpu" not in kobold(**dict(e, embed_n_gpu_layers=0)))

# ------------------------------------------------------------------ saying what is dropped
cfg = dict(BASE, flavour="koboldcpp", **dict(DEFAULTS, **e))
check("defaults are never reported as ignored", o.unsupported(cfg, "koboldcpp") == [],
      o.unsupported(cfg, "koboldcpp"))
check("-1 is neutral for a budget, not a value someone set",
      o.is_neutral("reasoning_budget", -1) and not o.is_neutral("reasoning_budget", 0))
named = o.unsupported(dict(cfg, alias="x", reasoning_format="none", n_ubatch=256), "koboldcpp")
check("what koboldcpp cannot express is named, one line each", len(named) == 3, named)
check("...and each line says which widget and why",
      all(":" in line for line in named)
      and any(line.startswith("alias:") for line in named), named)
check("llama-server has no such list", o.unsupported(dict(cfg, alias="x"), "llama-server") == [])

# ------------------------------------------------------------------ extra_args wins
check("extra_args is appended last, so a hand-written flag overrides a widget",
      llama(kv_cache_type="q8_0", extra=("-ctk", "f16"))[-2:] == ["-ctk", "f16"])

check.done()
