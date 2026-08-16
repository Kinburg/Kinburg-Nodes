"""Every menu path the pack registers, in one place.

A node's ``CATEGORY`` is only where it lands in ComfyUI's Add-Node menu — saved workflows key off
the id in ``NODE_CLASS_MAPPINGS``, so moving a node between folders never breaks a graph. What it
does break is *finding* things, which is why the paths live here rather than as 90 loose string
literals: rename a branch once and the whole pack follows.

The shape:

* ``Bestiary/`` — one folder per node suite (Chimera, Morpheus, Ouroboros, Siren). A suite is never
  split across two folders, so everything named "Siren …" is in exactly one place. The parts that
  several beasts share (``Sampler Settings``) sit at the root of ``Bestiary/`` instead of inside one
  of them, because burying a shared primitive in one suite's folder is how it gets mistaken for that
  suite's private business.
* everything else is grouped by what it does, and nests one level deeper only where a family is big
  enough to be worth collapsing (``flow/accumulators``, ``LLM/context`` …).

``tools/gen_readme_index.py`` audits this: a ``CATEGORY`` that is not one of these constants fails,
and so does a suite whose nodes have drifted into someone else's folder.
"""
ROOT = "Kinburg-Nodes"

# --- the bestiary: one folder per suite, plus the shelf they share ------------------------------
CAT_BESTIARY = ROOT + "/Bestiary"
CAT_CHIMERA = CAT_BESTIARY + "/Chimera"
CAT_MORPHEUS = CAT_BESTIARY + "/Morpheus"
CAT_OUROBOROS = CAT_BESTIARY + "/Ouroboros"
CAT_SIREN = CAT_BESTIARY + "/Siren"

# --- language & vision --------------------------------------------------------------------------
CAT_LLM = ROOT + "/LLM"
CAT_LLM_CONTEXT = CAT_LLM + "/context"
CAT_LLM_PRESETS = CAT_LLM + "/presets"
CAT_LLM_GGUF = CAT_LLM + "/GGUF"

# --- media ---------------------------------------------------------------------------------------
CAT_AUDIO = ROOT + "/audio"
CAT_IMAGE = ROOT + "/image"
CAT_IMAGE_COMPARE = CAT_IMAGE + "/compare"

# --- assets & recipes ------------------------------------------------------------------------------
CAT_MODEL = ROOT + "/model"
CAT_PROMPT = ROOT + "/prompt"
CAT_LORA = ROOT + "/lora"

# --- graph plumbing -------------------------------------------------------------------------------
CAT_FLOW = ROOT + "/flow"
CAT_FLOW_LOOPS = CAT_FLOW + "/loops"
CAT_FLOW_ACCUMULATORS = CAT_FLOW + "/accumulators"
CAT_FLOW_LIST = CAT_FLOW + "/list"

CAT_UTIL = ROOT + "/util"

#: Every path a node may declare. The audit rejects anything else.
ALL = frozenset(v for k, v in list(globals().items()) if k.startswith("CAT_"))

#: package folder → the suite folder its nodes belong in. Nodes from these packages may also sit on
#: the shared ``CAT_BESTIARY`` shelf (that is where ``Sampler Settings`` lives, in ``ouroboros/``),
#: but nowhere else — and no other package may put a node inside a suite folder.
SUITES = {
    "chimera": CAT_CHIMERA,
    "morpheus": CAT_MORPHEUS,
    "ouroboros": CAT_OUROBOROS,
    "siren": CAT_SIREN,
}
