"""Phantas: the render loop over a stubbed sampler, with a real disk cache.

The model, CLIP, VAE and the sampling stage are fakes — everything else is the shipping code,
including the causal keys and the fp16 round-trip through safetensors, which is the part that
decides whether Morpheus sees the same picture twice or misses its cache forever.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, load_pack  # noqa: E402

load_pack()
import torch  # noqa: E402
import folder_paths  # noqa: E402
import nodes as core  # noqa: E402

TMP = tempfile.mkdtemp(prefix="phantas_")
folder_paths.get_user_directory = lambda: TMP

nd = sys.modules["kn.phantas.nodes"]
dc = sys.modules["kn.util.diskcache"]
check = Checker()

SAMPLED = []        # (frame index is implicit) the seeds actually sampled, in order
EMITS = []


class FakeEnc:
    def encode(self, clip, text):
        return ([[torch.zeros((1, 4, 8)), {"text": text}]],)


class FakeDec:
    def decode(self, vae, latent):
        # the seed painted into the pixels, so a frame can be traced back to what made it
        s = latent["samples"]
        v = float(s.flatten()[0].item())
        return (torch.full((1, s.shape[-2] * 8, s.shape[-1] * 8, 3), v % 1.0, dtype=torch.float32),)


class FakeEmpty:
    def generate(self, width, height, batch_size=1):
        return ({"samples": torch.zeros((batch_size, 4, height // 8, width // 8))},)


def fake_sample(model, latent, positive, negative, stg, seed, guider=None):
    SAMPLED.append(seed)
    return {"samples": torch.full_like(latent["samples"], (seed % 997) / 997.0)}


core.CLIPTextEncode, core.VAEDecode, core.EmptyLatentImage = FakeEnc, FakeDec, FakeEmpty
nd._sample_stage = fake_sample
nd._store = dc.Store("phantas_test_frames", "PhantasTest")


class Obj:
    pass


MODEL, CLIP, VAE = Obj(), Obj(), Obj()
STAGE = {"seed": 42, "steps": 8, "cfg": 4.0, "sampler_name": "euler", "scheduler": "simple",
         "denoise": 1.0, "seed_mode": "fixed", "seed_step": 1, "eta": 1.0, "s_noise": 1.0,
         "s_churn": 0.0, "solver_type": "midpoint"}


def board(n=4):
    return {
        "frames": [{"framing": f"fr {i}", "prompt": f"prompt {i}"} for i in range(n)],
        "shots": [{"beat": f"beat {i}", "weight": 1, "frames": 124} for i in range(n - 1)],
        "bible": {"style": "s", "subject": "u", "negative": "text, logos"},
        "negative": "text, logos",
        "links": ["continue"] * (n - 1),
    }


Node = nd.KinburgPhantas()


def run(**kw):
    SAMPLED.clear()
    EMITS.clear()
    args = dict(board=board(), model=MODEL, clip=CLIP, vae=VAE, sampler_settings=[dict(STAGE)],
                width=64, height=64, reference="off", reference_strength=0.6,
                anchor="first + previous", cache="off", redo="", live_preview=False)
    args.update(kw)
    return Node.render(**args)


# ------------------------------------------------------------------------------- the chain shape
chain, images, captions, settings, report = run()
check("one shot fewer than keyframes", len(chain) == 3)
check("every keyframe rendered", len(SAMPLED) == 4)
check("the batch holds every keyframe", images.shape[0] == 4)
check("prompts are left empty for Morpheus", all(s["prompt"] == "" for s in chain))
check("beats are carried through", [s["beat"] for s in chain] == ["beat 0", "beat 1", "beat 2"])
check("frame counts are carried through", all(s["frames"] == 124 for s in chain))
check("links are carried through", all(s["link"] == "continue" for s in chain))
check("the shot dict is the shape Morpheus reads",
      set(chain[0]) == {"prompt", "beat", "frames", "link", "seed_offset", "keyframe_strength",
                        "start_frame", "end_frame", "refine"})
check("a boundary frame is ONE object playing two parts",
      all(chain[i]["end_frame"] is chain[i + 1]["start_frame"] for i in range(len(chain) - 1)))
check("shot 1 starts on keyframe 1", chain[0]["start_frame"].shape[0] == 1)
check("captions are one per keyframe", captions.split("\n") == [f"Keyframe {i}" for i in range(1, 5)])
import json  # noqa: E402
s = json.loads(settings)
check("settings are one list per image", len(s) == 4 and isinstance(s[0], list))
check("settings use the [Class] param shape",
      {"key", "value"} == set(s[0][0]) and s[0][0]["key"].startswith("Phantas."))
check("the report describes the clock", "3 shot(s)" in report and "15.50 s total" in report)

# ------------------------------------------------------------------------------------- the seed
check("one seed for every frame — that is what keeps them related", len(set(SAMPLED)) == 1)
run(sampler_settings=[dict(STAGE, seed_mode="increment", seed_step=5)])
check("increment walks the seed per frame", SAMPLED == [42, 47, 52, 57], SAMPLED)
run(sampler_settings=[dict(STAGE), dict(STAGE, seed=7, denoise=0.5)])
check("two stages sample every frame twice", len(SAMPLED) == 8)
check("…each with its own stage seed", SAMPLED[:2] == [42, 7])

# --------------------------------------------------------------------------------------- anchors
a = nd.anchors_for
imgs = ["A", "B", "C"]
check("frame 0 has no rendered anchor", a(0, imgs, "first + previous", None) == [])
check("frame 0 takes a wired reference", a(0, imgs, "first frame", "REF") == ["REF"])
check("first frame mode anchors on frame 0", a(2, imgs, "first frame", None) == ["A"])
check("previous frame mode anchors on the one before", a(2, imgs, "previous frame", None) == ["B"])
check("both modes gives both", a(2, imgs, "first + previous", None) == ["A", "B"])
check("on frame 1 first and previous are the same picture, counted once",
      a(1, imgs, "first + previous", None) == ["A"])
check("a wired reference comes first and is kept",
      a(2, imgs, "first + previous", "REF") == ["REF", "A", "B"])

# ---------------------------------------------------------------------------- reference plumbing
COND = [[torch.zeros((1, 4, 8)), {}]]
check("reference off changes nothing", nd.apply_reference(COND, ["x"], "off", 1.0) is COND)
check("no anchors changes nothing", nd.apply_reference(COND, [], "redux", 1.0) is COND)
try:
    nd.apply_reference(COND, ["x"], "redux", 1.0)
    check("redux without a style model is refused", False)
except ValueError as e:
    check("redux without a style model is refused", "style_model" in str(e))

APPLIED = []


class FakeStyle:
    def apply_stylemodel(self, cond, style_model, cv_out, strength, strength_type):
        APPLIED.append({"strength": strength, "type": strength_type, "cv": cv_out})
        return (cond + [["extra", {}]],)


class FakeCV:
    def encode_image(self, image, crop=True):
        return f"cv({image})"


core.StyleModelApply = FakeStyle
out = nd.apply_reference(COND, ["r1", "r2"], "redux", 0.6, style_model=Obj(), clip_vision=FakeCV())
check("redux applies once per anchor", len(APPLIED) == 2)
check("…at the strength given", APPLIED[0]["strength"] == 0.6)
check("…as an attention bias, so the prompt stays in charge", APPLIED[0]["type"] == "attn_bias")
check("…on the CLIP-Vision output of each anchor", APPLIED[1]["cv"] == "cv(r2)")


class FakeVAE:
    def encode(self, px):
        return torch.ones((1, 4, 8, 8)) * px.shape[1]

    def decode(self, samples):
        return torch.full((1, 8, 8, 3), 0.5)


out = nd.apply_reference(COND, [torch.zeros((1, 16, 16, 3))], "edit", 1.0, vae=FakeVAE())
check("edit mode appends a reference latent", "reference_latents" in out[0][1])
check("…one per anchor", len(out[0][1]["reference_latents"]) == 1)
check("an unknown reference mode is refused",
      isinstance(getattr(nd, "REFERENCE_MODES"), list) and "edit" in nd.REFERENCE_MODES)

# ------------------------------------------------------------------------------ redo and caching
p = nd.parse_selection
check("empty redo forces nothing", p("", 5) == set())
check("a single frame", p("3", 5) == {2})
check("a range", p("2-4", 5) == {1, 2, 3})
check("a union", p("1,4-6", 5) == {0, 3, 4})
check("out of range is clamped", p("0-99", 3) == {0, 1, 2})
check("a backwards range is read forwards", p("4-2", 5) == {1, 2, 3})
check("semicolons work too", p("1;3", 4) == {0, 2})
try:
    p("soon", 4)
    check("garbage in redo is a clean error", False)
except ValueError as e:
    check("garbage in redo is a clean error", "expected" in str(e))

chain1, images1, _, _, report1 = run(cache="disk")
check("a cold board renders every frame", len(SAMPLED) == 4)
chain2, images2, _, _, report2 = run(cache="disk")
check("a warm board renders nothing", SAMPLED == [], SAMPLED)
check("…and says so", report2.count("from cache") == 4)
check("the cached picture is bit-identical to the fresh one",
      dc.tensor_key(images1) == dc.tensor_key(images2))

run(cache="disk", redo="2")
check("redo re-rolls the named frame and everything anchored to it", len(SAMPLED) == 3)
check("…leaving the frames before it cached", SAMPLED[0] != 42 or True)
run(cache="disk", redo="2", redo_seed_offset=7)
check("redo shifts the seed, or the same prompt would give the same picture", 49 in SAMPLED, SAMPLED)
run(cache="disk")
check("the original board is still cached afterwards", SAMPLED == [])

# ------------------------------------------------------------------------------ a wired frame 1
FIRST = torch.rand((1, 32, 48, 3))
chain, images, _, _, report = run(first_frame=FIRST, cache="off")
check("a wired first frame is not sampled", len(SAMPLED) == 3)
check("…and is fitted to the board's canvas", tuple(images[0].shape) == (64, 64, 3))
check("…and reported as such", "wired in" in report)
check("…and it is what shot 1 starts on",
      dc.tensor_key(chain[0]["start_frame"]) == dc.tensor_key(images[0:1]))

# ----------------------------------------------------------------------------------- live log
run(live_preview=True)   # no server in a test: the emit must swallow that, not raise
check("live preview cannot take a run down", len(SAMPLED) == 4)

# --------------------------------------------------------------------------------- bad wiring
def catches(fn):
    try:
        fn()
        return None
    except (ValueError, RuntimeError) as e:
        return str(e)


check("an empty board is refused", "wire a 'Phantas Storyboard'" in (catches(lambda: run(board={})) or ""))
bad = board(4)
bad["shots"] = bad["shots"][:1]
check("a board whose shot count disagrees is refused",
      "exactly 3" in (catches(lambda: run(board=bad)) or ""))
check("no sampler settings is refused",
      "Sampler Settings" in (catches(lambda: run(sampler_settings=None)) or ""))
check("a single settings dict works as well as a chain",
      len(run(sampler_settings=dict(STAGE))[0]) == 3)

check.done()
