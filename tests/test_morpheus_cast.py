"""Morpheus as a music-video sampler: the song sliced per shot, and the cast attached by name.

Two things are worth pinning, and both are about WHICH, not HOW — H3 itself is faked:

**The clock.** Every shot hears its own slice of the song through Add Guide at frame 0. The slice has
to start where the shot's first frame sits on the WHOLE chain's timeline — after earlier shots' seam
trims and tail trims, and whatever `shots_range` renders — or the mouths drift off the words by a
little more at every seam. The song here is a ramp (each sample holds its own time in seconds), so a
slice's first sample says exactly where it was cut from.

**The cast.** A card is in a shot when the shot's text names it outside quotes, case as written —
and only then are its picture attached and its definition written. `<Picture K>` numbering follows
the wiring order among the attached pictures, because the prompt and the reference list must agree.

The pure prompt module (`morpheus/cast.py`) is checked first; the render loop then runs over a
fake model, VAE and sampler, with the real chain planning around them.
"""
import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, load_pack  # noqa: E402

pack = load_pack("kn_cast")
import torch  # noqa: E402

M = sys.modules["kn_cast.morpheus.nodes"]
C = sys.modules["kn_cast.morpheus.cast"]
check = Checker()


def card(name, kind="character", picture=None, **values):
    return {"name": name, "type": kind, "values": {"name": name, **values}, "picture": picture}


PIC_K, PIC_B, PIC_X = torch.full((1, 8, 16, 3), 0.1), torch.full((1, 8, 16, 3), 0.2), torch.full((1, 8, 8, 3), 0.3)
KEEN = card("Keen", picture=PIC_K, gender="female", hair_color="red", notes="loves bees")
BURG = card("Burg", picture=PIC_B, gender="male", outfit="black T-shirt")
CITY = card("Kharkiv", "entity", description="ruined street, smoke")
CAST = [KEEN, BURG, CITY]

# ------------------------------------------------------------------------------ who is in the shot
names = lambda text: [c["name"] for c in C.present(CAST, text)]  # noqa: E731
check("a named card is in the shot", names("Keen sings in Kharkiv") == ["Keen", "Kharkiv"])
check("...in wiring order, not in the order the text names them",
      names("Kharkiv at dusk, Burg left, Keen right") == ["Keen", "Burg", "Kharkiv"])
check("case counts — 'keen' the English word is not Keen", names("keen fans cheer") == [])
check("a possessive still names her", names("Keen's hair flies") == ["Keen"])
check("a name inside a longer word does not", names("Keenly, Burgundy light") == [])
check("text in quotes doesn't count — the band name on a T-shirt",
      names('Burg wears a T-shirt that says "Keen Burg No Death"') == ["Burg"])
check("...nor the sung line, in any of the usual quote marks",
      names("Keen sings «Burg, come back» and “Kharkiv”, „Burg“") == ["Keen"])
check("a negation still counts — the rule is to name only who is in frame",
      names("Keen sings alone, without Burg") == ["Keen", "Burg"])
check("names with spaces and Cyrillic work too",
      [c["name"] for c in C.present([card("Old Town"), card("Залізо")], "Залізо у Old Town")]
      == ["Old Town", "Залізо"])

# --------------------------------------------------------------------------------- the prompt
prompt, pics = C.assemble("Keen sings, Burg plays guitar.", [KEEN, BURG, CITY])
check("pictures come back in <Picture> order", pics == [PIC_K, PIC_B])
check("the sections are H3's own ref2va layout, in its order",
      [line for line in prompt.split("\n") if re.fullmatch(r"[a-z_]+:", line)]
      == ["subject_definitions:", "summary:", "retention_analysis:", "detailed_description:"], prompt)
check("a sheet is defined the way that stopped the clones",
      "<Subject 1> is Keen, the person in the multi-angle character reference sheet <Picture 1>, "
      "which shows the same single person" in prompt)
check("...with the card's looks as features to follow — and not its notes",
      "Features to follow exactly: Gender: female; Hair color: red." in prompt and "bees" not in prompt)
check("...and told not to bring the sheet's layout along",
      "Do not carry over the backdrop or the multi-view layout of <Picture 1>." in prompt)
check("the second sheet is <Picture 2>", "<Subject 2> is Burg, the person in the multi-angle "
      "character reference sheet <Picture 2>" in prompt)
check("a card without a picture is described in words only",
      "<Subject 3> is Kharkiv. ruined street, smoke" in prompt and "<Picture 3>" not in prompt)
check("each sheet person appears once, as one person",
      prompt.count("appears once, as one person") == 2)
check("the summary names who is in it",
      "[reference generation] A single continuous shot featuring Keen, Burg and Kharkiv." in prompt)
check("the shot's own text is the scene", prompt.endswith("[Shot 1] Keen sings, Burg plays guitar."))

prompt, pics = C.assemble("Burg alone.", [BURG])
check("numbering restarts per shot — Burg alone is <Picture 1>", "sheet <Picture 1>" in prompt and pics == [PIC_B])
prompt, pics = C.assemble("An empty street.", [])
check("a shot with nobody named is just summary + scene",
      pics == [] and prompt.startswith("summary:") and "subject_definitions" not in prompt)
prompt, _ = C.assemble("x", [card("Bar", "entity", picture=PIC_X, description="neon")])
check("an entity with a picture is an appearance anchor",
      "<Subject 1> is Bar, shown in <Picture 1>. Reference role: appearance anchor. neon" in prompt)
prompt, _ = C.assemble("Gritty rock look.\n\n[Shot 1] Keen sings.", [KEEN])
check("a scene that brings its own [Shot 1] keeps what stands before it as the description's opening",
      prompt.endswith("detailed_description:\nGritty rock look.\n\n[Shot 1] Keen sings."), prompt[-80:])

# ------------------------------------------------------------------- roster, as the writer is told
kept, notes = C.roster([KEEN, card(""), dict(KEEN), BURG])
check("the roster keeps named cards once, in wiring order", [c["name"] for c in kept] == ["Keen", "Burg"])
check("...and says why it dropped the others",
      any("no name" in n for n in notes) and any("wired twice" in n for n in notes), notes)
told = C.describe([KEEN, CITY])
check("the writer is told names and what each one is",
      told == "- Keen (a person, female): loves bees\n- Kharkiv (a place or thing): ruined street, smoke", told)
check("...but never their looks — the pictures carry those", "red" not in told)

# ------------------------------------------------------------------- Orpheus' lyrics, per shot
lyr = C.parse_shot_lyrics('1. Keen - "Ви думали, ми розпадемось"\n2. (instrumental - no words)\n'
                          '3. Keen - "Та ми обрали"; Burg - "свою долю"\nnot a row\n4. "no singer here"')
check("a shot's line comes back as (singer, words)", lyr[1] == [("Keen", "Ви думали, ми розпадемось")])
check("an instrumental shot has no words", lyr[2] == [])
check("a duet line keeps both singers, in order", lyr[3] == [("Keen", "Та ми обрали"), ("Burg", "свою долю")])
check("words without a singer are kept", lyr[4] == [("", "no singer here")])
check("anything that isn't a numbered row is ignored", set(lyr) == {1, 2, 3, 4})

# ------------------------------------------------------------------------------- the render loop
h3 = M.h3
FPS = h3.FPS
CALLS = []          # ("r2v" | "i2v" | "guide", info) in call order


class Out:
    def __init__(self, *result):
        self.result = result


class FakeAV:
    """Stands in for the sampled nested latent: knows its frame count, unbinds into two parts."""
    def __init__(self, frames):
        self.frames = frames

    def to(self, *_):
        return self

    def unbind(self):
        return torch.zeros(1, 1, self.frames), None


def _made(prompt, frames):
    """(conditioning, latent) in the shapes the sampler hands on — real enough for
    conditioning_set_values, and a latent that knows how many frames it holds."""
    return [[torch.zeros(1), {"prompt": prompt}]], {"samples": FakeAV(frames)}


class R2V:
    @staticmethod
    def execute(clip, prompt, width, height, length, ref_image_size="match", vae=None, audio_vae=None,
                ref_images=None, **_):
        CALLS.append(("r2v", {"prompt": prompt, "refs": list((ref_images or {}).values()),
                              "size": ref_image_size}))
        return Out(*_made(prompt, length))


class I2V:
    @staticmethod
    def execute(clip, vae, prompt, width, height, length, first_frame=None, last_frame=None):
        CALLS.append(("i2v", {"prompt": prompt, "first": first_frame is not None}))
        return Out(*_made(prompt, length))


class Guide:
    @staticmethod
    def execute(positive, latent, frame_idx, vae=None, audio_vae=None, image=None, audio=None):
        info = {"frame_idx": frame_idx, "image": image is not None}
        if audio is not None:
            w = audio["waveform"]
            info.update(start=float(w[0, 0, 0]), samples=int(w.shape[-1]))
        CALLS.append(("guide", info))
        return Out(positive)


class Guider:
    def __init__(self, m):
        pass

    def set_conds(self, cond):
        pass

    def sample(self, noise, latent, *a, **k):
        return latent


class Noise:
    def __init__(self, seed):
        pass

    def generate_noise(self, latent):
        return None


M.h3 = types.SimpleNamespace(FPS=FPS, align_frame_count=h3.align_frame_count, MAX_PIXELS=h3.MAX_PIXELS,
                             BASE_SHORT_EDGE=h3.BASE_SHORT_EDGE, MiniMaxH3ReferenceToVideo=R2V,
                             MiniMaxH3ImageToVideo=I2V, MiniMaxH3AddGuide=Guide)
M.Guider_Basic, M.Noise_RandomNoise = Guider, Noise


class VAE:
    def decode(self, lat):
        return torch.zeros(lat.shape[-1], 8, 16, 3)


class Model:
    model_options = {"transformer_options": {"minimax_h3_sigma_shift_video": 12.0}}
    load_device = "cpu"
    model = types.SimpleNamespace(latent_format=None)


SR = 2400                                   # 100 samples per frame
SONG_S = 40.0
SONG = {"waveform": (torch.arange(int(SONG_S * SR), dtype=torch.float32) / SR).reshape(1, 1, -1),
        "sample_rate": SR}
Dream = M.KinburgMorpheusDream()


def chain(*prompts, link="cut", seconds=5.17):
    shots = None
    for p in prompts:
        shots, _ = Dream.build(p, seconds, link, 0, shots=shots)
    return shots


def render(shots, **kw):
    CALLS.clear()
    args = dict(shots=shots, model=Model(), clip=object(), vae=VAE(), width=128, height=64, steps=4,
                scheduler="simple", sampler_name="euler", seed=1, shift_video=12.0, shift_audio=3.0,
                seam_trim=1, audio="concat", seam_fade_ms=0, cache="off", cache_tag="", trims="",
                live_preview=False, llm_keep_loaded=False, shots_range="", ref_image_size="max",
                audio_vae=object(), sigmas=torch.tensor([1.0, 0.5, 0.0]), sampler=object())
    args.update(kw)
    return M.KinburgMorpheus().render(**args)


guides = lambda: [c[1] for c in CALLS if c[0] == "guide"]  # noqa: E731
F = h3.align_frame_count(round(5.17 * FPS))              # 124

# three cut shots: each starts where the previous one ended
images, audio, *_ , report, prompts = render(chain("a", "b", "c"), song=SONG)
g = guides()
check("every shot hears the song at its frame 0", [x["frame_idx"] for x in g] == [0, 0, 0])
check("...each from where it sits on the timeline",
      [round(x["start"] * FPS) for x in g] == [0, F, 2 * F], [x["start"] for x in g])
check("...for its full sampled length", all(x["samples"] == F * SR // FPS for x in g))
check("without a cast the shots stay on the plain image-to-video path",
      [c[0] for c in CALLS if c[0] != "guide"] == ["i2v"] * 3)
check("the output track is the song itself, under exactly the rendered frames",
      audio["sample_rate"] == SR and audio["waveform"].shape[-1] == images.shape[0] * SR // FPS
      and float(audio["waveform"][0, 0, 0]) == 0.0)
check("the report says where the sound came from", "audio: the song" in report and "hears 5.17-10.33 s" in report,
      report)

_, model_audio, *_, report, _ = render(chain("a", "b"), song=SONG, song_audio="model")
check("song_audio = model: the track is what the model made, not the song",
      "the model's own audio" in report and "audio: decoded" in report
      and model_audio["sample_rate"] != SR, report)

# continuing shots drop their duplicated first frame; a tail trim moves every later shot's start
render(chain("a", "b", "c", link="continue"), song=SONG, trims="10")
g = guides()
out1 = F - 10                       # shot 1 keeps F - 10 frames
out2 = F - 1 - 9                    # shot 2 loses 1 at the head (the seam), paid back out of its 10 at the tail
check("a continuing shot's frame 0 is the frame before its first kept one",
      [round(x["start"] * FPS) for x in g] == [0, out1 - 1, out1 + out2 - 1], [round(x["start"] * FPS) for x in g])
check("...while the inherited start frame still goes in as the shot's first frame",
      [c[1]["first"] for c in CALLS if c[0] == "i2v"] == [False, True, True])

# shots_range renders shot 2 alone, but on the whole chain's clock
images, audio, *_ = render(chain("a", "b", "c"), song=SONG, shots_range="2")
check("a partial render still hears its own part of the song",
      [round(x["start"] * FPS) for x in guides()] == [F])
check("...and the track starts there too", round(float(audio["waveform"][0, 0, 0]) * FPS) == F)

# a song shorter than the chain
short = {"waveform": SONG["waveform"][..., : int(4.0 * SR)], "sample_rate": SR}
images, audio, *_ , report, _ = render(chain("a", "b"), song=short)
check("a shot that starts after the song ends gets no audio guide", len(guides()) == 1)
check("...says so", "past the end of the song" in report)
check("...and the track is padded with silence to the picture",
      audio["waveform"].shape[-1] == images.shape[0] * SR // FPS
      and float(audio["waveform"][0, 0, -1]) == 0.0)

try:
    render(chain("a"), song=SONG, audio_vae=None)
    check("a song without the audio vae is refused up front", False)
except ValueError as e:
    check("a song without the audio vae is refused up front", "audio_vae" in str(e))

# the cast
shots = chain("Keen sings in Kharkiv.", 'Burg in a T-shirt that says "Keen Burg No Death".', "keen fans cheer.")
_, _, _, _, report, prompts = render(shots, cast_1=KEEN, cast_2=BURG, cast_3=CITY, song=SONG)
r2v = [c[1] for c in CALLS if c[0] == "r2v"]
check("with a cast every shot is built as ref2va", len(r2v) == 3 and not any(c[0] == "i2v" for c in CALLS))
check("shot 1 gets Keen's sheet and Kharkiv in words",
      r2v[0]["refs"] == [PIC_K] and "<Subject 2> is Kharkiv." in r2v[0]["prompt"])
check("shot 2 gets Burg only — the quoted band name doesn't pull Keen in", r2v[1]["refs"] == [PIC_B]
      and "<Subject 1> is Burg" in r2v[1]["prompt"] and "is Keen" not in r2v[1]["prompt"])
check("shot 3 gets nobody — 'keen' is not Keen", r2v[2]["refs"] == [] and "subject_definitions" not in r2v[2]["prompt"])
check("the pictures go in at the chosen size", {x["size"] for x in r2v} == {"max"})
check("the song still rides on frame 0 in ref2va", len(guides()) == 3)
check("the prompts output shows what each shot really got",
      prompts.split("\n\n---\n\n")[0] == r2v[0]["prompt"])
check("the report lists who each shot got", "shot 1: Keen 🖼, Kharkiv" in report and "shot 3: nobody named" in report,
      report)

render(chain("Keen sings.", "Keen turns.", link="continue"), cast_1=KEEN, song=SONG)
check("in ref2va a continuing shot's start frame rides on the frame-0 guide with its song slice",
      [(x["image"], "start" in x) for x in guides()] == [(False, True), (True, True)], guides())

_, _, _, _, report, _ = render(chain("Keen"), cast_1=KEEN, cast_2=dict(KEEN), cast_3=card(""),
                               llm_config={"x": 1})
check("a card wired twice is used once, and said so", "wired twice" in report)
check("a nameless card is skipped, and said so", "has no name" in report)
check("the in-loop writer stands down for a cast, and says why", "in-loop writer is off" in report)
_, _, _, _, report, _ = render(chain("[Scene Overview]: Keen walks."), cast_1=KEEN)
check("a shot written in Morpheus' own template gets a warning", "own template" in report)

# keys: a plain chain must keep the cache keys it always had
render(chain("a"))
check("nothing extra is guided on a plain chain", guides() == [])

# ------------------------------------------------------------------ the Storyboard in cast mode
import tempfile  # noqa: E402

SB = sys.modules["kn_cast.morpheus.storyboard"]
ASKED = []


def fake_ask(cfg, system, user_prompt, images, unload_comfy, tag, emit=None):
    ASKED.append({"tag": tag, "system": system, "prompt": user_prompt})
    if "style bible" in tag:
        return "[STYLE]: Gritty rock-video look.\n[SUBJECT]: x\n[AUDIO BED]: y\n[NEGATIVE]: text", None
    if tag == "script":
        return "[SHOT 1]: one\n[SHOT 2]: two\n[SHOT 3]: three", None
    return ("[SCENE]: Keen sings at the mic in Kharkiv,\n Burg plays behind her.\n"
            "[END STATE]: close on Keen, still singing."), None


SB.KinburgMorpheusStoryboard._ask = staticmethod(fake_ask)
SB.shot_cache.load_json = lambda *a, **k: None
SB.shot_cache.save_json = lambda *a, **k: None
SB.shot_cache.prune = lambda *a, **k: None
LLM = Path(tempfile.mkdtemp()) / "m.gguf"
LLM.write_bytes(b"x")
CFG = {"model": SB.PLACEHOLDER, "model_path": str(LLM), "mmproj": SB.PLACEHOLDER, "mmproj_path": "",
       "system_prompt": "s", "max_tokens": 512, "n_ctx": 4096, "unload_comfy_models": False, "seed": 0}
LYR = ('1. Keen - "Ви думали, ми розпадемось від болю"\n2. (instrumental - no words)\n'
       '3. Keen - "Та ми обрали"; Burg - "свою власну долю"')


def write(**kw):
    a = dict(config=CFG, brief="a rock clip in ruined Kharkiv", shot_count=3, durations="5.17",
             anchor="continuous", link="cut", seed=0, cache="off", unload_after_run="keep loaded",
             script="auto", live_preview=False)
    a.update(kw)
    ASKED.clear()
    return SB.KinburgMorpheusStoryboard().write(**a)


shots_out, prompts, _, report, _ = write(shot_lyrics=LYR, cast_1=KEEN, cast_2=BURG, cast_3=CITY)
calls = {c["tag"].split(" (")[0]: c for c in ASKED}
check("the plan is written by the music-video planner", calls["script"]["system"] == SB.CAST_SCRIPT_SYSTEM)
check("...told the cast by name and what is sung over each shot",
      "- Keen (a person, female)" in calls["script"]["prompt"] and LYR in calls["script"]["prompt"])
check("every shot is written by the cast writer",
      all(c["system"] == SB.CAST_SHOT_SYSTEM for t, c in calls.items() if t.startswith("shot")))
check("shot 1's writer is told Keen sings, so Keen is in frame",
      "Keen sings in this shot, so Keen is in frame" in calls["shot 1/3"]["prompt"])
check("...and gets the words for the performance, not to quote",
      "do not quote them" in calls["shot 1/3"]["prompt"] and "розпадемось" in calls["shot 1/3"]["prompt"])
check("an instrumental shot says nobody sings", "Nobody sings" in calls["shot 2/3"]["prompt"])
check("a duet shot puts both singers in frame", "Keen and Burg sing in this shot" in calls["shot 3/3"]["prompt"])
check("a cut is a new take — the previous end state is not handed over",
      "first frame is fixed" not in calls["shot 2/3"]["prompt"])
check("the writer never sees the cast's looks", "Hair color" not in calls["shot 1/3"]["prompt"])

p = prompts.split("\n\n---\n\n")
check("a shot is the style, then [Shot 1] with the scene on one line",
      p[0].startswith("Gritty rock-video look.\n\n[Shot 1] Keen sings at the mic in Kharkiv, Burg plays behind her."),
      p[0][:120])
check("...and the sung words stamped verbatim after it",
      p[0].endswith('Keen sings: "Ви думали, ми розпадемось від болю"'), p[0][-60:])
check("an instrumental shot gets no words", "sings:" not in p[1])
check("a duet shot stamps each singer's words", p[2].endswith('Keen sings: "Та ми обрали" Burg sings: "свою власну долю"'))
check("no shot is left for the sampler's writer to rework", {s["refine"] for s in shots_out} == {"off"})
check("the report says it wrote for the cast", "cast: Keen, Burg, Kharkiv" in report and "words stamped on 2 of 3" in report,
      report)

# what the sampler then makes of it
render(shots_out, cast_1=KEEN, cast_2=BURG, cast_3=CITY)
r2v = [c[1] for c in CALLS if c[0] == "r2v"]
check("the sampler opens the description with the style and keeps the writer's [Shot 1]",
      "detailed_description:\nGritty rock-video look.\n\n[Shot 1] Keen sings" in r2v[0]["prompt"])
check("...and the stamped singer is what attaches her sheet", r2v[0]["refs"][0] is PIC_K)

_, prompts, _, report, _ = write(shot_lyrics=LYR, cast_1=KEEN, cast_2=BURG, cast_3=CITY, link="continue")
check("a continuing shot is handed the previous shot's end state",
      "first frame is fixed" in [c for c in ASKED if c["tag"].startswith("shot 2/3")][0]["prompt"])

_, _, _, report, _ = write(shot_lyrics='1. Gru - "hey"', cast_1=KEEN)
check("a singer with no card of that name is reported", "Gru sing(s)" in report)
_, _, _, report, _ = write(cast_1=KEEN)
check("a cast without lyrics is reported", "no 'shot_lyrics'" in report)

# ------------------------------------------------------------------------------- locations
ROOF = card("Roof", "entity", description="a flat roof at dusk — for the choruses")
STREET = card("Street", "entity", description="the ruined street")
check("in order: the clip's length is shared out evenly",
      C.places_in_order([100] * 6, ["cut"] * 6, 3) == [0, 0, 1, 1, 2, 2])
check("...by length, not by shot count", C.places_in_order([300, 100, 100, 100], ["cut"] * 4, 2) == [0, 1, 1, 1])
check("...and a continuing shot stays where the previous one was",
      C.places_in_order([100] * 4, ["cut", "cut", "continue", "cut"], 2) == [0, 0, 0, 1])
check("a plan line's bracket tag picks the place and is taken off the line",
      C.pick_place("(roof) Keen sings.", [STREET, ROOF]) == (ROOF, "Keen sings."))
check("...a place named in the line counts too", C.pick_place("Keen walks down Street.", [STREET, ROOF])[0] is STREET)
check("...and a line that names none says so", C.pick_place("(Moon) Keen sings.", [STREET, ROOF])[0] is None)
check("a section label in the lyrics doesn't disturb the words",
      C.parse_shot_lyrics('2. [Chorus] Keen - "hey"')[2] == [("Keen", "hey")])


def planned(tag, system, user_prompt, default):
    if tag == "script":
        return "[SHOT 1]: (Street) one\n[SHOT 2]: two, no place named\n[SHOT 3]: (Roof) three", None
    return default


BASE_ASK = fake_ask
SB.KinburgMorpheusStoryboard._ask = staticmethod(
    lambda cfg, system, user_prompt, images, unload_comfy, tag, emit=None:
    planned(tag, system, user_prompt, BASE_ASK(cfg, system, user_prompt, images, unload_comfy, tag, emit)))
_, prompts, _, report, script = write(shot_lyrics=LYR, cast_1=KEEN, cast_2=BURG,
                                      location_1=STREET, location_2=ROOF)
p = prompts.split("\n\n---\n\n")
plan_call = [c for c in ASKED if c["tag"] == "script"][0]["prompt"]
check("by section: the planner is given the places and told to keep one per kind of section",
      "(a place or thing): a flat roof at dusk — for the choruses" in plan_call and "every chorus" in plan_call)
shot_calls = [c["prompt"] for c in ASKED if c["tag"].startswith("shot")]
check("...each writer is told its one place, and the tag is off its direction",
      "takes place in Street" in shot_calls[0] and "Direction for this shot: one" in shot_calls[0])
check("...a line that names no place stays where the previous shot was", "takes place in Street" in shot_calls[1])
check("...and the tagged line moves to the new place", "takes place in Roof" in shot_calls[2])
check("a scene that forgot its place gets it put in front — that is what attaches the picture",
      "[Shot 1] In Street. Keen sings at the mic" in p[0] and "[Shot 1] In Roof. " in p[2], p[2][:80])
check("the script output keeps the tags, so pasting it back as beats keeps the places", "(Roof) three" in script)
check("the report counts the shots per place", "locations (by section): Street ×2, Roof ×1" in report, report)

_, prompts, _, report, _ = write(shot_lyrics=LYR, cast_1=KEEN, location_1=STREET, location_2=ROOF,
                                 locations="in order", shot_count=4)
check("in order: no choosing — two places over four equal shots",
      "locations (in order): Street ×2, Roof ×2" in report, report)
check("...and the planner is told where each shot is",
      "shot 3: Roof" in [c for c in ASKED if c["tag"] == "script"][0]["prompt"])
_, _, _, report, _ = write(location_1=STREET)
check("locations without a cast are ignored, and said so", "without a cast" in report)

SEC = '1. (instrumental - no words)\n2. [Verse 1] Keen - "a"\n3. [Verse 1] Keen - "b"\n4. [Chorus] Burg - "c"'
check("by section: a shot continues while the section holds and cuts where it changes",
      C.links_by_section(SEC, 1, 4) == ["cut", "cut", "continue", "cut"])
check("...counted from the first appended shot", C.links_by_section(SEC, 3, 2) == ["continue", "cut"])
check("...and wordless rows in a row are one take", C.links_by_section("1. (instrumental)\n2. (instrumental)", 1, 2) == ["cut", "continue"])
_, _, _, report, _ = write(shot_lyrics=SEC, cast_1=KEEN, cast_2=BURG, link="by section", shot_count=4)
check("link = by section: the report shows the links it derived", "cut, cut, continue, cut" in report, report)
plan_call = [c for c in ASKED if c["tag"] == "script"][0]["prompt"]
check("...the planner is told which shots carry on, and to vary the framing on a cut",
      "Shots 3 carry on" in plan_call and "differ clearly" in plan_call)
shot_calls = [c["prompt"] for c in ASKED if c["tag"].startswith("shot")]
check("...a continuing shot is told not to reverse the camera", "never reverses" in shot_calls[2], shot_calls[2])
check("...and a cut is shown where the last shot ended, to open differently",
      "starts on a cut" in shot_calls[3] and "starts on a cut" not in shot_calls[0], shot_calls[3])
_, _, _, report, _ = write(shot_lyrics=SEC, cast_1=KEEN, link="by section", links="cut", shot_count=4)
check("a filled 'links' still wins over by section", "open shots: cut" in report, report)
SB.KinburgMorpheusStoryboard._ask = staticmethod(BASE_ASK)

_, prompts, _, _, _ = write()
check("without a cast the Storyboard writes its own template, exactly as before",
      ASKED[1]["system"] == SB.SCRIPT_SYSTEM and "6. [Negative Prompt/Constraints]" in prompts)

check.done()
