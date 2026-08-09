"""Dream Board — the shot-derivation rule and the node's outputs. Synthetic fixtures only."""
import importlib.util
import json
import tempfile
import types
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import COMFY, PACK, comfy_on_path, fake_package, load_module, load_pack  # noqa: E402

comfy_on_path()

IN = Path(tempfile.mkdtemp()) / "input"
(IN / "kinburg_chat").mkdir(parents=True)
fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: str(IN)
fp.get_temp_directory = lambda: str(IN / "temp")
fp.get_output_directory = lambda: str(IN / "out")
sys.modules["folder_paths"] = fp

# Synthetic parent packages: dream_board does `from ..local_llm.attachments import resolve_refs`,
# and pulling the real morpheus package in would drag torch + the H3 nodes along.
root = types.ModuleType("kp")
root.__path__ = [str(PACK)]
sys.modules["kp"] = root
for sub in ("local_llm", "morpheus"):
    m = types.ModuleType("kp." + sub)
    m.__path__ = [str(PACK / sub)]
    sys.modules["kp." + sub] = m


def load(dotted, rel):
    spec = importlib.util.spec_from_file_location(dotted, str(PACK / rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


fake_nodes = types.ModuleType("kp.morpheus.nodes")
fake_nodes._frames_for = lambda sec: 17 * max(0, round((float(sec) * 24 - 5) / 17)) + 5
fake_nodes._seconds = lambda fr: fr / 24.0
sys.modules["kp.morpheus.nodes"] = fake_nodes

load("kp.local_llm.attachments", "local_llm/attachments.py")
db = load("kp.morpheus.dream_board", "morpheus/dream_board.py")

from PIL import Image

fails = []


def check(label, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + label + (("  " + str(extra)) if extra else ""))
    if not cond:
        fails.append(label)


def pic(name, size=(64, 36)):
    Image.new("RGB", size, (90, 90, 90)).save(IN / "kinburg_chat" / name)
    return {"name": name, "subfolder": "kinburg_chat", "type": "input"}


A, B, C = pic("a.png"), pic("b.png"), pic("c.png")


def msg(text, persona="Mia", role="a", img=None):
    m = {"r": role, "p": persona, "t": text}
    if img:
        m["img"] = img
    return m


def names(shots):
    """Compact view of a plan: per shot, (start, end, [message texts])."""
    out = []
    for s in shots:
        out.append((s["start"]["name"] if s["start"] else None,
                    s["end"]["name"] if s["end"] else None, len(s["msgs"])))
    return out


# ── the rule: pictures define the shots ─────────────────────────────────────────────────────
CHAT = [
    msg("the kitchen at dawn", role="u"),          # 0  before any picture
    msg("here, look", img=[A]),                    # 1  picture 1
    msg("she turns to the window"),                # 2
    msg("light on the counter"),                   # 3
    msg("and now this", img=[B]),                  # 4  picture 2
    msg("she steps outside"),                      # 5
    msg("last one", img=[C]),                      # 6  picture 3
    msg("she walks away"),                         # 7  tail
]
shots, bounds, notes = db.plan_board(CHAT)
check("3 pictures -> 2 bounded shots + a tail", len(shots) == 3, len(shots))
check("boundaries are the pictures in order", [b["name"] for b in bounds] == ["a.png", "b.png", "c.png"],
      [b["name"] for b in bounds])
check("shot 1 spans picture 1 -> picture 2", names(shots)[0][:2] == ("a.png", "b.png"), names(shots)[0])
check("shot 2 spans picture 2 -> picture 3", names(shots)[1][:2] == ("b.png", "c.png"), names(shots)[1])
check("the tail has no keyframes", names(shots)[2][:2] == (None, None), names(shots)[2])
check("messages before the first picture join shot 1", shots[0]["msgs"] == [0, 1, 2, 3],
      shots[0]["msgs"])
check("the message that sends a picture sets up the NEXT shot", shots[1]["msgs"] == [4, 5],
      shots[1]["msgs"])
check("the tail starts on the last picture's message", shots[2]["msgs"] == [6, 7], shots[2]["msgs"])
check("no notes for a clean board", not notes, notes)

# ── excluding things ────────────────────────────────────────────────────────────────────────
s2, b2, _ = db.plan_board(CHAT, noimg=["b.png"])
check("dropping a middle picture merges its two shots into one",
      len(s2) == 2 and names(s2)[0][:2] == ("a.png", "c.png"), names(s2))
check("...and the merged shot keeps both halves' messages", s2[0]["msgs"] == [0, 1, 2, 3, 4, 5],
      s2[0]["msgs"])

s3, _, _ = db.plan_board(CHAT, skip=[2, 3])
check("skipped messages leave the beats", s3[0]["msgs"] == [0, 1], s3[0]["msgs"])
check("...but the shot still exists with its keyframes",
      names(s3)[0][:2] == ("a.png", "b.png"), names(s3)[0])

# a picture whose own message is skipped is still a boundary — that is the persona bubble case,
# where the text is '[фото: …]' and only the picture matters
s4, b4, _ = db.plan_board(CHAT, skip=[1, 4])
check("a skipped message's picture still bounds the shot",
      [b["name"] for b in b4] == ["a.png", "b.png", "c.png"], [b["name"] for b in b4])
check("...and its text is out of the beats", 1 not in s4[0]["msgs"], s4[0]["msgs"])

# ── the degenerate counts ───────────────────────────────────────────────────────────────────
one = [msg("just this", img=[A]), msg("she leaves")]
s5, b5, _ = db.plan_board(one)
check("one picture -> no bounded shot, one text shot", len(s5) == 1 and s5[0]["tail"] is True,
      names(s5))
check("...and the picture is still a keyframe", [b["name"] for b in b5] == ["a.png"])

s6, b6, _ = db.plan_board([msg("all talk"), msg("no pictures", role="u")])
check("no pictures -> one text-only shot", len(s6) == 1 and not b6, (len(s6), b6))

s7, b7, _ = db.plan_board([])
check("an empty chat plans nothing", not s7 and not b7)

s8, _, _ = db.plan_board([msg("", img=[A]), msg("", img=[B])])
check("two pictures with no text still make a shot", len(s8) == 1 and s8[0]["msgs"] == [],
      names(s8))

# ── breaks ──────────────────────────────────────────────────────────────────────────────────
s9, _, n9 = db.plan_board(CHAT, breaks=[7])
check("a break in the tail splits it", len(s9) == 4 and s9[2]["msgs"] == [6] and s9[3]["msgs"] == [7],
      names(s9))
check("...with no complaint", not n9, n9)

s10, _, n10 = db.plan_board(CHAT, breaks=[3])
check("a break inside a keyframed span is ignored", len(s10) == 3, len(s10))
check("...and said so out loud", n10 and "#3" in n10[0], n10)

# ── durations ───────────────────────────────────────────────────────────────────────────────
check("no durations -> the default for every shot",
      db._durations([], 3) == [db.DEFAULT_DURATION] * 3, db._durations([], 3))
check("the last value repeats", db._durations([5, 8], 4) == [5, 8, 8, 8], db._durations([5, 8], 4))
check("extra values are trimmed", db._durations([5, 8, 9], 2) == [5, 8], db._durations([5, 8, 9], 2))
check("junk and zeros are ignored", db._durations(["x", 0, 6], 2) == [6, 6], db._durations(["x", 0, 6], 2))

# ── beats formatting ────────────────────────────────────────────────────────────────────────
check("a line is speaker-labelled", db._line(msg("hello")) == "Mia: hello", db._line(msg("hello")))
check("the user is labelled User", db._line(msg("hi", role="u")) == "User: hi")
check("newlines are collapsed — beats are positional",
      db._line(msg("one\n\ntwo   three")) == "Mia: one two three", db._line(msg("one\n\ntwo   three")))
check("an empty message makes no line", db._line(msg("   ")) == "")

# ── the node end to end ─────────────────────────────────────────────────────────────────────
node = db.KinburgDreamBoard()
state = json.dumps({"v": 1, "msgs": CHAT, "skip": [], "noimg": [], "breaks": [7],
                    "dur": [5.17, 6.58]})
chain, kf, beats, durs, lnks, count, report = node.build(state)
check("keyframes come out as a batch of 3", tuple(kf.shape)[0] == 3, tuple(kf.shape))
check("...in ComfyUI's IMAGE layout", tuple(kf.shape)[1:] == (36, 64, 3), tuple(kf.shape))
check("...normalised 0..1", 0.0 <= float(kf.min()) and float(kf.max()) <= 1.0,
      (float(kf.min()), float(kf.max())))
check("shot_count matches the plan", count == 4, count)
check("one beat line per shot", len(beats.split("\n")) == 4, beats.split("\n"))
check("beats carry the transcript", beats.split("\n")[0].startswith("User: the kitchen at dawn"),
      beats.split("\n")[0])
check("durations are emitted at full length, last repeating",
      durs == "5.17, 6.58, 6.58, 6.58", durs)
check("the report has a row per shot", report.count("\n") >= 5, report)
check("the report names the keyframes", "a" in report and "text only" in report, report[:200])

# an empty board is a clean error, not a traceback
try:
    node.build("")
    check("an empty board raises", False)
except ValueError as e:
    check("an empty board raises a readable error", "Update History" in str(e), str(e)[:60])

# a vanished keyframe file
gone = json.dumps({"msgs": [msg("x", img=[{"name": "ghost.png", "subfolder": "kinburg_chat",
                                           "type": "input"}]),
                            msg("y", img=[A])]})
try:
    node.build(gone)
    check("a missing keyframe raises", False)
except ValueError as e:
    check("a missing keyframe names the file", "ghost.png" in str(e), str(e)[:70])

# mixed sizes get cover-cropped to the first picture, and it is reported
wide = pic("wide.png", (128, 36))
mixed = json.dumps({"msgs": [msg("1", img=[A]), msg("2", img=[wide])]})
_, kf2, _, _, _, _, rep2 = node.build(mixed)
check("a differently shaped picture is cropped to the canvas",
      tuple(kf2.shape) == (2, 36, 64, 3), tuple(kf2.shape))
check("...and the crop is reported", "cropped to" in rep2, rep2[-160:])

# no pictures at all -> keyframes is None, which Storyboard reads as a text-only chain
chain3, kf3, beats3, _, _, count3, _ = node.build(json.dumps({"msgs": [msg("all talk")]}))
check("a text-only board emits no keyframes", kf3 is None)
check("...but still one shot and one beat line", count3 == 1 and beats3 == "Mia: all talk",
      (count3, beats3))

# a truncated / hand-mangled state must not explode
for junk in ("", "{", "[]", '{"msgs": "nope"}', '{"msgs": [1, 2]}'):
    st = db._parse_state(junk)
    check(f"state {junk!r:16} degrades to empty", st["msgs"] == [], st)

print("\n" + ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
