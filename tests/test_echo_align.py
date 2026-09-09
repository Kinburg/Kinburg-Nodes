"""The parts of Echo's aligner that do not need weights: the encode-window cap and the word score.

`align.py` imports torch only inside its functions, so the two pieces of arithmetic in it are
testable on their own — and they are the two that can silently lose data. A window split that drops
a line loses its timing entirely, and it does so quietly, on exactly the long sections nobody
notices until a subtitle is missing three minutes in.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "context", "echo", "siren", "timer", "util")
load_module("kn.util.anytype", "util/anytype.py")
load_module("kn.context.character_card", "context/character_card.py")
load_module("kn.timer.timer_nodes", "timer/timer_nodes.py")
load_module("kn.siren.cast", "siren/cast.py")
load_module("kn.siren.score", "siren/score.py")
load_module("kn.echo.translit", "echo/translit.py")
TR = load_module("kn.echo.track", "echo/track.py")
A = load_module("kn.echo.align", "echo/align.py")

check = Checker()


def block(window, n_lines, chars=20):
    return {"label": "Verse 1", "window": window,
            "lines": [{"text": "x" * chars,
                       "words": [{"text": "x" * chars, "key": "x"}]} for _ in range(n_lines)]}


# ------------------------------------------------------------------- a short block is not split
short = block((10.0, 40.0), 6)
parts = A._split_window(short, A.MAX_WINDOW)
check("a window inside the cap is left alone", len(parts) == 1 and parts[0][1] == (10.0, 40.0),
      parts[0][1] if parts else parts)
check("...and it keeps every line", len(parts[0][0]) == 6)


# ------------------------------------------------------------------------- a long one is split
long_block = block((0.0, 140.0), 12)
parts = A._split_window(long_block, 45.0)
check("a 140 s window becomes four", len(parts) == 4, len(parts))

# The failure that would be silent: a line in no window is a line with no timing.
seen = [ln for group, _ in parts for ln in group]
check("every line is in exactly one part",
      len(seen) == 12 and all(sum(ln is x for x in seen) == 1 for ln in long_block["lines"]),
      len(seen))
check("no part is empty", all(group for group, _ in parts))
check("the parts stay inside the block's own window",
      all(0.0 <= lo < hi <= 140.0 for _, (lo, hi) in parts), [w for _, w in parts])
check("they run in order", all(parts[i][1][0] <= parts[i + 1][1][0] for i in range(len(parts) - 1)),
      [w for _, w in parts])
# A word sitting on a split point has to be inside SOME window, so the seams overlap.
check("consecutive parts overlap at the seam",
      all(parts[i][1][1] > parts[i + 1][1][0] for i in range(len(parts) - 1)),
      [w for _, w in parts])
check("no part exceeds the cap by more than the seam overlap",
      all((hi - lo) <= 45.0 + 2.5 for _, (lo, hi) in parts),
      [round(hi - lo, 2) for _, (lo, hi) in parts])

# Splitting is by how much TEXT there is, not by line count: a long line takes longer to sing.
lopsided = {"label": "V", "window": (0.0, 120.0), "lines": [
    {"text": "x" * 400, "words": [{"text": "x" * 400, "key": "x"}]},
    {"text": "x", "words": [{"text": "x", "key": "x"}]},
    {"text": "x", "words": [{"text": "x", "key": "x"}]},
]}
lop = A._split_window(lopsided, 45.0)
check("a line that dwarfs the others gets a part of its own",
      len(lop[0][0]) == 1 and sum(len(g) for g, _ in lop) == 3, [len(g) for g, _ in lop])

# The degenerate shapes: one line cannot be split, and a zero-length window must not divide by zero.
check("a single line is never split", len(A._split_window(block((0.0, 200.0), 1), 45.0)) == 1)
check("a zero-width window does not raise",
      len(A._split_window(block((5.0, 5.0), 3), 45.0)) == 1)


# -------------------------------------------------------------------------- the word confidence
class Span:
    """A stand-in for `torchaudio.functional.TokenSpan`."""

    def __init__(self, start, end, score):
        self.start, self.end, self.score = start, end, score

    def __len__(self):
        return self.end - self.start


check("one token's score is its own", abs(A._score([Span(0, 4, 0.8)]) - 0.8) < 1e-9)
# Length-weighted, so a held vowel the model is sure of is not outvoted by a one-frame consonant.
mixed = A._score([Span(0, 9, 0.9), Span(9, 10, 0.1)])
check("the score is weighted by how long each token was held", abs(mixed - 0.82) < 1e-9, mixed)
check("a plain mean would have said 0.50 — it does not", abs(mixed - 0.5) > 0.3)
check("no tokens is 0.0 rather than a division by zero", A._score([]) == 0.0)


# ------------------------------------------------------ the window is caught placing the words
# The failure the author hit: the plan says a section is at 1:20, the singer came in at 1:35, the
# window stopped at 1:28. Forced alignment MUST place every word it is given, so it crams them
# against the last second it is allowed — early, over whoever is really singing, with silence where
# the words belong. From inside the window that is indistinguishable from a correct answer, so it is
# caught by geometry: words flat against an edge are words the window chose the position of.
def placed(pairs):
    return [({}, a, b, 0.7) for a, b in pairs]


check("words hard against the low edge are pinned",
      A._pinned(placed([(20.05, 21.0), (21.0, 22.0)]), 20.0, 40.0))
check("words hard against the high edge are pinned",
      A._pinned(placed([(38.0, 39.0), (39.0, 39.95)]), 20.0, 40.0))
check("words sitting comfortably inside are not",
      not A._pinned(placed([(25.0, 26.0), (30.0, 31.0)]), 20.0, 40.0))
check("the edge tolerance is a fraction of a second, not zero",
      A._pinned(placed([(20.3, 21.0)]), 20.0, 40.0) and
      not A._pinned(placed([(20.5, 21.0)]), 20.0, 40.0), A.PIN_EDGE)
check("nothing placed cannot be pinned", not A._pinned([], 0.0, 10.0))
check("the retry widens by a real amount",
      max(A.PIN_WIDEN_MIN, 30.0 * A.PIN_WIDEN) >= 18.0,
      max(A.PIN_WIDEN_MIN, 30.0 * A.PIN_WIDEN))

check("confidence is the mean over the placed words",
      abs(A._confidence(placed([(0, 1), (1, 2)])) - 0.7) < 1e-9)
check("no words is 0.0 confidence, not an error", A._confidence([]) == 0.0)
# The acceptance rule for a retry: a wider window is kept only when it is MORE sure. A section that
# genuinely starts at its window's edge comes back identical and loses nothing.
better = [({}, 0, 1, 0.9)]
check("a more confident retry wins", A._confidence(better) > A._confidence(placed([(0, 1)])))


# ------------------------------------------------------------------- where the weights will live
root = A.download_root()
check("the model root is under ComfyUI's own models folder, not a hidden cache",
      root.replace("\\", "/").endswith("stt/echo"), root)
check("...and it is its own folder, not shared with another pack's whisper",
      not root.replace("\\", "/").endswith("stt/whisper"))
check("is_downloaded answers without raising on an empty folder",
      A.is_downloaded() in (True, False), A.is_downloaded())
check("the download is named and sized before it is fetched",
      "MMS_FA" in A.MODEL_NAME and 1e9 < A.MODEL_BYTES < 2e9)

# =============================================== the loop: each section teaches the next one where
# The whole orchestration over a STUB aligner that behaves the way the real one has to: forced
# alignment must place every word it is given INSIDE the window it is given, so a window that does
# not contain the words crams them against an edge and the confidence collapses. That single
# property is what the anchoring is built against, and it is enough to test the loop without a GPU.
TRUTH = {                                       # the author's real take, section -> where it is
    "Verse 1": (14.32, 34.46), "Pre-Chorus": (38.61, 53.96), "Chorus 1": (55.95, 80.34),
    "Verse 2": (81.66, 100.75), "Bridge": (103.43, 116.66), "Chorus 2": (117.21, 146.10),
    # Never sung. Without a floor its words land confidently on top of Chorus 2, which is exactly
    # what happened on the real render: subtitles over a chorus that was singing something else.
    "Outro": (118.02, 125.77),
}
PLANNED = {"Verse 1": (12.8, 32.0), "Pre-Chorus": (32.0, 44.8), "Chorus 1": (44.8, 64.0),
           "Verse 2": (64.0, 89.6), "Bridge": (89.6, 102.4), "Chorus 2": (108.8, 128.0),
           "Outro": (128.0, 134.4)}
SONG = 179.2

A.mono_16k = lambda audio: (None, SONG)
A.load = lambda device=None, half=True: (None, None, None)
SEEN = []


def stub_window(model, tokenizer, aligner, wave, lo, hi, lines, star, label, notes):
    SEEN.append((label, round(lo, 2), round(hi, 2)))
    a, b = TRUTH[label]
    # Clamped into the window, because that is the one thing the real aligner cannot refuse to do.
    a2 = max(lo, min(a, hi))
    b2 = min(hi, max(b, a2 + 0.5))
    conf = max(0.0, 1.0 - (abs(a2 - a) + abs(b2 - b)) / 12.0)
    words = [w for ln in lines for w in ln["words"] if w["key"]]
    step = (b2 - a2) / max(1, len(words))
    return [(w, a2 + i * step, a2 + (i + 1) * step, conf) for i, w in enumerate(words)] or None


A._align_window = stub_window


def a_job():
    blocks = []
    for label, (p0, p1) in PLANNED.items():
        pad = max(TR.SLACK_MIN, (p1 - p0) * TR.SLACK)
        blocks.append({"label": label, "voice": "Nina", "start": p0, "end": p1,
                       "window": (max(0.0, p0 - pad), min(SONG, p1 + pad)),
                       "lines": [{"text": "x y z", "voice": "Nina", "label": label,
                                  "words": [{"text": w, "key": w} for w in ("aa", "bb", "cc")]}]})
    return {"total": SONG, "blocks": blocks, "language": "uk"}


def outcome(**kw):
    SEEN.clear()
    job = a_job()
    notes = A.align_job(job, {"waveform": None, "sample_rate": 16000}, **kw)
    got = {}
    for b in job["blocks"]:
        ws = [w for ln in b["lines"] for w in ln["words"] if w.get("start") is not None]
        if ws:
            got[b["label"]] = (ws[0]["start"], ws[-1]["end"],
                               sum(w["score"] for w in ws) / len(ws))
    return got, notes


anchored, anchored_notes = outcome(anchor=True, retry_pinned=False)
plain, _ = outcome(anchor=False, retry_pinned=False)

sung = [k for k in TRUTH if k != "Outro"]


def whole(got, k):
    """BOTH ends, not just the start. Every un-anchored window here contains its section's first
    word — what it does not contain is the last one, so the rest is crammed against the edge and
    the confidence collapses. Measuring the start alone says everything is fine."""
    return abs(got[k][0] - TRUTH[k][0]) < 1.0 and abs(got[k][1] - TRUTH[k][1]) < 1.0


good_a = [k for k in sung if whole(anchored, k)]
good_p = [k for k in sung if whole(plain, k)]
check("anchored, every sung section lands where it really is", len(good_a) == 6, good_a)
check("...where reading the plan alone gets one of six", len(good_p) == 1, good_p)
check("...and that one is Verse 1, exactly as on the real take", good_p == ["Verse 1"], good_p)
check("anchored confidence is high across the board",
      min(anchored[k][2] for k in sung) > 0.9, {k: round(anchored[k][2], 2) for k in sung})
check("...where un-anchored it collapses, which is what the real report showed",
      max(plain[k][2] for k in sung if k != "Verse 1") < 0.8,
      {k: round(plain[k][2], 2) for k in sung})

# The headline: the Outro was never sung, and the floor turns a confident wrong answer into a
# reported non-answer. Without the floor its words sit on top of a chorus that IS singing.
check("un-anchored, the Outro is laid on top of the chorus that is really singing",
      plain["Outro"][0] < plain["Chorus 2"][1], (plain["Outro"], plain["Chorus 2"]))
check("anchored, it is searched AFTER the chorus stopped instead",
      anchored["Outro"][0] >= TRUTH["Chorus 2"][1] - TR.ANCHOR_OVERLAP - 0.01, anchored["Outro"])
check("...so it no longer collides with anything",
      anchored["Outro"][0] >= anchored["Chorus 2"][1] - TR.ANCHOR_OVERLAP - 0.01,
      (anchored["Outro"], anchored["Chorus 2"]))
check("...and its confidence collapses, which is the signal that it was not sung",
      anchored["Outro"][2] < 0.4, round(anchored["Outro"][2], 2))
check("the collapse is SAID, not left to be noticed",
      any("previous section" in n and "Outro" in n for n in
          A.align_job(a_job(), {"waveform": None, "sample_rate": 16000}, anchor=True)),
      [n for n in anchored_notes])

# A section nobody could place must not poison the rest: it teaches the next one nothing.
BAD = dict(TRUTH)
BAD["Chorus 1"] = (5.0, 9.0)                   # nowhere near its window; comes back clamped and weak
TRUTH_BACKUP = dict(TRUTH)
TRUTH.update(BAD)
after_bad, _ = outcome(anchor=True, retry_pinned=False)
TRUTH.clear()
TRUTH.update(TRUTH_BACKUP)
check("a section that failed does not drag the ones after it off the music",
      abs(after_bad["Verse 2"][0] - TRUTH["Verse 2"][0]) < 1.5, after_bad["Verse 2"])
check("...which is what the confidence gate is for", TR.ANCHOR_TRUST > 0.0, TR.ANCHOR_TRUST)

# Windows must still be bounded — anchoring may not quietly grow them into the whole song.
widest = max(hi - lo for _, lo, hi in SEEN)
check("no anchored window approaches the length of the song", widest < SONG * 0.35,
      f"{widest:.1f} s of {SONG:.0f}")

check.done()
