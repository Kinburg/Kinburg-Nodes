"""Save Clip's timeline — everything that decides WHEN a picture is on screen.

No torch, no PyAV, no PIL: this is the arithmetic that can silently ruin a three-minute render, and
the only way to know it is right is to check it without rendering. The encoder itself is not tested
here — it is ComfyUI's own recipe (h264 + AAC through PyAV), and what could go wrong in it fails
loudly on the first frame.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "save_video", "siren", "context", "timer")
T = load_module("kn.save_video.timeline", "save_video/timeline.py")

check = Checker()

SIREN_PLAN = """\
Intro | - | 4 bars
Verse 1 | Nina | 16 bars
Chorus | Nina | 8 bars
Verse 2 | Nina | 16 bars
Chorus | Nina | 8 bars
Outro | - | 4 bars
END
"""

# ------------------------------------------------------------------------------ reading the plan
rows, notes = T.sections(SIREN_PLAN)
check("a Siren plan is read row by row", [r["label"] for r in rows] ==
      ["Intro", "Verse 1", "Chorus", "Verse 2", "Chorus", "Outro"])
check("...with the bars kept in proportion", rows[1]["weight"] == 4 * rows[0]["weight"])
check("...and the rows after END dropped", len(rows) == 6)

orph, _ = T.sections("3.50, 4.00, 8.00")
check("Orpheus's bare list of seconds is read too", [r["weight"] for r in orph] == [3.5, 4.0, 8.0])
check("...labelled so the report can still name them", orph[0]["label"] == "shot 1")
check("an empty plan is no plan, not an error", T.sections("   ") == ([], []))

# ------------------------------------------------------------------------------- laying them out
# The whole point: the plan's clock never reaches the video. The audio's does.
segs, notes = T.lay_out(rows, 6, T.LAYOUT_ORDER, 180.0)
check("the segments cover the song exactly",
      segs[0]["start"] == 0.0 and abs(segs[-1]["end"] - 180.0) < 1e-9)
check("...one slide per section, in order", [s["slide"] for s in segs] == [0, 1, 2, 3, 4, 5])
check("...in the plan's proportions, scaled onto the audio",
      abs((segs[1]["end"] - segs[1]["start"]) / (segs[0]["end"] - segs[0]["start"]) - 4.0) < 1e-9)
check("...and the stretch is reported rather than hidden",
      any("stretched" in n for n in notes), notes)

# a plan whose own length already matches says nothing about stretching
_, quiet = T.lay_out(rows, 6, T.LAYOUT_ORDER, sum(r["weight"] for r in rows))
check("a plan that already fits the song is not commented on",
      not any("stretched" in n for n in quiet), quiet)

# fewer slides than sections -> they cycle, and neighbours that repeat are FUSED
segs, notes = T.lay_out(rows, 2, T.LAYOUT_ORDER, 120.0)
check("fewer slides than sections cycle", [s["slide"] for s in segs] == [0, 1, 0, 1, 0, 1])
check("...and it is reported", any("cycle" in n for n in notes), notes)

three = [{"label": "A", "weight": 1.0}, {"label": "B", "weight": 1.0}, {"label": "C", "weight": 1.0}]
segs, _ = T.lay_out(three, 1, T.LAYOUT_ORDER, 30.0)
check("one picture over three sections is ONE segment, not three identical ones",
      len(segs) == 1 and segs[0]["start"] == 0.0 and segs[0]["end"] == 30.0)

# more slides than sections -> the extras subdivide, weighted
segs, _ = T.lay_out([{"label": "Short", "weight": 1.0}, {"label": "Long", "weight": 5.0}],
                    4, T.LAYOUT_ORDER, 60.0)
check("more slides than sections subdivide the LONG one",
      [s["label"] for s in segs] == ["Short", "Long 1/3", "Long 2/3", "Long 3/3"],
      [s["label"] for s in segs])
check("...each still getting its own picture", [s["slide"] for s in segs] == [0, 1, 2, 3])
check("...and the subdivisions are equal inside their section",
      abs((segs[1]["end"] - segs[1]["start"]) - (segs[2]["end"] - segs[2]["start"])) < 1e-9)

# by label: the chorus shot comes back
segs, _ = T.lay_out(rows, 6, T.LAYOUT_LABEL, 180.0)
check("by label, both choruses show the same slide",
      [s["slide"] for s in segs] == [0, 1, 2, 3, 2, 4], [s["slide"] for s in segs])
check("...and there are as many segments as sections", len(segs) == 6)

segs, notes = T.lay_out(rows, 2, T.LAYOUT_LABEL, 180.0)
check("by label with too few slides, the labels cycle and it says so",
      any("cycle" in n for n in notes), notes)

# even ignores the plan; no plan behaves the same way
segs, notes = T.lay_out(rows, 3, T.LAYOUT_EVEN, 90.0)
check("even splits the song equally", [round(s["end"], 6) for s in segs] == [30.0, 60.0, 90.0])
check("...and says the plan was ignored", any("ignored" in n for n in notes), notes)
check("no plan at all is the same even split",
      T.lay_out([], 3, T.LAYOUT_ORDER, 90.0)[0] == segs)

# a plan of nothing but zeros must not divide by zero
segs, notes = T.lay_out([{"label": "X", "weight": 0.0}], 2, T.LAYOUT_ORDER, 10.0)
check("an all-zero plan falls back to an even split instead of exploding",
      len(segs) == 2 and any("0" in n for n in notes))

# ------------------------------------------------------------------------------ the frame track
segs = [{"start": 0.0, "end": 5.0, "slide": 0, "label": "A"},
        {"start": 5.0, "end": 10.0, "slide": 1, "label": "B"}]
track = T.frame_track(segs, 10, 10.0, 0.0)
check("there is one entry per encoded frame", len(track) == 100)
check("...on the right side of the cut with no crossfade",
      track[49][1] == 0 and track[50][1] == 1)
check("...and nothing is mixed", all(f[3] == 0.0 for f in track))

track = T.frame_track(segs, 10, 10.0, 1.0)
mixed = [f for f in track if f[3] > 0]
check("a 1 s crossfade covers 10 frames at 10 fps", len(mixed) == 10, len(mixed))
check("...centred ON the cut, not after it",
      abs(mixed[0][0] - 4.5) < 0.06 and abs(mixed[-1][0] - 5.45) < 0.06,
      (mixed[0][0], mixed[-1][0]))
check("...running from one picture to the other",
      mixed[0][1] == 0 and mixed[0][2] == 1 and mixed[0][3] < 0.15 and mixed[-1][3] > 0.85,
      (mixed[0][3], mixed[-1][3]))

# a fade may never swallow a short section
short = [{"start": 0.0, "end": 1.0, "slide": 0, "label": "A"},
         {"start": 1.0, "end": 2.0, "slide": 1, "label": "B"}]
mixed = [f for f in T.frame_track(short, 25, 2.0, 4.0) if f[3] > 0]
check("a long crossfade is clamped to 40% of the shorter neighbour",
      abs(len(mixed) / 25.0 - 0.4) < 0.05, len(mixed) / 25.0)

check("a single segment has no transition at all",
      all(f[3] == 0.0 for f in T.frame_track(segs[:1], 10, 5.0, 2.0)))

# ------------------------------------------------------------------------------------ ken burns
check("no move at all when the amount is 0", T.ken_burns(0, 0.0) == (1.0, 1.0, (0.0, 0.0), (0.0, 0.0)))
z0, z1, a, b = T.ken_burns(0, 0.1)
check("the first slide pushes in", z0 == 1.0 and abs(z1 - 1.1) < 1e-9)
check("...and the next one pulls out, so four in a row don't all drift the same way",
      T.ken_burns(1, 0.1)[0] > T.ken_burns(1, 0.1)[1])
check("...deterministically — a re-render is the same clip", T.ken_burns(7, 0.1) == T.ken_burns(3, 0.1))
check("the pan starts and ends on opposite sides", a[0] == -b[0] and a[1] == -b[1])

# ------------------------------------------------------------------------------------ subtitles
lyric = [{"label": "Verse 1", "text": ["line one", "line two"]},
         {"label": "Chorus", "text": ["the hook"]},
         {"label": "Verse 2", "text": ["line three"]},
         {"label": "Chorus", "text": ["the hook"]}]
spans = T.section_spans(rows, 180.0)
check("the plan's sections span the whole song",
      spans[0]["start"] == 0.0 and abs(spans[-1]["end"] - 180.0) < 1e-9 and len(spans) == len(rows))
check("...keeping every label, so the words can be matched to them",
      [s["label"] for s in spans] == [r["label"] for r in rows])
check("no plan means no spans to time anything against", T.section_spans([], 180.0) == [])

body = T.srt(spans, lyric)
check("every sung section gets a cue", body.count(" --> ") == 4, body.count(" --> "))
check("...and the instrumental rows the plan added get none", "Intro" not in body)
check("...numbered from 1, in SubRip's own clock", body.startswith("1\n00:00:"))
check("...timed to the section, not to the plan's own clock",
      T.srt_time(spans[1]["start"]) in body, body[:80])
check("no lyrics, no file", T.srt(spans, []) == "")

# The regression that shipped: subtitles used to be timed off the VISIBLE segments, so a single
# still — the thing this node is mostly for — fused into one segment labelled
# "Intro + Verse 1 + Chorus + ..." and matched nothing at all.
one, _ = T.lay_out(rows, 1, T.LAYOUT_ORDER, 180.0)
check("one still over the whole song really is one segment", len(one) == 1)
check("...and its subtitles are STILL per section",
      T.srt(T.section_spans(rows, 180.0), lyric).count(" --> ") == 4)
check("...which is not what timing them off the segments would give", T.srt(one, lyric) == "")

even, _ = T.lay_out(rows, 3, T.LAYOUT_EVEN, 180.0)
check("an even layout has no section labels on its segments at all",
      all(not s["label"] for s in even))
check("...and its subtitles are unaffected, because they never read them",
      T.srt(T.section_spans(rows, 180.0), lyric).count(" --> ") == 4)

# ---------------------------------------------------------------------------------- the report
text = T.report(segs, 6, 12, (1920, 1080), 180.0, ["something to say"])
check("the report names the frame and the fps", "1920x1080" in text and "12 fps" in text)
check("...lists a line per segment", text.count("slide ") == len(segs))
check("...and carries the notes", "something to say" in text)
check("...in characters a cp1251 console can actually print",
      all(ch not in text for ch in "→←"), text[:60])

check.done()
