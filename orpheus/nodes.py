"""Orpheus — a finished track in, the shot boundaries of a music video out.

The node is thin on purpose: every decision it makes lives in `timing.py` (musical time against
H3's grid) or `detect.py` (where the music changes), both of which are tested without audio and
without a model. What is left here is the wiring, the report, and the one picture that makes the
whole thing checkable at a glance — the spectrogram with the cuts drawn on it. If the bright lines
do not sit on the transients, nothing downstream is worth rendering, and that is a half-second look
rather than a twenty-minute render.

**The two inputs that matter, and why they are not what they look like:**

`pace` is a **bias, not a length.** It decides how many shots the remainder should become, and the
cut then goes to the best *available* cue — so on a 128 BPM phrase grid, 7 and 9 produce byte-
identical plans, because the only candidates are 7.5 s apart. Turn it in whole units, not decimals.

`cut_on` costs real time, and how much moves with the tempo. At 120 BPM a 4-bar phrase is 8.00 s and
lands exactly on H3's grid: nothing is wasted. At 128 BPM the same phrase is 7.50 s and snaps up to
8.00 — 12 frames, 6.2%, thrown away on every shot — while *two* phrases is 15.00 s against a 15.083 s
ceiling and wastes 0.6%. The report prints the whole table for the tempo you actually have, because
that is a factor-of-ten decision hiding inside a dropdown.

**With a Siren `plan` wired**, section boundaries are not detected at all — they are *read*, from the
table you wrote, exact to the code. Detection then only fills in what the plan cannot know: the
accents inside a section. This is the normal case for your own songs and it is strictly better than
any detector, because the boundaries were authored rather than inferred.

**`cues` is deliberately not `beats`.** `Morpheus Storyboard` skips its own planning call when
`beats` is filled — "your lines win" — so wiring a list of musical facts into it would hand the shot
writer "0:48 — drop" as an entire director's note, and it would film exactly that. Read `cues`
through a `Show Text`, and merge the lines that earn it into `beats` yourself.
"""
import json
import time

import torch

from ..categories import CAT_ORPHEUS
from ..siren.cast import _bar_seconds, _parse_plan
from ..siren.scope import (_colorize, _draw_text, _grid, _label_strip, _nice_step, _resize,
                           _spectrogram_power, _to_db)
from ..timer.timer_nodes import _format_elapsed
from . import detect as det
from . import timing as T

#: The picture's dB window. Absolute, never auto-ranged — same rule as Siren Scope, so two runs of
#: the same track are comparable and a quiet mix renders dark instead of being silently boosted.
DB_FLOOR, DB_CEILING = -80.0, 0.0
SCOPE_N_FFT, SCOPE_N_MELS = 2048, 160


def _scope_image(sig, sr, duration, shots, bpm, beats_per_bar, offset, width, height):
    """The spectrogram with the plan drawn over it: bar grid, then one line per cut.

    Two colours, because they answer different questions. A **cut taken on a cue** is amber; a cut
    the planner had to make on the grid or on length alone is dim red. A row of amber lines sitting
    on the transients is the whole verification, and a red one tells you exactly where the music
    gave the planner nothing to work with — which is a `sensitivity` or `cut_on` problem, and is
    otherwise invisible until the render looks wrong.
    """
    hop = int(min(max(sig.numel() // max(1, int(width) * 2), 128), SCOPE_N_FFT))
    db = _to_db(_spectrogram_power(sig, sr, "mel", SCOPE_N_FFT, SCOPE_N_MELS, hop))
    norm = ((db - DB_FLOOR) / max(1e-6, DB_CEILING - DB_FLOOR)).clamp(0.0, 1.0)
    # Resize the INTENSITY, then colour it. The other order antialiases in RGB, which averages
    # across the ramp's hue turns and smears a bright transient into a muddy band.
    panel = _colorize(_resize(torch.flip(norm, dims=[0]), int(height), int(width)), "magma")

    step_sec = _nice_step(duration, int(width))
    _grid(panel, duration, bpm, beats_per_bar, offset, step_sec)

    def col(sec):
        return int(round(float(sec) / max(1e-9, duration) * (int(width) - 1)))

    amber = torch.tensor([1.00, 0.78, 0.20])
    dim = torch.tensor([0.85, 0.25, 0.25])
    for s in shots[:-1] if shots else []:
        c = col(s["end"])
        colour = amber if s["cue"] is not None else dim
        for x in (c - 1, c, c + 1):
            if 0 <= x < panel.shape[1]:
                a = 0.95 if x == c else 0.35
                panel[:, x] = panel[:, x] * (1.0 - a) + colour * a
    for s in shots:
        _draw_text(panel, str(s["index"] + 1), col(s["start"]) + 3, 3, 1, (1.0, 1.0, 1.0))

    strip = _label_strip(int(width), duration, step_sec, 1)
    return torch.cat([panel, strip], dim=0).unsqueeze(0)


class KinburgOrpheusScore:
    """Audio → where the cuts fall, as `durations` Phantas and Morpheus already take."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The finished track — straight out of Siren, or a LoadAudio for something off disk. It is analysed whole even when 'start_sec'/'end_sec' narrow the part you are cutting, because tempo and downbeat are measured far more reliably over the whole song than over a 20-second window."}),
                "cut_on": (T.CUT_UNITS, {"default": "2 bars", "tooltip": "The musical unit a cut may land on. Same vocabulary as Siren Section's 'snap'.\n\nThis is not only taste — it decides how much video gets generated and thrown away, and by how much depends on the tempo. A shot is 3-7 bars at a normal tempo, so 'bar' gives the planner more lines than it can use while 'phrase (4 bars)' may give it only two possible shot lengths. The report prints the full cost table for YOUR tempo; read it once per song."}),
                "pace": ("FLOAT", {"default": 7.0, "min": T.MIN_SECONDS, "max": T.MAX_SECONDS, "step": 0.25, "tooltip": "The shot length to aim for, in seconds — a BIAS, not a length.\n\nIt decides how many shots the rest of the track should become; the cut itself then goes to the best cue available, which is never exactly this number. So on a 7.5 s phrase grid, 7 and 9 give the SAME plan — there is nothing between the candidates to choose. Move it in whole seconds and watch the shot count, not the decimals."}),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How far above its own neighbourhood a moment must stand to become a candidate cut. Low = every drum hit is a candidate (the planner has plenty of choice and cuts nearer 'pace'); high = only the big moments, so shots stretch to reach one.\n\nIgnored for section boundaries when a 'plan' is wired — those are read, not detected."}),
                "bpm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.01, "tooltip": "0 = measure it from the audio. Anything else is used verbatim.\n\nType it for a song you generated: you know the tempo exactly, and a measured one carries a confidence that can be low on a ballad or anything without drums. The downbeat is measured either way — knowing the tempo does not tell you where bar 1 starts."}),
                "beats_per_bar": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "The time signature's top number. Only used to turn beats into bars, which is what 'cut_on' measures in."}),
            },
            "optional": {
                "plan": ("STRING", {"forceInput": True, "tooltip": "Siren Cast's or Siren Score's plan — 'label | voice | length' rows.\n\nWired, the section boundaries come from the TABLE rather than from the signal: exact, named, and impossible to miss. Detection then only adds the accents inside each section. This is the right way round for a song you wrote, and no detector can beat it."}),
                "start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip": "Begin the video here instead of at the top of the track — to skip a long intro, or to cut a teaser out of the middle. The whole track is still analysed."}),
                "end_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip": "Stop here. 0 = run to the end of the track."}),
                "cue_pull": ("FLOAT", {"default": T.DEFAULT_CUE_PULL, "min": 0.0, "max": 10.0, "step": 0.1, "advanced": True, "tooltip": "How many seconds of deviation from the ideal length a cue's importance is worth. Between two candidates it is their strength DIFFERENCE that pays, so at the default 2.0 a section boundary outranks a weak onset up to ~1.8 s further away and loses beyond that.\n\nRaise it to follow the music harder at the cost of uneven shots; drop it to 0 to ignore importance entirely and simply cut on the nearest line."}),
                "scope": ("BOOLEAN", {"default": True, "tooltip": "Render the spectrogram with the cuts drawn on it. Amber = the cut landed on a cue, dim red = the planner had nothing to cut on there. This is how you check a plan in half a second instead of a render."}),
                "scope_width": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 16, "advanced": True, "tooltip": "Width of that picture in pixels. The whole track always spans it, so a longer song simply gets less detail per second."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Print the report to the console. The same text is always on the 'report' output."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "FLOAT", "STRING", "IMAGE", "GEN_INFO")
    RETURN_NAMES = ("durations", "trims", "cues", "shot_count", "seconds", "report", "scope",
                    "gen_extra_info")
    FUNCTION = "run"
    CATEGORY = CAT_ORPHEUS
    DESCRIPTION = ("Read a track and decide where the cuts fall, so a music video is edited to its "
                   "own music. Outputs the 'durations' string Phantas and Morpheus already take, "
                   "plus the per-shot trim that puts each cut back exactly on the beat.")

    def run(self, audio, cut_on, pace, sensitivity, bpm, beats_per_bar, plan="", start_sec=0.0,
            end_sec=0.0, cue_pull=T.DEFAULT_CUE_PULL, scope=True, scope_width=1280, verbose=True):
        began = time.time()
        notes = []

        found = det.detect(audio, sensitivity=sensitivity, bpm=bpm, beats_per_bar=beats_per_bar)
        track_len = found["total"]
        if track_len <= 0:
            raise RuntimeError("[Orpheus] the audio is empty — nothing to cut.")

        start = max(0.0, min(float(start_sec), track_len))
        end = track_len if float(end_sec) <= 0 else min(float(end_sec), track_len)
        if end - start < T.MIN_SECONDS:
            raise RuntimeError(
                f"[Orpheus] the window {start:.2f}-{end:.2f} s is {end - start:.2f} s, shorter than "
                f"one shot ({T.MIN_SECONDS:.2f} s is the shortest H3 renders). Widen it.")

        bar_sec = found["bar_seconds"]
        step = T.unit_seconds(bar_sec, beats_per_bar, cut_on)
        offset = det.phase_for(found, step) if step else 0.0
        if not step:
            notes.append("no tempo, so there is no bar grid: cuts fall on detected moments alone, "
                         "and on plain length where there are none. Type the bpm.")

        cues = list(found["cues"])
        plan_rows = []
        if str(plan or "").strip():
            use_bpm = found["bpm"] if found["bpm"] > 0 else 0.0
            plan_rows, plan_notes = _parse_plan(plan, use_bpm, beats_per_bar)
            notes.extend(plan_notes)
            # The plan REPLACES detected section boundaries rather than joining them: it is exact and
            # named, and two competing "this is a section change" cues 0.4 s apart is how a cut lands
            # between the two rather than on either.
            cues = [c for c in cues if c["kind"] != "section"]
            at = 0.0
            for row in plan_rows:
                at += row["seconds"]
                if at < end - 1e-6:
                    cues.append(T.cue(at, 1.0, "section", row["label"]))
            if abs(at - track_len) > 1.0:
                notes.append(f"the plan adds up to {T.mmss(at)} but the audio is {T.mmss(track_len)} "
                             f"— the section boundaries will drift. Check the plan's lengths.")

        if step:
            cues = T.snap_cues(cues, step, offset=offset, total=end)
        shots = T.plan_shots(end, cues, preferred=pace, start=start, cue_pull=cue_pull,
                             step=step, offset=offset)
        notes.extend(T.warnings(shots, total=end))

        head = [f"[Orpheus] {T.mmss(end - start)} of {T.mmss(track_len)}, {len(shots)} shot(s)",
                "  " + det.confidence_note(found),
                f"  cutting on {cut_on}" + (f" = {step:.2f} s, downbeat {offset:.2f} s" if step else ""),
                "  " + T.describe(shots)]
        if plan_rows:
            head.append(f"  plan: {len(plan_rows)} section(s) read from the table, not detected")
        body = [T.timeline(shots), "", T.describe_options(step, cut_on) if step else ""]
        report = "\n".join(head + [""] + [b for b in body if b] +
                           ([""] + [f"  ! {n}" for n in notes] if notes else []))
        if verbose:
            print(report.replace("\n", "\n[Orpheus] "))

        img = torch.zeros(1, 8, 8, 3)
        if scope:
            sig, sr = det.to_mono(audio)
            img = _scope_image(sig, sr, track_len, shots, found["bpm"], beats_per_bar, offset,
                               int(scope_width), 320)

        elapsed = _format_elapsed(time.time() - began, "auto")
        params = {"track": f"{track_len:.2f} s", "window": f"{start:.2f}-{end:.2f} s",
                  "bpm": f"{found['bpm']:.2f} ({found['bpm_source']})",
                  "confidence": f"{found['confidence']:.2f}", "cut_on": cut_on,
                  "downbeat": f"{offset:.3f} s", "pace": float(pace),
                  "sensitivity": float(sensitivity), "shots": len(shots),
                  "trimmed": f"{sum(s['trim_frames'] for s in shots)} frames", "time": elapsed}
        gen_extra = json.dumps([{"class_type": "Orpheus", "ord": 1, "params": params}],
                               ensure_ascii=False)

        return (T.format_durations(shots), T.format_trims(shots), T.format_cues(shots), len(shots),
                float(end - start), report, img, gen_extra)


NODE_CLASS_MAPPINGS = {"KinburgOrpheusScore": KinburgOrpheusScore}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgOrpheusScore": "Orpheus (Audio → Shots) 🎶"}
