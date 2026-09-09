"""`Echo (Lyrics → Timing)` — where each word was actually sung.

The node is thin on purpose: it reads its inputs, hands `track.build` the job, hands `align.align_job`
the audio, and prints what came back. Every decision worth arguing about lives in one of those two
modules, next to the reasoning for it.

What it is FOR, in the order the outputs matter:

* **`timing`** — the structure `Save Clip` burns or writes, and the one `Orpheus` will read. One
  alignment, several consumers: it is expensive enough that computing it twice would be silly.
* **`plan`** — the wired plan with every section moved to where its words really are. The author's
  original complaint was that the plan does not match what the model sang; this is that complaint
  answered as data. Wire it back into `Save Clip` and the pictures land on the real section
  boundaries instead of the intended ones.
* **`report`** — how far each section moved and how sure the aligner is. This is the output to read
  first, because it is the one that says whether to believe the other two.

**The honest limit, stated here because it is where this will disappoint.** Forced alignment places
every word it is given, whether or not it was sung. AceStep swallows words, and with a long tail it
repeats the last phrase. A line the model never sang will still be laid somewhere, and the only
signal that it was invented is a low confidence — which is exactly why the confidence is in the
report and not hidden.
"""
import os
import time

from . import align as A
from . import subs as S
from . import track as T
from . import translit as R
from ..categories import CAT_ECHO
from ..context.character_card import VOICE_TYPE
from ..timer.timer_nodes import _format_elapsed


#: How much two sections may overlap before it is worth complaining about. A backing echo really
#: does start a fraction before its lead's last word ends, and a word's own boundary is only located
#: to within an emission frame — so a tenth of a second of overlap is arithmetic, not a misplacement.
CLASH = 0.5


def _plan_table(track, decimals=2, floor=0.05):
    """The corrected timeline, written back as a plan table `Save Clip` and `Orpheus` already read.

    Returns `(table, notes)`. Lengths rather than absolute times, because that is the shape both of
    those nodes take and they scale a plan onto the audio anyway. Instrumental rows are kept — a
    plan that lost its solos is a plan whose remaining sections no longer add up to the song.

    **Emitted in the order the words were FOUND, not the order the plan listed them.** A misplaced
    section can come back before the one it followed, and walking the plan's order then produces a
    row of negative length — which used to clamp to `0.00s` and be dropped by the parser further
    downstream, quietly deleting a section from a table whose whole purpose is to be trustworthy.
    Reordering is itself the loudest available signal that something is misplaced, so it is said.
    """
    secs = list(track.get("sections") or [])
    ordered = sorted(secs, key=lambda s: (s["start"], s.get("index", 0)))

    # Two different failures, one message. A section can come back BEFORE the one it followed
    # (the order changes), or it can come back INSIDE it (the order is untouched and the spans
    # overlap) — the second is the shape the author's real song produced, and checking only the
    # first missed it entirely.
    notes = []
    moved = [s["label"] for s, o in zip(secs, ordered) if s.get("index") != o.get("index")]
    clash = [lb for a, b in zip(ordered, ordered[1:]) if a["end"] > b["start"] + CLASH
             for lb in (a["label"], b["label"])]
    if moved or clash:
        who = ", ".join(dict.fromkeys(clash + moved))
        notes.append(f"the corrected plan could not simply follow the found spans — {who} overlap "
                     f"or came back out of sequence. Two sections cannot be sung at once, so at "
                     f"least one of them is timed against the wrong part of the song. The table is "
                     f"still written, in the order the words were found and with every row given a "
                     f"positive length, but check those sections before wiring it anywhere.")

    rows, at = [], 0.0
    for i, s in enumerate(ordered):
        # A section runs to where the next one's words begin, so the table stays contiguous even
        # though the aligned spans have gaps between them (the gaps are music).
        end = ordered[i + 1]["start"] if i + 1 < len(ordered) else track.get("total", s["end"])
        end = max(end, s["end"], at + floor)
        rows.append("{} | {} | {:.{d}f}s".format(s["label"] or f"section {i + 1}",
                                                 s["voice"] or "-", end - at, d=decimals))
        at = end
    return "\n".join(rows), notes


def _report(track, notes, elapsed):
    """The block printed to the console and put on `report`. Plain hyphens — see `save_video`."""
    from ..save_video.timeline import mmss

    lines = [f"Echo - {len(track['lines'])} line(s) aligned in {len(track['sections'])} section(s) "
             f"- {track['language']} - mean confidence {track['score']:.2f} - {elapsed}"]
    if track.get("spellable", 1.0) < 1.0:
        lines.append(f"  {track['spellable'] * 100:.0f}% of the words could be written in the "
                     f"aligner's alphabet")

    lines.append("  section              planned            found              drift   conf")
    for s in track["sections"]:
        p0, p1 = s["planned"]
        drift = s["start"] - p0
        # By section INDEX. Filtering by label averaged all three choruses of a song into one row.
        conf = [ln["score"] for ln in track["lines"] if ln.get("section") == s.get("index")]
        lines.append("  {:<20} {:>7} {:>7}   {:>7} {:>7}   {:+6.2f}s {:>5}".format(
            (s["label"] or "")[:20], mmss(p0), mmss(p1), mmss(s["start"]), mmss(s["end"]),
            drift, f"{sum(conf) / len(conf):.2f}" if conf else "  -  " if s["sung"] else "instr"))

    # Printed BEFORE the weak lines, because it is the louder failure: a collision means a section
    # was timed against the wrong part of the song, and its own confidence can look fine.
    hits = track.get("collisions") or []
    if hits:
        lines.append(f"  ! {len(hits)} place(s) where two different sections are on screen at "
                     f"once. That is what a section timed against the wrong part of the song looks "
                     f"like — the words appear early, over somebody else's line, and the real "
                     f"performance passes with no subtitle. Raise 'slack', and check that the "
                     f"lyrics' markers and the plan's labels repeat the same number of times:")
        for c in hits[:6]:
            lines.append(f"      {mmss(c['at'])}  '{c['a']}' over '{c['b']}'  {c['seconds']:.1f}s")

    weak = [ln for ln in track["lines"] if ln["found"] and ln["score"] < 0.4]
    if weak:
        lines.append(f"  ! {len(weak)} line(s) came back under 0.40 confidence — the model probably "
                     f"did not sing them as written. The lowest:")
        for ln in sorted(weak, key=lambda x: x["score"])[:5]:
            lines.append(f"      {mmss(ln['start'])} {ln['score']:.2f}  {ln['text'][:52]}")

    # Said out loud rather than done quietly: it is a heuristic about a held note, and the only way
    # to know it is not eating one is to see where it fired and listen to that spot.
    stranded = track.get("stranded") or []
    if stranded:
        lines.append(f"  {len(stranded)} line(s) ended on a word that ran into the silence after "
                     f"it and were pulled back "
                     f"({sum(s for _, s in stranded):.1f} s in total, longest "
                     f"{max(s for _, s in stranded):.1f} s in '{max(stranded, key=lambda x: x[1])[0]}')")

    quiet = track.get("quiet") or []
    if quiet:
        lines.append("  no words at all: " + ", ".join(
            f"{mmss(a)}-{mmss(b)}" for a, b in quiet[:8]) + ("  …" if len(quiet) > 8 else ""))
    lines.extend(f"  ! {n}" for n in notes)
    return "\n".join(lines)


class KinburgEchoAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The song — straight out of Siren, or a LoadAudio for something off disk. Nothing needs to be separated first: the aligner is looking for words it already knows, not trying to hear them out of nowhere.\n\nThis input is the CLOCK. Its length is what the plan's sections are scaled onto, so it has to be the whole track, start to end.\n\nNEVER put the instrumental stem here. It is also what gets listened to whenever 'vocals' is absent or rejected, and aligning a lyric against music with no voice in it produces timings that look confident and are nonsense.\n\nIf a separated vocal is all you want to wire, put it HERE and leave 'vocals' empty — a full-length stem is its own clock and the result is identical."}),
                "lyrics": ("STRING", {"forceInput": True, "tooltip": "The SAME lyric sheet that went to Siren, markers and all. This is the whole reason the node is cheap: the words are known, so the model is only asked WHERE they are, never what they were. A forced alignment cannot invent a word or drop one."}),
            },
            "optional": {
                "vocals": ("AUDIO", {"tooltip": "The isolated vocal, if you have one — from 'Vocals using MDX' or any other separator. This is what gets LISTENED to; 'audio' stays the clock and the fallback.\n\nWiring both is the belt-and-braces arrangement, and the reason to prefer it over putting the stem in 'audio' alone: a separator sometimes swallows a quiet or whispered passage entirely, and when that happens the mix is still there to be the clock and to take over.\n\nWhat it buys: the acoustic model was trained on speech, so a dense arrangement lowers its confidence on every word, worst on consonants. An isolated vocal raises the whole confidence column — which matters less for PLACING a word than for TRUSTING the placement. A weak reading and a wrong one both look like 0.3; at 0.8 against 0.3 they are telling you different things.\n\nWhat it does NOT fix: a section searched in the wrong window. That is arithmetic about the plan, not hearing, and no separation changes it.\n\nIt must be the same length as 'audio' — a separator that trimmed silence would shift every timing by however much it cut, so a mismatch is refused rather than quietly used."}),
                "plan": ("STRING", {"forceInput": True, "tooltip": "Siren Score's or Siren Cast's plan. Two jobs:\n\n• it says roughly where each section is, which is what bounds the search window — and the window is what keeps a three-minute song from costing three minutes of quadratic attention;\n• it carries the VOICE column, which is where per-singer colour comes from without any speaker diarization at all.\n\nLeave it empty and the lyric is spread over the song by syllable count with much wider windows. That works, and it is several times slower and less certain."}),
                "voice_1": (VOICE_TYPE, {"tooltip": "The band, as Character Card's 'voice' output — the same cards wired into Siren Score. Used to resolve a marker like '[Chorus - Nina]' to a member, so a solo line inside a duet chorus gets Nina's colour rather than the whole duet's.\n\nWith nothing wired the plan's own voice cell is used, which is usually right and cannot tell two singers apart inside one section."}),
                "voice_2": (VOICE_TYPE,),
                "voice_3": (VOICE_TYPE,),
                "voice_4": (VOICE_TYPE,),
                "language": (R.LANGUAGES, {"default": R.LANG_AUTO, "tooltip": "Which spelling rules turn the lyric into the aligner's 28-letter alphabet. 'auto' decides PER WORD from the letters that exist in only one language (ї є ґ against ы э ъ ё), falling back to whatever the lyric is mostly in — so a line with a Ukrainian and a Russian word in it is spelled correctly on both.\n\nSet it explicitly only if a song is in one language and 'auto' is guessing badly."}),
                "slack": ("FLOAT", {"default": T.SLACK, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How far outside its planned span a section is allowed to be found, as a fraction of its own length (never less than 3 s).\n\nThis is the dial for the author's actual complaint — that the plan and the performance disagree. Raise it when the report shows sections drifting up against their window's edge; lowering it makes the search stricter and faster, and risks pinning a section to a place it is not."}),
                "write_ass": ("BOOLEAN", {"default": True, "tooltip": "Write a .ass subtitle file next to the video folder. This is the FAST way to check the timing: open the song and the .ass in any player and the words light up as they are sung, in seconds, with no render at all. It is also restylable afterwards without re-encoding anything."}),
                "write_srt": ("BOOLEAN", {"default": False, "tooltip": "Also write a plain .srt — one cue per line, no karaoke. For anything that cannot read ASS."}),
                "filename_prefix": ("STRING", {"default": "clips/echo", "tooltip": "Save path prefix under ComfyUI/output, with a counter appended — the same convention Save Clip and Save Song use."}),
                "colors": ("STRING", {"multiline": True, "default": "", "tooltip": "One 'Name = #rrggbb' per line, e.g. 'Nina = #ff66cc'. Any singer not named here takes the next colour off a built-in palette, in order of first appearance — so this can stay empty and still give every voice its own colour.\n\nThe names are matched to the plan's voice column and to the cards, case- and space-insensitively."}),
                "font": ("STRING", {"default": "Arial", "tooltip": "Font name for the .ass file. It is resolved by the PLAYER, not here, so it has to be a font installed on whatever plays the file. Arial and Segoe UI both carry Cyrillic."}),
                "font_size": ("INT", {"default": 0, "min": 0, "max": 400, "tooltip": "0 = scale it to the frame (about 1/22 of the short side), which is what keeps a subtitle readable on both a 16:9 and a 9:16 clip. Anything else is used verbatim."}),
                "frame_width": ("INT", {"default": 1920, "min": 16, "max": 8192, "tooltip": "The frame the .ass is laid out for. It only sets PlayRes — the file scales to whatever it is played at — but getting it right makes the default font size land correctly."}),
                "frame_height": ("INT", {"default": 1080, "min": 16, "max": 8192}),
                "karaoke_sweep": ("BOOLEAN", {"default": True, "tooltip": "The highlight FILLS across each word as it is held (\\kf) rather than flipping at its start (\\k). Fill reads better on slow lines and worse on fast ones."}),
                "lead_in": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 3.0, "step": 0.05, "tooltip": "Seconds a line appears BEFORE its first word is sung, so it can be read rather than only followed. Nothing is highlighted during the wait."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "advanced": True, "tooltip": "Where the aligner runs. 'auto' is ComfyUI's own device. CPU works and is perhaps ten times slower; use it if something else is holding the VRAM."}),
                "half": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Run the encoder in fp16 on CUDA — half the weights and half the activation memory. The alignment itself is always done in fp32, so this costs nothing in precision that matters."}),
                "unload_after": ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Free the 1.2 GB of weights when the node is done. Off by default because subtitles get re-styled several times per song and reloading each time is seconds wasted; turn it on when the same graph goes straight into a video model that wants every byte."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True}),
                "anchor": ("BOOLEAN", {"default": True, "tooltip": "Search each section from where the PREVIOUS one's words actually stopped, instead of from where the plan says it begins.\n\nWhy, measured on a real take: the plan ran +1.5 s behind the performance at the first verse and +17.7 s by the second, growing steadily. A plan is not wrong at random, it is wrong PROGRESSIVELY — so the last section's error is the next section's best correction, and a window lands right on the first try instead of needing the emergency widening.\n\nThe half that matters more is the floor. A section cannot begin before the previous one stopped singing, so a section the model never sang can no longer be laid confidently on top of the one that IS being sung there. It gets searched where it belongs, finds nothing, and says so — a wrong answer becoming a reported non-answer.\n\nA section whose own confidence comes back low teaches the next one nothing, so one bad placement cannot walk the rest of the song off the music. Turn this off to read every window straight off the plan again."}),
            },
        }

    RETURN_TYPES = (T.TIMING_TYPE, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("timing", "plan", "ass_path", "srt_path", "report")
    FUNCTION = "run"
    CATEGORY = CAT_ECHO
    DESCRIPTION = ("Find where every word of a lyric was actually sung, by forced alignment against "
                   "the words themselves rather than by transcribing them. Writes a karaoke .ass, "
                   "and hands back both the timing and the plan corrected to match the performance.")

    def run(self, audio, lyrics, vocals=None, plan="", language=R.LANG_AUTO, slack=T.SLACK, write_ass=True,
            write_srt=False, filename_prefix="clips/echo", colors="", font="Arial", font_size=0,
            frame_width=1920, frame_height=1080, karaoke_sweep=True, lead_in=0.35, device="auto",
            half=True, unload_after=False, verbose=True, anchor=True, **kwargs):
        from ..siren.cast import _voices_in_order

        began = time.time()
        if audio is None or "waveform" not in audio:
            raise ValueError("[Echo] 'audio' is empty — there is nothing to align to.")
        if not str(lyrics or "").strip():
            raise ValueError("[Echo] 'lyrics' is empty. Wire the same lyric sheet that went to "
                             "Siren; the words are what makes this cheap and exact.")

        _, total = A.mono_16k(audio)
        voices = _voices_in_order(kwargs)
        job, notes = T.build(lyrics, plan, total, lang=language, slack=float(slack), voices=voices)

        # `audio` always defines the clock; `vocals` is only what gets listened to. A separator that
        # trimmed or padded would otherwise shift every timing by however much it moved, and the
        # subtitles would come out uniformly late with nothing in the report to explain it.
        heard = audio
        if vocals is not None and "waveform" in vocals:
            _, vocal_total = A.mono_16k(vocals)
            if abs(vocal_total - total) > 0.1:
                notes.append(f"the isolated vocal is {vocal_total:.2f} s and the song is "
                             f"{total:.2f} s — a separator that changes the length would shift "
                             f"every word by the difference, so THE MIX WAS LISTENED TO INSTEAD. "
                             f"If 'audio' is an instrumental stem, that is now aligning a lyric "
                             f"against music with no voice in it: put the whole song there.")
            else:
                heard = vocals
                notes.append("listened to the isolated vocal; the song's length came from 'audio'")

        if not A.is_downloaded():
            print(f"[Echo] the alignment model is not on disk yet — fetching {A.MODEL_NAME}, about "
                  f"{A.MODEL_BYTES / 1e9:.1f} GB, into {A.download_root()}. This happens once.")

        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(max(1, job["sung_blocks"]))
        except Exception:
            pass

        notes.extend(A.align_job(
            job, heard, device=None if device == "auto" else device, half=bool(half),
            anchor=bool(anchor), slack=float(slack),
            progress=(lambda done, n: pbar.update_absolute(done, n)) if pbar else None))
        if unload_after:
            A.unload()

        track = T.finish(job)
        colour_map, colour_notes = S.parse_colours(colors)
        notes.extend(colour_notes)

        # The style travels ON the timing, so `Save Clip` burns exactly what the .ass previewed.
        # Two places to set a colour is two places for them to disagree, and the whole point of
        # checking the .ass in a player is that it shows what the render will show.
        track["style"] = {"font": font, "size": int(font_size), "colors": colour_map,
                          "sweep": bool(karaoke_sweep), "lead_in": float(lead_in)}

        ass_path = srt_path = ""
        if write_ass or write_srt:
            import folder_paths
            folder, name, counter, _sub, _ = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(),
                int(frame_width), int(frame_height))
            base = os.path.join(folder, f"{name}_{counter:05}")
            if write_ass:
                ass_path = base + ".ass"
                _write(ass_path, S.ass(track, colour_map, int(frame_width), int(frame_height),
                                       font=font, size=int(font_size) or None,
                                       sweep=bool(karaoke_sweep), lead_in=float(lead_in)))
            if write_srt:
                srt_path = base + ".srt"
                _write(srt_path, S.srt(track))

        table, table_notes = _plan_table(track)
        notes.extend(table_notes)

        text = _report(track, notes, _format_elapsed(time.time() - began, "auto"))
        if verbose:
            print("[Echo] " + text.replace("\n", "\n[Echo] "))
        return (track, table, ass_path, srt_path, text)


def _write(path, body):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)


NODE_CLASS_MAPPINGS = {"KinburgEchoAlign": KinburgEchoAlign}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgEchoAlign": "Echo (Lyrics → Timing) 💬"}
