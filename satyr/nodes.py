"""`Satyr Score (Plan → Plan)` — check YuE2's own plan against the lyrics, and report what it will do.

The node is thin on purpose. Everything worth arguing about lives next door, beside the reasoning for
it: `notation` reads and edits the ABC, `bands` finds the two registers and moves a phrase between
them, `rewrite` changes how many syllables a phrase holds, and `layout` decides which words go where.

**It reads far more than it writes.** The first version rewrote every phrase — moved each one toward
its singer's register, cut each one to its line's syllables — and the songs came back sparse and sung
by a single voice. What the measurements since have established is narrower and more useful than what
was hoped for:

* **How many singers a song has is decided in `style`, and nowhere else.** A style naming no voices
  is sung by one singer from beginning to end however the plan is written.
* **Where they sing follows the register once it agrees with the markers — with a LoRA on the text
  encoder.** At one seed, a plan that contradicted its markers in five sections came back in a single
  voice; the same plan with those five moved an octave was sung by exactly the singers marked. On the
  base model the same correction turned one section of four: there, register is not a control.
* **The markers in the lyrics are worth keeping.** The model takes the performance from them —
  belts, growls, strain — and, with the LoRA, who sings where. Stripping them flattens a take.

So what comes out:

* **`report`** — read this one first, and it is the reason to wire the node at all. Which lyric block
  landed on which section, what register each uses, which block got too few notes to be sung in full,
  which got none. That last one is the failure worth catching: a measured plan gave an eight-syllable
  outro no notes, and no setting on any node would have sung it.
* **`abc`** — the plan with each section that contradicts its marker moved by whole octaves
  (`recast`), and byte-for-byte identical everywhere else.
* **`lyrics`** — the words, passed through as written unless `keep_markers` is off.

**The honest limits.** How many singers there are is settled in the style string, and without the
LoRA so, mostly, is where they sing. Nothing here makes a badly-shaped plan good — a block the plan gave no notes needs another plan, not
another setting. And the plan's LENGTH, which is where YuE2 most often disappoints, is not touched
here at all: that is `Satyr Trim`.
"""
import json
import time

from ..categories import CAT_SATYR
from ..context.character_card import VOICE_TYPE
from ..siren.cast import _roster, _voices_in_order
from ..timer.timer_nodes import _format_elapsed
from . import bands as B
from . import layout as L
from . import midi as MID
from . import notation as N
from . import trim as T


class KinburgSatyrScore:
    """A YuE2 ABC plan + marked-up lyrics → the same plan with the voices and syllables fixed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"forceInput": True, "tooltip": "The plan, from 'YuE2 Generate ABC' or pasted in by hand.\n\nIt is read, not trusted: the header gives the meter, unit and tempo, and everything below is indexed by section and phrase. Lines this node does not decide to change come back byte for byte, so a diff of a run shows the phrases that moved and nothing else."}),
                "lyrics": ("STRING", {"forceInput": True, "tooltip": "The lyrics with '[Verse 1 - Keen Burg]' style markers — the same format Siren Score reads.\n\nThe section name comes FIRST, then the member's name. A bracketed line that NAMES a member inside a section starts an exchange from that point, and a line wholly in ROUND brackets is a backing vocal: it gets a phrase of its own in the OTHER singer's register, which is how the model writes one when it bothers to.\n\nWire the 'lyrics' OUTPUT of this node into YuE2. By default it is this text unchanged: the markers are worth keeping, because the model takes the performance from them."}),
                "recast": ("BOOLEAN", {"default": True, "tooltip": "Move each section whose register contradicts its marker by whole octaves, so the plan agrees with the lyrics. ON by default.\n\nWith a LoRA on the text encoder this is what decides who sings. At one seed, a plan that contradicted its markers in five sections came back in ONE voice; the same plan with those five moved was sung by exactly the singers marked, its melody, rhythm and arrangement unchanged. On the base model the same correction turned one section of four — register alone is not what it listens to.\n\nIt moves whole sections only, with the pickup each one opens on, and takes the move back if it narrows the gap between the registers."}),
                "refit": ("BOOLEAN", {"default": False, "tooltip": "Cut each phrase to as many notes as its words have syllables. OFF by default, and the default is the recommendation.\n\nThe idea is sound — the plan's note count really is the syllable budget, a measured chorus of 47 syllables had been written 47 notes — but the edit is not. Removing a note conserves the bar it lives in, so it does not free time, it STRETCHES the notes that remain: a real run lost 96 notes, pushed the vocal line from 50% silence to 57%, and came back sounding like the singer was labouring through it.\n\nWhat the model itself does is sing a line at a natural rate and rest the remainder of the bar. Until this works the same way, leave it off and shorten the song by cutting bars instead. It is kept for the case where a phrase is far too short for its line and you would rather have the words than the rhythm."}),
                "keep_markers": ("BOOLEAN", {"default": True, "tooltip": "Pass the lyrics to the 'lyrics' output exactly as written, markers and all. ON by default, because that is what sounded better.\n\nThey were being stripped on the reasoning that YuE2 has no field for a singer and its own guidance says to keep instructions out of the lyrics. Listening says otherwise: the model reads them and takes the PERFORMANCE from them — 'powerful belts', 'deep growl', 'vocal duel, intense emotional peak' come back as strain and intensity, and stripping them flattens the take. With a LoRA on the text encoder they also decide who sings where, once 'recast' has made the plan agree with them.\n\nTurn it off if your lyrics carry stage directions in ROUND brackets: YuE2 reads those as a backing vocal and sings them aloud. Square brackets are safe either way."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Print the report to the console. The same text is always on the 'report' output."}),
            },
            "optional": {
                "voice_1": (VOICE_TYPE, {"tooltip": "A band member — a Character Card's 'voice' output. Wire the same cards the markers name.\n\nThe cards decide which band each singer gets: they are ranked on one ladder of voice types (soprano … bass, with a bare 'female'/'male' sitting between the ranges each usually covers) and the upper half takes the upper band. That has to be relative — the same tenor belongs above a bass and below a soprano.\n\nWith fewer than two wired there is nothing to alternate, and no phrase is recast."}),
                "voice_2": (VOICE_TYPE,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("abc", "lyrics", "report")
    FUNCTION = "run"
    CATEGORY = CAT_SATYR
    DESCRIPTION = ("Checks a YuE2 plan against the lyrics and reports what it will actually do "
                   "with them. It matches each lyric block to the section that carries it — by note "
                   "and syllable counts, because the model merges blocks and inserts wordless "
                   "sections, so pairing them in order gets nearly every one wrong — then says "
                   "which block is short of notes and which will not be sung at all. It also moves "
                   "each section whose register contradicts its marker by an octave: with a LoRA on "
                   "the text encoder, that is what brings every section back in the voice marked for "
                   "it. Cutting notes to fit the words stays off, because it stretches the ones that "
                   "remain. Read the report before rendering; most of what goes wrong with a plan "
                   "cannot be edited out, only caught early.")

    def run(self, abc, lyrics, recast, refit, keep_markers=True, verbose=True, **kwargs):
        started = time.time()
        if not str(abc or "").strip():
            raise RuntimeError(
                "[Satyr Score] the 'abc' input is empty. Wire it from 'YuE2 Generate ABC', or paste "
                "a plan in — this node rewrites a score, it does not compose one.")
        score = N.parse(abc)
        if not score.sections:
            raise RuntimeError(
                "[Satyr Score] nothing in 'abc' reads as a YuE2 score. It should open with X:/T:/M:/"
                "L:/Q:, declare a Vocal and an Ins voice, and then alternate '% section' labels with "
                "'V: Vocal' / 'V: Ins' music lines.")

        voices = _voices_in_order(kwargs)
        _, roster_notes = _roster(voices)
        broken = N.problems(score)
        blocks, _ = L.blocks(lyrics, voices)
        if not blocks:
            raise RuntimeError(
                "[Satyr Score] no section markers found in the lyrics. A section starts at a line "
                "like '[Verse 1 - Keen Burg]' — a bracketed line whose FIRST word is a section name "
                "(Intro / Verse / Pre-Chorus / Chorus / Bridge / Outro, or a synonym).")

        score, spots, notes = L.plan(score, lyrics, voices, refit=refit, recast=recast)
        report = L.report(score, spots, roster_notes + broken + notes)
        report += f"\n\n{_format_elapsed(time.time() - started, 'auto')}"
        if verbose:
            print("[Satyr Score]\n" + report)
        words = lyrics if keep_markers else L.strip_markers(blocks)
        return score.text(), words, report


class KinburgSatyrRead:
    """A YuE2 plan → its register map. Reads only; changes nothing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"forceInput": True, "tooltip": "A YuE2 plan, from 'YuE2 Generate ABC' or anywhere else."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Print the map to the console. The same text is always on the 'report' output."}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("report", "seconds")
    FUNCTION = "run"
    CATEGORY = CAT_SATYR
    DESCRIPTION = ("Shows what a YuE2 plan actually says: how long the song is in seconds, whether "
                   "it was written for one singer or two, where the two registers sit, and for "
                   "every phrase its timing, its syllable budget and which voice takes it. Read "
                   "this before changing anything — it is the output that says whether a plan has "
                   "two separated voices at all, and a plan with only one band has no female part "
                   "to move, however the style was written.")

    def run(self, abc, verbose=True):
        score = N.parse(abc)
        if not score.sections:
            raise RuntimeError("[Satyr Read] nothing in 'abc' reads as a YuE2 score.")
        phrases = B.read(score)
        report = B.report(score, phrases, B.find(phrases))
        for problem in N.problems(score):
            report += "\n! " + problem
        if verbose:
            print("[Satyr Read]\n" + report)
        return report, score.duration()


class KinburgSatyrTrim:
    """A YuE2 plan → the same plan with its silent stretches cut to length."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"forceInput": True, "tooltip": "The plan, from 'YuE2 Generate ABC' or from 'Satyr Score'."}),
                "intro_seconds": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 180.0, "step": 0.5, "tooltip": "How long the opening may run before anyone sings.\n\nThis is where the time goes. A measured plan opened with 25 bars — 40 seconds — of instrumental before the first word, and YuE2 writes one like that often. Cutting it is the single biggest thing that can be done to a plan, and it costs nothing: the bars removed hold no sung note.\n\nThe LAST bars are kept, not the first, because an introduction leads into the singing and its final bars are the approach.\n\n0 removes the introduction entirely. A number larger than it is leaves it alone."}),
                "outro_seconds": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 180.0, "step": 0.5, "tooltip": "How long the ending may run after the last word. A measured plan ran 29 seconds there.\n\nThe FIRST bars are kept: an outro leads out of the singing, and what the model writes at the end of one is usually its own figure repeated.\n\n0 removes it entirely."}),
                "between_seconds": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 180.0, "step": 0.5, "tooltip": "How long any instrumental stretch BETWEEN sung sections may run — an interlude, a break, a drop, a solo.\n\n0 removes them. These are usually short already, and they are also the ones a song most misses when they go, so this is the one to leave generous."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Print the report to the console. The same text is always on the 'report' output."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("abc", "report", "seconds")
    FUNCTION = "run"
    CATEGORY = CAT_SATYR
    DESCRIPTION = ("Shortens a YuE2 plan by cutting the stretches where nobody sings. A measured "
                   "plan ran 4:33 for a lyric AceStep sang in 3:00, and 61 of its 171 bars — 36% of "
                   "the song — carried no sung note at all: a 40-second introduction, a 29-second "
                   "outro. Dropping only those brought it to 3:45 with all 326 of its sung notes "
                   "untouched, and the take came back livelier. Nothing sung is ever removed, the "
                   "two singers are left exactly where the model put them, and the report proves "
                   "the note count did not change. Wire 'seconds' into 'Empty YuE2 Latent Audio'.")

    def run(self, abc, intro_seconds, outro_seconds, between_seconds, verbose=True):
        if not str(abc or "").strip():
            raise RuntimeError(
                "[Satyr Trim] the 'abc' input is empty. Wire it from 'YuE2 Generate ABC' or "
                "'Satyr Score'.")
        before = N.parse(abc)
        if not before.sections:
            raise RuntimeError("[Satyr Trim] nothing in 'abc' reads as a YuE2 score.")
        after, rows = T.trim(N.parse(abc), intro_seconds, outro_seconds, between_seconds)
        lost = T.sung_notes(before) - T.sung_notes(after)
        if lost:
            raise RuntimeError(
                f"[Satyr Trim] the cut would have removed {lost} sung note(s), which it must never "
                f"do. Nothing was changed — this is a bug, please report the plan that caused it.")
        report = T.report(before, after, rows)
        if verbose:
            print("[Satyr Trim]\n" + report)
        return after.text(), report, after.duration()


#: Frames of music per second. YuE2's own constant; read from core at call time rather than imported
#: at module scope, so a ComfyUI without YuE2 still loads the rest of this pack.
def _frames_per_second():
    try:
        from comfy.text_encoders.yue2 import FRAMES_PER_SECOND
    except Exception as exc:                                        # pragma: no cover - env specific
        raise RuntimeError(
            "[Satyr Music] this ComfyUI has no YuE2 support (comfy.text_encoders.yue2 is missing), "
            "so there is nothing for this node to drive. Update ComfyUI.") from exc
    return FRAMES_PER_SECOND


class KinburgSatyrMusic:
    """`YuE2 Generate Music` with the guidance scale its own reference implementation exposes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The YuE2 checkpoint's CLIP output — the same one 'YuE2 Generate Music' takes."}),
                "style": ("STRING", {"multiline": True, "dynamic_prompts": True, "tooltip": "Language, genre, vocal character, tempo, instruments. One description for the whole song: YuE2 has no way to say that the style changes partway through.\n\nThis is the field 'cfg_scale' amplifies, so it is the field to be precise in. 'dual vocals, powerful melodic female vocal, gritty male vocal' is what puts two singers in the song at all — without it a plan written for two registers still tends to come back in one voice."}),
                "lyrics": ("STRING", {"multiline": True, "dynamic_prompts": True, "tooltip": "The words. Wire the 'lyrics' output of 'Satyr Score' here: by default that is the text as written, square-bracket markers and all, and they are worth keeping — the model takes the performance from them. Round brackets are the exception: YuE2 sings whatever is inside them.\n\nAmplified by 'cfg_scale' along with the style."}),
                "abc": ("STRING", {"default": "", "multiline": True, "tooltip": "The plan. Leave it empty and the node falls back to 'off' mode exactly as the core node does.\n\nNote that the ABC is NOT amplified by 'cfg_scale': it sits in both branches of the guidance and cancels out. Raising the scale therefore pushes the style and the words harder while leaving the plan's own authority where it was."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "mode": (["full", "melody"], {"tooltip": "full: the plan carries chords. melody: melody only, for covers. An empty 'abc' ignores this and uses 'off'."}),
                "max_duration": ("FLOAT", {"default": 360.0, "min": 0.04, "max": 900.0, "step": 0.04, "tooltip": "Ceiling in seconds. Generation usually stops earlier — the plan's own length is what decides, which is why 'Satyr Trim' is the node that actually shortens a song."}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05, "tooltip": "How hard to push the style and the lyrics. 1.0 is off, and off is what the core node hardwires — this is the only reason this node exists.\n\nWhat it measurably does is DICTION. Words the model tends to swallow come through; the clearest case is a line in round brackets, which it otherwise sings too quietly or drops altogether. Raise this when the take is right but a phrase is mumbled.\n\nWhat it does not do is change who sings. That was the hope, and it was tested, and it does not.\n\nWhy diction and not voices: the negative branch is the instruction plus the ABC, with [Tags] and [Lyrics] removed, so the plan sits in both branches and cancels out of the difference. The scale amplifies the words and the style description against a plan whose authority is unchanged — it makes the model articulate what it was given rather than rearrange it.\n\nCost: anything other than 1.0 runs two branches, so roughly twice the VRAM and twice the time."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "advanced": True}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01, "advanced": True}),
                "top_k": ("INT", {"default": 100, "min": 1, "max": 32768, "advanced": True}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.01, "max": 10.0, "step": 0.01, "advanced": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "FLOAT")
    RETURN_NAMES = ("conditioning", "seconds")
    FUNCTION = "run"
    CATEGORY = CAT_SATYR
    DESCRIPTION = ("A drop-in replacement for 'YuE2 Generate Music' that exposes cfg_scale, which "
                   "YuE2's own reference implementation takes (0-20) and the ComfyUI node hardwires "
                   "to 1.0 — meaning guidance is switched off entirely. The negative branch keeps "
                   "the ABC and drops the style and the lyrics, so the scale amplifies those two "
                   "against a plan that stays at full strength in both branches. Everything else "
                   "behaves exactly like the core node. Above 1.0 it runs two branches: twice the "
                   "VRAM, twice the time.")

    def run(self, clip, style, lyrics, abc, seed, mode, max_duration, cfg_scale,
            temperature, top_p, top_k, repetition_penalty):
        rate = _frames_per_second()
        if not str(abc or "").strip():
            mode = "off"                       # exactly what the core node does with an empty plan
        tokens = clip.tokenize(style, lyrics=lyrics, cot=mode, seed=seed, abc=abc,
                               max_tokens=max(1, round(max_duration * rate)),
                               temperature=temperature, top_p=top_p, top_k=top_k,
                               repetition_penalty=repetition_penalty, cfg_scale=cfg_scale)
        if tokens.get("cfg_scale") != cfg_scale:
            raise RuntimeError(
                "[Satyr Music] this ComfyUI's YuE2 tokenizer ignored cfg_scale, so the node cannot "
                "do the one thing it is for. Use the core 'YuE2 Generate Music' node instead.")
        if cfg_scale != 1.0:
            print(f"[Satyr Music] guidance {cfg_scale} — two branches, so expect about twice the "
                  f"VRAM and twice the time of the core node.")
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return conditioning, conditioning[0][1]["yue2_frames"] / rate


def _find(path):
    """A MIDI path the way people actually give one: absolute, or a name in ComfyUI's input folder."""
    import os
    raw = str(path or "").strip().strip('"')
    if not raw:
        raise RuntimeError("[Satyr Import] no file given. Put a .mid path here, or the name of one "
                           "in ComfyUI's input folder.")
    if os.path.isfile(raw):
        return raw
    try:
        import folder_paths
        inside = os.path.join(folder_paths.get_input_directory(), raw)
    except Exception:
        inside = raw
    if os.path.isfile(inside):
        return inside
    raise RuntimeError(f"[Satyr Import] no file at {raw!r}, and none by that name in ComfyUI's "
                       f"input folder either.")


class KinburgSatyrImport:
    """A MIDI file → a YuE2 plan, with an honest account of what the conversion cost."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_path": ("STRING", {"default": "", "tooltip": "A .mid file: a full path, or just the name if it sits in ComfyUI's input folder.\n\nWhat it needs to contain is a melody. Everything else — tempo, time signature, key, and section labels from MIDI markers — is read if present and reported if missing."}),
                "vocal_track": ("STRING", {"default": "", "tooltip": "Which track becomes the sung line. A track name, or its number counting from 0. Leave empty and the node guesses: a name containing vocal / lead / melody wins, otherwise the first track with notes.\n\nThe report lists every track with its note count and pitch range, so one run tells you what to type here."}),
                "ins_track": ("STRING", {"default": "", "tooltip": "Which track becomes the instrumental line. Same rules. Leave empty for the first track the vocal did not take; leave it pointing at nothing and the Ins voice is written as rests, which is perfectly valid."}),
                "style_prefix": ("STRING", {"default": "", "multiline": True, "tooltip": "What the MIDI cannot know, put here: language, genre, vocal character, mood. The node appends what it CAN know — the tempo and the instruments General MIDI names — and hands back one style string.\n\nYuE2 asks for 'Language + Genre + Vocal Character + Tempo + Instruments', and a MIDI answers the last two exactly. It has nothing to say about the first three, and guessing them from note data would be making things up.\n\nLeave it empty to get the tempo and the band on their own, and join them to your own text however you like."}),
                "grid": (["auto", "1/4", "1/8", "1/16", "1/32"], {"tooltip": "The note grid everything is quantised onto — ABC's L: value.\n\nauto picks the coarsest grid the music actually fits, which keeps the plan readable: a 1/32 grid can write any rhythm and produces a line nobody can check. Set it by hand when the guess reads wrong, and read the report either way — it says how far the notes had to move to land on it.\n\nA MIDI quantised in a DAW converts exactly. A live take does not, and the report says so rather than pretending."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Print the report to the console. The same text is always on the 'report' output."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("abc", "style", "report", "seconds")
    FUNCTION = "run"
    CATEGORY = CAT_SATYR
    DESCRIPTION = ("Turns a MIDI file into a YuE2 plan, so a melody written in a DAW can be the one "
                   "the model sings. This is the only real control over the tune: who sings is "
                   "settled by the style string and the plan gets no say, but what they sing is "
                   "written here. Two tracks become the Vocal and Ins voices, notes are quantised "
                   "onto a grid chosen from the music, lengths the format cannot write are spelled "
                   "with ties, and a note crossing a barline stays ONE syllable. Everything lossy — "
                   "notes moved to reach the grid, chords flattened to their top note, a tempo map "
                   "reduced to one number — is counted in the report. MIDI carries no chord symbols, "
                   "so use mode 'melody' in YuE2, or wire this through 'Satyr Score' first to check "
                   "the tune against your lyrics. It also reads the band out of the file: the "
                   "tempo and every instrument General MIDI names go to the 'style' output, "
                   "appended to whatever you put in 'style_prefix'.")

    def run(self, midi_path, style_prefix, vocal_track, ins_track, grid, verbose=True):
        chosen = 0 if grid == "auto" else int(grid.split("/")[1])
        path = _find(midi_path)
        text, said = MID.convert(path, vocal=vocal_track, ins=ins_track, grid=chosen)
        heard = MID.style_of(MID.read(path)[0])
        style = ", ".join(x for x in (str(style_prefix or "").strip().rstrip(","), heard) if x)
        said.append(f"style from the file: {heard}")
        score = N.parse(text)
        broken = N.problems(score)
        report = "\n".join(
            [f"{score.bar_count()} bars = {int(score.duration() // 60)}:{score.duration() % 60:04.1f}"
             f" at {score.bpm} bpm, {score.meter[0]}/{score.meter[1]}, key {score.key}", ""]
            + ["  " + line for line in said]
            + ([""] + ["! " + problem for problem in broken] if broken else []))
        if broken:
            raise RuntimeError("[Satyr Import] the conversion produced a score that breaks the "
                               "format, which is a bug rather than bad input:\n" + report)
        if verbose:
            print("[Satyr Import]\n" + report)
        return text, style, report, score.duration()


def _plan_state(text):
    """The editor's carrier: `{"edited": plan text, "upstream": the last plan that came in}`."""
    try:
        state = json.loads(text or "{}")
    except ValueError:
        return {}
    return state if isinstance(state, dict) else {}


class KinburgSatyrEdit:
    """A YuE2 plan → the plan as you left it in the piano-roll editor (✏ on the node)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"forceInput": True, "lazy": True, "tooltip": "The plan to edit — from 'YuE2 Generate ABC', 'Satyr Trim' or 'Satyr Import'.\n\nRun the graph up to this node once and the plan is on the node for the editor to open. With 'use_edited' on this input is not evaluated at all, so whatever feeds it does not run — the same freeze as Show Text's saved text."}),
                "use_edited": ("BOOLEAN", {"default": False, "label_on": "🔒 edited plan (upstream not run)", "label_off": "upstream plan", "tooltip": "Which plan goes out.\n\nOn: the one you saved in the editor, and the upstream is not run at all. Saving in the editor turns this on.\n\nOff: the plan from the 'abc' input, passed through. Your edit is kept on the node — turn this back on to use it again."}),
                "plan_state": ("STRING", {"default": "", "tooltip": "The edited plan and the last plan that came in, kept by the editor. Not edited by hand."}),
            },
            "optional": {
                "lyrics": ("STRING", {"forceInput": True, "tooltip": "The lyrics with '[Verse 1 - Keen Burg]' markers — the same text Satyr Score reads. Only for the editor: it shows which syllable each Vocal note sings, which singer each section is marked for, and how many notes each section has for its words. The plan going out is not changed by it.\n\nThe words are laid onto the plan exactly the way Satyr Score lays them, so the editor and that node's report agree."}),
                "voice_1": (VOICE_TYPE, {"tooltip": "A band member — a Character Card's 'voice' output, the same cards Satyr Score takes. With two wired, the editor names the singer of each voice band and marks a section whose register contradicts its marker."}),
                "voice_2": (VOICE_TYPE,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("abc", "report", "seconds")
    FUNCTION = "run"
    CATEGORY = CAT_SATYR
    DESCRIPTION = ("Edit a YuE2 plan by hand in a piano roll — both voices, the chords, the bars and "
                   "the sections — and hear it before YuE2 does. Run the graph up to this node, press "
                   "✏ on it, change what you want and save: the edited plan goes out from then on and "
                   "the upstream is not run again. Everything you did not touch is written back "
                   "exactly as the model wrote it; what you did touch is written the way ComfyUI's "
                   "own exporter writes plans, which is the text YuE2 learned from. The two voice "
                   "bands are drawn and each phrase is coloured by its band — that shows who the "
                   "model wrote it for, not a switch: how many singers a song has is decided by "
                   "the style string.")

    def check_lazy_status(self, abc=None, use_edited=False, plan_state="", **kwargs):
        if use_edited and str(_plan_state(plan_state).get("edited") or "").strip():
            return []
        return ["abc"]

    def run(self, abc=None, use_edited=False, plan_state="", lyrics=None, **voices):
        # Core's SheetSage2 exporter, read at call time like Satyr Music's YuE2 constant, so a
        # ComfyUI without it still loads the rest of this pack.
        from . import edit as ED

        state = _plan_state(plan_state)
        edited = str(state.get("edited") or "")
        if use_edited and edited.strip():
            plan, source = edited, "edited plan — the upstream was not run"
            if state.get("editedReport"):
                source += f"\nlast save: {state['editedReport']}"
        else:
            if not str(abc or "").strip():
                raise RuntimeError("[Satyr Edit] the 'abc' input is empty. Wire it from 'YuE2 Generate "
                                   "ABC', 'Satyr Trim' or 'Satyr Import'.")
            plan = abc
            source = ("upstream plan — 'use_edited' is on, but nothing has been saved from the editor yet"
                      if use_edited else "upstream plan")
        score = N.parse(plan)
        if not score.sections:
            raise RuntimeError("[Satyr Edit] nothing in the plan reads as a YuE2 score.")
        model = ED.load(plan)
        summary = (f"{len(model['bars'])} bars = {int(model['seconds'] // 60)}:{model['seconds'] % 60:04.1f}"
                   f" at {score.bpm} bpm, {score.meter[0]}/{score.meter[1]}, key {score.key}")
        report = "\n".join([source, summary] + ["! " + p for p in N.problems(score)]
                           + ["  " + w for w in model["warnings"]])
        upstream = abc if abc is not None and str(abc).strip() else None
        # The lyrics and the voices go to the editor with every run — wired or not, so unwiring them
        # clears them there too.
        ui = {"satyr_plan": [{"upstream": upstream, "summary": summary if upstream is not None else None,
                              "lyrics": str(lyrics or ""), "voices": _voices_in_order(voices)}]}
        return {"ui": ui, "result": (plan, report, model["seconds"])}


NODE_CLASS_MAPPINGS = {
    "KinburgSatyrScore": KinburgSatyrScore,
    "KinburgSatyrRead": KinburgSatyrRead,
    "KinburgSatyrTrim": KinburgSatyrTrim,
    "KinburgSatyrMusic": KinburgSatyrMusic,
    "KinburgSatyrImport": KinburgSatyrImport,
    "KinburgSatyrEdit": KinburgSatyrEdit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KinburgSatyrScore": "Satyr Score (Plan → Plan) 🐐",
    "KinburgSatyrRead": "Satyr Read (Plan → Map) 🐐",
    "KinburgSatyrTrim": "Satyr Trim (Plan → Plan) 🐐",
    "KinburgSatyrMusic": "Satyr Music (Guided YuE2) 🐐",
    "KinburgSatyrImport": "Satyr Import (MIDI → Plan) 🐐",
    "KinburgSatyrEdit": "Satyr Edit (Plan → Plan) 🐐",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
