"""Character Card — fill a few fields, get one tidy character block for the LLM context.

Empty fields are skipped, so you only describe what matters. The `card` output is a small
Markdown block (a heading with the name + a bullet per filled attribute + free-form notes)
meant to be gathered by Context Collector and fed into an LLM node's `context` input, e.g.

    ### Vasya
    - Gender: male
    - Age: 35
    - Eyes: brown
    - Hair color: dark brown
    - Notes: quiet, keeps to himself

Giving the model such reference cards lets it weave each named character's looks into an
expanded image prompt (see README). Category `Kinburg-Nodes/LLM`.

**The `voice` output is for music.** A band member is one card: their looks go to the cover-art
prompt, their voice goes to Siren Cast, and the same card describes them to the LLM that writes
the lyrics — one place to edit, three consumers. The reason it is a SEPARATE typed output and not
just part of the Markdown: AceStep's caption is a music-production description, and pouring
"brown eyes, navy dress" into it only dilutes the part that decides how the vocal sounds. So
`voice` carries `voice_tags` alone (plus the name, which is how a plan row refers to the member).
"""

# (widget name, display label) in output order. All are plain strings; empty ones are dropped.
_FIELDS = [
    ("gender", "Gender"),
    ("age", "Age"),
    ("ethnicity", "Ethnicity / skin"),
    ("eye_color", "Eyes"),
    ("hair_color", "Hair color"),
    ("hair_style", "Hair style"),
    ("build", "Build"),
    ("height", "Height"),
    ("outfit", "Outfit / clothing"),
    ("features", "Distinctive features"),
]

# Typed hand-off to Siren Cast: {"name", "tags", "notes"}. Declared here rather than in siren/
# so that the card nodes — and the preset store that renders them — stay free of torch.
VOICE_TYPE = "KINBURG_VOICE"

# Voice fields, rendered after the appearance block and before Notes. `voice_tags` is the only one
# that reaches AceStep; `voice_notes` exists for the LLM that writes the lyrics.
_VOICE_FIELDS = [
    ("voice_tags", "Voice (music tags)"),
    ("voice_notes", "Voice"),
]
from ..categories import CAT_LLM_CONTEXT


class CharacterCard:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {"default": "", "tooltip": "Leave empty to omit this line from the card."}
        fields = {
            "name": ("STRING", {"default": "", "tooltip": "Character name / label — becomes the card heading and the name the LLM binds attributes to."}),
        }
        for key, label in _FIELDS:
            fields[key] = ("STRING", dict(opt, tooltip=f"{label}. " + opt["tooltip"]))
        fields["voice_tags"] = ("STRING", {"default": "", "tooltip": "How this member's VOICE sounds, as a music caption fragment — this is the text Siren Cast pastes onto the song's caption for the sections they sing.\n\nWrite production language, not looks: 'male lead vocal, raspy baritone, close-mic, slight grit on the high notes'. Keep it to the voice and its delivery; genre, tempo and mix belong in Siren Cast's own 'tags', which every section shares.\n\nEmpty = this card has no voice (a character who doesn't sing) and Siren Cast will say so if a plan row asks for them."})
        fields["voice_notes"] = ("STRING", {"multiline": True, "default": "", "tooltip": "The voice in prose, for the LLM that writes the lyrics — range, habits, what they never do ('avoids anything above E4, tends to talk-sing the verses'). Goes into the card as a '- Voice:' line and NEVER into the AceStep caption, so it can be as long as you like."})
        fields["notes"] = ("STRING", {"multiline": True, "default": "", "tooltip": "Anything else (personality, accessories, scars…). Added as a '- Notes:' line at the end of the card."})
        fields["save_preset_as"] = ("STRING", {"default": "", "tooltip": "Type a name to save this card to the Card Presets library on the next run. Works whether the fields are typed OR wired in from outside (saved at run time). Empty = don't save. (Re-runs overwrite the same name.)"})
        fields["tags"] = ("STRING", {"default": "", "tooltip": "Comma-separated tags for filtering the library in Card Presets (e.g. 'heroes, medieval'). Only used when save_preset_as is set. Empty = leave existing tags untouched."})
        return {"required": fields}

    RETURN_TYPES = ("STRING", VOICE_TYPE)
    RETURN_NAMES = ("card", "voice")
    FUNCTION = "run"
    CATEGORY = CAT_LLM_CONTEXT

    def run(self, name="", notes="", save_preset_as="", tags="", **kwargs):
        lines = []
        for key, label in _FIELDS:
            v = (kwargs.get(key) or "").strip()
            if v:
                lines.append(f"- {label}: {v}")
        for key, label in _VOICE_FIELDS:
            v = (kwargs.get(key) or "").strip()
            if v:
                lines.append(f"- {label}: {v}")
        notes = (notes or "").strip()
        if notes:
            lines.append(f"- Notes: {notes}")   # same bullet format as every other field

        result = ""
        if lines:
            header = (name or "").strip() or "Character"
            result = "\n".join([f"### {header}"] + lines)

        # Always emitted, even with nothing in it: Siren Cast reports "this member has no
        # voice_tags" far more usefully than a silently missing input would.
        # `gender` rides along because Siren Score has to read a lyric marker that says
        # "[Verse 1 - FEMALE vocal]" with no name in it, and match it to a member. It reads
        # `voice_tags` first (those usually say "female lead vocal…") and falls back to this.
        voice = {"name": (name or "").strip(),
                 "tags": (kwargs.get("voice_tags") or "").strip(),
                 "notes": (kwargs.get("voice_notes") or "").strip(),
                 "gender": (kwargs.get("gender") or "").strip()}

        # Backend save: captures the values as resolved at run time (typed OR wired in), which a
        # frontend button can't see. Overwrites the same name on re-runs.
        sp = (save_preset_as or "").strip()
        if sp and result:
            try:
                from ..card_presets.store import upsert
                values = {"name": name or "", "notes": notes}
                for key, _ in _FIELDS + _VOICE_FIELDS:
                    values[key] = (kwargs.get(key) or "")
                upsert(sp, "character", values, tags=(tags if (tags or "").strip() else None))
                print(f"[CharacterCard] saved preset '{sp}'")
            except Exception as e:
                print(f"[CharacterCard] save preset '{sp}' failed: {e}")

        return (result, voice)


NODE_CLASS_MAPPINGS = {"CharacterCard": CharacterCard}
NODE_DISPLAY_NAME_MAPPINGS = {"CharacterCard": "Character Card"}
