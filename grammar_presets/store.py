"""Grammar Presets store — built-in GBNF grammars + a user-editable layer on disk.

A built-in template forces an LLM's output into a fixed shape, so feeding a **vision** model a
photo returns exactly that structure (e.g. a Character Card / Entity Card as JSON). Users add
their own named grammars (persisted to ``data/store.json``). The chosen grammar wires into a
**Local LLM (GGUF)** node's ``grammar_override`` input.

Guarded so the package still imports without ComfyUI present (registry scan, tests).
"""
import json
import os
import threading

# The "no grammar" option — resolves to "" (an empty grammar_override = no constraint).
NONE = "🚫 None"

# Built-in grammars. JSON is the most robust shape to constrain (and to parse / read as context);
# the string production below (escapes + any non-control codepoint) matches the Vision Judge's,
# so it handles any language. Fields mirror the Character Card / Entity Card nodes; the model
# fills what it can see and uses "" for the rest.
CHARACTER_CARD = r'''root ::= ws "{" ws "\"name\"" ws ":" ws string "," ws "\"gender\"" ws ":" ws string "," ws "\"age\"" ws ":" ws string "," ws "\"ethnicity\"" ws ":" ws string "," ws "\"eye_color\"" ws ":" ws string "," ws "\"hair_color\"" ws ":" ws string "," ws "\"hair_style\"" ws ":" ws string "," ws "\"build\"" ws ":" ws string "," ws "\"height\"" ws ":" ws string "," ws "\"outfit\"" ws ":" ws string "," ws "\"features\"" ws ":" ws string "," ws "\"notes\"" ws ":" ws string ws "}" ws
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt/] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
ws ::= [ \t\n]*
'''

ENTITY_CARD = r'''root ::= ws "{" ws "\"name\"" ws ":" ws string "," ws "\"description\"" ws ":" ws string ws "}" ws
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt/] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
ws ::= [ \t\n]*
'''

# Siren Cast's `plan` block, one section per line. Not JSON on purpose — the node's parser reads
# this table, and a table is also what a songwriter can read back and edit by hand.
#
# The lengths are constrained to a musical set of BAR counts rather than free numbers: a section is
# written in bars, `Siren Cast` converts them with the bpm, and "3 bars" is almost always a slip.
# Labels are constrained to the usual section names so they line up with the `[Verse 1 - ...]`
# markers in the lyrics. The voice column can only be a name (or several joined by " + "), or "-"
# for no vocal — the names have to match the `name` on the Character Cards wired into Siren Cast,
# which is what the roster in the prompt is for.
#
# **A name may contain SPACES.** The first version of this rule allowed a single word, which did not
# make a model skip a two-word member — it made it write "GruBNik" for "Gru BNik", and that then
# missed the roster and fell through to being used as plain text. A grammar that cannot express the
# right answer does not produce a refusal; it produces a confident wrong answer.
#
# **The row count is bounded and the table ends with a required END line, and both matter.** A
# grammar with an open-ended `row+` never *forces* the model to finish: it only ever *permits* EOS,
# and EOS is a low-probability option that top_p / min_p happily prune — after which the only legal
# continuation left is another row, so the model writes rows until it hits max_tokens. The `{3,16}`
# bound makes the runaway finite; the END line makes stopping a word the model is glad to write,
# after which the grammar permits nothing but EOS. `_parse_plan` treats a bare END as end-of-table.
# Raise 16 if a song genuinely needs more sections.
SIREN_PLAN = r'''root ::= row{3,16} end
row ::= label " | " voice " | " bars "\n"
end ::= "END" "\n"?
label ::= section (" " [1-9])?
section ::= "Intro" | "Verse" | "Pre-Chorus" | "Chorus" | "Post-Chorus" | "Bridge" | "Solo" | "Break" | "Outro"
voice ::= "-" | name (" + " name)*
name ::= word (" " word)*
word ::= letter (letter | [0-9'-])*
letter ::= [A-Za-zА-Яа-яЁёІіЇїЄєҐґ]
bars ::= ("2" | "4" | "6" | "8" | "12" | "16" | "24" | "32") " bars"
'''

# The song config that feeds Siren Cast — a caption block, a blank line, then the metas as
# `key: value`. Companion to SIREN_PLAN: this pass runs BEFORE the lyrics (the plan needs the
# finished text to size its sections, this needs nothing but the brief).
#
# There is deliberately NO vocals line. Per-section voices come from the plan, and describing the
# timbres again in the caption pulls every section towards their average — which is the whole
# reason a chorus sung by a woman came out sounding like the verses. Siren Cast appends the cast
# itself when `cast_in_caption` is on. For an ensemble-level line (legitimately global, since it is
# true of the whole track) add:
#     ensemble-section ::= "*Ensemble:* " textline "\n"
#
# `keyscale` and `language` are pinned to the exact values Siren Cast's combos accept — a free
# character class happily emits "C sharp minor" or "ua", neither of which is in the list (Ukrainian
# is "uk"). `bpm` is held to 60-249 so a stray digit can't produce a 6 bpm song. The genre and
# instrument lines allow digits and '&', without which "90s alt-rock" and "808" are unwritable.
SIREN_CONFIG = r'''root ::= genre-section instruments-section bpm-line timesignature-line keyscale-line songname-line language-line

genre-section       ::= "*Genre:* " textline "\n"
instruments-section ::= "*Instruments:* " textline "\n\n"
bpm-line            ::= "bpm: " bpm "\n"
timesignature-line  ::= "timesignature: " ("2" | "3" | "4" | "6") "\n"
keyscale-line       ::= "keyscale: " keyroot " " ("major" | "minor") "\n"
songname-line       ::= "songname: " [-a-zA-Z0-9 '’а-яА-ЯёЁіІїЇєЄґҐ]+ "\n"
language-line       ::= "language: " language "\n"

textline ::= [-a-zA-Z0-9,'& ]+
bpm      ::= [6-9] [0-9] | "1" [0-9] [0-9] | "2" [0-4] [0-9]
keyroot  ::= "C#" | "Db" | "D#" | "Eb" | "F#" | "Gb" | "G#" | "Ab" | "A#" | "Bb" | "C" | "D" | "E" | "F" | "G" | "A" | "B"
language ::= "en" | "ru" | "uk" | "de" | "fr" | "es" | "it" | "pl" | "pt" | "ja" | "ko" | "zh" | "yue" | "tr" | "ar" | "he" | "hi" | "unknown"
'''

DEFAULTS = {
    "Character Card (JSON)": CHARACTER_CARD,
    "Entity Card (JSON)": ENTITY_CARD,
    "Siren Song Config (text)": SIREN_CONFIG,
    "Siren Voice Plan (table)": SIREN_PLAN,
}

_LOCK = threading.Lock()


def _store_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "store.json")


def _load_user():
    """User grammars as {name: text}; {} when absent/corrupt."""
    try:
        with open(_store_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    g = data.get("grammars") if isinstance(data.get("grammars"), dict) else {}
    return {name: text for name, text in g.items() if isinstance(name, str) and isinstance(text, str)}


def _save_user(grammars):
    p = _store_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"grammars": grammars}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def grammars_merged():
    """Built-in grammars with the user's own layered on top."""
    out = dict(DEFAULTS)
    out.update(_load_user())
    return out


def list_names():
    """Dropdown values: NONE first, then built-ins, then user-added (not already built-in)."""
    names = [NONE] + list(DEFAULTS.keys())
    for name in _load_user():
        if name not in DEFAULTS:
            names.append(name)
    return names


def get(name):
    """The selected preset's grammar text ("" for NONE / unknown)."""
    if not name or name == NONE:
        return ""
    return grammars_merged().get(name, "")


def full_data():
    """Everything the frontend needs."""
    return {
        "none": NONE,
        "order": list_names(),
        "grammars": grammars_merged(),
        "builtins": list(DEFAULTS.keys()),
    }


def upsert_grammar(name, text, delete=False):
    """Add/update (or delete) a user grammar. Built-ins can't be deleted."""
    name = (name or "").strip()
    if not name or name == NONE:
        raise ValueError("grammar name is required")
    with _LOCK:
        user = _load_user()
        if delete:
            if name in DEFAULTS and name not in user:
                raise ValueError("cannot delete a built-in grammar")
            user.pop(name, None)
        else:
            if not (text or "").strip():
                raise ValueError("grammar text is required")
            user[name] = text
        _save_user(user)
    return full_data()
