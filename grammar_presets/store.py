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

DEFAULTS = {
    "Character Card (JSON)": CHARACTER_CARD,
    "Entity Card (JSON)": ENTITY_CARD,
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
