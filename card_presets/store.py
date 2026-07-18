"""Card Presets store — a saved library of filled Character / Entity cards.

Each preset keeps the card **type** ("character" / "entity") and its field **values**; the
reader node (Card Presets) renders them back into the same Markdown block the card nodes emit,
so a saved character can be dropped into a workflow from a dropdown instead of re-describing the
same photo every time. Persisted to ``data/store.json``.

Guarded so the package still imports without ComfyUI present (registry scan, tests).
"""
import json
import os
import threading

NONE = "🚫 None"

_LOCK = threading.Lock()


def _store_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "store.json")


def _load():
    """{name: {"type": "character"|"entity", "values": {...}}}."""
    try:
        with open(_store_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
    out = {}
    for name, p in presets.items():
        if isinstance(name, str) and isinstance(p, dict) and isinstance(p.get("values"), dict):
            out[name] = {"type": p.get("type") or "character", "values": p["values"]}
    return out


def _save(presets):
    p = _store_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"presets": presets}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def names():
    """Dropdown values: NONE first, then saved preset names (sorted)."""
    return [NONE] + sorted(_load().keys())


def get(name):
    return _load().get(name)


def render(name):
    """Render a saved preset back into its card Markdown block ("" for NONE / unknown)."""
    if not name or name == NONE:
        return ""
    p = _load().get(name)
    if not p:
        return ""
    values = p.get("values") or {}
    # Render with the card nodes' own logic so the format always matches.
    if p.get("type") == "entity":
        from ..context.entity_card import EntityCard
        return EntityCard().run(name=values.get("name", ""),
                                description=values.get("description", ""))[0]
    from ..context.character_card import CharacterCard
    return CharacterCard().run(**values)[0]


def full_data():
    presets = _load()
    return {
        "none": NONE,
        "order": [NONE] + sorted(presets.keys()),
        "presets": {name: {"type": p.get("type", "character")} for name, p in presets.items()},
    }


def upsert(name, card_type, values, delete=False):
    """Add/update (or delete) a saved card preset."""
    name = (name or "").strip()
    if not name or name == NONE:
        raise ValueError("preset name is required")
    with _LOCK:
        presets = _load()
        if delete:
            presets.pop(name, None)
        else:
            ctype = "entity" if str(card_type).lower().startswith("entity") else "character"
            presets[name] = {"type": ctype, "values": values if isinstance(values, dict) else {}}
        _save(presets)
    return full_data()
