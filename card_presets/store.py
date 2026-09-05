"""Card Presets store — a saved library of filled Character / Entity cards.

Each preset keeps the card **type** ("character" / "entity"), its field **values**, and an
optional list of **tags** (free-form labels for filtering a growing library). The reader node
(Card Presets) renders the values back into the same Markdown block the card nodes emit, so a
saved character can be dropped into a workflow from a dropdown instead of re-describing the same
photo every time. Persisted to ``data/store.json``.

Cards enter the library three ways, all funnelling through :func:`upsert`:
  * **Card Save** — parse an LLM's grammar-constrained JSON straight into a preset (photo→card).
  * **Character Card / Entity Card** — their ``save_preset_as`` field (typed or wired-in values).
  * the Manage dialog — create / edit / rename / duplicate / retag / delete entries.

Guarded so the package still imports without ComfyUI present (registry scan, tests).
"""
import json
import os
import threading

NONE = "🚫 None"
ALL_TAGS = "🏷 All"  # loader filter sentinel — "don't filter by tag"

_LOCK = threading.Lock()


def _store_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "store.json")


def _norm_tags(tags):
    """Normalise tags to a deduped (case-insensitive) list of trimmed strings.

    ``None`` is passed through unchanged — callers use it to mean "leave existing tags alone".
    A comma-separated string is split; a list/tuple is taken as-is.
    """
    if tags is None:
        return None
    if isinstance(tags, str):
        tags = tags.split(",")
    out, seen = [], set()
    for t in tags or []:
        t = str(t).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def _load():
    """{name: {"type": "character"|"entity", "values": {...}, "tags": [...]}}."""
    try:
        with open(_store_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
    out = {}
    for name, p in presets.items():
        if isinstance(name, str) and isinstance(p, dict) and isinstance(p.get("values"), dict):
            out[name] = {
                "type": p.get("type") or "character",
                "values": p["values"],
                "tags": _norm_tags(p.get("tags")) or [],  # tolerate old presets with no tags
            }
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


def all_tags():
    """Every distinct tag across the library, sorted (case-insensitive)."""
    seen = {}
    for p in _load().values():
        for t in p.get("tags") or []:
            seen.setdefault(t.lower(), t)
    return [seen[k] for k in sorted(seen)]


def get(name):
    return _load().get(name)


# Fields the Manage dialog's editor draws, per card type. Read off the card nodes' OWN
# INPUT_TYPES so the editor gains a field the day a card node does, in the same order the block
# renders — and labelled exactly as the rendered bullet ("Eyes", not "eye_color").
_EDITOR_SKIP = ("save_preset_as", "tags")   # saving IS the dialog; tags have their own box


def _fields_of(cls, labels):
    out = []
    for key, spec in (cls.INPUT_TYPES().get("required") or {}).items():
        if key in _EDITOR_SKIP:
            continue
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        out.append({
            "key": key,
            "label": labels.get(key) or key.replace("_", " ").capitalize(),
            "multiline": bool(opts.get("multiline")),
            "tooltip": opts.get("tooltip") or "",
        })
    return out


def field_schema():
    """``{"character": [{key,label,multiline,tooltip}, …], "entity": [...]}`` for the editor."""
    try:
        from ..context.character_card import CharacterCard, _FIELDS, _VOICE_FIELDS
        from ..context.entity_card import EntityCard
    except Exception:  # pragma: no cover - import guard (registry scan without the pack)
        return {"character": [], "entity": []}
    char_labels = {"name": "Name", "notes": "Notes", **dict(_FIELDS), **dict(_VOICE_FIELDS)}
    return {
        "character": _fields_of(CharacterCard, char_labels),
        "entity": _fields_of(EntityCard, {"name": "Name", "description": "Description"}),
    }


def render_values(card_type, values):
    """Render raw card *values* into their Markdown block, via the card nodes' own logic.

    Shared by :func:`render` (saved preset) and the Card Save node (fresh JSON), so the block
    format always matches what the card nodes produce. Unknown extra keys are ignored; a stray
    ``save_preset_as`` is stripped so rendering never triggers a save.
    """
    values = {k: v for k, v in (values or {}).items() if k != "save_preset_as"}
    if str(card_type).lower().startswith("entity"):
        from ..context.entity_card import EntityCard
        return EntityCard().run(name=values.get("name", ""),
                                description=values.get("description", ""))[0]
    from ..context.character_card import CharacterCard
    return CharacterCard().run(**values)[0]


def render(name):
    """Render a saved preset back into its card Markdown block ("" for NONE / unknown)."""
    if not name or name == NONE:
        return ""
    p = _load().get(name)
    if not p:
        return ""
    return render_values(p.get("type"), p.get("values") or {})


def voice(name):
    """A saved character preset's `voice` block, for Siren Cast — ``None`` for NONE / unknown / an
    entity preset. Goes through the card node too, so a preset saved before the voice fields
    existed simply comes back with empty tags rather than raising."""
    if not name or name == NONE:
        return None
    p = _load().get(name)
    if not p or str(p.get("type", "")).lower().startswith("entity"):
        return None
    values = {k: v for k, v in (p.get("values") or {}).items() if k != "save_preset_as"}
    from ..context.character_card import CharacterCard
    return CharacterCard().run(**values)[1]


def full_data():
    presets = _load()
    return {
        "none": NONE,
        "all_tags": ALL_TAGS,
        "order": [NONE] + sorted(presets.keys()),
        "tags": all_tags(),
        "presets": {name: {"type": p.get("type", "character"), "tags": p.get("tags") or []}
                    for name, p in presets.items()},
        # Values are deliberately NOT here: the dialog fetches the one card it is about to edit.
        "schema": field_schema(),
    }


def upsert(name, card_type, values, tags=None, delete=False, old_name=None):
    """Add/update (or delete) a saved card preset.

    ``tags=None`` leaves an existing preset's tags untouched (and means "no tags" on create),
    so re-saving the same card from a node that doesn't set tags never wipes tags added later.
    Pass a list / comma-separated string to set them (``[]`` clears).

    ``old_name`` renames: the entry is written under ``name`` and the old key dropped in the same
    lock, inheriting its tags when none are given. One call, so a crash between two can't leave
    the library holding both copies.
    """
    name = (name or "").strip()
    if not name or name == NONE:
        raise ValueError("preset name is required")
    old_name = (old_name or "").strip()
    with _LOCK:
        presets = _load()
        if delete:
            presets.pop(name, None)
        else:
            ctype = "entity" if str(card_type).lower().startswith("entity") else "character"
            prev = presets.get(name) or (presets.get(old_name) if old_name else None) or {}
            if old_name and old_name != name:
                presets.pop(old_name, None)
            new_tags = _norm_tags(tags)
            if new_tags is None:  # keep existing tags (or none for a brand-new preset)
                new_tags = prev.get("tags") or []
            presets[name] = {
                "type": ctype,
                "values": values if isinstance(values, dict) else {},
                "tags": new_tags,
            }
        _save(presets)
    return full_data()


def retag(name, tags):
    """Set an existing preset's tags without touching its values (Manage-dialog edit)."""
    name = (name or "").strip()
    with _LOCK:
        presets = _load()
        if name in presets:
            presets[name]["tags"] = _norm_tags(tags) or []
            _save(presets)
    return full_data()
