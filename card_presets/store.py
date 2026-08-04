"""Card Presets store — a saved library of filled Character / Entity cards.

Each preset keeps the card **type** ("character" / "entity"), its field **values**, and an
optional list of **tags** (free-form labels for filtering a growing library). The reader node
(Card Presets) renders the values back into the same Markdown block the card nodes emit, so a
saved character can be dropped into a workflow from a dropdown instead of re-describing the same
photo every time. Persisted to ``data/store.json``.

Cards enter the library three ways, all funnelling through :func:`upsert`:
  * **Card Save** — parse an LLM's grammar-constrained JSON straight into a preset (photo→card).
  * **Character Card / Entity Card** — their ``save_preset_as`` field (typed or wired-in values).
  * the Manage dialog — delete / retag existing entries.

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


def full_data():
    presets = _load()
    return {
        "none": NONE,
        "all_tags": ALL_TAGS,
        "order": [NONE] + sorted(presets.keys()),
        "tags": all_tags(),
        "presets": {name: {"type": p.get("type", "character"), "tags": p.get("tags") or []}
                    for name, p in presets.items()},
    }


def upsert(name, card_type, values, tags=None, delete=False):
    """Add/update (or delete) a saved card preset.

    ``tags=None`` leaves an existing preset's tags untouched (and means "no tags" on create),
    so re-saving the same card from a node that doesn't set tags never wipes tags added later.
    Pass a list / comma-separated string to set them (``[]`` clears).
    """
    name = (name or "").strip()
    if not name or name == NONE:
        raise ValueError("preset name is required")
    with _LOCK:
        presets = _load()
        if delete:
            presets.pop(name, None)
        else:
            ctype = "entity" if str(card_type).lower().startswith("entity") else "character"
            new_tags = _norm_tags(tags)
            if new_tags is None:  # keep existing tags (or none for a brand-new preset)
                new_tags = (presets.get(name) or {}).get("tags") or []
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
