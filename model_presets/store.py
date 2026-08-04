"""Model Library store — model bundles (captured loader subgraphs) + their sampler presets.

Two things live here, keyed by a stable ``model_id`` the user chooses once:

  * the **bundle** — a recipe: the slice of prompt graph that assembles this model, exactly as
    Model Capture read it out of the graph (see replay.py for the shape and why it replays).
  * its **presets** — saved Sampler Settings chains (the ``SAMPLER_CFG`` list), plus the size the
    model likes and, when Settings Save was given them, the measured score / seconds that make
    "which settings are actually good here" a fact rather than a memory.

Presets come in two flavours so the same knowledge isn't copied per model:
  * a model's **own** presets, and
  * **shared** presets tagged with one or more ``families`` — visible to every model that declares
    a matching family (``flow-1024`` and friends). :func:`presets_for` merges both, own winning on
    a name clash.

A preset may carry ``overrides``: ``{"<class_type>.<input>": value}`` applied to the bundle's patch
nodes at load time, so "same model, shift 3 vs shift 5" is two presets over one bundle instead of
two near-identical bundles. Keyed by class rather than node id because re-capturing a bundle
renumbers its nodes; an override hits every node of that class in the recipe.

Persisted to ``data/store.json``. Guarded so the package still imports without ComfyUI present.
"""
import copy
import difflib
import json
import os
import threading

NONE = "🚫 None"
ALL_FAMILIES = "🏷 All"

_LOCK = threading.Lock()


def _store_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "store.json")


def _norm_list(items):
    """Deduped (case-insensitively), trimmed list of strings. ``None`` passes through as None so
    callers can mean "leave what's there alone"; a comma-separated string is split."""
    if items is None:
        return None
    if isinstance(items, str):
        items = items.split(",")
    out, seen = [], set()
    for it in items or []:
        it = str(it).strip()
        if it and it.lower() not in seen:
            seen.add(it.lower())
            out.append(it)
    return out


def _norm_preset(p):
    """Tolerate presets written by an older version (or by hand) without KeyErroring later."""
    if not isinstance(p, dict):
        return None
    stages = p.get("stages")
    if isinstance(stages, dict):          # a single stage, not yet a chain
        stages = [stages]
    if not isinstance(stages, list):
        stages = []
    out = {
        "stages": [s for s in stages if isinstance(s, dict)],
        "overrides": p.get("overrides") if isinstance(p.get("overrides"), dict) else {},
        "families": _norm_list(p.get("families")) or [],
        "tags": _norm_list(p.get("tags")) or [],
        "notes": str(p.get("notes") or ""),
        "default": bool(p.get("default")),
    }
    for k in ("width", "height"):
        try:
            v = int(p.get(k) or 0)
        except (TypeError, ValueError):
            v = 0
        out[k] = max(0, v)
    for k in ("score", "seconds"):
        try:
            out[k] = float(p[k]) if p.get(k) is not None else None
        except (TypeError, ValueError):
            out[k] = None
    return out


def _norm_model(m):
    if not isinstance(m, dict):
        return None
    recipe = m.get("recipe") if isinstance(m.get("recipe"), dict) else {}
    presets = {}
    for name, p in (m.get("presets") or {}).items():
        np = _norm_preset(p)
        if isinstance(name, str) and name and np is not None:
            presets[name] = np
    # No display name: `model_id` is free-form and is what every picker, report and dump shows, so a
    # second name would only be a thing to keep in sync. A `label` left by an earlier version is
    # dropped here on the next write.
    return {
        "families": _norm_list(m.get("families")) or [],
        "tags": _norm_list(m.get("tags")) or [],
        "notes": str(m.get("notes") or ""),
        "recipe": {"nodes": recipe.get("nodes") or {}, "outputs": recipe.get("outputs") or {}},
        "presets": presets,
    }


def _load():
    try:
        with open(_store_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    models = {}
    for mid, m in (data.get("models") or {}).items():
        nm = _norm_model(m)
        if isinstance(mid, str) and mid and nm is not None:
            models[mid] = nm
    shared = {}
    for name, p in (data.get("shared") or {}).items():
        np = _norm_preset(p)
        if isinstance(name, str) and name and np is not None:
            shared[name] = np
    return {"models": models, "shared": shared}


def _write(data):
    p = _store_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------ reading
def model_names():
    """Dropdown values for the model picker: NONE first, then the saved ids."""
    return [NONE] + sorted(_load()["models"].keys())


def get_model(model_id):
    return _load()["models"].get(model_id)


def fingerprint():
    """A cheap token that changes whenever the library file does — for nodes whose ``IS_CHANGED``
    can't identify the exact entry it depends on (see settings_node.IS_CHANGED). Missing file is a
    valid state, not an error: it just means an empty library."""
    try:
        st = os.stat(_store_path())
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "absent"


def all_families():
    seen = {}
    data = _load()
    for m in data["models"].values():
        for f in m.get("families") or []:
            seen.setdefault(f.lower(), f)
    for p in data["shared"].values():
        for f in p.get("families") or []:
            seen.setdefault(f.lower(), f)
    return [seen[k] for k in sorted(seen)]


def family_warnings(families, exclude_model=None, exclude_preset=None):
    """Flag families nothing else in the library uses — which is what a typo looks like.

    Creating a new family is legitimate, so this never blocks a save; it reports, and suggests the
    closest existing names. The reason it's worth the trouble: `flow-1204` instead of `flow-1024`
    raises no error anywhere — the shared preset simply never appears in any picker.

    The model / preset being saved is excluded from "known", so re-saving a typo doesn't make it
    look established just because it's already in the store.
    """
    wanted = _norm_list(families) or []
    if not wanted:
        return []
    data = _load()
    known = set()
    for mid, m in data["models"].items():
        if mid == exclude_model:
            continue
        known.update(f.lower() for f in m.get("families") or [])
    for name, p in data["shared"].items():
        if name == exclude_preset:
            continue
        known.update(f.lower() for f in p.get("families") or [])
    out = []
    for f in wanted:
        if f.lower() in known:
            continue
        msg = f"family '{f}' is new — nothing else in the library uses it"
        close = difflib.get_close_matches(f.lower(), sorted(known), n=2, cutoff=0.7)
        if close:
            msg += f" (did you mean {' or '.join(repr(c) for c in close)}?)"
        out.append(msg)
    return out


def presets_for(model_id, data=None):
    """``{name: preset}`` a model may use — its own, plus shared presets whose ``families``
    intersect the model's. Own presets win a name clash (the specific beats the generic)."""
    data = data or _load()
    model = data["models"].get(model_id)
    if not model:
        return {}
    fams = {f.lower() for f in model.get("families") or []}
    out = {}
    for name, p in data["shared"].items():
        if fams & {f.lower() for f in p.get("families") or []}:
            out[name] = p
    out.update(model.get("presets") or {})
    return out


def preset_names(model_id):
    return [NONE] + sorted(presets_for(model_id).keys())


def get_preset(model_id, name):
    if not name or name == NONE:
        return None
    return presets_for(model_id).get(name)


def default_preset_name(model_id):
    """The preset flagged as this model's default, if any — what the picker should land on."""
    for name, p in sorted(presets_for(model_id).items()):
        if p.get("default"):
            return name
    return None


def full_data():
    """Everything the frontend needs to drive the two dropdowns without a round-trip per pick.

    Recipes are summarised rather than sent whole: the picker only needs to show what a bundle is
    made of, and a bundle can be a dozen nodes.
    """
    data = _load()
    models = {}
    for mid, m in data["models"].items():
        recipe = m.get("recipe") or {}
        graph = recipe.get("nodes") or {}
        models[mid] = {
            "families": m.get("families") or [],
            "tags": m.get("tags") or [],
            "notes": m.get("notes") or "",
            "slots": sorted((recipe.get("outputs") or {}).keys()),
            "node_count": len(graph),
            "classes": sorted({n.get("class_type") for n in graph.values()
                               if isinstance(n, dict) and n.get("class_type")}),
            "presets": {name: {"tags": p.get("tags") or [], "score": p.get("score"),
                               "seconds": p.get("seconds"), "default": p.get("default"),
                               "stages": len(p.get("stages") or []), "shared": False,
                               # `overrides` is the COUNT (for the badge); `override_map` is the
                               # real thing, which the overrides editor needs to tick its rows.
                               "overrides": len(p.get("overrides") or {}),
                               "override_map": p.get("overrides") or {},
                               "width": p.get("width"), "height": p.get("height")}
                        for name, p in (m.get("presets") or {}).items()},
        }
        for name, p in presets_for(mid, data).items():
            if name not in models[mid]["presets"]:
                models[mid]["presets"][name] = {
                    "tags": p.get("tags") or [], "score": p.get("score"),
                    "seconds": p.get("seconds"), "default": p.get("default"),
                    "stages": len(p.get("stages") or []), "shared": True,
                    "overrides": len(p.get("overrides") or {}),
                    "override_map": p.get("overrides") or {},
                    "width": p.get("width"), "height": p.get("height")}
    return {
        "none": NONE,
        "all_families": ALL_FAMILIES,
        "order": [NONE] + sorted(data["models"].keys()),
        "families": all_families(),
        "models": models,
        "shared": {name: {"families": p.get("families") or [], "tags": p.get("tags") or [],
                          "stages": len(p.get("stages") or []), "score": p.get("score"),
                          "seconds": p.get("seconds"), "default": p.get("default"),
                          "overrides": len(p.get("overrides") or {}),
                          "override_map": p.get("overrides") or {},
                          "width": p.get("width"), "height": p.get("height")}
                   for name, p in data["shared"].items()},
    }


# ------------------------------------------------------------------------------------ writing
def upsert_model(model_id, recipe=None, families=None, tags=None, notes=None):
    """Add or update a model. Everything except ``model_id`` is optional and ``None`` means "keep
    what's there" — so re-capturing a bundle can't wipe the presets, families or notes that were
    added around it later."""
    model_id = (model_id or "").strip()
    if not model_id or model_id == NONE:
        raise ValueError("model id is required")
    with _LOCK:
        data = _load()
        m = data["models"].get(model_id) or _norm_model({})
        if recipe is not None:
            if not isinstance(recipe, dict) or not (recipe.get("nodes") or {}):
                raise ValueError("recipe must have nodes")
            m["recipe"] = {"nodes": recipe.get("nodes") or {},
                           "outputs": recipe.get("outputs") or {}}
        if notes is not None:
            m["notes"] = str(notes)
        for key, val in (("families", families), ("tags", tags)):
            nv = _norm_list(val)
            if nv is not None:
                m[key] = nv
        data["models"][model_id] = m
        _write(data)
    return full_data()


def delete_model(model_id):
    with _LOCK:
        data = _load()
        data["models"].pop((model_id or "").strip(), None)
        _write(data)
    return full_data()


def rename_model(old, new):
    """Rename in place, keeping the bundle and every preset — the alternative is an orphaned
    library, which is how preset collections die."""
    old, new = (old or "").strip(), (new or "").strip()
    if not old or not new:
        raise ValueError("both names are required")
    with _LOCK:
        data = _load()
        if old not in data["models"]:
            raise ValueError(f"no model '{old}'")
        if new in data["models"] and new != old:
            raise ValueError(f"'{new}' already exists")
        data["models"][new] = data["models"].pop(old)
        _write(data)
    return full_data()


def upsert_preset(model_id, name, preset, shared=False, delete=False):
    """Save (or delete) a preset. ``shared=True`` puts it in the family-wide pool instead of under
    one model. Setting ``default`` clears the flag on that model's other presets, since "the
    default" only means anything if there's one of it."""
    name = (name or "").strip()
    if not name or name == NONE:
        raise ValueError("preset name is required")
    with _LOCK:
        data = _load()
        if shared:
            bucket = data["shared"]
        else:
            model_id = (model_id or "").strip()
            if model_id not in data["models"]:
                raise ValueError(f"no model '{model_id}' — register it with Model Capture first")
            bucket = data["models"][model_id]["presets"]
        if delete:
            bucket.pop(name, None)
        else:
            np = _norm_preset(preset)
            if np is None:
                raise ValueError("preset must be an object")
            if not np["stages"]:
                raise ValueError("preset has no sampler stages")
            bucket[name] = np
            if np.get("default"):
                for other, p in bucket.items():
                    if other != name:
                        p["default"] = False
        _write(data)
    return full_data()


def retag_preset(model_id, name, tags=None, families=None, notes=None, set_default=None,
                 shared=False):
    """Edit a preset's metadata without touching its stages (the Manage dialog's job)."""
    name = (name or "").strip()
    with _LOCK:
        data = _load()
        bucket = data["shared"] if shared else (data["models"].get(model_id) or {}).get("presets")
        if not bucket or name not in bucket:
            raise ValueError(f"no preset '{name}'")
        p = bucket[name]
        for key, val in (("tags", tags), ("families", families)):
            nv = _norm_list(val)
            if nv is not None:
                p[key] = nv
        if notes is not None:
            p["notes"] = str(notes)
        if set_default is not None:
            p["default"] = bool(set_default)
            if p["default"]:
                for other, q in bucket.items():
                    if other != name:
                        q["default"] = False
        _write(data)
    return full_data()


def editable_fields(model_id):
    """The recipe's literal inputs, per node — what the Library dialog can offer to edit.

    Linked inputs are deliberately excluded: a link is the assembly's *wiring*, captured from the
    graph, not a setting. Everything else (a filename, a shift, a cfg, a dtype) is fair game, so a
    bundle can be retuned — or pointed at a different weights file — without re-capturing it.
    """
    model = get_model(model_id)
    if not model:
        return []
    from . import replay
    graph = (model.get("recipe") or {}).get("nodes") or {}
    out = []
    for nid in sorted(graph, key=lambda x: (len(x), x)):
        node = graph[nid]
        fields = {k: v for k, v in (node.get("inputs") or {}).items() if not replay.is_link(v)}
        linked = sorted(k for k, v in (node.get("inputs") or {}).items() if replay.is_link(v))
        out.append({"id": nid, "class_type": node.get("class_type"),
                    "values": fields, "linked": linked})
    return out


def _coerce(class_type, field, value):
    """Nudge a value to the type the node declares, so a form post can't put "3.0" where a float
    belongs. Unknown types and unknown fields pass through untouched."""
    try:
        from . import replay
        cls = replay._mappings().get(class_type)
        if cls is None:
            return value
        spec, _ = replay.input_spec(cls)
        declared = (spec.get(field) or (None, None))[0]
    except Exception:
        return value
    try:
        if declared == "INT":
            return int(float(value))
        if declared == "FLOAT":
            return float(value)
        if declared == "BOOLEAN":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
    except (TypeError, ValueError):
        return value
    return value


def update_recipe(model_id, values):
    """Write literal widget values into a model's bundle. ``values`` is ``{"<node_id>.<input>": v}``.

    Node ids are left alone rather than re-canonicalised: ids never enter the replay cache key
    (that's a content hash), and keeping them stable means the editor's fields don't shuffle
    underneath the user between saves. Returns ``(data, applied, rejected)`` so the dialog can say
    what landed instead of pretending everything did.
    """
    model_id = (model_id or "").strip()
    applied, rejected = [], []
    with _LOCK:
        data = _load()
        model = data["models"].get(model_id)
        if not model:
            raise ValueError(f"no model '{model_id}'")
        from . import replay
        graph = (model.get("recipe") or {}).get("nodes") or {}
        for key, value in (values or {}).items():
            nid, _, field = str(key).rpartition(".")
            node = graph.get(nid)
            if not isinstance(node, dict):
                rejected.append(f"{key}: no node {nid} in this bundle")
                continue
            inputs = node.get("inputs") or {}
            if field not in inputs:
                rejected.append(f"{key}: node {nid} has no input '{field}'")
                continue
            if replay.is_link(inputs[field]):
                rejected.append(f"{key}: '{field}' is wiring, not a setting")
                continue
            inputs[field] = _coerce(node.get("class_type"), field, value)
            applied.append(key)
        _write(data)
    return full_data(), applied, rejected


def set_overrides(model_id, name, overrides, shared=False):
    """Replace a preset's ``overrides`` map. Keys are ``"<class_type>.<input>"``; an empty map
    clears them. Keyed by class rather than node id on purpose — re-capturing a bundle renumbers
    its nodes, and an override that survives that is worth more than one that's precise."""
    name = (name or "").strip()
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be an object")
    clean = {}
    for key, value in overrides.items():
        cls_name, _, field = str(key).rpartition(".")
        if not cls_name or not field:
            raise ValueError(f"'{key}' is not a '<class_type>.<input>' key")
        clean[f"{cls_name}.{field}"] = _coerce(cls_name, field, value)
    with _LOCK:
        data = _load()
        bucket = data["shared"] if shared else (data["models"].get(model_id) or {}).get("presets")
        if not bucket or name not in bucket:
            raise ValueError(f"no preset '{name}'")
        bucket[name]["overrides"] = clean
        _write(data)
    return full_data()


def resolve(model_id, preset_name):
    """What Model Select needs in one call: ``(model, preset_or_None, recipe_with_overrides)``.

    The recipe returned is a *copy* with the preset's ``overrides`` already folded into the matching
    nodes' inputs, so replay stays oblivious to presets and the merkle cache sees the override as
    what it is — a changed patch input, not a changed model.
    """
    data = _load()
    model = data["models"].get(model_id)
    if not model:
        return None, None, None
    preset = presets_for(model_id, data).get(preset_name) if preset_name and preset_name != NONE \
        else None
    recipe = copy.deepcopy(model.get("recipe") or {})
    if preset:
        apply_overrides(recipe, preset.get("overrides") or {})
    return model, preset, recipe


def apply_overrides(recipe, overrides):
    """Fold ``{"<class_type>.<input>": value}`` into a recipe in place. Returns what it applied and
    what it couldn't, so ``info`` can say so instead of silently ignoring a stale override."""
    applied, unknown = [], []
    graph = (recipe or {}).get("nodes") or {}
    for key, value in (overrides or {}).items():
        if "." not in str(key):
            unknown.append(str(key))
            continue
        class_type, _, field = str(key).rpartition(".")
        hit = False
        for node in graph.values():
            if isinstance(node, dict) and node.get("class_type") == class_type \
                    and field in (node.get("inputs") or {}):
                node["inputs"][field] = value
                hit = True
        (applied if hit else unknown).append(f"{class_type}.{field}")
    return applied, unknown
