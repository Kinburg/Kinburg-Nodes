"""Criteria Builder — pick evaluation criteria with toggles and emit them as a `criteria` STRING.

Feeds the `criteria` field of **Critic Settings (GGUF)** (Ouroboros) and **Vision LLM Judge** —
both parse the same `name: description` per-line format. Instead of typing that by hand, tick the
criteria you want; the node assembles the string (each toggled criterion contributes a curated,
model-guiding description). Wire the output into either node's `criteria` field via right-click →
**Convert widget to input** on that field.

The catalog of criteria lives in `catalog.json` (shipped) and, if present, `catalog.user.json`
(your own additions — survives a git pull). Each entry becomes one BOOLEAN toggle. Category
`Kinburg-Nodes/LLM`.
"""
import json
import os
import re

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(NODE_DIR, "catalog.json")
USER_CATALOG = os.path.join(NODE_DIR, "catalog.user.json")
# Widget names reserved for the node's own inputs — a catalog key can't shadow these.
RESERVED = {"extra", "criteria_in"}

# Built-in fallback so the node still works if catalog.json is missing/corrupt (mirrors the five
# that were the old hand-typed default).
_FALLBACK = [
    {"key": "overall_quality", "default": True, "description": "overall image quality — matches the intended style, no artifacts, no excess noise, correct proportions, good color reproduction"},
    {"key": "prompt_compliance", "default": True, "description": "how accurately the image follows the generation prompt — subjects, attributes, counts and relations are present and correct"},
    {"key": "anatomy", "default": True, "description": "all required limbs present, no extra or missing limbs, correct placement, natural proportional body"},
    {"key": "camera", "default": True, "description": "camera angle and camera settings match the intent"},
    {"key": "text", "default": True, "description": "if the prompt requests text — present, character-accurate, correct color/font/size/placement; if the prompt has NO text, give the top score"},
]


def _sanitize_key(name):
    """Match Vision Judge's key derivation so dedup lines up with how `criteria` is later parsed."""
    return re.sub(r"[^a-z0-9_]+", "_", (name or "").strip().lower()).strip("_")


def _split_line(s):
    """Split a `criteria` line into (name, description) on the same separators the judge accepts."""
    for sep in (":", " — ", " - "):
        if sep in s:
            a, b = s.split(sep, 1)
            return a.strip(), b.strip()
    return s.strip(), ""


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[Kinburg CriteriaBuilder] failed to read {os.path.basename(path)}: {e}")
        return None


def _entries(obj):
    if isinstance(obj, dict):
        obj = obj.get("criteria", [])
    return obj if isinstance(obj, list) else []


def load_catalog():
    """Shipped catalog.json + optional catalog.user.json (user entries extend the list and override
    any matching key, keeping the original position). Ordered, deduped by sanitized key, reserved
    names dropped. Falls back to the built-in set when nothing loads."""
    base = _entries(_read_json(CATALOG))
    user = _entries(_read_json(USER_CATALOG))
    raw = (base if base else list(_FALLBACK)) + user
    out, pos = [], {}
    for e in raw:
        if not isinstance(e, dict):
            continue
        key = _sanitize_key(e.get("key", ""))
        if not key or key in RESERVED:
            continue
        entry = {"key": key,
                 "description": str(e.get("description", "")).strip(),
                 "default": bool(e.get("default", False)),
                 "group": str(e.get("group", "") or "")}
        if key in pos:            # a later (user) entry overrides an earlier one in place
            out[pos[key]] = entry
        else:
            pos[key] = len(out)
            out.append(entry)
    return out


class CriteriaBuilder:
    """Assemble a `criteria` string from toggles + optional custom/chained lines."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {}
        for e in load_catalog():
            req[e["key"]] = ("BOOLEAN", {"default": e["default"],
                                         "tooltip": e["description"] or e["key"]})
        return {
            "required": req,
            "optional": {
                "extra": ("STRING", {"multiline": True, "default": "", "tooltip": "Extra criteria, one per line ('name: description'), appended after the toggled ones. For anything not in the catalog."}),
                "criteria_in": ("STRING", {"forceInput": True, "tooltip": "Optional upstream criteria string merged FIRST — chain builders, or start from an existing set. Duplicates (by name) are removed."}),
            },
        }

    # Toggles are booleans and the catalog can grow, so no combo-membership check is needed.
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("criteria",)
    FUNCTION = "build"
    CATEGORY = "Kinburg-Nodes/LLM"
    DESCRIPTION = ("Build the Critic / Vision Judge 'criteria' string by ticking criteria instead of "
                   "typing them. Convert the target node's 'criteria' widget to an input and wire this "
                   "output in. Empty output = single overall score. Catalog: catalog.json (+ catalog.user.json).")

    def build(self, extra="", criteria_in="", **toggles):
        lines, seen = [], set()

        def _add(name, desc):
            key = _sanitize_key(name)
            if not key or key in seen:
                return
            seen.add(key)
            lines.append(f"{name.strip()}: {desc.strip()}" if desc.strip() else name.strip())

        # 1) upstream chain first (so an upstream builder's choices take precedence on a clash).
        for raw in (criteria_in or "").split("\n"):
            if raw.strip():
                _add(*_split_line(raw))
        # 2) toggled catalog entries, in catalog order.
        for e in load_catalog():
            if bool(toggles.get(e["key"], False)):
                _add(e["key"], e["description"])
        # 3) free-text extras last.
        for raw in (extra or "").split("\n"):
            if raw.strip():
                _add(*_split_line(raw))

        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {"CriteriaBuilder": CriteriaBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"CriteriaBuilder": "Criteria Builder 📋"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
