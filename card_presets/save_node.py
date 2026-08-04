"""Card Save — turn an LLM's JSON card into a saved preset (and a ready Markdown block).

Closes the loop that Grammar Presets opens: wire **Grammar Presets → Local LLM (GGUF)
(grammar_override) + photo → Card Save**, and the character / entity the model read off the
picture lands straight in the **Card Presets** library — no JSON Extract → 12-wire dance into a
Character Card. The grammar's JSON keys already mirror the card fields 1:1, so this node just
parses the JSON and stores it.

Inputs: ``json_string`` (the LLM output — prose around the JSON is tolerated), ``card_type``
(``auto`` detects character vs entity from the keys) and, optional, ``save_as`` (preset name —
empty falls back to the JSON's own ``name``) and ``tags`` (comma-separated labels for filtering
the library). Outputs the rendered ``card`` block (feed Context Collector in the same run),
``saved_as`` (the name used, "" if nothing was saved) and a ``report`` line. Never raises — a
bad / empty JSON yields an empty card and an explanatory report. Category ``Kinburg-Nodes/LLM``.
"""
from . import store

# Character grammar has these attribute keys; entity has just name + description. Used to
# auto-detect the card type when card_type == "auto".
_CHAR_KEYS = {"gender", "age", "ethnicity", "eye_color", "hair_color",
              "hair_style", "build", "height", "outfit", "features", "notes"}

_AUTO = "auto"


def _parse(text):
    """Parse `text` as JSON, tolerating prose around it. Returns (dict_or_None, error)."""
    try:
        from ..util.json_extract import _parse_json
        obj, err = _parse_json(text)
    except Exception as e:  # pragma: no cover - util import guard
        return None, f"parser unavailable: {e}"
    if err:
        return None, err
    if not isinstance(obj, dict):
        return None, "JSON is not an object (expected a card {…})"
    return obj, ""


def _detect_type(values):
    """Guess 'character' vs 'entity' from the JSON keys (entity = free-form name+description)."""
    keys = set(values)
    if "description" in keys and not (keys & _CHAR_KEYS):
        return "entity"
    return "character"


class CardSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "", "tooltip": "The card JSON — wire a Local LLM (GGUF) output constrained by a Grammar Presets grammar, or paste. Prose around the JSON is tolerated (first {…} is parsed)."}),
                "card_type": ([_AUTO, "character", "entity"], {"default": _AUTO, "tooltip": "auto = detect from the JSON keys (a name+description with no character attributes → entity, else character). Force it if auto guesses wrong."}),
            },
            "optional": {
                "save_as": ("STRING", {"default": "", "tooltip": "Preset name for the library. Empty = use the JSON's own 'name' field. Empty here AND in the JSON = render only, don't save. Re-saving the same name overwrites it."}),
                "tags": ("STRING", {"default": "", "tooltip": "Comma-separated tags to filter the library by in Card Presets (e.g. 'heroes, medieval'). Empty = leave existing tags untouched (edit/clear them in Card Presets → Manage)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("card", "saved_as", "report")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, json_string="", card_type=_AUTO, save_as="", tags=""):
        values, err = _parse(json_string)
        if err:
            report = f"parse error: {err}"
            print(f"[CardSave] {report}")
            return ("", "", report)

        ctype = _detect_type(values) if card_type == _AUTO else card_type

        name = (save_as or "").strip() or (str(values.get("name") or "")).strip()
        if not name:
            # Nothing to file it under (a photo often yields an empty name) — render as-is, don't save.
            card = store.render_values(ctype, values)
            report = f"parsed {ctype} card, not saved (no name in JSON and save_as empty)"
            print(f"[CardSave] {report}")
            return (card, "", report)

        # The resolved name is BOTH the library key and the card heading, so an explicit save_as
        # (the real name the model couldn't read off the photo) drives the block too.
        values = {**values, "name": name}
        card = store.render_values(ctype, values)

        # Empty tags string -> None (leave existing tags alone on overwrite).
        tags_arg = tags if (tags or "").strip() else None
        try:
            store.upsert(name, ctype, values, tags=tags_arg)
            saved = store.get(name) or {}
            tag_note = f", tags: {', '.join(saved.get('tags') or [])}" if saved.get("tags") else ""
            report = f"saved '{name}' ({ctype}{tag_note})"
            print(f"[CardSave] {report}")
            return (card, name, report)
        except Exception as e:
            report = f"save '{name}' failed: {e}"
            print(f"[CardSave] {report}")
            return (card, "", report)


NODE_CLASS_MAPPINGS = {"CardSave": CardSave}
NODE_DISPLAY_NAME_MAPPINGS = {"CardSave": "Card Save"}
