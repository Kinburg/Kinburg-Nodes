"""The parameter-dialect gap between SillyTavern and llama.cpp's server.

SillyTavern assembles one request body for a whole family of backends, and for a **Custom
(OpenAI-compatible)** endpoint it merges whatever sits in *Additional Parameters* on top of it as
YAML. llama.cpp's server validates every field it recognises and answers the entire request with a
400 when one has the wrong shape, so a single line in that box is enough to make every message
fail. The one that actually bites is `dry_sequence_breakers`: SillyTavern stores it as a JSON
**string** (`'["\n", ":", "\\"", "*"]'`) and only converts it to an array on the code paths for
its llama.cpp / KoboldCpp / Mancer backends — a Custom endpoint sends the string through as-is, and
llama.cpp answers

    Field 'dry_sequence_breakers': Error: dry_sequence_breakers must be a non-empty array of strings

(`tools/server/server-schema.cpp`, whose own comment reads "Currently, this is not compatible with
TextGen WebUI, Koboldcpp and SillyTavern format").

Two mechanisms live here, and the gateway uses both:

* :func:`normalize` fixes the shapes we already know about, before the request leaves;
* :func:`field_from_error` reads the offending field's name back out of a 400 so the gateway can
  drop that one field and retry once — which covers every field nobody has hit yet, including the
  ones a future llama.cpp will start validating.

Pure functions only: no HTTP, no ComfyUI, no llama.cpp.
"""
import json
import re

#: llama-server prefixes a rejected field's error with its name (`handle_with_catch` in
#: server-schema.cpp). That is all we need to name the offender and retry without it.
FIELD_ERROR_RE = re.compile(r"Field '([A-Za-z0-9_.\-]+)'")

#: Fields llama.cpp wants as an array of strings, and what to do with a bare string that is not
#: JSON: split it on commas (SillyTavern's own fallback for sequence breakers) or keep it whole.
STRING_LIST_FIELDS = {
    "dry_sequence_breakers": True,
    "stop": False,
    "banned_strings": False,
}


def field_from_error(text):
    """The field name out of a llama-server 400 body, or "" when it does not name one."""
    m = FIELD_ERROR_RE.search(text or "")
    return m.group(1) if m else ""


def as_string_list(value, split_commas=False):
    """A list of non-empty strings out of whatever arrived, or None if there is nothing usable.

    Handles the three shapes that turn up in practice: a real array, a JSON-serialised array (what
    SillyTavern stores), and a bare string — which is one entry unless the field is one where a
    comma-separated line is the conventional shorthand.
    """
    if isinstance(value, str):
        try:
            # strict=False so a breaker that arrived as a real newline rather than an escaped one
            # still parses instead of falling through to the comma split.
            value = json.loads(value, strict=False)
        except Exception:
            # No trimming: SillyTavern's own fallback is a bare `.split(',')`, and for sequence
            # breakers the whitespace IS the entry.
            value = value.split(",") if split_commas else [value]
    if isinstance(value, str):          # a JSON string that decoded to another string
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    out = [v for v in value if isinstance(v, str) and v != ""]
    return out or None


def normalize(body, drop=()):
    """Return ``(body, notes)`` — the request body llama.cpp will accept, and what was changed.

    `drop` is the user's own escape hatch: field names to remove unconditionally. Notes are short
    lines meant for the gateway log, so the user can see that a request was edited on its way
    through rather than wondering why a sampler stopped having an effect.
    """
    notes = []
    if not isinstance(body, dict):
        return body, notes
    out = dict(body)

    for key in drop:
        key = (key or "").strip()
        if key and key in out:
            out.pop(key)
            notes.append(f"dropped '{key}' (drop_fields)")

    for key, split_commas in STRING_LIST_FIELDS.items():
        if key not in out:
            continue
        before = out[key]
        fixed = as_string_list(before, split_commas)
        if fixed is None:
            out.pop(key)
            notes.append(f"dropped '{key}' — no usable strings in {before!r}")
        elif fixed != before:
            out[key] = fixed
            notes.append(f"'{key}': {type(before).__name__} -> array of {len(fixed)}")

    return out, notes
