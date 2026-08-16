"""Text Transform — string find/replace, regex, trim and case operations in one node.

Removes the need for third-party text nodes. The differentiator is safety: a regex is
compiled up front and any error (bad pattern, bad replacement backref) is reported on the
`error` output while the original text passes through unchanged — a typo can never crash
the whole run.

Operations:
- **replace** — literal substring replace (`pattern` -> `replacement`).
- **regex_replace** — `re.sub(pattern, replacement, text)`; `replacement` may use `\\1` groups.
- **regex_extract** — the FIRST match: group 1 if the pattern has groups, else the whole match.
- **regex_findall** — every match, joined by `join_with` (group 1 if present, else the match).
- **strip** — trim leading/trailing whitespace (or the exact chars in `pattern`, if given).
- **collapse_whitespace** — collapse all runs of whitespace to single spaces, trimmed.
- **lower / upper / title** — case conversion.

The `ignorecase` / `multiline` / `dotall` toggles apply to the regex_* operations.
"""
import re
from ..categories import CAT_UTIL

_OPERATIONS = [
    "replace", "regex_replace", "regex_extract", "regex_findall",
    "strip", "collapse_whitespace", "lower", "upper", "title",
]


def _flags(ignorecase, multiline, dotall):
    f = 0
    if ignorecase:
        f |= re.IGNORECASE
    if multiline:
        f |= re.MULTILINE
    if dotall:
        f |= re.DOTALL
    return f


class TextTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "The input text (type here or wire a STRING in)."}),
                "operation": (_OPERATIONS, {"default": "replace", "tooltip": "What to do. replace = literal; regex_* = pattern is a regular expression; strip/collapse/case need no pattern."}),
                "pattern": ("STRING", {"multiline": True, "default": "", "tooltip": "Search text (literal for 'replace'; a regex for the regex_* ops). For 'strip', the exact characters to trim (empty = whitespace)."}),
                "replacement": ("STRING", {"multiline": True, "default": "", "tooltip": "Replacement text for replace / regex_replace. In regex_replace you may reference groups as \\1, \\2, …"}),
            },
            "optional": {
                "ignorecase": ("BOOLEAN", {"default": False, "tooltip": "Regex ops: case-insensitive matching (re.IGNORECASE)."}),
                "multiline": ("BOOLEAN", {"default": False, "tooltip": "Regex ops: ^ and $ match at line boundaries (re.MULTILINE)."}),
                "dotall": ("BOOLEAN", {"default": False, "tooltip": "Regex ops: '.' also matches newlines (re.DOTALL)."}),
                "join_with": ("STRING", {"default": "\n", "tooltip": "regex_findall: string inserted between matches. Default is a newline."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("text", "count", "error")
    FUNCTION = "run"
    CATEGORY = CAT_UTIL

    def run(self, text="", operation="replace", pattern="", replacement="",
            ignorecase=False, multiline=False, dotall=False, join_with="\n"):
        text = text if isinstance(text, str) else str(text)

        # Non-regex ops: no pattern compilation, so they can't error.
        if operation == "replace":
            if pattern == "":
                return (text, 0, "")
            return (text.replace(pattern, replacement), text.count(pattern), "")
        if operation == "strip":
            return (text.strip(pattern) if pattern else text.strip(), 0, "")
        if operation == "collapse_whitespace":
            return (" ".join(text.split()), 0, "")
        if operation == "lower":
            return (text.lower(), 0, "")
        if operation == "upper":
            return (text.upper(), 0, "")
        if operation == "title":
            return (text.title(), 0, "")

        # Regex ops: compile once; report failure instead of crashing the run.
        try:
            rx = re.compile(pattern, _flags(ignorecase, multiline, dotall))
        except re.error as e:
            print(f"[TextTransform] invalid regex {pattern!r}: {e}")
            return (text, 0, f"invalid regex: {e}")

        try:
            if operation == "regex_replace":
                new_text, n = rx.subn(replacement, text)
                return (new_text, n, "")
            if operation == "regex_extract":
                m = rx.search(text)
                if not m:
                    return ("", 0, "")
                return (m.group(1) if m.groups() else m.group(0), 1, "")
            if operation == "regex_findall":
                matches = list(rx.finditer(text))
                vals = [(m.group(1) if m.groups() else m.group(0)) for m in matches]
                return (join_with.join(vals), len(vals), "")
        except re.error as e:  # e.g. an invalid replacement backreference
            print(f"[TextTransform] regex operation failed: {e}")
            return (text, 0, f"regex error: {e}")

        return (text, 0, "")


NODE_CLASS_MAPPINGS = {"KinburgTextTransform": TextTransform}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgTextTransform": "Text Transform"}
