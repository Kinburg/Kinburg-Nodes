"""JSON Extract — pull fields out of a JSON string by path into separate STRING outputs.

Pairs with the structured-output LLM nodes (e.g. output_format = ideogram4_json): the model
returns one JSON blob, and this node routes its sub-fields into distinct prompt inputs.

Authoring is one path per line in the `paths` field — no fixed slot count. Each line is either
a bare path or `path -> alias`; blank lines and `#` comments are ignored. The frontend grows one
`value_*` output per line and labels it by the alias (or the path's last key), so the graph is
self-documenting; the 🔍 Explore JSON button lets you click a field out of the last run / a pasted
sample instead of typing the path. Up to `_MAX_SLOTS` paths map to outputs (extras are reported).

Path syntax (a practical subset):
- dot keys:            `style_description.lighting`
- array index:         `elements[0]` or `elements.0`
- nested:              `compositional_deconstruction.elements[0].desc`
- negative index:      `elements[-1]`
- wildcard (join):     `elements[*].desc` or `elements.*.desc`  → every match joined by `array_join`
- whole document:      `$` or empty
A value that is itself an object/array comes out as compact JSON (so it can feed another JSON
Extract or a text node); strings pass through unchanged, numbers/bools become text. A `[*]`/`*`
segment expands to every element of a list (or every value of an object) and joins the results.

Outputs are `found` (True when every non-empty path resolved and the JSON parsed) and `report`
(a per-path hit/miss listing) FIRST, then `value_1..value_N`. Status is placed first on purpose so
the frontend can prune trailing unused value slots without disturbing the return-tuple mapping.

Non-JSON wrapping text is tolerated — if `json.loads` fails, the first {...} or [...] block is
parsed instead (LLMs sometimes add prose around the JSON).
"""
import re
import json
from ..categories import CAT_UTIL

_MAX_SLOTS = 12
_BRACKET_RE = re.compile(r"\[(-?\d+|\*)\]")
_PREVIEW_CAP = 240          # per-value chars shown in the in-node preview
_JSON_UI_CAP = 200_000      # don't ship a giant document to the browser for the Explore tree

_DEFAULT_PATHS = "# one path per line — `path` or `path -> alias`\n# e.g. style_description.lighting -> lighting\n"


def _parse_json(text):
    """Parse `text` as JSON; on failure try the first {...} / [...] block. Returns (obj, error)."""
    s = (text or "").strip()
    if not s:
        return None, "empty input"
    try:
        return json.loads(s), ""
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)), ""
        except Exception as e:
            return None, f"invalid JSON: {e}"
    return None, "no JSON found"


def _tokens(path):
    """Split a path like `a.b[0].c` into ['a','b','0','c']. A leading `$`/`$.` is stripped.

    `[n]`/`[-n]`/`[*]` are normalised to dot segments, so `*` survives as its own token.
    """
    p = path.strip()
    if p.startswith("$"):
        p = p[1:]
    p = _BRACKET_RE.sub(r".\1", p)          # a[0] -> a.0 ; a[*] -> a.*
    return [t for t in (seg.strip() for seg in p.split(".")) if t != ""]


def _resolve(obj, toks):
    """Walk `obj` by `toks`. Returns a list of matched values, or None if the path misses.

    A non-wildcard path yields a 1-element list. A `*` token expands to every element of a list
    (or every value of a dict); branches that miss the remaining path are skipped, so a wildcard
    over a valid container never returns None (it returns [] at worst).
    """
    if not toks:
        return [obj]
    t, rest = toks[0], toks[1:]

    if t == "*":
        if isinstance(obj, dict):
            items = obj.values()
        elif isinstance(obj, list):
            items = obj
        else:
            return None                     # can't wildcard a scalar
        out = []
        for it in items:
            r = _resolve(it, rest)
            if r is not None:
                out.extend(r)
        return out

    if isinstance(obj, dict):
        if t in obj:
            return _resolve(obj[t], rest)
        return None
    if isinstance(obj, list):
        try:
            idx = int(t)
        except ValueError:
            return None
        if -len(obj) <= idx < len(obj):
            return _resolve(obj[idx], rest)
        return None
    return None                             # can't descend into a scalar


def _to_str(v):
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(v, ensure_ascii=False)   # object / array -> compact JSON


def _label_from_path(path):
    """A short output label for a path: the last non-`*` token, or `{prev}_{index}` for a bare
    index, or 'root' for `$`/empty."""
    toks = [t for t in _tokens(path)]
    if not toks:
        return "root"
    real = [t for t in toks if t != "*"]
    if not real:
        return "root"
    last = real[-1]
    if re.fullmatch(r"-?\d+", last):        # bare array index — pair it with the container key
        prev = real[-2] if len(real) >= 2 else "item"
        return f"{prev}_{last.lstrip('-')}"
    return last


def _split_alias(line):
    """`path -> alias` / `path => alias` → (path, alias). No arrow → (line, '')."""
    for sep in ("->", "=>"):
        if sep in line:
            left, right = line.split(sep, 1)
            return left.strip(), right.strip()
    return line.strip(), ""


def parse_paths(text):
    """Parse the multiline `paths` field into a list of {path, alias, label}. Skips blank lines
    and `#` comments. `label` is the alias when given, else derived from the path."""
    entries = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        path, alias = _split_alias(s)
        if path == "":
            continue
        entries.append({"path": path, "alias": alias, "label": alias or _label_from_path(path)})
    return entries


def _trunc(s):
    s = s.replace("\n", "\\n")
    return s if len(s) <= _PREVIEW_CAP else s[:_PREVIEW_CAP] + "…"


class JSONExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "", "tooltip": "The JSON to read — wire an LLM JSON output here, or type/paste. Prose around the JSON is tolerated (first {…}/[…] is parsed)."}),
            },
            "optional": {
                "paths": ("STRING", {"multiline": True, "default": _DEFAULT_PATHS, "tooltip": "One path per line — `path` or `path -> alias`. Blank lines and `# comments` are ignored. Each line makes a labelled value_* output; the outputs are rebuilt when you click away from this field. Use 🔍 Explore JSON to click fields out of the last run."}),
                "default": ("STRING", {"default": "", "tooltip": "Value returned for a path that isn't found (and for every output if the JSON can't be parsed)."}),
                "array_join": ("STRING", {"default": ", ", "tooltip": "Separator used to join matches when a path uses a `[*]`/`*` wildcard (e.g. elements[*].desc)."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING") + ("STRING",) * _MAX_SLOTS
    RETURN_NAMES = ("found", "report") + tuple(f"value_{i}" for i in range(1, _MAX_SLOTS + 1))
    FUNCTION = "run"
    CATEGORY = CAT_UTIL

    def run(self, json_string="", paths="", default="", array_join=", ", **_):
        entries = parse_paths(paths)
        over = entries[_MAX_SLOTS:]
        entries = entries[:_MAX_SLOTS]

        values = [""] * _MAX_SLOTS
        obj, err = _parse_json(json_string)

        if err:
            for i in range(len(entries)):
                values[i] = default
            report = f"⚠ parse error: {err}"
            ui = {"kb_text": [report], "kb_json": [""]}
            return {"ui": ui, "result": (False, report) + tuple(values)}

        report_lines, preview_lines, all_found = [], [], True
        for i, e in enumerate(entries):
            toks = _tokens(e["path"])
            res = _resolve(obj, toks)
            if res is None:
                values[i] = default
                report_lines.append(f"{e['label']} ← '{e['path']}': NOT FOUND")
                preview_lines.append(f"✗ {e['label']}: (not found)")
                all_found = False
            else:
                if any(t == "*" for t in toks):
                    val = array_join.join(_to_str(v) for v in res)
                else:
                    val = _to_str(res[0])
                values[i] = val
                report_lines.append(f"{e['label']} ← '{e['path']}': ok")
                preview_lines.append(f"• {e['label']}: {_trunc(val)}")

        if over:
            msg = f"⚠ {len(over)} path(s) beyond the {_MAX_SLOTS}-slot limit ignored: " + ", ".join(o["path"] for o in over)
            report_lines.append(msg)
            preview_lines.append(msg)

        if report_lines:
            head = "✔ all paths found" if all_found else "✗ some paths missing"
            report = head + "\n" + "\n".join(report_lines)
        else:
            report = "(no paths set — JSON parsed OK)"
            preview_lines.append(report)

        compact = json.dumps(obj, ensure_ascii=False)
        ui = {
            "kb_text": ["\n".join(preview_lines)],
            "kb_json": [compact if len(compact) <= _JSON_UI_CAP else ""],
        }
        return {"ui": ui, "result": (all_found, report) + tuple(values)}


NODE_CLASS_MAPPINGS = {"KinburgJSONExtract": JSONExtract}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgJSONExtract": "JSON Extract"}
