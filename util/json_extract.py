"""JSON Extract — pull fields out of a JSON string by path into separate STRING outputs.

Pairs with the structured-output LLM nodes (e.g. output_format = ideogram4_json): the model
returns one JSON blob, and this node routes its sub-fields into distinct prompt inputs. Up to
six independent paths → six `value_*` outputs, plus a `found` flag (True when every non-empty
path resolved and the JSON parsed) and a `report` listing each path's hit/miss for debugging.

Path syntax (a practical subset):
- dot keys:            `style_description.lighting`
- array index:         `elements[0]` or `elements.0`
- nested:              `compositional_deconstruction.elements[0].desc`
- negative index:      `elements[-1]`
- whole document:      `$` or empty
A value that is itself an object/array comes out as compact JSON (so it can feed another JSON
Extract or a text node); strings pass through unchanged, numbers/bools become text.

Non-JSON wrapping text is tolerated — if `json.loads` fails, the first {...} or [...] block is
parsed instead (LLMs sometimes add prose around the JSON).
"""
import re
import json

_N_PATHS = 6
_BRACKET_RE = re.compile(r"\[(-?\d+)\]")


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
    """Split a path like `a.b[0].c` into ['a','b','0','c']. A leading `$`/`$.` is stripped."""
    p = path.strip()
    if p.startswith("$"):
        p = p[1:]
    p = _BRACKET_RE.sub(r".\1", p)          # a[0] -> a.0
    return [t for t in (seg.strip() for seg in p.split(".")) if t != ""]


def _get(obj, toks):
    """Walk `obj` by tokens; return (value, found)."""
    cur = obj
    for t in toks:
        if isinstance(cur, dict):
            if t in cur:
                cur = cur[t]
                continue
            return None, False
        if isinstance(cur, list):
            try:
                idx = int(t)
            except ValueError:
                return None, False
            if -len(cur) <= idx < len(cur):
                cur = cur[idx]
                continue
            return None, False
        return None, False       # can't descend into a scalar
    return cur, True


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


class JSONExtract:
    @classmethod
    def INPUT_TYPES(cls):
        paths = {
            "path_1": ("STRING", {"default": "", "tooltip": "Path into the JSON, e.g. style_description.lighting or elements[0].desc. Leave empty to skip this slot. '$' or empty = the whole document."}),
        }
        for i in range(2, _N_PATHS + 1):
            paths[f"path_{i}"] = ("STRING", {"default": "", "tooltip": f"Path #{i} (same syntax as path_1). Empty = skip."})
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "", "tooltip": "The JSON to read — wire an LLM JSON output here, or type/paste. Prose around the JSON is tolerated (first {…}/[…] is parsed)."}),
            },
            "optional": {
                **paths,
                "default": ("STRING", {"default": "", "tooltip": "Value returned for a path that isn't found (and for every output if the JSON can't be parsed)."}),
            },
        }

    RETURN_TYPES = ("STRING",) * _N_PATHS + ("BOOLEAN", "STRING")
    RETURN_NAMES = tuple(f"value_{i}" for i in range(1, _N_PATHS + 1)) + ("found", "report")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/util"

    def run(self, json_string="", default="", **kwargs):
        obj, err = _parse_json(json_string)
        paths = [(kwargs.get(f"path_{i}", "") or "").strip() for i in range(1, _N_PATHS + 1)]

        if err:
            report = f"parse error: {err}"
            return tuple([default] * _N_PATHS) + (False, report)

        values, report_lines, all_found = [], [], True
        for idx, path in enumerate(paths, 1):
            if not path:
                values.append("")           # unused slot — not part of `found`
                continue
            val, ok = _get(obj, _tokens(path))
            if ok:
                values.append(_to_str(val))
                report_lines.append(f"path_{idx} '{path}': ok")
            else:
                values.append(default)
                report_lines.append(f"path_{idx} '{path}': NOT FOUND")
                all_found = False

        report = "\n".join(report_lines) if report_lines else "(no paths set — JSON parsed OK)"
        return tuple(values) + (all_found, report)


NODE_CLASS_MAPPINGS = {"KinburgJSONExtract": JSONExtract}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgJSONExtract": "JSON Extract"}
