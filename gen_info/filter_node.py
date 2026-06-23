"""Generation Info Filter — turn several Generation Info dumps into per-image settings.

Feed it the `data` (GEN_INFO) output of one Generation Info node per branch/image. It lines
the dumps up field-by-field (a field = class_type + 1-based occurrence `ord` + param name)
and emits one block per input, separated by `separator`, ready for the compare node's
`settings` input.

Modes:
  * all                   — every field.
  * differences           — only fields whose value isn't identical across all inputs; each
                            block then shows ALL such fields (union), so the images line up.
  * custom                — only fields named in `custom_fields`.
  * differences + custom  — union of the two.

custom_fields, one selector per line:
  * ClassType            — all occurrences of that node, all its params
  * ClassType[n]         — the n-th occurrence (1-based), all its params
  * ClassType[n].param   — a single param of the n-th occurrence
  * ClassType.param      — that param on every occurrence

Alignment assumes the branches share structure (so the k-th occurrence of a class lines up
across inputs); a field missing from some inputs counts as a difference and shows as '—'.
"""
import json
import re
from collections import OrderedDict

_DATA_RE = re.compile(r"^data_(\d+)$")
_SELECTOR_RE = re.compile(r"^\s*([^\[\].]+?)\s*(?:\[\s*(\d+)\s*\])?\s*(?:\.\s*(.+?))?\s*$")
_ABSENT = "\x00absent"
_MODES = ["all", "differences", "custom", "differences + custom"]

HELP = """Generation Info Filter — custom_fields selectors (one per line):

  ClassType            every occurrence of that node, all its params
  ClassType[n]         the n-th occurrence (1-based), all its params
  ClassType[n].param   a single param of the n-th occurrence
  ClassType.param      that param on every occurrence

Names come straight from the Generation Info dump:
  [KSamplerSelect] sampler_name: euler   ->  KSamplerSelect.sampler_name
  the 2nd [PrimitiveString]              ->  PrimitiveString[2]

Modes:
  all                  every field
  differences          only fields that differ across inputs (each block shows them all)
  custom               only the fields named in custom_fields
  differences + custom union of the two
"""


def _idx(key):
    m = _DATA_RE.match(key)
    return int(m.group(1)) if m else 1 << 30


def _parse(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return []
    return []


def _norm(v):
    """Stable string used for difference comparison."""
    return " ".join((v if isinstance(v, str) else repr(v)).split())


def _disp(v, cap=300):
    s = " ".join((v if isinstance(v, str) else repr(v)).split())
    return s if len(s) <= cap else s[:cap] + "…"


def _raw(v):
    """Full (untruncated) string value for the structured output / report DB."""
    if isinstance(v, str):
        return v
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return str(v)
    return json.dumps(v, ensure_ascii=False)


def _field_map(data):
    """[{class_type, ord, params}] -> OrderedDict[(class_type, ord, key)] = value."""
    fm = OrderedDict()
    for entry in data:
        ct = entry.get("class_type", "?")
        od = int(entry.get("ord", 1))
        for k, val in entry.get("params", {}).items():
            fm[(ct, od, k)] = val
    return fm


def _parse_selectors(text):
    sels = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        m = _SELECTOR_RE.match(line)
        if not m:
            continue
        ct, ordn, param = m.group(1).strip(), m.group(2), m.group(3)
        sels.append((ct, int(ordn) if ordn else None, param.strip() if param else None))
    return sels


def _matches(key, sels):
    ct, od, k = key
    for sct, sord, sparam in sels:
        if sct == ct and (sord is None or sord == od) and (sparam is None or sparam == k):
            return True
    return False


class GenerationInfoFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_1": ("GEN_INFO",),
                "mode": (_MODES, {"default": "differences", "tooltip": "all = every field; differences = only fields that vary across inputs; custom = only fields in custom_fields; last = union of both."}),
                "custom_fields": ("STRING", {"multiline": True, "default": "", "tooltip": "One selector per line: ClassType / ClassType[n] / ClassType[n].param / ClassType.param ([n] is the 1-based occurrence)."}),
                "separator": ("STRING", {"default": "---", "tooltip": "Line placed between the per-image blocks. Match the compare node's settings_separator."}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "Skip empty / unconnected inputs (e.g. a bypassed branch) so the blocks stay aligned with the rest of the comparison."}),
            },
            "optional": {
                "data_2": ("GEN_INFO",),
            },
        }

    RETURN_TYPES = ("STRING", "GEN_SETTINGS", "STRING")
    RETURN_NAMES = ("settings", "settings_data", "help")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/util"

    def run(self, mode="differences", custom_fields="", separator="---", skip_empty=True, **kwargs):
        maps = []
        for key in sorted((k for k in kwargs if _DATA_RE.match(k)), key=_idx):
            v = kwargs.get(key)
            if v is None:
                continue
            fm = _field_map(_parse(v))
            if skip_empty and not fm:
                continue
            maps.append(fm)
        if not maps:
            return ("", "[]", HELP)

        # Global field order = first appearance across inputs (walk order preserved).
        order = list(OrderedDict((k, None) for fm in maps for k in fm).keys())

        sels = _parse_selectors(custom_fields)
        want_diff = mode in ("differences", "differences + custom")
        want_custom = mode in ("custom", "differences + custom")

        def differs(key):
            seen = {(_norm(fm[key]) if key in fm else _ABSENT) for fm in maps}
            return len(seen) > 1

        selected = []
        for key in order:
            keep = (mode == "all")
            keep = keep or (want_diff and differs(key))
            keep = keep or (want_custom and _matches(key, sels))
            if keep:
                selected.append(key)

        # A class needs a "#ord" tag only if more than one of its occurrences is shown.
        ords_per_class = {}
        for ct, od, _ in selected:
            ords_per_class.setdefault(ct, set()).add(od)

        def label(ct, od):
            return f"[{ct} #{od}]" if len(ords_per_class.get(ct, ())) > 1 else f"[{ct}]"

        def qkey(ct, od, k):
            base = f"{ct} #{od}" if len(ords_per_class.get(ct, ())) > 1 else ct
            return f"{base}.{k}"

        blocks, data_blocks = [], []
        for fm in maps:
            grouped, fields = OrderedDict(), []
            for key in selected:
                ct, od, k = key
                if key in fm:
                    grouped.setdefault((ct, od), []).append(f"{k}: {_disp(fm[key])}")
                    fields.append({"key": qkey(ct, od, k), "value": _raw(fm[key])})
                else:
                    grouped.setdefault((ct, od), []).append(f"{k}: —")
            lines = [f"{label(ct, od)} {', '.join(parts)}" for (ct, od), parts in grouped.items()]
            blocks.append("\n".join(lines))
            data_blocks.append(fields)

        joiner = "\n" + (separator or "---") + "\n"
        return (joiner.join(blocks), json.dumps(data_blocks, ensure_ascii=False), HELP)


NODE_CLASS_MAPPINGS = {"GenerationInfoFilter": GenerationInfoFilter}
NODE_DISPLAY_NAME_MAPPINGS = {"GenerationInfoFilter": "Generation Info Filter"}
