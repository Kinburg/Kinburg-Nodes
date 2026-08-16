"""Prompt Variations — expand a template into many prompts (a ComfyUI list).

Explore a prompt space in one node: write choices with `{a|b|c}` (nested groups allowed) and
optional `__wildcard__` references (one option per line from `<dir>/wildcard.txt`), and the node
emits the cartesian product as a per-item STRING **list** — feed it straight into
`For Each (Open)` → your sampler → an accumulator → `Image Compare` to render every variant.

Syntax:
- `{a|b|c}`        -> a choice; the output is the product of all choice groups.
- nested:          `portrait, {Rembrandt|soft} light{, dramatic|}`  (empty option = "nothing")
- `__name__`       -> replaced by the lines of `<wildcards_dir>/name.txt` (a `{...}` of them);
                      `__sub/name__` reads `<wildcards_dir>/sub/name.txt`. Missing file = left as-is.

`mode`:
- **all**    -> every combination, in order, capped at `limit`.
- **random** -> `limit` random combinations (reproducible via `seed`).
A hard internal ceiling guards against a runaway cartesian explosion.
"""
import os
import re
import random
from ..categories import CAT_PROMPT

MAX_COMBOS = 5000  # hard safety ceiling on how many combinations are ever materialized
_WILDCARD_RE = re.compile(r"__([A-Za-z0-9_\-/]+)__")


def _find_group(s):
    """(open, close) of the first top-level `{...}` group in s, or None."""
    depth, start = 0, -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    return start, i
    return None


def _split_top(body):
    """Split on `|` at brace-depth 0 (so nested groups stay intact)."""
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "{":
            depth += 1
            cur.append(ch)
        elif ch == "}":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _expand(s, cap=MAX_COMBOS):
    """Expand every `{a|b|…}` group into the cartesian product of strings, bounded to `cap`."""
    g = _find_group(s)
    if not g:
        return [s]
    a, b = g
    pre, body, post = s[:a], s[a + 1:b], s[b + 1:]
    out = []
    posts = _expand(post, cap)
    for opt in _split_top(body):
        for eo in _expand(opt, cap):
            for ep in posts:
                out.append(pre + eo + ep)
                if len(out) >= cap:
                    return out
    return out


def _clean(s):
    """Tidy a filled template for prompt use: collapse runs of spaces, and heal the dangling
    commas an empty choice can leave (', ,' / ' ,'), then strip."""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",(\s*,)+", ",", s)
    s = re.sub(r"(^|\n)[ ,]+", r"\1", s)
    s = re.sub(r"[ ,]+($|\n)", r"\1", s)
    return s.strip()


def _apply_wildcards(template, wdir):
    """Replace `__name__` with `{line1|line2|…}` from `<wdir>/name.txt`. Returns (text, missing)."""
    missing = []

    def repl(m):
        name = m.group(1)
        if wdir:
            path = os.path.join(wdir, *name.split("/")) + ".txt"
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f.read().splitlines()
                                 if ln.strip() and not ln.strip().startswith("#")]
                except Exception:
                    lines = []
                if lines:
                    return "{" + "|".join(lines) + "}"
        missing.append(name)
        return m.group(0)

    return _WILDCARD_RE.sub(repl, template), missing


def _default_wildcards_dir():
    try:
        import folder_paths
        d = os.path.join(folder_paths.base_path, "wildcards")
        return d if os.path.isdir(d) else ""
    except Exception:
        return ""


class PromptVariations:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {"multiline": True, "default": "{a photo|a painting} of a {red|blue|green} car", "tooltip": "Template with {a|b|c} choices (nesting allowed) and optional __wildcard__ refs. The output is the product of all choices."}),
                "mode": (["all", "random"], {"default": "all", "tooltip": "all = every combination (capped by limit); random = 'limit' random combinations (reproducible via seed)."}),
                "limit": ("INT", {"default": 25, "min": 1, "max": MAX_COMBOS, "tooltip": "Max prompts to output. In 'all' mode extra combinations past this are dropped (a console note is printed)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Randomness seed for 'random' mode (same seed = same picks)."}),
                "dedupe": ("BOOLEAN", {"default": True, "tooltip": "Drop duplicate resulting prompts (e.g. from overlapping choices)."}),
            },
            "optional": {
                "wildcards_dir": ("STRING", {"default": "", "tooltip": "Folder with <name>.txt files for __name__ refs. Empty = ComfyUI/wildcards if it exists. A missing file leaves the token unchanged."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("prompts", "count", "preview")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "run"
    CATEGORY = CAT_PROMPT

    def run(self, template="", mode="all", limit=25, seed=0, dedupe=True, wildcards_dir=""):
        template = template or ""
        if not template.strip():
            return ([], 0, "")

        wdir = (wildcards_dir or "").strip() or _default_wildcards_dir()
        substituted, missing = _apply_wildcards(template, wdir)
        if missing:
            print(f"[PromptVariations] wildcard file(s) not found (left as-is): {', '.join(sorted(set(missing)))}"
                  + ("" if wdir else " — no wildcards_dir set and no ComfyUI/wildcards folder"))

        combos = _expand(substituted, MAX_COMBOS)
        combos = [c for c in (_clean(x) for x in combos) if c != ""]
        if dedupe:
            seen, uniq = set(), []
            for c in combos:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
            combos = uniq

        exploded = len(combos) >= MAX_COMBOS
        if mode == "random" and len(combos) > limit:
            combos = random.Random(int(seed)).sample(combos, int(limit))
        elif len(combos) > limit:
            print(f"[PromptVariations] {len(combos)} combinations, keeping the first {limit} "
                  f"(raise 'limit' for more).")
            combos = combos[:limit]

        if exploded:
            print(f"[PromptVariations] hit the {MAX_COMBOS}-combination safety ceiling — "
                  f"the template may be too broad.")

        preview = "\n".join(combos)
        print(f"[PromptVariations] emitting {len(combos)} prompt(s).")
        return (combos, len(combos), preview)


NODE_CLASS_MAPPINGS = {"PromptVariations": PromptVariations}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptVariations": "Prompt Variations"}
