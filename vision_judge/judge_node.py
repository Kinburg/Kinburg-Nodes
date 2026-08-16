"""Vision LLM Judge — score a batch/list of images with a vision GGUF against your rubric.

For each image the node asks the vision model (the one carried by a **Local LLM Settings
(GGUF)** config with a **Vision Settings** mmproj) to rate it and returns a small structured
verdict — `score` / `tags` / `comment` — forced into shape by a **GBNF grammar** (the same
guaranteed-structure trick as the Ideogram preset), so any model returns clean JSON.

It reuses the exact machinery of the Local LLM nodes: `build_llm_request` +
`_generate_and_format`. The model is loaded once and kept warm across every image (the load
signature is identical), so only the first image pays the load cost.

Multi-criteria mode: the `criteria` field (one per line, `name` or `name: description`) makes the
judge score EACH criterion on the scale, with the overall `score` being their average. The grammar
is generated on the fly to force `{"scores": {…}, "tags", …}`. The field ships **pre-filled with an
example** (DEFAULT_CRITERIA); clear it for a single overall score (the original behaviour).

The two built-in prompts are editable: `system_prompt` (the judge persona; the default is
pre-filled) and `comment_style` (how the `comment` reads, default "one concise sentence"). Both
fall back to the built-in default when left blank. The JSON shape itself stays managed (it is
generated from `criteria` to match the grammar), so editing these can only change quality, never
break parsing.

Outputs:
- **summary** — a human-readable block per image (for a Show Text / preview node); in
  multi-criteria mode it also lists the per-criterion sub-scores.
- **results_json** — `[{index, score, score_max, tags, comment[, scores]}, …]` (`scores` present
  only in multi-criteria mode). Wire it into **Image Compare**'s `judge_data` input: it renders
  as a read-only judge section per image (stars / tags / comment), separate from your own review
  controls, with its own show/hide toggle.
- **best_index** — 0-based index of the highest-scoring image (ties → first).

Variant A (minimum): per-image verdict. Auto-selecting the best into the next stage / a
threshold loop-gate (variant B) is a later step.
"""
import os
import re
import json

from ..local_llm.llm_node import (
    LLM_CONFIG, build_llm_request, _generate_and_format, _shutdown_worker,
    _resolve_path, PLACEHOLDER, UNLOAD_MODES, resolve_unload,
)
from ..categories import CAT_LLM

# GBNF grammar forcing {"score": <int>, "tags": [<string>…], "comment": "<string>"}.
# Uses the standard JSON string production (escapes + any non-control codepoint), so a comment
# in any language — including Cyrillic — is allowed and always parses with json.loads.
JUDGE_GRAMMAR = r'''root ::= ws "{" ws "\"score\"" ws ":" ws int ws "," ws "\"tags\"" ws ":" ws tags ws "," ws "\"comment\"" ws ":" ws string ws "}" ws
tags ::= "[" ws (string (ws "," ws string)*)? ws "]"
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt/] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
int ::= digit digit?
digit ::= [0-9]
ws ::= [ \t\n]*
'''


def _grammar_tail():
    """The shared productions of JUDGE_GRAMMAR (everything but the `root` line) — reused verbatim
    when building the multi-criteria grammar, so the tricky string/char escapes stay correct."""
    return "\n".join(l for l in JUDGE_GRAMMAR.split("\n") if l and not l.startswith("root ::="))


def _sanitize_key(name):
    """A JSON/GBNF-safe object key from a free-text criterion name: lowercase, [a-z0-9_] only."""
    return re.sub(r"[^a-z0-9_]+", "_", (name or "").strip().lower()).strip("_")


def _parse_criteria(text, cap=12):
    """Parse the `criteria` field — one per line, `name` or `name: description`
    (also ` — ` / ` - ` as the separator). Returns a list of (key, label, description) with
    unique sanitized keys, capped at `cap`. Empty input → [] (→ single-score mode)."""
    out, seen = [], set()
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            label, desc = line.split(":", 1)
        elif " — " in line:
            label, desc = line.split(" — ", 1)
        elif " - " in line:
            label, desc = line.split(" - ", 1)
        else:
            label, desc = line, ""
        label, desc = label.strip(), desc.strip()
        key = _sanitize_key(label)
        if not key:
            continue
        base, k, n = key, key, 2
        while k in seen:
            k = f"{base}_{n}"; n += 1
        seen.add(k)
        out.append((k, label, desc))
        if len(out) >= cap:
            break
    return out


def _build_multi_grammar(keys):
    """A GBNF grammar forcing {"scores": {<key>: int, …}, "tags": [...], "comment": "..."} with a
    fixed integer sub-score for each criterion key, in order. Reuses _grammar_tail()."""
    clauses = ['ws "\\"' + k + '\\"" ws ":" ws int' for k in keys]
    scores_body = ' ws "," '.join(clauses)
    root = ('root ::= ws "{" ws "\\"scores\\"" ws ":" ws "{" ' + scores_body +
            ' ws "}" ws "," ws "\\"tags\\"" ws ":" ws tags ws "," '
            'ws "\\"comment\\"" ws ":" ws string ws "}" ws')
    return root + "\n" + _grammar_tail() + "\n"


JUDGE_SYSTEM = (
    "You are a strict, consistent image-quality judge. You are shown ONE image plus evaluation "
    "criteria. Assess the image only against those criteria and be objective. Reply with ONLY a "
    "single JSON object — no prose, no markdown, no code fences."
)

# Pre-filled example for the `criteria` field: a sensible multi-criteria starting point that also
# documents the "name: description" format. Edit or clear it (empty → single overall score).
DEFAULT_CRITERIA = (
    "overall_quality: style matches the prompt, no artifacts, no excess noise, correct proportions, good color reproduction\n"
    "anatomy: all required limbs present, no extra limbs, correct placement, natural pose, proportional body\n"
    "prompt_compliance: how accurately the image follows the generation prompt\n"
    "camera: camera angle and camera settings match the intent\n"
    "text: if the prompt requests text — present, character-accurate, correct color/font/size/placement (if the prompt has NO text, give the top score)"
)

HELP_TEXT = """# Vision LLM Judge — quick help

Rates each input image with a vision GGUF and returns a structured verdict.

## Wiring
- **config** — a `Local LLM Settings (GGUF)` node that HAS a `Vision Settings (GGUF)` (mmproj)
  attached. Without an mmproj the node errors (it needs vision).
- **images** — a batch OR an image list (mixed sizes fine).
- **rubric** — what to judge (e.g. "Rate anatomy, prompt adherence, sharpness. Penalize extra
  fingers, artifacts, watermarks.").
- **criteria** — one criterion per line (`name` or `name: description`) to score EACH separately
  (overall = their average). **Pre-filled with an example** (overall_quality / anatomy /
  prompt_compliance / camera / text) — edit it, or **clear it** for a single overall score.
- **prompts** (optional) — the per-image generation prompt(s), `---`-separated blocks (e.g. from
  Get Accumulator (prompts)); lets the judge assess prompt adherence.
- **system_prompt** (optional) — who the judge is; the built-in default is pre-filled, edit to
  change persona/strictness. Cleared → the built-in default is used.
- **comment_style** (optional) — how the `comment` reads; default `one concise sentence`. For a
  detailed review set e.g. `two to four sentences covering strengths and weaknesses` (and raise
  the Settings `max_tokens` so the long comment isn't cut off).

## Outputs
- **results_json** → wire into Image Compare's `judge_data` input: a read-only judge section
  per image (stars / tags / comment), toggled from the page header.
- **summary** → a Show Text node for a readable report.
- **best_index** → 0-based index of the top-scoring image.

## Tips
- Use a LOW temperature on the Settings node (e.g. 0.0–0.2) for stable, repeatable scores.
- The verdict shape is grammar-forced, so it's guaranteed on any model — but the *quality* of the
  scores is only as good as your vision model and how concrete your rubric is.
- It runs one inference per image; the model stays loaded across the whole batch.
"""


def _first(v, default=None):
    """INPUT_IS_LIST wraps every input in a list; take the first element (or default)."""
    if isinstance(v, list):
        return v[0] if v else default
    return v


def _flatten_images(images):
    """The `images` input (INPUT_IS_LIST) is a list of IMAGE tensors, each a [B,H,W,C] batch or a
    [H,W,C] frame, possibly of different sizes. Flatten to a list of single-frame [1,H,W,C]
    tensors, in order."""
    frames = []
    for v in (images if isinstance(images, list) else [images]):
        if v is None or not hasattr(v, "ndim"):
            continue
        if v.ndim == 4:
            for i in range(int(v.shape[0])):
                frames.append(v[i:i + 1])
        elif v.ndim == 3:
            frames.append(v[None, ...])
    return frames


def _split_prompts(text):
    """Per-image prompts: blocks separated by a '---' line (matches Image Compare / accumulators)."""
    if not text or not text.strip():
        return []
    blocks, cur = [], []
    for line in text.split("\n"):
        if line.strip() == "---":
            blocks.append("\n".join(cur).strip())
            cur = []
        else:
            cur.append(line)
    blocks.append("\n".join(cur).strip())
    return blocks


def _recover_obj(text):
    """Best-effort JSON object from the model's reply: parse whole, else the first {...} block."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _clamp_int(v, lo, hi, default):
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _clean_tags(tags):
    if not isinstance(tags, list):
        tags = [tags]
    return [str(t).strip() for t in tags if str(t).strip()]


def _clean_comment(obj):
    c = obj.get("comment", "")
    return (c if isinstance(c, str) else str(c)).strip()


def _parse_verdict(text, lo, hi):
    """Single-score verdict → {score, tags, comment}. None if no JSON object recovered."""
    obj = _recover_obj(text)
    if not isinstance(obj, dict):
        return None
    return {"score": _clamp_int(obj.get("score", lo), lo, hi, lo),
            "tags": _clean_tags(obj.get("tags", [])), "comment": _clean_comment(obj)}


def _parse_verdict_multi(text, lo, hi, keys):
    """Multi-criteria verdict → {score (=mean of sub-scores), scores:{key:int}, tags, comment}.
    None if no JSON object recovered."""
    obj = _recover_obj(text)
    if not isinstance(obj, dict):
        return None
    sc = obj.get("scores", {})
    if not isinstance(sc, dict):
        sc = {}
    scores = {k: _clamp_int(sc.get(k, lo), lo, hi, lo) for k in keys}
    # overall = mean of sub-scores, rounded half-up (4.5 -> 5, not Python's banker's 4). Scores are
    # non-negative, so int(x + 0.5) is a correct round-half-up.
    mean = sum(scores.values()) / len(scores) if scores else float(lo)
    overall = _clamp_int(int(mean + 0.5), lo, hi, lo)
    return {"score": overall, "scores": scores,
            "tags": _clean_tags(obj.get("tags", [])), "comment": _clean_comment(obj)}


class VisionLLMJudge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (LLM_CONFIG, {"tooltip": "A 'Local LLM Settings (GGUF)' node WITH a 'Vision Settings (GGUF)' (mmproj) attached. The judge needs vision."}),
                "images": ("IMAGE", {"tooltip": "Images to judge — a batch OR an image list (mixed sizes fine)."}),
                "rubric": ("STRING", {"multiline": True, "default": "Rate the overall image quality: anatomy/structure, sharpness and detail, and how well it matches the prompt. Penalize artifacts, extra or malformed limbs/fingers, and watermarks.", "tooltip": "What to judge each image on. Be concrete."}),
            },
            "optional": {
                "prompts": ("STRING", {"forceInput": True, "tooltip": "Optional per-image generation prompts, '---'-separated blocks (e.g. Get Accumulator (prompts)). Lets the judge assess prompt adherence."}),
                "score_min": ("INT", {"default": 1, "min": 0, "max": 100, "tooltip": "Lowest score on the scale."}),
                "score_max": ("INT", {"default": 5, "min": 1, "max": 100, "tooltip": "Highest score on the scale."}),
                "unload_after_run": (UNLOAD_MODES, {"default": "config default", "tooltip": "Free the model from VRAM after THIS node runs, without touching the shared config. 'config default' follows the Settings node; 'unload after run' frees VRAM before your image generation; 'keep loaded' stays warm."}),
                # New widgets are appended AFTER the originals so existing workflows keep their
                # positional widget values intact.
                "criteria": ("STRING", {"multiline": True, "default": DEFAULT_CRITERIA, "tooltip": "Multi-criteria mode: one criterion per line, 'name' or 'name: what it means'. The judge scores EACH on the score scale and the overall score is their average. Pre-filled with an example — edit it, or CLEAR it for a single overall score."}),
                "system_prompt": ("STRING", {"multiline": True, "default": JUDGE_SYSTEM, "tooltip": "Who the judge is (the model's system prompt). The built-in default is shown — edit to change persona/strictness. Cleared → falls back to the built-in default."}),
                "comment_style": ("STRING", {"default": "one concise sentence", "tooltip": "How the 'comment' field should read, inserted into the JSON instruction. Default 'one concise sentence'; e.g. 'two to four sentences covering strengths and weaknesses' for a detailed review (raise the Settings max_tokens for long comments)."}),
            },
        }

    INPUT_IS_LIST = True  # gather the whole batch/list and judge every image in ONE run
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("summary", "results_json", "best_index", "help")
    FUNCTION = "run"
    CATEGORY = CAT_LLM

    def _err(self, msg):
        print(f"[VisionJudge] ERROR: {msg}")
        return (f"[ERROR] {msg}", "[]", -1, HELP_TEXT)

    def run(self, config, images, rubric, prompts="", score_min=1, score_max=5,
            unload_after_run="config default", criteria="", system_prompt="", comment_style=""):
        config = _first(config, {}) or {}
        rubric = (_first(rubric, "") or "").strip()
        criteria = _first(criteria, "") or ""
        prompts = _first(prompts, "") or ""
        lo = int(_first(score_min, 1))
        hi = int(_first(score_max, 5))
        unload_after_run = _first(unload_after_run, "config default")
        # Externalized prompts: use the user's text, fall back to the built-in defaults if blank.
        system_prompt = (_first(system_prompt, "") or "").strip() or JUDGE_SYSTEM
        comment_style = (_first(comment_style, "") or "").strip() or "one concise sentence"
        if hi < lo:
            lo, hi = hi, lo

        crit = _parse_criteria(criteria)   # [] → single-score mode
        keys = [k for (k, _l, _d) in crit]
        multi = bool(keys)

        if not isinstance(config, dict):
            return self._err("config is not a Local LLM Settings bundle.")

        g = config.get
        model = _resolve_path(g("model", PLACEHOLDER), g("model_path", ""))
        if not model or not os.path.isfile(model):
            return self._err(f"model file not found: {model or '(none selected)'} — check the Settings node.")
        mmproj = _resolve_path(g("mmproj", PLACEHOLDER), g("mmproj_path", ""))
        if not mmproj or not os.path.isfile(mmproj):
            return self._err("the config has no mmproj — attach a 'Vision Settings (GGUF)' node. The judge needs a vision model.")

        frames = _flatten_images(images)
        if not frames:
            return self._err("no images connected.")
        per_prompts = _split_prompts(prompts)

        # A judge-scoped copy of the config: our system prompt + grammar-forced JSON, no stray
        # context/output-format from the user's Settings node.
        judge_cfg = dict(config)
        judge_cfg["system_prompt"] = system_prompt
        judge_cfg["context"] = ""
        judge_cfg["output_format"] = "gbnf_grammar"
        judge_cfg["grammar"] = _build_multi_grammar(keys) if multi else JUDGE_GRAMMAR
        judge_cfg["thinking_directive"] = "model default"
        judge_cfg["strip_think"] = True
        # Multi-criteria needs headroom for the extra fields.
        floor = 128 + (24 * len(keys) if multi else 0)
        judge_cfg["max_tokens"] = max(int(g("max_tokens", 512) or 512), floor)

        if multi:
            crit_lines = "\n".join(
                f'  - "{k}"' + (f': {d}' if d else (f' ({label})' if label.lower() != k else ''))
                for (k, label, d) in crit)
            instruction = (
                f'Return a JSON object with:\n'
                f'- "scores": an object giving an integer {lo}-{hi} (higher is better) for EACH of '
                f'these exact keys:\n{crit_lines}\n'
                f'- "tags": array of short lowercase keywords for notable qualities/problems '
                f'(e.g. "bad_hands", "off_prompt"; may be empty)\n'
                f'- "comment": {comment_style}.'
            )
        else:
            instruction = (
                f'Return a JSON object with: "score" (integer {lo}-{hi}, higher is better), '
                f'"tags" (array of short lowercase keywords for notable qualities or problems, '
                f'e.g. "bad_hands", "sharp", "off_prompt"; may be empty), and '
                f'"comment" ({comment_style}).'
            )

        want_unload_llm = resolve_unload(unload_after_run, config)
        want_unload_comfy = bool(g("unload_comfy_models", True))

        results = []
        n = len(frames)
        # The node's status bar advances one step per image (k of N) — not per token. The inner
        # per-token bar in _generate_and_format is disabled (show_progress=False) so it doesn't
        # fight this one for the same executing node.
        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(n)
        except Exception:
            pbar = None

        for i, frame in enumerate(frames):
            parts = [rubric or "Rate the overall quality of this image."]
            if i < len(per_prompts) and per_prompts[i].strip():
                parts.append("The image was generated from this prompt:\n" + per_prompts[i].strip())
            parts.append(instruction)
            user_prompt = "\n\n".join(parts)

            def _fallback(tag, comment):
                v = {"score": lo, "tags": [tag], "comment": comment}
                if multi:
                    v["scores"] = {k: lo for k in keys}
                return v

            err, ctx = build_llm_request(judge_cfg, user_prompt, image=frame)
            if err:
                verdict = _fallback("judge_error", err)
                print(f"[VisionJudge] image {i + 1}/{n}: request error: {err}")
            else:
                out = _generate_and_format(
                    ctx["req"], ctx["load_sig"], ctx["max_tokens"],
                    want_unload_comfy and i == 0,   # free ComfyUI VRAM once, before the first image
                    False,                          # keep the LLM warm across all images
                    ctx["directive"], ctx["strip_think"], ctx["answer_marker"], ctx["help"],
                    show_progress=False)
                text = out[0]
                if isinstance(text, str) and text.startswith("[ERROR]"):
                    verdict = _fallback("judge_error", text)
                    print(f"[VisionJudge] image {i + 1}/{n}: {text}")
                else:
                    verdict = _parse_verdict_multi(text, lo, hi, keys) if multi else _parse_verdict(text, lo, hi)
                    if verdict is None:
                        verdict = _fallback("parse_error", (text or "").strip()[:300])
                        print(f"[VisionJudge] image {i + 1}/{n}: could not parse verdict")
                    else:
                        extra = ("  " + " · ".join(f"{k} {verdict['scores'][k]}" for k in keys)) if multi else ""
                        print(f"[VisionJudge] image {i + 1}/{n}: ★{verdict['score']}/{hi}{extra} "
                              f"[{', '.join(verdict['tags'])}]")
            verdict["index"] = i
            results.append(verdict)
            if pbar is not None:
                try:
                    pbar.update_absolute(i + 1)
                except Exception:
                    pass

        if want_unload_llm:
            _shutdown_worker()

        # Build the outputs.
        summary_blocks, best_i, best_score = [], -1, None
        for r in results:
            s, tags, comment, idx = r["score"], r["tags"], r["comment"], r["index"]
            head = f"#{idx + 1}  ★{s}/{hi}"
            if multi and r.get("scores"):
                head += "  (" + " · ".join(f"{k} {r['scores'].get(k, lo)}/{hi}" for k in keys) + ")"
            head += f"  [{', '.join(tags)}]"
            summary_blocks.append(head + (f"\n{comment}" if comment else ""))
            if best_score is None or s > best_score:
                best_score, best_i = s, idx

        items = []
        for r in results:
            item = {"index": r["index"], "score": r["score"], "score_max": hi,
                    "tags": r["tags"], "comment": r["comment"]}
            if multi:
                item["scores"] = r.get("scores", {})
            items.append(item)
        results_json = json.dumps(items, ensure_ascii=False)
        return ("\n\n".join(summary_blocks), results_json, best_i, HELP_TEXT)


NODE_CLASS_MAPPINGS = {"VisionLLMJudge": VisionLLMJudge}
NODE_DISPLAY_NAME_MAPPINGS = {"VisionLLMJudge": "Vision LLM Judge"}
