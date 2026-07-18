"""Vision LLM Judge — score a batch/list of images with a vision GGUF against your rubric.

For each image the node asks the vision model (the one carried by a **Local LLM Settings
(GGUF)** config with a **Vision Settings** mmproj) to rate it and returns a small structured
verdict — `score` / `tags` / `comment` — forced into shape by a **GBNF grammar** (the same
guaranteed-structure trick as the Ideogram preset), so any model returns clean JSON.

It reuses the exact machinery of the Local LLM nodes: `build_llm_request` +
`_generate_and_format`. The model is loaded once and kept warm across every image (the load
signature is identical), so only the first image pays the load cost.

Outputs:
- **summary** — a human-readable block per image (for a Show Text / preview node).
- **results_json** — `[{index, score, score_max, tags, comment}, …]`. Wire it into **Image
  Compare**'s `judge_data` input: it renders as a read-only judge section per image (stars /
  tags / comment), separate from your own review controls, with its own show/hide toggle.
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

JUDGE_SYSTEM = (
    "You are a strict, consistent image-quality judge. You are shown ONE image plus evaluation "
    "criteria. Assess the image only against those criteria and be objective. Reply with ONLY a "
    "single JSON object — no prose, no markdown, no code fences."
)

HELP_TEXT = """# Vision LLM Judge — quick help

Rates each input image with a vision GGUF and returns a structured verdict.

## Wiring
- **config** — a `Local LLM Settings (GGUF)` node that HAS a `Vision Settings (GGUF)` (mmproj)
  attached. Without an mmproj the node errors (it needs vision).
- **images** — a batch OR an image list (mixed sizes fine).
- **rubric** — what to judge (e.g. "Rate anatomy, prompt adherence, sharpness. Penalize extra
  fingers, artifacts, watermarks.").
- **prompts** (optional) — the per-image generation prompt(s), `---`-separated blocks (e.g. from
  Get Accumulator (prompts)); lets the judge assess prompt adherence.

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


def _parse_verdict(text, lo, hi):
    """Parse the model's JSON verdict into {score, tags, comment}, coercing / clamping robustly.
    Returns None if no JSON object can be recovered at all."""
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None
    try:
        score = int(round(float(obj.get("score", lo))))
    except (TypeError, ValueError):
        score = lo
    score = max(lo, min(hi, score))
    tags = obj.get("tags", [])
    if not isinstance(tags, list):
        tags = [tags]
    tags = [str(t).strip() for t in tags if str(t).strip()]
    comment = obj.get("comment", "")
    comment = comment if isinstance(comment, str) else str(comment)
    return {"score": score, "tags": tags, "comment": comment.strip()}


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
            },
        }

    INPUT_IS_LIST = True  # gather the whole batch/list and judge every image in ONE run
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("summary", "results_json", "best_index", "help")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def _err(self, msg):
        print(f"[VisionJudge] ERROR: {msg}")
        return (f"[ERROR] {msg}", "[]", -1, HELP_TEXT)

    def run(self, config, images, rubric, prompts="", score_min=1, score_max=5,
            unload_after_run="config default"):
        config = _first(config, {}) or {}
        rubric = (_first(rubric, "") or "").strip()
        prompts = _first(prompts, "") or ""
        lo = int(_first(score_min, 1))
        hi = int(_first(score_max, 5))
        unload_after_run = _first(unload_after_run, "config default")
        if hi < lo:
            lo, hi = hi, lo

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
        judge_cfg["system_prompt"] = JUDGE_SYSTEM
        judge_cfg["context"] = ""
        judge_cfg["output_format"] = "gbnf_grammar"
        judge_cfg["grammar"] = JUDGE_GRAMMAR
        judge_cfg["thinking_directive"] = "model default"
        judge_cfg["strip_think"] = True
        judge_cfg["max_tokens"] = max(int(g("max_tokens", 512) or 512), 128)

        instruction = (
            f'Return a JSON object with: "score" (integer {lo}-{hi}, higher is better), '
            f'"tags" (array of short lowercase keywords for notable qualities or problems, '
            f'e.g. "bad_hands", "sharp", "off_prompt"; may be empty), and '
            f'"comment" (one concise sentence).'
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

            err, ctx = build_llm_request(judge_cfg, user_prompt, image=frame)
            if err:
                verdict = {"score": lo, "tags": ["judge_error"], "comment": err}
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
                    verdict = {"score": lo, "tags": ["judge_error"], "comment": text}
                    print(f"[VisionJudge] image {i + 1}/{n}: {text}")
                else:
                    verdict = _parse_verdict(text, lo, hi)
                    if verdict is None:
                        verdict = {"score": lo, "tags": ["parse_error"],
                                   "comment": (text or "").strip()[:300]}
                        print(f"[VisionJudge] image {i + 1}/{n}: could not parse verdict")
                    else:
                        print(f"[VisionJudge] image {i + 1}/{n}: ★{verdict['score']}/{hi} "
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
            summary_blocks.append(
                f"#{idx + 1}  ★{s}/{hi}  [{', '.join(tags)}]" + (f"\n{comment}" if comment else ""))
            if best_score is None or s > best_score:
                best_score, best_i = s, idx

        results_json = json.dumps(
            [{"index": r["index"], "score": r["score"], "score_max": hi,
              "tags": r["tags"], "comment": r["comment"]} for r in results], ensure_ascii=False)
        return ("\n\n".join(summary_blocks), results_json, best_i, HELP_TEXT)


NODE_CLASS_MAPPINGS = {"VisionLLMJudge": VisionLLMJudge}
NODE_DISPLAY_NAME_MAPPINGS = {"VisionLLMJudge": "Vision LLM Judge"}
