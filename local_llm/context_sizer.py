"""Context Sizer (GGUF) — measure exactly how many tokens a request needs and suggest an `n_ctx`.

Sizing `n_ctx` close to what a request actually uses saves VRAM (the KV cache shrinks with it),
which on a tight card lets you offload more layers to the GPU / avoid a spill or OOM. `n_ctx`
must still hold the whole input PLUS the generated output, so this node reports:

    suggested_n_ctx = round_up( (text + template + image tokens) + output_budget + margin , 256 )

Counting is **lean** — it does NOT load the full generation model, so it's cheap enough to sit
inline in a graph (no double-load / reload against the model that runs next):
- **text** via the vocab-only tokenizer (no weights);
- **image** via `mtmd_tokenize` on the clip/mmproj only (no LLM weights, no forward pass) — image
  tokens depend on the model + resolution and are counted for real, not guessed.

If the lean image path can't run weight-free on a given build, it **falls back** automatically to
an exact full-model prefill (heavier, but correct); the `info` output and console say which mode
was used. `output_budget` is taken from the config's `max_tokens`; `margin` is your safety headroom.

Inputs are all sockets (no text fields): wire the `config`, your prompt text into the auto-growing
`text_*` inputs, and optionally an `image` (a batch is fine — the node sizes for the largest image).
"""
import re

from .llm_node import (
    LLM_CONFIG, count_tokens, count_image_tokens, count_prompt,
    _shutdown_worker, UNLOAD_MODES, resolve_unload,
)
from ..categories import CAT_LLM

_IDX_RE = re.compile(r"^text_(\d+)$")


def _round_up(v, step=256):
    return int(((max(0, int(v)) + step - 1) // step) * step)


def _flatten_images(image):
    """A single [B,H,W,C] batch (or [H,W,C]) — or a list of them — into single-frame tensors."""
    frames = []
    for v in (image if isinstance(image, list) else [image]):
        if v is None or not hasattr(v, "ndim"):
            continue
        if v.ndim == 4:
            for i in range(int(v.shape[0])):
                frames.append(v[i:i + 1])
        elif v.ndim == 3:
            frames.append(v[None, ...])
    return frames


class ContextSizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (LLM_CONFIG, {"tooltip": "A 'Local LLM Settings (GGUF)' node — its model (and mmproj, for image sizing) and max_tokens are used."}),
            },
            "optional": {
                "text_1": ("STRING", {"forceInput": True, "tooltip": "Prompt text to size (e.g. your rubric / prompt). Connect-only; more text_* slots appear as you wire them and are joined with newlines. The config's system prompt is counted automatically."}),
                "image": ("IMAGE", {"tooltip": "Optional image(s) to size for — needs an mmproj on the config (Vision Settings). A batch is fine; the node reports the largest image's token cost."}),
                "margin": ("INT", {"default": 64, "min": 0, "max": 8192, "step": 16, "tooltip": "Safety headroom added on top of input + output when suggesting n_ctx."}),
                "unload_after_run": (UNLOAD_MODES, {"default": "config default", "tooltip": "Free the model from VRAM after THIS node runs, without touching the shared config. 'config default' follows the Settings node; 'unload after run' frees VRAM (a different model runs next); 'keep loaded' stays warm (the same model works next)."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("text_tokens", "image_tokens", "total_tokens", "suggested_n_ctx", "info")
    FUNCTION = "run"
    CATEGORY = CAT_LLM

    @staticmethod
    def _idx(key):
        m = _IDX_RE.match(key)
        return int(m.group(1)) if m else 1 << 30

    def run(self, config, image=None, margin=64, unload_after_run="config default", **kwargs):
        cfg = config if isinstance(config, dict) else {}
        try:
            return self._measure(cfg, image, margin, kwargs)
        finally:
            # This node can load the FULL model (Path B). Per-node override (default follows the
            # config), applied once at the end — each _measure probe reuses the warm model, so
            # unloading per probe would reload every time. Frees VRAM before your generation.
            if resolve_unload(unload_after_run, cfg):
                _shutdown_worker()

    def _measure(self, cfg, image, margin, kwargs):
        parts = []
        for k in sorted((k for k in kwargs if _IDX_RE.match(k)), key=self._idx):
            v = kwargs.get(k)
            if isinstance(v, str) and v:
                parts.append(v)
        user_text = "\n".join(parts)
        sys_text = (cfg.get("system_prompt") or "") if isinstance(cfg, dict) else ""
        combined = (sys_text.rstrip() + "\n" + user_text).strip() if sys_text.strip() else user_text

        frames = _flatten_images(image)
        mode = "lean (vocab + mtmd)"

        # Text: vocab-only tokenizer — no LLM weights. The chat-template overhead is small and the
        # `margin` covers it. (Replaced with an exact, template-inclusive count in the fallback.)
        text_tokens, terr = count_tokens(cfg, combined)
        if terr:
            return (-1, -1, -1, -1, f"[ERROR] {terr}")

        per_image = []
        if frames:
            counts, ierr = count_image_tokens(cfg, image)
            if ierr or counts is None:
                # Lean path unavailable — fall back to the exact full-model prefill (heavier).
                print(f"[ContextSizer] lean image count unavailable ({ierr}); "
                      f"falling back to full-model prefill.")
                mode = "fallback (full prefill)"
                base, e = count_prompt(cfg, user_text, image=None)
                if e:
                    return (text_tokens, -1, -1, -1, f"[ERROR] fallback failed: {e}")
                text_tokens = base  # exact, template-inclusive
                for fr in frames:
                    pt, e2 = count_prompt(cfg, user_text, image=fr)
                    if e2:
                        return (text_tokens, -1, -1, -1, f"[ERROR] fallback image failed: {e2}")
                    per_image.append(max(0, pt - base))
            else:
                per_image = counts

        image_tokens = max(per_image) if per_image else 0
        total = text_tokens + image_tokens
        output_budget = int(cfg.get("max_tokens", 512) or 512) if isinstance(cfg, dict) else 512
        suggested = _round_up(total + output_budget + int(margin))

        info = (f"mode: {mode}\n"
                f"text: {text_tokens} tok\n"
                f"image: {image_tokens} tok"
                + (f" (largest of {len(per_image)}: {per_image})" if per_image else " (no image)") + "\n"
                f"total input: {total} tok\n"
                f"+ output budget {output_budget} + margin {int(margin)}\n"
                f"→ suggested n_ctx: {suggested}")
        print(f"[ContextSizer] {mode}: text={text_tokens} image={image_tokens} total={total} "
              f"→ n_ctx {suggested}")
        return (text_tokens, image_tokens, total, suggested, info)


NODE_CLASS_MAPPINGS = {"KinburgContextSizer": ContextSizer}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgContextSizer": "Context Sizer (GGUF)"}
