"""Token Counter (GGUF) — count how many tokens a text is under a model's tokenizer.

Uses the same worker as the LLM nodes, but a **vocab-only** load (just the tokenizer, no
weights, no VRAM) — and reuses an already-loaded generation model if its path matches, so
counting never disturbs generation. No text is generated. Handy for budgeting a prompt against
a model's context window (`n_ctx`). Counts the raw text tokens (no BOS added).
"""
from .llm_node import LLM_CONFIG, count_tokens, _shutdown_worker, UNLOAD_MODES, resolve_unload


class TokenCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (LLM_CONFIG, {"tooltip": "A 'Local LLM Settings (GGUF)' node — only its model is used (for the tokenizer)."}),
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to tokenize (type here or wire a STRING in)."}),
            },
            "optional": {
                "unload_after_run": (UNLOAD_MODES, {"default": "config default", "tooltip": "Free the model from VRAM after THIS node runs, without touching the shared config. 'config default' follows the Settings node; 'unload after run' frees VRAM (a different model runs next); 'keep loaded' stays warm (the same model counts and then works)."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("token_count", "char_count", "info")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, config, text="", unload_after_run="config default"):
        cfg = config if isinstance(config, dict) else {}
        text = text if isinstance(text, str) else str(text)
        chars = len(text)
        try:
            tokens, err = count_tokens(cfg, text)
            if err:
                return (-1, chars, f"[ERROR] {err}")
            info = f"{tokens} tokens · {chars} chars" + (f" · {chars / tokens:.2f} chars/token" if tokens else "")
            return (tokens, chars, info)
        finally:
            # Per-node override (default follows the config). Counting itself is vocab-only, but
            # this also frees a warm generation model if one was up.
            if resolve_unload(unload_after_run, cfg):
                _shutdown_worker()


NODE_CLASS_MAPPINGS = {"KinburgTokenCounter": TokenCounter}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgTokenCounter": "Token Counter (GGUF)"}
