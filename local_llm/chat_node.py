"""Local LLM Chat (GGUF) — a self-contained multi-turn chat node.

The node itself is deliberately bare: just the **chat window**, the **user_message** field and the
**Send / Approve / Clear** buttons. Everything about *how* to generate — model, system prompt,
sampling, loader options, reasoning split, output format, and the optional vision inputs — comes in
through a single **`config`** link from a **Local LLM Settings (GGUF)** node.

Interaction (buttons live in web/chat_llm.js):
  • **📨 Send** runs the workflow up to this node: ``run()`` generates a reply from the stored
    history + your message (+ image), shows it, and BLOCKS the downstream branch
    (``ExecutionBlocker``) so nothing past the node runs while you chat.
  • **✅ Approve** runs with the gate open: ``run()`` skips generation and emits the **last reply**
    on ``text``, so it flows downstream immediately (no re-generation, any seed).

The conversation lives in a hidden, serialized ``history_json`` widget (persists in the workflow).
The frontend appends each turn; ``run()`` only reads it.
"""
import os
import json

from .llm_node import (
    HELP_TEXT, VISION_HELP_TEXT, PLACEHOLDER,
    _resolve_path, _encode_images, _generate_and_format, _apply_directive,
    _apply_output_format, _parse_extra_args, _with_context, _VISION_HANDLER_KEY,
)
from .settings_node import LLM_CONFIG

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except Exception:  # pragma: no cover - only inside ComfyUI
    ExecutionBlocker = None


def _last_assistant(history):
    for m in reversed(history):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


class LocalLLMChatGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (LLM_CONFIG, {"tooltip": "Wire a 'Local LLM Settings (GGUF)' node here — it carries the model, system prompt, sampling, vision, etc."}),
                # Single-line (hidden by the frontend, fed from the chat input box). Kept as a
                # widget so the message reaches run(); multiline would leave a stray DOM textarea.
                "user_message": ("STRING", {"default": "", "tooltip": "Filled from the chat input box; you don't edit this directly."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image for vision — attached to the current turn. Needs an mmproj set on the Settings node; if there's none, you'll get an error in the chat."}),
                # hidden chat state, managed by web/chat_llm.js
                "history_json": ("STRING", {"default": "[]"}),
                "nonce": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "approved": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "help")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Kinburg-Nodes/LLM"

    def run(self, config, user_message, image=None, history_json="[]", nonce=0, approved=False,
            unique_id=None):
        try:
            history = json.loads(history_json) if history_json else []
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []

        # ── Approve: release the last saved reply downstream, no generation. ──────────────
        if approved:
            return (_last_assistant(history), HELP_TEXT)

        def _blocked(reply, thoughts=""):
            gate = ExecutionBlocker(None) if ExecutionBlocker is not None else reply
            payload = json.dumps({"reply": reply, "thoughts": thoughts or ""}, ensure_ascii=False)
            return {"ui": {"kinburg_chatllm": [payload]}, "result": (gate, HELP_TEXT)}

        cfg = config if isinstance(config, dict) else {}
        g = cfg.get  # settings come from the config bundle; .get keeps it robust

        resolved = _resolve_path(g("model", PLACEHOLDER), g("model_path", ""))
        if not resolved or not os.path.isfile(resolved):
            return _blocked(f"[ERROR] Model file not found: {resolved or '(none selected)'} — check the Settings node.")

        # Vision: the image comes from THIS node's input; the mmproj/handler live in the config.
        mmproj_resolved = _resolve_path(g("mmproj", PLACEHOLDER), g("mmproj_path", ""))
        use_vision = False
        if image is not None:
            if mmproj_resolved and os.path.isfile(mmproj_resolved):
                use_vision = True
            else:
                return _blocked("[ERROR] An image is connected, but the Settings node has no mmproj set. "
                                "Add an mmproj (vision projector .gguf) in Local LLM Settings to chat about images.")
        help_text = VISION_HELP_TEXT if use_vision else HELP_TEXT

        try:
            extra = _parse_extra_args(g("extra_load_args", ""))
        except ValueError as e:
            return _blocked(f"[ERROR] {e}")

        images = []
        if use_vision:
            try:
                images = _encode_images(image, int(g("image_max_side", 1024)))
            except Exception as e:
                return _blocked(f"[ERROR] Failed to encode image: {e}")

        user_prompt, directive = _apply_directive(
            user_message, g("thinking_directive", "model default"), g("custom_directive", ""))
        # Append reference material (Character Card / Context Collector) to the system prompt.
        system = _with_context(g("system_prompt", "You are a helpful assistant."), g("context", ""))
        eff_format, eff_grammar, eff_system = _apply_output_format(
            g("output_format", "text"), g("grammar", ""), system)

        clean_history = [{"role": m["role"], "content": m["content"]} for m in history
                         if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                         and isinstance(m.get("content"), str)]

        req = {
            "model_path": resolved,
            "system_prompt": eff_system,
            "user_prompt": user_prompt,
            "history": clean_history,
            "max_tokens": int(g("max_tokens", 512)),
            "temperature": float(g("temperature", 0.7)),
            "top_p": float(g("top_p", 0.95)),
            "top_k": int(g("top_k", 40)),
            "min_p": float(g("min_p", 0.0)),
            "repeat_penalty": float(g("repeat_penalty", 1.1)),
            "stop": [s for s in (g("stop", "") or "").splitlines() if s.strip()],
            "n_ctx": int(g("n_ctx", 4096)),
            "n_gpu_layers": int(g("n_gpu_layers", -1)),
            "n_batch": int(g("n_batch", 512)),
            "flash_attn": bool(g("flash_attn", False)),
            "kv_cache_type": g("kv_cache_type", "f16"),
            "seed": int(g("seed", 0)),
            "output_format": eff_format,
            "grammar": eff_grammar,
            "extra_load_args": extra,
            "verbose": False,
        }
        if use_vision:
            req["mmproj_path"] = mmproj_resolved
            req["vision_handler"] = _VISION_HANDLER_KEY.get(g("vision_handler", "auto (MTMD)"), "auto")
            req["images"] = images

        load_sig = (resolved, req["n_ctx"], req["n_gpu_layers"], req["n_batch"], req["flash_attn"],
                    req["kv_cache_type"], mmproj_resolved if use_vision else None,
                    _VISION_HANDLER_KEY.get(g("vision_handler", "auto (MTMD)"), "auto") if use_vision else None,
                    json.dumps(extra, sort_keys=True, default=str))

        # Stream the reply text to the node as it generates, over ComfyUI's websocket.
        req["stream_text"] = True
        token_cb = None
        try:
            from server import PromptServer
            nid = str(unique_id[0] if isinstance(unique_id, list) else unique_id)

            def token_cb(delta):
                try:
                    PromptServer.instance.send_sync("kinburg.chatllm", {"id": nid, "delta": delta})
                except Exception:
                    pass
        except Exception:
            token_cb = None

        # Always split reasoning out (strip_think=True) so the chat shows a clean answer + a
        # separate 'thoughts' block, regardless of the config's strip_think.
        out = _generate_and_format(req, load_sig, req["max_tokens"], bool(g("unload_comfy_models", True)),
                                   bool(g("unload_llm_after_run", False)), directive,
                                   True, g("answer_marker", ""), help_text, token_cb=token_cb)
        return _blocked(out[0], out[1])  # (answer, thoughts)


NODE_CLASS_MAPPINGS = {"LocalLLMChatGGUF": LocalLLMChatGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"LocalLLMChatGGUF": "Local LLM Chat (GGUF)"}
