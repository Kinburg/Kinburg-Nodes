"""LLM Live Log — a UI-only node that shows a Local LLM (GGUF) node's output as it streams.

It has no inputs / outputs and never runs on the backend. The Local LLM (GGUF) node, when its
``live_preview`` toggle is on, emits ``kinburg.llm`` websocket events (start / delta / done); the
display lives entirely in ``web/llm_log.js``, which appends a block per generation and grows its
text token by token. Drop it anywhere on the canvas — it needs no connections, and shows the
stream from every Local LLM (GGUF) node (labelled by the source node). Category
``Kinburg-Nodes/LLM``. (Grammar/JSON runs don't stream — their result shows up on 'done'.)

Each block's header counts the generated tokens against the run's ``max_tokens`` ceiling and the
generation rate; on finish it adds the context fill (prompt + gen vs ``n_ctx``). Reasoning
(``<think>`` blocks, or everything before an ``answer_marker`` line) renders in a separate
collapsible section. The 'start' / 'done' payloads carry the figures and the marker — see
``LocalLLMGGUF.run``.
"""


class LLMLiveLog:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "noop"
    CATEGORY = "Kinburg-Nodes/LLM"
    OUTPUT_NODE = False
    DESCRIPTION = ("Live view of Local LLM (GGUF) output as it generates. Turn on the LLM node's "
                   "'live_preview' toggle, drop this node anywhere (no wiring needed), and watch "
                   "the text stream in, with a live tokens-used / max_tokens counter, budget bar "
                   "and tok/s per block (plus the context fill when it finishes). Reasoning is "
                   "folded into its own dim, collapsible section. Scroll up and the view stays "
                   "put while it keeps writing — the '↓ latest' pill jumps back. Text runs stream "
                   "token by token; a grammar/JSON run shows its result when it finishes.")

    def noop(self):
        return ()


NODE_CLASS_MAPPINGS = {"KinburgLLMLog": LLMLiveLog}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgLLMLog": "LLM Live Log"}
