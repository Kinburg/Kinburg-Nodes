"""Kinburg Live Log — a UI-only node that shows what the pack's LLM nodes are writing, live.

It has no inputs / outputs and never runs on the backend. Every LLM-side node in the pack, when its
``live_preview`` toggle is on, emits websocket events on ``kinburg.llm`` (``Local LLM (GGUF)``,
``Morpheus Storyboard``, ``Morpheus (Video Sampler)``) or ``kinburg.chatllm`` (``Local LLM Chat``);
the display lives entirely in ``web/llm_log.js``, which appends a block per generation and grows its
text token by token. Drop it anywhere on the canvas — it needs no connections, and shows every
source at once, each block labelled by its node plus, when the source says so, the individual call
("Morpheus Storyboard · shot 2/4 (2 keyframes)").

Blocks can also carry **images**: the frames a vision call was actually shown, and a ``frames``
event on its own — Morpheus sends each shot's last frame right after decoding it, which turns the
log into a live storyboard of the run and answers "what did this shot actually get?" by eye instead
of by digging through the cache. Thumbnails have a hover copy-to-clipboard button.

Each block's header counts the generated tokens against the run's ``max_tokens`` ceiling and the
generation rate; on finish it adds the context fill (prompt + gen vs ``n_ctx``). Reasoning
(``<think>`` blocks, or everything before an ``answer_marker`` line) renders in a separate
collapsible section. The 'start' / 'done' payloads carry the figures and the marker — see
``LocalLLMGGUF.run``.

``KinburgLLMLog`` is the previous, LLM-only id. It stays registered (and the same renderer drives
it, so it gains the images too) purely so workflows saved against it keep working; it is hidden
from the node picker.
"""

_SHARED_DESCRIPTION = (
    "Live view of every LLM node in the pack as it generates: Local LLM (GGUF), Chat, Morpheus "
    "Storyboard and the Morpheus sampler's in-loop writer. Turn on the source node's "
    "'live_preview' toggle, drop this node anywhere (no wiring needed), and watch the text stream "
    "in with a tokens-used / max_tokens counter, budget bar and tok/s per block (plus the context "
    "fill when it finishes). Vision calls show the frames they were given, and Morpheus posts each "
    "shot's last frame as it is decoded, so the log doubles as a live storyboard — thumbnails have "
    "a copy-to-clipboard button. Reasoning folds into its own dim, collapsible section. Scroll up "
    "and the view stays put while it keeps writing — the '↓ latest' pill jumps back.")


class KinburgLiveLog:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "noop"
    CATEGORY = "Kinburg-Nodes/LLM"
    OUTPUT_NODE = False
    DESCRIPTION = _SHARED_DESCRIPTION

    def noop(self):
        return ()


class LLMLiveLog(KinburgLiveLog):
    """The old id, kept loadable and out of the picker. Delete once no workflow references it."""
    DEPRECATED = True
    DESCRIPTION = "Superseded by 'Kinburg Live Log' — identical behaviour, kept so old graphs load."


NODE_CLASS_MAPPINGS = {"KinburgLiveLog": KinburgLiveLog, "KinburgLLMLog": LLMLiveLog}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgLiveLog": "Kinburg Live Log 📜",
                              "KinburgLLMLog": "LLM Live Log (old id)"}
