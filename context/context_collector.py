"""Context Collector — gather several text chunks (character cards, entity cards, world notes,
…) and frame them under one heading the LLM can latch onto.

The input list grows on demand (web/context.js): connect the last `item_N` slot and a new one
appears. Empty / unconnected slots are skipped. The framed `context` output goes into an LLM
node's `context` input. Frame it as a Markdown heading — pick its level ('#' vs '##' …) to set
the block's place in the context hierarchy — or as 'none' (title in bold, no heading markup).
Category `Kinburg-Nodes/LLM`.
"""
import re

_IDX_RE = re.compile(r"^item_(\d+)$")

_WRAP_MD = "markdown heading"
_WRAP_NONE = "none"
_WRAPPERS = [_WRAP_MD, _WRAP_NONE]

# Markdown heading markers (fewer '#' = higher level). Sets how deep the block title sits in
# the context hierarchy when wrapper = markdown heading.
_LEVELS = ["#", "##", "###", "####", "#####", "######"]


class ContextCollector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "item_1": ("STRING", {"forceInput": True}),
                "title": ("STRING", {"default": "Characters", "tooltip": "Heading placed at the top of the block (e.g. 'Characters', 'Reference'). Empty = no heading."}),
                "wrapper": (_WRAPPERS, {"default": _WRAP_MD, "tooltip": "How to frame the block: a Markdown heading (pick its level below) or none (title in bold, no heading markup)."}),
                "heading_level": (_LEVELS, {"default": "##", "tooltip": "Markdown heading level for the title (wrapper = markdown heading). '#' = H1 (top of the hierarchy), '##' = H2, … Ignored when wrapper = none."}),
                "separator": ("STRING", {"multiline": True, "default": "\n\n", "tooltip": "Inserted between the collected items. Default is a blank line."}),
            },
            "optional": {
                "item_2": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("context",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/LLM"

    @staticmethod
    def _index(key):
        m = _IDX_RE.match(key)
        return int(m.group(1)) if m else 1 << 30

    def run(self, title="Characters", wrapper=_WRAP_MD, heading_level="##", separator="\n\n", **kwargs):
        items = []
        for key in sorted((k for k in kwargs if _IDX_RE.match(k)), key=self._index):
            v = kwargs.get(key)
            if v is None:
                continue
            if not isinstance(v, str):
                v = str(v)
            if v.strip():
                items.append(v.strip())

        joined = separator.join(items)
        title = (title or "").strip()

        if wrapper == _WRAP_MD:
            level = heading_level if heading_level in _LEVELS else "##"
            parts = ([f"{level} {title}"] if title else []) + ([joined] if joined else [])
            return ("\n\n".join(parts),)

        # none: title in bold, items below, no heading markup.
        parts = ([f"**{title}**"] if title else []) + ([joined] if joined else [])
        return ("\n\n".join(parts),)


NODE_CLASS_MAPPINGS = {"ContextCollector": ContextCollector}
NODE_DISPLAY_NAME_MAPPINGS = {"ContextCollector": "Context Collector"}
