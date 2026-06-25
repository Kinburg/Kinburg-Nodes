"""Shared block separator for the Image Compare pipeline.

A line equal to BLOCK_SEP separates one per-image block (a prompt or a settings block)
from the next. The producers — Generation Info Filter and Get Accumulator (prompts) — join
their blocks with BLOCK_JOINER, and Image Compare (HTML) splits the incoming text back on
BLOCK_SEP. They all import from here so the contract can't silently drift.
"""

BLOCK_SEP = "---"

# Join string that places BLOCK_SEP on its own line between (possibly multi-line) blocks.
BLOCK_JOINER = "\n" + BLOCK_SEP + "\n"
