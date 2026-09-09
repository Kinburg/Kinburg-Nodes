"""Echo — the nymph who could only repeat the words others had already spoken.

Which is exactly the job. Siren writes the song and sings it; this suite says **when each word came
out**, so the lyric can be put back on the screen in time with the voice. It never decides what was
sung — the words are given, and inventing them is the one thing a forced alignment cannot do.

The split inside the package is the point of it:

* `translit.py` — the 28 characters the model can hear, and how a Cyrillic lyric is written in them.
* `track.py` — what is aligned, in what window. The windowing is what keeps the memory cost tied to
  the longest SECTION rather than to the length of the song.
* `subs.py` — the `.ass` and `.srt` files, karaoke tags and one colour per singer.
* `align.py` — the only file that needs weights, and it is the shortest of the four.

Three of those four are pure stdlib and are tested without a GPU, without audio and without the
1.2 GB of weights, which is why the arithmetic can be trusted before anything is rendered.
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F401

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
