"""Shared filename helpers.

`first_free` finds the smallest 1-based counter for which none of a set of extensions
collide in a directory — the pattern used when writing indexed output files
(`compare_0001.html`, `compare_01_0001.png` + `.txt`, …). Keeping it here removes the
duplicated `while os.path.exists(...)` loops.

Note: nodes that save into ComfyUI's own output tree should prefer
`folder_paths.get_save_image_path` (as Save Song does); this helper is for arbitrary
output folders (e.g. Image Compare's custom `output_dir`) where that isn't a fit.
"""
import os


def first_free(directory, base_fn, exts, start=1):
    """Smallest counter i >= start for which none of ``base_fn(i) + ext`` exist in
    ``directory``. Returns ``(i, base_fn(i))`` — the base name carries no extension.

    ``base_fn`` maps a counter to a base name (e.g. ``lambda k: f"compare_{k:04d}"``);
    ``exts`` is the list of extensions to check, each WITH its dot (``[".html"]`` or
    ``[".png", ".txt"]``).
    """
    i = start
    while True:
        base = base_fn(i)
        if not any(os.path.exists(os.path.join(directory, base + e)) for e in exts):
            return i, base
        i += 1
