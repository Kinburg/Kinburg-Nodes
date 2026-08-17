"""Disk cache for Morpheus dreams (MiniMax H3 shots) — latents, never pixels.

A 5-second shot at 1344x768 is ~1.5 GB of float32 pixels but only ~7 MB of fp16 latent, so what
lands on disk is the sampled latent pair (video + audio) plus the ONE decoded frame the next shot
needs as its keyframe. Decoding is cheap next to sampling; re-sampling is what we refuse to repeat.

Keys are **causal**: a shot's key folds in the previous shot's key, because a continuing shot's
first frame *is* the previous shot's last frame. Editing shot 5 of 8 therefore invalidates 5..8 and
lets 1..4 come straight off the disk.

Handoff *pixels* are deliberately never hashed. The fresh path and the cached path have to agree
bit-for-bit or every downstream shot would miss forever, so the sampler rounds every handoff frame
through fp16 (which is also how it is stored) instead of hoping two float32 VAE decodes match.

The machinery — keys, identities, the folder itself — lives in `util/diskcache.py` and is shared
with the other suites. What stays here is what is specific to a shot: where the folder is, and the
rule that an entry without a `video` tensor is not a hit.
"""
from ..util.diskcache import DEFAULT_LIMIT, Store, fingerprint, key, tensor_key  # noqa: F401

LIMIT_BYTES = DEFAULT_LIMIT  # pruned oldest-first after each run

_store = Store("morpheus_dreams", "Morpheus", LIMIT_BYTES)


def cache_dir():
    return _store.dir()


def load(k):
    """The cached tensors for a shot key, or None. Keys: video / audio / handoff."""
    return _store.load(k, require="video")


def save(k, video, audio=None, handoff=None, meta=None):
    """Write one shot. Returns None on success or the error string, so the caller can put a failed
    write in its report instead of leaving it in the log where nobody looks."""
    return _store.save(k, {"video": video, "audio": audio, "handoff": handoff}, meta)


def load_json(k):
    """A cached LLM answer, or None. Text, not tensors — the storyboard node's half of the cache."""
    return _store.load_json(k)


def save_json(k, obj):
    return _store.save_json(k, obj)


def prune(limit_bytes=LIMIT_BYTES):
    """Keep the cache under `limit_bytes`, dropping least-recently-used files first."""
    return _store.prune(limit_bytes)
