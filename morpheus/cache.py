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
"""
import hashlib
import json
import logging
import os

import torch

import comfy.utils
import folder_paths

LIMIT_BYTES = 8 * 1024 ** 3  # pruned oldest-first after each run
_SEP = "\x1f"


def cache_dir():
    d = os.path.join(folder_paths.get_user_directory(), "kinburg-nodes", "morpheus_dreams")
    os.makedirs(d, exist_ok=True)
    return d


def key(*parts):
    h = hashlib.sha1()
    h.update(_SEP.join("" if p is None else str(p) for p in parts).encode("utf-8", "replace"))
    return h.hexdigest()


def tensor_key(t):
    """Content hash of a user-supplied image (the only pixels we ever hash)."""
    if t is None:
        return "none"
    try:
        a = t.detach().cpu().contiguous()
        return hashlib.sha1(a.numpy().tobytes()).hexdigest()[:16] + f":{tuple(a.shape)}"
    except Exception:
        return "unhashable"


def fingerprint(*objs):
    """Cheap identity for a loaded model / clip / vae: architecture + parameter count + patch keys.

    Deliberately NOT a weight hash (too slow on a 30B text encoder). Two different checkpoints of
    the same architecture and size would collide — that is what the sampler's `cache_tag` widget is
    for. LoRA and other patches DO enter the key, because they change what gets sampled.
    """
    parts = []
    for obj in objs:
        if obj is None:
            parts.append("none")
            continue
        parts.append(type(obj).__name__)
        try:
            parts.append(str(sum(p.numel() for p in obj.parameters())))
        except Exception:
            pass
        try:
            patches = getattr(obj, "patches", None)
            if isinstance(patches, dict) and patches:
                parts.append("|".join(sorted(patches.keys())))
        except Exception:
            pass
    return key(*parts)


def _path(k):
    return os.path.join(cache_dir(), f"{k[:24]}.safetensors")


def load(k):
    """The cached tensors for a shot key, or None. Keys: video / audio / handoff."""
    p = _path(k)
    if not os.path.isfile(p):
        return None
    try:
        sd = comfy.utils.load_torch_file(p, safe_load=True)
    except Exception as e:
        logging.warning(f"[Morpheus] unreadable shot cache {os.path.basename(p)}: {e}")
        try:
            os.remove(p)
        except OSError:
            pass
        return None
    if "video" not in sd:
        return None
    try:  # LRU-ish: touch on hit so pruning drops what nobody re-uses
        os.utime(p, None)
    except OSError:
        pass
    return sd


def save(k, video, audio=None, handoff=None, meta=None):
    """Write one shot. Returns None on success or the error string, so the caller can put a failed
    write in its report instead of leaving it in the log where nobody looks.

    `.contiguous()` matters: sampled latents arrive as views into the packed tensor and decoded
    frames as a permuted view, `.to()` keeps those strides, and safetensors rejects a
    non-contiguous tensor outright ("You are trying to save a non contiguous tensor")."""
    def prep(t):
        return t.detach().to("cpu", torch.float16).contiguous()

    sd = {"video": prep(video)}
    if audio is not None:
        sd["audio"] = prep(audio)
    if handoff is not None:
        sd["handoff"] = prep(handoff)
    metadata = {str(a): str(b) for a, b in (meta or {}).items()}
    try:
        comfy.utils.save_torch_file(sd, _path(k), metadata=metadata or None)
        return None
    except Exception as e:
        logging.warning(f"[Morpheus] could not write shot cache: {e}")
        return str(e)


def load_json(k):
    """A cached LLM answer, or None. Text, not tensors — the storyboard node's half of the cache."""
    p = _path(k)[:-len(".safetensors")] + ".json"
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        os.utime(p, None)
        return data
    except Exception as e:
        logging.warning(f"[Morpheus] unreadable prompt cache {os.path.basename(p)}: {e}")
        return None


def save_json(k, obj):
    p = _path(k)[:-len(".safetensors")] + ".json"
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        return None
    except Exception as e:
        logging.warning(f"[Morpheus] could not write prompt cache: {e}")
        return str(e)


def prune(limit_bytes=LIMIT_BYTES):
    """Keep the cache under `limit_bytes`, dropping least-recently-used files first."""
    try:
        files = []
        for name in os.listdir(cache_dir()):
            if not name.endswith((".safetensors", ".json")):
                continue
            p = os.path.join(cache_dir(), name)
            st = os.stat(p)
            files.append((st.st_mtime, st.st_size, p))
    except OSError:
        return 0
    total = sum(f[1] for f in files)
    freed = 0
    for _, size, p in sorted(files):
        if total - freed <= limit_bytes:
            break
        try:
            os.remove(p)
            freed += size
        except OSError:
            pass
    return freed
