"""Disk cache primitives: causal keys, cheap identities, and a small tensor/JSON store.

Lifted out of the Morpheus shot cache when Phantas became the second thing in the pack that must
not redo expensive work. What is generic lives here; what a suite *stores*, and what makes an entry
usable, stays with the suite (`morpheus/cache.py`, `phantas/cache.py`) — "is this hit still valid"
is a question only the caller can answer.

Keys are **causal** by convention rather than by machinery: a step folds the previous step's key
into its own, so editing step 5 of 8 invalidates 5..8 and lets 1..4 come straight off the disk.
`key()` only joins and hashes; the causality is in what the caller passes it.

Nothing here imports torch, comfy or folder_paths at module level. `key` / `tensor_key` /
`fingerprint` are pure enough to run in a headless test, and a `Store` pulls the heavy imports only
when it actually touches the disk.
"""
import hashlib
import json
import logging
import os

DEFAULT_LIMIT = 8 * 1024 ** 3  # pruned oldest-first after each run
_SEP = "\x1f"


def key(*parts):
    """One cache key from any number of parts. `None` and `""` are the same part on purpose: an
    unset option and an empty one must not split the cache."""
    h = hashlib.sha1()
    h.update(_SEP.join("" if p is None else str(p) for p in parts).encode("utf-8", "replace"))
    return h.hexdigest()


def tensor_key(t):
    """Content hash of a user-supplied image — the only pixels the pack ever hashes."""
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
    the same architecture and size would collide — that is what a sampler's `cache_tag` widget is
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


class Store:
    """One folder of cache entries under ComfyUI's user directory.

    `subdir` names the folder and `tag` prefixes the warning lines — a cache file that turns out to
    be unreadable should say which suite wrote it. One instance per suite, at module level.
    """

    def __init__(self, subdir, tag, limit_bytes=DEFAULT_LIMIT):
        self.subdir = subdir
        self.tag = tag
        self.limit_bytes = limit_bytes

    def dir(self):
        import folder_paths
        d = os.path.join(folder_paths.get_user_directory(), "kinburg-nodes", self.subdir)
        os.makedirs(d, exist_ok=True)
        return d

    def path(self, k, ext=".safetensors"):
        return os.path.join(self.dir(), f"{k[:24]}{ext}")

    # ------------------------------------------------------------------------------------ tensors
    def load(self, k, require=None):
        """The tensors stored under `k`, or None — including when the entry exists but does not
        hold `require`, which is how a caller rejects something an older version wrote."""
        p = self.path(k)
        if not os.path.isfile(p):
            return None
        import comfy.utils
        try:
            sd = comfy.utils.load_torch_file(p, safe_load=True)
        except Exception as e:
            logging.warning(f"[{self.tag}] unreadable cache entry {os.path.basename(p)}: {e}")
            try:
                os.remove(p)
            except OSError:
                pass
            return None
        if require is not None and require not in sd:
            return None
        try:  # LRU-ish: touch on hit so pruning drops what nobody re-uses
            os.utime(p, None)
        except OSError:
            pass
        return sd

    def save(self, k, tensors, meta=None, dtype=None):
        """Write a `{name: tensor}` entry. Returns None on success or the error string, so a failed
        write can land in the caller's report instead of only in the log nobody reads.

        Stored fp16 and `.contiguous()`, both deliberately: sampled latents arrive as views into a
        packed tensor and decoded frames as a permuted view, `.to()` preserves those strides, and
        safetensors refuses a non-contiguous tensor outright.
        """
        import torch
        dt = torch.float16 if dtype is None else dtype
        sd = {n: t.detach().to("cpu", dt).contiguous()
              for n, t in (tensors or {}).items() if t is not None}
        if not sd:
            return "nothing to write"
        metadata = {str(a): str(b) for a, b in (meta or {}).items()}
        import comfy.utils
        try:
            comfy.utils.save_torch_file(sd, self.path(k), metadata=metadata or None)
            return None
        except Exception as e:
            logging.warning(f"[{self.tag}] could not write cache entry: {e}")
            return str(e)

    # --------------------------------------------------------------------------------------- json
    def load_json(self, k):
        """A cached answer as text rather than tensors — the writer nodes' half of the cache."""
        p = self.path(k, ".json")
        if not os.path.isfile(p):
            return None
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            os.utime(p, None)
            return data
        except Exception as e:
            logging.warning(f"[{self.tag}] unreadable text cache {os.path.basename(p)}: {e}")
            return None

    def save_json(self, k, obj):
        p = self.path(k, ".json")
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=1)
            return None
        except Exception as e:
            logging.warning(f"[{self.tag}] could not write text cache: {e}")
            return str(e)

    # -------------------------------------------------------------------------------------- prune
    def prune(self, limit_bytes=None):
        """Keep the folder under its limit, dropping least-recently-used files first."""
        limit = self.limit_bytes if limit_bytes is None else limit_bytes
        try:
            d = self.dir()
            files = []
            for name in os.listdir(d):
                if not name.endswith((".safetensors", ".json")):
                    continue
                p = os.path.join(d, name)
                st = os.stat(p)
                files.append((st.st_mtime, st.st_size, p))
        except OSError:
            return 0
        total = sum(f[1] for f in files)
        freed = 0
        for _, size, p in sorted(files):
            if total - freed <= limit:
                break
            try:
                os.remove(p)
                freed += size
            except OSError:
                pass
        return freed
