"""Send Image to Live Log — put a picture into ``Kinburg Live Log 📜`` from anywhere in a graph.

Morpheus and Phantas already post their frames into the log as they decode them, which is what turns
the log into a live storyboard of a run. This node hands that to any branch: wire an ``IMAGE`` in —
it passes straight through, so the node sits inline as a tap — and the picture appears in the log as
its own block, headed by this node's title, the moment execution reaches it.

It is the same channel and the same event the samplers use (``kinburg.llm``, ``event: frames``,
thumbnails from ``util.images.log_uris``), so the run's LLM text, the samplers' frames and whatever
you tapped land in one scrollback in the order they happened. Nothing is written to disk, and there
is no link to the log node — it has no inputs. A canvas with no log on it simply has nobody
listening, which costs a thumbnail encode and nothing else.

**Why not a Preview Image.** A preview shows the picture where the node sits, so a graph with a
dozen of them is a dozen places to look. The log is one place, in run order, with the text the LLM
nodes wrote interleaved — which is what answers "what did this run actually do", as opposed to "what
did this one node output".

**Batches.** ``frames`` decides what goes over the websocket, because a 300-frame video batch is 300
JPEGs down that socket and then held in the log's memory. *all* is capped at ``MAX_FRAMES`` evenly
spaced frames; either way the block header says which frames of how many it is showing, so a thinned
batch never reads as the whole thing.

**Caching.** ``always_run`` is on by default: without it a re-run whose branch is entirely cached
never executes this node, and the log stays empty exactly when you pressed Run to watch it fill.
Turn it off when the ``image`` passthrough feeds something expensive — a node that always re-runs
makes everything downstream of it re-run too.
"""
from ..categories import CAT_LLM
from ..util.images import log_uris

#: Ceiling on 'all', so one node cannot push a whole video down the websocket.
MAX_FRAMES = 32

FRAMES = ["all", "first", "last", "first & last",
          "4 evenly spaced", "8 evenly spaced", "16 evenly spaced"]

_DESCRIPTION = (
    "Push a picture into 'Kinburg Live Log 📜' — the same block-per-event log the pack's LLM nodes "
    "and the Morpheus / Phantas samplers write into, so a tap anywhere in the graph lands in one "
    "scrollback in run order. The image passes through untouched, so the node sits inline; no wire "
    "to the log is needed (it has none). Name the tap with 'label', add a 'note' for the block's "
    "text, and pick which frames of a batch to send.")


def _evenly(n, k):
    """`k` frame indices spread across `n`, first and last included. Fewer if `n` is smaller."""
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def pick_frames(n, mode):
    """Which frames of an `n`-frame batch go to the log, as indices in order.

    An unknown mode falls back to *all*: the value lives in saved workflows, so a mode this node no
    longer offers must still send something rather than nothing.
    """
    if n <= 0:
        return []
    if mode == "first":
        return [0]
    if mode == "last":
        return [n - 1]
    if mode == "first & last":
        return [0] if n == 1 else [0, n - 1]
    head = str(mode).split(" ", 1)[0]
    if head.isdigit():
        return _evenly(n, int(head))
    return _evenly(n, MAX_FRAMES)          # 'all', capped


def _count(image):
    """Frames in a ComfyUI IMAGE. A bare [H,W,C] frame counts as one."""
    shape = getattr(image, "shape", None)
    if shape is not None:
        return 1 if len(shape) == 3 else int(shape[0])
    return len(image)


def _frame(image, i):
    """Frame `i` as a one-frame batch, whatever the input's rank."""
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) == 3:
        return image[None, ...]
    return image[i:i + 1]


def build_payload(image, label, frames, note="", max_side=320):
    """The ``kinburg.llm`` frames event for this picture — pure, so a test can read it.

    The label carries the frame arithmetic ("4 of 120 frames") because the block header is the only
    place anyone will look for it.
    """
    n = _count(image)
    idx = pick_frames(n, frames)
    uris = log_uris([_frame(image, i) for i in idx], int(max_side))

    bits = [b for b in [(label or "").strip()] if b]
    if n > 1:
        bits.append(f"frame {idx[0] + 1} of {n}" if len(idx) == 1
                    else f"{len(idx)} of {n} frames")
    text = (note or "").strip()
    if idx and not uris:
        # log_uris never raises: it returns nothing and logs why. Say that in the block rather than
        # posting an empty one, which would read as "the branch produced no picture".
        text = ("⚠ the picture could not be encoded for the log — see the console"
                + (f"\n{text}" if text else ""))

    payload = {"event": "frames", "images": uris}
    if bits:
        payload["label"] = " · ".join(bits)
    if text:
        payload["text"] = text
    return payload


class KinburgSendImageToLog:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The picture to show in the log. It passes through this node untouched, so it can sit inline on a branch."}),
                "label": ("STRING", {"default": "", "tooltip": "What to call this tap in the block header, after the node's own title — 'before upscale', 'candidate 3'. Leave it empty and the title labels the block on its own, which is enough when the node is named for what it taps."}),
                "frames": (FRAMES, {"default": FRAMES[0], "tooltip": f"Which frames of a batch to send. 'all' is capped at {MAX_FRAMES} evenly spaced frames — a video batch would otherwise put hundreds of JPEGs down the websocket and into the log's memory. The header always says which frames of how many it is showing."}),
                "always_run": ("BOOLEAN", {"default": True, "tooltip": "On: the node re-runs on every Run, so the log fills even when the branch above it is fully cached — which is what a second Run at the same seed is. Off: it caches like any other node; use that when the image passthrough feeds something expensive, because a node that always re-runs makes everything downstream of it re-run too."}),
            },
            "optional": {
                "note": ("STRING", {"default": "", "multiline": True, "tooltip": "Text for the block's body — what you want to read next to the picture later: the seed, which branch this was, what you were testing. It goes nowhere but the log."}),
                "max_side": ("INT", {"default": 320, "min": 64, "max": 1024, "step": 32, "tooltip": "Longest side of the thumbnail sent to the log, in pixels (the log shows it at most 180px tall). Same encoder the samplers' frames use. Raise it to inspect detail, but it is a JPEG down a websocket on every run."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    OUTPUT_NODE = True          # a tap is usually a leaf: it must run with nothing wired downstream
    CATEGORY = CAT_LLM
    DESCRIPTION = _DESCRIPTION

    def run(self, image, label, frames, always_run, note="", max_side=320, unique_id=None):
        payload = build_payload(image, label, frames, note, max_side)
        try:
            from server import PromptServer
            nid = str(unique_id) if unique_id is not None else "sendimagetolog"
            PromptServer.instance.send_sync("kinburg.llm", {"id": nid, **payload})
        except Exception:  # pragma: no cover - a headless run has no server
            pass
        return (image,)

    @classmethod
    def IS_CHANGED(cls, always_run=True, **kwargs):
        # NaN is 'never equal to last time', so ComfyUI re-executes. Off, the constant leaves the
        # node cached on its inputs like any other.
        return float("NaN") if always_run else "cached"


NODE_CLASS_MAPPINGS = {"KinburgSendImageToLog": KinburgSendImageToLog}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSendImageToLog": "Send Image to Live Log 📜"}
