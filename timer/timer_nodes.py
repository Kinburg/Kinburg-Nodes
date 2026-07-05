"""Start Timer / Stop Timer — measure the wall-clock time of a slice of a workflow.

Insert `Start Timer` at the beginning of the slice and `Stop Timer` at the end: any value
(MODEL, LATENT, IMAGE, …) flows through their `passthrough` slot, and that data dependency
is what forces ComfyUI to run Start → slice → Stop in that order. The start timestamp
travels from Start to Stop as a plain number (`start_seconds`), so Stop just computes
`now - start`.

Caching note: both nodes return a changing IS_CHANGED (NaN) so they always re-execute —
otherwise Start would hand back a stale timestamp and the slice could be skipped entirely.
The side effect is that the wrapped slice is recomputed on every run (no caching) while the
timers are active; mute/bypass them when you're not measuring.
"""
import time
from datetime import datetime


from ..util.anytype import ANY


def _hms(sec):
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_elapsed(sec, fmt):
    if fmt == "seconds":
        return f"{sec:.3f} s"
    if fmt == "milliseconds":
        return f"{sec * 1000:.0f} ms"
    if fmt == "HH:MM:SS":
        return _hms(sec)
    if fmt == "human":
        if sec < 1:
            return f"{sec * 1000:.0f}ms"
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        if h:
            return f"{h}h {m}m {int(round(s))}s"
        if m:
            return f"{m}m {int(round(s))}s"
        return f"{s:.1f}s"
    # auto
    if sec < 1:
        return f"{sec * 1000:.0f} ms"
    if sec < 60:
        return f"{sec:.2f} s"
    if sec < 3600:
        m, s = divmod(int(round(sec)), 60)
        return f"{m:02d}:{s:02d}"
    return _hms(sec)


class StartTimer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_1": (ANY, {"tooltip": "Main line — passes through unchanged. The timer starts only after ALL connected inputs are ready, so tap every branch feeding your sampler (noise / guider / sampler / sigmas / latent) to start right before it runs, not as soon as one branch is ready."}),
                "time_format": ("STRING", {"default": "%Y-%m-%d %H:%M:%S", "tooltip": "strftime pattern for the 'start_time' string output."}),
            },
            "optional": {
                "any_2": (ANY, {"tooltip": "Extra dependency taps — more slots appear as you connect. They only gate WHEN the timer starts; just any_1 is passed through."}),
            },
        }

    RETURN_TYPES = (ANY, "FLOAT", "STRING")
    RETURN_NAMES = ("passthrough", "start_seconds", "start_time")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/util"

    def run(self, time_format="%Y-%m-%d %H:%M:%S", **kwargs):
        now = time.time()
        try:
            stamp = datetime.fromtimestamp(now).strftime(time_format or "%Y-%m-%d %H:%M:%S")
        except Exception:
            stamp = datetime.fromtimestamp(now).isoformat(sep=" ", timespec="seconds")
        return (kwargs.get("any_1"), now, stamp)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


class StopTimer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": (ANY, {"tooltip": "Any value — it passes through unchanged. Wiring it from the end of the slice is what stops the timer here."}),
                "start_seconds": ("FLOAT", {"default": 0.0, "forceInput": True, "tooltip": "Connect 'start_seconds' from the Start Timer node."}),
                "format": (["auto", "seconds", "milliseconds", "HH:MM:SS", "human"], {"default": "auto", "tooltip": "How to format the elapsed time string."}),
            }
        }

    RETURN_TYPES = (ANY, "STRING", "FLOAT")
    RETURN_NAMES = ("passthrough", "elapsed", "seconds")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Kinburg-Nodes/util"

    def run(self, passthrough, start_seconds=0.0, format="auto"):
        elapsed = max(0.0, time.time() - float(start_seconds))
        return (passthrough, _format_elapsed(elapsed, format), elapsed)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


NODE_CLASS_MAPPINGS = {"StartTimer": StartTimer, "StopTimer": StopTimer}
NODE_DISPLAY_NAME_MAPPINGS = {"StartTimer": "Start Timer", "StopTimer": "Stop Timer"}
