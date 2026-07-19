"""Stop-flag + PromptServer route for the Ouroboros loop.

A frontend "⏹ Stop loop" button POSTs the node id here; `run()` polls the flag between iterations
and stops gracefully — returning everything generated so far (unlike ComfyUI's Cancel, which
raises mid-sampling and discards the run). Guarded so the package still imports without
ComfyUI/aiohttp present.
"""
import threading

_LOCK = threading.Lock()
_STOP = set()  # node ids (str) with a pending stop request


def request_stop(node_id):
    with _LOCK:
        _STOP.add(str(node_id))


def clear_stop(node_id):
    with _LOCK:
        _STOP.discard(str(node_id))


def stop_requested(node_id):
    if node_id is None:
        return False
    with _LOCK:
        return str(node_id) in _STOP


try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/kinburg/ouroboros/stop")
    async def _stop(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        nid = body.get("id")
        if nid is None:
            return web.json_response({"ok": False, "error": "missing id"}, status=400)
        request_stop(nid)
        return web.json_response({"ok": True})

except Exception as e:  # pragma: no cover
    print(f"[Ouroboros] could not register stop route: {e}")
