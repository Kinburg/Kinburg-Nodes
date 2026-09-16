"""The backlog route behind the LLM Server Live Log node.

The pack's other live logs only ever show what arrives while they are watching, which is right for
them: they log a ComfyUI run, and if you were not there, there was no run. This one logs a server
that is talked to whenever SillyTavern feels like it — quite possibly for an hour with no browser
open at all. So the events are kept on the backend (see :mod:`events`) and a log node asks for the
backlog when it appears: open ComfyUI after a long chat and the log is already full.

`after` is the last seq the caller has, so a reconnecting node asks only for what it missed.
Guarded so the package still imports without ComfyUI/aiohttp present.
"""
from . import events, gateway

try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/kinburg/llm_server/log")
    async def _log(request):
        try:
            after = int(request.rel_url.query.get("after", 0))
        except (TypeError, ValueError):
            after = 0
        return web.json_response({
            "events": events.recent(after),
            "seq": events.last_seq(),
            "status": gateway.status(),
        })

    @PromptServer.instance.routes.post("/kinburg/llm_server/log/clear")
    async def _clear(request):
        # The ring, not just this node's view — otherwise a reload replays everything the user
        # just cleared away.
        return web.json_response({"ok": True, "seq": events.clear()})

except Exception as e:  # pragma: no cover
    print(f"[LLM gateway] could not register the log routes: {e}")
