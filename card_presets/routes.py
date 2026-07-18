"""PromptServer routes for Card Presets.

  GET  /kinburg/cards/data  -- saved preset names + their types
  POST /kinburg/cards/save  -- {name, type, values, delete?} add/update/delete a saved card

Guarded so the package still imports without ComfyUI/aiohttp present.
"""
from . import store

try:
    from server import PromptServer
    from aiohttp import web

    routes = PromptServer.instance.routes

    @routes.get("/kinburg/cards/data")
    async def _data(request):
        try:
            return web.json_response({"ok": True, **store.full_data()})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    @routes.post("/kinburg/cards/save")
    async def _save(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            data = store.upsert(
                body.get("name") or "",
                body.get("type") or "character",
                body.get("values") or {},
                delete=bool(body.get("delete")),
            )
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

except Exception as e:  # pragma: no cover
    print(f"[KinburgCards] could not register routes: {e}")
