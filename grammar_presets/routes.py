"""PromptServer routes for Grammar Presets.

  GET  /kinburg/grammars/data    -- built-in + user grammars, and the built-in name list
  POST /kinburg/grammars/preset  -- {name, text, delete?} add/update/delete a user grammar

Guarded so the package still imports without ComfyUI/aiohttp present.
"""
from . import store

try:
    from server import PromptServer
    from aiohttp import web

    routes = PromptServer.instance.routes

    @routes.get("/kinburg/grammars/data")
    async def _data(request):
        try:
            return web.json_response({"ok": True, **store.full_data()})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    @routes.post("/kinburg/grammars/preset")
    async def _preset(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            data = store.upsert_grammar(
                body.get("name") or "",
                body.get("text") or "",
                delete=bool(body.get("delete")),
            )
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

except Exception as e:  # pragma: no cover
    print(f"[KinburgGrammars] could not register routes: {e}")
