"""PromptServer routes for Prompt Presets.

  GET  /kinburg/grammars/data    -- preset names in dropdown order; per preset whether it is a
                                    built-in, an edited built-in, and which fields it fills
  GET  /kinburg/grammars/preset  -- ?name=X : one preset's prompt and grammar (what the editor opens on)
  POST /kinburg/grammars/preset  -- {name, prompt, grammar, old_name?, delete?} add/update/rename/delete

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

    @routes.get("/kinburg/grammars/preset")
    async def _get(request):
        name = request.query.get("name") or ""
        try:
            p = store.get(name)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        if p is None:
            return web.json_response({"ok": False, "error": f"no preset named '{name}'"}, status=404)
        return web.json_response({"ok": True, "name": name, **p})

    @routes.post("/kinburg/grammars/preset")
    async def _preset(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            data = store.upsert(
                body.get("name") or "",
                body.get("prompt") or "",
                body.get("grammar") or "",
                delete=bool(body.get("delete")),
                old_name=body.get("old_name") or "",
            )
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

except Exception as e:  # pragma: no cover
    print(f"[KinburgGrammars] could not register routes: {e}")
