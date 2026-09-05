"""PromptServer routes for Card Presets.

  GET  /kinburg/cards/data    -- saved preset names, their types + tags, the tag list, the
                                 editor's field schema
  GET  /kinburg/cards/preset  -- ?name=X : one card's full values (what the editor opens on)
  POST /kinburg/cards/save    -- {name, type, values, tags?, old_name?, delete?} add/update/
                                 rename/delete a saved card
  POST /kinburg/cards/tags    -- {name, tags} set an existing card's tags (no values needed)
  POST /kinburg/cards/render  -- {type, values} render unsaved values into their Markdown block
                                 (the editor's live preview — never touches the store)

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

    @routes.get("/kinburg/cards/preset")
    async def _preset(request):
        name = request.query.get("name") or ""
        try:
            p = store.get(name)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        if not p:
            return web.json_response({"ok": False, "error": f"no saved card named '{name}'"},
                                     status=404)
        return web.json_response({"ok": True, "name": name, "type": p.get("type", "character"),
                                  "tags": p.get("tags") or [], "values": p.get("values") or {}})

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
                tags=body.get("tags"),
                delete=bool(body.get("delete")),
                old_name=body.get("old_name") or "",
            )
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

    @routes.post("/kinburg/cards/tags")
    async def _tags(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            data = store.retag(body.get("name") or "", body.get("tags"))
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

    @routes.post("/kinburg/cards/render")
    async def _render(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            card = store.render_values(body.get("type") or "character", body.get("values") or {})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, "card": card})

except Exception as e:  # pragma: no cover
    print(f"[KinburgCards] could not register routes: {e}")
