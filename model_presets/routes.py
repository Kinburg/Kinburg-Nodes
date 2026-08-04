"""PromptServer routes for the Model Library.

  GET  /kinburg/models/data     -- models (label / families / slots / preset summaries) + shared pool
  POST /kinburg/models/model    -- {id, label?, families?, tags?, notes?, rename?, delete?}
  POST /kinburg/models/preset   -- {model, name, shared?, tags?, families?, notes?, set_default?,
                                    delete?}  (metadata only — stages come from Settings Save)
  GET  /kinburg/models/recipe   -- ?id=<model_id>  the full recipe, for the Manage dialog's detail view

Guarded so the package still imports without ComfyUI/aiohttp present (registry scan, tests).
"""
from . import capture_node, store

try:
    from server import PromptServer
    from aiohttp import web

    routes = PromptServer.instance.routes

    @routes.get("/kinburg/models/data")
    async def _data(request):
        try:
            return web.json_response({"ok": True, **store.full_data()})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    @routes.get("/kinburg/models/recipe")
    async def _recipe(request):
        try:
            mid = request.query.get("id") or ""
            model = store.get_model(mid)
            if not model:
                return web.json_response({"ok": False, "error": f"no model '{mid}'"}, status=404)
            recipe = model.get("recipe") or {}
            return web.json_response({
                "ok": True, "id": mid, "recipe": recipe,
                "summary": capture_node.describe(recipe.get("nodes") or {},
                                                 recipe.get("outputs") or {}),
                # Per-node literal inputs, so the dialog can render an editor without having to
                # work out for itself which inputs are wiring and which are settings.
                "editable": store.editable_fields(mid),
            })
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    @routes.post("/kinburg/models/recipe")
    async def _recipe_save(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            data, applied, rejected = store.update_recipe(body.get("id") or "",
                                                          body.get("values") or {})
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, "applied": applied, "rejected": rejected, **data})

    @routes.post("/kinburg/models/overrides")
    async def _overrides(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            data = store.set_overrides(body.get("model") or "", body.get("name") or "",
                                       body.get("overrides") or {},
                                       shared=bool(body.get("shared")))
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

    @routes.post("/kinburg/models/model")
    async def _model(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            mid = body.get("id") or ""
            if body.get("delete"):
                data = store.delete_model(mid)
            elif body.get("rename"):
                data = store.rename_model(mid, body["rename"])
            else:
                data = store.upsert_model(mid, families=body.get("families"),
                                          tags=body.get("tags"), notes=body.get("notes"))
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

    @routes.post("/kinburg/models/preset")
    async def _preset(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            if body.get("delete"):
                data = store.upsert_preset(body.get("model") or "", body.get("name") or "", None,
                                           shared=bool(body.get("shared")), delete=True)
            else:
                data = store.retag_preset(body.get("model") or "", body.get("name") or "",
                                          tags=body.get("tags"), families=body.get("families"),
                                          notes=body.get("notes"),
                                          set_default=body.get("set_default"),
                                          shared=bool(body.get("shared")))
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

except Exception as e:  # pragma: no cover
    print(f"[Kinburg ModelLibrary] could not register routes: {e}")
