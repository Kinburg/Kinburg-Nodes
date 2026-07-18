"""PromptServer routes for Prompt Presets — read the merged store, add/delete user
presets, save/delete setups.

  GET  /kinburg/presets/data     -- merged presets + built-in name lists + saved setups
  POST /kinburg/presets/preset   -- {category, name, text, delete?} add/update/delete a preset
  POST /kinburg/presets/category -- {action: add|rename|delete, name, new_name?} manage categories
  POST /kinburg/presets/setup    -- {name, slots, delete?} save/delete a named setup

Guarded so the package still imports without ComfyUI/aiohttp present.
"""
from . import store

try:
    from server import PromptServer
    from aiohttp import web

    routes = PromptServer.instance.routes

    @routes.get("/kinburg/presets/data")
    async def _data(request):
        try:
            return web.json_response({"ok": True, **store.full_data()})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    @routes.post("/kinburg/presets/preset")
    async def _preset(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            data = store.upsert_preset(
                (body.get("category") or "").strip(),
                body.get("name") or "",
                body.get("text") or "",
                delete=bool(body.get("delete")),
            )
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

    @routes.post("/kinburg/presets/category")
    async def _category(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        action = (body.get("action") or "").strip()
        name = body.get("name") or ""
        try:
            if action == "add":
                data = store.add_category(name)
            elif action == "rename":
                data = store.rename_category(name, body.get("new_name") or "")
            elif action == "delete":
                data = store.delete_category(name)
            else:
                raise ValueError(f"unknown action: {action}")
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

    @routes.post("/kinburg/presets/setup")
    async def _setup(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            data = store.upsert_setup(
                body.get("name") or "",
                body.get("slots") or [],
                delete=bool(body.get("delete")),
            )
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **data})

except Exception as e:  # pragma: no cover
    print(f"[KinburgPresets] could not register routes: {e}")
