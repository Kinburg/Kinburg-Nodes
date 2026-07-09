"""PromptServer route for Show Text (Markdown): write the node's text to a ``.md`` file.

  POST /kinburg/showtext/save  -- {path, text} -> writes <path>.md, returns the final path

Guarded so the package still imports without ComfyUI/aiohttp present.
"""
from .text_node import save_markdown

try:
    from server import PromptServer
    from aiohttp import web

    routes = PromptServer.instance.routes

    @routes.post("/kinburg/showtext/save")
    async def _save(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        path = (body.get("path") or "").strip()
        if not path:
            return web.json_response({"ok": False, "error": "save_path is empty"}, status=400)
        try:
            saved = save_markdown(path, body.get("text") or "")
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, "path": saved})

except Exception as e:  # pragma: no cover
    print(f"[KinburgShowText] could not register routes: {e}")
