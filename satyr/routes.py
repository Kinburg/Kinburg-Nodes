"""The two routes behind Satyr Edit's editor.

  POST /kinburg/satyr/edit/load   -- {abc}          → {ok, model}: a plan as the editor's notes
  POST /kinburg/satyr/edit/save   -- {base, model}  → {ok, abc, report, model}: the edit as a plan
  POST /kinburg/satyr/edit/words  -- {base, model, lyrics, voices} → {ok, sections, lines, singers,
                                     findings}: the lyrics laid onto the edit as Satyr Score lays them

`save` hands the model back re-read from the text it wrote, so the editor carries on from the plan
as it now stands — its bars then point at the new text's groups, and the next save keeps whatever
this one wrote as untouched. Guarded so the package still imports without ComfyUI/aiohttp present,
or without the core exporter `edit` writes through.
"""
try:
    from aiohttp import web
    from server import PromptServer

    from . import edit as ED

    routes = PromptServer.instance.routes

    @routes.post("/kinburg/satyr/edit/load")
    async def _load(request):
        body = await request.json()
        try:
            return web.json_response({"ok": True, "model": ED.load(str(body.get("abc") or ""))})
        except (ValueError, KeyError, TypeError) as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)

    @routes.post("/kinburg/satyr/edit/save")
    async def _save(request):
        body = await request.json()
        try:
            text, report = ED.save(str(body.get("base") or ""), body.get("model") or {})
        except (ValueError, KeyError, TypeError) as e:
            print(f"[Satyr Edit] not saved: {e}")
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        print("[Satyr Edit] saved: " + " · ".join(report))
        return web.json_response({"ok": True, "abc": text, "report": report, "model": ED.load(text)})

    @routes.post("/kinburg/satyr/edit/words")
    async def _words(request):
        body = await request.json()
        try:
            got = ED.words(str(body.get("base") or ""), body.get("model") or {}, str(body.get("lyrics") or ""),
                           body.get("voices") or [])
        except (ValueError, KeyError, TypeError) as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        return web.json_response({"ok": True, **got})

except Exception as e:  # pragma: no cover
    print(f"[Satyr Edit] could not register the editor routes: {e}")
