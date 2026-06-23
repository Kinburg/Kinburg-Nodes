"""PromptServer routes for the investigate report.

  POST /kinburg/report/save    -- the comparison page saves a run here (only writer)
  GET  /kinburg/report         -- the report browser page (report.html)
  GET  /kinburg/report/data    -- JSON of all results in a DB (+ filter metadata)
  GET  /kinburg/report/image   -- serves a report thumbnail (allow-listed paths only)

Guarded so the package still imports without ComfyUI/aiohttp present.
"""
import os

from . import db as _db

REPORT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html")

# Image paths become servable only after they show up in a /data response.
_ALLOWED_IMG = set()


def _register_img(p):
    if p:
        _ALLOWED_IMG.add(os.path.normcase(os.path.abspath(p)))


def _default_db():
    try:
        import folder_paths
        return os.path.join(folder_paths.get_output_directory(), "kinburg", "reports.db")
    except Exception:
        return os.path.join("output", "kinburg", "reports.db")


try:
    from server import PromptServer
    from aiohttp import web

    routes = PromptServer.instance.routes

    @routes.post("/kinburg/report/save")
    async def _save(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        db_path = (body.get("db_path") or "").strip()
        if not db_path:
            return web.json_response({"ok": False, "error": "db_path is required"}, status=400)
        try:
            res = _db.upsert_run(db_path, body.get("run") or {}, body.get("results") or [])
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **res})

    @routes.get("/kinburg/report")
    async def _page(request):
        if not os.path.isfile(REPORT_HTML):
            return web.Response(status=404, text="report.html missing")
        return web.FileResponse(REPORT_HTML, headers={"Content-Type": "text/html; charset=utf-8"})

    @routes.get("/kinburg/report/data")
    async def _data(request):
        db_path = (request.query.get("db", "") or "").strip() or _default_db()
        try:
            out = _db.fetch_results(db_path)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        for r in out["results"]:
            _register_img(r.get("image_path"))
        return web.json_response({"ok": True, "db_path": db_path, **out})

    @routes.post("/kinburg/report/result")
    async def _update(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        db_path = (body.get("db_path") or "").strip()
        if not db_path or body.get("id") is None:
            return web.json_response({"ok": False, "error": "db_path and id are required"}, status=400)
        try:
            res = _db.update_result(db_path, body["id"], status=body.get("status"),
                                    rating=body.get("rating"), comment=body.get("comment"), tags=body.get("tags"))
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **res})

    @routes.post("/kinburg/report/run/delete")
    async def _delrun(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        db_path = (body.get("db_path") or "").strip()
        run_key = (body.get("run_key") or "").strip()
        if not db_path or not run_key:
            return web.json_response({"ok": False, "error": "db_path and run_key are required"}, status=400)
        try:
            res = _db.delete_run(db_path, run_key)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True, **res})

    @routes.get("/kinburg/report/image")
    async def _image(request):
        p = request.query.get("path", "")
        key = os.path.normcase(os.path.abspath(p)) if p else ""
        if not p or key not in _ALLOWED_IMG or not os.path.isfile(p):
            return web.Response(status=404, text="not found")
        return web.FileResponse(p)

except Exception as e:  # pragma: no cover
    print(f"[KinburgReport] could not register routes: {e}")
