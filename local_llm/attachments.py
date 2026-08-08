"""Where chat pictures live, and the one route that deletes them.

Every picture that reaches a Local LLM Chat window — pasted, dropped, picked with 📎, or pushed by
Send Image to Chat — is written to ``<ComfyUI input>/kinburg_chat/`` and referred to afterwards by
ComfyUI's own ``{name, subfolder, type}`` triple. One folder and one naming scheme means the chat
window can show a thumbnail through ``/view``, the chat node can open the same file to send its
pixels, and this module can clean up after both.

``input`` rather than ``temp`` on purpose: temp is wiped on restart, and a chat saved inside a
workflow would come back with holes where its pictures used to be.
"""
import os

ATT_DIR = "kinburg_chat"


def att_base():
    """Absolute path of the attachment folder, created if it isn't there yet."""
    import folder_paths
    p = os.path.join(folder_paths.get_input_directory(), ATT_DIR)
    os.makedirs(p, exist_ok=True)
    return p


def resolve_refs(att):
    """Attachment refs -> (real paths, names that could not be found).

    Refs arrive inside a workflow JSON, so they are untrusted: each is resolved and then checked to
    be genuinely under its base directory before anything opens it.
    """
    import folder_paths
    bases = {"input": folder_paths.get_input_directory,
             "temp": folder_paths.get_temp_directory,
             "output": folder_paths.get_output_directory}
    paths, missing = [], []
    for a in att or []:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name") or "")
        if not name:
            continue
        get = bases.get(str(a.get("type") or "input"))
        if get is None:
            missing.append(name)
            continue
        base = os.path.abspath(get())
        p = os.path.abspath(os.path.join(base, str(a.get("subfolder") or ""), name))
        if not (p == base or p.startswith(base + os.sep)) or not os.path.isfile(p):
            missing.append(name)
            continue
        paths.append(p)
    return paths, missing


def discard(refs):
    """Delete attachment files. Returns (deleted, refused) counts.

    Stricter than `resolve_refs` on purpose: a delete only ever touches a file sitting DIRECTLY in
    the attachment folder. Resolving inside the base directory is enough to read a picture, but not
    to remove one — otherwise a hand-edited ref could talk this route into deleting somebody's
    LoadImage sources, which live in that same input tree.
    """
    try:
        allowed = os.path.abspath(att_base())
    except Exception:
        return 0, len(refs or [])
    deleted = refused = 0
    for a in refs or []:
        if not isinstance(a, dict):
            refused += 1
            continue
        name = str(a.get("name") or "")
        if not name or os.path.basename(name) != name:
            refused += 1                      # a path, not a filename
            continue
        p = os.path.abspath(os.path.join(allowed, name))
        if os.path.dirname(p) != allowed:
            refused += 1
            continue
        try:
            os.remove(p)
            deleted += 1
        except FileNotFoundError:
            pass                              # already gone: the outcome the caller wanted
        except Exception as e:
            print(f"[LocalLLM] could not delete attachment {name}: {e}")
            refused += 1
    return deleted, refused


# Guarded the same way as the other routes in this pack, so the package still imports without
# ComfyUI / aiohttp around (the test harnesses rely on that).
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/kinburg/chat/discard")
    async def _discard(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        refs = body.get("refs")
        if not isinstance(refs, list):
            return web.json_response({"ok": False, "error": "missing refs"}, status=400)
        deleted, refused = discard(refs)
        return web.json_response({"ok": True, "deleted": deleted, "refused": refused})

except Exception as e:  # pragma: no cover
    print(f"[LocalLLM] could not register the attachment route: {e}")
