"""Send Image to Chat — push a generated picture into a Local LLM Chat (GGUF) window.

The point of this node is what it does NOT do: it never links to the chat node. 📨 Send re-runs
everything upstream of the chat, so an IMAGE wired into it would drag a whole sampler branch along
on every message. Instead this node writes the picture into ComfyUI's ``input/kinburg_chat/`` —
the same folder a pasted one lands in — and hands the frontend a file reference. The chat window
then either stages it in its tray or hangs it on a message; nothing about that touches the graph.

**Who it comes from** (``send_as``):
  • *me (user)* — the picture joins the chat's tray and goes out with your next message, pixels and
    all, exactly as if you had pasted it.
  • *a persona* — the picture is hung on that persona's most recent bubble, so it reads as though
    they sent it. The MODEL is not shown it: llama.cpp only accepts images on user turns, and a
    persona has no need to look at a photo it supposedly took. By default it leaves no trace in the
    context either (``ctx: false`` on the attachment) — the picture usually came from something the
    persona had already described, so the description is in the conversation twice over otherwise.
    Turn on ``note_in_context`` when you do want a line about it, and give it a ``caption``.

**Caption.** Whatever the model reads once the pixels are gone (chat_node._content_of), so it wants
to be a sentence of plain prose, NOT the generation prompt: a paragraph of comma-separated tags in
the middle of a conversation teaches the persona to write in comma-separated tags. If your camera
persona produces both, split them — one line for the sampler, one for here.

**When** (``when``): *every run* pushes on each execution, *on button press* stashes the result and
waits for the 📌 on the node. Either way the file is named after a hash of its own pixels, so
re-running a branch that produced the same picture pushes the same reference and the chat quietly
recognises it instead of stacking duplicates.

The 📌 reads ``send_as`` / ``caption`` / ``shot`` / ``note_in_context`` **at the moment you press
it**, not as they were when the branch ran — deciding who a picture comes from is something you do
after looking at it. Only ``megapixels`` needs a re-run, because it changes the file rather than
the reference. That re-read lives in web/chat_send.js (``refsNow``); the rule it applies for
``ctx`` must stay in step with ``run()`` below.
"""
import hashlib
import json
import os

from .attachments import ATT_DIR, att_base
from ..categories import CAT_LLM

SEND_AS = ["the active persona", "me (user)",
           "persona 1", "persona 2", "persona 3", "persona 4", "persona 5", "persona 6"]
WHEN = ["on button press", "every run"]


def _resize_to_mp(pil, megapixels):
    """Scale so the picture is about `megapixels` MP, never up. 0 keeps it as it is."""
    if not megapixels or megapixels <= 0:
        return pil
    target = float(megapixels) * 1_000_000.0
    now = float(pil.width * pil.height)
    if now <= target:
        return pil
    from PIL import Image
    s = (target / now) ** 0.5
    return pil.resize((max(1, round(pil.width * s)), max(1, round(pil.height * s))),
                      Image.Resampling.LANCZOS)


def _save_frames(image, megapixels):
    """ComfyUI IMAGE -> file refs under input/kinburg_chat, one per frame.

    The name is a hash of the pixels that actually get written, which buys idempotency for free:
    a branch that re-runs and produces the same picture writes the same file, so the frontend can
    tell "this is the one I already have" from "this is a new one" by name alone.
    """
    import numpy as np
    from PIL import Image

    base = att_base()
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim == 3:
        arr = arr[None, ...]
    refs = []
    for frame in arr:
        pil = _resize_to_mp(
            Image.fromarray((np.clip(frame, 0.0, 1.0) * 255.0).astype("uint8")).convert("RGB"),
            megapixels)
        digest = hashlib.sha1(pil.tobytes() + f"{pil.width}x{pil.height}".encode()).hexdigest()[:12]
        name = f"kb_{digest}.png"
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            pil.save(path, format="PNG")
        refs.append({"name": name, "subfolder": ATT_DIR, "type": "input"})
    return refs


class LocalLLMChatSendImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The picture to put in the chat. A batch sends every frame."}),
                "send_as": (SEND_AS, {"default": SEND_AS[0], "tooltip": "Who the picture comes from. A persona: it hangs on that persona's last message, as though they sent it — the model is not shown the pixels. 'me (user)': it joins the chat's tray and goes out with your next message, which the model DOES see."}),
                "when": (WHEN, {"default": WHEN[0], "tooltip": "'on button press' saves the picture and waits for 📌 on this node — the usual choice, so you can look at the result first. 'every run' pushes it the moment this node executes. With 📌 you can still change send_as, caption, shot and note_in_context after the picture is generated; they are read when you press it. Only megapixels needs a re-run, since it changes the saved file itself."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.25, "tooltip": "Downscale to about this many megapixels before saving (never upscales). 0 = keep full size. Only affects the copy that goes to the chat; the image output passes through untouched."}),
            },
            "optional": {
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "One or two sentences of plain prose — what the model reads about this picture once the pixels are gone. NOT the generation prompt: tag soup in the conversation teaches the persona to write tag soup. Leave empty for a picture that is purely something to look at."}),
                "shot": ("STRING", {"default": "", "tooltip": "Optional keyframe label, e.g. 'shot 3 / end'. Stored with the picture so a chat can be read back as a storyboard; it never reaches the model."}),
                "note_in_context": ("BOOLEAN", {"default": False, "tooltip": "Persona pictures only. Off (default) the picture is purely visual and adds nothing to the context — right when the persona already described the scene it is a picture of. On, it leaves '[image: caption]' in that persona's message so the conversation records that a photo was sent."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    OUTPUT_NODE = True          # so the file reference reaches the frontend in the ui payload
    CATEGORY = CAT_LLM

    def run(self, image, send_as, when, megapixels, caption="", shot="",
            note_in_context=False, unique_id=None):
        try:
            refs = _save_frames(image, megapixels)
        except Exception as e:
            payload = {"error": f"couldn't save the picture: {e}"}
            return {"ui": {"kinburg_chatsend": [json.dumps(payload, ensure_ascii=False)]},
                    "result": (image,)}

        cap, sh = (caption or "").strip(), (shot or "").strip()
        to_user = send_as == "me (user)"
        for r in refs:
            if cap:
                r["caption"] = cap
            if sh:
                r["shot"] = sh
            # A persona's picture is visual by default: no marker in the context. What YOU send is
            # always remembered, same as a pasted one — you showed it to the model on purpose.
            if not to_user and not note_in_context:
                r["ctx"] = False

        payload = {"refs": refs, "as": send_as, "when": when}
        return {"ui": {"kinburg_chatsend": [json.dumps(payload, ensure_ascii=False)]},
                "result": (image,)}


NODE_CLASS_MAPPINGS = {"LocalLLMChatSendImage": LocalLLMChatSendImage}
NODE_DISPLAY_NAME_MAPPINGS = {"LocalLLMChatSendImage": "Send Image to Chat"}
