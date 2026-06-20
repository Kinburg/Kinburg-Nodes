import os
import io
import json
import base64
import urllib.parse

# Heavy / ComfyUI-only imports are guarded so the package still imports — and the
# Comfy Registry can enumerate its nodes — in an environment without ComfyUI present.
# Inside ComfyUI at runtime these are always available.
try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import folder_paths
except Exception:
    pass

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
VIEWER_TEMPLATE = os.path.join(NODE_DIR, "viewer.html")

_ALLOWED_HTML = set()


def _register_html(path):
    _ALLOWED_HTML.add(os.path.normcase(os.path.abspath(path)))


try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/image_compare")
    async def _serve_compare(request):
        p = request.query.get("path", "")
        key = os.path.normcase(os.path.abspath(p)) if p else ""
        if not p or key not in _ALLOWED_HTML or not key.endswith(".html") or not os.path.isfile(p):
            return web.Response(status=404, text="not found")
        return web.FileResponse(p, headers={"Content-Type": "text/html; charset=utf-8"})
except Exception as e:  # pragma: no cover
    print(f"[ImageCompare] could not register web route: {e}")


# ----------------------------------------------------------------------------- helpers
def _tensors_to_pil(images):
    """ComfyUI IMAGE tensor [B,H,W,C] float 0-1 -> list of PIL.Image (RGB)."""
    arr = images.cpu().numpy()
    return [Image.fromarray((np.clip(im, 0.0, 1.0) * 255.0).astype(np.uint8)) for im in arr]


def _pil_to_tensor_batch(pils):
    import torch
    arrs = [np.asarray(p.convert("RGB")).astype(np.float32) / 255.0 for p in pils]
    return torch.from_numpy(np.stack(arrs))


def _b64_png(pil):
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _load_font(size):
    candidates = ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf",
                  os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "arial.ttf")]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(text, font, draw, max_w):
    words = text.split()
    if not words:
        return [text]
    lines, cur = [], words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _overlay_caption(pil, text, position="bottom", font_size=0):
    text = (text or "").strip()
    if not text:
        return pil.convert("RGB")
    img = pil.convert("RGBA")
    W, H = img.size
    fs = font_size if font_size > 0 else max(14, W // 38)
    font = _load_font(fs)
    pad = max(6, fs // 3)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    lines = _wrap(text, font, draw, W - 2 * pad)
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + max(2, fs // 6)
    band_h = line_h * len(lines) + 2 * pad
    y0 = (H - band_h) if position == "bottom" else 0
    draw.rectangle([0, y0, W, y0 + band_h], fill=(0, 0, 0, 150))
    y = y0 + pad
    for ln in lines:
        draw.text((pad, y), ln, font=font, fill=(255, 255, 255, 255))
        y += line_h
    return Image.alpha_composite(img, overlay).convert("RGB")


def _split_prompts(text, sep):
    """Split full (possibly multi-line) prompts on a separator LINE (default '---')."""
    if not text or not text.strip():
        return []
    sep = (sep or "---").strip() or "---"
    blocks, cur = [], []
    for line in text.split("\n"):
        if line.strip() == sep:
            blocks.append("\n".join(cur).strip())
            cur = []
        else:
            cur.append(line)
    blocks.append("\n".join(cur).strip())
    return blocks


def _server_port():
    try:
        from comfy.cli_args import args
        if getattr(args, "port", None):
            return int(args.port)
    except Exception:
        pass
    try:
        from server import PromptServer
        if getattr(PromptServer.instance, "port", None):
            return int(PromptServer.instance.port)
    except Exception:
        pass
    return 8188


def _unique_html_path(out_dir, prefix):
    i = 1
    while True:
        name = f"{prefix}_{i:04d}.html"
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            return name, path
        i += 1


# ----------------------------------------------------------------------------- node
class ImageCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "captions": ("STRING", {"multiline": True, "default": "", "tooltip": "One caption per line, aligned with the image batch (missing lines become empty)"}),
                "title": ("STRING", {"default": "Image comparison"}),
                "columns": ("INT", {"default": 3, "min": 1, "max": 12, "tooltip": "Default number of columns in Grid mode"}),
                "overlay_captions": ("BOOLEAN", {"default": True, "tooltip": "Draw captions onto the 'images_captioned' output"}),
                "caption_position": (["bottom", "top"], {"default": "bottom"}),
                "font_size": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "Caption font size in px; 0 = auto from image width"}),
                "filename_prefix": ("STRING", {"default": "compare"}),
                "save_captioned_images": ("BOOLEAN", {"default": False, "tooltip": "Also save the captioned images as PNG files in the output folder"}),
            },
            "optional": {
                "prompts": ("STRING", {"multiline": True, "default": "", "tooltip": "Full generation prompts, separated by a line equal to 'prompt_separator'. Shown on the page (toggleable). Multi-line prompts are fine."}),
                "prompt_separator": ("STRING", {"default": "---", "tooltip": "A line equal to this string separates one prompt from the next"}),
                "show_prompts": ("BOOLEAN", {"default": True, "tooltip": "Initial visibility of prompts on the page (can be toggled there too)"}),
                "output_dir": ("STRING", {"default": "", "tooltip": "Custom save folder (absolute path). Empty = ComfyUI output. A served copy is also written to output so 'Open comparison' keeps working."}),
                "save_prompts_txt": ("BOOLEAN", {"default": False, "tooltip": "Save each prompt to a .txt file named like its image"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images_captioned", "html_path", "url")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/image/compare"
    OUTPUT_NODE = True

    def run(self, images, captions, title, columns, overlay_captions,
            caption_position, font_size, filename_prefix, save_captioned_images=False,
            prompts="", prompt_separator="---", show_prompts=True,
            output_dir="", save_prompts_txt=False):
        pils = _tensors_to_pil(images)
        n = len(pils)

        # Captions: one short tag per line, padded/truncated to the batch size.
        caps = captions.split("\n") if captions else []
        caps = [c.strip() for c in caps]
        caps = (caps + [""] * n)[:n]

        # Full generation prompts: separated by the separator line, may be multi-line.
        proms = _split_prompts(prompts, prompt_separator)
        proms = (proms + [""] * n)[:n]

        # Data for the interactive page uses the ORIGINAL (clean) images.
        items = [{"src": _b64_png(p), "caption": c, "prompt": pr}
                 for p, c, pr in zip(pils, caps, proms)]
        cfg = {"title": title, "columns": int(columns),
               "show_prompts": bool(show_prompts), "items": items}

        try:
            with open(VIEWER_TEMPLATE, "r", encoding="utf-8") as f:
                template = f.read()
        except Exception as e:
            return self._err(f"viewer.html missing: {e}", pils)

        html = template.replace("/*__COMPARE_DATA__*/null", json.dumps(cfg, ensure_ascii=True))
        prefix = filename_prefix or "compare"

        comfy_out = folder_paths.get_output_directory()
        out_dir = output_dir.strip() or comfy_out
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            return self._err(f"cannot create output_dir '{out_dir}': {e}", pils)

        name, path = _unique_html_path(out_dir, prefix)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        _register_html(path)
        q = urllib.parse.quote(os.path.abspath(path))
        url = f"http://127.0.0.1:{_server_port()}/image_compare?path={q}"
        print(f"[ImageCompare] saved {path}\n[ImageCompare] open: {url}")

        if overlay_captions:
            cap_pils = [_overlay_caption(p, c, caption_position, font_size) for p, c in zip(pils, caps)]
        else:
            cap_pils = [p.convert("RGB") for p in pils]
        out_tensor = _pil_to_tensor_batch(cap_pils)

        if save_captioned_images or save_prompts_txt:
            saved = 0
            for idx in range(n):
                k = 1
                while True:
                    base = f"{prefix}_{idx + 1:02d}_{k:04d}"
                    png = os.path.join(out_dir, base + ".png")
                    txt = os.path.join(out_dir, base + ".txt")
                    if not os.path.exists(png) and not os.path.exists(txt):
                        break
                    k += 1
                if save_captioned_images:
                    cap_pils[idx].save(png)
                if save_prompts_txt:
                    with open(txt, "w", encoding="utf-8") as f:
                        f.write(proms[idx])
                saved += 1
            print(f"[ImageCompare] exported files for {saved} image(s) to {out_dir}")

        return {"ui": {"compare_url": [url], "compare_path": [path]},
                "result": (out_tensor, path, url)}

    def _err(self, msg, pils):
        print(f"[ImageCompare] ERROR: {msg}")
        out = _pil_to_tensor_batch([p.convert("RGB") for p in pils]) if pils else None
        return {"ui": {"compare_url": [""], "compare_path": [""]},
                "result": (out, f"[ERROR] {msg}", "")}


NODE_CLASS_MAPPINGS = {"ImageCompareHTML": ImageCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageCompareHTML": "Image Compare (HTML)"}
