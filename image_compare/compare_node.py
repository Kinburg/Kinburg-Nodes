import os
import io
import json
import uuid
import base64
import urllib.parse

from ..util.separators import BLOCK_SEP

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


def _hex_to_rgb(hex_color, fallback=(255, 255, 255)):
    """'#RRGGBB' / '#RGB' (with or without '#') -> (r, g, b); fallback if invalid."""
    if not hex_color:
        return fallback
    s = str(hex_color).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return fallback
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback


def _norm_hex_or_none(value):
    """Canonical '#RRGGBB' for a valid hex string, else None."""
    rgb = _hex_to_rgb(value, None) if isinstance(value, str) and value.strip() else None
    return ("#%02X%02X%02X" % rgb) if rgb else None


def _parse_caption(raw):
    """A caption line is either plain text or a Color Caption JSON object
    ({"caption", "color", "band_color"}). Returns (text, color_hex_or_None,
    band_hex_or_None). Plain text keeps the old behavior (stripped, no colors)."""
    s = (raw or "").strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict) and "caption" in obj:
            text = obj.get("caption", "")
            text = text if isinstance(text, str) else str(text)
            return text, _norm_hex_or_none(obj.get("color")), _norm_hex_or_none(obj.get("band_color"))
    return s, None, None


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


def _overlay_caption(pil, text, position="bottom", font_size=0, color=None, band_color=None):
    text = (text or "").strip()
    if not text:
        return pil.convert("RGB")
    fill = _hex_to_rgb(color) + (255,)
    band = _hex_to_rgb(band_color, (0, 0, 0)) + (150,)
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
    draw.rectangle([0, y0, W, y0 + band_h], fill=band)
    y = y0 + pad
    for ln in lines:
        draw.text((pad, y), ln, font=font, fill=fill)
        y += line_h
    return Image.alpha_composite(img, overlay).convert("RGB")


def _settings_block(fields, cap=300):
    """Render one image's structured settings into display text, one line per field:

        [KSamplerSelect] sampler_name: res_multistep
        [PrimitiveInt] value: 164917466401748

    `fields` is a list of {"key", "value"} from Generation Info Filter's settings_data,
    where key is 'ClassType.param' (or 'ClassType #ord.param'). Values are squashed to a
    single line and capped so one huge value can't blow up the panel.
    """
    lines = []
    for f in fields or []:
        if not isinstance(f, dict):
            continue
        key = str(f.get("key", ""))
        val = f.get("value", "")
        val = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
        val = " ".join(val.split())
        if len(val) > cap:
            val = val[:cap] + "…"
        cls, dot, param = key.partition(".")
        lines.append(f"[{cls}] {param}: {val}" if dot else f"{key}: {val}" if key else val)
    return "\n".join(lines)


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
                "title": ("STRING", {"default": "Image comparison"}),
                "columns": ("INT", {"default": 3, "min": 1, "max": 12, "tooltip": "Default number of columns in Grid mode"}),
                "overlay_captions": ("BOOLEAN", {"default": True, "tooltip": "Draw captions onto the 'images_captioned' output"}),
                "caption_position": (["bottom", "top"], {"default": "bottom"}),
                "font_size": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "Caption font size in px; 0 = auto from image width"}),
                "filename_prefix": ("STRING", {"default": "compare"}),
                "save_captioned_images": ("BOOLEAN", {"default": False, "tooltip": "Also save the captioned images as PNG files in the output folder"}),
            },
            "optional": {
                "captions": ("STRING", {"forceInput": True, "tooltip": "One caption per line, aligned with the image batch (e.g. from Get Accumulator (captions)). Missing lines become empty."}),
                "prompts": ("STRING", {"forceInput": True, "tooltip": "Full generation prompts, one block per image separated by a '---' line (e.g. from Get Accumulator (prompts)). Shown on the page (toggleable). Multi-line prompts are fine."}),
                "show_prompts": ("BOOLEAN", {"default": False, "tooltip": "Initial visibility of prompts on the page (can be toggled there too)"}),
                "output_dir": ("STRING", {"default": "", "tooltip": "Custom save folder (absolute path). Empty = ComfyUI output. A served copy is also written to output so 'Open comparison' keeps working."}),
                "save_prompts_txt": ("BOOLEAN", {"default": False, "tooltip": "Save each prompt to a .txt file named like its image"}),
                "settings_data": ("GEN_SETTINGS", {"tooltip": "Structured per-image settings from Generation Info Filter's 'settings_data' output. Rendered under each image (one '[Class] param: value' line per field; toggleable) and stored by field (EAV) when you 'Save run to report'."}),
                "show_settings": ("BOOLEAN", {"default": True, "tooltip": "Initial visibility of settings on the page (can be toggled there too)"}),
                "report_db": ("STRING", {"default": "", "tooltip": "SQLite file the page's 'Save run to report' button writes to. Empty = <output>/kinburg/reports.db. The value is shown (and editable) on the page."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images_captioned", "html_path", "url")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/image/compare"
    OUTPUT_NODE = True

    def run(self, images, title, columns, overlay_captions,
            caption_position, font_size, filename_prefix,
            captions="", save_captioned_images=False,
            prompts="", show_prompts=False,
            output_dir="", save_prompts_txt=False,
            show_settings=True, settings_data="", report_db=""):
        pils = _tensors_to_pil(images)
        n = len(pils)

        # Report prep: save clean per-image PNGs to a stable, run-scoped folder and gather a
        # run_id + resolved DB path. The page's "Save run to report" button uses these; the
        # node itself never writes to the DB.
        comfy_out = folder_paths.get_output_directory()
        run_id = uuid.uuid4().hex[:12]
        report_db_resolved = report_db.strip() or os.path.join(comfy_out, "kinburg", "reports.db")
        report_paths = []
        try:
            rdir = os.path.join(comfy_out, "kinburg", "report_images", run_id)
            os.makedirs(rdir, exist_ok=True)
            for i, p in enumerate(pils):
                rp = os.path.join(rdir, f"{i:03d}.png")
                p.convert("RGB").save(rp)
                report_paths.append(rp)
        except Exception as e:
            print(f"[ImageCompare] could not save report images: {e}")
            report_paths = []
        report_paths = (report_paths + [""] * n)[:n]

        # Captions: one per line, padded/truncated to the batch size. Each line is
        # either plain text or a Color Caption JSON ({"caption","color","band_color"});
        # the colors (when present) tint the text and the band behind it, both on the
        # page and on the drawn images.
        caps, cap_colors, cap_bands = [], [], []
        for line in (captions.split("\n") if captions else []):
            text, color, band = _parse_caption(line)
            caps.append(text)
            cap_colors.append(color)
            cap_bands.append(band)
        caps = (caps + [""] * n)[:n]
        cap_colors = (cap_colors + [None] * n)[:n]
        cap_bands = (cap_bands + [None] * n)[:n]

        # Full generation prompts: blocks separated by a '---' line, may be multi-line.
        proms = _split_prompts(prompts, BLOCK_SEP)
        proms = (proms + [""] * n)[:n]

        # Structured per-image settings (from Generation Info Filter) drive both the report
        # DB (stored by field) and the on-page settings text (rendered one line per field).
        try:
            sdata = json.loads(settings_data) if isinstance(settings_data, str) and settings_data.strip() else []
        except (ValueError, TypeError):
            sdata = []
        if not isinstance(sdata, list):
            sdata = []
        sdata = (sdata + [[]] * n)[:n]
        setts = [_settings_block(fields) for fields in sdata]

        # Data for the interactive page uses the ORIGINAL (clean) images.
        items = [{"src": _b64_png(p), "caption": c, "prompt": pr, "settings": st,
                  "color": col, "band": band, "report_path": rpth, "settings_data": sd}
                 for p, c, pr, st, col, band, rpth, sd in
                 zip(pils, caps, proms, setts, cap_colors, cap_bands, report_paths, sdata)]
        cfg = {"title": title, "columns": int(columns), "show_prompts": bool(show_prompts),
               "show_settings": bool(show_settings), "run_id": run_id,
               "report_db": report_db_resolved, "items": items}

        try:
            with open(VIEWER_TEMPLATE, "r", encoding="utf-8") as f:
                template = f.read()
        except Exception as e:
            return self._err(f"viewer.html missing: {e}", pils)

        html = template.replace("/*__COMPARE_DATA__*/null", json.dumps(cfg, ensure_ascii=True))
        prefix = filename_prefix or "compare"

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
            cap_pils = [_overlay_caption(p, c, caption_position, font_size, col, band)
                        for p, c, col, band in zip(pils, caps, cap_colors, cap_bands)]
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
