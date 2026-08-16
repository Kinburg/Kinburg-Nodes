import os
import io
import re
import json
import uuid
import base64
import datetime
import urllib.parse

from ..util.separators import BLOCK_SEP
from ..util.paths import first_free
from ..categories import CAT_IMAGE_COMPARE

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
_ALLOWED_DIRS = {}   # token -> folder abspath (portable comparison bundles)
_DIRS_KEEP = 300     # how many tokens the on-disk registry remembers


def _registry_path():
    """Where the token → folder map is persisted, next to the report DB."""
    import folder_paths
    return os.path.join(folder_paths.get_output_directory(), "kinburg", "compare_dirs.json")


def _load_registry():
    """A node's 'Open comparison' URL is saved INTO the workflow, so it long outlives the process
    that minted it — but the token map used to live only in memory, so every ComfyUI restart turned
    every previously-issued link into a 404. The map is written to disk and re-read on a miss."""
    try:
        with open(_registry_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for tok, p in data.items():
                _ALLOWED_DIRS.setdefault(str(tok), str(p))
    except Exception:
        pass


def _register_html(path):
    _ALLOWED_HTML.add(os.path.normcase(os.path.abspath(path)))


def _register_dir(token, path):
    _ALLOWED_DIRS[token] = os.path.abspath(path)
    try:
        reg = _registry_path()
        os.makedirs(os.path.dirname(reg), exist_ok=True)
        keep = {}
        try:
            with open(reg, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict):
                keep = {str(k): str(v) for k, v in old.items()}
        except Exception:
            pass
        keep[token] = _ALLOWED_DIRS[token]
        if len(keep) > _DIRS_KEEP:           # dicts keep insertion order — drop the oldest
            keep = dict(list(keep.items())[-_DIRS_KEEP:])
        with open(reg, "w", encoding="utf-8") as f:
            json.dump(keep, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"[ImageCompare] could not persist the comparison-folder registry: {e}")


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

    @PromptServer.instance.routes.get("/image_compare_dir/{token}/{tail:.*}")
    async def _serve_compare_dir(request):
        # Serve any file inside a registered portable bundle folder, so the page's relative
        # image paths ("images/000.png") resolve when opened in-app.
        token = request.match_info.get("token", "")
        base = _ALLOWED_DIRS.get(token)
        if base is None:            # not minted by THIS process — a link from before a restart
            _load_registry()
            base = _ALLOWED_DIRS.get(token)
        tail = request.match_info.get("tail", "") or "index.html"
        if not base or not os.path.isdir(base):
            return web.Response(status=404, text="not found")
        target = os.path.normpath(os.path.join(base, tail))
        base_n, target_n = os.path.normcase(base), os.path.normcase(target)
        if not (target_n == base_n or target_n.startswith(base_n + os.sep)) or not os.path.isfile(target):
            return web.Response(status=404, text="not found")
        hdrs = {"Content-Type": "text/html; charset=utf-8"} if target.lower().endswith(".html") else None
        return web.FileResponse(target, headers=hdrs)
except Exception as e:  # pragma: no cover
    print(f"[ImageCompare] could not register web route: {e}")


# ----------------------------------------------------------------------------- helpers
def _first(v, default=None):
    """INPUT_IS_LIST wraps every input in a list; take the first element (or default)."""
    if isinstance(v, list):
        return v[0] if v else default
    return v


def _images_to_pils(images):
    """Flatten the `images` input (a list under INPUT_IS_LIST whose elements are IMAGE batch
    tensors or single frames, possibly of different sizes) into one flat list of PIL (RGB).
    Accepts a batch tensor and a ComfyUI image list alike."""
    pils = []
    for v in (images if isinstance(images, list) else [images]):
        if v is None:
            continue
        arr = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        if arr.ndim == 3:
            arr = arr[None, ...]
        for im in arr:
            pils.append(Image.fromarray((np.clip(im, 0.0, 1.0) * 255.0).astype(np.uint8)).convert("RGB"))
    return pils


def _pils_to_tensor_list(pils):
    """PILs -> a list of single-frame IMAGE tensors [1,H,W,C]. Used for the `images_captioned`
    output (OUTPUT_IS_LIST), so different-sized images stay separate — no batching/padding."""
    import torch
    return [torch.from_numpy(np.asarray(p.convert("RGB")).astype(np.float32) / 255.0)[None, ...]
            for p in pils]


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


_TIME_TOKEN_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?)?",
    re.I)


def _time_to_seconds(raw):
    """Best-effort parse of a displayed time string into float seconds (for sorting).

    Handles every Stop Timer format — 'HH:MM:SS' / 'MM:SS', '1h 2m 3s', '12.34 s',
    '890 ms', the 'human' shorthand ('45.0s', '890ms') — plus a plain number (seconds).
    Returns None when nothing parses, so callers can push those to the end when sorting.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Clock form: HH:MM:SS or MM:SS (only digits / colons / dot / spaces).
    if ":" in s and re.fullmatch(r"[\d:.\s]+", s):
        try:
            total = 0.0
            for part in s.split(":"):
                total = total * 60 + float(part)
            return total
        except ValueError:
            pass
    # Token form: sum of value+unit chunks; a unitless number counts as seconds.
    total, matched = 0.0, False
    for m in _TIME_TOKEN_RE.finditer(s):
        try:
            v = float(m.group(1))
        except (TypeError, ValueError):
            continue
        unit = (m.group(2) or "").lower()
        matched = True
        if unit in ("ms", "millisecond", "milliseconds"):
            total += v / 1000.0
        elif unit in ("m", "min", "mins", "minute", "minutes"):
            total += v * 60.0
        elif unit in ("h", "hr", "hrs", "hour", "hours"):
            total += v * 3600.0
        else:  # "", s, sec, secs, second, seconds
            total += v
    return total if matched else None


def _psnr(x, y):
    """PSNR in dB between two float arrays in [0,1] of the same shape. None if identical
    (infinite)."""
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return None
    return round(float(10.0 * np.log10(1.0 / mse)), 2)


def _ssim(a_pil, b_pil):
    """Structural Similarity (0..1, higher = more similar) on the luminance channel, with a
    Gaussian window — the standard formulation, in torch (no extra deps). Assumes equal size.
    Returns None if the images are too small for a window."""
    import torch
    import torch.nn.functional as F

    def gray(p):
        arr = np.asarray(p.convert("L")).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, None]

    x, y = gray(a_pil), gray(b_pil)
    h, w = x.shape[-2:]
    win = min(11, h, w)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return None
    sigma = 1.5
    coords = torch.arange(win, dtype=torch.float32) - (win - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :])[None, None]
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y = F.conv2d(x, window), F.conv2d(y, window)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    var_x = F.conv2d(x * x, window) - mu_x2
    var_y = F.conv2d(y * y, window) - mu_y2
    cov = F.conv2d(x * y, window) - mu_xy
    smap = ((2 * mu_xy + c1) * (2 * cov + c2)) / ((mu_x2 + mu_y2 + c1) * (var_x + var_y + c2))
    return round(float(smap.mean().item()), 4)


def _metrics_for(cand_pil, ref_pil):
    """(ssim, psnr) of a candidate vs a reference; the reference is resized to the candidate's
    size first (metrics need matching dimensions)."""
    ref = ref_pil if ref_pil.size == cand_pil.size else ref_pil.resize(cand_pil.size, Image.Resampling.LANCZOS)
    x = np.asarray(cand_pil.convert("RGB")).astype(np.float32) / 255.0
    y = np.asarray(ref.convert("RGB")).astype(np.float32) / 255.0
    psnr = _psnr(x, y)
    try:
        ssim = _ssim(cand_pil, ref)
    except Exception as e:
        print(f"[ImageCompare] SSIM failed: {e}")
        ssim = None
    return ssim, psnr


def _metrics_str(ssim, psnr, is_ref=False):
    parts = []
    if ssim is not None:
        parts.append(f"SSIM {ssim:.3f}")
    parts.append("PSNR ∞" if psnr is None else f"PSNR {psnr:.1f} dB")
    s = " · ".join(parts)
    return (s + " (reference)") if is_ref else s


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
    _, base = first_free(out_dir, lambda k: f"{prefix}_{k:04d}", [".html"])
    name = base + ".html"
    return name, os.path.join(out_dir, name)


# ----------------------------------------------------------------------------- node
class ImageCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to compare — a batch OR an image list (e.g. from Get Accumulator (images list)); different sizes are fine."}),
                "title": ("STRING", {"default": "Image comparison"}),
                "columns": ("INT", {"default": 3, "min": 1, "max": 12, "tooltip": "Default number of columns in Grid mode"}),
                "overlay_captions": ("BOOLEAN", {"default": True, "tooltip": "Draw captions onto the 'images_captioned' output"}),
                "caption_position": (["bottom", "top"], {"default": "bottom"}),
                "font_size": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "Caption font size in px; 0 = auto from image width"}),
                "filename_prefix": ("STRING", {"default": "compare"}),
                "save_captioned_images": ("BOOLEAN", {"default": False, "tooltip": "Also save the captioned images as PNG files in the output folder"}),
                "auto_collect": ("BOOLEAN", {"default": True, "tooltip": "Re-wire every Get Accumulator in the graph automatically, right before the workflow is queued — so a Set you just added, removed, muted or bypassed is picked up without a click. Off: only the '🔌 Collect All' button collects. (Purely an editor convenience; the backend ignores this value.)"}),
            },
            "optional": {
                "captions": ("STRING", {"forceInput": True, "tooltip": "One caption per line, aligned with the image batch (e.g. from Get Accumulator (captions)). Missing lines become empty."}),
                "prompts": ("STRING", {"forceInput": True, "tooltip": "Full generation prompts, one block per image separated by a '---' line (e.g. from Get Accumulator (prompts)). Shown on the page (toggleable). Multi-line prompts are fine."}),
                "times": ("STRING", {"forceInput": True, "tooltip": "Per-image generation time — one entry per line, aligned with the image batch (e.g. Stop Timer's 'elapsed' collected via Get Accumulator (texts) with a newline separator). Shown under each image and used for the grid's 'Time' sort. Strings like '12.34 s', '1m 30s', '890 ms' or '00:01:30' are parsed for sorting."}),
                "output_dir": ("STRING", {"default": "", "tooltip": "Custom save folder (absolute path). Empty = ComfyUI output. A served copy is also written to output so 'Open comparison' keeps working."}),
                "embed_images": ("BOOLEAN", {"default": False, "tooltip": "How the comparison is saved. Off (default): a portable FOLDER '<prefix>_<datetime>/' with a light index.html + an images/ subfolder (relative links) — open it offline, zip/share it, or open it in-app. On: a single self-contained .html with every image inlined as base64 (one file, much larger). Both open from the node's URL output."}),
                "save_prompts_txt": ("BOOLEAN", {"default": False, "tooltip": "Save each prompt to a .txt file named like its image"}),
                "settings_data": ("GEN_SETTINGS", {"tooltip": "Structured per-image settings from Generation Info Filter's 'settings_data' output. Rendered under each image (one '[Class] param: value' line per field; toggleable) and stored by field (EAV) when you 'Save run to report'."}),
                "report_db": ("STRING", {"default": "", "tooltip": "SQLite file the page's 'Save run to report' button writes to. Empty = <output>/kinburg/reports.db. The value is shown (and editable) on the page."}),
                "reference": ("IMAGE", {"tooltip": "Optional baseline to measure similarity against (e.g. the source of an upscale/img2img, or the fp16 output when checking a quantized model). When connected, SSIM + PSNR are computed for every image vs this reference and shown on the page (with a 'Similarity' sort)."}),
                "reference_index": ("INT", {"default": -1, "min": -1, "max": 4096, "tooltip": "Used only when no 'reference' image is connected: 0-based index of the image IN THIS BATCH to use as the baseline (every image is measured against it). -1 = off (no metrics)."}),
                "judge_data": ("STRING", {"forceInput": True, "tooltip": "Optional AI verdicts — wire the Vision LLM Judge's 'results_json' here. Each image gets a read-only judge section (stars / tags / comment), separate from your own review, with its own page toggle."}),
            },
        }

    INPUT_IS_LIST = True  # gather the whole batch/list into ONE page (don't run per-image)
    # images_captioned is a LIST (one frame per image), so mixed sizes stay separate; the two
    # string outputs are single values.
    OUTPUT_IS_LIST = (True, False, False)
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images_captioned", "html_path", "url")
    FUNCTION = "run"
    CATEGORY = CAT_IMAGE_COMPARE
    OUTPUT_NODE = True

    def run(self, images, title, columns, overlay_captions,
            caption_position, font_size, filename_prefix,
            captions="", save_captioned_images=False,
            prompts="", times="",
            output_dir="", save_prompts_txt=False,
            settings_data="", report_db="", embed_images=False,
            reference=None, reference_index=-1, judge_data="",
            auto_collect=True,          # editor-only (the Collect All button lives on this node)
            show_prompts=False, show_settings=True):   # retired widgets: still honoured if an old
                                                       # API prompt sends them, otherwise defaulted
        # INPUT_IS_LIST: every input arrives wrapped in a list. Unwrap the scalar widgets and
        # flatten the images (a batch or a list, possibly mixed sizes) into one list of PIL.
        title = _first(title, "Image comparison")
        columns = int(_first(columns, 3))
        overlay_captions = bool(_first(overlay_captions, True))
        caption_position = _first(caption_position, "bottom")
        font_size = int(_first(font_size, 0))
        filename_prefix = _first(filename_prefix, "compare")
        save_captioned_images = bool(_first(save_captioned_images, False))
        captions = _first(captions, "") or ""
        prompts = _first(prompts, "") or ""
        show_prompts = bool(_first(show_prompts, False))
        times = _first(times, "") or ""
        output_dir = _first(output_dir, "") or ""
        embed_images = bool(_first(embed_images, False))
        save_prompts_txt = bool(_first(save_prompts_txt, False))
        show_settings = bool(_first(show_settings, True))
        settings_data = _first(settings_data, "") or ""
        report_db = _first(report_db, "") or ""
        reference_index = int(_first(reference_index, -1))
        judge_data = _first(judge_data, "") or ""

        pils = _images_to_pils(images)
        n = len(pils)
        if n == 0:
            return self._err("no images connected", [])

        # Where clean per-image PNGs live + how the page references them:
        #  embed_images=False → a portable folder  <prefix>_<datetime>/  holding images/ + a light
        #    index.html with RELATIVE src ("images/000.png"): self-contained — zip it, open the
        #    html offline, or open it in-app (served via /image_compare_dir/<token>/).
        #  embed_images=True  → pixels are inlined as base64 in a single .html; the clean PNGs
        #    still go to the report tree so "Save run to report" works.
        comfy_out = folder_paths.get_output_directory()
        run_id = uuid.uuid4().hex[:12]
        report_db_resolved = report_db.strip() or os.path.join(comfy_out, "kinburg", "reports.db")
        prefix = filename_prefix or "compare"
        out_dir = output_dir.strip() or comfy_out
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            return self._err(f"cannot create output_dir '{out_dir}': {e}", pils)

        bundle_dir = ""
        if embed_images:
            imgdir = os.path.join(comfy_out, "kinburg", "report_images", run_id)
        else:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            _, folder_name = first_free(
                out_dir, lambda k, s=stamp: f"{prefix}_{s}" if k == 1 else f"{prefix}_{s}_{k}", [""])
            bundle_dir = os.path.join(out_dir, folder_name)
            imgdir = os.path.join(bundle_dir, "images")

        report_paths = []
        try:
            os.makedirs(imgdir, exist_ok=True)
            for i, p in enumerate(pils):
                rp = os.path.join(imgdir, f"{i:03d}.png")
                p.convert("RGB").save(rp)
                report_paths.append(rp)
        except Exception as e:
            print(f"[ImageCompare] could not save images: {e}")
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

        # Per-image generation time: one entry per line (like captions). Keep the display
        # string as-is and precompute a numeric seconds value so the page can sort by it.
        time_lines = [t.strip() for t in (times.split("\n") if times else [])]
        time_lines = (time_lines + [""] * n)[:n]
        time_secs = [_time_to_seconds(t) for t in time_lines]

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

        # Similarity metrics (SSIM / PSNR) vs a reference — only when one is available: a connected
        # `reference` image wins; otherwise `reference_index` picks one from the batch (-1 = off).
        # The reference is resized to each candidate's size. Cheap (numpy/torch), no extra deps.
        ref_pils = _images_to_pils(reference) if reference is not None else []
        ref_pil = ref_pils[0] if ref_pils else None
        ref_self = None
        if ref_pil is None and 0 <= reference_index < n:
            ref_pil, ref_self = pils[reference_index], reference_index
        metrics_disp, m_ssim, m_psnr = [""] * n, [None] * n, [None] * n
        if ref_pil is not None:
            for i, p in enumerate(pils):
                ss, ps = _metrics_for(p, ref_pil)
                m_ssim[i], m_psnr[i] = ss, ps
                metrics_disp[i] = _metrics_str(ss, ps, is_ref=(i == ref_self))

        # Judge verdicts (Vision LLM Judge's results_json): parsed into a per-index map and shown
        # as a read-only judge section on the page, separate from the user's own review controls.
        try:
            jraw = json.loads(judge_data) if isinstance(judge_data, str) and judge_data.strip() else []
        except (ValueError, TypeError):
            jraw = []
        jmap = {}
        if isinstance(jraw, list):
            for v in jraw:
                idx = v.get("index") if isinstance(v, dict) else None
                if isinstance(idx, int) and not isinstance(idx, bool):
                    jmap[idx] = {
                        "score": v.get("score"),
                        "score_max": v.get("score_max"),
                        "tags": v.get("tags") if isinstance(v.get("tags"), list) else [],
                        "comment": v.get("comment") if isinstance(v.get("comment"), str) else "",
                        # Per-criterion sub-scores (Vision Judge multi-criteria mode); absent → {}.
                        "scores": v.get("scores") if isinstance(v.get("scores"), dict) else {},
                    }
        judge_list = [jmap.get(i) for i in range(n)]

        # Per-image `src`: relative "images/NNN.png" for the portable folder (embed off) — resolves
        # both offline (open the folder's index.html) and in-app (served under /image_compare_dir).
        # With embed on, pixels are inlined as base64. A missing file falls back to base64 so the
        # page never shows a broken picture.
        def src_of(p, rpth):
            if not embed_images and rpth and os.path.exists(rpth):
                return "images/" + urllib.parse.quote(os.path.basename(rpth))
            return _b64_png(p)

        # Data for the interactive page uses the ORIGINAL (clean) images.
        items = [{"src": src_of(p, rpth), "caption": c, "prompt": pr, "settings": st,
                  "time": tm, "time_seconds": ts, "metrics": mtr, "metric_ssim": ms, "metric_psnr": mp,
                  "judge": jv, "color": col, "band": band, "report_path": rpth, "settings_data": sd}
                 for p, c, pr, st, tm, ts, col, band, rpth, sd, mtr, ms, mp, jv in
                 zip(pils, caps, proms, setts, time_lines, time_secs, cap_colors, cap_bands,
                     report_paths, sdata, metrics_disp, m_ssim, m_psnr, judge_list)]
        cfg = {"title": title, "columns": int(columns), "show_prompts": bool(show_prompts),
               "show_settings": bool(show_settings), "run_id": run_id,
               "report_db": report_db_resolved, "items": items}

        try:
            with open(VIEWER_TEMPLATE, "r", encoding="utf-8") as f:
                template = f.read()
        except Exception as e:
            return self._err(f"viewer.html missing: {e}", pils)

        html = template.replace("/*__COMPARE_DATA__*/null", json.dumps(cfg, ensure_ascii=True))

        if embed_images:
            # Single self-contained .html (base64) in out_dir, served by the single-file route.
            _, path = _unique_html_path(out_dir, prefix)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            _register_html(path)
            q = urllib.parse.quote(os.path.abspath(path))
            url = f"http://127.0.0.1:{_server_port()}/image_compare?path={q}"
        else:
            # Light index.html inside the portable folder (relative image paths); served as a
            # folder under a token so those relative paths resolve in-app too.
            path = os.path.join(bundle_dir, "index.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            token = uuid.uuid4().hex[:12]
            _register_dir(token, bundle_dir)
            url = f"http://127.0.0.1:{_server_port()}/image_compare_dir/{token}/index.html"
        print(f"[ImageCompare] saved {path}\n[ImageCompare] open: {url}")

        if overlay_captions:
            cap_pils = [_overlay_caption(p, c, caption_position, font_size, col, band)
                        for p, c, col, band in zip(pils, caps, cap_colors, cap_bands)]
        else:
            cap_pils = [p.convert("RGB") for p in pils]
        # images_captioned is a list output — one frame per image, any sizes, no padding.
        out_tensor = _pils_to_tensor_list(cap_pils)

        if save_captioned_images or save_prompts_txt:
            export_dir = bundle_dir or out_dir     # keep exports inside the portable folder
            saved = 0
            for idx in range(n):
                _, base = first_free(export_dir, lambda k, i=idx: f"{prefix}_{i + 1:02d}_{k:04d}",
                                     [".png", ".txt"])
                png = os.path.join(export_dir, base + ".png")
                txt = os.path.join(export_dir, base + ".txt")
                if save_captioned_images:
                    cap_pils[idx].save(png)
                if save_prompts_txt:
                    with open(txt, "w", encoding="utf-8") as f:
                        f.write(proms[idx])
                saved += 1
            print(f"[ImageCompare] exported files for {saved} image(s) to {export_dir}")

        return {"ui": {"compare_url": [url], "compare_path": [path]},
                "result": (out_tensor, path, url)}

    def _err(self, msg, pils):
        print(f"[ImageCompare] ERROR: {msg}")
        # images_captioned is OUTPUT_IS_LIST -> output 0 must be a list (empty if no images).
        out = _pils_to_tensor_list([p.convert("RGB") for p in pils]) if pils else []
        return {"ui": {"compare_url": [""], "compare_path": [""]},
                "result": (out, f"[ERROR] {msg}", "")}


NODE_CLASS_MAPPINGS = {"ImageCompareHTML": ImageCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageCompareHTML": "Image Compare (HTML)"}
