"""Collage — arrange images into a grid on a solid color or background image.

Builds a single collage IMAGE from either a wired `input_images` batch or, when none is
connected, every image in `folder_path` (natural-sorted). Each image is fit into its grid
cell with its aspect ratio preserved (letterboxed); the letterbox border is filled with the
image's own top-left pixel color so it blends into a matching background. `cols`, `gap` and
`margin` lay out the grid; the cell size is derived so the whole grid fits the output canvas.
"""
import os
import re

# Heavy / ComfyUI-only imports are guarded so the package still imports (and the Registry
# can enumerate nodes) without ComfyUI present. At runtime these are always available.
try:
    import torch
    import numpy as np
    from PIL import Image
except Exception:
    pass

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _hex_rgb(value, fallback=(255, 255, 255)):
    """'#RRGGBB' / '#RGB' (with or without '#') -> (r, g, b) 0..255; fallback if invalid."""
    s = str(value).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return fallback
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _tensor_to_pils(image):
    """ComfyUI IMAGE tensor [B,H,W,C] float 0..1 -> list of PIL.Image (RGB)."""
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:  # a bare [H,W,C] frame -> add the batch dim
        arr = arr[None, ...]
    return [Image.fromarray((np.clip(im, 0.0, 1.0) * 255.0).astype(np.uint8)).convert("RGB")
            for im in arr]


def _fit_cell(img, cell_w, cell_h, default_fill):
    """Fit `img` into cell_w x cell_h preserving aspect ratio, centered. The letterbox border
    is filled with the image's own top-left pixel color, falling back to `default_fill`."""
    iw, ih = img.size
    scale = min(cell_w / iw, cell_h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    try:
        fill = img.getpixel((0, 0))
        fill = (fill, fill, fill) if isinstance(fill, int) else tuple(fill[:3])
    except Exception:
        fill = default_fill
    cell = Image.new("RGB", (cell_w, cell_h), fill)
    cell.paste(resized, ((cell_w - nw) // 2, (cell_h - nh) // 2))
    return cell


class CollageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_width": ("INT", {"default": 2480, "min": 1, "max": 16384, "tooltip": "Width of the output collage, in px."}),
                "output_height": ("INT", {"default": 3508, "min": 1, "max": 16384, "tooltip": "Height of the output collage, in px."}),
                "cols": ("INT", {"default": 3, "min": 1, "max": 100, "tooltip": "Number of columns; the row count is derived from the image count."}),
                "gap": ("INT", {"default": 10, "min": 0, "max": 2000, "tooltip": "Spacing between cells, in px."}),
                "margin": ("INT", {"default": 20, "min": 0, "max": 2000, "tooltip": "Outer margin around the grid, in px."}),
                "background_color": ("STRING", {"default": "#FFFFFF", "tooltip": "Canvas color as HEX (#RRGGBB), used when no background_image is connected."}),
            },
            "optional": {
                "input_images": ("IMAGE", {"tooltip": "Images to arrange. If not connected, images are loaded from folder_path instead."}),
                "background_image": ("IMAGE", {"tooltip": "Optional background, stretched to the output size. Overrides background_color (first frame only)."}),
                "folder_path": ("STRING", {"default": "", "tooltip": "Fallback source: load every image in this folder (natural-sorted) when input_images isn't connected."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_collage"
    CATEGORY = "Kinburg-Nodes/image"

    def create_collage(self, output_width, output_height, cols, gap, margin,
                        background_color, input_images=None, background_image=None, folder_path=""):
        bg_rgb = _hex_rgb(background_color)

        # Background: a connected image (stretched to the canvas) wins over the solid color.
        if background_image is not None:
            canvas = _tensor_to_pils(background_image)[0].resize(
                (output_width, output_height), Image.Resampling.LANCZOS)
        else:
            canvas = Image.new("RGB", (output_width, output_height), bg_rgb)

        # Source: the wired batch, else the folder fallback.
        images = _tensor_to_pils(input_images) if input_images is not None else self._load_folder(folder_path)
        if not images:  # empty batch — return the bare background rather than crashing
            return (self._to_tensor(canvas),)

        # Grid geometry: cell size derived so every row fits the content area. The cell
        # dimensions are clamped to >= 1 so an over-large margin/gap can't make them negative.
        num = len(images)
        rows = (num + cols - 1) // cols
        content_w = output_width - 2 * margin
        content_h = output_height - 2 * margin
        cell_w = max(1, (content_w - (cols - 1) * gap) // cols)
        cell_h = max(1, (content_h - (rows - 1) * gap) // rows)

        for i, img in enumerate(images):
            cell = _fit_cell(img, cell_w, cell_h, bg_rgb)
            r, c = divmod(i, cols)
            canvas.paste(cell, (margin + c * (cell_w + gap), margin + r * (cell_h + gap)))

        return (self._to_tensor(canvas),)

    @staticmethod
    def _to_tensor(pil):
        """PIL.Image -> ComfyUI IMAGE tensor [1,H,W,C] float 0..1."""
        arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    @staticmethod
    def _load_folder(folder_path):
        if not folder_path or not os.path.isdir(folder_path):
            raise ValueError(f"Collage: no input_images connected and folder_path '{folder_path}' is not a folder.")
        pils = []
        for name in sorted(os.listdir(folder_path), key=_natural_key):
            p = os.path.join(folder_path, name)
            if os.path.isfile(p) and name.lower().endswith(_IMG_EXTS):
                try:
                    pils.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    print(f"[Collage] failed to load {p}: {e}")
        if not pils:
            raise ValueError(f"Collage: no images ({', '.join(_IMG_EXTS)}) found in '{folder_path}'.")
        return pils


NODE_CLASS_MAPPINGS = {"CustomCollageNode": CollageNode}
NODE_DISPLAY_NAME_MAPPINGS = {"CustomCollageNode": "Collage"}
