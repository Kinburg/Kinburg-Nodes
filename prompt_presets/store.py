"""Prompt-preset store — built-in presets + a user-editable JSON layer on disk.

Five categories (camera / aesthetics / light / medium / background). Each is an ordered
dict of {preset name -> prompt fragment}. The built-ins live here in code; the user's own
presets and saved "setups" (named combinations of all five dropdowns) are persisted to
``data/store.json`` next to this file and merged on top of the built-ins.

Guarded so the package still imports without ComfyUI present (registry scan, tests).
"""
import json
import os
import threading

# The empty option every dropdown carries; resolves to "".
NONE = "🚫 None"

# Category order is the order the dropdowns / outputs appear in.
DEFAULTS = {
    "camera": {
        "Close-up Portrait": "close-up portrait, 85mm lens, shallow depth of field, eye-level",
        "Wide Establishing": "wide-angle establishing shot, 24mm lens, deep depth of field",
        "Cinematic Anamorphic": "cinematic anamorphic shot, shallow depth of field, subtle lens flare",
        "Extreme Macro": "extreme macro shot, 100mm macro lens, razor-thin depth of field, fine detail",
        "Low Angle Hero": "dramatic low-angle hero shot, looking up at the subject, 35mm lens",
        "High Angle Top-Down": "high-angle top-down shot, bird's-eye view",
        "Dutch Angle": "dynamic dutch angle, tilted frame, sense of tension",
        "Aerial Drone": "aerial drone shot, high altitude, sweeping perspective",
        "Fisheye": "fisheye lens, 8mm, heavy barrel distortion, ultra-wide field of view",
        "35mm Film Still": "35mm film still, natural perspective, candid framing",
    },
    "aesthetics": {
        "Photorealistic": "photorealistic, hyper-detailed, true-to-life color, high dynamic range",
        "Cinematic": "cinematic, filmic color grading, dramatic composition, moody atmosphere",
        "Anime": "anime style, cel shading, clean linework, vibrant colors",
        "Oil Painting": "oil painting, visible brushstrokes, rich texture, classical composition",
        "Watercolor": "watercolor painting, soft washes, bleeding pigments, paper texture",
        "Cyberpunk": "cyberpunk aesthetic, neon-soaked, high-tech low-life, futuristic grime",
        "Vaporwave": "vaporwave aesthetic, pastel neon, retro 80s, glitch, dreamy",
        "Minimalist": "minimalist, clean negative space, simple composition, muted palette",
        "Baroque": "baroque, ornate, dramatic chiaroscuro, opulent detail",
        "3D Render": "octane render, 3d, physically based rendering, subsurface scattering, ray tracing",
        "Comic Ink": "comic book ink style, bold outlines, halftone shading, dynamic",
        "Surreal": "surreal, dreamlike, impossible juxtapositions, otherworldly",
    },
    "light": {
        "Golden Hour": "golden hour lighting, warm soft sunlight, long shadows",
        "Blue Hour": "blue hour, cool twilight glow, soft ambient light",
        "Studio Softbox": "studio lighting, large softbox, soft even illumination, clean",
        "Rembrandt": "Rembrandt lighting, dramatic side light, defined shadow triangle on the cheek",
        "Rim / Backlight": "rim lighting, strong backlight, glowing edges, near silhouette",
        "Neon Glow": "neon lighting, colorful glow, reflections, night ambiance",
        "Hard Noon Sun": "harsh midday sun, hard shadows, high contrast",
        "Candlelight": "warm candlelight, flickering glow, intimate, low key",
        "Overcast Diffused": "overcast soft diffused light, even, low contrast",
        "Volumetric Rays": "volumetric lighting, god rays, atmospheric haze, light shafts",
        "Moonlight": "moonlight, cool blue tones, soft nighttime glow",
    },
    "medium": {
        "Digital Photograph": "digital photograph, DSLR, sharp focus, high resolution",
        "35mm Film": "shot on 35mm film, natural film grain, analog color",
        "Polaroid": "polaroid photo, faded colors, soft focus, vintage border",
        "Oil on Canvas": "oil on canvas, textured brushwork",
        "Pencil Sketch": "graphite pencil sketch, hatching, paper grain",
        "Charcoal Drawing": "charcoal drawing, smudged shading, expressive marks",
        "Digital Illustration": "digital illustration, clean rendering, painterly finish",
        "3D Sculpt": "3d sculpt, zbrush, detailed geometry, clay render",
        "Vector Art": "flat vector art, clean shapes, bold colors",
        "Stained Glass": "stained glass artwork, leaded segments, luminous colors",
        "Claymation": "claymation, stop-motion, handmade clay texture",
        "Pixel Art": "pixel art, 16-bit, limited palette, crisp pixels",
    },
    "background": {
        "Studio Seamless": "plain seamless studio backdrop, solid color, clean",
        "Lush Forest": "lush green forest, dense foliage, dappled light",
        "Urban Street": "busy urban street, city buildings, bokeh lights",
        "Mountain Vista": "majestic mountain range, distant peaks, expansive sky",
        "Beach Sunset": "tropical beach at sunset, ocean waves, warm sky",
        "Outer Space": "deep space, stars, nebula, cosmic",
        "Cozy Interior": "cozy interior, warm furnishings, soft window light",
        "Cyberpunk City": "futuristic cyberpunk city, neon signs, rain-slick streets",
        "Desert Dunes": "vast desert, rolling sand dunes, clear sky",
        "Abstract Gradient": "abstract gradient background, smooth color blend, soft bokeh",
        "Snowy Landscape": "snowy winter landscape, soft snowfall, muted tones",
        "Solid Black": "solid black background, isolated subject",
    },
}

CATEGORIES = list(DEFAULTS.keys())

_LOCK = threading.Lock()


def _store_path():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", "store.json")


def _load_user():
    """The user layer as {"categories": {...}, "setups": {...}}; {} when absent/corrupt."""
    try:
        with open(_store_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    cats = data.get("categories") if isinstance(data.get("categories"), dict) else {}
    setups = data.get("setups") if isinstance(data.get("setups"), dict) else {}
    return {"categories": cats, "setups": setups}


def _save_user(data):
    p = _store_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def categories_merged():
    """Built-in presets with the user's own presets layered on top, per category."""
    user = _load_user()["categories"]
    out = {}
    for cat, presets in DEFAULTS.items():
        merged = dict(presets)
        for name, text in (user.get(cat) or {}).items():
            if isinstance(name, str) and isinstance(text, str):
                merged[name] = text
        out[cat] = merged
    return out


def options_for(cat):
    """Dropdown values for a category: NONE first, then preset names (built-in + user)."""
    return [NONE] + list(categories_merged().get(cat, {}).keys())


def resolve(cat, name):
    """A selected preset name -> its prompt fragment ("" for NONE / unknown)."""
    if not name or name == NONE:
        return ""
    return categories_merged().get(cat, {}).get(name, "")


def full_data():
    """Everything the frontend needs: merged presets, which names are built-in, and setups."""
    user = _load_user()
    return {
        "none": NONE,
        "order": CATEGORIES,
        "categories": categories_merged(),
        "builtins": {cat: list(p.keys()) for cat, p in DEFAULTS.items()},
        "setups": user["setups"],
    }


def upsert_preset(cat, name, text, delete=False):
    """Add/update (or delete) a user preset. Built-in names can't be deleted."""
    if cat not in DEFAULTS:
        raise ValueError(f"unknown category: {cat}")
    name = (name or "").strip()
    if not name or name == NONE:
        raise ValueError("preset name is required")
    with _LOCK:
        data = _load_user()
        cat_user = data["categories"].setdefault(cat, {})
        if delete:
            if name in DEFAULTS[cat] and name not in cat_user:
                raise ValueError("cannot delete a built-in preset")
            cat_user.pop(name, None)
            if not cat_user:
                data["categories"].pop(cat, None)
        else:
            cat_user[name] = text or ""
        _save_user(data)
    return full_data()


def upsert_setup(name, values, delete=False):
    """Save/delete a named setup (a chosen preset name per category)."""
    name = (name or "").strip()
    if not name:
        raise ValueError("setup name is required")
    with _LOCK:
        data = _load_user()
        if delete:
            data["setups"].pop(name, None)
        else:
            values = values or {}
            data["setups"][name] = {c: (values.get(c) or NONE) for c in DEFAULTS}
        _save_user(data)
    return full_data()
