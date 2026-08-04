"""Siren Compare — a page for listening to several takes against each other.

Image Compare answers "which of these looks better". This is the same idea for music, and it is a
separate node rather than a mode of that one because almost nothing carries over: there is no SSIM
for a song, captions are not burnt into anything, and the whole interaction is a transport — play,
switch, loop — which an image grid has no notion of.

What *is* shared is the delivery: a portable folder (audio + data + index.html with relative links)
registered under a token and served by Image Compare's existing `/image_compare_dir/` route, which
already streams arbitrary files with Range support — exactly what audio seeking needs — and opens
offline just the same. It also shares the accumulator idiom, the `captions` / `prompts` string
conventions and Generation Info's `GEN_SETTINGS`, so the nodes you already wire up work here.

**This node ships data, not pictures.** It writes each take's mel spectrogram as a 16-bit dB matrix
packed into a PNG and lets the page do the drawing. Pre-rendering finished images instead would mean
one file per combination of mode x colour map x dB floor x channel — measured at 288 renders, about
a minute and 119 MB *per track* — and would still leave the continuous controls (a dB slider, a diff
span) and live A/B diffs impossible. The matrix is ~580 kB, less than two finished pictures, and
covers all of it: recolouring, re-windowing and diffing any pair are then plain array arithmetic in
the browser. The FFT stays here, computed once, so there is no second implementation to drift.
"""
import json
import os
import uuid

import torch

from ..image_compare.compare_node import (
    _first, _register_dir, _register_html, _server_port, _time_to_seconds, _unique_html_path,
)
from .scope import _metrics, db_matrix_png

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_TEMPLATE = os.path.join(NODE_DIR, "player.html")
FORMATS = {
    "FLAC (lossless)": ("flac", ""),
    "MP3 V0": ("mp3", "V0"),
    "Opus 192k": ("opus", "192k"),
}
MIME = {"flac": "audio/flac", "mp3": "audio/mpeg", "opus": "audio/ogg"}


def _data_uri(raw, mime):
    import base64
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")


def _blocks(text, n, allow_lines=True):
    """One free-text block per track.

    A `---` line always means "these are separate blocks". What happens WITHOUT one depends on the
    field, and the difference is not cosmetic:

    * `allow_lines=True` (notes): if the line count happens to match the track count, treat one line
      as one track. That is the Get Accumulator (captions) convention, and captions are one-liners.
    * `allow_lines=False` (lyrics): never split on newlines. Lyrics are inherently multi-line, so a
      two-line lyric next to two takes would otherwise be torn in half and half of it hung on the
      wrong track — silently, and looking plausible. No separator means one lyric shared by all.

    Short input is padded and long input truncated, so a mismatch can never shift a block onto the
    wrong take."""
    raw = (text or "").replace("\r\n", "\n").strip("\n")
    if not raw.strip():
        return [""] * n
    if "\n---\n" in raw:
        parts = [p.strip("\n") for p in raw.split("\n---\n")]
    else:
        lines = raw.split("\n")
        parts = lines if (allow_lines and n > 1 and len(lines) == n) else [raw] * n
    return (parts + [""] * n)[:n]


def _lines_for(text, n, fallback=""):
    """One line per take, in order — the Get Accumulator (captions) contract. Missing lines get
    `fallback` rather than shifting the remaining ones onto the wrong take."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return [(lines[i] if i < len(lines) else fallback) for i in range(n)]


def _labels_for(text, n):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return [(lines[i] if i < len(lines) else f"Take {i + 1}") for i in range(n)]


def _settings_for(raw, n):
    """Generation Info Filter's `settings_data`: a JSON list of per-item [{key, value}, …]."""
    try:
        data = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
    except (ValueError, TypeError):
        data = []
    if not isinstance(data, list):
        data = []
    out = []
    for i in range(n):
        fields = data[i] if i < len(data) and isinstance(data[i], list) else []
        row = {}
        for f in fields:
            if isinstance(f, dict) and f.get("key") is not None:
                v = f.get("value", "")
                v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                row[str(f["key"])] = " ".join(v.split())[:300]
        out.append(row)
    return out


def _rms_db(wave):
    r = float(wave.pow(2).mean().sqrt())
    return 20.0 * float(torch.log10(torch.tensor(max(r, 1e-10))))


class KinburgSirenCompare:
    """Several takes on one page: synchronised transport, instant solo, live scopes, measurements."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audios": ("AUDIO", {"tooltip": "The takes to compare. Wire a 'Get Accumulator (audio)' here and drop a 'Set Accumulator (audio)' on each branch — same idiom as Image Compare, and the Collect button wires them in index order. A single AUDIO works too."}),
                "title": ("STRING", {"default": "Siren comparison"}),
                "filename_prefix": ("STRING", {"default": "siren_compare"}),
                "audio_format": (list(FORMATS.keys()), {"default": "FLAC (lossless)", "tooltip": "How the takes are written into the bundle.\n\nFLAC is the default and should stay the default: you are comparing fine detail, and a lossy codec would add differences of its own on top of the ones you are trying to hear. MP3/Opus only to make a bundle small enough to send someone."}),
                "scope_columns": ("INT", {"default": 0, "min": 0, "max": 12000, "tooltip": "Time resolution of the shipped spectrogram data, in columns. 0 = auto, which is what you want.\n\nAuto gives 25 columns per second — one per AceStep latent frame, 40 ms — clamped to 1200…6000. That is finer than the screen so there is real detail to find when you zoom in, without making the bundle silly: a 3-minute track comes to about 1.6 MB.\n\nThis is the only scope setting that has to be decided here, because changing it means recomputing the spectrogram. Colours, dB window, mode, channel, zoom and the diff are all live on the page."}),
                "auto_collect": ("BOOLEAN", {"default": True, "tooltip": "Re-wire every Get Accumulator in the graph automatically, right before the workflow is queued — so a Set you just added, removed, muted or bypassed is picked up without a click. Off: only the '🔌 Collect All' button collects. (Purely an editor convenience; the backend ignores this value.)"}),
                "self_contained": ("BOOLEAN", {"default": False, "tooltip": "How the comparison is saved.\n\nOFF (default): a portable FOLDER — a light index.html plus audio/ and scopes/ with relative links. Small, quick to write, and what you want for opening it in-app.\n\nON: ONE .html with every take and every spectrogram inlined as a data: URI. Bigger (base64 adds a third), but it is the only version that works when somebody OPENS IT STRAIGHT OFF DISK — a browser refuses to fetch local media from a file:// page, so a zipped folder plays nothing and draws nothing. Turn this on to send a comparison to someone, and pick MP3 or Opus first unless the point is lossless: a 3-minute FLAC take inlines to about 40 MB."}),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300, "tooltip": "Starting value for the page's bar/beat grid — you can change it there without re-running. 0 = a plain time ruler."}),
                "beats_per_bar": ("INT", {"default": 4, "min": 1, "max": 16}),
                "n_fft": ("INT", {"default": 2048, "min": 256, "max": 16384, "advanced": True, "tooltip": "FFT window for the shipped spectrogram data. Larger = finer frequency detail, blurrier timing."}),
                "n_mels": ("INT", {"default": 192, "min": 16, "max": 1024, "advanced": True, "tooltip": "Mel bands in the shipped data. 192 is the most that stays fully populated at n_fft 2048 — ask for 256 and two bands come out empty, i.e. dead rows in the picture."}),
            },
            "optional": {
                "labels": ("STRING", {"forceInput": True, "tooltip": "One label per line, in track order — wire a 'Get Accumulator (captions)', which already joins with newlines. Missing labels become 'Take N'."}),
                "notes": ("STRING", {"forceInput": True, "tooltip": "Free text shown per track: a Siren report, a Generation Info dump, your own remarks. Takes either convention — blocks split by a '---' line (Get Accumulator (prompts)) or one line per track (captions). A single paragraph is shared by every take."}),
                "times": ("STRING", {"forceInput": True, "tooltip": "Generation time per take, ONE LINE EACH in track order — wire Siren (Music Sampler)'s 'time' output through a Set/Get Accumulator (texts) with a newline separator, exactly as Image Compare takes its times.\n\nShown next to each take's name and in the Measurements tab, where a 'vs fastest' row turns it into the question you actually have: what did the extra stage cost. '12.34 s', '1m 30s', '890 ms' and '00:01:30' all parse."}),
                "lyrics": ("STRING", {"forceInput": True, "tooltip": "Song lyrics for the page's side panel. ONE block (no '---') = the same words for every take, which is the usual case when you are comparing settings. Split it with '---' lines when the takes were sung from different lyrics."}),
                "settings_data": ("GEN_SETTINGS", {"tooltip": "Structured per-take settings from Generation Info Filter's 'settings_data' — the same output Image Compare takes. Becomes the page's Settings tab, with a 'differences only' view that shows exactly which setting moved between takes."}),
                "output_dir": ("STRING", {"default": "", "tooltip": "Where the bundle folder is written. Empty = the ComfyUI output folder."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("html_path", "url")
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/audio"
    OUTPUT_NODE = True
    DESCRIPTION = ("A page for comparing takes BY EAR. Every track is decoded into one Web Audio "
                   "clock, so they play in perfect sync and soloing between them mid-phrase is "
                   "instant and silent. Click a scope to seek, shift-drag to loop a bar, 'match "
                   "level' trims them to equal RMS so the louder one doesn't simply win, 'blind' "
                   "hides the labels. The spectrograms ship as DATA, so colours, dB window, channel "
                   "and A/B diff are all live on the page — no re-running to change the view. "
                   "Measurements, Settings and Notes tabs each have a 'differences only' mode, and "
                   "the lyrics sit in a side panel. Writes a portable folder that opens offline.")

    def run(self, audios, title, filename_prefix, audio_format, scope_columns, auto_collect,
            self_contained, bpm, beats_per_bar, n_fft, n_mels, labels=None, notes=None,
            times=None, lyrics=None, settings_data=None, output_dir=None):
        import folder_paths
        from ..save_song.song_node import _encode_audio

        # INPUT_IS_LIST wraps every input; `audios` is the one that genuinely is a list.
        title = _first(title, "Siren comparison")
        prefix = _first(filename_prefix, "siren_compare") or "siren_compare"
        audio_format = _first(audio_format, "FLAC (lossless)")
        cols = int(_first(scope_columns, 0))
        bpm = int(_first(bpm, 0))
        beats_per_bar = int(_first(beats_per_bar, 4))
        n_fft = int(_first(n_fft, 2048))
        n_mels = int(_first(n_mels, 192))
        labels = _first(labels, "") or ""
        notes = _first(notes, "") or ""
        times = _first(times, "") or ""
        lyrics = _first(lyrics, "") or ""
        settings_raw = _first(settings_data, "") or ""
        embed = bool(_first(self_contained, False))
        out_root = (_first(output_dir, "") or "").strip() or folder_paths.get_output_directory()

        takes = [a for a in (audios if isinstance(audios, list) else [audios])
                 if isinstance(a, dict) and a.get("waveform") is not None]
        if not takes:
            return self._err("nothing wired — connect an AUDIO (or a Get Accumulator (audio)).")

        n = len(takes)
        names = _labels_for(labels, n)
        time_lines = _lines_for(times, n)
        time_secs = [_time_to_seconds(t) for t in time_lines]
        note_blocks = _blocks(notes, n)
        lyric_blocks = _blocks(lyrics, n, allow_lines=False)
        settings_rows = _settings_for(settings_raw, n)
        fmt, quality = FORMATS.get(audio_format, ("flac", ""))

        bundle = os.path.join(out_root, f"{prefix}_{_stamp()}")
        os.makedirs(os.path.join(bundle, "audio"), exist_ok=True)
        os.makedirs(os.path.join(bundle, "scopes"), exist_ok=True)

        waves = []
        for a in takes:
            w = a["waveform"].detach().cpu().float()
            if w.ndim == 2:
                w = w.unsqueeze(0)
            waves.append((w, int(a.get("sample_rate", 44100))))
        # ONE time base for every matrix, so column x is the same instant on all of them.
        max_dur = max(w.shape[-1] / float(sr) for w, sr in waves)
        if cols <= 0:
            # 25 columns per second — one per AceStep latent frame (40 ms). Finer than any screen,
            # so zooming in finds real detail rather than interpolation, and still modest on disk.
            cols = int(min(6000, max(1200, round(max_dur * 25))))

        entries, warnings, total_bytes = [], [], 0
        meta = None
        for i, (wave, sr) in enumerate(waves):
            dur = wave.shape[-1] / float(sr)
            mono = wave[0].mean(dim=0) if wave.shape[1] > 1 else wave[0, 0]
            chans = {"mono": mono}
            if wave.shape[1] > 1:
                chans["left"], chans["right"] = wave[0, 0], wave[0, 1]
            mats = {}
            for cname, sig in chans.items():
                png, meta = db_matrix_png(sig, sr, cols, n_fft, n_mels, duration=max_dur)
                rel = f"scopes/{i:03d}_{cname}.png"
                path_png = os.path.join(bundle, rel)
                png.save(path_png, optimize=True)
                if embed:
                    # A data: URI is the one form a file:// page can still read: it is same-origin,
                    # so it neither trips CORS on fetch nor taints the canvas that getImageData
                    # needs. Relative paths do both, which is why a zipped folder shows nothing.
                    with open(path_png, "rb") as f:
                        mats[cname] = _data_uri(f.read(), "image/png")
                else:
                    mats[cname] = rel

            rel_audio = f"audio/{i:03d}.{fmt}"
            try:
                data = _encode_audio(wave[0], sr, fmt, quality)
            except Exception as e:
                warnings.append(f"track {i + 1} ({names[i]}) could not be encoded as {fmt}: {e}")
                continue
            with open(os.path.join(bundle, rel_audio), "wb") as f:
                f.write(data)
            total_bytes += len(data)
            src_audio = _data_uri(data, MIME.get(fmt, "audio/flac")) if embed else rel_audio

            entries.append({
                "label": names[i], "file": src_audio, "matrices": mats,
                "duration": round(dur, 3), "rms_db": round(_rms_db(wave), 2),
                "metrics": _metrics(wave, sr), "notes": note_blocks[i],
                "lyrics": lyric_blocks[i], "settings": settings_rows[i], "gain_db": 0.0,
                "time": time_lines[i], "time_seconds": time_secs[i],
            })

        if not entries:
            return self._err("every track failed to encode — see the warnings above.")

        # Level matching trims DOWN to the quietest take rather than boosting up to the loudest:
        # with several unmuted at once the sum would otherwise clip the output.
        target = min(e["rms_db"] for e in entries)
        for e in entries:
            e["match_db"] = round(target - e["rms_db"], 2)

        # Generation time rides in the metrics dict so it lands in the Measurements table and picks
        # up 'differences only' and blind mode for free. 'vs fastest' is the question actually being
        # asked when you compare sampler settings: what did the extra stage cost.
        known = [e["time_seconds"] for e in entries if isinstance(e["time_seconds"], (int, float))]
        fastest = min(known) if known else None
        for e in entries:
            if not e["time"]:
                continue
            e["metrics"]["generation time"] = e["time"]
            s = e["time_seconds"]
            if fastest and isinstance(s, (int, float)) and len(known) > 1:
                e["metrics"]["time vs fastest"] = (
                    "fastest" if abs(s - fastest) < 1e-9 else f"+{(s / fastest - 1) * 100:.0f}%")

        durs = [e["duration"] for e in entries]
        if max(durs) - min(durs) > 0.05:
            warnings.append(f"the takes are not the same length ({min(durs):.2f}-{max(durs):.2f} s)"
                            f" — everything is laid out on the longest, so the shorter ones simply "
                            f"stop early")
        mem_mb = sum(d * 2 * 4 * 48000 for d in durs) / (1024 * 1024)
        if mem_mb > 700:
            warnings.append(f"these tracks decode to roughly {mem_mb:.0f} MB in the browser — that "
                            f"is a lot for one page; drop a take or shorten them if it struggles")

        cfg = {"title": title, "tracks": entries, "duration": round(max(durs), 3),
               "bpm": bpm, "beats_per_bar": beats_per_bar,
               "matrix": meta or {"bands": n_mels, "cols": cols, "db_min": -120.0, "db_max": 0.0},
               "footer": ("\n".join("⚠ " + w for w in warnings) if warnings else "")}
        try:
            with open(PLAYER_TEMPLATE, "r", encoding="utf-8") as f:
                template = f.read()
        except Exception as e:
            return self._err(f"player.html missing: {e}")
        html = template.replace("/*__SIREN_DATA__*/null", json.dumps(cfg, ensure_ascii=True))

        if embed:
            # One file, everything inlined — the only version that survives being zipped, mailed and
            # double-clicked. The folder is still written alongside it, so nothing is lost.
            import urllib.parse
            _, path = _unique_html_path(out_root, prefix)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            _register_html(path)
            url = (f"http://127.0.0.1:{_server_port()}/image_compare"
                   f"?path={urllib.parse.quote(os.path.abspath(path))}")
            size = os.path.getsize(path) / 1048576
            if size > 120:
                warnings.append(f"the self-contained page is {size:.0f} MB — fine to open, awkward "
                                f"to send. Re-run with MP3 or Opus if it is going to somebody.")
            print(f"[Siren Compare] self-contained: {size:.1f} MB → {path}")
        else:
            path = os.path.join(bundle, "index.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            token = uuid.uuid4().hex[:12]
            _register_dir(token, bundle)
            url = f"http://127.0.0.1:{_server_port()}/image_compare_dir/{token}/index.html"
        print(f"[Siren Compare] {len(entries)} track(s), {total_bytes / 1048576:.1f} MB of {fmt} + "
              f"{n_mels}x{cols} matrices → {bundle}")
        for w in warnings:
            print(f"[Siren Compare] ⚠ {w}")
        print(f"[Siren Compare] open: {url}")
        return {"ui": {"compare_url": [url]}, "result": (path, url)}

    def _err(self, msg):
        print(f"[Siren Compare] {msg}")
        return {"ui": {"compare_url": [""]}, "result": (f"[ERROR] [Siren Compare] {msg}", "")}


def _stamp():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


NODE_CLASS_MAPPINGS = {"KinburgSirenCompare": KinburgSirenCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSirenCompare": "Siren Compare (Audio) 🧜"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
