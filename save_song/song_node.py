"""Save Song — save an AUDIO clip (with a chosen quality) plus an optional cover image and
lyrics, and preview it right on the node with a built-in player and the cover.

Required input is ``audio``; ``image`` (cover) and ``lyrics`` (text) are optional. The audio is
encoded with PyAV exactly like ComfyUI's own Save Audio node — pick **FLAC** (lossless) or an
**MP3 / Opus** bitrate via the ``quality`` dropdown. The cover is written as a JPEG next to the
audio and lyrics as a ``.txt``; all three share one counter-based base name
(``song_00001.flac`` / ``.jpg`` / ``.txt``) under ComfyUI's output folder.

The cover and the lyrics also go **inside** the audio file — an ID3v2.4 tag for MP3, Vorbis
comments for FLAC and Opus — because the files next to it are gone the moment the song is copied
onto a phone. See ``tagging.py``; the rest of the fields come from the **Song Tags** node, and the
title falls back to the file's own name so a player never shows a blank.

The node returns the ``audio`` passthrough and the saved ``path``, plus a ``ui`` payload the
frontend (web/save_song.js) turns into an ``<audio>`` player and the cover image.
"""
import os

from . import tagging

# quality label -> (container format, bitrate/quality token)
_QUALITY = {
    "FLAC (lossless)": ("flac", None),
    "MP3 320k": ("mp3", "320k"),
    "MP3 V0 (VBR)": ("mp3", "V0"),
    "MP3 128k": ("mp3", "128k"),
    "Opus 192k": ("opus", "192k"),
    "Opus 128k": ("opus", "128k"),
    "Opus 96k": ("opus", "96k"),
}
QUALITY_OPTIONS = list(_QUALITY.keys())

# Sample rates libopus accepts; anything else is resampled up to the next one (48k max).
_OPUS_RATES = [8000, 12000, 16000, 24000, 48000]


def _encode_audio(waveform, sample_rate, fmt, quality, tags=None, lyrics="", cover=None,
                  cover_size=None):
    """Encode one waveform ([C, T] float torch tensor) to `fmt`; return the file bytes.

    Mirrors ComfyUI's SaveAudio encoder (PyAV): interleaved float frames, libopus/libmp3lame/
    flac streams, and the Opus sample-rate fixup.

    `tags` / `lyrics` / `cover` are written into the file itself when given; leave them out and the
    bytes are exactly what this function has always returned. Tagging never costs you the song: if
    anything in it goes wrong it is reported and skipped, and the audio is still written.
    """
    import io
    import av

    if fmt == "opus":
        if sample_rate > 48000:
            new_sr = 48000
        elif sample_rate not in _OPUS_RATES:
            new_sr = next((r for r in sorted(_OPUS_RATES) if r > sample_rate), 48000)
        else:
            new_sr = sample_rate
        if new_sr != sample_rate:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sample_rate, new_sr)
            sample_rate = new_sr

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=fmt)
    layout = "mono" if waveform.shape[0] == 1 else "stereo"
    if fmt == "opus":
        stream = container.add_stream("libopus", rate=sample_rate, layout=layout)
        stream.bit_rate = {"64k": 64000, "96k": 96000, "128k": 128000,
                           "192k": 192000, "320k": 320000}.get(quality, 128000)
    elif fmt == "mp3":
        stream = container.add_stream("libmp3lame", rate=sample_rate, layout=layout)
        if quality == "V0":
            stream.codec_context.qscale = 1
        elif quality == "128k":
            stream.bit_rate = 128000
        elif quality == "320k":
            stream.bit_rate = 320000
    else:  # flac
        stream = container.add_stream("flac", rate=sample_rate, layout=layout)

    # Vorbis comments have to be in place before the first packet is muxed, because that is when the
    # headers are written. MP3 is handled after the fact instead — see below.
    if fmt != "mp3" and (tags or lyrics or cover):
        try:
            comments = tagging.vorbis_comments(tags, lyrics, cover, cover_size)
            # FLAC keeps them on the CONTAINER and Opus on the STREAM. Put Opus's on the container
            # and they vanish without a word — no error, just an untagged file.
            (stream.metadata if fmt == "opus" else container.metadata).update(comments)
        except Exception as e:
            print(f"[KinburgSaveSong] could not tag the {fmt}: {e}")

    frame = av.AudioFrame.from_ndarray(
        waveform.movedim(0, 1).reshape(1, -1).float().numpy(), format="flt", layout=layout)
    frame.sample_rate = sample_rate
    frame.pts = 0
    container.mux(stream.encode(frame))
    container.mux(stream.encode(None))
    container.close()
    raw = buf.getvalue()

    if fmt == "mp3" and (tags or lyrics or cover):
        try:
            tag = tagging.id3v2_tag(
                tags, lyrics, cover,
                encoder=f"Kinburg Save Song (libmp3lame {quality or 'default'})")
            # Replaces the muxer's own tag (one TSSE frame) rather than sitting in front of it: two
            # ID3 tags in one file is exactly the sort of thing a strict reader stops on.
            if tag:
                raw = tag + tagging.strip_id3(raw)
        except Exception as e:
            print(f"[KinburgSaveSong] could not tag the mp3: {e}")
    return raw


class KinburgSaveSong:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The song to save (required)."}),
                "filename_prefix": ("STRING", {"default": "songs/song", "tooltip": "Save path prefix under ComfyUI/output. A counter is appended; the cover (.jpg) and lyrics (.txt) share the same base name."}),
                "quality": (QUALITY_OPTIONS, {"default": "MP3 V0 (VBR)", "tooltip": "Audio format & quality. FLAC = lossless; MP3/Opus = compressed at the chosen bitrate."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional cover art — saved as a JPEG next to the audio, shown on the node, and embedded IN the audio file so it survives being copied off somewhere."}),
                "lyrics": ("STRING", {"forceInput": True, "tooltip": "Optional lyrics/text (input only) — saved as a .txt next to the audio and embedded in the file itself (ID3 'USLT' for MP3, a LYRICS comment for FLAC/Opus), which is the frame a phone scrolls along to. Wire a STRING here."}),
                "image_quality": ("INT", {"default": 90, "min": 1, "max": 100, "tooltip": "JPEG quality for the cover (higher = larger, better). The same JPEG is written next to the song and embedded in it, so this is also the cover's weight inside the file."}),
                "tags": ("SONG_TAGS", {"tooltip": "Artist, album, year, genre and anything else — wire a 'Song Tags' node. Without one the file still gets its title (the file's own name), plus the cover and lyrics wired above."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Kinburg-Nodes/audio"

    def run(self, audio, filename_prefix="songs/song", quality="MP3 V0 (VBR)",
            image=None, lyrics="", image_quality=90, tags=None):
        import folder_paths

        if audio is None or "waveform" not in audio:
            raise ValueError("Save Song: 'audio' is empty (nothing to save).")

        fmt, q = _QUALITY.get(quality, ("flac", None))
        out_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir)
        base = f"{filename}_{counter:05}"

        # AUDIO waveform is [B, C, T]; save the first item.
        wf = audio["waveform"]
        w0 = (wf[0] if wf.dim() == 3 else wf).cpu()
        text = lyrics or ""

        # The cover is encoded BEFORE the audio, because the same JPEG bytes go into the file's tag
        # and onto disk next to it — one encode, one picture, no chance of the two disagreeing.
        cover, cover_size = None, None
        if image is not None:
            try:
                cover, cover_size = _cover_jpeg(image, int(image_quality))
            except Exception as e:
                print(f"[KinburgSaveSong] cover encode failed: {e}")

        meta = dict(tags or {})
        # A song with no title shows up as nothing at all in a player, so the file's own name — the
        # thing you would have used to find it anyway — is a better default than blank.
        meta["title"] = str(meta.get("title") or "").strip() or base

        audio_name = f"{base}.{fmt}"
        audio_path = os.path.join(full_output_folder, audio_name)
        with open(audio_path, "wb") as f:
            f.write(_encode_audio(w0, int(audio["sample_rate"]), fmt, q, tags=meta, lyrics=text,
                                  cover=cover, cover_size=cover_size))

        # Standard UI keys so ComfyUI renders a native player / cover preview AND registers the
        # files in Media Assets (the frontend keys off "audio"/"images", not a custom key).
        ui = {"audio": [{"filename": audio_name, "subfolder": subfolder, "type": "output"}]}

        if text.strip():
            with open(os.path.join(full_output_folder, base + ".txt"), "w",
                      encoding="utf-8", newline="\n") as f:
                f.write(text)

        if cover is not None:
            try:
                img_name = base + ".jpg"
                with open(os.path.join(full_output_folder, img_name), "wb") as f:
                    f.write(cover)
                ui["images"] = [{"filename": img_name, "subfolder": subfolder, "type": "output"}]
            except Exception as e:
                print(f"[KinburgSaveSong] cover save failed: {e}")

        return {"ui": ui, "result": (audio, audio_path)}


def _cover_jpeg(image, quality):
    """First frame of an IMAGE batch as JPEG bytes, plus its size for the picture block."""
    import io

    import numpy as np
    from PIL import Image
    arr = image[0].detach().cpu().numpy()  # [B,H,W,C] -> first frame [H,W,C]
    pil = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), pil.size


NODE_CLASS_MAPPINGS = {"KinburgSaveSong": KinburgSaveSong}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSaveSong": "Save Song"}
