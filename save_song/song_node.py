"""Save Song — save an AUDIO clip (with a chosen quality) plus an optional cover image and
lyrics, and preview it right on the node with a built-in player and the cover.

Required input is ``audio``; ``image`` (cover) and ``lyrics`` (text) are optional. The audio is
encoded with PyAV exactly like ComfyUI's own Save Audio node — pick **FLAC** (lossless) or an
**MP3 / Opus** bitrate via the ``quality`` dropdown. The cover is written as a JPEG next to the
audio; lyrics are written as a ``.txt`` next to it and embedded in the audio file's metadata.
All three share one counter-based base name (``song_00001.flac`` / ``.jpg`` / ``.txt``) under
ComfyUI's output folder.

The node returns the ``audio`` passthrough and the saved ``path``, plus a ``ui`` payload the
frontend (web/save_song.js) turns into an ``<audio>`` player and the cover image.
"""
import os

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


def _encode_audio(waveform, sample_rate, fmt, quality):
    """Encode one waveform ([C, T] float torch tensor) to `fmt`; return the file bytes.

    Mirrors ComfyUI's SaveAudio encoder (PyAV): interleaved float frames, libopus/libmp3lame/
    flac streams, and the Opus sample-rate fixup.
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

    frame = av.AudioFrame.from_ndarray(
        waveform.movedim(0, 1).reshape(1, -1).float().numpy(), format="flt", layout=layout)
    frame.sample_rate = sample_rate
    frame.pts = 0
    container.mux(stream.encode(frame))
    container.mux(stream.encode(None))
    container.close()
    buf.seek(0)
    return buf.getvalue()


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
                "image": ("IMAGE", {"tooltip": "Optional cover art — saved as JPEG next to the audio and shown on the node."}),
                "lyrics": ("STRING", {"forceInput": True, "tooltip": "Optional lyrics/text (input only) — saved as a .txt next to the audio. Wire a STRING here."}),
                "image_quality": ("INT", {"default": 90, "min": 1, "max": 100, "tooltip": "JPEG quality for the cover (higher = larger, better)."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Kinburg-Nodes/audio"

    def run(self, audio, filename_prefix="songs/song", quality="MP3 V0 (VBR)",
            image=None, lyrics="", image_quality=90):
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

        audio_name = f"{base}.{fmt}"
        audio_path = os.path.join(full_output_folder, audio_name)
        with open(audio_path, "wb") as f:
            f.write(_encode_audio(w0, int(audio["sample_rate"]), fmt, q))

        # Standard UI keys so ComfyUI renders a native player / cover preview AND registers the
        # files in Media Assets (the frontend keys off "audio"/"images", not a custom key).
        ui = {"audio": [{"filename": audio_name, "subfolder": subfolder, "type": "output"}]}

        if text.strip():
            with open(os.path.join(full_output_folder, base + ".txt"), "w",
                      encoding="utf-8", newline="\n") as f:
                f.write(text)

        if image is not None:
            try:
                img_name = self._save_cover(image, full_output_folder, base, int(image_quality))
                ui["images"] = [{"filename": img_name, "subfolder": subfolder, "type": "output"}]
            except Exception as e:
                print(f"[KinburgSaveSong] cover save failed: {e}")

        return {"ui": ui, "result": (audio, audio_path)}

    @staticmethod
    def _save_cover(image, folder, base, quality):
        import numpy as np
        from PIL import Image
        arr = image[0].detach().cpu().numpy()  # [B,H,W,C] -> first frame [H,W,C]
        pil = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")).convert("RGB")
        name = base + ".jpg"
        pil.save(os.path.join(folder, name), format="JPEG", quality=quality)
        return name


NODE_CLASS_MAPPINGS = {"KinburgSaveSong": KinburgSaveSong}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSaveSong": "Save Song"}
