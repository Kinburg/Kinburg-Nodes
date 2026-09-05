"""Save Clip — one picture (or a slideshow) over a song, written straight to an mp4.

The video sibling of **Save Song**: same counter-based naming under ComfyUI's output folder, the
same "quality in words, not in codec flags" dropdown, the same `SONG_TAGS` going into the file
itself, and a player on the node when it is done. What it adds is the only interesting part — WHEN
each picture is on screen.

**Why this is not `Create Video` + `Save Video`.** Those take an IMAGE batch and encode one frame
per item, so a three-minute song at 30 fps means a batch of 5 400 frames — about 17 GB of tensor
for a 1024x1024 picture, before the encoder has seen anything. This node never builds that batch:
it holds at most a couple of prepared slides and feeds the encoder frame by frame, so the memory
it needs is set by the frame SIZE and never by the song's length.

**Timing comes from the audio, the plan only supplies proportions** — see ``timeline.py`` for why
that is the only way a slideshow cannot drift off its own song. Wire `Siren Score`'s plan (bars,
with the section labels on it), `Orpheus`'s `durations` (seconds, already cut to the music), or
nothing at all for an even split.

Category ``Kinburg-Nodes/video``.
"""
import os

from . import timeline as T
from ..categories import CAT_VIDEO

# label -> (crf, x264 preset, audio bitrate). CRF is the dial that matters and 18/21/26 is the
# usable span of it; the preset only trades encode time for a few percent of size.
_QUALITY = {
    "High (crf 18, big file)": (18, "slow", 256000),
    "Balanced (crf 21)": (21, "medium", 192000),
    "Small (crf 26, for messengers)": (26, "faster", 128000),
}
QUALITY_OPTIONS = list(_QUALITY.keys())

SIZE_SOURCE = "source (from the image)"
_SIZES = {
    SIZE_SOURCE: None,
    "1920x1080 (16:9)": (1920, 1080),
    "1280x720 (16:9, light)": (1280, 720),
    "1080x1920 (9:16, phone)": (1080, 1920),
    "1080x1080 (1:1)": (1080, 1080),
}
SIZE_OPTIONS = list(_SIZES.keys())

FIT_BLUR = "pad with a blurred copy"
FIT_BLACK = "pad with black"
FIT_CROP = "crop to fill"
FIT_OPTIONS = [FIT_BLUR, FIT_BLACK, FIT_CROP]


def _even(n):
    """yuv420p needs both sides even, and an odd one fails inside libx264 rather than here."""
    return max(2, int(n) // 2 * 2)


def _to_pil(frame):
    """One IMAGE frame ([H,W,C] float 0..1) -> RGB PIL."""
    import numpy as np
    from PIL import Image
    arr = frame[..., :3].detach().cpu().numpy()
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8"), mode="RGB")


def _fitted(pil, size, fit):
    """`pil` laid onto a `size` canvas by one of the three fit rules."""
    from PIL import Image, ImageFilter
    tw, th = size
    sw, sh = pil.size
    if (sw, sh) == (tw, th):
        return pil
    cover = max(tw / sw, th / sh)
    contain = min(tw / sw, th / sh)

    if fit == FIT_CROP:
        w, h = max(tw, int(round(sw * cover))), max(th, int(round(sh * cover)))
        big = pil.resize((w, h), Image.LANCZOS)
        return big.crop(((w - tw) // 2, (h - th) // 2, (w - tw) // 2 + tw, (h - th) // 2 + th))

    w, h = max(1, int(round(sw * contain))), max(1, int(round(sh * contain)))
    front = pil.resize((w, h), Image.LANCZOS)
    if fit == FIT_BLUR:
        bw, bh = max(tw, int(round(sw * cover))), max(th, int(round(sh * cover)))
        back = pil.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - tw) // 2, (bh - th) // 2, (bw - tw) // 2 + tw, (bh - th) // 2 + th))
        # Radius scaled to the frame: a fixed one is a smear at 4K and a visible copy at 720p.
        back = back.filter(ImageFilter.GaussianBlur(max(8, min(tw, th) // 16)))
    else:
        back = Image.new("RGB", (tw, th), (0, 0, 0))
    back.paste(front, ((tw - w) // 2, (th - h) // 2))
    return back


class _Slides:
    """The prepared pictures, at most a handful alive at once.

    Frames are produced in time order and a dissolve needs two pictures, so a cache of three is
    enough to never prepare the same slide twice in a row — and it is what keeps a 400-image batch
    from costing 400 full-size canvases.
    """

    def __init__(self, images, size, fit, work_scale):
        self.images, self.size, self.fit, self.work = images, size, fit, work_scale
        self.count = int(images.shape[0])
        self._cache = {}

    def __call__(self, i):
        i = int(i) % self.count
        hit = self._cache.get(i)
        if hit is None:
            w, h = self.size
            target = (max(2, int(round(w * self.work))), max(2, int(round(h * self.work))))
            hit = _fitted(_to_pil(self.images[i]), target, self.fit)
            if len(self._cache) >= 3:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = hit
        return hit


def _compose(slides, segments, t, a, b, mix, size, amount):
    """The one frame at `t` — a Ken Burns view of each side of the dissolve, blended."""
    import numpy as np
    from PIL import Image

    def view(seg_index):
        pil = slides(segments[seg_index]["slide"])
        if amount <= 0:
            return pil if pil.size == size else pil.resize(size, Image.LANCZOS)
        seg = segments[seg_index]
        span = max(1e-6, seg["end"] - seg["start"])
        p = max(0.0, min(1.0, (t - seg["start"]) / span))
        z0, z1, (x0, y0), (x1, y1) = T.ken_burns(seg_index, amount)
        return _view(pil, size, z0 + (z1 - z0) * p, x0 + (x1 - x0) * p, y0 + (y1 - y0) * p)

    front = view(a)
    if mix <= 0.0 or b == a:
        return np.asarray(front)
    return np.asarray(Image.blend(front, view(b), max(0.0, min(1.0, mix))))


def _view(pil, size, z, cx, cy):
    """The Ken Burns window: crop at zoom `z`, centred by `cx`/`cy` in -1..1, resized to `size`."""
    from PIL import Image
    tw, th = size
    sw, sh = pil.size
    cw, ch = max(2, int(round(sw / z))), max(2, int(round(sh / z)))
    x = (sw - cw) / 2 * (1 + max(-1.0, min(1.0, cx)))
    y = (sh - ch) / 2 * (1 + max(-1.0, min(1.0, cy)))
    x, y = int(round(max(0, min(sw - cw, x)))), int(round(max(0, min(sh - ch, y))))
    box = pil.crop((x, y, x + cw, y + ch))
    return box if (cw, ch) == (tw, th) else box.resize((tw, th), Image.LANCZOS)


class KinburgSaveClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The picture, or the batch of slides. One image = one still behind the whole song; a batch = a slideshow, laid out by 'layout' below.\n\nNothing is resampled in time: the batch is NOT the video's frames, it is the list of pictures. A 4-image batch over a 3-minute song is four shots, not four frames."}),
                "audio": ("AUDIO", {"tooltip": "The song. Its length IS the video's length — the slides are scaled onto it, so the last picture always ends on the last sample."}),
                "filename_prefix": ("STRING", {"default": "clips/clip", "tooltip": "Save path prefix under ComfyUI/output. A counter is appended; the subtitles (.srt) share the same base name."}),
                "quality": (QUALITY_OPTIONS, {"default": "Balanced (crf 21)", "tooltip": "h264 + AAC in an mp4, at one of three CRFs. 'High' is visually lossless on a still and roughly doubles the file; 'Small' is what survives a messenger's own re-encode."}),
            },
            "optional": {
                "plan": ("STRING", {"forceInput": True, "tooltip": "Where the pictures change. Two shapes are read:\n\n• Siren Score's / Siren Cast's plan — 'label | voice | 16 bars' rows. The section LABELS come with it, which is what makes 'by section label' possible.\n• Orpheus's 'durations' — a plain comma list of seconds, already cut to the music.\n\nOnly the PROPORTIONS are used: the plan is scaled onto the real length of the audio, so no bpm is needed here and the slideshow cannot drift off the song. Leave it empty to split the song evenly."}),
                "layout": (T.LAYOUTS, {"default": T.LAYOUT_ORDER, "tooltip": "How the slides are handed to the plan's sections.\n\n• one slide per section — the picture changes on the section boundary. More slides than sections and the extra ones subdivide the longest sections; fewer and they cycle.\n• by section label — every row called 'Chorus' gets the SAME slide, so the chorus shot comes back the way it does in a cut video. Costs nothing: the labels are already in the plan.\n• even — the plan is ignored and the song is split equally."}),
                "fps": ("INT", {"default": 12, "min": 1, "max": 60, "tooltip": "Nothing moves in a slideshow, so this is mostly file size and encode time — 12 is plenty for stills and a third of the frames of 30.\n\nRaise it to 24-30 when 'crossfade' or 'ken_burns' is on: those are the only things here that actually move, and at 12 fps a dissolve steps rather than flows."}),
                "frame_size": (SIZE_OPTIONS, {"default": SIZE_SOURCE, "tooltip": "The video's frame. 'source' takes the first image's own size (rounded down to even sides, which yuv420p requires). The rest are the shapes the platforms want; how a picture that isn't that shape is laid onto them is 'fit'."}),
                "fit": (FIT_OPTIONS, {"default": FIT_BLUR, "tooltip": "What to do when the picture and the frame are different shapes.\n\n• pad with a blurred copy — the picture whole, on a blurred enlargement of itself. The usual choice for a square cover in a 16:9 frame.\n• pad with black — the picture whole, on black.\n• crop to fill — no bars, but the edges of the picture are gone."}),
                "crossfade": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.1, "tooltip": "Seconds of dissolve at each change of picture, centred ON the cut — half before, half after — so the moment the two pictures are equal is the moment the music turns.\n\n0 = hard cuts. Clamped to 40% of the shorter neighbour, so a long fade cannot swallow a short section."}),
                "ken_burns": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.4, "step": 0.01, "advanced": True, "tooltip": "Slow zoom and drift over each slide, as a fraction (0.08 = the picture creeps 8% closer across its section). A still held for three minutes reads as a broken video; this is what makes it read as a shot.\n\nIt is off by default because it is not free: every frame becomes unique, so the file and the encode time grow several times over. With it on, use 24 fps or more."}),
                "lyrics": ("STRING", {"forceInput": True, "tooltip": "The lyrics, with their '[Verse 1 - ...]' markers — the SAME text that went to Siren. Each section's own lines are written to a .srt next to the video, timed to that section's segment.\n\nOnly works with a Siren plan wired (the labels are what the two are matched on). Sections with no sung lines get no cue."}),
                "tags": ("SONG_TAGS", {"tooltip": "Artist, album, year, genre — wire a 'Song Tags' node, the same one Save Song takes. Written into the mp4's own metadata; the title falls back to the file's name."}),
                "verbose": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Print the report to the console. The same text is always on the 'report' output."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "path", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT_VIDEO
    DESCRIPTION = ("A still (or a slideshow) over a song, written to an mp4 with the picture "
                   "changing where the music does. Wire Siren Score's plan and the slides land on "
                   "the section boundaries; wire nothing and the song is split evenly. The song's "
                   "own length decides the video's, so the pictures cannot drift off it.")

    def run(self, image, audio, filename_prefix="clips/clip", quality="Balanced (crf 21)",
            plan="", layout=T.LAYOUT_ORDER, fps=12, frame_size=SIZE_SOURCE, fit=FIT_BLUR,
            crossfade=0.5, ken_burns=0.0, lyrics="", tags=None, verbose=True):
        import av
        import folder_paths
        import numpy as np
        from fractions import Fraction

        if audio is None or "waveform" not in audio:
            raise ValueError("Save Clip: 'audio' is empty (there is nothing to time the slides to).")
        if image is None or int(image.shape[0]) == 0:
            raise ValueError("Save Clip: 'image' is empty (there is nothing to show).")

        wf = audio["waveform"]
        wave = (wf[0] if wf.dim() == 3 else wf).cpu()
        sample_rate = int(audio["sample_rate"])
        total = wave.shape[-1] / float(sample_rate)
        if total <= 0:
            raise ValueError("Save Clip: the audio holds no samples.")

        # ------------------------------------------------------------------ what is on screen when
        rows, notes = T.sections(plan)
        segments, more = T.lay_out(rows, int(image.shape[0]), layout, total)
        notes.extend(more)

        src_h, src_w = int(image.shape[1]), int(image.shape[2])
        size = _SIZES.get(frame_size) or (src_w, src_h)
        size = (_even(size[0]), _even(size[1]))

        fps = max(1, int(fps))
        amount = max(0.0, float(ken_burns))
        # With a move on, the slide is prepared LARGER than the frame so the crop downsamples into
        # it. Prepare it at frame size and every zoomed frame would be an upscale of a still.
        slides = _Slides(image, size, fit, 1.0 + amount)
        track = T.frame_track(segments, fps, total, float(crossfade))

        crf, preset, audio_rate = _QUALITY.get(quality, _QUALITY["Balanced (crf 21)"])

        out_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir, size[0], size[1])
        base = f"{filename}_{counter:05}"
        name = base + ".mp4"
        path = os.path.join(full_output_folder, name)

        meta = dict(tags or {})
        meta["title"] = str(meta.get("title") or "").strip() or base

        # ------------------------------------------------------------------------------- encoding
        # movflags: 'faststart' moves the index to the front (a clip starts playing before it has
        # downloaded), 'use_metadata_tags' is what lets the tags below survive into the file.
        with av.open(path, mode="w", options={"movflags": "use_metadata_tags+faststart"}) as out:
            for key, value in meta.items():
                if value not in (None, ""):
                    out.metadata[str(key)] = str(value)

            video = out.add_stream("h264", rate=Fraction(fps, 1))
            video.width, video.height = size
            video.pix_fmt = "yuv420p"
            video.options = {"crf": str(crf), "preset": preset}

            # BOTH streams before the first packet. A stream added after the header has been
            # written has a time_base of 0, and the mux fails with "cannot rebase to zero time".
            layout_name = {1: "mono", 2: "stereo", 6: "5.1"}.get(int(wave.shape[0]), "stereo")
            track_audio = out.add_stream("aac", rate=sample_rate, layout=layout_name)
            track_audio.bit_rate = audio_rate

            pbar = None
            try:
                from comfy.utils import ProgressBar
                pbar = ProgressBar(len(track))
            except Exception:
                pass

            still = {}          # segment index -> the encoded-ready frame, when nothing moves
            for i, (t, a, b, mix) in enumerate(track):
                if amount <= 0 and mix <= 0.0:
                    frame = still.get(a)
                    if frame is None:
                        still = {a: av.VideoFrame.from_ndarray(
                            np.asarray(slides(segments[a]["slide"])), format="rgb24"
                        ).reformat(format="yuv420p")}
                        frame = still[a]
                else:
                    frame = av.VideoFrame.from_ndarray(
                        _compose(slides, segments, t, a, b, mix, size, amount),
                        format="rgb24").reformat(format="yuv420p")
                frame.pts = i
                out.mux(video.encode(frame))
                if pbar is not None and (i % 32 == 0 or i == len(track) - 1):
                    pbar.update_absolute(i + 1, len(track))
            out.mux(video.encode(None))

            # Same recipe ComfyUI's own encoder uses — one float-planar frame, let AAC chunk it.
            frame = av.AudioFrame.from_ndarray(
                wave.float().contiguous().numpy(), format="fltp", layout=layout_name)
            frame.sample_rate = sample_rate
            frame.pts = 0
            out.mux(track_audio.encode(frame))
            out.mux(track_audio.encode(None))

        # ------------------------------------------------------------------------------ subtitles
        srt_written = ""
        if str(lyrics or "").strip():
            try:
                from ..siren.score import _split_sections
                secs, _ = _split_sections(lyrics)
                # The PLAN's sections, not the visual segments: where the words are has nothing to
                # do with how many pictures there are.
                body = T.srt(T.section_spans(rows, total), secs)
                if body.strip():
                    srt_written = base + ".srt"
                    with open(os.path.join(full_output_folder, srt_written), "w",
                              encoding="utf-8", newline="\n") as f:
                        f.write(body)
                elif not rows:
                    notes.append("the lyrics were given but no plan was — there is nothing to time "
                                 "them against, so no .srt was written. Wire Siren Score's 'plan'.")
                else:
                    notes.append("no subtitle cue matched a section — the lyrics' markers and the "
                                 "plan's labels have to be the same words, so wire both from Siren")
            except Exception as e:
                notes.append(f"the .srt could not be written: {e}")

        if srt_written:
            notes.append(f"subtitles written to {srt_written}")
        text = T.report(segments, int(image.shape[0]), fps, size, total, notes)
        if verbose:
            print("[Save Clip] " + text.replace("\n", "\n[Save Clip] "))

        video_out = None
        try:
            from comfy_api.input_impl import VideoFromFile
            video_out = VideoFromFile(path)
        except Exception as e:                       # older ComfyUI: the file is still on disk
            print(f"[Save Clip] no VIDEO object for this ComfyUI build: {e}")

        ui = {"images": [{"filename": name, "subfolder": subfolder, "type": "output"}],
              "animated": (True,)}
        return {"ui": ui, "result": (video_out, path, text)}


NODE_CLASS_MAPPINGS = {"KinburgSaveClip": KinburgSaveClip}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSaveClip": "Save Clip"}
