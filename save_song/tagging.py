"""Tags that ride INSIDE the file — ID3v2.4 for MP3, Vorbis comments for FLAC and Opus.

A `.txt` and a `.jpg` next to the song are lost the moment it is copied onto a phone, and every
player on earth reads its title, cover and lyrics out of the file itself. So this module writes them
where they belong.

**No new dependency.** PyAV is already here (it is what ComfyUI's own Save Audio encodes with) and it
carries text metadata for all three formats — but NOT the cover: attaching a picture through PyAV
means an `attached_pic` video stream, and that path is broken both ways in PyAV 17 (setting
`stream.disposition` raises TypeError, and muxing the stream tries to open an mjpeg *encoder* and
fails with EINVAL). The two remaining formats are just bytes, so they are written here:

* **MP3** — an ID3v2.4 tag built frame by frame and prepended to the encoded stream. That is all an
  ID3 tag is: a header, a synchsafe length and a run of frames ahead of the first MPEG frame.
* **FLAC / Opus** — a `METADATA_BLOCK_PICTURE` comment: the FLAC picture block, base64'd, which is
  the standard way a cover travels in a Vorbis comment. PyAV passes it through untouched.

Both were checked by writing a file and reading it back with ffmpeg: the picture comes out as an
`attached_pic` stream, byte-identical to the JPEG that went in, and UTF-8 text (Cyrillic included)
survives intact.

One quirk worth knowing, since it is invisible until tags go missing: for **FLAC** the comments go on
the CONTAINER (`container.metadata`), for **Opus** they go on the STREAM (`stream.metadata`). Set
them on the container for Opus and they are silently dropped — no error, just an untagged file.
"""
import base64
import struct

# The canonical fields, in the order they are written. Anything else a user names goes through
# `extra`, where a handful of well-known names still land in a real frame rather than a TXXX.
_ID3 = {
    "title": "TIT2", "artist": "TPE1", "album": "TALB", "album_artist": "TPE2",
    "track": "TRCK", "date": "TDRC", "genre": "TCON",
}
_VORBIS = {
    "title": "TITLE", "artist": "ARTIST", "album": "ALBUM", "album_artist": "ALBUMARTIST",
    "track": "TRACKNUMBER", "date": "DATE", "genre": "GENRE",
}
# Extras a player actually understands, so `bpm: 120` becomes TBPM and not a TXXX nobody reads.
_EXTRA_ID3 = {
    "bpm": "TBPM", "composer": "TCOM", "publisher": "TPUB", "copyright": "TCOP",
    "isrc": "TSRC", "disc": "TPOS", "language": "TLAN", "mood": "TMOO", "key": "TKEY",
}
_EXTRA_VORBIS = {
    "bpm": "BPM", "composer": "COMPOSER", "publisher": "ORGANIZATION", "copyright": "COPYRIGHT",
    "isrc": "ISRC", "disc": "DISCNUMBER", "language": "LANGUAGE", "mood": "MOOD", "key": "KEY",
}
FIELDS = list(_ID3.keys()) + ["comment"]
LANG = b"eng"      # ISO-639-2 for USLT/COMM. Players key off the frame, not the language.


def _clean(v):
    return "" if v is None else str(v).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalise(tags):
    """A tags dict from anywhere (the node, a hand-written dict) into the canonical shape."""
    t = dict(tags or {})
    out = {k: _clean(t.get(k)) for k in FIELDS}
    extra = {}
    for k, v in (t.get("extra") or {}).items():
        k, v = _clean(k), _clean(v)
        if k and v:
            extra[k] = v
    out["extra"] = extra
    return out


# ------------------------------------------------------------------------------ ID3v2.4 (MP3)
def _syncsafe(n):
    """Seven bits per byte, so a size can never contain a run that looks like a frame sync."""
    return bytes(((n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F))


def _frame(fid, payload):
    return fid.encode("ascii") + _syncsafe(len(payload)) + b"\x00\x00" + payload


def _text_frame(fid, value):
    # Encoding byte 3 = UTF-8, which only ID3v2.4 allows — and 2.4 is what ffmpeg writes here
    # anyway, so nothing that reads these files is being asked for something new.
    return _frame(fid, b"\x03" + value.encode("utf-8") + b"\x00")


def strip_id3(raw):
    """Drop a leading ID3v2 tag. The muxer writes its own (a lone TSSE); ours replaces it whole
    rather than being stapled on in front, which would leave the file with two tags."""
    if len(raw) < 10 or raw[:3] != b"ID3":
        return raw
    size = 0
    for b in raw[6:10]:
        size = (size << 7) | (b & 0x7F)
    end = 10 + size + (10 if raw[5] & 0x10 else 0)      # bit 4 of the flags = a footer is present
    return raw[end:] if end <= len(raw) else raw


def id3v2_tag(tags, lyrics="", cover=None, encoder=""):
    """Build a complete ID3v2.4 tag. Returns b"" when there is nothing worth writing."""
    t = normalise(tags)
    body = b""
    for field, fid in _ID3.items():
        if t[field]:
            body += _text_frame(fid, t[field])
    if t["comment"]:
        body += _frame("COMM", b"\x03" + LANG + b"\x00" + t["comment"].encode("utf-8"))
    for k, v in t["extra"].items():
        fid = _EXTRA_ID3.get(k.lower())
        body += (_text_frame(fid, v) if fid else
                 _frame("TXXX", b"\x03" + k.encode("utf-8") + b"\x00" + v.encode("utf-8")))
    if _clean(lyrics):
        # USLT, not a comment: this is the frame phones and car stereos scroll along to.
        body += _frame("USLT", b"\x03" + LANG + b"\x00" + _clean(lyrics).encode("utf-8"))
    if encoder:
        body += _text_frame("TSSE", encoder)
    if cover:
        # 0x03 = "cover (front)", the type every player looks for first.
        body += _frame("APIC", b"\x03" + b"image/jpeg\x00" + b"\x03" + b"Cover\x00" + cover)
    if not body:
        return b""
    return b"ID3\x04\x00\x00" + _syncsafe(len(body)) + body


# --------------------------------------------------------------- Vorbis comments (FLAC / Opus)
def vorbis_comments(tags, lyrics="", cover=None, size=None):
    t = normalise(tags)
    out = {}
    for field, name in _VORBIS.items():
        if t[field]:
            out[name] = t[field]
    if t["comment"]:
        out["COMMENT"] = t["comment"]
    for k, v in t["extra"].items():
        out[_EXTRA_VORBIS.get(k.lower(), k.upper())] = v
    if _clean(lyrics):
        out["LYRICS"] = _clean(lyrics)
    if cover:
        out["METADATA_BLOCK_PICTURE"] = picture_comment(cover, size)
    return out


def picture_comment(jpeg, size=None):
    """A FLAC picture block, base64'd — how a cover travels in a Vorbis comment.

    The declared width/height are advisory (players read the JPEG itself), so an unknown size is
    written as 0 rather than guessed at."""
    w, h = size or (0, 0)
    mime, desc = b"image/jpeg", b"Cover"
    block = (struct.pack(">I", 3)                                   # picture type: cover (front)
             + struct.pack(">I", len(mime)) + mime
             + struct.pack(">I", len(desc)) + desc
             + struct.pack(">IIII", int(w), int(h), 24, 0)          # w, h, bit depth, palette size
             + struct.pack(">I", len(jpeg)) + jpeg)
    return base64.b64encode(block).decode("ascii")
