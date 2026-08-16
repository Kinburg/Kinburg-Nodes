"""Song Tags — the fields that go inside the saved file, gathered on their own node.

A separate node rather than eight more widgets on Save Song: this is filled in once and then sits
there, while Save Song is the node you keep re-running. Wire the `tags` output into Save Song and it
writes them as ID3v2.4 (MP3) or Vorbis comments (FLAC / Opus).

Everything here is optional — an empty field is simply not written, and Save Song still fills in the
title from the filename and embeds the cover and lyrics it already has.
"""
import json
from ..categories import CAT_AUDIO

_HINT = "Leave empty to skip this tag."


class KinburgSongTags:
    @classmethod
    def INPUT_TYPES(cls):
        s = lambda tip, **kw: ("STRING", {"default": "", "tooltip": tip, **kw})
        return {
            "required": {
                "title": s("Track title. Empty = Save Song uses the file's own name, so the player "
                           "shows 'song_00007' rather than nothing at all."),
                "artist": s(_HINT),
                "album": s(_HINT),
                "album_artist": s("Who the ALBUM is credited to. Worth setting when the tracks have "
                                  "different artists, or a player will file each one separately."),
                "track": s("Track number. '3' or '3/12' both work."),
                "year": s("Year, or a full date — '2026' and '2026-08-16' are both valid ID3v2.4 "
                          "timestamps."),
                "genre": s("Free text; ID3v2.4 dropped the numeric genre list."),
                "comment": s(_HINT, multiline=True),
                "extra": ("STRING", {"default": "", "multiline": True, "tooltip":
                          "Anything else, one 'name: value' per line.\n\nNames a player actually "
                          "understands land in the real field — bpm, composer, publisher, "
                          "copyright, isrc, disc, language, mood, key — and everything else is "
                          "written as a custom tag (TXXX / a named Vorbis comment), which is where "
                          "'seed: 12345' or 'sampler: chimera' belongs."}),
            },
            "optional": {
                "settings": ("GEN_SETTINGS", {"tooltip":
                             "Generation Info Filter's 'settings_data' — every field it selected is "
                             "written as a custom tag, so the song carries the settings that made "
                             "it. Takes the FIRST dump: this node describes one song, not a batch. "
                             "The 'extra' lines win where the names collide."}),
            },
        }

    RETURN_TYPES = ("SONG_TAGS",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "run"
    CATEGORY = CAT_AUDIO
    DESCRIPTION = ("The tags Save Song writes INTO the file — title, artist, album, year, genre and "
                   "anything else you name. Kept on its own node because it is filled in once, "
                   "while Save Song is the one you keep re-running. Wire 'tags' into Save Song; "
                   "empty fields are skipped, and Generation Info's settings can ride along as "
                   "custom tags so the song remembers how it was made.")

    def run(self, title="", artist="", album="", album_artist="", track="", year="", genre="",
            comment="", extra="", settings=None):
        tags = {"title": title, "artist": artist, "album": album, "album_artist": album_artist,
                "track": track, "date": year, "genre": genre, "comment": comment}
        # Settings first, so a hand-typed 'extra' line overrides what the graph reported.
        pairs = _from_settings(settings)
        pairs.update(_parse_pairs(extra))
        tags["extra"] = pairs
        return (tags,)


def _parse_pairs(text):
    """'name: value' per line. Split on the FIRST colon only — values are full of them (times,
    URLs, 'ratio: 16:9')."""
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition(":")
        if not sep:
            continue
        name, value = name.strip(), value.strip()
        if name and value:
            out[name] = value
    return out


def _from_settings(raw):
    """Generation Info Filter's settings_data: a JSON list of per-item [{key, value}, …]."""
    try:
        data = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, list):
        return {}
    fields = next((d for d in data if isinstance(d, list) and d), [])
    out = {}
    for f in fields:
        if not isinstance(f, dict) or f.get("key") is None:
            continue
        v = f.get("value", "")
        v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        v = " ".join(v.split())
        if v:
            out[str(f["key"])] = v
    return out


NODE_CLASS_MAPPINGS = {"KinburgSongTags": KinburgSongTags}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgSongTags": "Song Tags"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
