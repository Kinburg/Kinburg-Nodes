"""Echo's subtitle files — the three things about ASS that are byte order, not taste.

A karaoke line is not text with times on it: it is one string cut into runs whose lengths must add
up to the event's own length, coloured by a pair of fields whose names are the wrong way round, in a
colour format that puts blue first. Each of those is a silent failure — the file loads, the player
shows something, and the something is wrong. So each is pinned here.

Pure stdlib: no torch, no model, no player.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "echo")
S = load_module("kn.echo.subs", "echo/subs.py")

check = Checker()


def line(words, voice="Nina", label="Verse 1"):
    """`[(text, start, end)]` -> one track line."""
    ws = [{"text": t, "start": a, "end": b} for t, a, b in words]
    return {"start": ws[0]["start"], "end": ws[-1]["end"], "voice": voice, "label": label,
            "text": " ".join(w["text"] for w in ws), "words": ws}


# ---------------------------------------------------------------------------- colour is BGR, and
# AA is transparency. A hex code straight off a picker comes out with red and blue swapped if this
# is wrong, which reads as a palette bug and is a byte order.
check("#ff66cc is &H00CC66FF (blue first, red last)", S.ass_colour("#ff66cc") == "&H00CC66FF",
      S.ass_colour("#ff66cc"))
check("pure red is &H000000FF", S.ass_colour("#ff0000") == "&H000000FF", S.ass_colour("#ff0000"))
check("pure blue is &H00FF0000", S.ass_colour("#0000ff") == "&H00FF0000", S.ass_colour("#0000ff"))
check("alpha 0 means OPAQUE", S.ass_colour("#ffffff", 0).startswith("&H00"))
check("alpha 255 means invisible", S.ass_colour("#ffffff", 255).startswith("&HFF"))
check("a colour with no hash still parses", S.ass_colour("ff66cc") == "&H00CC66FF")
check("junk falls back to white rather than raising", S.ass_colour("nope") == "&H00FFFFFF",
      S.ass_colour("nope"))


# ---------------------------------------------------------------------------------- the clock
check("ass_time is centiseconds", S.ass_time(62.34) == "0:01:02.34", S.ass_time(62.34))
check("59.999 s carries into the minute", S.ass_time(59.999) == "0:01:00.00", S.ass_time(59.999))
check("an hour is a single digit", S.ass_time(3661.5) == "1:01:01.50", S.ass_time(3661.5))
check("negative time is clamped, not signed", S.ass_time(-3) == "0:00:00.00")
check("srt_time never writes a four-digit millisecond", S.srt_time(0.9999) == "00:00:01,000",
      S.srt_time(0.9999))


# ------------------------------------------------------- the \k runs must ADD UP to the event
# This is the claim the whole karaoke sweep rests on. Rounding each word's own duration drifts by up
# to half a centisecond per word; over a 12-word line that is a visible lag by the end. Taken as
# differences of rounded absolute times it cannot drift at all — so: random lines, awkward numbers.
rng = random.Random(20260905)
drift = []
for trial in range(400):
    t = rng.uniform(0, 200)
    ws = []
    for _ in range(rng.randint(1, 14)):
        dur = rng.uniform(0.07, 1.7)
        ws.append((f"w{len(ws)}", t, t + dur))
        t += dur + (rng.uniform(0, 0.9) if rng.random() < 0.4 else 0.0)   # sometimes a gap
    ln = line(ws)
    text, start, end = S._karaoke(ln, lead_in=rng.choice([0.0, 0.35, 1.0]))
    runs = sum(int(v) for v in __import__("re").findall(r"\\k f?(\d+)".replace(" ", ""), text))
    want = int(round((end - start) * 100))
    if runs != want:
        drift.append((trial, runs, want))
check("the \\k runs sum to the event's own length, on 400 random lines", drift == [], drift[:3])

# The lead-in is a real silent run, not a shifted start: the line must be readable before it is sung.
text, start, end = S._karaoke(line([("Живий", 10.0, 10.5), ("я", 10.5, 10.8)]), lead_in=0.5)
check("the event opens half a second early", abs(start - 9.5) < 1e-9, start)
# Two runs before the first word is CORRECT and worth spelling out in full: the first is the silent
# lead-in (nothing highlighted while the line is being read), the second is the word's own 0.5 s.
check("...with a silent run covering the wait, then the word's own",
      text == "{\\k50}{\\k50}Живий{\\k30}я", text)
check("the event ends on the last word", abs(end - 10.8) < 1e-9, end)

# A gap between two words gets its own run, or the highlight runs ahead of the singer.
gapped, _, _ = S._karaoke(line([("a", 0.0, 0.5), ("b", 2.0, 2.5)]), lead_in=0.0, sweep=False)
check("a gap between words is a silent run", gapped == "{\\k50}a{\\k150}{\\k50}b", gapped)
check("sweep picks \\kf over \\k",
      S._karaoke(line([("a", 0.0, 0.5)]), 0.0, sweep=True)[0].startswith("{\\kf"))


# ------------------------------------------------------------------------------ escaping
check("a brace cannot open an override block", S.ass_text("a {b} c") == "a \\{b\\} c",
      S.ass_text("a {b} c"))
check("a newline becomes \\N", S.ass_text("a\nb") == "a\\Nb", S.ass_text("a\nb"))
check("CRLF is one break, not two", S.ass_text("a\r\nb") == "a\\Nb", S.ass_text("a\r\nb"))
check("a style name with a comma cannot shift the row",
      "," not in S._safe_style("Nina, the loud one"), S._safe_style("Nina, the loud one"))
check("an empty voice still names a style", S._safe_style("") == "Default")


# -------------------------------------------------------------------------- colours per voice
mapping, notes = S.parse_colours("Nina = #ff66cc\n Gru : 00aaff \n# a comment\nbroken line\n")
check("a mapping row parses", mapping.get("nina") == "#ff66cc", mapping)
check("the separator is loose", mapping.get("gru") == "#00aaff", mapping)
check("a comment is silent", "a comment" not in str(notes), notes)
check("a broken row is reported, not dropped in silence", any("broken" in n for n in notes), notes)

got = S.assign_colours(["Nina", "Gru", "Nina", "Ivan"], mapping)
check("an assigned colour is used", got["nina"] == "#ff66cc")
check("an unassigned voice takes the next palette colour", got["ivan"] == S.PALETTE[0], got)
check("a voice is only counted once", len(got) == 3, got)
check("assignment is stable across runs",
      S.assign_colours(["Nina", "Gru", "Ivan"], mapping) == got)


# ----------------------------------------------------------------------------- the whole file
track = {"total": 30.0, "lines": [
    line([("Живий", 2.0, 2.6), ("я", 2.6, 2.9), ("живий", 2.9, 3.8)], "Nina", "Verse 1"),
    line([("Don't", 6.0, 6.4), ("stop", 6.4, 7.0)], "Gru", "Chorus"),
    line([("тиша", 9.0, 9.9)], "", "Bridge"),
]}
doc = S.ass(track, mapping, width=1080, height=1920)
check("the file has all three blocks",
      all(b in doc for b in ("[Script Info]", "[V4+ Styles]", "[Events]")))
check("PlayRes is the real frame", "PlayResX: 1080" in doc and "PlayResY: 1920" in doc)
check("one style row per voice, plus Default",
      doc.count("\nStyle: ") == 3, doc.count("\nStyle: "))
check("three events", doc.count("\nDialogue: ") == 3, doc.count("\nDialogue: "))
check("a line with no voice falls back to the Default style", ",Default,Bridge," in doc)
check("Nina's style carries HER colour as PrimaryColour",
      [r for r in doc.splitlines() if r.startswith("Style: Nina")][0].split(",")[3] == "&H00CC66FF",
      [r for r in doc.splitlines() if r.startswith("Style: Nina")][0])
check("...and the idle colour as SecondaryColour",
      [r for r in doc.splitlines() if r.startswith("Style: Nina")][0].split(",")[4]
      == S.ass_colour(S.DEFAULT_IDLE))
check("the font size scales off the SHORT side of a phone frame",
      "," + str(max(20, round(1080 / 22.0))) + "," in doc.split("[V4+ Styles]")[1][:400],
      max(20, round(1080 / 22.0)))
check("every style row has the 23 fields the Format line declares",
      all(len(r.split(",")) == 23 for r in doc.splitlines() if r.startswith("Style: ")),
      [len(r.split(",")) for r in doc.splitlines() if r.startswith("Style: ")])

# The lyric goes into the file as the author typed it — a subtitle that has been lowercased or
# stripped of its punctuation by the aligner's needs is the failure this whole design avoids.
check("the displayed word keeps its capital", "Живий" in doc)
empty = S.ass({"lines": []})
check("an empty track is still a valid file", "[Events]" in empty and "Dialogue:" not in empty)

# ------------------------------------------------------------------------------------- srt
plain = S.srt(track)
check("one srt cue per line", plain.count(" --> ") == 3, plain.count(" --> "))
per_word = S.srt(track, per_line=False)
check("per-word srt is the debugging view", per_word.count(" --> ") == 6, per_word.count(" --> "))
check("srt numbering starts at 1", plain.startswith("1\n"), plain[:8])
check("a zero-length cue is dropped rather than written",
      S.srt({"lines": [line([("a", 5.0, 5.0)])]}) == "")

check.done()
