"""The alphabet the aligner can hear, and how a sung line is written in it.

`MMS_FA`'s dictionary is **28 characters**: `a`-`z`, an apostrophe, and the `*` star token. That is
not a shortcut anyone took — the model was trained on text put through `uroman`, a romanizer, which
is exactly how one acoustic model covers 1130 languages instead of one alphabet each. So a
Ukrainian line does not fail to align, it fails to be *spellable*, and the fix is a romanizer rather
than a different model.

`uroman` itself is a Perl program with a Python wrapper that is not installed here and would be a
dependency for thirty lines of table. What it does to Cyrillic is a per-character substitution with
two forks, and both forks are in this file.

**The one thing that must not be confused: what is SHOWN and what is ALIGNED are different
strings.** `"Живий!"` is displayed with its capital and its exclamation mark; it is aligned as
`zhyvyi`. Every token here therefore carries both, and the romanized form is allowed to come out
empty (a line of `"..."`, a digit, an emoji) without the word disappearing off the screen — it is
carried by its neighbour instead. Losing that distinction is how a subtitle track ends up in
lowercase Latin.

**Why the language fork exists.** Two letters are pronounced differently in the two languages the
author writes in, and both are common:

* `г` is [ɦ] in Ukrainian (`h`) and [ɡ] in Russian (`g`) — `ґ` is the Ukrainian `g`;
* `и` is [ɪ] in Ukrainian (`y`) and [i] in Russian (`i`).

Getting them backwards does not usually move a word's boundaries — the rest of the word still
matches and Viterbi is forgiving — but it costs confidence on exactly the short words (`і`, `и`,
`що`) where a karaoke wipe is most visible. The guess is made **per word**, because the author
writes lines with both languages in them, and falls back to whatever the lyric is mostly in.

**The one thing here that is a guess, and how to settle it.** `uroman` is not installed, so the
table below is a phonetic reading of Cyrillic rather than a copy of what the model was trained on.
The glides are where the two could differ: this writes `й ю я` as `y yu ya`, and `uroman` may write
them `j ju ja` — both are in the dictionary. It is not worth arguing about, because the aligner
hands back a **per-word confidence** and the node prints the mean of it. If Ukrainian lines come
back consistently less confident than English ones on the same track, swap the four glides and
measure again; a table that is wrong shows up as a number, not as a wrong subtitle.

Pure stdlib. No torch, no model, no network: this is the half of alignment that can be tested by
reading it.
"""
import re
import unicodedata

#: What `MMS_FA` can spell. Anything outside this set never reaches the aligner.
ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz'")

#: The token the model was given for "there is audio here that the transcript does not cover".
#: Not produced by romanizing anything — it is inserted around a line, never inside a word.
STAR = "*"

LANG_AUTO = "auto"
LANGUAGES = [LANG_AUTO, "uk", "ru", "en"]

#: Letters that exist in exactly one of the two Cyrillic alphabets. This is the whole language
#: detector: a word carrying `ї` is Ukrainian and a word carrying `ы` is Russian, and no amount of
#: statistics beats a letter that the other language does not have.
_UK_ONLY = frozenset("їієґ")
_RU_ONLY = frozenset("ыэъё")

# The shared table. `г` and `и` are deliberately absent — they are the fork, and a default here
# would silently make one language's spelling the other's.
_COMMON = {
    "а": "a", "б": "b", "в": "v", "д": "d", "е": "e", "ж": "zh", "з": "z",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ю": "yu", "я": "ya",
    # Both soft signs are silent. Dropped rather than mapped to an apostrophe: the apostrophe IS a
    # character in the dictionary, so writing one where nothing is pronounced spends a real frame.
    "ь": "", "ъ": "",
}
_UK = {**_COMMON, "г": "h", "ґ": "g", "и": "y", "і": "i", "ї": "yi", "є": "ye"}
_RU = {**_COMMON, "г": "g", "и": "i", "ы": "y", "э": "e", "ё": "yo",
       # A Russian lyric can still carry these — a name, a borrowed word, a line in the other
       # language that the per-word guess called Russian. Spelling them is better than dropping.
       "і": "i", "ї": "yi", "є": "ye", "ґ": "g"}

#: Latin letters that Unicode does NOT decompose, because they are letters in their own right
#: rather than a base plus a mark. `café` normalizes to `cafe` on its own; `cœur` normalizes to
#: `cœur` and then loses the `œ` entirely, which turns a word into `cur`. Found by a test, not by
#: reading the standard.
_LATIN_EXTRAS = {"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "ł": "l", "đ": "d", "ð": "d",
                 "þ": "th", "ħ": "h", "ŋ": "ng", "ı": "i"}

#: Typographic apostrophes, all of which mean the dictionary's `'`. Ukrainian uses one *inside*
#: words (`м'яко`), so this is not punctuation that can simply be stripped.
_APOSTROPHES = "'’ʼ‘´`"

_HAS_CYRILLIC = re.compile(r"[а-яёіїєґА-ЯЁІЇЄҐ]")
#: What counts as one word for alignment. Letters of either alphabet, digits (so a number is
#: *noticed* rather than silently split), and the apostrophes above.
_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ" + _APOSTROPHES + r"]+")


def guess_language(text):
    """`'uk'` / `'ru'` / `'en'` / `None` for text with nothing to go on.

    Only ever consulted for Cyrillic; Latin is spelled the same either way. `None` rather than a
    default, so the caller decides whether an undecidable word inherits the lyric's language or the
    node's setting — the two are different answers and this function knows neither.
    """
    lowered = str(text or "").lower()
    if not _HAS_CYRILLIC.search(lowered):
        return "en" if any(c.isalpha() for c in lowered) else None
    uk = sum(1 for c in lowered if c in _UK_ONLY)
    ru = sum(1 for c in lowered if c in _RU_ONLY)
    if uk > ru:
        return "uk"
    if ru > uk:
        return "ru"
    return None


def romanize(word, lang=LANG_AUTO, fallback="uk"):
    """One word -> the letters the aligner can hear, or `''` if none of it survives.

    `fallback` is what an undecidable Cyrillic word is spelled as — normally the language guessed
    over the whole lyric, which is why it is a parameter rather than a constant.
    """
    text = unicodedata.normalize("NFKC", str(word or "")).lower()
    for a in _APOSTROPHES[1:]:
        text = text.replace(a, "'")

    if lang == LANG_AUTO:
        lang = guess_language(text) or (fallback if _HAS_CYRILLIC.search(text) else "en")
    table = _UK if lang == "uk" else _RU if lang == "ru" else {}

    out = []
    for ch in text:
        if ch in table:
            out.append(table[ch])
            continue
        if ch in ALPHABET:
            out.append(ch)
            continue
        if ch in _LATIN_EXTRAS:
            out.append(_LATIN_EXTRAS[ch])
            continue
        # A Latin letter with a diacritic (é, ł, ß) decomposes to a plain one plus a combining mark;
        # the mark is not in the alphabet and drops out on the next pass. Anything that decomposes
        # to nothing spellable — a digit, an emoji, a dash — is simply absent from the result.
        for part in unicodedata.normalize("NFKD", ch):
            if part in ALPHABET:
                out.append(part)

    # An apostrophe is only a character when it sits between letters. One at either end is a quote
    # mark, and a leading one would make the word start on a sound nobody sings.
    return "".join(out).strip("'")


def words(line, lang=LANG_AUTO, fallback="uk"):
    """A sung line -> `[{text, key, at, to}]`, in order.

    `text` is the word as it will appear on screen, capitals intact. `key` is what the aligner is
    asked to find, and may be `''` — the caller must carry such a word on its neighbour's timing
    rather than drop it, or the subtitle loses a word the singer sang.

    `at`/`to` are the word's character offsets **in the original line**, and they are not
    bookkeeping: a karaoke line is drawn as one string cut into pieces, so every comma, dash and
    space between two words has to end up attached to one of them. With the offsets the caller can
    take `line[at:next.at]` and the pieces reassemble into exactly the line the author typed. Without
    them the punctuation is simply gone, and a lyric sheet turns into a word list.
    """
    line = str(line or "")
    return [{"text": m.group(0), "key": romanize(m.group(0), lang, fallback),
             "at": m.start(), "to": m.end()} for m in _WORD.finditer(line)]


def spellable(text, lang=LANG_AUTO, fallback="uk"):
    """How much of `text` the aligner can actually hear, 0..1, by word.

    The number the node reports before spending a GPU on anything: a lyric that comes out at 0.2 is
    written in an alphabet this table does not cover, and no amount of tuning the audio side will
    fix that. Empty text is 1.0 — there is nothing that failed.
    """
    ws = words(text, lang, fallback)
    if not ws:
        return 1.0
    return sum(1 for w in ws if w["key"]) / len(ws)
