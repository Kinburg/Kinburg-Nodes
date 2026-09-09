"""Echo's romanizer: the 28 characters `MMS_FA` can hear, and what a Cyrillic lyric becomes in them.

Two claims carry the rest. **Nothing outside the model's dictionary ever reaches it** — a table that
emits one stray character produces a token id that does not exist and the aligner raises somewhere
deep inside Viterbi, which is a miserable place to read a bug from. And **what is shown is not what
is aligned**: every token keeps its original spelling, so a subtitle track cannot come out in
lowercase Latin no matter what the aligner needed.

Pure stdlib — no torch, no model, no network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "echo")
R = load_module("kn.echo.translit", "echo/translit.py")

check = Checker()


# ------------------------------------------------------------------ the dictionary is the contract
# `MMS_FA.get_dict()` is 28 entries + the blank: a-z, an apostrophe, and the star. This suite hard-
# codes it rather than importing torchaudio, so it runs without weights — if the bundle ever changes
# its alphabet, the aligner will say so and this check is what gets updated.
check("the alphabet is a-z plus an apostrophe", len(R.ALPHABET) == 27 and "'" in R.ALPHABET,
      len(R.ALPHABET))

CORPUS = [
    "Живий! Я живий, і кожен подих — мій",
    "Лише пусті знаки, лише пусті слова",
    "Я помню чудное мгновенье, передо мной явилась ты",
    "Don't stop me now, I'm having such a good time",
    "Ще не вмерла України і слава, і воля",
    "Ой у лузі червона калина похилилася",
    "м'яко, з'їв, п'ять — апостроф всередині слова",
    "Мій рідний край, où le cœur bat, naïve café",
]
outside = []
for line in CORPUS:
    for w in R.words(line):
        for ch in w["key"]:
            if ch not in R.ALPHABET:
                outside.append((line[:20], w["text"], ch))
check("no romanized word carries a character the model has no id for", outside == [], outside[:4])


# ------------------------------------------------------------------------ the two-language fork
# `г` and `и` are the whole reason a language is asked for at all.
check("uk: г is h", R.romanize("гора", "uk") == "hora", R.romanize("гора", "uk"))
check("ru: г is g", R.romanize("гора", "ru") == "gora", R.romanize("гора", "ru"))
check("uk: ґ is the hard g", R.romanize("ґанок", "uk") == "ganok", R.romanize("ґанок", "uk"))
check("uk: и is y", R.romanize("сини", "uk") == "syny", R.romanize("сини", "uk"))
check("ru: и is i", R.romanize("сини", "ru") == "sini", R.romanize("сини", "ru"))
check("uk: ї is yi", R.romanize("їжак", "uk") == "yizhak", R.romanize("їжак", "uk"))
check("ru: ы is y", R.romanize("ты", "ru") == "ty", R.romanize("ты", "ru"))
check("ru: ё is yo", R.romanize("всё", "ru") == "vsyo", R.romanize("всё", "ru"))
check("щ is four letters", R.romanize("що", "uk") == "shcho", R.romanize("що", "uk"))

# The soft sign is silent, and is dropped rather than written as an apostrophe: the apostrophe is a
# real character in the dictionary and would cost a frame on a sound nobody makes.
check("ь is silent", R.romanize("день", "uk") == "den", R.romanize("день", "uk"))
check("ъ is silent", R.romanize("объезд", "ru") == "obezd", R.romanize("объезд", "ru"))

# …but a Ukrainian apostrophe sits INSIDE a word and is a real separation.
check("an inner apostrophe survives", R.romanize("м'яко", "uk") == "m'yako", R.romanize("м'яко", "uk"))
check("a typographic apostrophe is the same character",
      R.romanize("м’яко", "uk") == R.romanize("м'яко", "uk"), R.romanize("м’яко", "uk"))
check("a quote mark around a word is not", R.romanize("'слово'", "uk") == "slovo",
      R.romanize("'слово'", "uk"))


# --------------------------------------------------------------------------- guessing per word
check("ї makes a word Ukrainian", R.guess_language("Україна") == "uk")
check("ы makes a line Russian", R.guess_language("ты и мы") == "ru")
check("latin is english", R.guess_language("hello there") == "en")
check("a word with no marker letter is undecidable", R.guess_language("вода") is None,
      R.guess_language("вода"))
check("undecidable falls back to the lyric's language",
      R.romanize("гора", R.LANG_AUTO, fallback="ru") == "gora")
check("...and the marker still wins over the fallback",
      R.romanize("гори", R.LANG_AUTO, fallback="ru") == "gori" and
      R.romanize("ґоґи", R.LANG_AUTO, fallback="ru") == "gogy",
      R.romanize("ґоґи", R.LANG_AUTO, fallback="ru"))
# The mixed line the author actually writes: one Ukrainian word, one Russian, on the same row.
mixed = R.words("Їжак і ёжик", R.LANG_AUTO, fallback="uk")
check("a mixed line romanizes each word in its own language",
      [w["key"] for w in mixed] == ["yizhak", "i", "yozhik"], [w["key"] for w in mixed])


# ------------------------------------------------------------------- shown vs aligned, and gaps
ws = R.words("Живий! Я живий")
check("punctuation and capitals survive in the DISPLAY text",
      [w["text"] for w in ws] == ["Живий", "Я", "живий"], [w["text"] for w in ws])
check("...while the aligned key is lowercase latin",
      [w["key"] for w in ws] == ["zhyvyy", "ya", "zhyvyy"], [w["key"] for w in ws])

# A word that romanizes to nothing must still BE a word: the caller carries it on a neighbour's
# timing. Dropping it here is how a line loses a word between the lyric sheet and the screen.
odd = R.words("раз 2 три")
check("a digit is a token with an empty key, not a missing word",
      len(odd) == 3 and odd[1]["text"] == "2" and odd[1]["key"] == "", odd)
check("an unspellable word is reported, not hidden", R.spellable("раз 2 три") < 1.0,
      R.spellable("раз 2 три"))
check("a fully spellable line reports 1.0", R.spellable("раз два три") == 1.0)
check("empty text is 1.0 rather than a division by zero", R.spellable("") == 1.0)
check("a line of pure punctuation has no words", R.words("… — !!!") == [], R.words("… — !!!"))

# Latin with diacritics: the decomposition drops the mark and keeps the letter, so a borrowed word
# aligns as its plain spelling instead of vanishing.
check("é decomposes to e", R.romanize("café", "en") == "cafe", R.romanize("café", "en"))
check("a ligature expands", R.romanize("cœur", "en") == "coeur", R.romanize("cœur", "en"))


# ------------------------------------------------------------------------------ never raises
for junk in [None, "", 123, "🔥🔥", "\n\t  ", "'''", "-–—"]:
    try:
        R.words(junk)
        R.romanize(junk)
        R.spellable(junk)
    except Exception as e:
        check(f"junk input {junk!r} does not raise", False, e)
check("junk input never raises", True)

check.done()
