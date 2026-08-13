"""Siren Score: reading a real lyric sheet into a plan.

The fixture is a real song — a real marker layout, with the traps that layout actually contains:
`[Pre-chorus - …]` (whose own hyphen breaks a naive split), `[Bridge/Chaos - …]`, headers that
describe only the drums while the voice hides in the annotation line below, and a member whose name
is two words.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, fake_package, load_module  # noqa: E402

fake_package("kn", "context", "timer", "util", "siren")
load_module("kn.util.anytype", "util/anytype.py")
CardMod = load_module("kn.context.character_card", "context/character_card.py")
load_module("kn.timer.timer_nodes", "timer/timer_nodes.py")
cast = load_module("kn.siren.cast", "siren/cast.py")
score = load_module("kn.siren.score", "siren/score.py")

check = Checker()
Score = score.KinburgSirenScore

LYRICS = """[Intro - deep hypnotic spoken word - MALE vocal]
[cold mechanical electronic beat, rhythmic clicking of a keyboard]
Синій БЛЮКРИЙ... він не мигтить.
Я бачу себе у склі... але це НЕ Я.
Це лише тінь, затиснута між нулями.
Ми — ПРИВИДИ в машині.

[Verse 1 - hypnotic melodic FEMALE vocal]
Ідеальний код, ідеальна ТРАЄКТОРІЯ,
Логічний ланцюг замість пульсу у венах.
Електричний струм — моя НОВА кров,
Ця в'язниця без стін.

[Pre-chorus - building energy, gritty distorted guitars]
[FEMALE melodic vocal shifting to tension]
Свідомість розбита на СЕКТОРІ,
Руки тремтять у пошуках СПРАВЖНЬОГО.
Душа завантажується в холодний СЕРВЕР,
Де кожна емоція — це просто ФАЙЛ!

[Chorus - massive explosion of sound, wall of distorted guitars and fast drums]
[powerful FEMALE scream/belt]
ЦЕ ЦИФРОВИЙ СМЕРЧ! РУЙНУЄ ВСЕ!
Я ХОЧУ ЧУТИ БІТТЯ, А НЕ КОД!
ВІДДАЙ МЕНІ ЖИТТЯ, А НЕ ПІКСЕЛІ!
Я НЕ ФУНКЦІЯ! Я... ЖИВА!

[Bridge/Chaos - aggressive rhythmic MALE vocals, shouting]
[industrial crash sound, heavy blast beats]
ТИ — ЛИШЕ ДАНІ! ТИ — ЛИШЕ СТАТИСТИКА!
СВІТ ПЕРЕТВОРИВСЯ НА МЕРТВУ МАТЕМАТИКУ!
ПРОДУКТИВНІСТЬ! КОНТРОЛЬ!
МИ — ЦИФРИ В ЇХ ТРІЗНЕННІ!

[Outro - whispered FEMALE vocal]
[sudden silence, single distorted low note lingering]
Що залишилося... від МЕНЕ?
"""

GRU = {"name": "Gru BNik", "tags": "male vocal, aggressive, deep, hypnotic", "notes": "",
       "gender": "male"}
KEEN = {"name": "Keen Burg", "tags": "female vocal, melodic, deep, hypnotic", "notes": "",
        "gender": "female"}
ARGS = dict(bpm=145, timesignature="4", pad_to_seconds=60.0, tail_bars="0",
            pad_placement=score.PAD_END, pad_block_bars=16, min_bars=4, instrumental_bars=4,
            arrangement_notes=True, duets=score.DUET_SPLIT, verbose=False)

# --------------------------------------------------------------------------------- reading markers
secs, snotes = score._split_sections(LYRICS)
check("every section is found, in order",
      [s["label"] for s in secs] == ["Intro", "Verse 1", "Pre-Chorus", "Chorus", "Bridge", "Outro"],
      [s["label"] for s in secs])
check("'[Pre-chorus - …]' is NOT read as a Chorus — its own hyphen is the trap",
      secs[2]["label"] == "Pre-Chorus")
check("'[Bridge/Chaos - …]' lands on Bridge and keeps Chaos as marker text",
      secs[4]["label"] == "Bridge" and secs[4]["marker"].startswith("Chaos"), secs[4]["marker"])
check("a verse number is picked up, a chorus needs none",
      secs[1]["label"] == "Verse 1" and secs[3]["label"] == "Chorus")
check("annotation lines are annotations, not lyrics",
      [s["lines"] for s in secs] == [4, 4, 4, 4, 4, 1], [s["lines"] for s in secs])
check("...and they are kept with their section, which is where the voice often hides",
      "powerful FEMALE scream/belt" in secs[3]["notes"], secs[3]["notes"])
check("nothing was reported for a clean sheet", snotes == [], snotes)
check("an unbracketed 'Verse 1:' header works too",
      [s["label"] for s in score._split_sections("Verse 1:\nline one\nChorus:\nline two")[0]]
      == ["Verse 1", "Chorus"])
def _err(fn):
    try:
        fn()
    except Exception as e:
        return str(e)
    return ""


check("no markers at all is refused, and the message says what a marker looks like",
      "[Verse 1 - Nina]" in _err(lambda: Score().run(lyrics="just some words", **ARGS)),
      _err(lambda: Score().run(lyrics="just some words", **ARGS))[:80])

# --------------------------------------------------------------------------------- who sings what
check("'female' is never matched as 'male'",
      score._gender_of("powerful FEMALE scream") == "female"
      and score._gender_of("aggressive MALE vocals") == "male"
      and score._gender_of("no gender here") is None)

plan, report = Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN, **ARGS)
rows, notes = cast._parse_plan(plan, 145, 4)
check("the plan Siren Score writes parses back through Siren Cast's own parser, cleanly",
      len(rows) == 6 and notes == [], (len(rows), notes))
check("MALE in the marker resolves to the one male member",
      [r["voice_raw"] for r in rows][0] == "Gru BNik", rows[0]["voice_raw"])
check("FEMALE resolves to the one female member",
      [r["voice_raw"] for r in rows][1:4] == ["Keen Burg"] * 3, [r["voice_raw"] for r in rows])
check("the Bridge goes back to the male member", rows[4]["voice_raw"] == "Gru BNik")
check("two-word names come out whole, not run together",
      "GruBNik" not in plan and "KeenBurg" not in plan)
check("a name in the marker beats a gender word in it",
      cast._parse_plan(Score().run(lyrics="[Verse 1 - Gru BNik, FEMALE backing]\nline\n",
                                   voice_1=GRU, voice_2=KEEN, **ARGS)[0], 145, 4)[0][0]["voice_raw"]
      == "Gru BNik")
check("'both names' mode keeps the duet as written, in the marker's order",
      cast._parse_plan(Score().run(lyrics="[Chorus - Keen Burg + Gru BNik]\nline\n",
                                   voice_1=GRU, voice_2=KEEN,
                                   **{**ARGS, "duets": score.DUET_ASIS})[0],
                       145, 4)[0][0]["voice_raw"] == "Keen Burg + Gru BNik")
check("with nobody wired, the marker's own words become the vocal description",
      "spoken word" in cast._parse_plan(Score().run(lyrics=LYRICS, **ARGS)[0], 145,
                                        4)[0][0]["voice_raw"], )
check("an ambiguous gender is resolved AND flagged",
      any("2 members match" in n for n in Score().run(
          lyrics="[Verse 1 - FEMALE vocal]\nline\n", voice_1=KEEN,
          voice_2={"name": "Ada", "tags": "female vocal, breathy", "gender": "female"},
          **ARGS)[1].splitlines()))
check("a section whose marker says nothing about a voice gets no vocal",
      cast._parse_plan(Score().run(lyrics="[Solo - twelve-string guitar]\nline\n",
                                   voice_1=GRU, **ARGS)[0], 145, 4)[0][0]["voice_raw"] == "-")

# ---------------------------------------------------------------- round brackets are not lyrics
# A newer lyric style puts production notes in ROUND brackets, and they were being counted as sung
# lines: "(Distorted bassline, haunting atmospheric electric guitar)" gave an instrumental intro 16
# syllables of imaginary singing. Backing echoes live in there too — "(Живий!)" — and those are sung
# OVER the line above rather than after it, so neither kind adds duration.
PARENS = ("[Intro - Instrumental]\n(Distorted bassline, haunting atmospheric electric guitar)\n"
          "[Chorus - Bella]\nСЕРЦЕ!\n(Живий!)\nШлях!\n")
psecs, _ = score._split_sections(PARENS, [{"name": "Bella", "tags": "female vocal"}])
check("an instrumental intro whose only content is a production note has NO sung lines",
      psecs[0]["lines"] == 0 and psecs[0]["syl"] == 0, (psecs[0]["lines"], psecs[0]["syl"]))
check("...and the note is kept, so it can still reach the caption",
      any("Distorted bassline" in n for n in psecs[0]["notes"]), psecs[0]["notes"])
check("a backing echo does not add a line either — it is sung over the line above",
      psecs[1]["lines"] == 2, psecs[1]["lines"])
check("...and its syllables are not counted twice",
      psecs[1]["syl"] == score._syllables("СЕРЦЕ! Шлях!"), psecs[1]["syl"])
check("a line that only CONTAINS brackets is still a lyric",
      score._split_sections("[Verse 1 - x]\nСталеве... (сталеве...)\n")[0][0]["lines"] == 1)

# ------------------------------------------------------------------------- two voices at one time
# The plan carries one audio code per 200 ms and the caption is one description, so a section naming
# two singers asks for a blend and gets one. Per-line markers are the way out: they alternate.
EXCHANGE = ("[Bridge - Gru BNik + Keen Burg - shouted male and piercing female]\n"
            "[frantic chaotic drums, peak breakdown]\n"
            "[Gru BNik - deep aggressive growls]\nЦЕ ХАОС!\n"
            "[Keen Burg - high-pitched piercing screams]\nХТО МЕНЕ ЗАМІНИТЬ?!\n"
            "[Gru BNik - aggressive shouted male vocal]\nСИСТЕМА ПАДАЄ!\n"
            "[Keen Burg - powerful strained female vocal]\nМій розум ТРІЩИТЬ!\n")
ex_plan, ex_rep = Score().run(lyrics=EXCHANGE, voice_1=GRU, voice_2=KEEN, **ARGS)
ex_rows, _ = cast._parse_plan(ex_plan, 145, 4)
check("inner markers split the section, one sub-section per voice",
      len(ex_rows) == 4 and all(r["label"] == "Bridge" for r in ex_rows),
      [r["label"] for r in ex_rows])
check("...and the voices alternate, which is the thing the model CAN do",
      [r["voice_raw"] for r in ex_rows] == ["Gru BNik", "Keen Burg", "Gru BNik", "Keen Burg"],
      [r["voice_raw"] for r in ex_rows])
check("the header that only announced the exchange did not become an empty section too",
      not any("no sung lines" in n for n in ex_rep.splitlines()),
      [n for n in ex_rep.splitlines() if "⚠" in n])
BAR145 = 60 / 145 * 4
BAR145 = 60 / 145 * 4
tight = Score().run(lyrics=EXCHANGE, voice_1=GRU, voice_2=KEEN,
                    **{**ARGS, "pad_to_seconds": 14.0})[0]
check("a one-line exchange may go down to 2 bars, where a whole section's floor is min_bars",
      min(int(ln.split("|")[2].strip().split()[0]) for ln in tight.splitlines() if "|" in ln) == 2,
      [int(ln.split("|")[2].strip().split()[0]) for ln in tight.splitlines() if "|" in ln])
check("the parent's arrangement note goes to the FIRST sub-section only, not all four",
      sum("frantic chaotic drums" in r["extra"] for r in ex_rows) == 1,
      [r["extra"] for r in ex_rows])
check("no band member's name reaches any caption, not even the one that isn't singing there",
      not any(n in r["extra"] for r in ex_rows for n in ("Gru", "Keen", "BNik", "Burg")),
      [r["extra"] for r in ex_rows])
check("'both names' mode does not split at all — it is the A/B baseline",
      len(cast._parse_plan(Score().run(lyrics=EXCHANGE, voice_1=GRU, voice_2=KEEN,
                                       **{**ARGS, "duets": score.DUET_ASIS})[0], 145, 4)[0]) == 1)

duet_plan, duet_rep = Score().run(lyrics="[Chorus - Keen Burg + Gru BNik - powerful harmonies]\n"
                                         "l1\nl2\nl3\nl4\n", voice_1=GRU, voice_2=KEEN, **ARGS)
duet_row = cast._parse_plan(duet_plan, 145, 4)[0][0]
check("a header duet with nothing to split on gives the FIRST-named singer the lead",
      duet_row["voice_raw"] == "Keen Burg", duet_row["voice_raw"])
check("...and the others become a short note, in words, about the harmonies",
      "with male backing harmonies" in duet_row["extra"], duet_row["extra"])
check("...and it is said out loud, with what to write instead",
      any("come back as a blend" in n and "alternate" in n for n in duet_rep.splitlines()),
      [n for n in duet_rep.splitlines() if "⚠" in n])
check("the backing note survives even with arrangement_notes off — it is not decoration",
      "with male backing harmonies" in cast._parse_plan(
          Score().run(lyrics="[Chorus - Keen Burg + Gru BNik]\nl1\nl2\n", voice_1=GRU, voice_2=KEEN,
                      **{**ARGS, "arrangement_notes": False})[0], 145, 4)[0][0]["extra"])

# ------------------------------------------------------------- a long caption is not a detailed one
LONG = ("[Chorus - Keen Burg - powerful female vocal]\n[one, two, three, four, five, six]\n"
        "l1\nl2\n")
long_row = cast._parse_plan(Score().run(lyrics=LONG, voice_1=KEEN, **ARGS)[0], 145, 4)[0][0]
check(f"a caption addition is capped at {score.MAX_CLAUSES} clauses",
      len(long_row["extra"].split(",")) == score.MAX_CLAUSES, long_row["extra"])
check("...and the dropped ones are reported, not silently lost",
      any("clause(s) past the first" in n
          for n in Score().run(lyrics=LONG, voice_1=KEEN, **ARGS)[1].splitlines()))
check("the report states the rate the plan ACTUALLY came out at",
      "syllables/s over the sung sections" in Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN,
                               **ARGS)[1].splitlines()[0])

# ------------------------------------------------------------------------------------ how long
BAR145 = 60 / 145 * 4
check("syllables are counted per vowel group — exact for Slavic",
      score._syllables("Синій БЛЮКРИЙ... він не мигтить.") == 8,
      score._syllables("Синій БЛЮКРИЙ... він не мигтить."))
check("...and English's silent final 'e' is not counted",
      score._syllables("time the same") == 3 and score._syllables("e") == 1,
      score._syllables("time the same"))

# ------------------------------------------------------------- lengths run backwards from the target
# The old model asked for a singing RATE and let the song come out however long it came out. That put
# the slack between the words and the wanted length INSIDE the vocal sections, where the model fills
# it rather than singing slower — a 6-second intro came back 40 seconds long. Now the length is given
# and the words are shared across it, so slack can only land where it was asked for: the tail.
for weights, floors, avail in (([60, 42, 49, 67, 11, 9, 6, 7, 49, 8], [2] * 10, 60),
                               ([10, 10], [2, 2], 20), ([100, 1], [2, 2], 40),
                               ([5], [4], 30), ([3, 3, 3], [2, 2, 2], 7)):
    sh = score._apportion(weights, floors, avail)
    label = f"{len(weights)} section(s) over {avail} bars"
    check(f"[{label}] every share is even — the plan's unit is 2 bars",
          all(b % 2 == 0 for b in sh), sh)
    check(f"[{label}] no share is below its floor", all(b >= f for b, f in zip(sh, floors)), sh)
    check(f"[{label}] the budget is spent, not approximated",
          sum(sh) == avail - avail % 2 or sum(sh) == sum(max(2, f) for f in floors), (sum(sh), avail))
    check(f"[{label}] more syllables never get fewer bars",
          all(sh[i] >= sh[j] for i in range(len(sh)) for j in range(len(sh))
              if weights[i] > weights[j] and floors[i] <= floors[j]), list(zip(weights, sh)))
check("the same lyric always apportions the same way",
      score._apportion([7, 7, 7], [2, 2, 2], 20) == score._apportion([7, 7, 7], [2, 2, 2], 20))
check("floors that cannot fit come back as floors, for the caller to report",
      score._apportion([1, 1, 1], [8, 8, 8], 6) == [8, 8, 8])

plan60 = Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN, **ARGS)[0]
plan120 = Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN,
                      **{**ARGS, "pad_to_seconds": 120.0})[0]


def _bars(plan_):
    return [int(ln.split("|")[2].strip().split()[0]) for ln in plan_.splitlines() if "|" in ln]


check("the plan lands on the target length, to the bar",
      abs(sum(_bars(plan60)) * BAR145 - 60.0) <= BAR145, sum(_bars(plan60)) * BAR145)
check("doubling the target roughly doubles the bars — the words spread, they are not repeated",
      1.8 <= sum(_bars(plan120)) / sum(_bars(plan60)) <= 2.2,
      (sum(_bars(plan60)), sum(_bars(plan120))))
check("a longer target makes every sung section longer, none shorter",
      all(b120 >= b60 for b60, b120 in zip(_bars(plan60), _bars(plan120))),
      (_bars(plan60), _bars(plan120)))

instr = Score().run(lyrics="[Intro - drums]\n[Verse 1 - Gru BNik]\nline one\nline two\n",
                    voice_1=GRU, **{**ARGS, "instrumental_bars": 8, "pad_to_seconds": 60.0})[0]
check("a section with no sung lines takes instrumental_bars and is NOT in the proportional share",
      _bars(instr)[0] == 8, _bars(instr))
check("...so the rest of the target goes to the words",
      abs(sum(_bars(instr)) * BAR145 - 60.0) <= BAR145, sum(_bars(instr)) * BAR145)

# The rate is an output now, and it is the number that says whether the target suits the lyric.
slow = Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN,
                   **{**ARGS, "pad_to_seconds": 200.0})[1]
check("a target far too long for the words is called out by the rate it produces",
      any("syllables/s" in n and "below" in n for n in slow.splitlines()),
      [n for n in slow.splitlines() if "⚠" in n][:1])
check("...and it names the length this lyric would actually suit",
      any("wants about" in n for n in slow.splitlines()))
check("...and points at the roomiest section, so it is obvious where the room went",
      any("roomiest is" in n for n in slow.splitlines()))
fast = Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN,
                   **{**ARGS, "pad_to_seconds": 24.0, "min_bars": 2})[1]
check("a target far too short says the words are crammed, the other way round",
      any("crammed" in n and "tightest is" in n for n in fast.splitlines()),
      [n for n in fast.splitlines() if "⚠" in n][:1])
check("a target that suits the lyric says nothing about the rate",
      not any("syllables/s" in n for n in
              Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN,
                          **{**ARGS, "pad_to_seconds": 75.0})[1].splitlines() if "⚠" in n),
      [n for n in Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN,
                              **{**ARGS, "pad_to_seconds": 75.0})[1].splitlines() if "⚠" in n])
check("the report prints the rate it came out at, since nobody chose it",
      "syllables/s over the sung sections" in
      Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN, **ARGS)[1].splitlines()[0])
check("the floor sits between the rate that worked and the rate that got filled",
      2.34 < score.SANE_RATE[0] < 3.28, score.SANE_RATE)

check("a plain field takes the time signature a text parser hands it — a dropdown cannot",
      [score._beats(x) for x in ("4", 4, " 4/4 ", "3", "", None)] == [4, 4, 4, 3, 4, 4])

# ----------------------------------------------------------------------------- the instrumental tail
# The tail used to be a residual: target duration minus what the lyrics were worth, which is how a
# six-section lyric ended up with forty bars behind it. It is now an explicit small number, because
# with `lyrics_in_negative` off a long tail is actively bad — the model repeats the last phrase to
# fill it and clips the ends of held notes, while 2-8 bars buys a last chorus and a proper ending.
PAD = dict(ARGS, bpm=120, pad_to_seconds=120.0, tail_bars="32", pad_block_bars=16)


def _tail_rows(**kw):
    plan = Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN, **{**PAD, **kw})[0]
    rows_ = cast._parse_plan(plan, 120, 4)[0]
    return rows_, [r for r in rows_ if r["label"] in score.PAD_LABELS and r["voice_raw"] == "-"]


base, _ = _tail_rows(tail_bars="0")
check("tail 0 adds nothing at all",
      not any(r["label"] in score.PAD_LABELS for r in base), [r["label"] for r in base])
for choice in score.TAIL_BARS:
    rows_, pads = _tail_rows(tail_bars=choice)
    want = int(choice)
    got = sum(round(r["asked_seconds"] / 2.0) for r in pads)
    check(f"tail {choice} bars comes out as {want} bars over {len(pads)} row(s)",
          got == want and len(pads) <= max(1, -(-want // 16)), (got, len(pads)))
    check(f"...and tail {choice} leaves the sung sections exactly as they were",
          [r["voice_raw"] for r in rows_ if r["label"] not in score.PAD_LABELS]
          == [r["voice_raw"] for r in base], choice)

rows32, pads32 = _tail_rows(tail_bars="32")
check("a 32-bar tail is split, not left as one 32-bar row", len(pads32) == 2)
check("...and it alternates Solo / Break",
      [r["label"] for r in pads32] == ["Solo", "Break"], [r["label"] for r in pads32])
rep32 = Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN, **PAD)[1]
check("the header separates what the lyrics are worth from what was added",
      "+32 bars instrumental" in rep32.splitlines()[0], rep32.splitlines()[0])
check("...and a tail this long is called out, with what it does to the take",
      any("is a long tail" in n and "repeat the last phrase" in n for n in rep32.splitlines()),
      [n for n in rep32.splitlines() if "tail" in n])
rep4 = Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN, **{**PAD, "tail_bars": "4"})[1]
check("a short tail draws no complaint — it is the point of the dial",
      not any("is a long tail" in n for n in rep4.splitlines()))
check("...and is still reported as added, so it is never a surprise",
      any("bars of instrumental were added" in n for n in rep4.splitlines()))
check("the dial offers 0 and a musical spread, nothing in between to fiddle with",
      score.TAIL_BARS[0] == "0" and all(int(b) % 2 == 0 for b in score.TAIL_BARS[1:])
      and int(score.TAIL_BARS[-1]) == 32, score.TAIL_BARS)


def _shape(**kw):
    """Section labels of the plan, instrumental ones starred."""
    p_ = Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN, **{**PAD, **kw})[0]
    return [ln.split("|")[0].strip() + ("*" if ln.split("|")[1].strip() == "-" else "")
            for ln in p_.splitlines() if "|" in ln]


ends = _shape(pad_placement=score.PAD_ENDS)
check("'intro + outro' still opens with instrumental and ends with it",
      ends[0].endswith("*") and ends[-1].endswith("*"), ends)
btwn = _shape(pad_placement=score.PAD_BETWEEN)
check("'between sections' still opens with one and puts the rest after a chorus or a bridge",
      btwn[0].endswith("*")
      and any(btwn[i + 1].endswith("*") for i, x in enumerate(btwn[:-1])
              if x.split()[0] in ("Chorus", "Bridge")), btwn)
check("every placement adds the same number of bars — only the position differs",
      len([x for x in ends if x.endswith("*")]) == len([x for x in btwn if x.endswith("*")]))

# ----------------------------------------------------------------------- redundancy in the caption
check("the marker's delivery note survives when it adds something",
      "spoken word" in rows[0]["extra"], rows[0]["extra"])
check("...and 'MALE vocal' does not, because the card already says it",
      "MALE vocal" not in rows[0]["extra"], rows[0]["extra"])
check("a marker that only repeats the card leaves no note at all",
      rows[1]["extra"] == "", rows[1]["extra"])
check("'whispered' is the whole point of the outro's marker and it is kept",
      "whispered" in rows[5]["extra"], rows[5]["extra"])
check("arrangement text is kept whatever it says",
      "wall of distorted guitars" in rows[3]["extra"], rows[3]["extra"])
check("trimming is fragment by fragment, so one new word keeps only its own fragment",
      score._trim_redundant("deep hypnotic spoken word, MALE vocal", GRU["tags"])
      == "deep hypnotic spoken word")
check("with no tags to compare against, nothing is trimmed",
      score._trim_redundant("male vocal", "") == "male vocal")
check("a member's NAME never reaches the caption — it is a credit, not a music description",
      cast._parse_plan(Score().run(lyrics="[Verse 1 - Keen Burg - hypnotic melodic female vocal]\n"
                                          "l1\nl2\nl3\nl4\n", voice_1=GRU, voice_2=KEEN,
                                   **ARGS)[0], 145, 4)[0][0]["extra"] == "",
      cast._parse_plan(Score().run(lyrics="[Verse 1 - Keen Burg - hypnotic melodic female vocal]\n"
                                          "l1\n", voice_1=GRU, voice_2=KEEN, **ARGS)[0],
                       145, 4)[0][0]["extra"])
check("...and a two-word name goes whole, not leaving 'Burg' stranded",
      "Burg" not in score._trim_redundant("Keen Burg - powerful harmonies", "female vocal",
                                          ["Keen Burg"]),
      score._trim_redundant("Keen Burg - powerful harmonies", "female vocal", ["Keen Burg"]))
check("a duet marker keeps only what describes the sound — the backing note plus the harmonies",
      cast._parse_plan(Score().run(lyrics="[Chorus - Keen Burg + Gru BNik - powerful harmonies]\n"
                                          "l1\nl2\nl3\nl4\n", voice_1=GRU, voice_2=KEEN,
                                   **ARGS)[0], 145, 4)[0][0]["extra"]
      == "with male backing harmonies, powerful harmonies")
check("words that only glue names together go with them",
      score._trim_redundant("Keen Burg and Gru BNik together", "female vocal",
                            ["Keen Burg", "Gru BNik"]) == "")
check("arrangement_notes off leaves the 4th column empty everywhere",
      all(not r["extra"] for r in cast._parse_plan(
          Score().run(lyrics=LYRICS, voice_1=GRU, voice_2=KEEN,
                      **{**ARGS, "arrangement_notes": False})[0], 145, 4)[0]))

# A '[Chorus]' with no lines under it is an LLM meaning "repeat the chorus". AceStep has no repeat,
# so it silently becomes instrumental — in the middle of the song. It has to be said out loud.
_, rep_empty = Score().run(lyrics="[Verse 1 - Keen Burg]\nl1\nl2\n[Chorus]\n", voice_1=KEEN, **ARGS)
check("a vocal section with no sung lines is reported as the 'repeat' mistake it usually is",
      any("Chorus has no sung lines" in n and "no repeat" in n for n in rep_empty.splitlines()),
      [n for n in rep_empty.splitlines() if "⚠" in n])
_, rep_instr = Score().run(lyrics="[Intro - drums]\n[Verse 1 - Keen Burg]\nl1\n[Solo]\n[Outro]\n",
                           voice_1=KEEN, **ARGS)
check("...but an Intro / Solo / Outro with no lines is legitimately instrumental and stays quiet",
      not any("no sung lines" in n for n in rep_instr.splitlines()),
      [n for n in rep_instr.splitlines() if "⚠" in n])


# The card is where a voice is described; Score has to read exactly what it emits.
_, voice = CardMod.CharacterCard().run(name="Keen Burg", gender="female",
                                       voice_tags="female vocal, airy")
check("the card's voice carries the gender Score needs for a nameless marker",
      voice.get("gender") == "female", voice)
check("...and that is enough to resolve '[Chorus - FEMALE belt]'",
      cast._parse_plan(Score().run(lyrics="[Chorus - FEMALE belt]\nline\n", voice_1=voice,
                                   **ARGS)[0], 145, 4)[0][0]["voice_raw"] == "Keen Burg")

check.done()
