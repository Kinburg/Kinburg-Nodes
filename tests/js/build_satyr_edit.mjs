import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STUBS } from "./stubs.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "..", "web") + path.sep;
const OUT = path.join(HERE, "run_satyr_edit.mjs");
// Harness for web/satyr_edit.js (the node) and the model half of web/satyr_roll.js (the editor).
//
// The node's job is bookkeeping, and every piece of it decides which plan YuE2 gets:
//   * plan_state is the backend input, carried by the panel itself — the auto-created text row must
//     be gone and the panel must serialise both plans;
//   * a run hands the upstream plan back, a frozen run hands back nothing and must not wipe it;
//   * the editor opens on what the node will SEND — the edit while use_edited is on, the upstream
//     otherwise — and saving stores the edit and turns use_edited on.
// The editor's canvas is checked by eye; what it computes from the model is checked here.

const strip = (p) => fs.readFileSync(p, "utf8")
  .split("\n").filter((l) => !/^import\s/.test(l)).join("\n")
  .replace(/^export\s+(function|const|async function)/gm, "$1");

const EXTRA = String.raw`
globalThis.alert = (m) => { ALERTS.push(m); };
const ALERTS = [];
const ROUTES = [];
const REPLY = {
  "/kinburg/satyr/edit/load": (b) => ({ ok: true, model: { echo: b.abc } }),
  "/kinburg/satyr/edit/save": (b) => ({ ok: true, abc: "SAVED:" + b.base, report: ["12 bars · 0:24.0", "kept"],
                                        model: { saved: true } }),
  "/kinburg/satyr/edit/words": (b) => ({ ok: true, sections: [], lines: [], singers: {}, findings: [], echo: b }),
};
api.fetchApi = (route, opt) => {
  const body = JSON.parse(opt.body);
  ROUTES.push([route, body]);
  return Promise.resolve({ status: 200, statusText: "OK", json: () => Promise.resolve(REPLY[route](body)) });
};
const OPENED_EDITORS = [];
`;

const TESTS = String.raw`
const fails = [];
const check = (label, cond, extra) => {
  console.log((cond ? "  ok   " : "  FAIL ") + label + (extra !== undefined ? "  " + extra : ""));
  if (!cond) fails.push(label);
};
openEditor = (opts) => { OPENED_EDITORS.push(opts); return { close() {} }; };

// ── the editor's arithmetic ────────────────────────────────────────────────────────────────────
const model = {
  unit: [1, 16], bpm: 120,
  bars: [{ w: 16, m: [4, 4], o: 0 }, { w: 16, m: [4, 4], o: 1 }, { w: 16, m: [4, 4], o: 2 }, { w: 8, m: [2, 4], o: 3 }],
  sections: [{ label: "verse", n: 2, o: 0 }, { label: "", n: 0, o: 1 }, { label: "chorus", n: 2, o: 2 }],
  notes: [
    { v: 0, t: 0, d: 4, p: 60, j: 0 }, { v: 0, t: 4, d: 4, p: 62, j: 0 }, { v: 0, t: 8, d: 12, p: 64, j: 0 },
    { v: 0, t: 32, d: 4, p: 76, j: 0 }, { v: 0, t: 36, d: 4, p: 77, j: 1 },
    { v: 1, t: 0, d: 32, p: 48, j: 0 },
  ],
  chords: [{ t: 0, c: "Am" }, { t: 8, c: "" }, { t: 32, c: "F/C" }],
  keys: [{ t: 0, k: "Am" }, { t: 32, k: "Em" }],
  bands: { split: 70, low: [60, 64], high: [76, 77] },
};
const L = layout(model);
check("bars start where the widths say", JSON.stringify(L.starts) === "[0,16,32,48]" && L.total === 56,
      JSON.stringify(L.starts));
check("a unit of 1/16 at 120 bpm lasts 1/8 s", Math.abs(L.unitSec - 0.125) < 1e-12, L.unitSec);
check("sections span their bars, an empty one is a point",
      JSON.stringify(L.sections.map((s) => [s.label, s.from, s.to])) === '[["verse",0,32],["",32,32],["chorus",32,56]]',
      JSON.stringify(L.sections.map((s) => [s.label, s.from, s.to])));
check("a chord lasts until the next change, and '' is no chord",
      JSON.stringify(L.chords.map((c) => [c.t, c.end, c.value])) === '[[0,8,"Am"],[8,32,""],[32,56,"F/C"]]');
check("keys are spans too", JSON.stringify(L.keys.map((k) => [k.t, k.end, k.value])) === '[[0,32,"Am"],[32,56,"Em"]]');
check("a section boundary starts a new phrase even without a rest", L.phrases.length === 2, L.phrases.length);
check("each phrase takes the band its median sits in",
      L.phrases.map((p) => p.band).join() === "low,high" && model.notes[3]._band === "high" && model.notes[0]._band === "low",
      L.phrases.map((p) => p.band).join());
const rested = layout({ ...model, notes: [{ v: 0, t: 0, d: 2, p: 60, j: 0 }, { v: 0, t: 6, d: 2, p: 60, j: 0 },
                                          { v: 0, t: 9, d: 2, p: 60, j: 0 }], bands: null });
check("a rest of a beat or more splits a phrase, a shorter one does not", rested.phrases.length === 2,
      rested.phrases.map((p) => p.notes.length).join());
check("one band, no colouring by band", rested.phrases.every((p) => p.band === null));
check("barAt finds the bar a unit falls in", barAt(L.starts, 0) === 0 && barAt(L.starts, 31) === 1 &&
      barAt(L.starts, 32) === 2 && barAt(L.starts, 55) === 3);
check("chord tones: a minor triad under middle C", JSON.stringify(chordTones("Am")) === "[57,60,64]",
      JSON.stringify(chordTones("Am")));
check("...a slash bass goes an octave lower still", JSON.stringify(chordTones("F/C")) === "[36,53,57,60]",
      JSON.stringify(chordTones("F/C")));
check("...the exporter's own qualities are known", JSON.stringify(chordTones("C#m7b5")) === "[49,52,55,59]" &&
      JSON.stringify(chordTones("Bbmaj7")) === "[58,62,65,69]" && JSON.stringify(chordTones("Gm(maj7)")) === "[55,58,62,66]");
check("...an unknown quality still sounds its triad, and junk sounds nothing",
      chordTones("Dadd9").length === 3 && chordTones("N.C.").length === 0);
check("pitch names: middle C is C4", pitchName(60) === "C4" && pitchName(69) === "A4" && pitchName(59) === "B3");
check("time reads as m:ss.s", mmss(0) === "0:00.0" && mmss(83.25) === "1:23.3" && mmss(-1) === "0:00.0", mmss(83.25));

// ── the edits ──────────────────────────────────────────────────────────────────────────────────
// Every operation must leave the model the way satyr/edit.py expects it: these are those rules.
const broken = (m) => {
  const S = barStarts(m), total = S.at(-1), errs = [];
  if (m.sections.reduce((a, s) => a + s.n, 0) !== m.bars.length) errs.push("sections do not add up to the bars");
  for (const n of m.notes) if (n.t < 0 || n.d < 1 || n.t + n.d > total) errs.push("a note outside the song " + JSON.stringify(n));
  for (const v of [0, 1]) {
    const ns = m.notes.filter((n) => n.v === v).sort((a, b) => a.t - b.t);
    for (let i = 1; i < ns.length; i++) if (ns[i - 1].t + ns[i - 1].d > ns[i].t) errs.push("two notes at once in voice " + v);
    for (let i = 0; i < ns.length; i++) if (ns[i].j && !(i && ns[i - 1].t + ns[i - 1].d === ns[i].t)) errs.push("a join with nothing before it");
  }
  if (!m.keys.length || m.keys[0].t !== 0) errs.push("no key at the start");
  for (const [list, f] of [["keys", "k"], ["chords", "c"]]) {
    for (let i = 1; i < m[list].length; i++) {
      if (m[list][i].t <= m[list][i - 1].t) errs.push(list + " out of order");
      if (m[list][i][f] === m[list][i - 1][f]) errs.push(list + " repeat a value");
    }
    for (const e of m[list]) if (e.t < 0 || e.t >= total) errs.push(list + " past the end");
  }
  return errs;
};
// intro(2) verse(2) chorus(2) outro(1), 4/4 at L:1/16; a Vocal note held from the verse into the
// chorus; the chorus in E minor; chords on every bar.
const song = () => ({
  unit: [1, 16], bpm: 96, key: "D#m",
  bars: Array.from({ length: 7 }, (_, i) => ({ w: 16, m: [4, 4], o: i })),
  sections: [{ label: "intro", n: 2, o: 0 }, { label: "verse", n: 2, o: 1 }, { label: "chorus", n: 2, o: 2 }, { label: "outro", n: 1, o: 3 }],
  notes: [
    { v: 1, t: 0, d: 8, p: 51, j: 0 }, { v: 1, t: 16, d: 8, p: 54, j: 0 },
    { v: 0, t: 32, d: 4, p: 60, j: 0 }, { v: 0, t: 36, d: 4, p: 62, j: 1 }, { v: 0, t: 48, d: 4, p: 63, j: 0 },
    { v: 0, t: 60, d: 8, p: 65, j: 0 },                                  // held from the verse into the chorus
    { v: 0, t: 72, d: 4, p: 74, j: 0 }, { v: 0, t: 80, d: 8, p: 76, j: 0 },
    { v: 1, t: 96, d: 16, p: 39, j: 0 },
  ],
  chords: [{ t: 0, c: "D#m" }, { t: 16, c: "B" }, { t: 32, c: "F#" }, { t: 48, c: "C#" }, { t: 64, c: "Em" }, { t: 96, c: "" }],
  keys: [{ t: 0, k: "D#m" }, { t: 64, k: "Em" }],
  labels: ["intro", "verse", "chorus", "outro", "bridge"], bands: null,
});
const vocal = (m) => m.notes.filter((n) => n.v === 0).sort((a, b) => a.t - b.t).map((n) => [n.t, n.d, n.p]);
const ev = (list) => list.map((e) => [e.t, e.k ?? e.c]);
const labels = (m) => m.sections.map((s) => s.label + ":" + s.n).join(" ");
const okAfter = (label, m) => { const e = broken(m); check(label + " — and the model is still whole", !e.length, e.join("; ")); };
check("the fixture itself is whole", !broken(song()).length, broken(song()).join("; "));

let m = deleteBars(song(), 2, 4);
check("deleting the verse removes its bars and its section", m.bars.length === 5 && labels(m) === "intro:2 chorus:2 outro:1", labels(m));
check("...and every note that started in it, the held one too", JSON.stringify(vocal(m)) === "[[40,4,74],[48,8,76]]",
      JSON.stringify(vocal(m)));
check("...the chorus keeps its key and its chord from its first beat", JSON.stringify(ev(m.keys)) === '[[0,"D#m"],[32,"Em"]]'
      && m.chords.find((c) => c.t === 32)?.c === "Em", JSON.stringify(ev(m.keys)) + JSON.stringify(ev(m.chords)));
check("...and the bars that stay keep their origins", m.bars.map((b) => b.o).join() === "0,1,4,5,6");
okAfter("delete", m);

m = deleteBars(song(), 4, 6);
check("deleting the chorus stops the note held into it at its edge", vocal(m).some(([t, d]) => t === 60 && d === 4),
      JSON.stringify(vocal(m)));
check("...and the outro keeps the key it was in, the chorus's E minor", JSON.stringify(ev(m.keys)) === '[[0,"D#m"],[64,"Em"]]',
      JSON.stringify(ev(m.keys)));
okAfter("delete held", m);

m = deleteBars(song(), 0, 2);
check("deleting from the start leaves a key at 0", m.keys[0].t === 0 && m.keys[0].k === "D#m");
check("...and the verse opens on the chord it had", m.chords[0].t === 0 && m.chords[0].c === "F#", JSON.stringify(ev(m.chords)));
okAfter("delete start", m);

m = insertBars(song(), 2, 1);
check("an empty bar inserted before the verse joins the verse", labels(m) === "intro:2 verse:3 chorus:2 outro:1", labels(m));
check("...is new to the plan", m.bars[2].o === null && m.bars[3].o === 2);
check("...holds no note, and everything after moves along", !m.notes.some((n) => n.t >= 32 && n.t < 48)
      && vocal(m)[0][0] === 48, JSON.stringify(vocal(m)));
check("...and carries the chord of the bar before it", inForce(m.chords, "c", 32) === "B" && inForce(m.chords, "c", 48) === "F#");
okAfter("insert", m);

m = duplicateBars(song(), 2, 4);
check("duplicated verse bars follow it, inside the verse", labels(m) === "intro:2 verse:4 chorus:2 outro:1", labels(m));
check("...as bars new to the plan", m.bars.slice(4, 6).every((b) => b.o === null) && m.bars[6].o === 4);
check("...with the same notes, the held one cut at the copy's end", JSON.stringify(vocal(m).filter(([t]) => t >= 64 && t < 96))
      === "[[64,4,60],[68,4,62],[80,4,63],[92,4,65]]", JSON.stringify(vocal(m)));
check("...the original held note now stops where the copy begins", vocal(m).some(([t, d]) => t === 60 && d === 4));
check("...and the chorus still starts in its own key after the copy", inForce(m.keys, "k", 96) === "Em" && inForce(m.keys, "k", 80) === "D#m");
okAfter("duplicate", m);

m = song();
check("moving the chorus before the verse", moveSection(m, 2, -1) === true && labels(m) === "intro:2 chorus:2 verse:2 outro:1", labels(m));
check("...takes its bars, origins and all", m.bars.map((b) => b.o).join() === "0,1,4,5,2,3,6");
check("...takes its key and gives the verse its own back", inForce(m.keys, "k", 32) === "Em" && inForce(m.keys, "k", 64) === "D#m"
      && inForce(m.keys, "k", 96) === "Em", JSON.stringify(ev(m.keys)));
check("...takes its notes", vocal(m).filter(([t]) => t >= 32 && t < 64).map(([, , p]) => p).join() === "74,76",
      JSON.stringify(vocal(m)));
okAfter("move", m);
check("the first section cannot move earlier", moveSection(song(), 0, -1) === false);
m = song(); moveSection(m, 2, -1); moveSection(m, 1, 1);
check("...and moving it back restores the order and the bars", labels(m) === labels(song()) && m.bars.map((b) => b.o).join() === "0,1,2,3,4,5,6");

m = duplicateSection(song(), 1);
check("a duplicated section is a section of its own, new to the plan", labels(m) === "intro:2 verse:2 verse:2 chorus:2 outro:1"
      && m.sections[2].o === null, labels(m));
okAfter("duplicate section", m);
m = deleteSection(song(), 3);
check("deleting the outro", labels(m) === "intro:2 verse:2 chorus:2" && m.bars.length === 6);
okAfter("delete section", m);
m = renameSection(song(), 3, "  fade-out ");
check("a section is renamed, spaces tidied", m.sections[3].label === "fade-out");

check("keys: D# minor up a whole tone is F minor", transposeKey("D#m", 2) === "Fm");
check("...and back", transposeKey("Fm", -2) === "D#m");
check("...named the way SheetSage2 names them", transposeKey("F#", 1) === "G" && transposeKey("C", 6) === "F#"
      && transposeKey("Bbm", 1) === "Bm" && transposeKey("Am", 3) === "Cm" && transposeKey("C", -1) === "B");
check("...an unreadable key is left alone", transposeKey("Dmix", 2) === "Dmix");
check("flat keys know they are flat", keyFlats("Fm") && keyFlats("F") && keyFlats("Dm") && keyFlats("Bb")
      && !keyFlats("D#m") && !keyFlats("C") && !keyFlats("E"));
check("chords keep their quality and bass", transposeChord("C#m7b5", 1, true) === "Dm7b5"
      && transposeChord("F/C", 1, false) === "F#/C#" && transposeChord("F/C", 1, true) === "Gb/Db"
      && transposeChord("Bbmaj7", 2, false) === "Cmaj7" && transposeChord("Gm(maj7)", -2, true) === "Fm(maj7)");
check("...and something that is not a chord is left alone", transposeChord("N.C.", 3, false) === "N.C.");

m = transposeSong(song(), 2);
check("the whole song up a whole tone: every note", vocal(m).map(([, , p]) => p).join() === "62,64,65,67,76,78");
check("...every key", JSON.stringify(ev(m.keys)) === '[[0,"Fm"],[64,"F#m"]]', JSON.stringify(ev(m.keys)));
check("...every chord, spelt for the key it lands in", JSON.stringify(ev(m.chords)) ===
      '[[0,"Fm"],[16,"Db"],[32,"Ab"],[48,"Eb"],[64,"F#m"],[96,""]]', JSON.stringify(ev(m.chords)));
okAfter("transpose song", m);

m = transposeRange(song(), 4, 6, 1);
check("the chorus alone up a semitone: its notes move, the rest do not",
      vocal(m).map(([, , p]) => p).join() === "60,62,63,65,75,77" && m.notes.find((n) => n.t === 96).p === 39,
      vocal(m).map(([, , p]) => p).join());
check("...its key modulates and the outro gets the old one back", JSON.stringify(ev(m.keys)) === '[[0,"D#m"],[64,"Fm"],[96,"Em"]]',
      JSON.stringify(ev(m.keys)));
check("...and so do its chords", inForce(m.chords, "c", 64) === "Fm" && inForce(m.chords, "c", 96) === "",
      JSON.stringify(ev(m.chords)));
okAfter("transpose range", m);

m = song();
const held = m.notes.find((n) => n.t === 60), next = m.notes.find((n) => n.t === 72);
next.t = 64;                              // dragged back over the end of the held note
enforceMono(m, [next]);
check("a note dragged over another cuts it back", held.d === 4 && broken(m).length === 0, JSON.stringify(vocal(m)));
m = song();
const cover = { v: 0, t: 28, d: 16, p: 70, j: 0 };
addNote(m, cover);
check("a long note drawn over others swallows what it covers and trims what it starts on",
      JSON.stringify(vocal(m).filter(([t]) => t < 48)) === "[[28,16,70]]" , JSON.stringify(vocal(m)));
m = song();
addNote(m, { v: 0, t: 58, d: 4, p: 70, j: 0 });
check("a note drawn into the start of another pushes that one's start back",
      JSON.stringify(vocal(m).filter(([t]) => t >= 56 && t < 72)) === "[[58,4,70],[62,6,65]]", JSON.stringify(vocal(m)));
okAfter("draw", m);

m = song();
const second = m.notes.find((n) => n.t === 36);
check("a joined note can be unjoined", toggleJoin(m, second) === true && second.j === 0);
check("...and joined again", toggleJoin(m, second) === true && second.j === 1);
check("a note with a gap before it cannot be joined", toggleJoin(m, m.notes.find((n) => n.t === 48)) === false);
m = song(); m.notes.find((n) => n.t === 32).t = 30;
fixJoins(m);
check("a join left with nothing right before it is dropped", m.notes.find((n) => n.t === 36).j === 0);

m = setChord(song(), 8, "G#m");
check("a chord set mid-bar is a change there", inForce(m.chords, "c", 8) === "G#m" && inForce(m.chords, "c", 7) === "D#m");
m = setChord(song(), 8, "D#m");
check("...setting the chord already in force adds nothing", m.chords.length === song().chords.length);
m = removeChordAt(song(), 20);
check("removing a chord lets the one before carry on", inForce(m.chords, "c", 20) === "D#m" && m.chords.length === 5,
      JSON.stringify(ev(m.chords)));

m = song(); m.notes[2]._band = "low";
check("what goes over the wire leaves the editor's own notes behind", !("_band" in plain(m).notes[2]) && plain(m).notes.length === 9);
m = song(); m.notes.push({ v: 1, t: 100, d: 30, p: 40, j: 0 }); m.keys = [{ t: 16, k: "D#m" }];
tidy(m);
check("tidy cuts a note at the end of the song and restores a key at 0",
      m.notes.at(-1).d === 12 - 0 && m.keys[0].t === 0, JSON.stringify(m.notes.at(-1)) + JSON.stringify(m.keys));

// ── the words on the notes ─────────────────────────────────────────────────────────────────────
// The verse has three onsets (32, 48, 60 — the note at 36 is joined to 32), each opening a phrase of
// its own, and four syllables in two lines; the chorus two onsets and two syllables. The intro and
// the outro have no notes.
const syl = (s, first, last, line, block) => ({ s, word: s, first, last, line, block, backing: false, voice: "" });
const WORDS = { lines: ["сире ни", "та", "кохан"], singers: { high: ["Keen"], low: ["Burg"] }, findings: [],
  blocks: [{ label: "Intro" }, { label: "Verse" }, { label: "Chorus" }],
  sections: [
    { label: "intro", voice: "", band: null, blocks: [], syllables: [] },
    { label: "verse", voice: "Burg", band: "low", blocks: ["Verse"],
      syllables: [syl("си", true, false, 0, 1), syl("ре", false, true, 0, 1), syl("ни", true, true, 0, 1), syl("та", true, true, 1, 1)] },
    { label: "chorus", voice: "Keen", band: "high", blocks: ["Chorus"], syllables: [syl("ко", true, false, 2, 2), syl("хан", false, true, 2, 2)] },
    { label: "outro", voice: "", band: null, blocks: [], syllables: [] },
  ] };
m = song();
const at = (t) => m.notes.find((n) => n.v === 0 && n.t === t);
const sungOn = (ts) => ts.map((t) => at(t)._syl?.s || (at(t)._hold ? "~" : ".")).join();
let L3 = layout(m, WORDS);
check("the lyric goes onto the song's phrases line by line, squeezed onto fewer notes where it must",
      sungOn([32, 48, 60, 72, 80]) === "сире,ни,та,ко,хан", sungOn([32, 48, 60, 72, 80]));
check("...a joined note carries the syllable before it and takes none", at(36)._syl === null && !at(36)._hold);
check("...and the Ins voice is never given words", m.notes.filter((n) => n.v === 1).every((n) => !n._syl));
check("each lyric block knows its notes against its syllables",
      L3.words.map((w) => w.label + ":" + w.notes + "/" + w.wants).join() === "Verse:3/4,Chorus:2/2",
      L3.words.map((w) => w.label + ":" + w.notes + "/" + w.wants).join());
check("...and the blocks are the song's sections by words: the first opens the song, each runs to the next",
      L3.words.map((w) => w.from + "-" + w.to).join() === "0-72,72-112", L3.words.map((w) => w.from + "-" + w.to).join());
check("the words a selection sings, as text", wordsOf([at(48), at(32), at(60)]) === "сире ни / та"
      && wordsOf(m.notes) === "сире ни / та / кохан", wordsOf(m.notes));
check("words laid on a different set of sections are not forced onto this one",
      layout(m, { ...WORDS, sections: WORDS.sections.slice(1) }).words === null && m.notes.every((n) => !n._syl));
const mb = { ...song(), bands: { split: 70, low: [60, 65], high: [74, 76] } };
check("a block reports the band its notes sit in", layout(mb, WORDS).words.map((w) => w.was).join() === "low,high",
      layout(mb, WORDS).words.map((w) => w.was).join());
// What the render showed: words the plan wrote no notes for are sung over the music — an intro's over
// its two silent bars, a bridge over an interlude — and take no notes from what follows.
const INTRO = { ...WORDS, lines: [...WORDS.lines, "три години ночі"], sections: WORDS.sections.map((s, i) => (i === 0
  ? { ...s, voice: "Keen", syllables: ["три", "го", "ди", "ни", "но", "чі"].map((x) => syl(x, true, true, 3, 0)) } : s)) };
L3 = layout(m, INTRO);
check("lines with no notes of their own are sung over a silence, and the notes after keep their words",
      L3.free.length === 1 && L3.free[0].from === 0 && L3.free[0].to === 32 && L3.free[0].syl.map((s) => s.s).join() === "три,го,ди,ни,но,чі"
      && sungOn([32, 48, 60]) === "сире,ни,та", JSON.stringify(L3.free.map((f) => [f.from, f.to])) + " " + sungOn([32, 48, 60]));
check("...and their singer's run spans it all the same", L3.runs.some((r) => r.free && r.from === 0 && r.to === 32));
// A chorus with three notes more than its two syllables, all in one phrase.
const crowded = () => {
  const c = song();
  c.notes.push({ v: 0, t: 76, d: 4, p: 74, j: 0 }, { v: 0, t: 88, d: 2, p: 78, j: 0 }, { v: 0, t: 90, d: 2, p: 79, j: 0 });
  return c;
};
m = crowded();
layout(m, WORDS);
check("a line given more notes than syllables holds its last over the rest",
      sungOn([72, 76, 80, 88, 90]) === "ко,хан,~,~,~", sungOn([72, 76, 80, 88, 90]));
m = song();
deleteNotes(m, [at(48)]);
check("delete a note and the words close up on the notes left",
      layout(m, WORDS).words[0].notes === 2 && sungOn([32, 60]) === "сире ни,та", sungOn([32, 60]));
m = song();
m.sections[1].sings = -1;
layout(m, WORDS);
check("a section pinned to nothing sings nothing, its words go elsewhere", [32, 48, 60].every((t) => !at(t)._syl));
const one = { unit: [1, 16], bpm: 96, key: "C", bars: [{ w: 16, m: [4, 4], o: 0 }], sections: [{ label: "verse", n: 1, o: 0 }],
  notes: [{ v: 0, t: 0, d: 1, p: 60, j: 0 }, { v: 0, t: 1, d: 1, p: 62, j: 0 }, { v: 0, t: 2, d: 6, p: 64, j: 0 }],
  chords: [], keys: [{ t: 0, k: "C" }], labels: ["verse"], bands: null };
layout(one, { lines: ["c d e f g"], singers: {}, findings: [], blocks: [{ label: "Verse" }],
  sections: [{ label: "verse", voice: "", band: null, blocks: ["Verse"], syllables: ["c", "d", "e", "f", "g"].map((s) => syl(s, true, true, 0, 0)) }] });
check("squeezed, the longer notes take more syllables", one.notes.map((n) => n._syl.s).join("|") === "c|d|e f g",
      one.notes.map((n) => n._syl?.s).join("|"));

// ── lines onto phrases ─────────────────────────────────────────────────────────────────────────
check("lines go onto phrases in the runs whose counts agree: a line broken by a breath spans two phrases",
      JSON.stringify(groupLines([12, 12], [6, 6, 8, 5])) === "[[0,1,0,2],[1,2,2,4]]", JSON.stringify(groupLines([12, 12], [6, 6, 8, 5])));
check("...and two short lines share a phrase", JSON.stringify(groupLines([13, 12], [26])) === "[[0,2,0,1]]");
check("lines the phrases cannot hold go over a silence long enough to sing them, the rest stay on their notes",
      JSON.stringify(groupLines([10, 16, 16], [11, { room: 45 }])) === "[[0,1,0,1],[1,3,1,2]]",
      JSON.stringify(groupLines([10, 16, 16], [11, { room: 45 }])));
check("...a silence nobody needs is passed over", JSON.stringify(groupLines([6, 6], [{ room: 40 }, 6, 6])) === "[[0,1,1,2],[1,2,2,3]]",
      JSON.stringify(groupLines([6, 6], [{ room: 40 }, 6, 6])));
check("...one to one when they already agree", JSON.stringify(groupLines([5, 7], [5, 7])) === "[[0,1,0,1],[1,2,1,2]]");
const pinned = { item: [{ pin: null, section: 0 }, { pin: 1, section: 1 }], line: [{ block: 0, in: null }, { block: 1, in: 1 }] };
check("a pin keeps a block's lines in its section", JSON.stringify(groupLines([6, 6], [12, 6], pinned)) === "[[0,1,0,1],[1,2,1,2]]",
      JSON.stringify(groupLines([6, 6], [12, 6], pinned)));

// ── the Vocal rhythm lock ──────────────────────────────────────────────────────────────────────
// What the lock compares: everything about the sung notes but their pitch.
m = song();
const rhythm0 = vocalRhythm(m);
transposeNotes(m, m.notes.filter((n) => n.v === 0), 12);
m.notes.find((n) => n.v === 1).t += 2;
check("the rhythm the lock guards ignores pitches and the Ins voice", vocalRhythm(m) === rhythm0);
m = insertBars(song(), 0, 1);
check("...and whole bars coming in before the notes", vocalRhythm(m) === rhythm0);
m = song();
const r1 = vocalRhythm(deleteNotes(m, [m.notes.find((n) => n.v === 0 && n.t === 48)]));
m = song();
m.notes.find((n) => n.v === 0 && n.t === 48).t += 2;
const r2 = vocalRhythm(m);
m = song();
toggleJoin(m, m.notes.find((n) => n.v === 0 && n.t === 36));
check("...but not a sung note removed, moved or unjoined", r1 !== rhythm0 && r2 !== rhythm0 && vocalRhythm(m) !== rhythm0);

// ── who sings where ────────────────────────────────────────────────────────────────────────────
const RUNW = { lines: WORDS.lines, findings: [], singers: { high: ["Keen"], low: ["Burg"] },
  blocks: [{ label: "Intro" }, { label: "Verse" }, { label: "Chorus" }],
  sections: WORDS.sections.map((s, i) => ({ ...s, syllables: s.syllables.map((y) => ({ ...y, unit: i === 1 ? 0 : 1, block: i, voice: i === 1 ? "Burg" : "Keen" })) })) };
m = { ...song(), bands: { split: 70, low: [60, 65], high: [74, 76] } };
let runs = layout(m, RUNW).runs;
check("the lyric's runs fall where they are sung, joined notes riding along",
      runs.map((r) => r.label + ":" + r.voice + ":" + r.from + "-" + r.to + ":" + r.notes.length).join() === "Verse:Burg:32-68:4,Chorus:Keen:72-88:2",
      runs.map((r) => r.label + ":" + r.voice + ":" + r.from + "-" + r.to + ":" + r.notes.length).join());
check("...each knowing the band its singer belongs in and the band its notes sit in",
      runs.map((r) => r.wants + "/" + r.band).join() === "low/low,high/high" && !runs.some(runClash),
      runs.map((r) => r.wants + "/" + r.band).join());
const SWAPW = { ...RUNW, singers: { high: ["Burg"], low: ["Keen"] } };
runs = layout(m, SWAPW).runs;
check("...and a run sung in the other singer's band says so", runs.every(runClash));

// ── the lyric sheet ────────────────────────────────────────────────────────────────────────────
const SHEET = "[Verse - Burg]\n(soft guitar)\n  сире ни  \n\nта\n[Chorus - Keen]\nкохан\n";
const sheet = sungRows(SHEET, ["сире ни", "нема такого", "та", "кохан"]);
check("the sung lines are found in the lyric as written, a line it cannot find skipped over",
      JSON.stringify([...sheet.at]) === "[[2,0],[4,2],[6,3]]" && sheet.rows.length === 8, JSON.stringify([...sheet.at]));

// ── the node ───────────────────────────────────────────────────────────────────────────────────
const ext = EXTS.find((e) => e.name === "Kinburg.SatyrEdit");
check("the extension registers", !!ext);
function NodeType() {}
await ext.beforeRegisterNodeDef(NodeType, { name: "KinburgSatyrEdit" });
const mk = () => {
  const n = new NodeType();
  n.title = "Satyr Edit";
  const changed = [];
  n.widgets = [{ name: "use_edited", value: false, callback: (v) => changed.push(v) },
               { name: "plan_state", value: "" }];
  n.addDOMWidget = (name, type, el, opts) => { n._dom = { name, type, el, opts }; n.widgets.push({ name, ...opts }); };
  n.setDirtyCanvas = () => {};
  n.graph = { change: () => { n._changed = (n._changed || 0) + 1; } };
  n._useChanged = changed;
  n.onNodeCreated();
  return n;
};
const node = mk();
const state = () => JSON.parse(node._dom.opts.getValue());
const ui = node._kbUi;
check("the auto-created plan_state row is gone", node.widgets.filter((w) => w.name === "plan_state").length === 1
      && node.widgets.find((w) => w.name === "plan_state").type === undefined);
check("the panel IS plan_state, and it serialises", node._dom.name === "plan_state" && node._dom.opts.serialize === true);
check("a fresh node has no plan, and says how to get one",
      state().upstream === "" && ui.status.textContent.includes("run the graph") && ui.open.disabled === true,
      ui.status.textContent);

node.onExecuted({ satyr_plan: [{ upstream: "PLAN-A", summary: "12 bars" }] });
check("a run hands the upstream plan to the node", state().upstream === "PLAN-A" && state().upstreamSummary === "12 bars");
check("...and the panel says it is sending it", ui.status.textContent.startsWith("sending the upstream plan")
      && ui.open.disabled === false, ui.status.textContent);
node.onExecuted({ satyr_plan: [{ upstream: null, summary: null }] });
check("a frozen run hands back nothing, and the upstream is kept", state().upstream === "PLAN-A");
node.onExecuted({ satyr_plan: [{ upstream: null, summary: null, lyrics: "[Verse - Burg]\nсире ни\n\n[Chorus - Keen]\nкохан",
                                 voices: [{ name: "Keen" }, { name: "Burg" }] }] });
check("the lyrics and voices that came with a run are kept on the node",
      state().lyrics.startsWith("[Verse - Burg]") && state().voices.map((v) => v.name).join() === "Keen,Burg");
check("...and the panel says so", ui.words.textContent === "lyrics: 2 lines · voices: Keen, Burg", ui.words.textContent);
node.onExecuted({ something_else: [1] });
check("a message for someone else is ignored", state().upstream === "PLAN-A");

ui.open.fire("click");
await tick(); await tick();
check("not frozen: the editor opens on the upstream plan",
      ROUTES.at(-1)[0] === "/kinburg/satyr/edit/load" && ROUTES.at(-1)[1].abc === "PLAN-A"
      && OPENED_EDITORS.at(-1).base === "PLAN-A" && OPENED_EDITORS.at(-1).model.echo === "PLAN-A");
check("...with the lyrics", OPENED_EDITORS.at(-1).lyrics === state().lyrics);
const asked = await OPENED_EDITORS.at(-1).words({ m: 2 }, "PLAN-A");
check("...and a way to have them laid onto the edit, lyrics and voices included",
      ROUTES.at(-1)[0] === "/kinburg/satyr/edit/words" && asked.echo.base === "PLAN-A" && asked.echo.model.m === 2
      && asked.echo.lyrics === state().lyrics && asked.echo.voices.length === 2);
const got = await OPENED_EDITORS.at(-1).save({ m: 1 }, "PLAN-A");
check("saving posts the model against the plan it was opened on",
      ROUTES.at(-1)[0] === "/kinburg/satyr/edit/save" && ROUTES.at(-1)[1].base === "PLAN-A" && ROUTES.at(-1)[1].model.m === 1);
check("...stores the edit on the node", state().edited === "SAVED:PLAN-A" && state().editedSummary === "12 bars · 0:24.0");
check("...turns use_edited on, through its callback", node.widgets[0].value === true && node._useChanged.at(-1) === true);
check("...tells the graph it changed", node._changed >= 1);
check("...and hands the editor the plan to carry on from", got.abc === "SAVED:PLAN-A" && got.model.saved === true);
check("the panel now says the edit is going out", ui.status.textContent.startsWith("🔒 sending the edited plan"),
      ui.status.textContent);
check("...and what the save kept and re-rendered, where it cannot be missed",
      ui.saved.textContent === "last save: kept" && state().editedReport === "kept", ui.saved.textContent);

ui.open.fire("click");
await tick(); await tick();
check("frozen: the editor opens on the edit, not the upstream", OPENED_EDITORS.at(-1).base === "SAVED:PLAN-A");

node.widgets[0].value = false;
node.widgets[0].callback(false);
check("unfrozen with an edit kept: the panel says so", ui.status.textContent.startsWith("sending the upstream plan")
      && ui.note.textContent.includes("an edit is kept"), ui.note.textContent);
ui.open.fire("click");
await tick(); await tick();
check("...and the editor opens on the upstream again", OPENED_EDITORS.at(-1).base === "PLAN-A");

const copy = mk();
copy._dom.opts.setValue(node._dom.opts.getValue());
check("a workflow reload restores both plans", JSON.parse(copy._dom.opts.getValue()).edited === "SAVED:PLAN-A"
      && JSON.parse(copy._dom.opts.getValue()).upstream === "PLAN-A");
copy._dom.opts.setValue("{not json");
check("a mangled value loads as empty instead of throwing", JSON.parse(copy._dom.opts.getValue()).upstream === "");

node.widgets[0].value = true;
ui.discard.fire("click", { preventDefault() {} });
check("discarding drops the edit and unfreezes", state().edited === "" && node.widgets[0].value === false
      && state().upstream === "PLAN-A");
check("...and its save report with it", state().editedReport === "" && ui.saved.textContent === "");

const empty = mk();
empty._kbUi.open.fire("click");
await tick();
check("no plan at all: opening says why instead of calling the server", ALERTS.length === 1
      && ALERTS[0].includes("run the graph"), ALERTS[0]);

console.log("\n" + (fails.length ? "FAILED: " + fails.join(", ") : "ALL PASS"));
process.exit(fails.length ? 1 : 0);
`;

fs.writeFileSync(OUT, [STUBS, EXTRA, strip(WEB + "satyr_roll.js"), strip(WEB + "satyr_edit.js"), TESTS].join("\n"));
console.log("wrote " + OUT);
