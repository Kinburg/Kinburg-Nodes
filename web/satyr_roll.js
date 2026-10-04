// Satyr Edit — the piano roll a YuE2 plan is edited in.
//
// Opened by web/satyr_edit.js with the model satyr/edit.py reads out of a plan: bars (width in L:
// units, meter, origin), sections (label, bar count), notes {v, t, d, p, j} with t and d in units,
// and chord and key changes {t, c} / {t, k}. ABC never reaches this file. Nothing here talks to
// ComfyUI either, so the model operations run in node for the tests.
//
// The band line: asked for two singers, YuE2 writes them into the one Vocal line as two pitch bands
// (satyr/bands.py), so the line is drawn and every Vocal phrase is coloured by the side its median
// sits on. That shows who the model wrote a phrase for. It is not a switch — moving a phrase across
// flipped the singer once at a fixed seed and never reliably since; the style decides the singers.
//
// Every edit goes through one of the model operations below and leaves the model in the shape
// satyr/edit.py expects: sections adding up to the bars, notes inside the song and one at a time per
// voice, a key at 0, chord and key changes sorted with no repeats. Bars that were in the plan keep
// their origin `o`; bars made here have none, which is how the writer knows to render them.

export const VOICES = ["Vocal", "Ins"];
const NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
const PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

// Geometry of the canvas, in CSS pixels.
const GUTTER = 54, BLK_H = 19, PLAN_H = 14, SEC_H = BLK_H + PLAN_H, RULER_H = 22, CHORD_H = 20, SING_H = 18, LYR_H = 18, OVER_H = 38;
const SING_Y = SEC_H + RULER_H + CHORD_H;
const LYR_Y = SING_Y + SING_H;
const TOP = LYR_Y + LYR_H;

const COLORS = {
  bg: "#15171c", white: "#1e2127", black: "#191b20", cRow: "#2b2f37",
  bar: "#4a505c", beat: "#31363f", unit: "#252930",
  vocalLow: "#f0a24e", vocalHigh: "#ef7fa6", vocal: "#f0a24e", ins: "#4fb0bd",
  lowTint: "rgba(240,162,78,0.045)", highTint: "rgba(239,127,166,0.05)", split: "#c56a8a",
  text: "#c9ced8", dim: "#7d8492", cursor: "#e8e8e8", play: "#7CFC9A",
  secA: "#2a2f3a", secB: "#242832", chord: "#9fb4d8", key: "#e3c27a",
  pick: "#ffffff", range: "rgba(120,160,255,0.10)", rangeEdge: "rgba(120,160,255,0.55)",
  lyric: "#eadfbe", backing: "#9fb4d8", unsung: "#e3a24a", clash: "#ff8a80",
  runHigh: "rgba(239,127,166,0.22)", runLow: "rgba(240,162,78,0.22)", runNone: "rgba(160,170,190,0.14)",
};

export function pitchName(p) {
  return NAMES_SHARP[((p % 12) + 12) % 12] + (Math.floor(p / 12) - 1);
}

export function mmss(sec) {
  sec = Math.max(0, sec);
  const m = Math.floor(sec / 60), s = sec - m * 60;
  return `${m}:${s < 10 ? "0" : ""}${s.toFixed(1)}`;
}

// ------------------------------------------------------------------------------------- layout
// Everything derived from the model that the drawing, the hit-testing and the playback share.
// `words` is what satyr/edit.py laid onto the plan when the lyrics were last asked for.
export function layout(model, words) {
  const starts = [];
  let total = 0;
  for (const b of model.bars) { starts.push(total); total += b.w; }
  const [un, ud] = model.unit;
  const unitSec = (un / ud) * 4 * 60 / model.bpm;
  const sections = [];
  let b0 = 0;
  for (const s of model.sections) {
    const b1 = b0 + s.n;
    sections.push({ label: s.label, b0, b1,
      from: b0 < starts.length ? starts[b0] : total, to: b1 < starts.length ? starts[b1] : total });
    b0 = b1;
  }
  const spans = (events, field) => events.map((e, i) => ({
    t: e.t, end: i + 1 < events.length ? events[i + 1].t : total, value: e[field] }));
  const out = { starts, total, unitSec, sections, chords: spans(model.chords, "c"), keys: spans(model.keys, "k"),
                phrases: phrases(model, starts, sections) };
  const sung = attachWords(model, sections, words, total);
  out.words = sung?.blocks || null;
  out.free = sung?.free || [];
  out.runs = sung ? singerRuns(model, words, out.free) : [];
  return out;
}

// Who sings where: the lyric's runs — the lines one voice sings in a row, or one bracketed backing
// line — laid where their syllables fall, from the first note to the last, joined and held notes
// riding with the syllable before them; lines sung over a silence (`free`) share its length. Each
// knows the band the lyrics' singer belongs in (`wants`) and the band its notes sit in (`band`); the
// two disagreeing is what an octave up or down is for.
export function singerRuns(model, words, free) {
  const runs = [];
  let cur = null;
  for (const n of model.notes.filter((x) => x.v === 0).sort((a, b) => a.t - b.t)) {
    const s = n._syl;
    if (s && s.unit !== cur?.unit) {
      cur = { unit: s.unit, block: s.block, voice: s.voice, backing: s.backing, from: n.t, to: n.t + n.d, notes: [] };
      runs.push(cur);
    } else if (!s && !n.j && !n._hold) {
      cur = null;
    }
    if (!cur) continue;
    cur.notes.push(n);
    cur.to = Math.max(cur.to, n.t + n.d);
  }
  for (const span of free) {
    const length = span.to - span.from;
    let at = 0;
    for (const unit of [...new Set(span.syl.map((s) => s.unit))]) {
      const mine = span.syl.filter((s) => s.unit === unit), s = mine[0];
      runs.push({ unit, block: s.block, voice: s.voice, backing: s.backing, notes: [], free: true,
                  from: span.from + Math.round((length * at) / span.syl.length),
                  to: span.from + Math.round((length * (at + mine.length)) / span.syl.length) });
      at += mine.length;
    }
  }
  runs.sort((a, b) => a.from - b.from);
  const high = words?.singers?.high || [], low = words?.singers?.low || [];
  runs.forEach((r, i) => {
    const bands = r.notes.map((n) => n._band).filter(Boolean);
    r.band = bands.length ? (bands.filter((b) => b === "high").length * 2 >= bands.length ? "high" : "low") : null;
    r.wants = high.includes(r.voice) ? "high" : low.includes(r.voice) ? "low" : null;
    r.label = words?.blocks?.[r.block]?.label || "";
    r.opens = i === 0 || runs[i - 1].block !== r.block;
  });
  return runs;
}

const runClash = (r) => !!(r.band && r.wants && r.band !== r.wants);

// The lyrics on the notes, the way YuE2 sang them on a real render: the whole lyric, line after line,
// onto the whole song's phrases — and onto its silences of two bars or more, where the plan has no
// notes and the model sings over the music. Nothing about a section's label decides it. The model
// had written its intro's lines into the first bars of the section it labelled `verse`, sung the
// first pre-chorus over a long break in that same section, a bridge over an interlude with no notes
// at all, and a chorus of 52 syllables whole before that interlude — each block where its words fit,
// so each block is where its words are (`blocks`, drawn as the song's real sections). How the lines
// go on is groupLines; inside a run of phrases a syllable per note, the last held over any notes
// left (`_hold`), or several on the longer notes when there are too few.
//
// Satyr Score's matcher only carries the block order and the editor's pins: a section pinned to a
// block takes that block's lines and no others, one pinned to nothing takes none. A section's
// `pickup` notes, the ones before its barline that run into it, count as its own for that.
//
// A joined note carries on the syllable before it and takes none. Done here rather than in the
// backend so the words follow the notes while they are dragged. → {blocks: per lyric block, where
// its words are sung, who it is marked for, the band that singer should be in and the band its
// notes sit in, notes against syllables; free: the lines sung over silence, [{from, to, syl}]}, or
// null when there are no words, or they were laid onto a different set of sections.
export function attachWords(model, sections, words, total) {
  for (const n of model.notes) { n._syl = null; n._hold = false; }
  if (!words || !Array.isArray(words.sections) || words.sections.length !== sections.length) return null;
  const vocal = model.notes.filter((n) => n.v === 0 && !n.j).sort((a, b) => a.t - b.t);
  const opens = phraseOpens(model), gap = 2 * model.bars[0].w;      // two bars of silence: room to sing over
  const from = sections.map((s, i) => {
    const before = i ? vocal.filter((n) => n.t >= sections[i - 1].from && n.t < s.from) : [];
    const k = Math.min(words.sections[i].pickup || 0, before.length);
    return k ? before[before.length - k].t : s.from;
  });
  const owner = (t) => from.findLastIndex((f) => f <= t);
  // The song as phrases and silences, in order; a silence ends where a section's words start, and a
  // section with no notes is one silence however short.
  const items = [];
  const silence = (a, b) => {
    while (a < b) {
      const s = Math.max(0, owner(a)), end = Math.min(b, s + 1 < from.length ? from[s + 1] : b);
      const whole = a <= from[s] && end >= (s + 1 < from.length ? from[s + 1] : total);
      if (end - a >= gap || whole) items.push({ from: a, to: end, section: s });
      a = end;
    }
  };
  let end = 0;
  for (const n of vocal) {
    if (items.length && items.at(-1).notes && !opens.has(n)) {
      items.at(-1).notes.push(n);
    } else {
      silence(end, n.t);
      items.push({ notes: [n], section: Math.max(0, owner(n.t)) });
    }
    end = Math.max(end, n.t + n.d);
  }
  silence(end, total);
  const syl = words.sections.flatMap((w) => w.syllables), lines = [];
  syl.forEach((s, k) => { if (!k || s.line !== syl[k - 1].line) lines.push([]); lines.at(-1).push(s); });
  const pinned = model.sections.map((s) => (typeof s.sings === "number" ? s.sings : null));
  const home = new Map(pinned.map((b, s) => [b, s]).filter(([b]) => b !== null && b >= 0));
  const pins = pinned.some((b) => b !== null) ? {
    item: items.map((it) => ({ pin: pinned[it.section], section: it.section })),
    line: lines.map((l) => ({ block: l[0].block, in: home.has(l[0].block) ? home.get(l[0].block) : null })),
  } : null;
  const counts = lines.map((l) => l.length);
  const shape = items.map((it) => (it.notes ? it.notes.length : { room: (it.to - it.from) / 2 }));
  // Pins that cannot all be met are let go rather than leave words with nowhere to go; a song with
  // too few places for its lines has them all on its notes as one run.
  const matched = new Map(words.sections.flatMap((w, s) => w.syllables.map((x) => [x.block, s])));
  const homes = { line: lines.map((l) => matched.get(l[0].block)), item: items.map((it) => it.section) };
  const plan = (pins && groupLines(counts, shape, pins, homes)) || groupLines(counts, shape, null, homes)
    || [[0, lines.length, 0, items.length]];
  const free = [];
  for (const [l0, l1, i0, i1] of plan) {
    const mine = lines.slice(l0, l1).flat();
    const notes = items.slice(i0, i1).filter((it) => it.notes).flatMap((it) => it.notes);
    if (!notes.length) { free.push({ from: items[i0]?.from ?? 0, to: items[i1 - 1]?.to ?? total, syl: mine }); continue; }
    if (mine.length <= notes.length) {
      notes.forEach((n, k) => { n._syl = mine[k] || null; n._hold = k >= mine.length; });
      continue;
    }
    let at = 0;
    shares(mine.length, notes.map((n) => n.d)).forEach((c, k) => {
      const some = mine.slice(at, at + c);
      at += c;
      notes[k]._syl = some.length < 2 ? some[0]
        : { ...some[0], s: some.map((y, q) => (q && y.first ? " " : "") + y.s).join(""), last: some.at(-1).last,
            lines: [...new Set(some.map((y) => y.line))] };
    });
  }
  return { blocks: blockRows(words, vocal, free, syl, total), free };
}

// Per lyric block: where its words begin — the song's real sections, the first one opening the song
// — who it is marked for, the band that singer belongs in and the band its notes sit in, and its
// notes against its syllables.
function blockRows(words, vocal, free, syl, total) {
  const high = words.singers?.high || [], low = words.singers?.low || [];
  const rows = [];
  for (const b of [...new Set(syl.map((s) => s.block))]) {
    const mine = syl.filter((s) => s.block === b);
    const notes = vocal.filter((n) => n._syl?.block === b);
    const spans = free.filter((f) => f.syl.some((s) => s.block === b));
    const bands = notes.map((n) => n._band).filter(Boolean);
    const voice = mine.find((s) => s.voice)?.voice || "";
    rows.push({ block: b, label: words.blocks?.[b]?.label || "", voice, wants: mine.length, notes: notes.length,
                band: high.includes(voice) ? "high" : low.includes(voice) ? "low" : null,
                was: bands.length ? (bands.filter((x) => x === "high").length * 2 >= bands.length ? "high" : "low") : null,
                free: spans.length, from: Math.min(...notes.map((n) => n.t), ...spans.map((f) => f.from)) });
  }
  rows.sort((a, c) => a.from - c.from);
  rows.forEach((r, k) => { r.to = k + 1 < rows.length ? rows[k + 1].from : total; });
  if (rows.length) rows[0].from = 0;
  return rows;
}

// The Vocal onsets that open a phrase: those after a beat or more of silence, and the first.
function phraseOpens(model) {
  const starts = barStarts(model), opens = new Set();
  let end = -Infinity;
  for (const n of model.notes.filter((x) => x.v === 0).sort((a, b) => a.t - b.t)) {
    if (!n.j && n.t - end >= beatUnits(model, model.bars[Math.min(barAt(starts, n.t), model.bars.length - 1)])) opens.add(n);
    end = Math.max(end, n.t + n.d);
  }
  return opens;
}

// Lines onto the song, both kept in order: a run of lines takes a run of phrases (counts of notes,
// with no silence between them), or one silence long enough to sing over (`{room}`, in syllables).
// The runs are chosen so syllables and notes come out as close as they can — a line broken by a
// breath spans two phrases, two short lines share one. Joining more than one costs a little, so a
// pair that already agrees stays a pair. A miss of more than two counts barely more than two: a
// stretch thirty syllables short is one place the model fitted words in, not thirty lines each a
// little off — the render had its first lines exactly on their phrases — though more than two
// syllables a note is a stretch, and costs a tenth for every one past that. A silence takes lines at a
// flat cost while they fit its room — the cost of a two-syllable miss, so words go over silence only
// where the notes really cannot hold them — or none at no cost; a phrase may go without words,
// dearer than holding a syllable over it, so a pickup keeps its words. `pins`, when given, are the editor's:
// `pins.item[q]` is {pin: the block item q's section must sing — or -1, none — or null; section},
// and `pins.line[r]` is {block: line r's block, in: the one section that block is pinned to, or
// null}. `homes`, when given, is a weak hint where counts tie: the section Satyr Score's matcher put
// each line's block in ({line}) and the section of each item ({item}); a line placed elsewhere costs
// a fifth of a syllable — never enough to beat a better fit, as the intro its matcher misplaced shows.
// → [[first line, end, first item, end]] for every run that sings, or null when nothing fits.
export function groupLines(lines, items, pins = null, homes = null) {
  const n = lines.length, m = items.length;
  const L = lines.reduce((acc, x) => { acc.push(acc.at(-1) + x); return acc; }, [0]);
  const miss = (d) => Math.min(d, 2) + 0.05 * Math.max(0, d - 2);
  const away = (i, a, j) => {
    let k = 0;
    if (homes) for (let r = i; r < a; r++) if (homes.line[r] !== homes.item[j]) k += 1;
    return 0.2 * k;
  };
  const best = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(Infinity));
  const back = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(null));
  const relax = (a, b, cost, from) => { if (cost < best[a][b]) { best[a][b] = cost; back[a][b] = from; } };
  // Whether lines i..a-1 may go on items whose pins and sections are `want` and `secs`.
  const fits = (i, a, want, secs) => {
    if (!pins) return true;
    if (want.has(-1) || want.size > 1) return false;
    for (let r = i; r < a; r++) {
      const line = pins.line[r];
      if (want.size && !want.has(line.block)) return false;
      if (line.in !== null && (secs.size > 1 || !secs.has(line.in))) return false;
    }
    return true;
  };
  best[0][0] = 0;
  for (let i = 0; i <= n; i++) {
    for (let j = 0; j < m; j++) {
      const here = best[i][j];
      if (here === Infinity) continue;
      const want = new Set(), secs = new Set();
      const take = (q) => { if (pins) { if (pins.item[q].pin !== null) want.add(pins.item[q].pin); secs.add(pins.item[q].section); } };
      if (typeof items[j] !== "number") {
        relax(i, j + 1, here, [i, j, false]);
        take(j);
        for (let a = i + 1; a <= Math.min(n, i + 24); a++) {
          if (!fits(i, a, want, secs)) continue;
          relax(a, j + 1, here + 2 + 0.05 * Math.max(0, L[a] - L[i] - items[j].room) + 0.5 * (a - i - 1) + away(i, a, j),
                [i, j, true]);
        }
        continue;
      }
      relax(i, j + 1, here + 1 + miss(items[j]), [i, j, false]);
      let notes = 0;
      for (let b = j + 1; b <= Math.min(m, j + 12) && typeof items[b - 1] === "number"; b++) {
        notes += items[b - 1];
        take(b - 1);
        for (let a = i + 1; a <= Math.min(n, i + 12); a++) {
          if (!fits(i, a, want, secs)) continue;
          const crammed = 0.1 * Math.max(0, L[a] - L[i] - 2 * notes);
          relax(a, b, here + miss(Math.abs(L[a] - L[i] - notes)) + crammed + 0.5 * (a - i - 1) + 0.5 * (b - j - 1) + away(i, a, j),
                [i, j, true]);
        }
      }
    }
  }
  if (best[n][m] === Infinity) return null;
  const out = [];
  for (let a = n, b = m; a || b;) {
    const [i, j, sings] = back[a][b];
    if (sings) out.unshift([i, a, j, b]);
    [a, b] = [i, j];
  }
  return out;
}

// `total` as one count per weight, each at least one and the rest by weight — largest remainder, so
// the counts add back up exactly.
function shares(total, weights) {
  const mass = weights.reduce((a, w) => a + w, 0), spare = total - weights.length;
  const exact = weights.map((w) => (spare * w) / mass);
  const out = exact.map((e) => 1 + Math.floor(e));
  const left = total - out.reduce((a, c) => a + c, 0);
  [...exact.keys()].sort((x, y) => (exact[y] % 1) - (exact[x] % 1)).slice(0, left).forEach((k) => { out[k] += 1; });
  return out;
}

// The words a set of notes sings, as text: syllables of one word run together, words spaced, lines
// split by " / ".
export function wordsOf(notes) {
  let out = "", line = null;
  for (const n of [...notes].filter((x) => x._syl).sort((a, b) => a.t - b.t)) {
    const s = n._syl;
    if (out) out += s.line !== line ? " / " : s.first ? " " : "";
    out += s.s;
    line = s.line;
  }
  return out;
}

// Runs of Vocal notes: a new phrase wherever the line rests for a beat or more, and at every section
// boundary — a verse and the chorus after it are often two different singers with no breath between.
// Each phrase gets the band its median sits in, which is what decides who sings it.
function phrases(model, starts, sections) {
  const vocal = model.notes.filter((n) => n.v === 0).sort((a, b) => a.t - b.t);
  const sectionOf = (t) => sections.findIndex((s) => s.from <= t && t < s.to);
  const out = [];
  let cur = null;
  for (const n of vocal) {
    const beat = beatUnits(model, model.bars[barAt(starts, n.t)]);
    const sec = sectionOf(n.t);
    if (cur && n.t - cur.end < beat && sec === cur.section) {
      cur.notes.push(n); cur.end = Math.max(cur.end, n.t + n.d);
    } else {
      cur = { notes: [n], start: n.t, end: n.t + n.d, section: sec };
      out.push(cur);
    }
  }
  for (const ph of out) {
    const ps = ph.notes.map((n) => n.p).sort((a, b) => a - b);
    const mid = ps.length % 2 ? ps[(ps.length - 1) / 2] : (ps[ps.length / 2 - 1] + ps[ps.length / 2]) / 2;
    ph.median = mid;
    ph.band = model.bands ? (mid < model.bands.split ? "low" : "high") : null;
    for (const n of ph.notes) n._band = ph.band;
  }
  return out;
}

export function barAt(starts, t) {
  let lo = 0, hi = starts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (starts[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

function beatUnits(model, bar) {
  return Math.max(1, Math.round(model.unit[1] / (model.unit[0] * bar.m[1])));
}

function valueAt(spans, t) {
  for (let i = spans.length - 1; i >= 0; i--) if (spans[i].t <= t) return spans[i].value;
  return "";
}

// --------------------------------------------------------------------------------- chords → tones
const QUALITY = {
  "": [0, 4, 7], m: [0, 3, 7], dim: [0, 3, 6], aug: [0, 4, 8], "7": [0, 4, 7, 10], maj7: [0, 4, 7, 11],
  m7: [0, 3, 7, 10], dim7: [0, 3, 6, 9], m7b5: [0, 3, 6, 10], sus4: [0, 5, 7], sus2: [0, 2, 7],
  "6": [0, 4, 7, 9], m6: [0, 3, 7, 9], "7sus4": [0, 5, 7, 10], "m(maj7)": [0, 3, 7, 11],
};
const CHORD_RE = /^([A-G])(##|bb|#|b)?(.*?)(?:\/([A-G])(##|bb|#|b)?)?$/;

function accidental(a) {
  return !a ? 0 : a === "#" ? 1 : a === "##" ? 2 : a === "b" ? -1 : -2;
}

function pcOf(letter, acc) {
  return (((PC[letter] + accidental(acc)) % 12) + 12) % 12;
}

// The pitches a chord symbol sounds, voiced low under the melody: root in the octave below middle C,
// a slash bass an octave under that. An unknown quality plays as its triad rather than not at all.
export function chordTones(text) {
  const m = CHORD_RE.exec(String(text || "").trim());
  if (!m) return [];
  const root = pcOf(m[1], m[2]);
  const shape = QUALITY[m[3]] || (m[3].startsWith("m") && !m[3].startsWith("maj") ? QUALITY.m : QUALITY[""]);
  const tones = shape.map((i) => 48 + root + i);
  if (m[4]) tones.unshift(36 + pcOf(m[4], m[5]));
  return tones;
}

// ------------------------------------------------------------------------------------ keys & chords
// Accidental counts of the signatures the format names (satyr/notation.py has the same table), and
// the spelling SheetSage2 itself uses for each tonic — so a transposed key is one the exporter would
// have written.
const MAJOR = { C: 0, G: 1, D: 2, A: 3, E: 4, B: 5, "F#": 6, "C#": 7, F: -1, Bb: -2, Eb: -3, Ab: -4, Db: -5, Gb: -6, Cb: -7 };
const MINOR = { A: 0, E: 1, B: 2, "F#": 3, "C#": 4, "G#": 5, "D#": 6, "A#": 7, D: -1, G: -2, C: -3, F: -4, Bb: -5, Eb: -6, Ab: -7 };
const TONIC_MAJOR = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const TONIC_MINOR = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"];
const KEY_RE = /^([A-G])(#|b)?(m?)$/;

// Does this key write its accidentals as flats? Chord roots follow it, the way the exporter spells them.
export function keyFlats(name) {
  const m = KEY_RE.exec(String(name || "").trim());
  if (!m) return false;
  return ((m[3] ? MINOR : MAJOR)[m[1] + (m[2] || "")] ?? 0) < 0;
}

// A key moved by `semis`, named the way SheetSage2 names that tonic. A key it cannot read (a mode,
// say) is returned unchanged rather than guessed at.
export function transposeKey(name, semis) {
  const m = KEY_RE.exec(String(name || "").trim());
  if (!m) return name;
  const pc = (((pcOf(m[1], m[2]) + semis) % 12) + 12) % 12;
  return m[3] ? TONIC_MINOR[pc] + "m" : TONIC_MAJOR[pc];
}

export function transposeChord(text, semis, flats) {
  const m = CHORD_RE.exec(String(text || "").trim());
  if (!m) return text;
  const names = flats ? NAMES_FLAT : NAMES_SHARP;
  const move = (letter, acc) => names[(((pcOf(letter, acc) + semis) % 12) + 12) % 12];
  return move(m[1], m[2]) + m[3] + (m[4] ? "/" + move(m[4], m[5]) : "");
}

// ------------------------------------------------------------------------------- model operations
export function barStarts(model) {
  const out = [];
  let t = 0;
  for (const b of model.bars) { out.push(t); t += b.w; }
  out.push(t);
  return out;
}

export function sectionSpans(model) {
  let b = 0;
  return model.sections.map((s) => { const r = { b0: b, b1: b + s.n }; b += s.n; return r; });
}

export function sectionOfBar(model, bar) {
  let b = 0;
  for (let i = 0; i < model.sections.length; i++) {
    const n = model.sections[i].n;
    if (bar >= b && bar < b + n) return i;
    b += n;
  }
  return model.sections.length - 1;
}

// The value in force at `t` — an event exactly at `t` included.
export function inForce(events, field, t) {
  let v = "";
  for (const e of events) { if (e.t > t) break; v = e[field]; }
  return v;
}

// Sorted changes inside the song, the later of two at one point kept, repeats dropped — the same rule
// satyr/edit.py applies on the way in.
export function changes(events, field, total) {
  const out = [];
  let last = null;
  for (const e of [...events].sort((a, b) => a.t - b.t)) {
    if (!(e.t >= 0 && e.t < total)) continue;
    if (out.length && out[out.length - 1].t === e.t) {
      out.pop();
      last = out.length ? out[out.length - 1][field] : null;
    }
    if (e[field] !== last) { out.push({ t: e.t, [field]: e[field] }); last = e[field]; }
  }
  return out;
}

// A joined note continues the syllable of the note right before it; anything else is a new one.
export function fixJoins(model) {
  for (const v of [0, 1]) {
    let prev = null;
    for (const n of model.notes.filter((x) => x.v === v).sort((a, b) => a.t - b.t)) {
      if (n.j && !(prev && prev.t + prev.d === n.t)) n.j = 0;
      prev = n;
    }
  }
}

// The Vocal line's rhythm — everything but its pitches: where each sung note starts inside its bar,
// how long it lasts and whether it carries on the syllable before. YuE2 wrote that rhythm for the
// words' stresses, so while it is locked an edit that changes it is refused. Whole bars coming or
// going around the notes leave it as it was; a note added, removed, moved, stretched or joined does not.
export function vocalRhythm(model) {
  const starts = barStarts(model);
  return model.notes.filter((n) => n.v === 0).sort((a, b) => a.t - b.t)
    .map((n) => { const b = barAt(starts, n.t); return `${n.t - starts[b]}+${n.d}${n.j ? "j" : ""}`; }).join(" ");
}

// Put the model back in shape after an edit.
export function tidy(model) {
  const total = barStarts(model).at(-1);
  model.notes = model.notes.filter((n) => n.d >= 1 && n.t >= 0 && n.t < total);
  for (const n of model.notes) if (n.t + n.d > total) n.d = total - n.t;
  let keys = changes(model.keys, "k", total);
  if (!keys.length || keys[0].t > 0) keys = changes([{ t: 0, k: keys[0]?.k ?? model.key ?? "C" }, ...keys], "k", total);
  model.keys = keys;
  model.chords = changes(model.chords, "c", total);
  fixJoins(model);
  return model;
}

// Bars [b0, b1) as a piece that can be put back anywhere: the bars, the notes that START inside (cut
// at its end), and the chord and key in force at its start followed by the changes inside it.
export function extract(model, b0, b1) {
  const S = barStarts(model), t0 = S[b0], t1 = S[b1];
  const events = (list, field) => [{ t: 0, [field]: inForce(model[list], field, t0) },
    ...model[list].filter((e) => e.t > t0 && e.t < t1).map((e) => ({ t: e.t - t0, [field]: e[field] }))];
  return {
    bars: model.bars.slice(b0, b1).map((b) => ({ ...b, m: [...b.m] })),
    width: t1 - t0,
    notes: model.notes.filter((n) => n.t >= t0 && n.t < t1).map((n) => ({ ...n, t: n.t - t0, d: Math.min(n.d, t1 - n.t) })),
    chords: events("chords", "c"),
    keys: events("keys", "k"),
  };
}

// Remove bars [b0, b1). A note that starts inside goes with them; one held into them stops at their
// edge; whatever chord and key were in force at the far edge carry on from the near one.
export function deleteBars(model, b0, b1) {
  if (b1 <= b0) return model;
  const S = barStarts(model), t0 = S[b0], t1 = S[b1], w = t1 - t0;
  const kept = [];
  for (const n of model.notes) {
    if (n.t >= t0 && n.t < t1) continue;
    if (n.t < t0 && n.t + n.d > t0) n.d = t0 - n.t;
    if (n.t >= t1) n.t -= w;
    kept.push(n);
  }
  model.notes = kept;
  for (const [list, field] of [["keys", "k"], ["chords", "c"]]) {
    const carry = inForce(model[list], field, t1);
    const out = model[list].filter((e) => e.t < t0).concat({ t: t0, [field]: carry },
      model[list].filter((e) => e.t >= t1).map((e) => ({ ...e, t: e.t - w })));
    model[list] = out;
  }
  model.bars.splice(b0, b1 - b0);
  let b = 0;
  model.sections = model.sections.filter((s) => {
    const from = b, to = b + s.n;
    b = to;
    if (!s.n) return true;
    s.n -= Math.max(0, Math.min(to, b1) - Math.max(from, b0));
    return s.n > 0;
  });
  return tidy(model);
}

// Put a piece from `extract` in before bar `at`. `place` is {extend: section index} to grow that
// section, or {section: {label, o}, index} for a section of its own at that position. A note held
// across the insertion point stops there; what was in force there resumes after the piece.
export function insertFragment(model, at, frag, place) {
  const S = barStarts(model), ta = S[at], w = frag.width;
  for (const n of model.notes) {
    if (n.t >= ta) n.t += w;
    else if (n.t + n.d > ta) n.d = ta - n.t;
  }
  model.notes.push(...frag.notes.map((n) => ({ ...n, t: n.t + ta })));
  for (const [list, field] of [["keys", "k"], ["chords", "c"]]) {
    const resume = inForce(model[list], field, ta);
    model[list] = model[list].map((e) => (e.t >= ta ? { ...e, t: e.t + w } : e))
      .concat(frag[list].map((e) => ({ ...e, t: e.t + ta })), { t: ta + w, [field]: resume });
  }
  model.bars.splice(at, 0, ...frag.bars);
  if (place.section) model.sections.splice(place.index, 0, { label: place.section.label, n: frag.bars.length, o: place.section.o ?? null });
  else model.sections[place.extend].n += frag.bars.length;
  return tidy(model);
}

// `count` empty bars before bar `at`, inside the section `at` is in, in its meter. The chord and key
// of the bar before carry through them.
export function insertBars(model, at, count) {
  const S = barStarts(model), ref = model.bars[Math.min(at, model.bars.length - 1)];
  const before = Math.max(0, S[at] - 1);
  const frag = {
    bars: Array.from({ length: count }, () => ({ w: ref.w, m: [...ref.m], o: null })),
    width: ref.w * count, notes: [],
    chords: [{ t: 0, c: inForce(model.chords, "c", before) }], keys: [{ t: 0, k: inForce(model.keys, "k", before) }],
  };
  return insertFragment(model, at, frag, { extend: sectionOfBar(model, Math.min(at, model.bars.length - 1)) });
}

export function duplicateBars(model, b0, b1) {
  const frag = extract(model, b0, b1);
  for (const b of frag.bars) b.o = null;
  return insertFragment(model, b1, frag, { extend: sectionOfBar(model, b1 - 1) });
}

export function deleteSection(model, si) {
  const { b0, b1 } = sectionSpans(model)[si];
  if (b1 > b0) return deleteBars(model, b0, b1);
  model.sections.splice(si, 1);
  return model;
}

export function duplicateSection(model, si) {
  const { b0, b1 } = sectionSpans(model)[si], sec = model.sections[si];
  if (b1 === b0) { model.sections.splice(si + 1, 0, { label: sec.label, n: 0, o: null }); return model; }
  const frag = extract(model, b0, b1);
  for (const b of frag.bars) b.o = null;
  return insertFragment(model, b1, frag, { section: { label: sec.label, o: null }, index: si + 1 });
}

// Swap section `si` with its neighbour in direction `dir` (-1 earlier, +1 later). Its bars keep their
// origins, so a section moved whole can still be written back as the lines it came from.
export function moveSection(model, si, dir) {
  const j = si + dir;
  if (j < 0 || j >= model.sections.length) return false;
  const sec = model.sections[si];
  if (!sec.n) {
    model.sections.splice(si, 1);
    model.sections.splice(j, 0, sec);
    return true;
  }
  const { b0, b1 } = sectionSpans(model)[si];
  const frag = extract(model, b0, b1);
  deleteBars(model, b0, b1);
  const spans = sectionSpans(model);
  const at = dir < 0 ? spans[j].b0 : spans[si].b1;
  insertFragment(model, at, frag, { section: { label: sec.label, o: sec.o }, index: j });
  return true;
}

export function renameSection(model, si, label) {
  model.sections[si].label = String(label || "").split(/\s+/).filter(Boolean).join(" ");
  return model;
}

export function transposeNotes(model, notes, semis) {
  for (const n of notes) n.p = Math.max(0, Math.min(127, n.p + semis));
  return model;
}

// The whole song: every note, every key change and every chord, respelled for the key it lands in.
export function transposeSong(model, semis) {
  transposeNotes(model, model.notes, semis);
  model.keys = model.keys.map((e) => ({ t: e.t, k: transposeKey(e.k, semis) }));
  model.chords = model.chords.map((e) => ({
    t: e.t, c: e.c ? transposeChord(e.c, semis, keyFlats(inForce(model.keys, "k", e.t))) : "" }));
  if (model.key) model.key = transposeKey(model.key, semis);
  return tidy(model);
}

// Bars [b0, b1) up or down: their notes, their chords and their key — a modulation for that stretch,
// with the key and the chord that were in force restored right after it.
export function transposeRange(model, b0, b1, semis) {
  const S = barStarts(model), t0 = S[b0], t1 = S[b1], total = S.at(-1);
  for (const n of model.notes) if (n.t >= t0 && n.t < t1) n.p = Math.max(0, Math.min(127, n.p + semis));
  const shift = (list, field, fn) => {
    const first = inForce(model[list], field, t0), after = inForce(model[list], field, t1);
    const out = model[list].filter((e) => e.t < t0 || e.t >= t1)
      .concat({ t: t0, [field]: fn(first, t0) },
        model[list].filter((e) => e.t > t0 && e.t < t1).map((e) => ({ t: e.t, [field]: fn(e[field], e.t) })));
    if (t1 < total && !model[list].some((e) => e.t === t1)) out.push({ t: t1, [field]: after });
    model[list] = changes(out, field, total);
  };
  shift("keys", "k", (k) => transposeKey(k, semis));
  shift("chords", "c", (c, t) => (c ? transposeChord(c, semis, keyFlats(inForce(model.keys, "k", t))) : ""));
  return tidy(model);
}

// A voice sings one note at a time. The `winners` — the notes just moved, drawn or stretched — keep
// their place; any other note of the same voice under them is cut back, cut short from the front,
// or removed if it is covered entirely.
export function enforceMono(model, winners) {
  const win = [...winners].sort((a, b) => a.t - b.t);
  const mine = new Set(win);
  for (const n of model.notes) {
    if (mine.has(n)) continue;
    for (const w of win) {
      if (w.v !== n.v || n.d < 1) continue;
      const end = n.t + n.d, wEnd = w.t + w.d;
      if (n.t < w.t && end > w.t) n.d = w.t - n.t;
      else if (n.t >= w.t && n.t < wEnd) {
        if (end > wEnd) { n.t = wEnd; n.d = end - wEnd; n.j = 0; } else n.d = 0;
      }
    }
  }
  return tidy(model);
}

export function addNote(model, note) {
  model.notes.push(note);
  return enforceMono(model, [note]);
}

export function deleteNotes(model, notes) {
  const gone = new Set(notes);
  model.notes = model.notes.filter((n) => !gone.has(n));
  return tidy(model);
}

// Join a note to the one right before it (one syllable carried onto a new pitch) or undo that. Only a
// note that starts exactly where the previous one of its voice ends can be joined.
export function toggleJoin(model, note) {
  const prev = model.notes.filter((n) => n.v === note.v && n.t < note.t).sort((a, b) => b.t - a.t)[0];
  if (!note.j && !(prev && prev.t + prev.d === note.t)) return false;
  note.j = note.j ? 0 : 1;
  return true;
}

// Set the chord from `t` on ("" = no chord from here).
export function setChord(model, t, text) {
  model.chords = model.chords.filter((e) => e.t !== t).concat({ t, c: String(text || "").trim().replace(/"/g, "") });
  return tidy(model);
}

// Remove the chord change that starts the span `t` is in; the chord before it carries on.
export function removeChordAt(model, t) {
  const start = model.chords.filter((e) => e.t <= t).reduce((a, e) => (a === null || e.t > a ? e.t : a), null);
  if (start === null) return model;
  model.chords = model.chords.filter((e) => e.t !== start);
  return tidy(model);
}

// What goes over the wire: the model without the editor's own bookkeeping (keys starting with "_").
export function plain(model) {
  return JSON.parse(JSON.stringify(model, (k, v) => (k.startsWith("_") ? undefined : v)));
}

// ------------------------------------------------------------------------------------- the editor
export function openEditor(opts) {
  injectStyle();
  const st = {
    model: opts.model, base: opts.base, save: opts.save, title: opts.title || "Satyr Edit",
    ppu: 6, rowH: 11, scrollX: 0, scrollY: 0, cursor: 0, hover: null,
    show: { 0: true, 1: true, chords: true, bands: true },
    play: null, audio: null, dirty: false, saving: false,
    sel: new Set(), barSel: null, secSel: null, undo: [], redo: [], drag: null, inline: null,
    grid: 0, drawVoice: 0,
    lyrics: String(opts.lyrics || ""), wordsFn: opts.words || null, words: null, rev: 0,
    lock: true,
  };
  tidy(st.model);
  st.L = layout(st.model, st.words);
  st.grid = gridChoices(st.model)[Math.min(1, gridChoices(st.model).length - 1)].units;

  const root = el("div", "kbse-overlay");
  const win = el("div", "kbse-window");
  const bar = el("div", "kbse-bar");
  const tools = el("div", "kbse-bar kbse-tools");
  const body = el("div", "kbse-body");
  const main = el("div", "kbse-main");
  const grip = el("div", "kbse-grip");
  const side = el("div", "kbse-side");
  const canvas = el("canvas", "kbse-canvas");
  const toast = el("div", "kbse-toast");
  const foot = el("div", "kbse-foot");
  const hoverLine = el("span", "kbse-hover");
  const msgLine = el("span", "kbse-msg");
  foot.append(hoverLine, msgLine);
  main.append(canvas, toast);
  body.append(main, grip, side);
  win.append(bar, tools, body, foot);
  root.append(win);
  document.body.append(root);
  st.root = root; st.main = main; st.canvas = canvas; st.toast = toast; st.hoverLine = hoverLine; st.msgLine = msgLine;
  st.side = side; st.grip = grip;
  grip.title = "Drag to make the lyrics wider or narrower";
  grip.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    grip.setPointerCapture(e.pointerId);
    const move = (ev) => sideWidth(st, body.getBoundingClientRect().right - ev.clientX - grip.offsetWidth);
    const up = () => {
      grip.removeEventListener("pointermove", move);
      grip.removeEventListener("pointerup", up);
      try { localStorage.setItem(SIDE_KEY, String(st.sideW)); } catch (err) { /* private window */ }
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", up);
  });
  let saved = NaN;
  try { saved = Number(localStorage.getItem(SIDE_KEY)); } catch (err) { /* private window */ }
  sideWidth(st, saved || 280);
  showSide(st, !!st.lyrics.trim());
  buildSide(st);
  // Keys typed into the editor's own fields stop here, so ComfyUI never acts on them behind it.
  root.addEventListener("keydown", (e) => e.stopPropagation());

  // --- row 1: listening and looking
  const title = el("span", "kbse-title");
  title.textContent = "🐐 " + st.title;
  const playBtn = button("▶ Play", "Play from the cursor (Space)", () => (st.play ? stop(st) : play(st)));
  const timeBox = el("span", "kbse-time");
  const toggles = [
    toggle("Vocal", COLORS.vocal, () => st.show[0], (v) => { st.show[0] = v; }),
    toggle("Ins", COLORS.ins, () => st.show[1], (v) => { st.show[1] = v; }),
    toggle("Chords", COLORS.chord, () => st.show.chords, (v) => { st.show.chords = v; }),
    toggle("Bands", COLORS.split, () => st.show.bands, (v) => { st.show.bands = v; }),
  ];
  for (const t of toggles) t.el.title = "Show or hide — a hidden voice is not played and cannot be edited";
  const lyricsTgl = toggle("Lyrics", COLORS.lyric, () => st.sideOn, (v) => showSide(st, v));
  lyricsTgl.el.title = "The lyric sheet beside the plan: what is on screen is lit, the line at the cursor brighter; " +
    "click a line to go to it and pick its notes";
  lyricsTgl.el.addEventListener("click", () => lyricsTgl.flip());
  const stats = el("span", "kbse-stats");
  const saveBtn = button("💾 Save", "Write the plan back to the node", () => doSave(st));
  const closeBtn = button("✕", "Close (Esc)", () => close(st));
  const notesBtn = button("ⓘ", "", () => say(st, (st.words?.findings || []).join(" · ")));
  notesBtn.style.display = "none";
  bar.append(title, playBtn, timeBox, sep(),
    button("−", "Zoom out (Ctrl+wheel)", () => zoom(st, 1 / 1.4, null)),
    button("+", "Zoom in (Ctrl+wheel)", () => zoom(st, 1.4, null)),
    button("Fit", "Show the whole song", () => fitAll(st)), sep(), ...toggles.map((t) => t.el), lyricsTgl.el, notesBtn,
    el("span", "kbse-spacer"), stats, saveBtn, closeBtn);
  for (const t of toggles) t.el.addEventListener("click", () => { t.flip(); prune(st); draw(st); });

  // --- row 2: editing
  const undoBtn = button("↶", "Undo (Ctrl+Z)", () => undo(st));
  const redoBtn = button("↷", "Redo (Ctrl+Y)", () => redo(st));
  const grid = select(gridChoices(st.model).map((g) => [g.units, g.label]), st.grid, (v) => { st.grid = Number(v); });
  grid.title = "The grid notes snap to when drawn, moved or stretched";
  const drawInto = select([[0, "Vocal"], [1, "Ins"]], 0, (v) => { st.drawVoice = Number(v); });
  drawInto.title = "Which voice a double-click draws into";
  const tempo = el("input", "kbse-num");
  tempo.type = "number"; tempo.min = "20"; tempo.max = "400"; tempo.step = "1";
  tempo.title = "Tempo, quarter notes per minute — changes how long every bar lasts";
  tempo.addEventListener("change", () => {
    const bpm = Math.round(Number(tempo.value));
    if (!(bpm >= 20 && bpm <= 400) || bpm === st.model.bpm) { tempo.value = st.model.bpm; return; }
    edit(st, (m) => { m.bpm = bpm; });
  });
  const ctx = el("span", "kbse-ctx");
  const lockBtn = button("", "Locked: Vocal notes keep their timing — they go up and down, nothing else. YuE2 wrote " +
    "that rhythm for the words' stresses: a sung note added, removed, moved, stretched, joined or split shifts which " +
    "syllable falls on which beat, and the model sings the line wrong. Whole bars can still come and go, and Ins " +
    "notes move freely.\n\nUnlock to change the Vocal rhythm anyway.", () => {
    st.lock = !st.lock;
    say(st, st.lock ? "🔒 Vocal rhythm locked: sung notes go up and down only"
                    : "🔓 Vocal rhythm unlocked: sung notes can be moved, stretched, drawn and deleted");
    refresh(st);
  });
  tools.append(undoBtn, redoBtn, sep(), lockBtn, sep(), label("Grid"), grid, label("Draw into"), drawInto, sep(),
    label("Transpose"),
    ...[[-12, "An octave down"], [-1, "A semitone down"], [1, "A semitone up"], [12, "An octave up"]].map(([s, what]) =>
      button(s > 0 ? `+${s}` : `−${-s}`, `${what}: the selected notes; or the selected bars or section with their chords ` +
        "and key; or, with nothing selected, the whole song", () => transpose(st, s))),
    sep(), label("♩="), tempo, sep(), ctx);
  st.ui = { playBtn, timeBox, stats, saveBtn, undoBtn, redoBtn, tempo, ctx, toggles, notesBtn, lockBtn };

  // --- canvas events
  canvas.addEventListener("wheel", (e) => onWheel(st, e), { passive: false });
  canvas.addEventListener("pointerdown", (e) => onDown(st, e));
  canvas.addEventListener("pointermove", (e) => onMove(st, e));
  canvas.addEventListener("pointerup", (e) => onUp(st, e));
  canvas.addEventListener("dblclick", (e) => onDouble(st, e));
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerleave", () => { st.hover = null; showHover(st); });
  st.onKey = (e) => onKey(st, e);
  window.addEventListener("keydown", st.onKey, true);
  st.ro = new ResizeObserver(() => resize(st));
  st.ro.observe(main);

  resize(st);
  fitAll(st);
  centerPitches(st);
  refresh(st);
  say(st, (st.model.warnings || []).length ? "⚠ " + st.model.warnings.join(" · ") : "");
  askWords(st);
  return { close: () => close(st), state: st };
}

// ------------------------------------------------------------------------------ the lyric sheet
// The whole lyric beside the roll, as written — markers, directions and all — at a width the viewer
// drags to and keeps. A line the plan sings is lit while its notes are on screen, brighter where the
// cursor is, and a click on it goes there and picks its notes.
const SIDE_KEY = "kinburg.satyrEdit.lyricsWidth";

function sideWidth(st, w) {
  st.sideW = Math.round(Math.max(160, Math.min(w, window.innerWidth * 0.6)));
  st.side.style.width = st.sideW + "px";
}

function showSide(st, on) {
  st.sideOn = on;
  st.side.style.display = st.grip.style.display = on ? "" : "none";
}

// Which written line is which sung one: the backend's sung lines are the lyric's own lines, stripped,
// in order — so each is looked for from where the last one was found.
export function sungRows(lyrics, sung) {
  const rows = String(lyrics || "").split(/\r?\n/);
  const at = new Map();
  let from = 0;
  sung.forEach((line, k) => {
    const want = String(line).trim();
    const i = rows.findIndex((r, j) => j >= from && r.trim() === want);
    if (i >= 0) { at.set(i, k); from = i + 1; }
  });
  return { rows, at };
}

function buildSide(st) {
  st.side.innerHTML = "";
  st.sideLines = [];
  st.sideKey = "";
  const { rows, at } = sungRows(st.lyrics, st.words?.lines || []);
  rows.forEach((text, i) => {
    const row = el("div", "kbse-line");
    row.textContent = text || " ";
    if (at.has(i)) {
      const k = at.get(i);
      row.classList.add("sung");
      row.addEventListener("click", () => goToLine(st, k));
      st.sideLines[k] = row;
    } else if (/^\s*\[.*\]\s*$/.test(text)) row.classList.add("mark");
    st.side.append(row);
  });
}

// A sung line: put the cursor on its first note, bring it into view and pick its notes (joined ones
// too), ready for an octave up or down.
function goToLine(st, k) {
  const vocal = st.model.notes.filter((n) => n.v === 0).sort((a, b) => a.t - b.t);
  const picked = [];
  let on = false;
  for (const n of vocal) {
    if (n._syl) on = (n._syl.lines || [n._syl.line]).includes(k);
    else if (!n.j && !n._hold) on = false;
    if (on) picked.push(n);
  }
  const v = view(st);
  if (!picked.length) {
    // A line sung over the music where the plan has no notes: go to where it falls.
    const free = freeLines(st.L).find((x) => x.line === k);
    if (!free) return;
    st.cursor = Math.round(free.from);
    st.scrollX = st.cursor - (v.rollW / st.ppu) * 0.15;
    clampScroll(st);
    say(st, `“${st.words.lines[k]}” has no notes: the model sings it over the music here`);
    refresh(st);
    return;
  }
  const ps = picked.map((n) => n.p);
  st.sel = new Set(picked); st.secSel = null; st.barSel = null;
  st.cursor = picked[0].t;
  st.scrollX = picked[0].t - (v.rollW / st.ppu) * 0.15;
  st.scrollY = (st.range.hi - (Math.min(...ps) + Math.max(...ps)) / 2) * st.rowH - v.rollH / 2;
  clampScroll(st);
  say(st, `“${st.words.lines[k]}” — ${picked.length} note(s) picked: +12 / −12 hands them to the other voice`);
  refresh(st);
}

// The lines sung over the music where the plan has no notes, each with the stretch it is laid over:
// a silence's lines share it evenly.
function freeLines(L) {
  const out = [];
  for (const free of L.free) {
    const lines = [...new Set(free.syl.map((s) => s.line))], span = free.to - free.from;
    lines.forEach((line, k) => out.push({ line, from: free.from + (span * k) / lines.length,
                                          to: free.from + (span * (k + 1)) / lines.length }));
  }
  return out;
}

// Light the lines on screen and the one at the cursor (or the playhead); only touches the page when
// that changes.
function markSide(st) {
  if (!st.sideOn || !st.sideLines.length) return;
  const v = view(st), t0 = tOf(st, GUTTER), t1 = tOf(st, v.w), at = st.play ? st.playAt || 0 : st.cursor;
  const seen = new Set();
  let here = -1, hereT = -1;
  for (const n of st.model.notes) {
    if (n.v !== 0 || !n._syl) continue;
    if (n.t + n.d > t0 && n.t < t1) for (const line of n._syl.lines || [n._syl.line]) seen.add(line);
    if (n.t <= at && n.t > hereT) { here = (n._syl.lines || [n._syl.line]).at(-1); hereT = n.t; }
  }
  for (const free of freeLines(st.L)) {
    if (free.to > t0 && free.from < t1) seen.add(free.line);
    if (free.from <= at && at < free.to) here = free.line;
  }
  const key = [...seen].join(",") + "|" + here;
  if (key === st.sideKey) return;
  st.sideKey = key;
  st.sideLines.forEach((row, k) => {
    row?.classList.toggle("seen", seen.has(k));
    row?.classList.toggle("here", k === here);
  });
  const target = seen.has(here) || !seen.size ? here : Math.min(...seen);
  st.sideLines[target]?.scrollIntoView({ block: "nearest" });
}

// Ask the backend to lay the lyrics onto the plan as it stands — after opening and after every edit,
// a moment after the last one, and only an answer for the model as it still is gets used.
function askWords(st) {
  if (!st.wordsFn || !st.lyrics.trim()) return;
  clearTimeout(st._words);
  st._words = setTimeout(async () => {
    const rev = st.rev;
    try {
      const got = await st.wordsFn(plain(st.model), st.base);
      if (rev !== st.rev || !st.root.isConnected) return;
      const lines = JSON.stringify(st.words?.lines || []);
      st.words = got;
      st.L = layout(st.model, st.words);
      if (JSON.stringify(got.lines || []) !== lines) buildSide(st);
      const notes = got.findings || [];
      st.ui.notesBtn.style.display = notes.length ? "" : "none";
      st.ui.notesBtn.textContent = `ⓘ ${notes.length}`;
      st.ui.notesBtn.title = "What Satyr Score would say about this plan and these lyrics:\n\n• " + notes.join("\n• ");
      refresh(st);
    } catch (e) {
      say(st, "⚠ lyrics: " + e.message);
    }
  }, 250);
}

// The model changed: what was laid on it is out of date until the next answer.
function changed(st) {
  st.rev += 1;
  askWords(st);
}

function close(st) {
  if (st.dirty && !confirm("Close the editor without saving? Your changes since the last save are lost.")) return;
  stop(st);
  try { st.audio?.close(); } catch (e) { /* already closed */ }
  window.removeEventListener("keydown", st.onKey, true);
  st.ro?.disconnect();
  st.root.remove();
}

async function doSave(st) {
  if (st.saving) return;
  closeInline(st, true);
  st.saving = true;
  st.ui.saveBtn.disabled = true;
  say(st, "saving…");
  try {
    const pins = st.model.sections.map((s) => s.sings);
    const got = await st.save(plain(st.model), st.base);
    st.base = got.abc;
    st.model = got.model;
    // Pins are the editor's, not the plan's: carried over onto the sections the save handed back.
    if (st.model.sections.length === pins.length) {
      st.model.sections.forEach((s, i) => { if (pins[i] !== undefined) s.sings = pins[i]; });
    }
    st.sel = new Set(); st.barSel = null; st.secSel = null;
    st.L = layout(st.model, st.words);
    st.dirty = false;
    changed(st);
    say(st, "saved — " + got.report.join(" · "));
    flash(st, "✓ Saved — " + got.report.slice(1).join(" · "), "ok");
  } catch (e) {
    say(st, "⚠ not saved: " + e.message);
    flash(st, "⚠ Not saved: " + e.message, "bad");
  } finally {
    st.saving = false;
    st.ui.saveBtn.disabled = false;
    refresh(st);
  }
}

// A message that cannot be missed, over the roll for a few seconds; the footer keeps it afterwards.
function flash(st, text, kind) {
  st.toast.textContent = text;
  st.toast.className = "kbse-toast show " + kind;
  clearTimeout(st._toast);
  st._toast = setTimeout(() => { st.toast.className = "kbse-toast"; }, 7000);
}

function say(st, text) {
  st.msgLine.textContent = text || "";
  st.msgLine.title = text || "";
}

// --------------------------------------------------------------------------------- edits & undo
function snapshot(st) {
  return JSON.stringify(plain(st.model));
}

const LOCKED = "🔒 the Vocal rhythm is locked: YuE2 wrote it for the words' stresses, and a sung note added, " +
  "removed or moved shifts which syllable falls on which beat — up and down only (unlock 🔒 to change it)";

// Run one edit on the model as a single undo step — refused, and put back, if the Vocal rhythm is
// locked and the edit changed it. False when nothing was done.
function edit(st, fn) {
  const before = snapshot(st), rhythm = st.lock ? vocalRhythm(st.model) : null;
  const picked = new Set([...st.sel].map((n) => `${n.v}:${n.t}:${n.p}`)), secSel = st.secSel, barSel = st.barSel;
  const result = fn(st.model);
  if (result === false) return false;
  tidy(st.model);
  if (rhythm !== null && vocalRhythm(st.model) !== rhythm) {
    restore(st, before);
    st.sel = new Set(st.model.notes.filter((n) => picked.has(`${n.v}:${n.t}:${n.p}`)));
    st.secSel = secSel; st.barSel = barSel;
    say(st, LOCKED);
    refresh(st);
    return false;
  }
  commit(st, before);
  return true;
}

function commit(st, before) {
  st.undo.push(before);
  if (st.undo.length > 200) st.undo.shift();
  st.redo = [];
  st.dirty = true;
  relayout(st);
  changed(st);
}

function restore(st, json) {
  st.model = JSON.parse(json);
  st.sel = new Set(); st.barSel = null; st.secSel = null;
  relayout(st);
  changed(st);
}

function undo(st) {
  if (!st.undo.length) return;
  st.redo.push(snapshot(st));
  restore(st, st.undo.pop());
  st.dirty = true;
  refresh(st);
}

function redo(st) {
  if (!st.redo.length) return;
  st.undo.push(snapshot(st));
  restore(st, st.redo.pop());
  st.dirty = true;
  refresh(st);
}

function relayout(st) {
  st.L = layout(st.model, st.words);
  st.cursor = Math.min(st.cursor, st.L.total);
  const before = st.range;
  st.range = pitchRange(st);
  if (before) st.scrollY += (st.range.hi - before.hi) * st.rowH;      // keep the view where it was
  clampScroll(st);
  refresh(st);
}

// Selection that no longer points at anything visible is dropped.
function prune(st) {
  for (const n of st.sel) if (!st.show[n.v] || !st.model.notes.includes(n)) st.sel.delete(n);
  refresh(st);
}

// Selected notes move alone; selected bars (or a section) move with their chords and key; with
// nothing selected the whole song moves.
function transpose(st, semis) {
  const how = `${semis > 0 ? "up" : "down"} ${Math.abs(semis)} semitone${Math.abs(semis) > 1 ? "s" : ""}`;
  if (st.sel.size) {
    const picked = [...st.sel];
    edit(st, (m) => { transposeNotes(m, picked, semis); enforceMono(m, picked); });
    say(st, `${picked.length} note(s) moved ${how}`);
  } else if (st.barSel) {
    const { b0, b1 } = st.barSel;
    edit(st, (m) => transposeRange(m, b0, b1, semis));
    say(st, `bars ${b0 + 1}–${b1} moved ${how}, with their chords and key — now ${inForce(st.model.keys, "k", st.L.starts[b0])}`);
  } else {
    edit(st, (m) => transposeSong(m, semis));
    say(st, `the whole song, its key and its chords moved ${how} — now ${st.model.keys[0].k}`);
  }
}

function deleteSelection(st) {
  if (st.sel.size) {
    const n = st.sel.size;
    if (edit(st, (m) => deleteNotes(m, [...st.sel]))) {
      st.sel = new Set();
      say(st, `${n} note(s) deleted`);
    }
  } else if (st.secSel !== null) {
    const label = st.model.sections[st.secSel].label;
    if (edit(st, (m) => deleteSection(m, st.secSel))) {
      st.secSel = null; st.barSel = null;
      say(st, `section '${label}' deleted`);
    }
  } else if (st.barSel) {
    const { b0, b1 } = st.barSel;
    if (edit(st, (m) => deleteBars(m, b0, b1))) {
      st.barSel = null;
      say(st, `${b1 - b0} bar(s) deleted`);
    }
  }
  refresh(st);
}

// The grid choices a plan's L: allows, as note values: at L:1/16 that is 1/16, 1/8, 1/4, 1/2 and the bar.
function gridChoices(model) {
  const [un, ud] = model.unit;
  const out = [];
  for (const k of [1, 2, 4, 8]) {
    const num = un * k, den = ud;
    const g = gcd(num, den);
    if (den / g <= 32 && num / g === 1) out.push({ units: k, label: `1/${den / g}` });
  }
  out.push({ units: 0, label: "bar" });
  return out;
}

function gcd(a, b) {
  return b ? gcd(b, a % b) : a;
}

function gridAt(st, t) {
  const b = barAt(st.L.starts, Math.max(0, Math.min(t, st.L.total - 1)));
  return { b, start: st.L.starts[b], size: st.grid || st.model.bars[b].w, w: st.model.bars[b].w };
}

function snap(st, t, mode = "round") {
  const g = gridAt(st, t);
  const k = (t - g.start) / g.size;
  return Math.max(g.start, Math.min(g.start + g.w, g.start + (mode === "floor" ? Math.floor(k) : Math.round(k)) * g.size));
}

// ----------------------------------------------------------------------------------- the panels
function refresh(st) {
  const L = st.L, m = st.model;
  const key = m.keys.length ? m.keys[0].k : "";
  const syl = L.words ? L.words.reduce((a, w) => a + w.wants, 0) : 0;
  const sung = L.words ? L.words.reduce((a, w) => a + w.notes, 0) : 0;
  st.ui.stats.textContent = `${m.bars.length} bars · ${mmss(L.total * L.unitSec)} · ${key} · ♩=${m.bpm}` +
    (L.words ? ` · ${sung} notes / ${syl} syllables` : "") + (st.dirty ? " · edited" : "");
  st.ui.stats.title = L.words ? "Vocal notes in the plan against syllables in the lyrics. They need not match: the " +
    "model squeezes words onto fewer notes, holds them over more, and sings over the music where there are none" : "";
  st.ui.timeBox.textContent = mmss((st.play ? st.playAt || 0 : st.cursor) * L.unitSec);
  st.ui.undoBtn.disabled = !st.undo.length;
  st.ui.redoBtn.disabled = !st.redo.length;
  st.ui.lockBtn.textContent = st.lock ? "🔒 Vocal rhythm" : "🔓 Vocal rhythm";
  if (document.activeElement !== st.ui.tempo) st.ui.tempo.value = m.bpm;
  actions(st);
  draw(st);
}

// The right end of the editing row: what can be done to what is selected.
function actions(st) {
  const box = st.ui.ctx;
  box.innerHTML = "";
  const m = st.model;
  if (st.secSel !== null && m.sections[st.secSel]) {
    const si = st.secSel, s = m.sections[si], span = sectionSpans(m)[si];
    const names = [...new Set([s.label, ...(m.labels || [])].filter(Boolean))];
    const rename = select(names.map((x) => [x, x]), s.label, (v) => edit(st, (mm) => renameSection(mm, si, v)));
    rename.title = "Rename the section — YuE2 reads these labels";
    box.append(label(`Section ${span.b1 > span.b0 ? `bars ${span.b0 + 1}–${span.b1}` : "(empty)"}:`), rename,
      button("Duplicate", "Copy the section in right after itself", () => {
        if (edit(st, (mm) => duplicateSection(mm, si))) { st.secSel = si + 1; st.barSel = barsOf(st, si + 1); }
        refresh(st);
      }),
      button("Delete", "Remove the section and its bars (Del)", () => deleteSelection(st)),
      button("◀ Move", "Swap with the section before", () => {
        if (edit(st, (mm) => moveSection(mm, si, -1))) { st.secSel = si - 1; st.barSel = barsOf(st, si - 1); refresh(st); }
      }),
      button("Move ▶", "Swap with the section after", () => {
        if (edit(st, (mm) => moveSection(mm, si, 1))) { st.secSel = si + 1; st.barSel = barsOf(st, si + 1); refresh(st); }
      }));
    if (st.words?.blocks?.length) {
      const opts = [["auto", "auto — matched by counts"], ["-1", "nothing — instrumental"],
        ...st.words.blocks.map((b, k) => [String(k), `${k + 1}. ${b.label}${b.voice ? " · " + b.voice : ""} (${b.syllables})`])];
      const sings = select(opts, s.sings === undefined ? "auto" : String(s.sings), (v) => edit(st, (mm) => {
        if (v === "auto") delete mm.sections[si].sings;
        else mm.sections[si].sings = Number(v);
      }));
      sings.title = "Which lyric block this section sings. 'auto' leaves it to Satyr Score's matcher, which goes by " +
        "counts; pin a block where it is wrong. Two cases the counts get wrong: an interlude the model sang the " +
        "bridge over, with no notes at all, loses the bridge to the chorus after it — pin it to the bridge; and an " +
        "instrumental intro is handed the intro's words although the model sang them at the start of the verse — " +
        "pin it to nothing.";
      box.append(label("Sings:"), sings);
    }
  } else if (st.barSel) {
    const { b0, b1 } = st.barSel, n = b1 - b0;
    box.append(label(`Bars ${b0 + 1}–${b1} (${n}):`),
      button("Delete", "Remove these bars from both voices (Del)", () => deleteSelection(st)),
      button("Duplicate", "Copy these bars in right after them", () => {
        if (edit(st, (mm) => duplicateBars(mm, b0, b1))) st.barSel = { b0: b1, b1: b1 + n };
        refresh(st);
      }),
      button(`Insert ${n} empty before`, "Empty bars in front of these, in their meter", () => {
        if (edit(st, (mm) => insertBars(mm, b0, n))) st.barSel = { b0, b1: b0 + n };
        refresh(st);
      }));
  } else if (st.sel.size) {
    const sung = wordsOf(st.sel);
    const text = label(sung ? `${st.sel.size} note(s): “${sung}”` : `${st.sel.size} note(s) selected`);
    text.className = "kbse-label kbse-hint";
    text.title = (sung ? `“${sung}”\n\n` : "") + "drag to move · arrows to nudge · Del to delete · double-click to " +
      "join to the note before · Transpose moves only these · Ctrl+C copies them";
    box.append(button("Copy", "Copy these notes (Ctrl+C) — then click the ruler where they should go and Paste",
      () => copyNotes(st)));
    if (st.clip?.length) box.append(button("Paste", "Paste the copied notes at the cursor (Ctrl+V)", () => paste(st)));
    box.append(text);
  } else {
    if (st.clip?.length) box.append(button("Paste", "Paste the copied notes at the cursor (Ctrl+V)", () => paste(st)));
    const hint = label((st.lock ? "click a note · drag it up or down (the Vocal rhythm is locked; Ins moves freely) · "
      : "click a note · drag it to move · drag its end to stretch · double-click empty space to draw · right-click a " +
        "note to delete · ") + "click a singer's run to pick its notes · drag the ruler to pick bars · click a section · " +
      "double-click the chord lane · Ctrl+C / Ctrl+V copy notes to the cursor");
    hint.className = "kbse-label kbse-hint";
    hint.title = hint.textContent;
    box.append(hint);
  }
}

function copyNotes(st) {
  if (!st.sel.size) { say(st, "select the notes to copy first"); return; }
  const picked = [...st.sel].sort((a, b) => a.t - b.t), t0 = picked[0].t;
  st.clip = picked.map((n) => ({ v: n.v, t: n.t - t0, d: n.d, p: n.p, j: n.j }));
  say(st, `${picked.length} note(s) copied — click the ruler where they should go, then Ctrl+V`);
  refresh(st);
}

// The copied notes at the cursor, voices and pitches as they were; what they land on is cut back.
function paste(st) {
  if (!st.clip?.length) { say(st, "nothing copied yet — select notes and press Ctrl+C"); return; }
  const at = st.cursor, total = st.L.total;
  let made = [];
  const done = edit(st, (m) => {
    made = st.clip.map((n) => ({ ...n, t: n.t + at })).filter((n) => n.t < total);
    m.notes.push(...made);
    enforceMono(m, made);
  });
  if (!done) { refresh(st); return; }
  st.sel = new Set(made.filter((n) => st.model.notes.includes(n)));
  st.barSel = null; st.secSel = null;
  say(st, `${st.sel.size} note(s) pasted at bar ${barAt(st.L.starts, Math.min(at, total - 1)) + 1}`);
  refresh(st);
}

function barsOf(st, si) {
  const span = sectionSpans(st.model)[si];
  return span && span.b1 > span.b0 ? { b0: span.b0, b1: span.b1 } : null;
}

// ----------------------------------------------------------------------------------- view maths
function view(st) {
  const w = st.cssW, h = st.cssH;
  return { w, h, rollTop: TOP, rollBottom: h - OVER_H, rollLeft: GUTTER, rollW: w - GUTTER, rollH: h - OVER_H - TOP };
}
const xOf = (st, t) => GUTTER + (t - st.scrollX) * st.ppu;
const tOf = (st, x) => st.scrollX + (x - GUTTER) / st.ppu;
function pitchRange(st) {
  const ps = st.model.notes.map((n) => n.p);
  const lo = Math.min(36, ...ps) - 3, hi = Math.max(84, ...ps) + 3;
  return { lo, hi };
}
const yOf = (st, p) => TOP + (st.range.hi - p) * st.rowH - st.scrollY;
const pOf = (st, y) => st.range.hi - Math.floor((y - TOP + st.scrollY) / st.rowH);

function resize(st) {
  const r = st.canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  st.cssW = Math.max(200, Math.floor(r.width));
  st.cssH = Math.max(200, Math.floor(r.height));
  st.canvas.width = Math.round(st.cssW * dpr);
  st.canvas.height = Math.round(st.cssH * dpr);
  st.canvas.style.width = st.cssW + "px";
  st.canvas.style.height = st.cssH + "px";
  st.dpr = dpr;
  st.range = pitchRange(st);
  // Rows as tall as the window allows, so a tall window gives bigger notes rather than empty octaves.
  st.rowH = Math.max(9, Math.min(16, Math.floor(view(st).rollH / (st.range.hi - st.range.lo + 1))));
  clampScroll(st);
  draw(st);
}

// Where the lowest row of the song's range ends — below it the roll is empty, and not clickable.
function contentBottom(st) {
  return Math.min(view(st).rollBottom, yOf(st, st.range.lo) + st.rowH);
}

function clampScroll(st) {
  const v = view(st);
  const maxX = Math.max(0, st.L.total - v.rollW / st.ppu * 0.9);
  st.scrollX = Math.min(Math.max(0, st.scrollX), maxX);
  const content = (st.range.hi - st.range.lo + 1) * st.rowH;
  st.scrollY = Math.min(Math.max(0, st.scrollY), Math.max(0, content - v.rollH));
}

function fitAll(st) {
  const v = view(st);
  st.ppu = Math.max(0.4, Math.min(40, v.rollW / Math.max(1, st.L.total)));
  st.scrollX = 0;
  clampScroll(st);
  draw(st);
}

// Start with the sung notes in the middle of the roll rather than at the top of an empty range.
function centerPitches(st) {
  const ps = st.model.notes.map((n) => n.p);
  if (!ps.length) return;
  const mid = (Math.min(...ps) + Math.max(...ps)) / 2;
  const v = view(st);
  st.scrollY = (st.range.hi - mid) * st.rowH - v.rollH / 2;
  clampScroll(st);
  draw(st);
}

function zoom(st, factor, atX) {
  const v = view(st);
  const x = atX ?? GUTTER + v.rollW / 2;
  const t = tOf(st, x);
  st.ppu = Math.max(0.4, Math.min(40, st.ppu * factor));
  st.scrollX = t - (x - GUTTER) / st.ppu;
  clampScroll(st);
  draw(st);
}

// ---------------------------------------------------------------------------------------- input
function onWheel(st, e) {
  e.preventDefault();
  const rect = st.canvas.getBoundingClientRect();
  if (e.ctrlKey || e.metaKey) {
    zoom(st, e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - rect.left);
    return;
  }
  const sideways = e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY);
  if (sideways) st.scrollX += (e.shiftKey ? e.deltaY : e.deltaX) / st.ppu;
  else st.scrollY += e.deltaY;
  clampScroll(st);
  draw(st);
}

function local(st, e) {
  const r = st.canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

// What is under a point: a lane, a note (and whether its end), the empty roll, or the overview.
function hit(st, x, y) {
  const v = view(st);
  if (y >= v.rollBottom) return { area: "overview" };
  if (x < GUTTER) return { area: "keys" };
  const t = tOf(st, x);
  if (y < BLK_H) return { area: "blocks", t };
  if (y < SEC_H) return { area: "sections", t };
  if (y < SEC_H + RULER_H) return { area: "ruler", t };
  if (y < SING_Y) return { area: "chords", t };
  if (y < LYR_Y) return { area: "singers", t };
  if (y < TOP) return { area: "lyrics", t };
  if (y >= contentBottom(st)) return { area: "none" };
  const n = noteAt(st, x, y);
  if (n) {
    const width = n.d * st.ppu;
    return { area: "note", note: n, edge: width >= 9 && x >= xOf(st, n.t + n.d) - Math.min(6, width / 3), t };
  }
  return { area: "roll", t, p: pOf(st, y) };
}

function onDown(st, e) {
  closeInline(st, true);
  const { x, y } = local(st, e);
  const h = hit(st, x, y);
  try { st.canvas.setPointerCapture(e.pointerId); } catch (err) { /* synthetic event */ }
  if (e.button === 2) { onRight(st, h); return; }
  if (e.button !== 0) return;
  const L = st.L;
  if (h.area === "overview") {
    st.drag = { kind: "overview" };
    overviewJump(st, x);
    return;
  }
  if (h.area === "ruler") {
    const t = Math.max(0, Math.min(L.total, h.t));
    const b = barAt(L.starts, Math.min(t, L.total - 1));
    setCursor(st, x);
    st.sel = new Set(); st.secSel = null;
    st.barSel = { b0: b, b1: b + 1 };
    st.drag = { kind: "bars", anchor: b };
  } else if (h.area === "blocks") {
    // A block of the lyric picks every note its words are sung on — to move it to the other voice.
    const w = (L.words || []).find((x) => x.from <= h.t && h.t < x.to);
    st.barSel = null; st.secSel = null;
    if (w) {
      const picked = blockNotes(st, w);
      st.sel = new Set(e.shiftKey || e.ctrlKey || e.metaKey ? [...st.sel, ...picked] : picked);
      say(st, `${w.label}: ${picked.length} note(s) picked` + (picked.length ? " — +12 / −12 hands them to the other voice" : " — it has none, it is sung over the music"));
    }
  } else if (h.area === "sections") {
    const si = L.sections.findIndex((s) => s.from <= h.t && h.t < s.to);
    st.sel = new Set();
    st.secSel = si >= 0 ? si : null;
    st.barSel = si >= 0 ? barsOf(st, si) : null;
  } else if (h.area === "note") {
    const n = h.note;
    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      if (st.sel.has(n)) st.sel.delete(n); else st.sel.add(n);
    } else if (!st.sel.has(n)) st.sel = new Set([n]);
    st.barSel = null; st.secSel = null;
    if (st.sel.has(n)) {
      // With the Vocal rhythm locked, anything sung moves up and down only — and takes the rest along.
      const fixed = st.lock && [...st.sel].some((m) => m.v === 0);
      st.drag = { kind: h.edge && !fixed ? "stretch" : "move", fixed, x0: x, y0: y, note: n, before: snapshot(st),
                  moved: false, plain: !(e.shiftKey || e.ctrlKey || e.metaKey),
                  origin: new Map([...st.sel].map((m) => [m, { t: m.t, p: m.p, d: m.d }])) };
    }
  } else if (h.area === "roll") {
    if (!(e.shiftKey || e.ctrlKey || e.metaKey)) st.sel = new Set();
    st.barSel = null; st.secSel = null;
    st.drag = { kind: "box", x0: x, y0: y, x1: x, y1: y, base: new Set(st.sel) };
  } else if (h.area === "lyrics") {
    // Picking words picks the Vocal notes that sing them — the way to transpose "these lines".
    st.barSel = null; st.secSel = null;
    st.drag = { kind: "words", t0: h.t, base: e.shiftKey || e.ctrlKey || e.metaKey ? new Set(st.sel) : new Set() };
    pickWords(st, h.t, h.t);
  } else if (h.area === "singers") {
    // A singer's run picks its notes: an octave up or down hands them to the other voice.
    const r = L.runs.find((x) => x.from <= h.t && h.t < x.to);
    st.barSel = null; st.secSel = null;
    if (r) {
      st.sel = new Set(e.shiftKey || e.ctrlKey || e.metaKey ? [...st.sel, ...r.notes] : r.notes);
      say(st, runStory(r));
    }
  }
  refresh(st);
}

// The notes a lyric block's words are sung on, joined and held ones included.
function blockNotes(st, w) {
  const picked = [];
  let on = false;
  for (const n of st.model.notes.filter((x) => x.v === 0).sort((a, b) => a.t - b.t)) {
    if (n._syl) on = n._syl.block === w.block;
    else if (!n.j && !n._hold) on = false;
    if (on) picked.push(n);
  }
  return picked;
}

function pickWords(st, a, b) {
  const t0 = Math.min(a, b), t1 = Math.max(a, b);
  st.sel = new Set(st.drag.base);
  if (!st.show[0]) return;
  for (const n of st.model.notes) if (n.v === 0 && n.t <= t1 && n.t + n.d > t0) st.sel.add(n);
}

function onMove(st, e) {
  const { x, y } = local(st, e);
  const d = st.drag;
  if (!d) {
    st.hover = { x, y };
    const h = hit(st, x, y);
    st.canvas.style.cursor = h.area === "note" ? (st.lock && h.note.v === 0 ? "ns-resize" : h.edge ? "ew-resize" : "grab")
      : h.area === "ruler" || h.area === "lyrics" ? "text" : h.area === "singers" || h.area === "blocks" ? "pointer" : "default";
    showHover(st);
    return;
  }
  if (d.kind === "overview") { overviewJump(st, x); return; }
  if (d.kind === "words") { pickWords(st, d.t0, tOf(st, x)); refresh(st); return; }
  if (d.kind === "bars") {
    const t = Math.max(0, Math.min(st.L.total - 1, tOf(st, x)));
    const b = barAt(st.L.starts, t);
    st.barSel = { b0: Math.min(d.anchor, b), b1: Math.max(d.anchor, b) + 1 };
    refresh(st);
    return;
  }
  if (d.kind === "box") {
    d.x1 = x; d.y1 = y;
    const t0 = tOf(st, Math.min(d.x0, x)), t1 = tOf(st, Math.max(d.x0, x));
    const pHi = pOf(st, Math.min(d.y0, y)), pLo = pOf(st, Math.max(d.y0, y));
    st.sel = new Set(d.base);
    for (const n of st.model.notes) {
      if (st.show[n.v] && n.p >= pLo && n.p <= pHi && n.t < t1 && n.t + n.d > t0) st.sel.add(n);
    }
    refresh(st);
    return;
  }
  const total = st.L.total;
  if (d.kind === "move") {
    const o = d.origin.get(d.note);
    const dt = d.fixed ? 0 : snap(st, o.t + (x - d.x0) / st.ppu) - o.t;
    const dp = Math.round((d.y0 - y) / st.rowH);
    let lo = -Infinity, hi = Infinity;
    for (const [, v] of d.origin) { lo = Math.max(lo, -v.t); hi = Math.min(hi, total - v.t - v.d); }
    const move = Math.max(lo, Math.min(hi, dt));
    for (const [m, v] of d.origin) { m.t = v.t + move; m.p = Math.max(0, Math.min(127, v.p + dp)); }
    d.moved = d.moved || move !== 0 || dp !== 0;
  } else if (d.kind === "stretch") {
    const o = d.origin.get(d.note);
    const end = Math.min(total, Math.max(o.t + 1, snap(st, o.t + o.d + (x - d.x0) / st.ppu)));
    d.note.d = end - o.t;
    d.moved = d.moved || d.note.d !== o.d;
  }
  st.L = layout(st.model, st.words);
  st.hover = { x, y };
  showHover(st);
  draw(st);
}

function onUp(st, e) {
  const d = st.drag;
  st.drag = null;
  try { st.canvas.releasePointerCapture(e.pointerId); } catch (err) { /* not captured */ }
  if (!d) return;
  if ((d.kind === "move" || d.kind === "stretch") && d.moved) {
    enforceMono(st.model, d.kind === "move" ? [...d.origin.keys()] : [d.note]);
    for (const n of [...st.sel]) if (!st.model.notes.includes(n)) st.sel.delete(n);
    commit(st, d.before);
    say(st, d.kind === "move" ? `${d.origin.size} note(s) moved` : "note stretched");
  } else if (d.kind === "move" && d.plain) {
    st.sel = new Set([d.note]);         // a plain click on one of several picks just that one
  }
  refresh(st);
}

function onDouble(st, e) {
  const { x, y } = local(st, e);
  const h = hit(st, x, y);
  if (h.area === "note") {
    const before = snapshot(st);
    if (st.lock && h.note.v === 0) say(st, LOCKED);
    else if (toggleJoin(st.model, h.note)) {
      commit(st, before);
      say(st, h.note.j ? "joined: this note now carries on the syllable of the one before it"
                       : "unjoined: this note is a syllable of its own again");
    } else say(st, "only a note that starts exactly where the previous one ends can be joined to it");
  } else if (h.area === "roll") {
    const v = st.drawVoice;
    if (!st.show[v]) { say(st, `${VOICES[v]} is hidden — show it to draw into it`); return; }
    const t = snap(st, h.t, "floor");
    const g = gridAt(st, t);
    if (t >= st.L.total) return;
    const note = { v, t, d: Math.max(1, Math.min(g.size, st.L.total - t)), p: Math.max(0, Math.min(127, h.p)), j: 0 };
    if (edit(st, (m) => addNote(m, note))) {
      st.sel = new Set([note]);
      say(st, `${VOICES[v]} ${pitchName(note.p)} drawn`);
    }
  } else if (h.area === "chords") {
    openChord(st, h.t);
  } else if (h.area === "sections") {
    const si = st.L.sections.findIndex((s) => s.from <= h.t && h.t < s.to);
    if (si >= 0) { st.secSel = si; st.barSel = barsOf(st, si); st.sel = new Set(); }
  }
  refresh(st);
}

function onRight(st, h) {
  if (h.area === "note") {
    if (edit(st, (m) => deleteNotes(m, [h.note]))) {
      st.sel.delete(h.note);
      say(st, "note deleted");
    }
  } else if (h.area === "chords") {
    const before = inForce(st.model.chords, "c", h.t);
    if (before) { edit(st, (m) => removeChordAt(m, h.t)); say(st, `chord ${before} removed`); }
  }
  refresh(st);
}

function onKey(st, e) {
  const tag = (document.activeElement?.tagName || "").toUpperCase();
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  const key = e.key, mod = e.ctrlKey || e.metaKey;
  const handled = () => { e.preventDefault(); e.stopPropagation(); };
  if (key === "Escape") {
    handled();
    if (st.sel.size || st.barSel || st.secSel !== null) { st.sel = new Set(); st.barSel = null; st.secSel = null; refresh(st); }
    else close(st);
    return;
  }
  if (key === " ") { handled(); st.play ? stop(st) : play(st); return; }
  if (key === "Home") { handled(); st.cursor = 0; st.scrollX = 0; refresh(st); return; }
  if (mod && (key === "z" || key === "Z") && !e.shiftKey) { handled(); undo(st); return; }
  if (mod && (key === "y" || ((key === "z" || key === "Z") && e.shiftKey))) { handled(); redo(st); return; }
  if (mod && (key === "c" || key === "C")) { handled(); copyNotes(st); return; }
  if (mod && (key === "v" || key === "V")) { handled(); paste(st); return; }
  if (mod && (key === "a" || key === "A")) {
    handled();
    st.sel = new Set(st.model.notes.filter((n) => st.show[n.v]));
    st.barSel = null; st.secSel = null;
    refresh(st);
    return;
  }
  if (key === "Delete" || key === "Backspace") { handled(); deleteSelection(st); return; }
  if (st.sel.size && (key === "ArrowUp" || key === "ArrowDown")) {
    handled();
    const semis = (key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 12 : 1);
    const picked = [...st.sel];
    edit(st, (m) => { transposeNotes(m, picked, semis); enforceMono(m, picked); });
    return;
  }
  if (st.sel.size && (key === "ArrowLeft" || key === "ArrowRight")) {
    handled();
    const picked = [...st.sel];
    const step = (key === "ArrowRight" ? 1 : -1) * (st.grid || st.model.bars[0].w);
    const lo = Math.min(...picked.map((n) => n.t)), hi = Math.max(...picked.map((n) => n.t + n.d));
    if (lo + step < 0 || hi + step > st.L.total) return;
    edit(st, (m) => { for (const n of picked) n.t += step; enforceMono(m, picked); });
    return;
  }
  // Everything else stays with the editor while it is open, so ComfyUI's own shortcuts (delete a
  // node, queue a prompt) cannot fire behind it.
  e.stopPropagation();
}

function setCursor(st, x) {
  const L = st.L;
  const t = Math.max(0, Math.min(L.total, Math.round(tOf(st, x))));
  // Snap to the beat: a cursor between two sixteenths is never what anyone meant.
  const b = barAt(L.starts, t);
  const beat = beatUnits(st.model, st.model.bars[b]);
  st.cursor = Math.min(L.total, L.starts[b] + Math.round((t - L.starts[b]) / beat) * beat);
  refresh(st);
}

function overviewJump(st, x) {
  const v = view(st);
  const t = (x / v.w) * st.L.total;
  st.scrollX = t - (v.rollW / st.ppu) / 2;
  clampScroll(st);
  draw(st);
}

function noteAt(st, x, y) {
  const t = tOf(st, x), p = pOf(st, y);
  for (let i = st.model.notes.length - 1; i >= 0; i--) {
    const n = st.model.notes[i];
    if (!st.show[n.v]) continue;
    if (n.p === p && n.t <= t && t < n.t + n.d) return n;
  }
  return null;
}

// The chord lane's own little editor: type a chord where you double-clicked, Enter to set it, an empty
// field for "no chord from here", Esc to leave it as it was.
function openChord(st, t) {
  const L = st.L;
  const at = Math.max(0, Math.min(L.total - 1, snapBeat(st, t)));
  const input = el("input", "kbse-inline");
  const current = st.model.chords.find((e) => e.t === at);
  input.value = current ? current.c : inForce(st.model.chords, "c", at);
  input.placeholder = "chord, e.g. Am7";
  input.title = "Enter: set the chord from here · empty: no chord from here · Esc: cancel";
  input.style.left = Math.max(GUTTER, xOf(st, at)) + "px";
  input.style.top = (SEC_H + RULER_H) + "px";
  st.main.append(input);
  st.inline = { input, at, done: false };
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") { e.preventDefault(); closeInline(st, true); }
    if (e.key === "Escape") { e.preventDefault(); closeInline(st, false); }
  });
  input.addEventListener("blur", () => closeInline(st, true));
  input.focus();
  input.select();
}

function closeInline(st, keep) {
  const ed = st.inline;
  if (!ed || ed.done) return;
  ed.done = true;
  st.inline = null;
  const text = ed.input.value.trim();
  ed.input.remove();
  if (!keep) return;
  if (text === (st.model.chords.find((e) => e.t === ed.at)?.c ?? inForce(st.model.chords, "c", ed.at))) return;
  edit(st, (m) => setChord(m, ed.at, text));
  say(st, text ? `chord ${text} from bar ${barAt(st.L.starts, ed.at) + 1}` : "no chord from here");
}

function snapBeat(st, t) {
  const b = barAt(st.L.starts, Math.max(0, Math.min(t, st.L.total - 1)));
  const beat = beatUnits(st.model, st.model.bars[b]);
  return st.L.starts[b] + Math.floor((t - st.L.starts[b]) / beat) * beat;
}

function showHover(st) {
  const h = st.hover;
  if (!h) { st.hoverLine.textContent = ""; return; }
  const v = view(st), L = st.L;
  const parts = [];
  if (h.x >= GUTTER) {
    const t = Math.max(0, Math.min(L.total - 1, Math.floor(tOf(st, h.x))));
    const b = barAt(L.starts, t);
    const si = L.sections.findIndex((s) => s.from <= t && t < s.to);
    parts.push(`bar ${b + 1}`, mmss(t * L.unitSec));
    const block = h.y < BLK_H ? (L.words || []).find((w) => w.from <= t && t < w.to) : null;
    if (block) {                                    // the song's sections by words: who sings, and how
      st.hoverLine.textContent = parts.concat(blockStory(block)).join(" · ");
      return;
    }
    if (h.y >= BLK_H && h.y < SEC_H && si >= 0) {   // the plan's own sections
      st.hoverLine.textContent = parts.concat(sectionStory(st, si)).join(" · ");
      return;
    }
    const run = h.y >= SING_Y && h.y < LYR_Y ? L.runs.find((r) => r.from <= t && t < r.to) : null;
    if (run) {                                      // the singers lane: whose run it is, and where it sits
      st.hoverLine.textContent = parts.concat(runStory(run)).join(" · ");
      return;
    }
    if (si >= 0 && L.sections[si].label) parts.push(L.sections[si].label);
    const chord = valueAt(L.chords, t), key = valueAt(L.keys, t);
    if (chord) parts.push(`chord ${chord}`);
    if (key) parts.push(`key ${key}`);
    if (h.y >= LYR_Y && h.y < TOP) {                // the lyrics lane: the line sung here
      const n = st.model.notes.filter((x) => x.v === 0 && x._syl && x.t <= t).sort((a, c) => c.t - a.t)[0];
      if (n) parts.push(`“${st.words.lines[n._syl.line]}”`);
    }
  }
  if (h.y >= v.rollTop && h.y < contentBottom(st)) {
    parts.push(pitchName(pOf(st, h.y)));
    const n = noteAt(st, h.x, h.y);
    if (n) parts.push(`${VOICES[n.v]} ${pitchName(n.p)} · ${n.d} unit${n.d > 1 ? "s" : ""}` +
                      (n.j ? " · joined (same syllable)" : "") +
                      (n._band && n.v === 0 ? ` · ${n._band === "high" ? "upper" : "lower"} voice` : "") +
                      (n._syl ? ` · sings “${n._syl.s}” of “${n._syl.word}” in “${st.words.lines[n._syl.line]}”` +
                                (n._syl.voice ? ` (${n._syl.voice})` : "")
                              : n._hold ? " · holds the syllable before" : ""));
  }
  st.hoverLine.textContent = parts.join(" · ");
}

// Everything known about one section's words, for the status line.
// A plan section, for the status line: its label as YuE2 wrote it and the lyric blocks sung in it.
function sectionStory(st, si) {
  const s = st.L.sections[si], pin = st.model.sections[si]?.sings;
  const sung = (st.L.words || []).filter((w) => w.from < s.to && w.to > s.from).map((w) => w.label);
  return [`plan section '${s.label || "—"}', bars ${s.b0 + 1}–${s.b1}`,
          sung.length ? `sung here: ${sung.join(" + ")}` : "nothing sung here",
          pin === undefined ? "click to rename, move, duplicate, delete or pin what it sings"
            : `pinned to ${pin < 0 ? "nothing" : st.words?.blocks?.[pin]?.label || "a block"}`];
}

// A lyric block, for the status line: who sings it, in which band, and how its words meet its notes.
function blockStory(w) {
  const out = [w.label || "(no label)"];
  if (w.voice) out.push(`marked for ${w.voice}` + (w.band ? ` (${w.band === "high" ? "upper" : "lower"} band)` : ""));
  if (w.was) out.push(`sits in the ${w.was === "high" ? "upper" : "lower"} band` + (clashes(w) ? " ⚠ not the singer's" : ""));
  out.push(`${w.notes} notes for ${w.wants} syllables` + (!w.notes ? "" : w.wants > w.notes + 1 ? " — squeezed onto the longer notes"
    : w.notes > w.wants + 1 ? " — some lines hold their last syllable" : ""));
  if (w.free) out.push(w.notes ? "some lines sung over the music where the plan has no notes"
                               : "no notes: the model sings it over the music, as it did a bridge");
  out.push("click to pick its notes");
  return out;
}

// -------------------------------------------------------------------------------------- drawing
function draw(st) {
  if (st._raf) return;
  st._raf = requestAnimationFrame(() => { st._raf = 0; paint(st); });
}

function paint(st) {
  const c = st.canvas.getContext("2d");
  const v = view(st), L = st.L, m = st.model;
  c.setTransform(st.dpr, 0, 0, st.dpr, 0, 0);
  c.fillStyle = COLORS.bg;
  c.fillRect(0, 0, v.w, v.h);
  const t0 = Math.max(0, Math.floor(tOf(st, GUTTER))), t1 = Math.min(L.total, Math.ceil(tOf(st, v.w)));

  // --- the roll: rows, bands, grid, picked bars, notes
  const yEnd = contentBottom(st);
  c.save();
  c.beginPath();
  c.rect(GUTTER, v.rollTop, v.rollW, Math.max(0, yEnd - v.rollTop));
  c.clip();
  const pTop = Math.min(st.range.hi, pOf(st, v.rollTop)), pBottom = Math.max(st.range.lo, pOf(st, yEnd - 1));
  for (let p = pBottom; p <= pTop; p++) {
    const y = yOf(st, p);
    c.fillStyle = NAMES_SHARP[((p % 12) + 12) % 12].length > 1 ? COLORS.black : COLORS.white;
    c.fillRect(GUTTER, y, v.rollW, st.rowH);
    if (p % 12 === 0) { c.fillStyle = COLORS.cRow; c.fillRect(GUTTER, y + st.rowH - 1, v.rollW, 1); }
  }
  if (st.show.bands && m.bands) {
    const ySplit = TOP + (st.range.hi - m.bands.split + 0.5) * st.rowH - st.scrollY;
    c.fillStyle = COLORS.highTint;
    c.fillRect(GUTTER, v.rollTop, v.rollW, Math.max(0, ySplit - v.rollTop));
    c.fillStyle = COLORS.lowTint;
    c.fillRect(GUTTER, ySplit, v.rollW, Math.max(0, v.rollBottom - ySplit));
    c.strokeStyle = COLORS.split;
    c.setLineDash([6, 4]);
    c.beginPath(); c.moveTo(GUTTER, ySplit + 0.5); c.lineTo(v.w, ySplit + 0.5); c.stroke();
    c.setLineDash([]);
  }
  gridLines(st, c, v, t0, t1, v.rollTop, v.rollBottom);
  if (st.barSel) {
    const x0 = xOf(st, L.starts[st.barSel.b0]), x1 = xOf(st, st.barSel.b1 < L.starts.length ? L.starts[st.barSel.b1] : L.total);
    c.fillStyle = COLORS.range;
    c.fillRect(x0, v.rollTop, x1 - x0, v.rollH);
  }
  c.font = "10px system-ui, sans-serif";
  c.textBaseline = "middle";
  for (const voice of [1, 0]) {
    if (!st.show[voice]) continue;
    for (const n of m.notes) {
      if (n.v !== voice || n.t + n.d < t0 || n.t > t1) continue;
      const x = xOf(st, n.t), y = yOf(st, n.p), w = Math.max(2, n.d * st.ppu - 1);
      if (y + st.rowH < v.rollTop || y > v.rollBottom) continue;
      c.fillStyle = voice === 1 ? COLORS.ins : !st.show.bands || !n._band ? COLORS.vocal
        : n._band === "high" ? COLORS.vocalHigh : COLORS.vocalLow;
      c.fillRect(x, y + 1, w, st.rowH - 2);
      if (n.j) {               // a melisma: drawn tied to the note before it
        c.fillStyle = "rgba(255,255,255,0.75)";
        c.fillRect(x - 1, y + st.rowH / 2 - 1, 3, 2);
      }
      // The syllable on the note itself, where it fits — the pitch and the word in one place; a note
      // the syllable before is held over shows "~".
      const sung = n._syl ? n._syl.s : n._hold ? "~" : "";
      if (sung && st.rowH >= 12 && c.measureText(sung).width + 4 <= w) {
        c.fillStyle = "#24170a";
        c.fillText(sung, x + 2, y + st.rowH / 2 + 0.5);
      }
      if (st.sel.has(n)) {
        c.strokeStyle = COLORS.pick;
        c.lineWidth = 1.5;
        c.strokeRect(x + 0.5, y + 1.5, w - 1, st.rowH - 3);
        c.lineWidth = 1;
      }
    }
  }
  if (st.drag?.kind === "box") {
    const d = st.drag;
    c.strokeStyle = "rgba(255,255,255,0.7)";
    c.setLineDash([4, 3]);
    c.strokeRect(Math.min(d.x0, d.x1) + 0.5, Math.min(d.y0, d.y1) + 0.5, Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
    c.setLineDash([]);
  }
  c.restore();

  // --- the playhead and the cursor, over the roll and the lanes
  line(c, xOf(st, st.cursor), 0, v.rollBottom, COLORS.cursor, [3, 3]);
  if (st.play) line(c, xOf(st, st.playAt || 0), 0, v.rollBottom, COLORS.play, []);

  bandLabels(st, c, v);
  lanes(st, c, v, t0, t1);
  keyboard(st, c, v);
  overview(st, c, v);
  markSide(st);
}

function gridLines(st, c, v, t0, t1, yTop, yBottom) {
  const L = st.L, m = st.model;
  const b0 = barAt(L.starts, t0), b1 = barAt(L.starts, Math.max(t0, t1 - 1));
  for (let b = b0; b <= b1; b++) {
    const s = L.starts[b], bar = m.bars[b], beat = beatUnits(m, bar);
    const step = st.ppu >= 7 ? 1 : st.ppu * beat >= 7 ? beat : bar.w;
    for (let u = 0; u < bar.w; u += step) {
      const x = Math.round(xOf(st, s + u)) + 0.5;
      c.strokeStyle = u === 0 ? COLORS.bar : u % beat === 0 ? COLORS.beat : COLORS.unit;
      c.beginPath(); c.moveTo(x, yTop); c.lineTo(x, yBottom); c.stroke();
    }
  }
}

function line(c, x, y0, y1, color, dash) {
  c.save();
  c.strokeStyle = color;
  c.setLineDash(dash);
  c.beginPath(); c.moveTo(Math.round(x) + 0.5, y0); c.lineTo(Math.round(x) + 0.5, y1); c.stroke();
  c.restore();
}

function lanes(st, c, v, t0, t1) {
  const L = st.L;
  c.save();
  c.beginPath(); c.rect(GUTTER, 0, v.rollW, TOP); c.clip();
  c.fillStyle = COLORS.bg; c.fillRect(GUTTER, 0, v.rollW, TOP);
  c.font = "11px system-ui, sans-serif";
  c.textBaseline = "middle";
  // The song's sections as its words have them: each lyric block from where its words begin.
  (L.words || []).forEach((w, i) => {
    if (w.to <= t0 || w.from >= t1) return;
    const x0 = xOf(st, w.from), x1 = xOf(st, w.to);
    c.fillStyle = i % 2 ? COLORS.secA : COLORS.secB;
    c.fillRect(x0, 1, Math.max(2, x1 - x0 - 1), BLK_H - 2);
    const [text, color] = blockTitle(w);
    c.fillStyle = color;
    const xl = Math.max(x0, GUTTER);              // a label stays in view while its band does
    clipText(c, text, xl + 5, BLK_H / 2, Math.max(0, x1 - xl - 8));
  });
  // Under them, the plan's own sections as YuE2 labelled them — what renaming, moving and deleting
  // work on.
  c.font = "10px system-ui, sans-serif";
  L.sections.forEach((s, i) => {
    if (s.to <= t0 || (s.from >= t1 && s.from !== s.to)) return;
    const x0 = xOf(st, s.from), x1 = xOf(st, s.to);
    c.fillStyle = "#1b1e24";
    c.fillRect(x0, BLK_H, Math.max(2, x1 - x0 - 1), PLAN_H - 1);
    c.fillStyle = COLORS.bar;
    c.fillRect(Math.round(x0), BLK_H, 1, PLAN_H - 1);
    if (i === st.secSel) {
      c.strokeStyle = COLORS.rangeEdge;
      c.lineWidth = 2;
      c.strokeRect(x0 + 1, BLK_H + 1, Math.max(2, x1 - x0 - 3), PLAN_H - 3);
      c.lineWidth = 1;
    }
    c.fillStyle = s.b0 === s.b1 ? COLORS.dim : "#9aa3b2";
    const xl = Math.max(x0, GUTTER);
    clipText(c, sectionTitle(st, i), xl + 5, BLK_H + PLAN_H / 2, Math.max(0, x1 - xl - 8));
  });
  c.font = "11px system-ui, sans-serif";
  // ruler: bar numbers where they fit, picked bars, key changes
  const yR = SEC_H;
  c.fillStyle = "#1a1d23"; c.fillRect(GUTTER, yR, v.rollW, RULER_H);
  if (st.barSel) {
    const x0 = xOf(st, L.starts[st.barSel.b0]), x1 = xOf(st, st.barSel.b1 < L.starts.length ? L.starts[st.barSel.b1] : L.total);
    c.fillStyle = COLORS.rangeEdge;
    c.fillRect(x0, yR, x1 - x0, RULER_H);
  }
  const every = Math.max(1, Math.ceil(28 / Math.max(1, st.model.bars[0].w * st.ppu)));
  const b0 = barAt(L.starts, t0), b1 = barAt(L.starts, Math.max(t0, t1 - 1));
  for (let b = b0; b <= b1; b++) {
    const x = Math.round(xOf(st, L.starts[b])) + 0.5;
    c.strokeStyle = COLORS.bar;
    c.beginPath(); c.moveTo(x, yR + (b % every ? 14 : 4)); c.lineTo(x, yR + RULER_H); c.stroke();
    if (b % every === 0) { c.fillStyle = COLORS.dim; c.fillText(String(b + 1), x + 3, yR + 8); }
  }
  for (const k of L.keys) {
    if (k.t < t0 - 1 || k.t > t1) continue;
    c.fillStyle = COLORS.key;
    c.fillText("K:" + k.value, xOf(st, k.t) + 3, yR + 17);
  }
  // chords
  const yC = SEC_H + RULER_H;
  if (st.show.chords) {
    for (const ch of L.chords) {
      if (ch.end <= t0 || ch.t >= t1 || !ch.value) continue;
      const x0 = xOf(st, ch.t), x1 = xOf(st, ch.end);
      c.fillStyle = "rgba(159,180,216,0.10)";
      c.fillRect(x0, yC + 2, Math.max(1, x1 - x0 - 1), CHORD_H - 4);
      c.fillStyle = COLORS.chord;
      clipText(c, ch.value, x0 + 3, yC + CHORD_H / 2, Math.max(0, x1 - x0 - 5));
    }
  }
  singersLane(st, c, v, t0, t1);
  lyricsLane(st, c, v, t0, t1);
  c.restore();
  // the cursor's flag in the ruler
  const xc = xOf(st, st.cursor);
  if (xc >= GUTTER) {
    c.fillStyle = COLORS.cursor;
    c.beginPath(); c.moveTo(xc - 5, SEC_H); c.lineTo(xc + 5, SEC_H); c.lineTo(xc, SEC_H + 7); c.fill();
  }
}

// A plan section's name in its lane, pinned or not.
function sectionTitle(st, i) {
  const s = st.L.sections[i];
  return (st.model.sections[i]?.sings !== undefined ? "📌 " : "") + (s.label || "—") + (s.b0 === s.b1 ? " (empty)" : "");
}

// A lyric block's name where its words are sung, with who it is marked for and its notes against
// its syllables. Amber when those are far apart, red when its notes sit in the other singer's band.
function blockTitle(w) {
  let text = `${w.label || "—"} · ${w.voice ? w.voice + " · " : ""}${w.notes}/${w.wants}`, color = COLORS.text;
  if (Math.abs(w.notes - w.wants) > Math.max(2, w.wants * 0.08)) color = COLORS.unsung;
  if (clashes(w)) { text += " ⚠"; color = COLORS.clash; }
  return [text, color];
}

function clashes(info) {
  return !!(info && info.band && info.was && info.band !== info.was);
}

// Syllables as text: one word's run together, words spaced.
function sylText(list) {
  return list.map((s, i) => (i && s.first ? " " : "") + s.s).join("");
}

// The singers lane, over the lyrics: each run the lyrics give one voice, where it is sung, in the
// colour of that singer's band — outlined red where its notes sit in the other singer's band.
function singersLane(st, c, v, t0, t1) {
  c.fillStyle = "#16181d";
  c.fillRect(GUTTER, SING_Y, v.rollW, SING_H);
  for (const r of st.L.runs) {
    if (r.to <= t0 || r.from >= t1) continue;
    const x0 = xOf(st, r.from), x1 = xOf(st, r.to), bad = runClash(r);
    c.fillStyle = r.wants === "high" ? COLORS.runHigh : r.wants === "low" ? COLORS.runLow : COLORS.runNone;
    c.fillRect(x0, SING_Y + 2, Math.max(2, x1 - x0 - 1), SING_H - 4);
    if (bad) {
      c.strokeStyle = COLORS.clash;
      c.strokeRect(x0 + 0.5, SING_Y + 2.5, Math.max(1, x1 - x0 - 2), SING_H - 5);
    }
    c.fillStyle = bad ? COLORS.clash : COLORS.text;
    const who = r.backing ? `(${r.voice || "backing"})` : r.voice || "—";
    const xl = Math.max(x0, GUTTER);
    clipText(c, (r.opens && r.label ? r.label + " · " : "") + who + (bad ? " ⚠" : ""), xl + 4, SING_Y + SING_H / 2, x1 - xl - 7);
  }
}

// One run, for the status line.
function runStory(r) {
  const band = (b) => (b === "high" ? "upper" : "lower");
  return `${r.label ? r.label + " · " : ""}${r.backing ? "backing · " : ""}${r.voice || "no singer named"}` +
    (r.wants ? ` — belongs in the ${band(r.wants)} band` : "") +
    (r.band ? `, its notes sit in the ${band(r.band)} band` + (runClash(r) ? " ⚠ — +12 / −12 hands them to the other voice" : "") : "") +
    ` · ${r.notes.length} notes · click to pick them`;
}

// The lyrics lane, under the chords: each syllable at the note that sings it. Zoomed out, where a
// syllable has no room, each LINE is written from its first note instead.
function lyricsLane(st, c, v, t0, t1) {
  const L = st.L, y = LYR_Y + LYR_H / 2;
  c.fillStyle = "#17191e";
  c.fillRect(GUTTER, LYR_Y, v.rollW, LYR_H);
  c.font = "11px system-ui, sans-serif";
  if (!st.lyrics.trim() || !L.words) {
    c.fillStyle = COLORS.dim;
    c.fillText(!st.lyrics.trim() ? "wire lyrics into the node to see here which syllable each note sings"
                                 : "laying the words on the notes…", GUTTER + 6, y);
    return;
  }
  const sung = st.model.notes.filter((n) => n.v === 0 && n._syl).sort((a, b) => a.t - b.t);
  const lines = st.ppu * 4 < 18;
  for (let i = 0; i < sung.length; i++) {
    const n = sung[i], s = n._syl;
    if (n.t > t1) break;
    const starts = i === 0 || sung[i - 1]._syl.line !== s.line;
    if (lines && !starts) continue;
    let j = i + 1;
    if (lines) while (j < sung.length && sung[j]._syl.line === s.line) j++;
    const until = j < sung.length ? sung[j].t : L.total;
    if (until < t0) continue;
    const x = xOf(st, n.t), room = xOf(st, until) - x - 4;
    if (starts) { c.fillStyle = "rgba(234,223,190,0.45)"; c.fillRect(Math.round(x), LYR_Y + 3, 1, LYR_H - 6); }
    c.fillStyle = s.backing ? COLORS.backing : COLORS.lyric;
    const text = lines ? (st.words.lines[s.line] || s.s) : s.s + (s.last ? "" : "-");
    clipText(c, text, x + 3, y, room);
  }
  // Lines with no notes, laid over the silence they are sung over — the way the model sang a bridge
  // over an interlude.
  for (const free of freeLines(L)) {
    if (free.to <= t0 || free.from >= t1) continue;
    const x0 = xOf(st, free.from), x1 = xOf(st, free.to);
    c.fillStyle = "rgba(234,223,190,0.45)";
    c.fillRect(Math.round(x0), LYR_Y + 3, 1, LYR_H - 6);
    c.fillStyle = COLORS.backing;
    clipText(c, st.words.lines[free.line] || "", x0 + 3, y, x1 - x0 - 6);
  }
}

function clipText(c, text, x, y, maxW) {
  if (maxW < 8) return;
  let s = text;
  while (s.length > 1 && c.measureText(s).width > maxW) s = s.slice(0, -2) + "…";
  c.fillText(s, x, y);
}

function keyboard(st, c, v) {
  c.save();
  c.fillStyle = "#121418";
  c.fillRect(0, 0, GUTTER, v.rollBottom);
  c.beginPath(); c.rect(0, v.rollTop, GUTTER, v.rollH); c.clip();
  c.font = "10px system-ui, sans-serif";
  c.textBaseline = "middle";
  const yEnd = contentBottom(st);
  const pTop = Math.min(st.range.hi, pOf(st, v.rollTop)), pBottom = Math.max(st.range.lo, pOf(st, yEnd - 1));
  const hoverP = st.hover && st.hover.y >= v.rollTop && st.hover.y < yEnd ? pOf(st, st.hover.y) : null;
  for (let p = pBottom; p <= pTop; p++) {
    const y = yOf(st, p), black = NAMES_SHARP[((p % 12) + 12) % 12].length > 1;
    c.fillStyle = p === hoverP ? "#3a4150" : black ? "#2a2d33" : "#d8dbe0";
    c.fillRect(0, y + 0.5, black ? GUTTER * 0.62 : GUTTER - 1, st.rowH - 1);
    if (p % 12 === 0 || p === hoverP) {
      c.fillStyle = p === hoverP ? "#fff" : "#555b66";
      c.fillText(pitchName(p), GUTTER - 26, y + st.rowH / 2);
    }
  }
  c.restore();
  c.fillStyle = "#121418";
  c.fillRect(0, 0, GUTTER, TOP);
}

// The two voices' names, on the roll just inside the keyboard, either side of the band line.
function bandLabels(st, c, v) {
  if (!st.show.bands || !st.model.bands) return;
  const ySplit = TOP + (st.range.hi - st.model.bands.split + 0.5) * st.rowH - st.scrollY;
  if (ySplit < v.rollTop + 12 || ySplit > v.rollBottom - 12) return;
  c.font = "10px system-ui, sans-serif";
  c.textBaseline = "middle";
  const pill = (text, y, color) => {
    const w = c.measureText(text).width + 8;
    c.fillStyle = "rgba(16,18,22,0.85)";
    c.fillRect(GUTTER + 3, y - 7, w, 14);
    c.fillStyle = color;
    c.fillText(text, GUTTER + 7, y);
  };
  const who = st.words?.singers || {};
  pill("▲ " + ((who.high || []).join(", ") || "upper voice"), ySplit - 9, COLORS.vocalHigh);
  pill("▼ " + ((who.low || []).join(", ") || "lower voice"), ySplit + 9, COLORS.vocalLow);
}

function overview(st, c, v) {
  const L = st.L, m = st.model, y0 = v.rollBottom, h = OVER_H;
  c.fillStyle = "#101216";
  c.fillRect(0, y0, v.w, h);
  if (!L.total) return;
  const sx = v.w / L.total;
  L.sections.forEach((s, i) => {
    c.fillStyle = i % 2 ? "#1b1f27" : "#171a21";
    c.fillRect(s.from * sx, y0, (s.to - s.from) * sx, h);
  });
  const { lo, hi } = st.range;
  for (const n of m.notes) {
    if (!st.show[n.v]) continue;
    c.fillStyle = n.v ? COLORS.ins : COLORS.vocal;
    const y = y0 + 3 + (hi - n.p) / (hi - lo + 1) * (h - 6);
    c.fillRect(n.t * sx, y, Math.max(1, n.d * sx), 1.5);
  }
  const vx = st.scrollX * sx, vw = (v.rollW / st.ppu) * sx;
  c.strokeStyle = "rgba(255,255,255,0.55)";
  c.strokeRect(Math.round(vx) + 0.5, y0 + 0.5, Math.max(4, vw), h - 1);
  c.fillStyle = COLORS.cursor;
  c.fillRect(st.cursor * sx, y0, 1, h);
  if (st.play) { c.fillStyle = COLORS.play; c.fillRect((st.playAt || 0) * sx, y0, 1.5, h); }
}

// ------------------------------------------------------------------------------------- playback
// Plain oscillators, scheduled a little ahead of the clock: no samples, nothing downloaded. Vocal is
// a triangle, Ins a filtered saw, chords a quiet sine pad — enough to hear the line, the band and
// the harmony, which is what you are checking before YuE2 spends minutes on it.
function play(st) {
  stop(st);
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) { say(st, "⚠ this browser has no Web Audio"); return; }
  const ctx = st.audio || (st.audio = new Ctx());
  if (ctx.state === "suspended") ctx.resume();
  const L = st.L, from = st.cursor >= L.total ? 0 : st.cursor;
  const events = [];
  for (const n of st.model.notes) {
    if (!st.show[n.v] || n.t + n.d <= from) continue;
    events.push({ s: Math.max(n.t, from), e: n.t + n.d, p: n.p, kind: n.v });
  }
  if (st.show.chords) {
    for (const ch of L.chords) {
      if (!ch.value || ch.end <= from) continue;
      for (const p of chordTones(ch.value)) events.push({ s: Math.max(ch.t, from), e: ch.end, p, kind: 2 });
    }
  }
  events.sort((a, b) => a.s - b.s);
  const master = ctx.createGain();
  master.gain.value = 0.8;
  master.connect(ctx.destination);
  const P = { ctx, master, t0: ctx.currentTime + 0.08, from, events, i: 0, unitSec: L.unitSec, total: L.total };
  st.play = P;
  const tick = () => {
    if (st.play !== P) return;
    const horizon = ctx.currentTime + 0.5;
    while (P.i < P.events.length) {
      const ev = P.events[P.i];
      const s = P.t0 + (ev.s - P.from) * P.unitSec;
      if (s > horizon) break;
      tone(ctx, master, ev, s, P.t0 + (ev.e - P.from) * P.unitSec);
      P.i++;
    }
    if (ctx.currentTime > P.t0 + (P.total - P.from) * P.unitSec + 0.3) stop(st);
  };
  P.timer = setInterval(tick, 60);
  tick();
  const frame = () => {
    if (st.play !== P) return;
    st.playAt = P.from + Math.max(0, ctx.currentTime - P.t0) / P.unitSec;
    const v = view(st), x = xOf(st, st.playAt);
    if (x > v.w - 40 || x < GUTTER) { st.scrollX = st.playAt - 2 / st.ppu; clampScroll(st); }
    st.ui.timeBox.textContent = mmss(st.playAt * P.unitSec);
    paint(st);
    P.raf = requestAnimationFrame(frame);
  };
  P.raf = requestAnimationFrame(frame);
  st.ui.playBtn.textContent = "■ Stop";
}

function tone(ctx, out, ev, s, e) {
  const osc = ctx.createOscillator(), g = ctx.createGain();
  osc.type = ev.kind === 0 ? "triangle" : ev.kind === 1 ? "sawtooth" : "sine";
  osc.frequency.value = 440 * Math.pow(2, (ev.p - 69) / 12);
  const peak = ev.kind === 0 ? 0.22 : ev.kind === 1 ? 0.06 : 0.035;
  const attack = 0.008, release = Math.min(0.06, Math.max(0.01, (e - s) / 3));
  g.gain.setValueAtTime(0, s);
  g.gain.linearRampToValueAtTime(peak, s + attack);
  g.gain.setValueAtTime(peak, Math.max(s + attack, e - release));
  g.gain.linearRampToValueAtTime(0, e);
  let head = osc;
  if (ev.kind === 1) {
    const f = ctx.createBiquadFilter();
    f.type = "lowpass"; f.frequency.value = 1400;
    osc.connect(f); head = f;
  }
  head.connect(g);
  g.connect(out);
  osc.start(s);
  osc.stop(e + 0.02);
}

function stop(st) {
  const P = st.play;
  if (!P) return;
  st.play = null;
  clearInterval(P.timer);
  cancelAnimationFrame(P.raf);
  try { P.master.disconnect(); } catch (e) { /* already gone */ }
  st.ui.playBtn.textContent = "▶ Play";
  refresh(st);
}

// ---------------------------------------------------------------------------------------- DOM bits
function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function button(text, tip, onClick) {
  const b = el("button", "kbse-btn");
  b.textContent = text;
  b.title = tip;
  b.addEventListener("click", (e) => { e.stopPropagation(); onClick(); });
  return b;
}

function label(text) {
  const s = el("span", "kbse-label");
  s.textContent = text;
  return s;
}

function select(options, value, onChange) {
  const s = el("select", "kbse-select");
  for (const [v, text] of options) {
    const o = el("option");
    o.value = String(v);
    o.textContent = text;
    if (String(v) === String(value)) o.selected = true;
    s.append(o);
  }
  s.value = String(value);
  s.addEventListener("change", () => { onChange(s.value); s.blur(); });
  return s;
}

function sep() {
  return el("span", "kbse-sep");
}

function toggle(text, color, get, set) {
  const b = el("button", "kbse-btn kbse-toggle");
  const dot = el("span", "kbse-dot");
  dot.style.background = color;
  b.append(dot, document.createTextNode(text));
  const sync = () => b.classList.toggle("off", !get());
  sync();
  return { el: b, flip: () => { set(!get()); sync(); } };
}

function injectStyle() {
  if (document.getElementById("kbse-style")) return;
  const s = el("style");
  s.id = "kbse-style";
  s.textContent = `
.kbse-overlay{position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.55);display:flex;align-items:stretch;justify-content:stretch;padding:18px;box-sizing:border-box}
.kbse-window{flex:1;display:flex;flex-direction:column;background:#15171c;border:1px solid #2e333d;border-radius:8px;overflow:hidden;box-shadow:0 10px 40px rgba(0,0,0,.6);font:12px system-ui,sans-serif;color:#c9ced8}
.kbse-bar{display:flex;align-items:center;gap:6px;padding:7px 10px;background:#1b1e24;border-bottom:1px solid #2a2e37;flex-wrap:wrap}
.kbse-tools{padding:5px 10px;background:#181b20}
.kbse-title{font-weight:600;color:#e6e9ef;margin-right:6px}
.kbse-btn{background:#262a32;color:#d6dae2;border:1px solid #363b46;border-radius:5px;padding:3px 9px;font:12px system-ui,sans-serif;cursor:pointer;display:inline-flex;align-items:center;gap:5px}
.kbse-btn:hover{background:#2f3440}
.kbse-btn:disabled{opacity:.4;cursor:default}
.kbse-toggle.off{opacity:.45}
.kbse-dot{width:9px;height:9px;border-radius:2px;display:inline-block}
.kbse-sep{width:1px;height:18px;background:#343944;margin:0 4px}
.kbse-spacer{flex:1}
.kbse-label{color:#8e96a5}
.kbse-select,.kbse-num{background:#20242b;color:#d6dae2;border:1px solid #363b46;border-radius:5px;padding:2px 4px;font:12px system-ui,sans-serif}
.kbse-num{width:52px}
.kbse-ctx{display:inline-flex;align-items:center;gap:6px;flex-wrap:wrap;color:#8e96a5;flex:1 1 220px;min-width:0}
.kbse-hint{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0;max-width:100%}
.kbse-time{font-variant-numeric:tabular-nums;color:#9aa3b2;min-width:52px}
.kbse-stats{color:#9aa3b2;margin-right:6px}
.kbse-body{flex:1;display:flex;min-height:0}
.kbse-main{flex:1;position:relative;min-height:0;min-width:0}
.kbse-grip{flex:0 0 5px;cursor:col-resize;background:#1b1e24;border-left:1px solid #2a2e37;touch-action:none}
.kbse-grip:hover{background:#2c3342}
.kbse-side{flex:0 0 auto;overflow:auto;background:#17191e;padding:8px 6px 28px;box-sizing:border-box;line-height:1.55}
.kbse-line{padding:0 6px;border-radius:3px;white-space:pre-wrap;overflow-wrap:anywhere;color:#5f6673}
.kbse-line.sung{color:#d6dae2;cursor:pointer}
.kbse-line.sung:hover{background:#21252d}
.kbse-line.mark{color:#e3c27a;margin-top:4px}
.kbse-line.seen{background:rgba(120,160,255,.10)}
.kbse-line.here{background:rgba(120,160,255,.30);color:#fff}
.kbse-canvas{position:absolute;inset:0;display:block;touch-action:none}
.kbse-inline{position:absolute;z-index:2;width:110px;height:${CHORD_H}px;box-sizing:border-box;background:#0f1114;color:#fff;border:1px solid #7a9be0;border-radius:3px;padding:0 4px;font:12px system-ui,sans-serif}
.kbse-toast{position:absolute;left:50%;top:${TOP + 10}px;transform:translateX(-50%);padding:8px 14px;border-radius:6px;font:600 13px system-ui,sans-serif;pointer-events:none;opacity:0;transition:opacity .25s;max-width:80%;text-align:center}
.kbse-toast.show{opacity:1}
.kbse-toast.ok{background:#1e3a29;color:#b9f5c9;border:1px solid #2f6a45}
.kbse-toast.bad{background:#3d1f22;color:#ffc1c6;border:1px solid #7a343b}
.kbse-foot{display:flex;gap:16px;padding:5px 10px;background:#1b1e24;border-top:1px solid #2a2e37;color:#8e96a5;min-height:16px;white-space:nowrap;overflow:hidden}
.kbse-hover{flex:1 1 0;min-width:0;overflow:hidden;text-overflow:ellipsis}
.kbse-msg{flex:0 1 auto;max-width:75%;text-align:right;overflow:hidden;text-overflow:ellipsis}
`;
  document.head.append(s);
}
