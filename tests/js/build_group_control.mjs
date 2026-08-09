import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STUBS } from "./stubs.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "..", "web") + path.sep;
const OUT = path.join(HERE, "run_group_control.mjs");
// Harness for web/group_control.js — the group LINKS engine: the pure resolver (resolveLinks) and
// the graph-facing side (toggleEntry / pollLinks / setEntryState) driven against a hand-built graph
// of groups and nodes. No DOM is involved: every panel function bails out when the node has no
// _kbEls, which is exactly the headless case, so what runs here is the real engine.

const strip = (p) => fs.readFileSync(p, "utf8")
  .split("\n").filter((l) => !/^import\s/.test(l)).join("\n")
  .replace(/^export\s+(function|const|async function)/gm, "$1");

// What the stubs don't carry: LiteGraph's mode constants (read through window) and a graph that
// has groups as well as nodes.
const EXTRA = `
globalThis.LiteGraph = { ALWAYS: 0, NEVER: 2, BYPASS: 4 };
globalThis.window.LiteGraph = globalThis.LiteGraph;
app.graph._groups = [];
app.graph.change = () => {};
`;

const EXPOSE = `
globalThis.GC = { resolveLinks, treeEntries, collectEntries, stateOf, applyMode, onOf, entriesNow,
                  setEntryState, toggleEntry, applyLinks, pollLinks, snapshotLinks, activeLinks,
                  offForEntry, removeFromLinks, describeLinks, explainLink, linkLabel, bulkApply,
                  openLinksDialog };
`;

const TESTS = `
const fails = [];
const check = (label, cond, extra) => {
  console.log((cond ? "  ok   " : "  FAIL ") + label + (extra !== undefined ? "  " + extra : ""));
  if (!cond) fails.push(label);
};
const GC = globalThis.GC;
const ALW = 0, BYP = 4, NEV = 2;    // LiteGraph modes: Always / Bypass / Never

// ── the pure resolver ───────────────────────────────────────────────────────────────────────
const follow = (ms, extra) => Object.assign(
  { id: "f" + ms.map((m) => m[0]).join(""), type: "follow", off: "bypass",
    members: ms.map((m) => ({ name: m[0], pol: m[1] })) }, extra || {});
const oneof = (names, extra) => Object.assign(
  { id: "o" + names.join(""), type: "oneof", off: "bypass",
    members: names.map((n) => ({ name: n })) }, extra || {});
const res = (links, drivers, states) =>
  GC.resolveLinks(links, drivers, (n) => (states && n in states ? states[n] : null));
const flat = (r) => [...r.assign].map((kv) => kv[0] + "=" + (kv[1].on ? "on" : "off")).join(",");

const TOGGLE = [follow([["A", 1], ["B", -1]])];
let r = res(TOGGLE, [{ name: "A", on: true }]);
check("toggle: A on -> B off", flat(r) === "A=on,B=off", flat(r));
r = res(TOGGLE, [{ name: "A", on: false }]);
check("toggle: A off -> B on", flat(r) === "A=off,B=on", flat(r));
r = res(TOGGLE, [{ name: "B", on: true }]);
check("toggle drives from either end", flat(r) === "B=on,A=off", flat(r));

const THREE = [follow([["A", 1], ["B", -1], ["C", -1]])];
r = res(THREE, [{ name: "A", on: false }]);
check("A off -> B and C on", flat(r) === "A=off,B=on,C=on", flat(r));
r = res(THREE, [{ name: "A", on: true }]);
check("A on -> B and C off", flat(r) === "A=on,B=off,C=off", flat(r));
r = res(THREE, [{ name: "B", on: true }]);
check("...and flipping a minus member stays consistent", flat(r) === "B=on,A=off,C=on", flat(r));

r = res([follow([["A", 1], ["B", 1]])], [{ name: "A", on: false }]);
check("two plus members are a mirror", flat(r) === "A=off,B=off", flat(r));

r = res([follow([["A", 1], ["B", -1]]), follow([["B", 1], ["C", -1]])], [{ name: "A", on: true }]);
check("links chain: A on -> B off -> C on", flat(r) === "A=on,B=off,C=on", flat(r));

r = res([follow([["A", 1], ["B", -1]]), follow([["B", 1], ["C", -1]]),
         follow([["C", 1], ["A", -1]])], [{ name: "A", on: true }]);
// A ring of three opposites cannot be satisfied. First assignment wins, so it settles on the
// driver's own answer instead of oscillating, and says which groups it could not satisfy.
check("a contradictory cycle terminates", flat(r) === "A=on,B=off,C=off", flat(r));
check("...and is reported rather than oscillating",
      r.conflicts.map((c) => c.name).sort().join() === "B,C",
      JSON.stringify(r.conflicts.map((c) => c.name)));

const RADIO = [oneof(["A", "B", "C"])];
r = res(RADIO, [{ name: "B", on: true }]);
check("one-of: switching B on switches the rest off", flat(r) === "B=on,A=off,C=off", flat(r));
r = res(RADIO, [{ name: "B", on: false }]);
check("one-of: switching one off leaves the rest alone", flat(r) === "B=off", flat(r));
r = res([oneof(["A", "B", "C"], { requireOne: true })], [{ name: "B", on: false }],
        { A: false, B: false, C: false });
check("one-of + requireOne brings the next member in", flat(r) === "B=off,A=on,C=off", flat(r));
r = res([oneof(["A", "B", "C"], { requireOne: true })], [{ name: "B", on: false }],
        { A: false, B: false, C: true });
check("...but not when one is already on", flat(r) === "B=off", flat(r));

r = res([], [{ name: "A", on: true }]);
check("no links, nothing to propagate", flat(r) === "A=on", flat(r));

// ── a hand-built graph: four flat groups, plus a parent holding two children ─────────────────
let nid = 0;
const N = [], G = [];
const put = (x, y) => { const n = { id: ++nid, pos: [x, y], size: [20, 20], mode: ALW }; N.push(n); return n; };
const grp = (title, x, y, w, h) => { G.push({ title, bounding: [x, y, w, h] }); };
grp("A", 0, 0, 100, 100);       const na = put(10, 10);
grp("B", 200, 0, 100, 100);     const nb = put(210, 10);
grp("C", 400, 0, 100, 100);     const nc = put(410, 10);
grp("X", 600, 0, 100, 100);     put(610, 10);
grp("P", 0, 200, 400, 200);
grp("C1", 10, 210, 100, 100);   const n1 = put(20, 220);
grp("C2", 200, 210, 100, 100);  put(210, 220);
app.graph._groups = G;
app.graph._nodes = N;

const node = { properties: {} };
const at = (name) => GC.entriesNow(node).find((e) => e.name === name);
const state = (name) => GC.stateOf(at(name));
const reset = (links) => {
  for (const n of N) n.mode = ALW;
  node.properties = links ? { _kbLinks: links } : {};
  node._kbLinkSnap = null;
  node._kbLinksPaused = false;
};

const tree = GC.treeEntries();
check("the tree finds the nested groups",
      tree.filter((e) => e.parent && e.parent.name === "P").map((e) => e.name).join() === "C1,C2",
      tree.filter((e) => e.parent).map((e) => e.name + "<" + e.parent.name).join());
check("...at depth 1", tree.filter((e) => e.depth === 1).length === 2);

// ── the engine on that graph ─────────────────────────────────────────────────────────────────
reset([follow([["A", 1], ["B", -1]])]);
GC.toggleEntry(node, at("A"), true);
check("live: switching A on bypasses B", state("B") === "bypass", state("B"));
GC.toggleEntry(node, at("A"), false);
check("live: switching A off brings B back", state("A") === "bypass" && state("B") === "always",
      state("A") + "/" + state("B"));

reset([follow([["A", 1], ["B", -1]], { off: "never" })]);
GC.toggleEntry(node, at("A"), true);
check("a link's off mode can be Never", nb.mode === NEV, nb.mode);

reset([follow([["A", 1], ["B", -1], ["C", -1]])]);
GC.toggleEntry(node, at("A"), false);
check("live: A off brings B and C on", state("B") === "always" && state("C") === "always",
      state("B") + "/" + state("C"));
GC.toggleEntry(node, at("A"), true);
check("live: A on takes B and C off", state("B") === "bypass" && state("C") === "bypass",
      state("B") + "/" + state("C"));

reset([oneof(["A", "B", "C"])]);
GC.toggleEntry(node, at("B"), true);
check("live one-of: only B survives", state("A") === "bypass" && state("C") === "bypass",
      state("A") + "/" + state("C"));

reset([follow([["A", 1], ["B", -1]], { enabled: false })]);
GC.toggleEntry(node, at("A"), true);
check("a disabled link is inert", state("B") === "always", state("B"));

// A change made anywhere else — Ctrl+B on the canvas, the core group menu — drives the links too.
reset([follow([["A", 1], ["B", -1]])]);
nb.mode = BYP;                       // the consistent starting state: A on, B off
GC.pollLinks(node);                  // first look only learns
check("the first poll never fires", state("B") === "bypass", state("B"));
na.mode = BYP;                       // as if Ctrl+B had switched A off on the canvas
GC.pollLinks(node);
check("a change made outside the panel drives the link", state("B") === "always", state("B"));
GC.pollLinks(node);
check("...and the engine does not chase its own work", state("A") === "bypass", state("A"));

reset([follow([["A", 1], ["B", -1]])]);
node._kbLinksPaused = true;
nb.mode = BYP;
GC.pollLinks(node);
na.mode = BYP;
GC.pollLinks(node);
check("paused links do nothing", state("B") === "bypass", state("B"));

// A bulk switch is an explicit override — the links stay out of it.
reset([follow([["A", 1], ["B", -1]])]);
node._kbAllEntries = GC.treeEntries();      // bulkEntries() reads the panel's cached list
GC.bulkApply(node, ALW);
check("all on means all on, links or not", state("A") === "always" && state("B") === "always",
      state("A") + "/" + state("B"));
delete node._kbAllEntries;

// ── nesting: a parent must not flatten the arrangement inside it ─────────────────────────────
reset();
GC.applyMode(at("C2"), BYP);                 // one child off, one on
GC.setEntryState(node, at("P"), false, "bypass");
check("switching the parent off takes the children with it",
      state("C1") === "bypass" && state("P") === "bypass", state("C1") + "/" + state("P"));
GC.setEntryState(node, at("P"), true);
check("...and switching it back on restores the inner arrangement",
      state("C1") === "always" && state("C2") === "bypass", state("C1") + "/" + state("C2"));

// The child's own rule is applied after the parent's blanket sweep, not before it — and the state
// is re-read as the targets go in, so a child that already looked right isn't skipped.
reset([follow([["X", 1], ["P", 1]]), follow([["X", 1], ["C1", -1]])]);
n1.mode = BYP;
GC.toggleEntry(node, at("X"), true);
check("a parent's sweep does not overwrite a nested group's own rule",
      state("C1") === "bypass" && state("C2") === "always", state("C1") + "/" + state("C2"));

// ── links data upkeep ────────────────────────────────────────────────────────────────────────
reset([follow([["A", 1], ["B", -1], ["C", -1]]), follow([["A", 1], ["X", -1]])]);
GC.removeFromLinks(node, "A");
const left = node.properties._kbLinks || [];
check("unlinking drops the group everywhere", left.length === 1, JSON.stringify(left.map((l) => l.members.map((m) => m.name))));
check("...and a link left with one member goes too",
      left[0].members.map((m) => m.name).join() === "B,C", JSON.stringify(left[0].members));

reset([follow([["A", 1], ["B", -1]]), follow([["A", 1], ["Gone", -1]])]);
GC.toggleEntry(node, at("A"), true);
check("a link naming a group that isn't there is skipped, not fatal", state("B") === "bypass", state("B"));

reset([follow([["A", 1], ["B", -1]], { off: "never" })]);
check("offForEntry reads the link", GC.offForEntry(node, "A") === "never", GC.offForEntry(node, "A"));
check("...and falls back to Bypass off a link", GC.offForEntry(node, "C") === "bypass");
check("the badge tooltip names what follows",
      /switches OFF B/.test(GC.describeLinks(node, "A")), GC.describeLinks(node, "A"));
const expl = GC.explainLink(node.properties._kbLinks[0]);
check("the editor explains the rule in words",
      expl.indexOf("A") >= 0 && expl.indexOf("B") >= 0 && expl.indexOf("on") >= 0, expl);
check("an empty link says so", /at least two/.test(GC.explainLink({ type: "follow", members: [] })));

// ── the editor ───────────────────────────────────────────────────────────────────────────────
// Only a smoke test — there is no layout here — but it does walk the whole builder, which is the
// one part of this file a browser-less run can still keep honest.
reset([follow([["A", 1], ["B", -1]]), oneof(["A", "C"], { requireOne: true })]);
GC.openLinksDialog(node);
check("the links editor builds", !!node._kbDlg);
const seen = [];
(function walk(e) {
  if (e.textContent) seen.push(e.textContent);
  if (e.placeholder) seen.push(e.placeholder);
  for (const c of e.children || []) walk(c);
})(node._kbDlg.box);
const text = seen.join(" | ");
check("...with a card per link", text.indexOf("Group links") >= 0 && text.indexOf("A ⇄ B") >= 0, text.slice(0, 90));
check("...the members as chips", text.indexOf("+") >= 0 && text.indexOf("−") >= 0);
check("...and the one-of extras", text.indexOf("keep one always on") >= 0);
node._kbDlg.close();
check("...and it closes", node._kbDlg === null);

console.log("\\n" + (fails.length ? "FAILED: " + fails.join(", ") : "ALL PASS"));
process.exit(fails.length ? 1 : 0);
`;

fs.writeFileSync(OUT, STUBS + EXTRA + strip(WEB + "group_control.js") + EXPOSE + TESTS, "utf8");
console.log("wrote " + path.basename(OUT));
