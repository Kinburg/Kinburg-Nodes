import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STUBS } from "./stubs.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "..", "web") + path.sep;
const OUT = path.join(HERE, "run_ouroboros_log.mjs");
// Harness for web/ouroboros_log.js — the Ouroboros Live Log node, driven the way ComfyUI drives
// it: the extension patches the node TYPE's prototype, LiteGraph news up instances and calls
// onNodeCreated / onConfigure / onRemoved, nodes enter and leave app.graph, and the backend's
// events arrive on the shared websocket listener.
//
// What it keeps honest is the failure that was actually reported: the log going quiet. A log
// showing the wrong rows is a nuisance; a log showing NOTHING while the loop runs looks like the
// sampler hung. So the checks are about DELIVERY —
//
//   * every live node gets every event, whatever the other nodes on the canvas do (one node's
//     row throwing used to escape the listener and cost every node after it that event);
//   * a node the frontend rebuilt without telling us is dropped, so a stale copy can neither take
//     the live one's place nor record the run's history a second time under the same id;
//   * and the one silence that is NOT a bug — a re-run ComfyUI served from its execution cache,
//     so the sampler never ran at all — says so instead of showing an empty log.

const strip = (p) => fs.readFileSync(p, "utf8")
  .split("\n").filter((l) => !/^import\s/.test(l)).join("\n")
  .replace(/^export\s+(function|const|async function)/gm, "$1");

// The websocket + LiteGraph halves of ComfyUI that the extension talks to.
const EXTRA = String.raw`
globalThis.performance = { now: () => Date.now() };

const BUS = [];                      // [channel, handler] pairs the module registered
api.addEventListener = (ch, fn) => { BUS.push([ch, fn]); };
const fire = (ch, detail) => { for (const [c, fn] of BUS) if (c === ch) fn({ detail }); };
const send = (d) => fire("kinburg.ouroboros", d);

// LiteGraph.createNode: a new instance off the patched prototype, then onNodeCreated().
const mkType = () => function NodeType() {};
const mkNode = (Type, id, cls) => {
  const n = new Type();
  n.id = id;
  n.size = [400, 500];
  n.comfyClass = cls || "KinburgOuroborosLog";
  n.addDOMWidget = function () {};
  n.setSize = function (s) { this.size = s; };
  n.onNodeCreated?.();
  n.graph = app.graph;               // LGraph.add()
  app.graph._nodes.push(n);
  return n;
};
// LGraph.remove(): the node loses its graph. "told" is whether the frontend also ran onRemoved —
// it does not always, which is the whole point of the liveness sweep.
const rmNode = (n, told) => {
  app.graph._nodes = app.graph._nodes.filter((x) => x !== n);
  n.graph = null;
  if (told) n.onRemoved?.();
};
`;

const EXPOSE = `
globalThis.OB = { instances, logStore, applyEvent, restore, liveNodes };
`;

const TESTS = String.raw`
const fails = [];
const check = (label, cond, extra) => {
  console.log((cond ? "  ok   " : "  FAIL ") + label + (extra !== undefined ? "  " + extra : ""));
  if (!cond) fails.push(label);
};
const OB = globalThis.OB;   // send / fire / mkType / mkNode / rmNode are in scope from EXTRA

const ext = EXTS.find((e) => e.name === "Kinburg.OuroborosLog");
check("the extension registers", !!ext);

const Type = mkType();
await ext.beforeRegisterNodeDef(Type, { name: "KinburgOuroborosLog" });
check("it listens on kinburg.ouroboros", BUS.some(([c]) => c === "kinburg.ouroboros"));

const rows = (n) => n._obEls.list.children.length;
const status = (n) => n._obEls.statusTxt.textContent;
const iter = (i) => ({ type: "iteration", i, total: 3, score: 4, score_max: 5, seed: 1,
                       gen_seconds: 2, prompt: "p" + i });

// -- one node, one run ------------------------------------------------------------------------
const a = mkNode(Type, 5);
send({ type: "start", total: 3, ts: "10:00:00" });
send(iter(1));
send(iter(2));
check("a fresh node is found and every iteration lands", rows(a) === 2, rows(a));
check("the status line follows", /iteration 2\/3/.test(status(a)), status(a));

// -- the second run of the same workflow -------------------------------------------------------
send({ type: "start", total: 3, ts: "10:05:00" });
check("a new run empties the log", rows(a) === 0, rows(a));
send(iter(1));
check("...and then writes into it again", rows(a) === 1, rows(a));

// -- the frontend rebuilds the node and tells us (tab switch / undo / reload) -------------------
rmNode(a, true);
const b = mkNode(Type, 5);          // same workflow id, new object
b.onConfigure();
check("the replacement replays what the old one had", rows(b) === 1, rows(b));
send(iter(2));
check("...and live events keep landing on it", rows(b) === 2, rows(b));

// -- the same rebuild, but onRemoved never fired -----------------------------------------------
rmNode(b, false);
const c = mkNode(Type, 5);          // third object, same id; the dead one is still in the set
c.onConfigure();
send(iter(3));
check("the live node still gets the event", rows(c) === 3, rows(c));
check("a node that left the canvas is dropped even unannounced", !OB.instances.has(b));
const seen = (OB.logStore.get(5) || []).filter((d) => d.type === "iteration" && d.i === 3).length;
check("one event is recorded ONCE, not once per stale copy", seen === 1, seen);

// -- one bad node must not silence the others --------------------------------------------------
const d1 = mkNode(Type, 7);
const d2 = mkNode(Type, 8);
const before = rows(d2);
d1._obEls.list.appendChild = () => { throw new Error("detached"); };
send(iter(4));
check("a node that throws does not eat the event for the next one", rows(d2) === before + 1,
      rows(d2) + " vs " + before);

// -- the silence that is not a bug: ComfyUI served the sampler from its cache -------------------
mkNode(mkType(), 11, "KinburgOuroboros");
fire("execution_start", { prompt_id: "p1" });
fire("execution_cached", { nodes: [11], prompt_id: "p1" });
fire("execution_success", { prompt_id: "p1" });
check("a cached re-run explains itself instead of sitting blank",
      /cache/.test(status(d2)), status(d2));

fire("execution_start", { prompt_id: "p2" });
fire("execution_cached", { nodes: [99], prompt_id: "p2" });   // some other node, not ours
send({ type: "start", total: 2, ts: "11:00:00" });
fire("execution_success", { prompt_id: "p2" });
check("a run that DID happen gets no such notice", !/cache/.test(status(d2)), status(d2));

console.log("\n" + (fails.length ? "FAILED: " + fails.join(", ") : "ALL PASS"));
process.exit(fails.length ? 1 : 0);
`;

fs.writeFileSync(OUT, STUBS + EXTRA + strip(WEB + "ouroboros_log.js") + EXPOSE + TESTS, "utf8");
console.log("wrote " + path.basename(OUT));
