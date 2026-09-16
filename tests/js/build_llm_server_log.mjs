import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STUBS } from "./stubs.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "..", "web") + path.sep;
const OUT = path.join(HERE, "run_llm_server_log.mjs");
// Harness for web/llm_server_log.js — the LLM Server Live Log node, driven the way ComfyUI drives
// it: the extension patches the node TYPE's prototype, LiteGraph news up instances and calls
// onNodeCreated / onConfigure / onRemoved, nodes enter and leave app.graph, and the gateway's
// events arrive on the shared websocket listener.
//
// Three things are worth a test here, and they are not the pretty ones:
//
//   * a request is TWO events. The row opened when it arrives must be the row filled in when it
//     finishes — a second row instead would double every line in a long chat, and an unmatched
//     'done' must still show rather than vanish;
//   * this log fetches a BACKLOG, because the server is talked to while ComfyUI is closed. The
//     backlog and the live websocket overlap by design, so the same event arrives twice and must
//     render once;
//   * it must not leak the conversation. The backend decides whether message text is sent at all
//     (`log_text`); the renderer's job is to put nothing on the canvas that was not in the event.

const strip = (p) => fs.readFileSync(p, "utf8")
  .split("\n").filter((l) => !/^import\s/.test(l)).join("\n")
  .replace(/^export\s+(function|const|async function)/gm, "$1");

const EXTRA = String.raw`
globalThis.performance = { now: () => Date.now() };

const BUS = [];                      // [channel, handler] pairs the module registered
api.addEventListener = (ch, fn) => { BUS.push([ch, fn]); };
const fire = (ch, detail) => { for (const [c, fn] of BUS) if (c === ch) fn({ detail }); };

let SEQ = 0;
const ev = (d) => ({ t: "10:00:00", seq: ++SEQ, ...d });
const send = (d) => fire("kinburg.llmserver", d);

// What /kinburg/llm_server/log answers with, and every call made to it.
let BACKLOG = { events: [], seq: 0, status: null };
const CALLS = [];
api.fetchApi = (route, opt) => {
  CALLS.push([route, opt]);
  return Promise.resolve({ status: 200, json: () => Promise.resolve(BACKLOG) });
};

const mkType = () => function NodeType() {};
const mkNode = (Type, id, cls) => {
  const n = new Type();
  n.id = id;
  n.size = [460, 500];
  n.comfyClass = cls || "LLMServerLog";
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

// Every string rendered anywhere under a row, so a test can ask "is this text on the canvas?"
const textOf = (node) => {
  const walk = (e) => (e.children || []).reduce((acc, c) => acc + " " + walk(c), e.textContent || "");
  return (node._lsEls.list.children || []).map(walk).join(" ");
};
`;

const EXPOSE = `
globalThis.LS = { instances, logStore, applyEvent, restore, liveNodes, fmtSampler, fmtCtx };
`;

const TESTS = String.raw`
const fails = [];
const check = (label, cond, extra) => {
  console.log((cond ? "  ok   " : "  FAIL ") + label + (extra !== undefined ? "  " + extra : ""));
  if (!cond) fails.push(label);
};
const LS = globalThis.LS;

const ext = EXTS.find((e) => e.name === "Kinburg.LLMServerLog");
check("the extension registers", !!ext);

const Type = mkType();
await ext.beforeRegisterNodeDef(Type, { name: "LLMServerLog" });
check("it listens on kinburg.llmserver", BUS.some(([c]) => c === "kinburg.llmserver"));

const rows = (n) => n._lsEls.list.children.length;
const status = (n) => n._lsEls.statusTxt.textContent;

// -- a node appearing asks for the backlog, because the chat happened while nobody was looking --
BACKLOG = {
  seq: 2,
  status: { listening: "http://127.0.0.1:5001", model_loaded: true, model: "m.gguf", requests: 7 },
  events: [ev({ kind: "gateway", event: "listening", addr: "http://127.0.0.1:5001" }),
           ev({ kind: "model", event: "ready", role: "chat", model: "m.gguf", ms: 4600 })],
};
const a = mkNode(Type, 5);
await tick();
check("a fresh node fetches the backlog",
      CALLS.some(([r]) => String(r).startsWith("/kinburg/llm_server/log")), CALLS[0]);
check("...and renders what happened before it existed", rows(a) === 2, rows(a));
check("...with the gateway's live status, not the backlog's last line",
      /5001 .* m\.gguf/.test(status(a)) && /7 requests/.test(status(a)), status(a));

// -- the backlog and the websocket overlap: the same event must render once -------------------
for (const d of BACKLOG.events) send(d);
check("an event already replayed from the backlog is not rendered twice", rows(a) === 2, rows(a));

// -- one request is two events, and ONE row ----------------------------------------------------
const req = ev({ kind: "request", method: "POST", path: "/v1/chat/completions", role: "chat",
                 messages: 24, chars: 18400, stream: true, n_ctx: 8192,
                 sampler: { temperature: 1.05, min_p: 0.05, dry_multiplier: 0.8, dry_base: 1.75 } });
send(req);
check("a request opens a row when it ARRIVES", rows(a) === 3, rows(a));
check("...showing the shape of what was sent", /24 msg/.test(textOf(a)), textOf(a).slice(-120));
check("...and the sampler settings the client chose",
      /temp 1\.05/.test(textOf(a)) && /min_p 0\.05/.test(textOf(a)) && /dry 0\.80/.test(textOf(a)));
check("...marked as still running", /…/.test(textOf(a)));

send(ev({ kind: "request_done", id: req.seq, status: 200, ms: 12400,
          prompt_tokens: 1420, gen_tokens: 180, tps: 14.5 }));
check("finishing fills in the SAME row rather than adding another", rows(a) === 3, rows(a));
const done = textOf(a);
check("...with the status, the time and the tokens",
      /200/.test(done) && /12s/.test(done) && /1420\+180 tok/.test(done), done.slice(-160));
check("...and the context fill against the server's n_ctx",
      /ctx 1420\/8192 \(17%\)/.test(done), done.slice(-90));

// -- a 'done' whose 'request' we never saw must still show up ----------------------------------
send(ev({ kind: "request_done", id: 9999, status: 200, ms: 120, path: "/v1/embeddings" }));
check("an unmatched result is a row of its own, not a silent drop", rows(a) === 4, rows(a));

// -- errors ------------------------------------------------------------------------------------
const bad = ev({ kind: "request", method: "POST", path: "/v1/chat/completions" });
send(bad);
send(ev({ kind: "request_done", id: bad.seq, status: 400, ms: 30,
          error: "Field 'dry_sequence_breakers': must be a non-empty array" }));
const errRow = a._lsEls.list.children[rows(a) - 1];
check("a failed request is marked as one", errRow.classList.contains("ls-err"));
check("...and says what the server actually complained about",
      /dry_sequence_breakers/.test(textOf(a)), textOf(a).slice(-120));

// -- the conversation is the backend's to send, and it did not ---------------------------------
check("nothing renders message text that was not in the event",
      !/▸/.test(textOf(a)) && !/◂/.test(textOf(a)));
const talk = ev({ kind: "request", method: "POST", path: "/v1/chat/completions",
                  prompt: "what do you see?" });
send(talk);
send(ev({ kind: "request_done", id: talk.seq, status: 200, ms: 900, reply: "a room" }));
check("...and renders it when it IS (log_text on)",
      /what do you see\?/.test(textOf(a)) && /a room/.test(textOf(a)));

// -- polling must not bury the log -------------------------------------------------------------
const before = rows(a);
send(ev({ kind: "probe", path: "/v1/models" }));
send(ev({ kind: "probe", path: "/v1/models" }));
send(ev({ kind: "probe", path: "/v1/models" }));
check("a run of identical polls collapses into one row", rows(a) === before + 1, rows(a) - before);
check("...with a counter", /\/v1\/models ×3/.test(textOf(a)), textOf(a).slice(-60));
send(ev({ kind: "note", text: "dry_sequence_breakers: str -> array of 4" }));
send(ev({ kind: "probe", path: "/v1/models" }));
check("a real row breaks the run, so the next poll starts a new one", rows(a) === before + 3,
      rows(a) - before);

// -- lifecycle drives the status line ----------------------------------------------------------
send(ev({ kind: "model", event: "unloaded", role: "chat", reason: "a ComfyUI prompt was queued" }));
check("an unload says so in the status line", /unloaded/.test(status(a)), status(a));
send(ev({ kind: "model", event: "loading", role: "chat", model: "m.gguf" }));
check("...and a load does too", /loading m\.gguf/.test(status(a)), status(a));
send(ev({ kind: "model", event: "ready", role: "embed", model: "e.gguf", ms: 900 }));
check("the embedding model gets a row but does not claim the status line",
      /loading m\.gguf/.test(status(a)), status(a));

// -- the frontend rebuilds the node (tab switch / undo / reload) --------------------------------
// A node is only known-dead once we have SEEN it alive: onNodeCreated runs before LGraph.add()
// sets node.graph, so a sweep that trusted a null graph outright would drop a node that is merely
// still being built. Same rule as the Ouroboros log — one liveness rule for the pack, not two.
const wasRows = rows(a);
rmNode(a, true);
const b = mkNode(Type, 5);          // same workflow id, new object
b.onConfigure();
await tick();
check("the replacement replays what the old one had", rows(b) === wasRows, rows(b) + " vs " + wasRows);
check("...from memory, without asking the backend again",
      CALLS.length === 1, CALLS.length);

// -- the same rebuild, but onRemoved never fired ------------------------------------------------
send(ev({ kind: "note", text: "b is alive" }));   // a sweep runs, and sees b on the canvas
rmNode(b, false);
const c = mkNode(Type, 5);
c.onConfigure();
await tick();
send(ev({ kind: "note", text: "still here" }));
check("the live node still gets the event", /still here/.test(textOf(c)));
check("a node that left the canvas is dropped even unannounced", !LS.instances.has(b));
const seen = (LS.logStore.get(5) || []).filter((d) => d.text === "still here").length;
check("one event is recorded ONCE, not once per stale copy", seen === 1, seen);

// -- one bad node must not silence the others ---------------------------------------------------
const d1 = mkNode(Type, 7);
const d2 = mkNode(Type, 8);
await tick();
const d2rows = rows(d2);
d1._lsEls.list.appendChild = () => { throw new Error("detached"); };
send(ev({ kind: "note", text: "fan out" }));
check("a node that throws does not eat the event for the next one", rows(d2) === d2rows + 1,
      rows(d2) + " vs " + d2rows);

// -- the log is bounded -------------------------------------------------------------------------
const e1 = mkNode(Type, 12);
await tick();
for (let i = 0; i < 420; i++) send(ev({ kind: "note", text: "n" + i }));
check("an endless chat does not grow an endless DOM", rows(e1) <= 400, rows(e1));
check("...and it is the OLDEST rows that go", /n419/.test(textOf(e1)) && !/ n0 /.test(textOf(e1)));

// -- clear empties the backend's ring too, or a reload would bring it all back -------------------
const callsBefore = CALLS.length;
e1._lsEls.status.children[1].onclick({ preventDefault() {}, stopPropagation() {} });
check("clear empties the view", rows(e1) === 0, rows(e1));
check("...and asks the gateway to forget it as well",
      CALLS.slice(callsBefore).some(([r, o]) => String(r).includes("/log/clear") && o?.method === "POST"),
      JSON.stringify(CALLS.slice(callsBefore)));

// -- the sampler line, on its own ----------------------------------------------------------------
check("an empty sampler renders nothing", LS.fmtSampler({}) === "" && LS.fmtSampler(null) === "");
check("only what was sent is shown",
      LS.fmtSampler({ temperature: 0.7, top_k: 40 }) === "temp 0.70 · top_k 40",
      LS.fmtSampler({ temperature: 0.7, top_k: 40 }));
check("a full context is flagged", LS.fmtCtx({ prompt_tokens: 7800, n_ctx: 8192 }).warn);
check("...and a roomy one is not", !LS.fmtCtx({ prompt_tokens: 1000, n_ctx: 8192 }).warn);
check("no n_ctx means no percentage to invent",
      LS.fmtCtx({ prompt_tokens: 1000 }).text === "ctx 1000", LS.fmtCtx({ prompt_tokens: 1000 }).text);

console.log("\n" + (fails.length ? "FAILED: " + fails.join(", ") : "ALL PASS"));
process.exit(fails.length ? 1 : 0);
`;

fs.writeFileSync(OUT, [STUBS, EXTRA, strip(WEB + "llm_server_log.js"), EXPOSE, TESTS].join("\n"));
console.log(OUT);
