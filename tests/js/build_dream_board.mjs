import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STUBS } from "./stubs.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "..", "web") + path.sep;
const OUT = path.join(HERE, "run_dream_board.mjs");
// Harness for web/dream_board.js: chat_llm.js + dream_board.js in one scope, DOM stubbed.
// dream_board goes in a block (both files declare `CLASS`) and hands its internals out through a
// global, so the JS shot-derivation can be tested against the SAME fixtures as the python one.

const strip = (p) => fs.readFileSync(p, "utf8")
  .split("\n").filter((l) => !/^import\s/.test(l)).join("\n")
  .replace(/^export\s+(function|const|async function)/gm, "$1");

const EXPOSE = `
globalThis.DB = { planBoard, ST, stateJSON, loadState, textOf, nth, updateHistory, setup,
                  render, renderShots, heldRefs, gone };
`;

const TESTS = `
const fails = [];
const check = (label, cond, extra) => {
  console.log((cond ? "  ok   " : "  FAIL ") + label + (extra !== undefined ? "  " + extra : ""));
  if (!cond) fails.push(label);
};
const { planBoard } = globalThis.DB;

// ── the SAME fixture the python harness uses ────────────────────────────────────────────────
const A = { name: "a.png", subfolder: "kinburg_chat", type: "input" };
const B = { name: "b.png", subfolder: "kinburg_chat", type: "input" };
const C = { name: "c.png", subfolder: "kinburg_chat", type: "input" };
const m = (t, p = "Mia", r = "a", img = null) => (img ? { r, p, t, img } : { r, p, t });
const CHAT = [
  m("the kitchen at dawn", "Mia", "u"),
  m("here, look", "Mia", "a", [A]),
  m("she turns to the window"),
  m("light on the counter"),
  m("and now this", "Mia", "a", [B]),
  m("she steps outside"),
  m("last one", "Mia", "a", [C]),
  m("she walks away"),
];
const shape = (p) => p.shots.map((s) => [s.start?.name ?? null, s.end?.name ?? null, s.msgs]);

let p = planBoard(CHAT, [], [], []);
check("3 pictures -> 2 bounded shots + a tail", p.shots.length === 3, p.shots.length);
check("boundaries in order", p.bounds.map((b) => b.name).join() === "a.png,b.png,c.png",
      p.bounds.map((b) => b.name).join());
check("shot 1 = a -> b, messages 0..3",
      JSON.stringify(shape(p)[0]) === JSON.stringify(["a.png", "b.png", [0, 1, 2, 3]]),
      JSON.stringify(shape(p)[0]));
check("shot 2 = b -> c, messages 4..5",
      JSON.stringify(shape(p)[1]) === JSON.stringify(["b.png", "c.png", [4, 5]]),
      JSON.stringify(shape(p)[1]));
check("the tail is text-only from the last picture",
      JSON.stringify(shape(p)[2]) === JSON.stringify([null, null, [6, 7]]), JSON.stringify(shape(p)[2]));
check("no notes on a clean board", !p.notes.length, p.notes);

p = planBoard(CHAT, [], ["b.png"], []);
check("dropping a middle picture merges the two shots",
      p.shots.length === 2 && JSON.stringify(shape(p)[0]) === JSON.stringify(["a.png", "c.png", [0, 1, 2, 3, 4, 5]]),
      JSON.stringify(shape(p)));

p = planBoard(CHAT, [2, 3], [], []);
check("skipped messages leave the beats", JSON.stringify(p.shots[0].msgs) === "[0,1]",
      JSON.stringify(p.shots[0].msgs));

p = planBoard(CHAT, [1, 4], [], []);
check("a skipped message's picture is still a boundary",
      p.bounds.map((b) => b.name).join() === "a.png,b.png,c.png", p.bounds.map((b) => b.name).join());

p = planBoard([m("just this", "Mia", "a", [A]), m("she leaves")], [], [], []);
check("one picture -> a single text shot", p.shots.length === 1 && p.shots[0].tail === true,
      JSON.stringify(shape(p)));

p = planBoard([m("all talk"), m("no pictures", "Mia", "u")], [], [], []);
check("no pictures -> one text-only shot", p.shots.length === 1 && !p.bounds.length);

p = planBoard([], [], [], []);
check("an empty chat plans nothing", !p.shots.length && !p.bounds.length);

p = planBoard([m("", "Mia", "a", [A]), m("", "Mia", "a", [B])], [], [], []);
check("two pictures with no text still make a shot",
      p.shots.length === 1 && p.shots[0].msgs.length === 0, JSON.stringify(shape(p)));

p = planBoard(CHAT, [], [], [7]);
check("a break in the tail splits it",
      p.shots.length === 4 && JSON.stringify(p.shots[2].msgs) === "[6]"
      && JSON.stringify(p.shots[3].msgs) === "[7]", JSON.stringify(shape(p)));
check("...with no complaint", !p.notes.length, p.notes);

p = planBoard(CHAT, [], [], [3]);
check("a break inside a keyframed span is ignored", p.shots.length === 3, p.shots.length);
check("...and reported", p.notes.length === 1 && p.notes[0].includes("#3"), p.notes);

// helpers shared with the python side
check("textOf collapses whitespace", globalThis.DB.textOf({ t: "a\\n\\n b  c" }) === "a b c",
      globalThis.DB.textOf({ t: "a\\n\\n b  c" }));
check("nth repeats the last value", globalThis.DB.nth([5, 8], 3, 1) === 8);
check("nth falls back when empty", globalThis.DB.nth([], 0, 5.17) === 5.17);

// ── the node, through the real extension hook ──────────────────────────────────────────────
const NodeType = function () {};
NodeType.prototype = {};
await extBy("Kinburg.DreamBoard").beforeRegisterNodeDef(NodeType, { name: "KinburgDreamBoard" });

function mkBoard() {
  const n = {
    id: 5, title: "Dream Board", properties: {}, widgets: [{ name: "board_state", value: "" }],
    size: [420, 460], setSize(s) { n.size = s; }, setDirtyCanvas() {},
    addWidget(type, name, value, callback, options) {
      const w = { type, name, value, callback, options: options || {} };
      n.widgets.push(w); return w;
    },
    addDOMWidget(name, t, element, opts) {
      const w = { name, type: t, element, ...(opts || {}) };
      n.widgets.push(w); n._dom = w; return w;
    },
  };
  NodeType.prototype.onNodeCreated.call(n);
  return n;
}

const board = mkBoard();
check("board_state's auto widget is spliced out",
      !board.widgets.some((w) => w.name === "board_state" && w.type !== "kinburg_dreamboard"),
      board.widgets.map((w) => w.name).join());
check("it carries the state itself", board._dom?.name === "board_state" && !!board._dom.getValue);
check("the picker and the button are there",
      !!board.widgets.find((w) => w.name === "→ chat") && !!board.widgets.find((w) => w.name === "⟳ Update History"));
check("an empty board says what to do",
      JSON.stringify(board._dbView.children).includes("Update History"), "");

// a chat node to pull from — one persona is private, the "camera" pattern
const chat = { id: 9, title: "Chat", comfyClass: "LocalLLMChatGGUF", properties: {
  _kbPersonas: [{ label: "Mia" }, { label: "Camera", private: true }] },
  widgets: [], inputs: [{ name: "persona_1", link: 1 }, { name: "persona_2", link: 2 },
                        { name: "persona_3", link: null }, { name: "persona_4", link: null },
                        { name: "persona_5", link: null }, { name: "persona_6", link: null }],
  getInputNode: () => ({ title: "x", constructor: { title: "y" }, widgets: [] }),
  setDirtyCanvas() {},
  _kbState: { v: 1, user: "", nonce: 0, approved: false, persona: 1, turn: null, att: [],
    history: [
      { role: "assistant", content: "here, look", persona: "Mia",
        att: [{ name: "a.png", subfolder: "kinburg_chat", type: "input", ctx: false }] },
      { role: "user", content: "tighter, more bokeh", persona: "Camera" },   // an instruction TO a camera
      { role: "assistant", content: "1girl, kitchen, soft light", persona: "Camera" },
      { role: "user", content: "you look lovely", persona: "Mia" },          // dialogue, must stay
      { role: "user", content: "untagged aside" },                           // no persona, must stay
      { role: "digest", content: "a summary" },
      { role: "assistant", content: "and now this", persona: "Mia",
        att: [{ name: "b.png", subfolder: "kinburg_chat", type: "input" }] },
    ] } };
app.graph._nodes = [chat, board];

board.widgets.find((w) => w.name === "⟳ Update History").callback();
const st = globalThis.DB.ST(board);
check("Update History pulled the conversation", st.msgs.length === 6, st.msgs.length);
check("the digest was left out", !st.msgs.some((x) => x.t === "a summary"));
check("pictures came across as refs",
      st.msgs[0].img?.[0]?.name === "a.png" && st.msgs[0].img[0].ctx === undefined,
      JSON.stringify(st.msgs[0].img));

// auto-unticking goes BY PERSONA, not by role: both halves of a camera exchange are noise, while
// the author's dialogue with the character is half the story and has to survive
const skipped = st.skip.map((i) => st.msgs[i].r + ":" + st.msgs[i].p).sort().join(" | ");
check("both halves of the camera exchange start unticked", skipped === "a:Camera | u:Camera",
      skipped);
check("dialogue with the character is kept",
      !st.skip.some((i) => st.msgs[i].t === "you look lovely"), JSON.stringify(st.skip));
check("an untagged aside is kept", !st.skip.some((i) => st.msgs[i].t === "untagged aside"));
const beatsOf = (s) => s.msgs.map((i) => st.msgs[i].t).join(" / ");
const plan0 = planBoard(st.msgs, st.skip, st.noimg, st.breaks);
check("...so the shot's direction has the dialogue and none of the camera talk",
      /you look lovely/.test(beatsOf(plan0.shots[0])) && !/bokeh|1girl/.test(beatsOf(plan0.shots[0])),
      beatsOf(plan0.shots[0]));
check("the source chat is remembered", st.src === "9", st.src);
check("two pictures -> one bounded shot",
      plan0.shots.filter((s) => !s.tail).length === 1);

// picks survive a re-pull, and only new messages get auto-unticked
st.skip = [3];
chat._kbState.history.push({ role: "assistant", content: "tags again", persona: "Camera" });
board.widgets.find((w) => w.name === "⟳ Update History").callback();
check("an existing pick is kept across a re-pull", globalThis.DB.ST(board).skip.includes(3),
      JSON.stringify(globalThis.DB.ST(board).skip));
check("...and the newly arrived camera turn is unticked too",
      globalThis.DB.ST(board).skip.includes(6), JSON.stringify(globalThis.DB.ST(board).skip));

// a stale noimg entry is dropped rather than kept forever
globalThis.DB.ST(board).noimg.push("ghost.png");
board.widgets.find((w) => w.name === "⟳ Update History").callback();
check("a picture that left the chat drops out of noimg",
      !globalThis.DB.ST(board).noimg.includes("ghost.png"), globalThis.DB.ST(board).noimg);

// state round-trip
const json = board._dom.getValue();
const fresh = mkBoard();
fresh._dom.setValue(json);
check("state round-trips through the widget",
      globalThis.DB.ST(fresh).msgs.length === globalThis.DB.ST(board).msgs.length
      && globalThis.DB.ST(fresh).src === "9", globalThis.DB.ST(fresh).src);
for (const junk of ["", "{", "[]", '{"msgs":"nope"}']) {
  const b = mkBoard();
  b._dom.setValue(junk);
  check("junk state " + JSON.stringify(junk).padEnd(16) + " degrades to empty",
        globalThis.DB.ST(b).msgs.length === 0);
}

// two chat nodes and no pick -> a note, not a guess
const other = { ...chat, id: 11, _kbState: chat._kbState };
app.graph._nodes = [chat, other, mkBoard()];
const lonely = mkBoard();
app.graph._nodes.push(lonely);
lonely.widgets.find((w) => w.name === "⟳ Update History").callback();
check("two chats and no target -> a note", /pick one/.test(lonely._dbNote || ""), lonely._dbNote);
check("...and nothing was pulled", globalThis.DB.ST(lonely).msgs.length === 0);

// ── the live bug: a board's snapshot must protect its files ──────────────────────────────────
await (async () => {
  app.graph._nodes = [];
  const chat2 = { id: 20, comfyClass: "LocalLLMChatGGUF", properties: {}, widgets: [],
    inputs: [{ name: "persona_1", link: 1 }], setDirtyCanvas() {},
    getInputNode: () => ({ title: "x", constructor: { title: "y" }, widgets: [] }),
    _kbState: { v: 1, user: "", nonce: 0, approved: false, persona: 1, turn: null, att: [],
      history: [
        { role: "assistant", content: "one", persona: "Mia",
          att: [{ name: "kb_aaa.png", subfolder: "kinburg_chat", type: "input" }] },
        { role: "assistant", content: "two", persona: "Mia",
          att: [{ name: "kb_bbb.png", subfolder: "kinburg_chat", type: "input" }] },
      ] } };
  const bd = mkBoard();
  app.graph._nodes = [chat2, bd];
  bd.widgets.find((w) => w.name === "⟳ Update History").callback();
  check("the board snapshotted both pictures", globalThis.DB.ST(bd).msgs.length === 2);
  const held = globalThis.DB.heldRefs();
  check("heldRefs reports what a board still names",
        held.includes("kb_aaa.png") && held.includes("kb_bbb.png"), held.join());

  // now take one picture out of the CHAT — the file must survive, because the board still shows it
  UPLOADS.length = 0;
  dropAtt(chat2, 1, 0);
  await tick();
  check("removing a picture the board still holds deletes NOTHING",
        UPLOADS.length === 0, UPLOADS.map((u) => u[0]).join());

  // with no board around, the same removal does delete it
  app.graph._nodes = [chat2];
  instances.delete(bd);
  setHistory(chat2, [{ role: "assistant", content: "one", persona: "Mia",
                       att: [{ name: "kb_ccc.png", subfolder: "kinburg_chat", type: "input" }] }]);
  UPLOADS.length = 0;
  dropAtt(chat2, 0, 0);
  await tick();
  check("...while an unheld picture is still cleaned up",
        UPLOADS.at(-1)?.[0] === "/kinburg/chat/discard", UPLOADS.at(-1)?.[0]);
  check("...and it is the right file",
        JSON.parse(UPLOADS.at(-1)[1].body).refs[0].name === "kb_ccc.png");
  instances.add(bd);
  app.graph._nodes = [];
})();

// ── a stale snapshot is visible before the run ───────────────────────────────────────────────
await (async () => {
  app.graph._nodes = [];
  const bd = mkBoard();
  const st = globalThis.DB.ST(bd);
  st.msgs = [
    { r: "a", p: "Mia", t: "one", img: [{ name: "kb_x.png", subfolder: "kinburg_chat", type: "input" }] },
    { r: "a", p: "Mia", t: "two", img: [{ name: "kb_y.png", subfolder: "kinburg_chat", type: "input" }] },
  ];
  globalThis.DB.render(bd);
  const cells = [];
  (function walk(el) {
    for (const c of el.children || []) {
      if (c.classList?.contains("kb-db-pic")) cells.push(c);
      walk(c);
    }
  })(bd._dbView);
  check("both pictures drew a cell", cells.length === 2, cells.length);
  check("a healthy picture is marked as boundary K1", !cells[0].classList.contains("gone"));

  // the browser reports the second thumbnail 404s
  const img = cells[1].children.find((c) => c.tagName === "IMG");
  img.fire("error");
  check("a 404 thumbnail marks the picture gone", cells[1].classList.contains("gone"));
  check("...and is remembered on the node", globalThis.DB.gone(bd).has("kb_y.png"));
  const notes = [];
  (function walk(el) {
    for (const c of el.children || []) {
      if (c.classList?.contains("kb-db-note")) notes.push(c.textContent);
      walk(c);
    }
  })(bd._dbShots);
  check("...and the shot panel warns before you run",
        notes.some((t) => /no longer on disk/.test(t)), notes.join(" | "));
  check("a redraw keeps the mark", (globalThis.DB.render(bd), globalThis.DB.gone(bd).has("kb_y.png")));
  app.graph._nodes = [];
})();

console.log("\\n" + (fails.length ? "FAILED: " + fails.join(", ") : "ALL PASS"));
process.exit(fails.length ? 1 : 0);
`;

fs.writeFileSync(OUT, STUBS + strip(WEB + "chat_llm.js") + "\n{\n" + strip(WEB + "dream_board.js")
  + EXPOSE + "\n}\n" + TESTS, "utf8");
console.log("wrote " + path.basename(OUT));
