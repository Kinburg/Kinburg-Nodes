import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STUBS } from "./stubs.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "..", "web") + path.sep;
const OUT = path.join(HERE, "run_chat.mjs");
// Assemble a runnable harness: chat_llm.js minus its imports, with DOM/app stubs in front and
// the assertions behind. Written out as one .mjs so `node` can execute it.

const SRC = WEB + "chat_llm.js";

const SEND = WEB + "chat_send.js";

// Both files, imports stripped and concatenated: chat_send.js imports from chat_llm.js, so once
// they share a scope the export/import pair simply falls away. `export` keywords go too.
const strip = (p) => fs.readFileSync(p, "utf8")
  .split("\n")
  .filter((l) => !/^import\s/.test(l))
  .join("\n")
  .replace(/^export\s+(function|const|async function)/gm, "$1");

// chat_send.js goes in a block: as real modules the two files each have their own `CLASS`, and
// concatenating them would otherwise be a redeclaration. Its extension object still escapes,
// through the registerExtension stub.
const body = strip(SRC) + "\n{\n" + strip(SEND) + "\n}\n";

const TESTS = `
// ── assertions ──────────────────────────────────────────────────────────────────────────────
const fails = [];
const check = (label, cond, extra) => {
  console.log((cond ? "  ok   " : "  FAIL ") + label + (extra !== undefined ? "  " + extra : ""));
  if (!cond) fails.push(label);
};

const upNode = {
  title: "Character", constructor: { title: "Local LLM Settings (GGUF)" },
  widgets: [{ name: "system_prompt", value: "you are x" }, { name: "model", value: "m.gguf" }],
};

function mkNode() {
  const n = {
    id: 1, properties: {}, setDirtyCanvas() {},
    widgets: [{ name: "unload_on_approve", value: true }],
    inputs: [{ name: "persona_1", link: 7 }, { name: "persona_2", link: null },
             { name: "persona_3", link: null }, { name: "persona_4", link: null },
             { name: "persona_5", link: null }, { name: "persona_6", link: null }],
    getInputNode: (s) => (s === 0 ? upNode : null),
    _kbView: new El("div"), _kbJump: new El("button"), _kbTray: new El("div"),
    _kbClip: new El("button"),
    _kbBar: new El("div"), _kbStat: new El("div"), _kbInput: new El("textarea"),
  };
  clearPending(n);
  ST(n);
  n._kbEditIdx = -1; n._kbStick = true; n._kbBarH = 0; n._kbStatH = 0; n._kbTrayH = 0;
  n._kbAttErr = "";
  return n;
}
const A = { name: "a.png", subfolder: "kinburg_chat", type: "input" };
const B = { name: "b.png", subfolder: "kinburg_chat", type: "input" };

// A node put through the real setup(), so the widgets and the DOM it builds are the shipping ones.
function mkFullNode() {
  const n = mkNode();
  n.size = [400, 600];
  n.setSize = (s) => { n.size = s; };
  n.addWidget = (type, name, value, callback, options) => {
    const w = { type, name, value, callback, options: options || {} , ...(options || {}) };
    n.widgets.push(w);
    return w;
  };
  n.addDOMWidget = (name, t, element, opts) => {
    const w = { name, type: t, element, ...(opts || {}) };
    n.widgets.push(w);
    return w;
  };
  setup(n);
  return n;
}
const widgetOf = (n, name) => (n.widgets || []).find((w) => w.name === name);

// PCOUNT
check("PCOUNT is 6", PCOUNT === 6, PCOUNT);
const six = mkNode();
six.inputs.forEach((i) => { i.link = 7; });
check("all six persona inputs are seen as connected",
      connected(six).length === 6, connected(six).length);
check("getFold caps 'summarised by' at 6", getFold({ properties: { _kbFold: { by: 6 } } }).by === 6);
check("getPMeta returns six rows", getPMeta(six).length === 6, getPMeta(six).length);

// sending still works end to end
QUEUED.length = 0;
const free = mkNode();
free._kbInput.value = "hello";
doSend(free);
check("Send queues", QUEUED.length === 1, QUEUED.length);
check("the turn is staged", ST(free).user === "hello" && ST(free).turn.mode === "turn");
check("node goes busy", busy(free));

// ↻ replays through doSend
QUEUED.length = 0;
const rep = mkNode();
setHistory(rep, [{ role: "user", content: "q" }, { role: "assistant", content: "a", persona: "Character" }]);
rerunAt(rep, 0);
check("↻ replays and slices back to the turn",
      QUEUED.length === 1 && getHistory(rep).length === 0, getHistory(rep).length);

// ⤵ Archive
QUEUED.length = 0;
const fold = mkNode();
setHistory(fold, Array.from({ length: 40 }, (_, i) =>
  ({ role: i % 2 ? "assistant" : "user", content: "m" + i, persona: "Character" })));
doFold(fold);
check("⤵ Archive queues", QUEUED.length === 1, QUEUED.length);
check("archive button enabled", archiveBtn(mkNode(), { upto: 4, count: 4 }, false).disabled === false);

// the per-message actions are all live again
const findAct = (el, label) => {
  for (const c of el.children || []) {
    if (c.textContent === label) return c;
    const hit = findAct(c, label);
    if (hit) return hit;
  }
  return null;
};
const bub = makeBubble(mkNode(), { role: "assistant", content: "a", persona: "Character" }, 0, null);
for (const act of ["↻", "✎", "🗑"]) {
  check(act + " enabled", findAct(bub, act)?.disabled === false);
}

// a full redraw must not throw with six personas + the pill present
try { render(six); renderPersonaBar(six); renderStats(six); check("render/bar/stats survive 6 personas", true); }
catch (e) { check("render/bar/stats survive 6 personas", false, e.message); }

// ── attachments ─────────────────────────────────────────────────────────────────────────────
await (async () => {
  // upload -> tray
  UPLOADS.length = 0;
  const n = mkNode();
  await addFiles(n, [mkFile("shot.png"), mkFile("wide.png")]);
  check("both files were uploaded", UPLOADS.length === 2, UPLOADS.length);
  check("upload goes to /upload/image", UPLOADS[0][0] === "/upload/image", UPLOADS[0][0]);
  check("upload is a POST", UPLOADS[0][1]?.method === "POST");
  const form = UPLOADS[0][1]?.body?.d || [];
  check("upload targets input/kinburg_chat",
        form.some(([k, v]) => k === "subfolder" && v === ATT_DIR)
        && form.some(([k, v]) => k === "type" && v === "input"), JSON.stringify(form.map(x => [x[0], x[1]])));
  check("tray holds the names the server returned",
        getAtt(n).map((a) => a.name).join(",") === "up1.png,up2.png",
        getAtt(n).map((a) => a.name).join(","));
  check("tray row is visible", !n._kbTray.classList.contains("kb-lc-hide"));
  check("tray reserves height", n._kbTrayH > 0, n._kbTrayH);

  // a failed upload is reported, not swallowed
  UPLOAD_FAILS = true;
  const bad = mkNode();
  await addFiles(bad, [mkFile("nope.png")]);
  UPLOAD_FAILS = false;
  check("a failed upload adds nothing to the tray", getAtt(bad).length === 0);
  check("a failed upload says so", /couldn't attach/.test(bad._kbAttErr || ""), bad._kbAttErr);

  // an empty tray costs no height
  const empty = mkNode();
  renderTray(empty);
  check("empty tray is hidden", empty._kbTray.classList.contains("kb-lc-hide"));
  check("empty tray reserves nothing", empty._kbTrayH === 0, empty._kbTrayH);

  // ✕ takes off exactly one
  const two = mkNode();
  setAtt(two, [A, B]);
  renderTray(two);
  findAct(two._kbTray, "✕").fire("click");
  check("✕ removes just that one", getAtt(two).map((a) => a.name).join(",") === "b.png",
        getAtt(two).map((a) => a.name).join(","));
  check("✕ is disabled mid-turn", (() => {
    const t = mkNode(); setAtt(t, [A]); t._kbMode = "turn"; renderTray(t);
    return findAct(t._kbTray, "✕").disabled === true;
  })());
  check("📎 is disabled mid-turn", (() => {
    const t = mkNode(); t._kbMode = "turn"; renderTray(t);
    return t._kbClip.disabled === true;
  })());
})();

// a picture with no words is still a turn
QUEUED.length = 0;
const only = mkNode();
setAtt(only, [A]);
check("planTurn calls an image-only send a turn", planTurn(only).mode === "turn",
      planTurn(only).mode);
doSend(only);
check("image-only Send queues", QUEUED.length === 1, QUEUED.length);
check("the tray is NOT cleared on send (the backend still has to read it)",
      getAtt(only).length === 1, getAtt(only).length);
check("no empty pending bubble is drawn", !only._kbPending);

// the tray moves onto the message when the reply lands — through the REAL onExecuted
const NodeType = function () {};
NodeType.prototype = {};
await extBy("Kinburg.LocalLLMChat").beforeRegisterNodeDef(NodeType, { name: "LocalLLMChatGGUF" });
const reply = (n, text) => NodeType.prototype.onExecuted.call(n, {
  kinburg_chatllm: [JSON.stringify({ reply: text, thoughts: "", stats: {} })],
});

const landed = mkNode();
setAtt(landed, [A, B]);
landed._kbMode = "turn";
landed._kbPending = "look at this";
ST(landed).turn = { mode: "turn", persona: "Character" };
reply(landed, "I see a dress");
const hl = getHistory(landed);
check("the user message got the pictures", hl[0]?.att?.length === 2,
      JSON.stringify(hl[0]?.att?.map((a) => a.name)));
check("the reply did not", !hl[1]?.att);
check("the tray is emptied once the turn lands", getAtt(landed).length === 0);
check("the tray row collapses again", landed._kbTray.classList.contains("kb-lc-hide"));

// a picture with no words still produces a user message to hang it on
const wordless = mkNode();
setAtt(wordless, [A]);
wordless._kbMode = "turn";
wordless._kbPending = "";
ST(wordless).turn = { mode: "turn", persona: "Character" };
reply(wordless, "nice");
check("an image-only send still creates the user message",
      getHistory(wordless)[0]?.role === "user" && getHistory(wordless)[0]?.att?.length === 1,
      JSON.stringify(getHistory(wordless)[0]));

// a turn that never lands leaves the pictures staged for another go
const lost = mkNode();
setAtt(lost, [A]);
lost._kbMode = "turn";
clearPending(lost);
check("a failed turn keeps its pictures in the tray", getAtt(lost).length === 1);

// state round-trip
const rt = mkNode();
setAtt(rt, [A]);
const json = JSON.parse(stateJSON(rt));
check("att is serialized into chat_state", json.att?.length === 1, JSON.stringify(json.att));
const rt2 = mkNode();
loadState(rt2, JSON.stringify(json));
check("att survives a reload", getAtt(rt2)[0]?.name === "a.png", getAtt(rt2)[0]?.name);
loadState(rt2, JSON.stringify({ history: [] }));
check("a state with no att degrades to an empty tray", getAtt(rt2).length === 0);

// ↻ puts the picture back in front of the model
QUEUED.length = 0;
const again = mkNode();
setHistory(again, [{ role: "user", content: "look", att: [A] },
                   { role: "assistant", content: "nice", persona: "Character" }]);
rerunAt(again, 1);
check("↻ restages the turn's picture", getAtt(again)[0]?.name === "a.png",
      getAtt(again).map((a) => a.name).join(","));
check("↻ still queues", QUEUED.length === 1, QUEUED.length);

// bubbles keep the picture forever
const withPic = makeBubble(mkNode(), { role: "user", content: "look", att: [A] }, 0, null);
const shown = (function find(el) {
  for (const c of el.children || []) {
    if (c.tagName === "IMG") return c;
    const hit = find(c);
    if (hit) return hit;
  }
  return null;
})(withPic);
check("the bubble renders the picture", !!shown);
check("...from ComfyUI's /view", /\\/view\\?filename=a\\.png/.test(shown?.src || ""), shown?.src);
check("...pointing at the right subfolder+type",
      /subfolder=kinburg_chat/.test(shown?.src || "") && /type=input/.test(shown?.src || ""));

// ── Send Image to Chat: the sendToChat door ─────────────────────────────────────────────────
const P1 = { name: "p1.png", subfolder: "kinburg_chat", type: "input", ctx: false };
const P2 = { name: "p2.png", subfolder: "kinburg_chat", type: "input" };
const asChat = (n) => { n.comfyClass = "LocalLLMChatGGUF"; return n; };
const soleChat = (n) => { app.graph._nodes = n ? [n] : []; return n; };

// no chat node at all
soleChat(null);
check("with no chat node it refuses", sendToChat({ refs: [P2], as: "me (user)" }).ok === false);

// "me (user)" -> the tray
const c1 = soleChat(asChat(mkNode()));
let res = sendToChat({ refs: [P2], as: "me (user)" });
check("me (user) lands in the tray", res.ok && getAtt(c1)[0]?.name === "p2.png", res.msg);
res = sendToChat({ refs: [P2], as: "me (user)" });
check("the same file twice is not stacked", getAtt(c1).length === 1, getAtt(c1).length);
check("...and it says so", /already/.test(res.msg), res.msg);

// a persona -> its last bubble
const c2 = soleChat(asChat(mkNode()));
setHistory(c2, [
  { role: "user", content: "hi" },
  { role: "assistant", content: "old one", persona: "Character" },
  { role: "user", content: "and?" },
  { role: "assistant", content: "[фото: селфи]", persona: "Character" },
]);
res = sendToChat({ refs: [P1], as: "persona 1" });
const h2 = getHistory(c2);
check("a persona picture hangs on its LAST bubble",
      res.ok && h2[3].att?.[0]?.name === "p1.png" && !h2[1].att, res.msg);
check("no new message was invented", h2.length === 4, h2.length);
check("the tray was left alone", getAtt(c2).length === 0);
check("the hidden flag survives the trip", h2[3].att[0].ctx === false);

// "the active persona" resolves through the chat's own selection
const c3 = soleChat(asChat(mkNode()));
setHistory(c3, [{ role: "assistant", content: "mine", persona: "Character" }]);
res = sendToChat({ refs: [P2], as: "the active persona" });
check("'the active persona' resolves", res.ok && getHistory(c3)[0].att?.length === 1, res.msg);

// a persona that has not spoken yet gets a bubble to hang it on
const c4 = soleChat(asChat(mkNode()));
res = sendToChat({ refs: [P1], as: "persona 1" });
const h4 = getHistory(c4);
check("a silent persona gets a bubble", res.ok && h4.length === 1 && h4[0].role === "assistant",
      JSON.stringify(h4));
check("...tagged with its name", h4[0].persona === "Character", h4[0].persona);

// an unwired persona is refused rather than guessed at
const c5 = soleChat(asChat(mkNode()));
check("an unwired persona is refused", sendToChat({ refs: [P2], as: "persona 4" }).ok === false,
      sendToChat({ refs: [P2], as: "persona 4" }).msg);

// mid-turn the chat is left alone
const c6 = soleChat(asChat(mkNode()));
c6._kbMode = "turn";
res = sendToChat({ refs: [P2], as: "me (user)" });
check("mid-turn it refuses", res.ok === false && getAtt(c6).length === 0, res.msg);

// several chat nodes need an explicit target
const a1 = asChat(mkNode()), a2 = asChat(mkNode());
a2.id = 2;
app.graph._nodes = [a1, a2];
check("chatNodes finds both", chatNodes().length === 2, chatNodes().length);
check("two chat nodes and no target -> refused",
      sendToChat({ refs: [P2], as: "me (user)" }).ok === false);
res = sendToChat({ refs: [P2], as: "me (user)", target: a2 });
check("an explicit target is honoured",
      res.ok && getAtt(a2).length === 1 && getAtt(a1).length === 0, res.msg);

// nothing to send
check("empty refs are refused", sendToChat({ refs: [], as: "me (user)", target: a1 }).ok === false);
check("junk refs are refused",
      sendToChat({ refs: [{}, null], as: "me (user)", target: a1 }).ok === false);
app.graph._nodes = [];

// ── 🗑 Clear also deletes the files ──────────────────────────────────────────────────────────
await (async () => {
  // allAtt sees both places a picture can be
  const n = mkNode();
  setAtt(n, [A]);
  setHistory(n, [{ role: "user", content: "x", att: [B] },
                 { role: "assistant", content: "y", persona: "Character" }]);
  check("allAtt collects tray + history",
        allAtt(n).map((a) => a.name).sort().join(",") === "a.png,b.png",
        allAtt(n).map((a) => a.name).join(","));
  check("allAtt on an empty chat is empty", allAtt(mkNode()).length === 0);

  // the real button, through the real setup()
  app.graph._nodes = [];
  const full = asChat(mkFullNode());
  app.graph._nodes = [full];
  setAtt(full, [A]);
  setHistory(full, [{ role: "user", content: "look", att: [B] }]);
  const clear = widgetOf(full, "🗑 Clear");
  check("the Clear button is there", !!clear);
  check("setup built the tray and the 📎", !!full._kbTray && !!full._kbClip);

  UPLOADS.length = 0;
  clear.callback();
  await tick();
  check("Clear empties the history", getHistory(full).length === 0);
  check("Clear empties the tray too", getAtt(full).length === 0);
  check("Clear asks the backend to delete", UPLOADS.at(-1)?.[0] === "/kinburg/chat/discard",
        UPLOADS.at(-1)?.[0]);
  const body = JSON.parse(UPLOADS.at(-1)[1].body);
  check("...both pictures, by bare filename",
        body.refs.map((r) => r.name).sort().join(",") === "a.png,b.png",
        JSON.stringify(body.refs));
  check("...and nothing but the name travels", Object.keys(body.refs[0]).join() === "name",
        Object.keys(body.refs[0]));

  // an empty chat asks for nothing
  UPLOADS.length = 0;
  widgetOf(full, "🗑 Clear").callback();
  await tick();
  check("clearing an empty chat posts nothing", UPLOADS.length === 0, UPLOADS.length);

  // a picture another chat still shows must survive
  const keeper = asChat(mkNode());
  keeper.id = 9;
  setHistory(keeper, [{ role: "assistant", content: "", persona: "Character", att: [A] }]);
  const goer = asChat(mkNode());
  goer.id = 8;
  app.graph._nodes = [keeper, goer];
  UPLOADS.length = 0;
  await discardUnused([A, B, B]);
  const kept = JSON.parse(UPLOADS.at(-1)[1].body).refs.map((r) => r.name);
  check("a file another chat still uses is spared", !kept.includes("a.png"), kept);
  check("...and the rest is deduped", kept.join(",") === "b.png", kept);
  app.graph._nodes = [];
})();

// ── removing a picture that was sent by mistake ─────────────────────────────────────────────
await (async () => {
  const n = asChat(mkNode());
  app.graph._nodes = [n];
  setHistory(n, [
    { role: "assistant", content: "here you go", persona: "Character", att: [A, B] },
    { role: "user", content: "thanks" },
  ]);
  UPLOADS.length = 0;
  dropAtt(n, 0, 0);
  await tick();
  const h = getHistory(n);
  check("✕ removes just that picture", h[0].att.length === 1 && h[0].att[0].name === "b.png",
        JSON.stringify(h[0].att));
  check("...and keeps the reply it hung on", h[0].content === "here you go" && h.length === 2);
  check("...and deletes its file",
        JSON.parse(UPLOADS.at(-1)[1].body).refs[0].name === "a.png",
        UPLOADS.at(-1)?.[1]?.body);

  // the last picture off a carrier bubble takes the bubble with it
  setHistory(n, [{ role: "assistant", content: "", persona: "Character", att: [A] }]);
  dropAtt(n, 0, 0);
  await tick();
  check("an empty carrier bubble goes too", getHistory(n).length === 0, getHistory(n).length);

  // but a bubble with words stays, picture or not
  setHistory(n, [{ role: "assistant", content: "words", persona: "Character", att: [A] }]);
  dropAtt(n, 0, 0);
  await tick();
  check("a bubble with words survives losing its picture",
        getHistory(n).length === 1 && !getHistory(n)[0].att, JSON.stringify(getHistory(n)));

  // 🗑 on the whole message discards what it carried
  setHistory(n, [{ role: "user", content: "look", att: [A, B] }]);
  UPLOADS.length = 0;
  deleteAt(n, 0);
  await tick();
  check("🗑 on a message deletes its pictures too",
        JSON.parse(UPLOADS.at(-1)[1].body).refs.map((r) => r.name).sort().join(",") === "a.png,b.png",
        UPLOADS.at(-1)?.[1]?.body);

  // a picture still shown elsewhere in the same chat is not deleted
  setHistory(n, [{ role: "user", content: "one", att: [A] },
                 { role: "assistant", content: "two", persona: "Character", att: [A] }]);
  UPLOADS.length = 0;
  dropAtt(n, 0, 0);
  await tick();
  check("a picture still shown further down is spared", UPLOADS.length === 0, UPLOADS.length);
  app.graph._nodes = [];
})();

// ── 📌 reads the widgets as they are NOW, not as they were at generation time ────────────────
await (async () => {
  const SendType = function () {};
  SendType.prototype = {};
  await extBy("Kinburg.ChatSendImage").beforeRegisterNodeDef(SendType, { name: "LocalLLMChatSendImage" });

  const sender = {
    id: 42, title: "Send Image to Chat", properties: {}, setDirtyCanvas() {},
    widgets: [{ name: "send_as", value: "persona 1" }, { name: "caption", value: "" },
              { name: "shot", value: "" }, { name: "note_in_context", value: false }],
    addWidget(type, name, value, callback, options) {
      const w = { type, name, value, callback, options: options || {} };
      this.widgets.push(w);
      return w;
    },
  };
  SendType.prototype.onNodeCreated.call(sender);
  const btn = sender.widgets.find((w) => w.name === "📌 Send to chat");
  check("the sender has its 📌", !!btn);

  const chat = asChat(mkNode());
  app.graph._nodes = [chat];
  setHistory(chat, [{ role: "assistant", content: "first", persona: "Character" }]);

  // the branch runs while "persona 1" is picked
  SendType.prototype.onExecuted.call(sender, {
    kinburg_chatsend: [JSON.stringify({
      refs: [{ name: "gen.png", subfolder: "kinburg_chat", type: "input" }],
      as: "persona 1", when: "on button press",
    })],
  });
  check("the reference was stashed", !!sender.properties._kbSend);
  check("nothing was sent yet", !getHistory(chat)[0].att);

  // ...and only THEN do you decide it should come from you
  sender.widgets.find((w) => w.name === "send_as").value = "me (user)";
  sender.widgets.find((w) => w.name === "caption").value = "  a red dress  ";
  btn.callback();
  check("📌 honours send_as as it is now, not at generation time",
        getAtt(chat).length === 1 && !getHistory(chat)[0].att,
        JSON.stringify({ tray: getAtt(chat).length, onMsg: !!getHistory(chat)[0].att }));
  check("...and picks up the caption typed afterwards",
        getAtt(chat)[0].caption === "a red dress", getAtt(chat)[0].caption);
  check("yours is never hidden from the model", getAtt(chat)[0].ctx === undefined,
        getAtt(chat)[0]);

  // switch back to a persona and the ctx rule flips with it
  setAtt(chat, []);
  sender.widgets.find((w) => w.name === "send_as").value = "persona 1";
  btn.callback();
  const put = getHistory(chat)[0].att?.[0];
  check("a persona picture goes on its bubble", !!put, JSON.stringify(getHistory(chat)[0]));
  check("...and is hidden from the model by default", put?.ctx === false, put);
  getHistory(chat)[0].att = [];
  sender.widgets.find((w) => w.name === "note_in_context").value = true;
  btn.callback();
  check("note_in_context, flipped after the run, is honoured",
        getHistory(chat)[0].att[0].ctx === undefined, getHistory(chat)[0].att[0]);

  // no run yet -> a useful nudge instead of a silent nothing
  const fresh = { id: 43, properties: {}, setDirtyCanvas() {}, widgets: [],
                  addWidget(t, name, v, cb, o) { const w = { type: t, name, value: v, callback: cb, options: o || {} }; this.widgets.push(w); return w; } };
  SendType.prototype.onNodeCreated.call(fresh);
  const b2 = fresh.widgets.find((w) => w.name === "📌 Send to chat");
  b2.callback();
  check("📌 before any run says so", /run this branch first/.test(b2.name), b2.name);
  app.graph._nodes = [];
})();

console.log("\\n" + (fails.length ? "FAILED: " + fails.join(", ") : "ALL PASS"));
process.exit(fails.length ? 1 : 0);
`;

fs.writeFileSync(OUT, STUBS + body + TESTS, "utf8");
console.log("wrote " + path.basename(OUT));
