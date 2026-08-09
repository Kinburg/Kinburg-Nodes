import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Local LLM Chat (GGUF) — a chat window on the node.
//   • Three zones inside one DOM widget: the scrolling message view, the persona chip bar, the
//     input box. The view is an absolutely-positioned layer inside a relative/overflow:hidden box
//     so its content never inflates the node's measured height.
//   • 📨 Send streams the reply live (over the "kinburg.chatllm" websocket event); reasoning
//     (<think>…) shows in an open "💭 thinking" block during generation, then collapses into a
//     "💭 reasoning" toggle with the answer as the main text.
//   • Every message has ⧉ copy · ✎ edit · ↻ resend-from-here · 🗑 delete. The conversation lives
//     in node._kbState.history; all of that is pure array surgery on it.
//   • Personas: one chip per connected persona_1..6 input, each a whole Local LLM Settings bundle
//     (own model / sampling / system prompt). Clicking a chip only SELECTS it — Send is the sole
//     trigger. Chip labels, trigger texts, per-persona retention and privacy live in
//     node.properties._kbPersonas; what actually reaches the model is worked out per request by
//     windowedOut() + privateOut() (mirrored in chat_node.py), never frozen into the history.
//   • Send with an empty box picks the mode: same persona as the last reply → continue that reply
//     where it was cut off; a different persona → a turn with no user message at all.
//   • Pictures: paste / drop / 📎 into the window and they queue in the tray above the input, go
//     out with the next message, and stay in its bubble for good. The MODEL sees them only on
//     that one turn — after it they are a text marker (chat_node._content_of). Nothing here is a
//     graph link, so none of it re-runs an image branch when you Send.

const CLASS = "LocalLLMChatGGUF";
const PCOUNT = 6;
const STICK_SLACK = 24;
const VIEW_MIN = 96, INPUT_H = 56, GAP = 6, BAR_MIN = 24, STAT_MIN = 14, TRAY_MIN = 46;

// Pasted / dropped pictures land here, under ComfyUI's own input dir, so /view serves them and
// the node opens the same file. input (not temp) on purpose: temp is wiped on restart and a chat
// saved in a workflow would come back with holes where its pictures were.
const ATT_DIR = "kinburg_chat";

const instances = new Set();
const busy = (node) => node._kbMode != null;

// ── state ───────────────────────────────────────────────────────────────────────────────────
// Everything the backend needs travels as ONE json string, and the chat window itself carries it
// (see the addDOMWidget call in setup). The Vue frontend draws a 24px row for EVERY entry in
// node.widgets — `type = "hidden"` and a zeroed computeSize/computeLayoutSize change nothing, as
// measured on 1.45.21 — so the six little carriers this node used to need showed up as 168px of
// dead grey space under the chat.
const ST_VERSION = 1;

function ST(node) {
  if (!node._kbState || typeof node._kbState !== "object") {
    node._kbState = { v: ST_VERSION, user: "", history: [], nonce: 0, approved: false,
                      persona: 1, turn: null, att: [] };
  }
  return node._kbState;
}
const stateJSON = (node) => JSON.stringify(ST(node));

function loadState(node, raw) {
  let s = null;
  try { s = (typeof raw === "string" && raw) ? JSON.parse(raw) : raw; } catch (e) { s = null; }
  if (!s || typeof s !== "object") return;
  const st = ST(node);
  st.user = typeof s.user === "string" ? s.user : "";
  st.history = Array.isArray(s.history) ? s.history : [];
  st.nonce = Number.isFinite(+s.nonce) ? +s.nonce : 0;
  st.approved = false;                 // never resume a workflow mid-Approve
  st.persona = Number.isFinite(+s.persona) ? +s.persona : 1;
  st.turn = (s.turn && typeof s.turn === "object") ? s.turn : null;
  st.att = Array.isArray(s.att) ? s.att : [];
}

// Workflows saved before chat_state existed carry eleven positional widget values:
// [chat, Send, Approve, Clear, unload_on_approve, user_message, history_json, nonce, approved,
//  active_persona, turn_json]. The first five line up with the new layout (so unload_on_approve
// restores itself); lift the rest into the single state.
function migrateLegacy(node, info) {
  const vals = info?.widgets_values;
  if (!Array.isArray(vals) || vals.length < 11) return false;
  let hist;
  try { hist = JSON.parse(vals[6] || "[]"); } catch (e) { return false; }
  if (!Array.isArray(hist)) return false;
  const st = ST(node);
  st.user = typeof vals[5] === "string" ? vals[5] : "";
  st.history = hist;
  st.nonce = Number(vals[7]) || 0;
  st.persona = Number(vals[9]) || 1;
  try { st.turn = vals[10] ? JSON.parse(vals[10]) : null; } catch (e) { st.turn = null; }
  return true;
}

function reorder(node, order) {
  const rank = (w) => { const i = order.indexOf(w.name); return i === -1 ? order.length : i; };
  node.widgets.sort((a, b) => rank(a) - rank(b));
}

function injectStyle() {
  if (document.getElementById("kb-llmchat-style")) return;
  const s = document.createElement("style");
  s.id = "kb-llmchat-style";
  s.textContent = `
  /* Never un-hide with style.display="" — an empty string DELETES the property and the element
     falls back to display:block, which breaks these flex/grid layouts. Toggle this class instead. */
  .kb-lc-hide{display:none !important;}
  .kb-lc-wrap{position:relative;display:flex;flex-direction:column;gap:6px;width:100%;height:100%;box-sizing:border-box;overflow:hidden;}
  /* min-height:0 is load-bearing: without it this flex item refuses to shrink and a wrapped chip
     row pushes the input box out of the clipped wrap. */
  .kb-lc-box{flex:1 1 auto;position:relative;min-height:0;overflow:hidden;}
  .kb-lc-view{position:absolute;inset:0;overflow-y:auto;display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;background:#181818;border:1px solid #2b2b2b;border-radius:6px;font:12px/1.45 -apple-system,Segoe UI,sans-serif;}
  .kb-lc-view .kb-lc-msg{position:relative;max-width:85%;min-width:104px;padding:5px 9px;border-radius:10px;white-space:pre-wrap;word-break:break-word;}
  .kb-lc-view .kb-lc-user{align-self:flex-end;background:#274b6d;color:#eaf2fb;border-bottom-right-radius:3px;}
  .kb-lc-view .kb-lc-asst{align-self:flex-start;background:#2c2c33;color:#e6e6ea;border-bottom-left-radius:3px;}
  .kb-lc-view .kb-lc-msg.kb-lc-muted{opacity:.55;}
  .kb-lc-view .kb-lc-msg.kb-lc-muted:hover{opacity:.9;}
  .kb-lc-view .kb-lc-role{display:block;font-size:10px;opacity:.6;margin-bottom:1px;}
  .kb-lc-view .kb-lc-empty{margin:auto;color:#6d6d75;font-style:italic;text-align:center;}
  .kb-lc-view .kb-lc-acts{position:absolute;top:2px;right:3px;display:flex;gap:1px;padding:1px;border-radius:5px;background:#00000099;opacity:0;pointer-events:none;transition:opacity .1s;}
  .kb-lc-view .kb-lc-msg:hover .kb-lc-acts,.kb-lc-view .kb-lc-msg:focus-within .kb-lc-acts{opacity:1;pointer-events:auto;}
  .kb-lc-act{border:0;background:transparent;color:inherit;cursor:pointer;font-size:11px;line-height:1;padding:2px 4px;border-radius:4px;opacity:.75;}
  .kb-lc-act:hover{opacity:1;background:rgba(255,255,255,.16);}
  .kb-lc-act[disabled]{opacity:.25;cursor:default;}
  .kb-lc-view .kb-lc-msg.editing{max-width:100%;width:100%;}
  .kb-lc-view .kb-lc-msg.editing .kb-lc-acts,.kb-lc-view .kb-lc-msg.editing .kb-lc-think{display:none;}
  /* Height is set in JS to fit the message (floor 100px, capped to the view) — see fitEditor. */
  .kb-lc-ta{display:block;width:100%;min-height:100px;box-sizing:border-box;resize:vertical;background:#12121a;color:#e6e6ea;border:1px solid #4a86c4;border-radius:5px;padding:4px 6px;font:12px/1.45 inherit;outline:none;}
  .kb-lc-edbtns{display:flex;justify-content:flex-end;gap:5px;margin-top:4px;}
  .kb-lc-edbtn{border:1px solid #4a4a52;background:#2c2c33;color:#d8d8de;border-radius:4px;font:11px/1.3 inherit;padding:3px 9px;cursor:pointer;}
  .kb-lc-edbtn:hover{background:#3a3a44;}
  .kb-lc-edbtn.primary{background:#2f5b86;border-color:#4a86c4;color:#eaf2fb;}
  .kb-lc-view .kb-lc-think{margin:1px 0 4px;font-size:11px;}
  .kb-lc-view .kb-lc-think summary{cursor:pointer;opacity:.7;user-select:none;}
  .kb-lc-view .kb-lc-think-body{margin-top:3px;padding:4px 7px;border-left:2px solid #4a4a52;color:#b9b9c0;white-space:pre-wrap;word-break:break-word;font-style:italic;}
  .kb-lc-jump{position:absolute;right:14px;bottom:8px;z-index:2;border:1px solid #4a4a52;background:#2c2c33ee;color:#d8d8de;border-radius:11px;font:10px/1.3 -apple-system,Segoe UI,sans-serif;padding:3px 9px;cursor:pointer;}
  .kb-lc-jump:hover{background:#3a3a44;}
  .kb-lc-jump.hot{border-color:#4a86c4;color:#eaf2fb;}
  .kb-lc-stat{flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:7px;padding:0 2px;font:10px/1.4 -apple-system,Segoe UI,sans-serif;color:#8a8a94;}
  .kb-lc-track{flex:0 0 84px;height:4px;border-radius:2px;background:#2b2b33;overflow:hidden;}
  .kb-lc-track i{display:block;height:100%;background:#4a86c4;}
  .kb-lc-track i.warm{background:#c9a227;}
  .kb-lc-track i.hot{background:#c4564a;}
  .kb-lc-cut{color:#d7a55a;}
  .kb-lc-arch{margin-left:auto;border:1px solid #4a4a52;background:#2c2c33;color:#d8d8de;border-radius:9px;font:10px/1.3 inherit;padding:2px 8px;cursor:pointer;}
  .kb-lc-arch:hover{background:#3a3a44;color:#fff;}
  .kb-lc-arch.due{border-color:#c9a227;color:#e6cf8a;}
  .kb-lc-arch[disabled]{opacity:.4;cursor:default;}
  .kb-lc-view .kb-lc-digest{align-self:stretch;max-width:100%;background:#22252b;border:1px dashed #4a5561;color:#c4ccd4;}
  .kb-lc-view .kb-lc-digest .kb-lc-role{opacity:.75;color:#8fb0c8;}
  .kb-lc-bar{flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:4px;padding:0 1px;}
  .kb-lc-chip{flex:0 0 auto;max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;border:1px solid #3a3a44;background:#26262e;color:#c8c8d0;border-radius:11px;font:11px/1.3 -apple-system,Segoe UI,sans-serif;padding:3px 9px;cursor:pointer;}
  .kb-lc-chip:hover{background:#32323c;color:#eee;}
  .kb-lc-chip.on{background:#2f5b86;border-color:#4a86c4;color:#eaf2fb;}
  .kb-lc-chip[disabled]{opacity:.4;cursor:default;}
  .kb-lc-inputwrap{position:relative;flex:0 0 auto;}
  /* padding-right leaves room for the 📎, which is overlaid rather than given a row of its own. */
  .kb-lc-input{display:block;width:100%;height:56px;resize:none;box-sizing:border-box;background:#181818;color:#e6e6e6;border:1px solid #2b2b2b;border-radius:6px;padding:6px 30px 6px 8px;font:12px/1.45 -apple-system,Segoe UI,sans-serif;outline:none;}
  .kb-lc-clip{position:absolute;right:5px;bottom:5px;z-index:2;border:1px solid #3a3a44;background:#26262eee;color:#c8c8d0;border-radius:5px;font:11px/1 -apple-system,Segoe UI,sans-serif;padding:4px 5px;cursor:pointer;}
  .kb-lc-clip:hover{background:#32323c;color:#eee;}
  .kb-lc-clip[disabled]{opacity:.4;cursor:default;}
  .kb-lc-tray{flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:4px;padding:0 1px;}
  .kb-lc-thumb{position:relative;flex:0 0 auto;width:44px;height:44px;border:1px solid #3a3a44;border-radius:5px;overflow:hidden;background:#101014;}
  .kb-lc-thumb img{display:block;width:100%;height:100%;object-fit:cover;}
  .kb-lc-thumbx{position:absolute;top:0;right:0;border:0;background:#000000b0;color:#e6e6ea;font:10px/1 sans-serif;padding:2px 3px;border-radius:0 0 0 4px;cursor:pointer;}
  .kb-lc-thumbx:hover{background:#c4564a;color:#fff;}
  .kb-lc-thumbx[disabled]{opacity:.35;cursor:default;}
  .kb-lc-atterr{flex:0 0 auto;color:#d98b80;font:10px/1.3 -apple-system,Segoe UI,sans-serif;}
  .kb-lc-view .kb-lc-att{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;}
  .kb-lc-attcell{position:relative;max-width:100%;}
  .kb-lc-attimg{display:block;max-width:100%;max-height:180px;border-radius:6px;cursor:zoom-in;}
  /* Same reveal-on-hover as the message actions, so a picture is clean until you reach for it. */
  .kb-lc-attx{position:absolute;top:3px;right:3px;border:0;border-radius:4px;background:#000000b0;color:#e6e6ea;font:10px/1 sans-serif;padding:3px 4px;cursor:pointer;opacity:0;pointer-events:none;transition:opacity .1s;}
  .kb-lc-view .kb-lc-msg:hover .kb-lc-attx{opacity:1;pointer-events:auto;}
  .kb-lc-attx:hover{background:#c4564a;color:#fff;}
  .kb-lc-attx[disabled]{opacity:.25;cursor:default;}
  .kb-lc-wrap.kb-lc-drag .kb-lc-box{outline:2px dashed #4a86c4;outline-offset:-2px;}
  `;
  document.head.appendChild(s);
}

// ── attachments ─────────────────────────────────────────────────────────────────────────────
// A picture is carried as a ComfyUI file ref — {name, subfolder, type} — and never as a graph
// link: 📨 Send re-runs everything upstream of this node, so an IMAGE input would drag its whole
// generation branch along on every message. `st.att` is the tray staged for the next send; on the
// way back onExecuted stamps it onto the user message, where it stays for good.
//
// `caption` and `shot` are part of the ref from day one even though nothing fills them yet: the
// caption is what the model reads once the pixels are gone (chat_node._content_of) and `shot`
// labels a keyframe, so a chat can be read back as a storyboard. Both are the sender node's job.

function getAtt(node) {
  const a = ST(node).att;
  return Array.isArray(a) ? a : (ST(node).att = []);
}
function setAtt(node, arr) {
  ST(node).att = Array.isArray(arr) ? arr : [];
}

function attUrl(a) {
  const q = "/view?filename=" + encodeURIComponent(a?.name || "")
    + "&subfolder=" + encodeURIComponent(a?.subfolder || "")
    + "&type=" + encodeURIComponent(a?.type || "input");
  return api.apiURL ? api.apiURL(q) : q;
}

// ComfyUI's own upload endpoint, so the file lands where /view can serve it and the node can open
// it. It renames on collision and hands back the name it actually used — always trust the reply.
async function uploadImage(file) {
  const body = new FormData();
  body.append("image", file, file.name || "pasted.png");
  body.append("subfolder", ATT_DIR);
  body.append("type", "input");
  const r = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!r || r.status !== 200) throw new Error("upload returned " + (r ? r.status : "nothing"));
  const d = await r.json();
  if (!d?.name) throw new Error("upload returned no filename");
  return { name: d.name, subfolder: d.subfolder ?? ATT_DIR, type: d.type || "input" };
}

async function addFiles(node, files) {
  if (busy(node)) return;
  node._kbAttErr = "";
  for (const f of files) {
    try {
      getAtt(node).push(await uploadImage(f));
    } catch (e) {
      console.error("[Kinburg chat] attach failed:", e);
      node._kbAttErr = "couldn't attach " + (f.name || "the image") + " — see the console";
    }
    renderTray(node);
    node.setDirtyCanvas?.(true, true);
  }
}

const imagesIn = (list) => [...(list || [])].filter((f) => (f.type || "").startsWith("image/"));

// Every picture this chat is holding: staged in the tray, or hanging on a message.
function allAtt(node) {
  const out = [...getAtt(node)];
  for (const m of getHistory(node)) {
    if (m && Array.isArray(m.att)) out.push(...m.att);
  }
  return out.filter((a) => a && a.name);
}

// Anything OUTSIDE the chats that also holds attachment references and would be broken by deleting
// their files. Dream Board is one: it keeps a snapshot of the conversation, so a picture removed
// from the chat after that snapshot was taken is still named in its shot list. Registered rather
// than hard-coded, so a chat knows nothing about who else reads its pictures.
const refHolders = new Set();
export function registerRefHolder(fn) {
  if (typeof fn === "function") refHolders.add(fn);
}

// Delete the files behind `refs` — but only the ones nothing in this graph is still showing.
// Content-addressed names mean the same picture in two places IS one file, so removing one copy
// must not blank the other. CALL THIS AFTER the removal: it reads the live state to decide what is
// still wanted. A copy open in some other workflow is beyond what we can see from here — and
// re-running the branch that made it writes the very same file back.
async function discardUnused(refs) {
  const inUse = new Set();
  for (const n of chatNodes()) for (const a of allAtt(n)) inUse.add(a.name);
  for (const fn of refHolders) {
    try { for (const nm of (fn() || [])) inUse.add(nm); } catch (e) { /* a holder must never block a delete path */ }
  }
  const doomed = [];
  const seen = new Set();
  for (const a of refs || []) {
    if (!a?.name || inUse.has(a.name) || seen.has(a.name)) continue;
    seen.add(a.name);
    doomed.push({ name: a.name });
  }
  if (!doomed.length) return;
  try {
    await api.fetchApi("/kinburg/chat/discard", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refs: doomed }),
    });
  } catch (e) {
    // Losing the files is untidy, not broken — the chat is already cleared either way.
    console.warn("[Kinburg chat] could not delete attachment files:", e);
  }
}

// ── the door Send Image to Chat comes in through ────────────────────────────────────────────
// Exported rather than reached for: web/chat_send.js knows only these two functions, so the chat's
// state stays private and the sender node never has to understand history surgery.

export function chatNodes() {
  return (app.graph?._nodes || []).filter((n) => (n.comfyClass || n.type) === CLASS);
}

// A snapshot of the conversation for Dream Board 🎬, in the compact shape its board_state stores:
// role, speaker, text, and any pictures as plain file refs. Digests are left out — a brief is not
// something that happened. `private` names the personas whose whole job is writing image prompts
// (the "camera" pattern), so the board can leave them out of the story by default.
export function chatSnapshot(node) {
  const msgs = getHistory(node).filter((m) => m && (m.role === "user" || m.role === "assistant"))
    .map((m) => {
      const o = { r: m.role === "user" ? "u" : "a", p: String(m.persona || ""),
                  t: String(m.content || "") };
      const att = (m.att || []).filter((a) => a && a.name).map((a) => {
        const r = { name: a.name, subfolder: a.subfolder || "", type: a.type || "input" };
        if (a.caption) r.caption = a.caption;
        if (a.shot) r.shot = a.shot;
        return r;                       // `ctx` is the chat's model-visibility flag; irrelevant here
      });
      if (att.length) o.img = att;
      return o;
    });
  return { msgs, private: privateSet(node), title: String(node.title || "") };
}

// Which persona slot a "send_as" value names, or -1. Only a wired one counts.
function personaSlotFor(node, as) {
  if (as === "the active persona") {
    const i = getActive(node) - 1;
    return i >= 0 ? i : -1;
  }
  const m = /^persona (\d+)$/.exec(String(as || ""));
  if (!m) return -1;
  const i = Number(m[1]) - 1;
  return (i >= 0 && i < PCOUNT && isConnected(node, i)) ? i : -1;
}

// Put pictures into a chat: either your tray, or a persona's last bubble. Idempotent by filename,
// which is what makes "every run" safe — a re-executed branch that produced the same picture pushes
// the same reference, and it lands nowhere twice.
export function sendToChat({ refs, as, target }) {
  const list = (Array.isArray(refs) ? refs : []).filter((r) => r && r.name);
  if (!list.length) return { ok: false, msg: "nothing to send" };

  const all = chatNodes();
  const chat = target || (all.length === 1 ? all[0] : null);
  if (!chat) {
    return { ok: false, msg: all.length ? "several chat nodes — pick one" : "no chat node here" };
  }
  if (busy(chat)) return { ok: false, msg: "the chat is mid-turn" };

  const put = (arr) => {
    const have = new Set(arr.map((a) => a?.name));
    let n = 0;
    for (const r of list) if (!have.has(r.name)) { arr.push(r); n++; }
    return n;
  };

  if (as === "me (user)") {
    const n = put(getAtt(chat));
    renderTray(chat);
    chat.setDirtyCanvas?.(true, true);
    return { ok: true, msg: n ? "📎 " + n + " in your tray" : "already attached" };
  }

  const slot = personaSlotFor(chat, as);
  if (slot < 0) return { ok: false, msg: "that persona isn't wired" };
  const who = personaLabel(chat, slot);

  // Hang it on the persona's most recent turn rather than inventing a message: the reply that
  // asked for the picture is almost always the one it belongs to, and reusing it means the
  // context gains nothing at all.
  const h = getHistory(chat);
  let at = -1;
  for (let i = h.length - 1; i >= 0; i--) {
    if (h[i]?.role === "assistant" && String(h[i].persona || "") === who) { at = i; break; }
  }
  if (at < 0) {
    h.push({ role: "assistant", content: "", persona: who });   // it hasn't spoken yet
    at = h.length - 1;
  }
  const msg = h[at];
  if (!Array.isArray(msg.att)) msg.att = [];
  const n = put(msg.att);
  setHistory(chat, h);
  render(chat);
  chat.setDirtyCanvas?.(true, true);
  return { ok: true, msg: n ? "🖼 " + n + " from " + who : "already attached" };
}

// ── history ─────────────────────────────────────────────────────────────────────────────────

function getHistory(node) {
  const h = ST(node).history;
  return Array.isArray(h) ? h : (ST(node).history = []);
}
function setHistory(node, arr) {
  ST(node).history = Array.isArray(arr) ? arr : [];
}

// ── personas ────────────────────────────────────────────────────────────────────────────────
// Each persona_N input carries a whole Local LLM Settings bundle, so the backend owns the model
// and the system prompt. Only the presentation bits — chip label, trigger message, "withhold from
// context" — live here, in node.properties, so they serialize with the workflow.

// keep: null = every turn stays in the context, 0 = none of them, N = the last N.
function normKeep(v, legacyMute) {
  if (v === "" || v === null || v === undefined) return legacyMute ? 0 : null;  // pre-keep meta
  const n = Math.trunc(Number(v));
  return Number.isFinite(n) && n >= 0 ? n : null;
}

function getPMeta(node) {
  const raw = node.properties?._kbPersonas;
  const out = [];
  for (let i = 0; i < PCOUNT; i++) {
    // node.properties is hand-editable in litegraph's Properties panel — assume nothing.
    const p = (Array.isArray(raw) && raw[i] && typeof raw[i] === "object") ? raw[i] : null;
    out.push({
      label: String(p?.label ?? ""), trigger: String(p?.trigger ?? ""),
      keep: normKeep(p?.keep, !!p?.mute), private: !!p?.private,
    });
  }
  return out;
}
function setPMeta(node, arr) {
  node.properties = node.properties || {};
  node.properties._kbPersonas = arr.map((p) => {
    const keep = normKeep(p?.keep, false);
    const o = { label: String(p?.label ?? ""), trigger: String(p?.trigger ?? "") };
    if (keep !== null) o.keep = keep;   // absent = keep everything, and drops the legacy `mute`
    if (p?.private) o.private = true;
    return o;
  });
}

// {personaLabel: keep} for the personas that actually have a window — sent to the backend and
// used here to dim. Only connected personas can appear in the history, so only they matter.
function keepMap(node) {
  const meta = getPMeta(node);
  const map = {};
  for (const i of connected(node)) {
    if (meta[i].keep !== null) map[personaLabel(node, i)] = meta[i].keep;
  }
  return map;
}

// ── archiving ───────────────────────────────────────────────────────────────────────────────
// One pass folds at most this many messages, so the summariser's own prompt can't itself overflow
// n_ctx on a very long chat. Press again to fold the next block — the brief is cumulative.
const FOLD_BATCH = 30;

function getFold(node) {
  const f = node.properties?._kbFold;
  const num = (v, d, lo, hi) => {
    const n = Math.trunc(Number(v));
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d;
  };
  return { at: num(f?.at, 70, 10, 100), keep: num(f?.keep, 8, 2, 200), by: num(f?.by, 0, 0, PCOUNT) };
}
function setFold(node, f) {
  node.properties = node.properties || {};
  node.properties._kbFold = { at: f.at, keep: f.keep, by: f.by };
}

const digestIndex = (h) => h.findIndex((m) => m && m.role === "digest");

// Which messages the next ⤵ would fold: everything older than the last `keep` real messages that
// isn't already archived, hidden by hand, aged out, or private. `upto` is exclusive.
function foldPlan(node) {
  const h = getHistory(node);
  const { keep } = getFold(node);
  const aged = windowedOut(h, keepMap(node));
  const priv = privateOut(h, privateSet(node), "");   // "" → nobody's own, so all private turns
  let seen = 0, end = h.length;
  for (let i = h.length - 1; i >= 0; i--) {
    if (!h[i] || h[i].role === "digest") continue;
    if (seen >= keep) break;
    seen++; end = i;
  }
  let count = 0, upto = 0;
  for (let i = 0; i < end; i++) {
    const m = h[i];
    if (!m || (m.role !== "user" && m.role !== "assistant") || !m.content) continue;
    if (m.ctx === false || m.fold || aged.has(i) || priv.has(i)) continue;
    if (count >= FOLD_BATCH) break;
    count++; upto = i + 1;
  }
  return { upto, count };
}

// Names of the personas that keep their turns to themselves.
function privateSet(node) {
  const meta = getPMeta(node);
  return connected(node).filter((i) => meta[i].private).map((i) => personaLabel(node, i));
}

// Indices a private persona withholds from whoever is speaking. MUST match _private_out in
// local_llm/chat_node.py. The speaker always sees its own turns — that's what lets it iterate.
function privateOut(h, names, speaker) {
  const out = new Set();
  if (!names || !names.length) return out;
  const set = new Set(names.map(String));
  h.forEach((m, i) => {
    if (!m || typeof m !== "object") return;
    const who = String(m.persona || "");
    if (who && who !== speaker && set.has(who)) out.add(i);
  });
  return out;
}

// Indices withheld by the retention windows. MUST match _windowed_out in local_llm/chat_node.py:
// a persona's turn is its reply plus the user message that prompted it, and only its `keep` most
// recent turns survive. Counting the persona's OWN turns is deliberate — see the python docstring.
function windowedOut(h, map) {
  const out = new Set();
  const turns = new Map();
  h.forEach((m, i) => {
    if (m && m.role === "assistant") {
      const who = String(m.persona || "");
      if (who) { if (!turns.has(who)) turns.set(who, []); turns.get(who).push(i); }
    }
  });
  for (const [who, idxs] of turns) {
    const n = map[who];
    if (!(typeof n === "number" && Number.isFinite(n) && n >= 0)) continue;
    for (const i of (n ? idxs.slice(0, -n) : idxs)) {
      out.add(i);
      const prev = h[i - 1];
      if (prev && prev.role === "user" && String(prev.persona || "") === who) out.add(i - 1);
    }
  }
  return out;
}

const slotOf = (node, i) => (node.inputs || []).findIndex((inp) => inp?.name === "persona_" + (i + 1));

function isConnected(node, i) {
  const s = slotOf(node, i);
  return s >= 0 && node.inputs[s].link != null;
}
function connected(node) {
  const a = [];
  for (let i = 0; i < PCOUNT; i++) if (isConnected(node, i)) a.push(i);
  return a;
}
function upstream(node, i) {
  const s = slotOf(node, i);
  if (s < 0 || node.inputs[s].link == null) return null;
  try { return node.getInputNode?.(s) || null; } catch (e) { return null; }
}
const upWidget = (node, i, name) => upstream(node, i)?.widgets?.find((w) => w.name === name)?.value;

// A Settings node still carrying its default title tells us nothing — four of them would all read
// "Local LLM Settings (GGUF)". Only a title the user actually changed is worth showing.
function upstreamTitle(node, i) {
  const up = upstream(node, i);
  if (!up) return "";
  const t = String(up.title || "");
  const def = up.constructor?.title;
  return (def && t === def) ? "" : t;
}
function personaLabel(node, i) {
  return getPMeta(node)[i].label || upstreamTitle(node, i) || ("Persona " + (i + 1));
}
const personaSystem = (node, i) => String(upWidget(node, i, "system_prompt") ?? "");

// Which model a config input resolves to, as far as the graph can tell client-side. Used only to
// warn that picking this chip will cost a model reload.
function configModel(node, i) {
  if (!upstream(node, i)) return null;
  const m = upWidget(node, i, "model"), mp = upWidget(node, i, "model_path");
  if (m == null && mp == null) return null;
  return String(m ?? "") + "|" + String(mp ?? "");
}
function needsReload(node, i) {
  const now = configModel(node, getActive(node) - 1);
  const next = configModel(node, i);
  return !!(now && next && now !== next);
}

// 1..4 — always a persona that is actually wired. 0 only when nothing is (a broken graph).
function getActive(node) {
  const v = Math.trunc(Number(ST(node).persona) || 0);
  if (v >= 1 && v <= PCOUNT && isConnected(node, v - 1)) return v;
  const on = connected(node);
  return on.length ? on[0] + 1 : 0;
}
function setActive(node, v) {
  ST(node).persona = (v >= 1 && v <= PCOUNT) ? v : 0;
}
function activeLabel(node) {
  const pi = getActive(node) - 1;
  return pi >= 0 ? personaLabel(node, pi) : "";
}
// Resolve a recorded persona name back to an index for ↻; "" means the turn had no persona.
function personaIndexByLabel(node, name) {
  if (!name) return -1;
  for (let i = 0; i < PCOUNT; i++) if (isConnected(node, i) && personaLabel(node, i) === name) return i;
  return getActive(node) - 1;
}

// ── scrolling ───────────────────────────────────────────────────────────────────────────────
// Auto-follow is opt-out by position: pin to the bottom only while the view IS at the bottom, so
// scrolling up mid-stream (to read or edit an older message) freezes the view instead of yanking.

function atBottom(node) {
  const s = node._kbView;
  if (!s) return true;
  return s.scrollHeight - s.scrollTop - s.clientHeight <= STICK_SLACK;
}
function showJump(node, visible, hot) {
  const j = node._kbJump;
  if (!j) return;
  j.classList.toggle("kb-lc-hide", !visible);
  j.classList.toggle("hot", !!(visible && hot));
}
function stick(node) {
  const s = node._kbView;
  if (!s) return;
  if (node._kbStick === false) showJump(node, true, true);
  else s.scrollTop = s.scrollHeight;
}
function toBottom(node) {
  const s = node._kbView;
  if (!s) return;
  node._kbStick = true;
  s.scrollTop = s.scrollHeight;
  showJump(node, false);
}

// ── bubbles ─────────────────────────────────────────────────────────────────────────────────

// Split raw model output into reasoning (<think>…</think>) and the answer. Handles an unclosed
// <think> mid-stream. (Models using a plain marker instead of tags stream as answer text; the
// final, authoritative split comes from the backend on completion.)
function parseThink(raw) {
  const open = raw.indexOf("<think>");
  if (open === -1) return { thinking: "", answer: raw };
  const before = raw.slice(0, open);
  const close = raw.indexOf("</think>", open);
  if (close === -1) return { thinking: raw.slice(open + 7), answer: before, inThink: true };
  return { thinking: raw.slice(open + 7, close), answer: before + raw.slice(close + 8) };
}

function actBtn(label, title, fn, disabled) {
  const b = document.createElement("button");
  b.className = "kb-lc-act";
  b.textContent = label;
  b.title = title;
  if (disabled) b.disabled = true;
  b.addEventListener("pointerdown", (e) => e.stopPropagation());
  b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); if (!b.disabled) fn(); });
  return b;
}

function copyBtn(getText) {
  const b = actBtn("⧉", "Copy message", () => {
    navigator.clipboard?.writeText(getText()).then(() => {
      b.textContent = "✓"; setTimeout(() => { b.textContent = "⧉"; }, 900);
    }).catch(() => {});
  });
  return b;
}

function reasoningEl(thoughts, open, label) {
  const d = document.createElement("details");
  d.className = "kb-lc-think";
  d.open = !!open;
  const sum = document.createElement("summary"); sum.textContent = label;
  const body = document.createElement("div"); body.className = "kb-lc-think-body"; body.textContent = thoughts;
  d.append(sum, body);
  return { details: d, body, summary: sum };
}

// Grow the editor to the message it holds — a two-line box for a twenty-line reply is useless.
// Floor 100px, capped so the editor still fits inside the scroll view.
function fitEditor(node, ta) {
  const cap = Math.max(140, (node._kbView?.clientHeight || 420) - 60);
  ta.style.height = "auto";
  ta.style.height = Math.min(cap, Math.max(100, (ta.scrollHeight || 0) + 4)) + "px";
}

function editorEl(node, i, content) {
  const wrap = document.createElement("div");
  const ta = document.createElement("textarea");
  ta.className = "kb-lc-ta";
  ta.value = node._kbEditDraft != null ? node._kbEditDraft : content;
  ta.addEventListener("pointerdown", (e) => e.stopPropagation());
  ta.addEventListener("wheel", (e) => e.stopPropagation());
  ta.addEventListener("input", () => { node._kbEditDraft = ta.value; fitEditor(node, ta); });
  ta.addEventListener("keydown", (e) => {
    e.stopPropagation();  // keep Delete / b / Ctrl+A out of ComfyUI's global hotkeys
    if (e.key === "Escape") { e.preventDefault(); cancelEdit(node); }
    else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveEdit(node, i, ta.value); }
  });

  const row = document.createElement("div");
  row.className = "kb-lc-edbtns";
  const cancel = document.createElement("button");
  cancel.className = "kb-lc-edbtn"; cancel.textContent = "Cancel"; cancel.title = "Esc";
  cancel.addEventListener("pointerdown", (e) => e.stopPropagation());
  cancel.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); cancelEdit(node); });
  const save = document.createElement("button");
  save.className = "kb-lc-edbtn primary"; save.textContent = "Save"; save.title = "Ctrl+Enter";
  save.addEventListener("pointerdown", (e) => e.stopPropagation());
  save.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); saveEdit(node, i, ta.value); });
  row.append(cancel, save);

  wrap.append(ta, row);
  // scrollHeight only means anything once the element is in the document.
  setTimeout(() => {
    try { fitEditor(node, ta); ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); } catch (e) {}
  }, 0);
  return wrap;
}

// i is the index in the history; i < 0 means the in-flight turn (no history entry to act on yet).
// `why` is null, "aged" (pushed out by the retention window) or "private" (its persona keeps its
// turns to itself and someone else is speaking).
function makeBubble(node, m, i, why) {
  const isUser = m.role === "user";
  const isDigest = m.role === "digest";
  const content = m.content || "";
  const byHand = m.ctx === false;
  const hidden = !isDigest && (byHand || !!why);

  const b = document.createElement("div");
  b.className = "kb-lc-msg " + (isDigest ? "kb-lc-digest" : (isUser ? "kb-lc-user" : "kb-lc-asst"))
    + (hidden ? " kb-lc-muted" : "");

  if (isDigest) {
    const head = document.createElement("span");
    head.className = "kb-lc-role";
    head.textContent = "🗂 archived summary";
    head.title = "Stands in for the dimmed messages above — the model reads this instead.\n"
      + "✎ to fix it by hand; 🗑 to drop it and put the originals back.";
    b.appendChild(head);
    const acts = document.createElement("div");
    acts.className = "kb-lc-acts";
    acts.appendChild(copyBtn(() => content));
    if (i >= 0) {
      const off = busy(node);
      acts.appendChild(actBtn("✎", "Edit the summary", () => startEdit(node, i), off));
      acts.appendChild(actBtn("🗑", "Drop the summary and un-archive its messages",
                              () => deleteAt(node, i), off));
    }
    b.appendChild(acts);
    if (i >= 0 && node._kbEditIdx === i) {
      b.classList.add("editing");
      b.appendChild(editorEl(node, i, content));
      return b;
    }
    const txt = document.createElement("span");
    txt.textContent = content;
    b.appendChild(txt);
    return b;
  }

  // Turns are always tagged with their persona, but naming them only helps once there is more
  // than one — a lone persona is just "the model".
  const named = connected(node).length > 1 && m.persona;
  const roleEl = document.createElement("span");
  roleEl.className = "kb-lc-role";
  roleEl.textContent = (hidden ? "🚫 " : "") + (isUser ? "You" : (named ? m.persona : "LLM"));
  if (byHand) roleEl.title = "Hidden by hand — 👁 puts it back in the context";
  else if (why === "aged") {
    const n = keepMap(node)[String(m.persona || "")];
    roleEl.title = "Aged out of " + (m.persona || "this persona") + "'s "
      + (n ? "last-" + n + "-turns window" : "context") + " — still here, but the model no longer sees it";
  } else if (why === "private") {
    roleEl.title = (m.persona || "This persona") + " keeps its turns to itself — "
      + (activeLabel(node) || "the active persona") + " won't see this one. Pick its own chip and it will.";
  } else if (why === "folded") {
    roleEl.title = "Archived — the model reads the 🗂 summary instead. Delete the summary to "
      + "bring this back.";
  }
  b.appendChild(roleEl);

  const acts = document.createElement("div");
  acts.className = "kb-lc-acts";
  acts.appendChild(copyBtn(() => content));
  if (i >= 0) {
    const off = busy(node);
    if (byHand) acts.appendChild(actBtn("👁", "Put this message back in the context", () => unhide(node, i), off));
    acts.appendChild(actBtn("✎", "Edit this message", () => startEdit(node, i), off));
    acts.appendChild(actBtn("↻", isUser ? "Resend from here (drops everything below)"
                                        : "Regenerate this reply (drops everything below)",
                            () => rerunAt(node, i), off));
    acts.appendChild(actBtn("🗑", "Delete this message", () => deleteAt(node, i), off));
  }
  b.appendChild(acts);

  if (i >= 0 && node._kbEditIdx === i) {
    b.classList.add("editing");
    b.appendChild(editorEl(node, i, content));
    return b;
  }
  if (!isUser && m.thoughts) b.appendChild(reasoningEl(m.thoughts, false, "💭 reasoning").details);
  const textEl = document.createElement("span");
  textEl.textContent = content;
  b.appendChild(textEl);
  // The picture stays in the bubble for good, even though the model stopped seeing it after that
  // turn — the chat is the record, the context is only what fits.
  if (Array.isArray(m.att) && m.att.length) b.appendChild(attRow(node, m, i));
  return b;
}

// The pictures on a message. Each gets its own ✕ — a picture sent by the wrong persona, or sent by
// mistake, has to be removable without binning the reply it is hanging on.
function attRow(node, m, i) {
  const row = document.createElement("div");
  row.className = "kb-lc-att";
  (m.att || []).forEach((a, k) => {
    if (!a || !a.name) return;
    const cell = document.createElement("div");
    cell.className = "kb-lc-attcell";
    const img = document.createElement("img");
    img.className = "kb-lc-attimg";
    img.src = attUrl(a);
    img.alt = a.caption || a.name;
    img.title = (a.caption ? a.caption + "\n" : "") + a.name
      + (a.shot ? "\nshot: " + a.shot : "") + "\nClick to open it full size.";
    img.addEventListener("pointerdown", (e) => e.stopPropagation());
    img.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation();
      window.open(attUrl(a), "_blank");
    });
    cell.appendChild(img);
    if (i >= 0) {
      const x = document.createElement("button");
      x.className = "kb-lc-attx";
      x.textContent = "✕";
      x.title = "Remove this picture from the chat (the file goes too, unless it is used elsewhere)";
      if (busy(node)) x.disabled = true;
      x.addEventListener("pointerdown", (e) => e.stopPropagation());
      x.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        if (!x.disabled) dropAtt(node, i, k);
      });
      cell.appendChild(x);
    }
    row.appendChild(cell);
  });
  return row;
}

// Take one picture off a message. A bubble left with neither text nor pictures was only ever there
// to carry them (Send Image to Chat makes one for a persona that hasn't spoken yet), so it goes too.
function dropAtt(node, i, k) {
  if (busy(node)) return;
  const h = getHistory(node);
  const m = h[i];
  if (!m || !Array.isArray(m.att) || !m.att[k]) return;
  const gone = m.att.splice(k, 1);
  if (!m.att.length) delete m.att;
  if (!m.att && !String(m.content || "").trim() && m.role === "assistant") h.splice(i, 1);
  setHistory(node, h);
  node._kbEditIdx = -1; node._kbEditDraft = null;
  render(node);
  renderStats(node);
  discardUnused(gone);        // after the removal, so "still in use" is answered honestly
}

// The in-progress assistant bubble whose text/reasoning we update as tokens stream in. `seed` is
// the already-written text a 'continue' turn is extending — the deltas land after it.
function makeStreamingBubble(node, label, seed) {
  const b = document.createElement("div");
  b.className = "kb-lc-msg kb-lc-asst";
  const roleEl = document.createElement("span");
  roleEl.className = "kb-lc-role"; roleEl.textContent = label;
  const think = reasoningEl("", true, "💭 thinking…");
  think.details.classList.add("kb-lc-hide");
  const answerEl = document.createElement("span");
  // Escape hatch: if a run dies somewhere the error listeners can't see, this unsticks the node.
  const acts = document.createElement("div");
  acts.className = "kb-lc-acts";
  acts.appendChild(actBtn("✕", "Give up on this turn (the reply is discarded)", () => {
    clearPending(node); render(node); renderPersonaBar(node);
  }));
  b.append(roleEl, acts, think.details, answerEl);
  return { bubble: b, refs: { thinkDetails: think.details, thinkBody: think.body, answerEl, seed: seed || "" } };
}

function updateStreaming(node) {
  const refs = node._kbStreamEls;
  if (!refs) return;
  const { thinking, answer } = parseThink(node._kbStream || "");
  refs.thinkDetails.classList.toggle("kb-lc-hide", !thinking);
  if (thinking) refs.thinkBody.textContent = thinking;
  const shown = refs.seed + answer;
  refs.answerEl.textContent = shown || (thinking ? "" : "…");
  stick(node);
}

function render(node) {
  const el = node._kbView;
  if (!el) return;
  // Preserve the reading position when the user has scrolled away (editing an old message must
  // not yank the view to the bottom); otherwise follow the newest content.
  const keep = node._kbStick === false ? el.scrollTop : null;

  node._kbStreamEls = null;
  el.innerHTML = "";
  const h = getHistory(node);
  // A 'continue' turn extends the last reply, so that reply is drawn as the live bubble instead.
  const contIdx = node._kbMode === "continue" ? h.length - 1 : -1;

  if (!h.length && !busy(node)) {
    const e = document.createElement("div");
    e.className = "kb-lc-empty";
    e.textContent = "(no messages yet — type below and Send)";
    el.appendChild(e);
  }
  // Derived every render, never stored — the 🚫 marks slide down as the conversation moves, and
  // switching chips re-answers "what is the persona I just picked about to see?".
  const aged = windowedOut(h, keepMap(node));
  const priv = privateOut(h, privateSet(node), activeLabel(node));
  h.forEach((m, i) => {
    if (i === contIdx || !m || typeof m !== "object") return;
    const why = m.fold ? "folded" : (aged.has(i) ? "aged" : (priv.has(i) ? "private" : null));
    el.appendChild(makeBubble(node, m, i, why));
  });

  if (busy(node)) {
    // Truthy, not != null: a picture sent with no words leaves _kbPending as "", and an empty
    // bubble reads as a glitch. The tray below still shows what is going out.
    if (node._kbPending) el.appendChild(makeBubble(node, { role: "user", content: node._kbPending }, -1));
    const who = (connected(node).length > 1 && activeLabel(node)) || "LLM";
    const seed = contIdx >= 0 ? String(h[contIdx]?.content || "") : "";
    const label = node._kbMode === "fold" ? "🗂 archiving…"
      : (contIdx >= 0 ? who + " · continuing" : who);
    const s = makeStreamingBubble(node, label, seed);
    node._kbStreamEls = s.refs;
    el.appendChild(s.bubble);
    updateStreaming(node);
  }
  if (keep != null) el.scrollTop = keep;
  else toBottom(node);
  renderStats(node);   // the ⤵ count follows the history, so keep it in step with every redraw
}

// ── persona chip bar ────────────────────────────────────────────────────────────────────────
// Kept separate from render() on purpose: redrawing the bar must never null _kbStreamEls.

// offsetHeight, NOT getBoundingClientRect: the DOM-widget layer is transform:scale()d by the
// canvas zoom, so a rect would make the node's min height breathe with the zoom level.
const rowHeight = (el, min) =>
  (el && !el.classList.contains("kb-lc-hide")) ? (el.offsetHeight || min) : 0;

function measureRows(node) {
  const b = rowHeight(node._kbBar, BAR_MIN);
  const s = rowHeight(node._kbStat, STAT_MIN);
  const t = rowHeight(node._kbTray, TRAY_MIN);
  if (b !== node._kbBarH || s !== node._kbStatH || t !== node._kbTrayH) {
    node._kbBarH = b; node._kbStatH = s; node._kbTrayH = t;
    node.setDirtyCanvas?.(true, true);
  }
}

// ── attachment tray ─────────────────────────────────────────────────────────────────────────
// Its own row rather than a strip inside the input, so it can collapse to nothing: with an empty
// tray this is display:none and costs zero height, exactly like the chip row and the meter.

function renderTray(node) {
  const el = node._kbTray;
  if (!el) return;
  const att = getAtt(node);
  el.innerHTML = "";
  if (node._kbClip) node._kbClip.disabled = busy(node);
  el.classList.toggle("kb-lc-hide", !att.length && !node._kbAttErr);
  if (!att.length && !node._kbAttErr) { measureRows(node); return; }

  att.forEach((a, i) => {
    const t = document.createElement("div");
    t.className = "kb-lc-thumb";
    const img = document.createElement("img");
    img.src = attUrl(a);
    img.alt = a.name || "";
    img.title = a.name + "\nGoes out with your next message.";
    t.appendChild(img);
    const x = document.createElement("button");
    x.className = "kb-lc-thumbx";
    x.textContent = "✕";
    x.title = "Take this one off";
    if (busy(node)) x.disabled = true;
    x.addEventListener("pointerdown", (e) => e.stopPropagation());
    x.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation();
      if (x.disabled) return;
      const cur = getAtt(node).slice();
      const gone = cur.splice(i, 1);
      setAtt(node, cur);
      renderTray(node);
      node.setDirtyCanvas?.(true, true);
      discardUnused(gone);      // unstaged and unwanted — don't leave the file behind
    });
    t.appendChild(x);
    el.appendChild(t);
  });

  if (node._kbAttErr) {
    const e = document.createElement("span");
    e.className = "kb-lc-atterr";
    e.textContent = "⚠ " + node._kbAttErr;
    el.appendChild(e);
  }
  measureRows(node);
}

// ── context meter ───────────────────────────────────────────────────────────────────────────
// Last turn's KV-cache fill, straight from the worker (the same numbers LLM Live Log shows).
// Kept in node.properties so it survives the tab switch that destroys the node.

function getStats(node) {
  const s = node.properties?._kbStats;
  return (s && typeof s === "object" && Number(s.n_ctx) > 0) ? s : null;
}
function setStats(node, s) {
  node.properties = node.properties || {};
  if (s && Number(s.n_ctx) > 0) node.properties._kbStats = s;
  else delete node.properties._kbStats;
}

const groupNum = (v) => String(Math.round(Number(v) || 0)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

function renderStats(node) {
  const el = node._kbStat;
  if (!el) return;
  const s = getStats(node);
  const plan = foldPlan(node);
  el.innerHTML = "";
  el.classList.toggle("kb-lc-hide", !s && !plan.count);
  if (!s && !plan.count) { measureRows(node); return; }
  if (!s) { el.appendChild(archiveBtn(node, plan, false)); measureRows(node); return; }

  const ctx = Number(s.n_ctx) || 0;
  const used = Math.max(0, Number(s.context_used) || 0);
  const pct = Math.max(0, Math.min(100, Math.round((used / ctx) * 100)));

  const track = document.createElement("div");
  track.className = "kb-lc-track";
  const fill = document.createElement("i");
  fill.className = pct >= 90 ? "hot" : (pct >= 75 ? "warm" : "");
  fill.style.width = pct + "%";
  track.appendChild(fill);

  const txt = document.createElement("span");
  txt.textContent = "ctx " + groupNum(used) + " / " + groupNum(ctx) + " · " + pct + "%"
    + (s.output_tokens ? " · " + groupNum(s.output_tokens) + " out" : "")
    + (s.seconds ? " · " + s.seconds + "s" : "");
  txt.title = "KV-cache fill after the last turn — prompt + reply, including the chat template";
  el.append(track, txt);

  if (s.finish_reason === "length") {
    const cut = document.createElement("span");
    cut.className = "kb-lc-cut";
    cut.textContent = "⚠ cut off — Send with an empty box to continue it";
    el.appendChild(cut);
  }
  if (plan.count) el.appendChild(archiveBtn(node, plan, pct >= getFold(node).at));
  measureRows(node);
}

function archiveBtn(node, plan, due) {
  const b = document.createElement("button");
  b.className = "kb-lc-arch" + (due ? " due" : "");
  b.textContent = "⤵ Archive " + plan.count;
  b.title = "Fold the " + plan.count + " oldest message(s) into one brief the model reads instead.\n"
    + "They stay in the chat, dimmed; deleting the brief puts them back."
    + (due ? "\nThe context is past your " + getFold(node).at + "% mark." : "");
  if (busy(node)) b.disabled = true;
  b.addEventListener("pointerdown", (e) => e.stopPropagation());
  b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); if (!b.disabled) doFold(node); });
  return b;
}

function chipEl(node, text, title, active, disabled, onClick) {
  const b = document.createElement("button");
  b.className = "kb-lc-chip" + (active ? " on" : "");
  b.textContent = text;
  b.title = title;
  if (disabled) b.disabled = true;
  b.addEventListener("pointerdown", (e) => e.stopPropagation());
  b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); if (!b.disabled) onClick(); });
  return b;
}

// Clicking a chip only SELECTS a persona — Send is the sole trigger for a generation.
function pick(node, i) {
  setActive(node, i + 1);
  renderPersonaBar(node);
  node.setDirtyCanvas?.(true, true);
}

function renderPersonaBar(node) {
  const bar = node._kbBar;
  if (!bar) return;
  const on = connected(node);
  setActive(node, getActive(node));     // normalise the widget so the backend agrees with the UI

  bar.innerHTML = "";
  // A single persona is just the node's config — there is nothing to switch between, so no row.
  bar.classList.toggle("kb-lc-hide", on.length < 2);
  if (on.length < 2) { measureRows(node); return; }

  const off = busy(node);
  const act = getActive(node);
  const meta = getPMeta(node);
  const mark = (i) => (needsReload(node, i) ? " ⟳" : "");
  const reloadTip = (i) => (needsReload(node, i) ? "\n⟳ different model — picking this reloads it" : "");

  for (const i of on) {
    const sys = personaSystem(node, i).trim().replace(/\s+/g, " ").slice(0, 140);
    const keep = meta[i].keep;
    const tip = "persona_" + (i + 1)
      + (sys ? "\n" + sys + "…" : "")
      + (keep === 0 ? "\n🚫 its turns never reach the context"
         : keep !== null ? "\n🚫 only its last " + keep + " turn(s) stay in the context" : "")
      + (meta[i].private ? "\n🔒 private — the other personas don't see its turns" : "")
      + (meta[i].trigger ? "\nTrigger: " + meta[i].trigger : "")
      + reloadTip(i);
    bar.appendChild(chipEl(node, personaLabel(node, i) + mark(i), tip, act === i + 1, off,
                           () => pick(node, i)));
  }
  bar.appendChild(chipEl(node, "⚙", "Persona labels, trigger messages and context retention",
                         false, off, () => personaDialog(node)));
  measureRows(node);
}

// Plain DOM overlay — NOT window.prompt, which the ComfyUI desktop app (Electron) forbids.
function personaDialog(node) {
  const draft = getPMeta(node);
  const overlay = document.createElement("div");
  Object.assign(overlay.style, { position: "fixed", inset: "0", background: "rgba(0,0,0,0.55)",
    zIndex: 10000, display: "flex", alignItems: "center", justifyContent: "center" });
  const box = document.createElement("div");
  Object.assign(box.style, { background: "#222", color: "#eee", border: "1px solid #444",
    borderRadius: "8px", padding: "16px", minWidth: "460px", maxWidth: "660px", maxHeight: "80vh",
    overflow: "auto", font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" });

  const close = () => { document.removeEventListener("keydown", onKey, true); overlay.remove(); };
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  // These are real text inputs; don't let their keys reach ComfyUI's global hotkeys.
  box.addEventListener("keydown", (e) => e.stopPropagation());

  const h = document.createElement("div");
  h.textContent = "🎭 Personas";
  Object.assign(h.style, { fontSize: "15px", fontWeight: "600", marginBottom: "4px" });
  const sub = document.createElement("div");
  sub.textContent = "Model, sampling and system prompt come from the Settings node wired into "
    + "each persona_N input. This is only how their chips look and behave.";
  Object.assign(sub.style, { opacity: "0.6", marginBottom: "12px", fontSize: "12px" });
  box.append(h, sub);

  const field = (label, value, hint, onInput) => {
    const w = document.createElement("label");
    w.style.cssText = "display:flex;align-items:center;gap:8px;margin-top:6px;";
    const t = document.createElement("span");
    t.textContent = label;
    t.style.cssText = "flex:0 0 92px;opacity:.7;font-size:12px;";
    const inp = document.createElement("input");
    inp.type = "text"; inp.value = value; inp.placeholder = hint;
    inp.style.cssText = "flex:1;min-width:0;background:#1a1a1a;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 8px;";
    inp.addEventListener("input", () => onInput(inp.value));
    w.append(t, inp);
    return w;
  };

  // ── node-level: archiving ──
  const fold = getFold(node);
  const arch = document.createElement("div");
  arch.style.cssText = "border:1px solid #333;border-radius:6px;padding:8px 10px;margin-bottom:14px;";
  const ah = document.createElement("div");
  ah.textContent = "🗂 Archiving";
  ah.style.cssText = "font-size:12px;opacity:.75;font-weight:600;";
  arch.appendChild(ah);
  arch.appendChild(field("Nag at", String(fold.at), "70",
    (v) => { fold.at = Math.max(10, Math.min(100, Math.trunc(Number(v)) || 70)); }));
  arch.appendChild(field("Keep verbatim", String(fold.keep), "8",
    (v) => { fold.keep = Math.max(2, Math.min(200, Math.trunc(Number(v)) || 8)); }));

  const sw = document.createElement("label");
  sw.style.cssText = "display:flex;align-items:center;gap:8px;margin-top:6px;";
  const st = document.createElement("span");
  st.textContent = "Summarised by";
  st.style.cssText = "flex:0 0 92px;opacity:.7;font-size:12px;";
  const sel = document.createElement("select");
  sel.style.cssText = "flex:1;min-width:0;background:#1a1a1a;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 8px;";
  const opt = (val, label) => { const o = document.createElement("option"); o.value = String(val); o.textContent = label; if (fold.by === val) o.selected = true; sel.appendChild(o); };
  opt(0, "the active persona's model (no reload)");
  for (let i = 0; i < PCOUNT; i++) if (isConnected(node, i)) opt(i + 1, personaLabel(node, i));
  sel.addEventListener("change", () => { fold.by = Math.trunc(Number(sel.value)) || 0; });
  sw.append(st, sel);
  arch.appendChild(sw);

  const ahint = document.createElement("div");
  ahint.textContent = "⤵ Archive folds the older turns into one editable brief the model reads "
    + "instead of them. 'Keep verbatim' is how many recent messages are never folded; 'Nag at' only "
    + "colours the button once the context passes that %. Nothing is deleted — the originals stay in "
    + "the chat, dimmed, and deleting the brief puts them back.";
  ahint.style.cssText = "margin:8px 0 0;font-size:11px;opacity:.5;line-height:1.4;";
  arch.appendChild(ahint);
  box.appendChild(arch);

  let any = false;
  for (let i = 0; i < PCOUNT; i++) {
    if (!isConnected(node, i)) continue;
    any = true;
    const row = document.createElement("div");
    row.style.cssText = "border:1px solid #333;border-radius:6px;padding:8px 10px;margin-bottom:8px;";
    const head = document.createElement("div");
    head.textContent = "persona_" + (i + 1) + (upstreamTitle(node, i) ? " ← " + upstreamTitle(node, i) : "");
    head.style.cssText = "font-size:12px;opacity:.75;font-weight:600;";
    row.appendChild(head);

    const sys = personaSystem(node, i).trim();
    if (sys) {
      const pv = document.createElement("div");
      pv.textContent = sys.replace(/\s+/g, " ").slice(0, 220) + (sys.length > 220 ? "…" : "");
      pv.style.cssText = "margin-top:4px;font-size:11px;opacity:.5;font-style:italic;";
      row.appendChild(pv);
    }

    row.appendChild(field("Chip label", draft[i].label,
      upstreamTitle(node, i) || ("Persona " + (i + 1)), (v) => { draft[i].label = v; }));
    row.appendChild(field("Trigger", draft[i].trigger,
      "optional — sent as your message instead of an empty turn", (v) => { draft[i].trigger = v; }));
    row.appendChild(field("Keep in context", draft[i].keep === null ? "" : String(draft[i].keep),
      "all", (v) => { draft[i].keep = normKeep(v, false); }));

    const hint = document.createElement("div");
    hint.textContent = "How many of this persona's most recent turns still reach the model. "
      + "Blank = all. 0 = none. A prompt-writer's long drafts are usually worth 1–2; older ones "
      + "stay in the chat, dimmed, but stop costing context.";
    hint.style.cssText = "margin:6px 0 0 100px;font-size:11px;opacity:.5;line-height:1.4;";
    row.appendChild(hint);

    const pw = document.createElement("label");
    pw.style.cssText = "display:flex;align-items:flex-start;gap:8px;margin-top:9px;font-size:12px;cursor:pointer;";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = draft[i].private;
    cb.style.marginTop = "2px";
    cb.addEventListener("change", () => { draft[i].private = cb.checked; });
    const ct = document.createElement("span");
    ct.innerHTML = "";
    ct.textContent = "Private — only you and this persona see its turns. It still reads its own "
      + "back (that's how it revises a draft); the other personas never see them at all.";
    ct.style.cssText = "opacity:.8;line-height:1.4;";
    pw.append(cb, ct);
    row.appendChild(pw);
    box.appendChild(row);
  }
  if (!any) {
    const e = document.createElement("div");
    e.textContent = "(no persona inputs connected)";
    e.style.opacity = "0.6";
    box.appendChild(e);
  }

  const f = document.createElement("div");
  f.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:14px;";
  const cancel = document.createElement("button");
  cancel.textContent = "Cancel";
  Object.assign(cancel.style, { background: "#333", color: "#eee", border: "1px solid #555", borderRadius: "4px", padding: "6px 12px", cursor: "pointer" });
  cancel.onclick = close;
  const save = document.createElement("button");
  save.textContent = "Save";
  Object.assign(save.style, { background: "#3b82f6", color: "#fff", border: "1px solid #555", borderRadius: "4px", padding: "6px 12px", cursor: "pointer" });
  save.onclick = () => {
    setPMeta(node, draft);
    setFold(node, fold);
    renderPersonaBar(node);
    render(node);
    renderStats(node);
    node.setDirtyCanvas?.(true, true);
    close();
  };
  f.append(cancel, save);
  box.appendChild(f);

  overlay.appendChild(box);
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
  document.addEventListener("keydown", onKey, true);
  document.body.appendChild(overlay);
}

// ── history mutations ───────────────────────────────────────────────────────────────────────

function startEdit(node, i) {
  if (busy(node)) return;
  node._kbEditIdx = i; node._kbEditDraft = null;
  render(node);
}
function cancelEdit(node) {
  node._kbEditIdx = -1; node._kbEditDraft = null;
  render(node);
}
function saveEdit(node, i, text) {
  const h = getHistory(node);
  if (h[i]) { h[i].content = text; setHistory(node, h); }   // thoughts / persona / ctx survive
  node._kbEditIdx = -1; node._kbEditDraft = null;
  render(node);
}
// Clear a by-hand `ctx:false` (the retention window isn't cleared here — change it in ⚙).
function unhide(node, i) {
  if (busy(node)) return;
  const h = getHistory(node);
  if (!h[i]) return;
  delete h[i].ctx;
  setHistory(node, h);
  render(node);
}
function deleteAt(node, i) {
  if (busy(node)) return;
  const h = getHistory(node);
  if (!h[i]) return;
  // Binning the brief un-archives everything it stood for — there is only ever one.
  if (h[i].role === "digest") for (const m of h) { if (m && m.fold) delete m.fold; }
  const gone = Array.isArray(h[i].att) ? h[i].att.slice() : [];
  h.splice(i, 1);
  setHistory(node, h);
  node._kbEditIdx = -1; node._kbEditDraft = null;
  render(node);
  renderStats(node);
  if (gone.length) discardUnused(gone);   // the message took its pictures with it
}
// Where the turn that produced h[i] begins, and how to replay it. A reply is paired with the user
// message directly above it — unless there isn't one, which is exactly what a 'fresh' turn looks
// like. (Scanning further back would swallow the previous persona's turn, which is the bug this
// replaces: ↻ on a generator's reply used to re-run the chat persona's message before it.)
function turnAt(h, i) {
  const m = h[i];
  if (!m || typeof m !== "object" || m.role === "digest") return null;  // a brief isn't a turn
  if (m.role === "user") {
    return { start: i, mode: "turn", text: String(m.content || ""), persona: m.persona || "" };
  }
  const prev = h[i - 1];
  if (prev && prev.role === "user") {
    return { start: i - 1, mode: "turn", text: String(prev.content || ""),
             persona: prev.persona || m.persona || "" };
  }
  return { start: i, mode: "fresh", text: "", persona: m.persona || "" };
}

function rerunAt(node, i) {
  if (busy(node)) return;
  const h = getHistory(node);
  const t = turnAt(h, i);
  if (!t) return;
  // The file is still on disk, so a replay can put the picture back in front of the model even
  // though the stored turn had long since decayed to its text marker.
  const back = Array.isArray(h[t.start]?.att) ? h[t.start].att.slice() : [];
  if (t.mode === "turn" && !t.text.trim() && !back.length) return;
  setAtt(node, back.concat(getAtt(node)));
  setHistory(node, h.slice(0, t.start));                 // onExecuted re-appends the whole turn
  node._kbEditIdx = -1; node._kbEditDraft = null;
  doSend(node, { persona: personaIndexByLabel(node, t.persona), mode: t.mode, text: t.text });
}

// ── sending ─────────────────────────────────────────────────────────────────────────────────

function clearPending(node) {
  node._kbMode = null;
  node._kbPending = null;
  node._kbStream = "";
  node._kbStreamEls = null;
}

// What pressing Send means right now.
//   turn     — there is text (typed, replayed by ↻, or the persona's trigger)
//   continue — empty box and the last reply is the active persona's own: resume it
//   fresh    — empty box, someone else spoke last: generate with NO user message at all
function planTurn(node) {
  const typed = (node._kbInput?.value || "").trim();
  if (typed) return { mode: "turn", text: typed };
  // A picture with no words is still "look at this" — a turn, not a continuation. It also has to
  // be a turn for the pixels to reach the model at all: 'fresh' and 'continue' send none.
  if (getAtt(node).length) return { mode: "turn", text: "" };

  const h = getHistory(node);
  const last = h[h.length - 1];
  const who = activeLabel(node);
  if (last && last.role === "assistant" && String(last.persona || "") === who
      && typeof last.content === "string" && last.content) {
    return { mode: "continue", text: "" };
  }
  const pi = getActive(node) - 1;
  const trigger = pi >= 0 ? getPMeta(node)[pi].trigger.trim() : "";
  if (trigger) return { mode: "turn", text: trigger };
  return { mode: "fresh", text: "" };
}

// Archiving: not a chat turn. The older messages go to a summariser as a one-shot task and come
// back as the 🗂 brief; onExecuted marks what they replaced.
function doFold(node) {
  if (busy(node)) return;
  const plan = foldPlan(node);
  if (!plan.count) return;
  const st = ST(node);
  st.turn = { mode: "fold", persona: activeLabel(node), keep: keepMap(node),
              private: privateSet(node), upto: plan.upto, summarizer: getFold(node).by };
  st.user = "";
  node._kbMode = "fold";
  node._kbPending = null;
  node._kbStream = "";
  node._kbEditIdx = -1; node._kbEditDraft = null;
  node._kbStick = true;
  render(node);
  renderStats(node);
  renderPersonaBar(node);
  renderTray(node);       // archiving isn't a chat turn: the tray waits, just greyed out
  st.nonce = ((st.nonce | 0) + 1) % 0x7fffffff;
  app.queuePrompt(0);
}

// opts: {persona} 0..3 (default: the active one), {mode, text} to force a replay (used by ↻).
function doSend(node, opts) {
  if (busy(node)) return;
  const o = opts || {};
  let idx = (o.persona == null) ? getActive(node) - 1 : o.persona;
  if (!(idx >= 0 && idx < PCOUNT && isConnected(node, idx))) idx = getActive(node) - 1;
  if (idx < 0) return;                        // no Settings node wired anywhere
  setActive(node, idx + 1);

  const plan = o.mode ? { mode: o.mode, text: String(o.text || "") } : planTurn(node);
  if (plan.mode === "turn" && !plan.text && !getAtt(node).length) {
    renderPersonaBar(node); render(node); return;
  }
  // A stray Enter on an untouched chat has nothing to work from at all.
  if (plan.mode === "fresh" && !getHistory(node).length) {
    renderPersonaBar(node); render(node); return;
  }

  const st = ST(node);
  st.turn = { mode: plan.mode, persona: activeLabel(node),
              keep: keepMap(node), private: privateSet(node) };
  st.user = plan.text;

  node._kbMode = plan.mode;
  node._kbPending = plan.mode === "turn" ? plan.text : null;
  node._kbStream = "";                         // reset the streaming accumulator
  node._kbEditIdx = -1; node._kbEditDraft = null;
  node._kbStick = true;                        // a new turn always snaps to the bottom
  render(node);                                // shows the pending pair / the live bubble
  renderPersonaBar(node);                      // busy → chips disabled
  // The tray is deliberately NOT cleared here — st.att is what the backend reads for this turn's
  // pixels, and onExecuted moves it onto the message once the reply lands. A turn that dies on
  // the way therefore leaves the pictures staged, ready for another go.
  renderTray(node);
  st.nonce = ((st.nonce | 0) + 1) % 0x7fffffff;  // force one turn per press past ComfyUI's cache
  app.queuePrompt(0);
}

// ── websocket / lifecycle events ────────────────────────────────────────────────────────────

// One listener for all chat nodes: append the streamed delta to the matching node.
api.addEventListener("kinburg.chatllm", ({ detail }) => {
  if (!detail) return;
  const node = (app.graph?._nodes || []).find((n) => String(n.id) === String(detail.id));
  if (!node || !node._kbStreamEls) return;
  node._kbStream = (node._kbStream || "") + (detail.delta || "");
  updateStreaming(node);
  node.setDirtyCanvas?.(true, true);
});

// A pending turn disables Send, the chips and every message action, so it must never get stuck.
// The failure can happen anywhere upstream, so don't bother matching node ids — unstick every chat
// node. Deliberately NOT hooked to execution_success: a turn queued behind another job would be
// unstuck by that job finishing, killing the live bubble. For the rare "the node never ran at all"
// case there's the ✕ on the pending bubble.
function unstickAll() {
  for (const n of instances) {
    if (!busy(n)) continue;
    clearPending(n);
    render(n);
    renderPersonaBar(n);
    renderTray(n);          // the turn never landed, so its pictures stay staged — re-enable them
    n.setDirtyCanvas?.(true, true);
  }
}
for (const ev of ["execution_error", "execution_interrupted"]) {
  api.addEventListener(ev, unstickAll);
}

// ── node setup ──────────────────────────────────────────────────────────────────────────────

function setup(node) {
  injectStyle();
  // Drop the widget ComfyUI auto-creates for chat_state — the chat window carries the value
  // itself (below), so the node ends up with no invisible rows eating vertical space. Splice in
  // place: Vue renders from this very array and misses a reassignment. The legacy names are
  // listed so a stale cached node definition can't leave orphan rows behind either.
  const carriers = ["chat_state", "user_message", "history_json", "nonce", "approved",
                    "active_persona", "turn_json"];
  for (let i = node.widgets.length - 1; i >= 0; i--) {
    if (carriers.includes(node.widgets[i].name)) node.widgets.splice(i, 1);
  }

  clearPending(node);            // _kbMode / _kbPending / _kbStream / _kbStreamEls
  ST(node);                      // seed the state object before anything renders
  node._kbEditIdx = -1;
  node._kbEditDraft = null;
  node._kbStick = true;
  node._kbBarH = 0;
  node._kbStatH = 0;
  node._kbTrayH = 0;
  node._kbAttErr = "";

  const wrap = document.createElement("div");
  wrap.className = "kb-lc-wrap";

  const box = document.createElement("div");
  box.className = "kb-lc-box";
  const view = document.createElement("div");
  view.className = "kb-lc-view";
  const jump = document.createElement("button");
  jump.className = "kb-lc-jump kb-lc-hide";
  jump.textContent = "↓ latest";
  jump.addEventListener("pointerdown", (e) => e.stopPropagation());
  jump.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); toBottom(node); });
  box.append(view, jump);

  const stat = document.createElement("div");
  stat.className = "kb-lc-stat kb-lc-hide";
  stat.addEventListener("pointerdown", (e) => e.stopPropagation());

  const bar = document.createElement("div");
  bar.className = "kb-lc-bar kb-lc-hide";

  const tray = document.createElement("div");
  tray.className = "kb-lc-tray kb-lc-hide";

  const input = document.createElement("textarea");
  input.className = "kb-lc-input";
  input.placeholder = "Type a message…  (Enter = send, Shift+Enter = new line, empty = let the persona continue)";
  input.title = "Ctrl+V or drop a picture here to send it with your message — or use 📎.";

  // The 📎 is overlaid in the corner of the input instead of sitting in the tray row, so the tray
  // can still collapse to nothing when there is nothing attached.
  const inputWrap = document.createElement("div");
  inputWrap.className = "kb-lc-inputwrap";
  const picker = document.createElement("input");
  picker.type = "file";
  picker.accept = "image/*";
  picker.multiple = true;
  picker.style.display = "none";
  picker.addEventListener("change", () => {
    const files = imagesIn(picker.files);
    picker.value = "";                       // so picking the same file twice still fires
    if (files.length) addFiles(node, files);
  });
  const clip = document.createElement("button");
  clip.className = "kb-lc-clip";
  clip.textContent = "📎";
  clip.title = "Attach a picture (or just paste / drop one into the chat)";
  clip.addEventListener("pointerdown", (e) => e.stopPropagation());
  clip.addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    if (!busy(node)) picker.click();
  });
  inputWrap.append(input, clip, picker);

  wrap.append(box, stat, bar, tray, inputWrap);
  node._kbView = view;
  node._kbJump = jump;
  node._kbStat = stat;
  node._kbBar = bar;
  node._kbTray = tray;
  node._kbClip = clip;
  node._kbInput = input;

  view.addEventListener("wheel", (e) => { view.scrollTop += e.deltaY; e.preventDefault(); e.stopPropagation(); }, { passive: false });
  // Any scroll re-decides whether we follow: parked at the bottom → follow, anywhere above → freeze.
  view.addEventListener("scroll", () => {
    const bottom = atBottom(node);
    node._kbStick = bottom;
    showJump(node, !bottom, false);
  });
  input.addEventListener("pointerdown", (e) => e.stopPropagation());
  input.addEventListener("wheel", (e) => e.stopPropagation());
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doSend(node); }
  });
  // Ctrl+V of an image. Both preventDefault and stopPropagation are needed: ComfyUI's own paste
  // handler would otherwise drop a Load Image node onto the canvas. A text paste is left alone.
  input.addEventListener("paste", (e) => {
    const files = imagesIn(e.clipboardData?.files);
    if (!files.length) return;
    e.preventDefault(); e.stopPropagation();
    addFiles(node, files);
  });
  // Same story for a dropped file — the canvas would otherwise take it.
  const dragging = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  wrap.addEventListener("dragover", (e) => {
    if (!dragging(e)) return;
    e.preventDefault(); e.stopPropagation();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
    wrap.classList.add("kb-lc-drag");
  });
  wrap.addEventListener("dragleave", (e) => {
    if (e.target === wrap) wrap.classList.remove("kb-lc-drag");
  });
  wrap.addEventListener("drop", (e) => {
    wrap.classList.remove("kb-lc-drag");
    const files = imagesIn(e.dataTransfer?.files);
    if (!files.length) return;
    e.preventDefault(); e.stopPropagation();
    addFiles(node, files);
  });

  // Named after the backend input on purpose: graphToPrompt reads inputs[widget.name] =
  // widget.value, and a DOM widget's value is whatever getValue returns. So this one element is
  // both the chat UI and the state carrier — no extra widget, no extra row.
  node.addDOMWidget("chat_state", "kinburg_llmchat", wrap, {
    serialize: true,
    getValue: () => stateJSON(node),
    setValue: (v) => loadState(node, v),
    getMinHeight: () => VIEW_MIN + GAP + INPUT_H
      + (node._kbStatH ? node._kbStatH + GAP : 0)
      + (node._kbBarH ? node._kbBarH + GAP : 0)
      + (node._kbTrayH ? node._kbTrayH + GAP : 0),
    getMaxHeight: () => 100000,
  });

  // Both rows wrap to a second line on narrow nodes — keep the reserved height in sync.
  try {
    node._kbRO = new ResizeObserver(() => measureRows(node));
    node._kbRO.observe(bar);
    node._kbRO.observe(stat);
    node._kbRO.observe(tray);
  } catch (e) { node._kbRO = null; }

  node.addWidget("button", "📨 Send", null, () => doSend(node), { serialize: false });
  node.addWidget("button", "✅ Approve", null, async () => {
    const st = ST(node);
    st.approved = true;
    try { await app.queuePrompt(0); } finally { st.approved = false; }
  }, { serialize: false });
  node.addWidget("button", "🗑 Clear", null, () => {
    // Personas are untouched — the history, the tray and the pictures behind them go. Clearing is
    // the one moment we can be sure those files are finished with, so it doubles as the cleanup.
    const refs = allAtt(node);
    const ask = "Clear this chat?" + (refs.length
      ? "\n\n" + refs.length + " attached picture(s) will be deleted from input/" + ATT_DIR + " too."
      : "");
    if (!confirm(ask)) return;
    setHistory(node, []);
    setAtt(node, []);
    setStats(node, null);                       // the meter described a context that's now gone
    clearPending(node);
    node._kbEditIdx = -1; node._kbEditDraft = null;
    node._kbAttErr = "";
    if (node._kbInput) node._kbInput.value = "";
    render(node);
    renderStats(node);
    renderPersonaBar(node);
    renderTray(node);
    if (refs.length) discardUnused(refs);       // fire and forget: the chat is empty either way
  }, { serialize: false });

  reorder(node, ["chat_state", "📨 Send", "✅ Approve", "🗑 Clear", "unload_on_approve"]);

  render(node);
  renderStats(node);
  renderPersonaBar(node);
  renderTray(node);
  if ((node.size?.[1] || 0) < 560) node.setSize([Math.max(node.size?.[0] || 0, 380), 600]);
}

app.registerExtension({
  name: "Kinburg.LocalLLMChat",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      setup(this);
      instances.add(this);
      return r;
    };

    // Send finished → the ui payload has the final answer + thoughts. Record the turn (with the
    // authoritative split) and clear the streaming state. Approve returns no ui payload.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const raw = message?.kinburg_chatllm?.[0];
      if (raw == null) return;
      let p; try { p = JSON.parse(raw); } catch (e) { return; }
      if (p.reply != null) {
        const h = getHistory(this);
        // _kbMode/_kbPending are transient — a tab switch destroys the node mid-turn. The turn
        // descriptor rides in the serialized state, so it survives and says what this reply is for.
        const stored = ST(this).turn || {};
        const mode = this._kbMode || stored.mode || "turn";
        const pname = activeLabel(this);
        // Only the persona tag is stored — whether the turn is in the context is decided by the
        // retention window at request time, so it can slide instead of being frozen here.
        const tag = (o) => { if (pname) o.persona = pname; return o; };

        if (mode === "fold" && p.fold) {
          // Mark what the brief now stands for, then put (or replace) the brief at the top.
          const upto = Math.max(0, Math.min(Number(p.fold.upto) || 0, h.length));
          const id = 1 + h.reduce((mx, m) => Math.max(mx, Number(m?.fold) || 0), 0);
          for (let k = 0; k < upto; k++) {
            const m = h[k];
            if (m && (m.role === "user" || m.role === "assistant") && !m.fold) m.fold = id;
          }
          const entry = { role: "digest", content: p.reply, fold: id };
          const di = digestIndex(h);
          if (di >= 0) h.splice(di, 1, entry); else h.unshift(entry);
        } else if (mode === "continue" && h.length && h[h.length - 1]?.role === "assistant") {
          const last = h[h.length - 1];          // extend the truncated reply in place
          last.content = String(last.content || "") + p.reply;
          if (p.thoughts) last.thoughts = (last.thoughts ? last.thoughts + "\n\n" : "") + p.thoughts;
        } else {
          const user = this._kbPending != null ? this._kbPending
            : (mode === "turn" ? (ST(this).user || "") : "");
          // The tray belongs to the message it went out with — move it, don't copy, so the same
          // pictures can't ride the next turn too. From here on the model only reads their marker.
          const sent = getAtt(this);
          if (user || sent.length) {
            const um = tag({ role: "user", content: user });        // APPEND — ↻ relies on this
            if (sent.length) um.att = sent;
            h.push(um);
          }
          setAtt(this, []);
          const msg = tag({ role: "assistant", content: p.reply });
          if (p.thoughts) msg.thoughts = p.thoughts;
          h.push(msg);
        }
        setHistory(this, h);
        setStats(this, p.stats || null);
        clearPending(this);
        this._kbStick = true;
        ST(this).user = "";
        if (this._kbInput) this._kbInput.value = "";
        render(this);
        renderStats(this);
        renderPersonaBar(this);
        renderTray(this);
      }
      this.setDirtyCanvas?.(true, true);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure?.apply(this, arguments);
      // chat_state's setValue has already run for new-format workflows; older ones kept the chat
      // in six separate widget values, so lift those across.
      migrateLegacy(this, info);
      render(this);
      renderStats(this);
      renderPersonaBar(this);
      renderTray(this);
      return r;
    };

    // Wiring or unwiring a persona_N input adds/removes its chip; re-wiring config changes the ⟳.
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConnectionsChange?.apply(this, arguments);
      if (this._kbBar) { renderPersonaBar(this); this.setDirtyCanvas?.(true, true); }
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      instances.delete(this);
      try { this._kbRO?.disconnect(); } catch (e) {}
      this._kbRO = null;
      return onRemoved?.apply(this, arguments);
    };
  },
});
