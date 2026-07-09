import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Local LLM Chat (GGUF) — a chat window on the node.
//   • The chat area fills/stretches with the node (getMinHeight/getMaxHeight) and scrolls.
//   • 📨 Send streams the reply live (over the "kinburg.chatllm" websocket event); reasoning
//     (<think>…) shows in an open "💭 thinking" block during generation, then collapses into a
//     "💭 reasoning" toggle with the answer as the main text.
//   • Each message has a ⧉ copy button. Conversation lives in a hidden, serialized history_json.

const CLASS = "LocalLLMChatGGUF";
const wv = (node, name) => node.widgets?.find((w) => w.name === name);

function hideWidget(node, name) {
  const w = wv(node, name);
  if (!w) return;
  w.type = "hidden";
  w.computeSize = () => [0, -4];
  w.hidden = true;
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
  .kb-lc-wrap{position:relative;width:100%;height:100%;box-sizing:border-box;}
  .kb-lc-view{position:absolute;top:0;left:0;right:0;bottom:62px;overflow-y:auto;display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;background:#181818;border:1px solid #2b2b2b;border-radius:6px;font:12px/1.45 -apple-system,Segoe UI,sans-serif;}
  .kb-lc-view .kb-lc-msg{position:relative;max-width:85%;padding:5px 9px;border-radius:10px;white-space:pre-wrap;word-break:break-word;}
  .kb-lc-view .kb-lc-user{align-self:flex-end;background:#274b6d;color:#eaf2fb;border-bottom-right-radius:3px;}
  .kb-lc-view .kb-lc-asst{align-self:flex-start;background:#2c2c33;color:#e6e6ea;border-bottom-left-radius:3px;}
  .kb-lc-view .kb-lc-role{display:block;font-size:10px;opacity:.6;margin-bottom:1px;}
  .kb-lc-view .kb-lc-empty{margin:auto;color:#6d6d75;font-style:italic;text-align:center;}
  .kb-lc-view .kb-lc-copy{position:absolute;top:3px;right:4px;border:0;background:transparent;color:inherit;opacity:0;cursor:pointer;font-size:11px;line-height:1;padding:2px 4px;border-radius:4px;}
  .kb-lc-view .kb-lc-msg:hover .kb-lc-copy{opacity:.5;}
  .kb-lc-view .kb-lc-copy:hover{opacity:1;background:rgba(255,255,255,.14);}
  .kb-lc-view .kb-lc-think{margin:1px 0 4px;font-size:11px;}
  .kb-lc-view .kb-lc-think summary{cursor:pointer;opacity:.7;user-select:none;}
  .kb-lc-view .kb-lc-think-body{margin-top:3px;padding:4px 7px;border-left:2px solid #4a4a52;color:#b9b9c0;white-space:pre-wrap;word-break:break-word;font-style:italic;}
  .kb-lc-input{position:absolute;left:0;right:0;bottom:0;height:56px;resize:none;box-sizing:border-box;background:#181818;color:#e6e6e6;border:1px solid #2b2b2b;border-radius:6px;padding:6px 8px;font:12px/1.45 -apple-system,Segoe UI,sans-serif;outline:none;}
  `;
  document.head.appendChild(s);
}

function getHistory(node) {
  try {
    const a = JSON.parse(wv(node, "history_json")?.value || "[]");
    return Array.isArray(a) ? a : [];
  } catch (e) { return []; }
}
function setHistory(node, arr) {
  const w = wv(node, "history_json");
  if (w) w.value = JSON.stringify(arr);
}

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

function copyBtn(getText) {
  const b = document.createElement("button");
  b.className = "kb-lc-copy"; b.textContent = "⧉"; b.title = "Copy message";
  b.addEventListener("pointerdown", (e) => e.stopPropagation());
  b.addEventListener("click", (e) => {
    e.stopPropagation();
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

function makeBubble(role, content, thoughts) {
  const b = document.createElement("div");
  b.className = "kb-lc-msg " + (role === "user" ? "kb-lc-user" : "kb-lc-asst");
  const roleEl = document.createElement("span"); roleEl.className = "kb-lc-role"; roleEl.textContent = role === "user" ? "You" : "LLM";
  b.appendChild(roleEl);
  if (role !== "user" && thoughts) b.appendChild(reasoningEl(thoughts, false, "💭 reasoning").details);
  const textEl = document.createElement("span"); textEl.textContent = content;
  b.appendChild(textEl);
  b.appendChild(copyBtn(() => content));
  return b;
}

// The in-progress assistant bubble whose text/reasoning we update as tokens stream in.
function makeStreamingBubble() {
  const b = document.createElement("div");
  b.className = "kb-lc-msg kb-lc-asst";
  const roleEl = document.createElement("span"); roleEl.className = "kb-lc-role"; roleEl.textContent = "LLM";
  const think = reasoningEl("", true, "💭 thinking…");
  think.details.style.display = "none";
  const answerEl = document.createElement("span");
  b.append(roleEl, think.details, answerEl);
  return { bubble: b, refs: { thinkDetails: think.details, thinkBody: think.body, answerEl } };
}

function updateStreaming(node) {
  const refs = node._kbStreamEls;
  if (!refs) return;
  const { thinking, answer } = parseThink(node._kbStream || "");
  if (thinking) { refs.thinkDetails.style.display = ""; refs.thinkBody.textContent = thinking; }
  else { refs.thinkDetails.style.display = "none"; }
  refs.answerEl.textContent = answer || (thinking ? "" : "…");
  const v = node._kbView; if (v) v.scrollTop = v.scrollHeight;
}

function render(node) {
  const el = node._kbView;
  if (!el) return;
  node._kbStreamEls = null;
  el.innerHTML = "";
  const h = getHistory(node);
  if (!h.length && node._kbPending == null) {
    const e = document.createElement("div");
    e.className = "kb-lc-empty";
    e.textContent = "(no messages yet — type below and Send)";
    el.appendChild(e);
  }
  for (const m of h) el.appendChild(makeBubble(m.role, m.content || "", m.thoughts || ""));
  if (node._kbPending != null) {
    el.appendChild(makeBubble("user", node._kbPending, ""));
    const s = makeStreamingBubble();
    node._kbStreamEls = s.refs;
    el.appendChild(s.bubble);
    updateStreaming(node);
  }
  el.scrollTop = el.scrollHeight;
}

function doSend(node) {
  const msg = (node._kbInput?.value || "").trim();
  if (!msg) return;
  const um = wv(node, "user_message");
  if (um) um.value = msg;                 // feed the (hidden) backend widget
  node._kbPending = msg;
  node._kbStream = "";                     // reset the streaming accumulator
  render(node);                            // shows the user bubble + a live assistant bubble
  const nw = wv(node, "nonce");
  if (nw) nw.value = ((nw.value || 0) + 1) % 0x7fffffff;  // force one turn per press
  app.queuePrompt(0);
}

// One websocket listener for all chat nodes: append the streamed delta to the matching node.
api.addEventListener("kinburg.chatllm", ({ detail }) => {
  if (!detail) return;
  const node = (app.graph?._nodes || []).find((n) => String(n.id) === String(detail.id));
  if (!node || !node._kbStreamEls) return;
  node._kbStream = (node._kbStream || "") + (detail.delta || "");
  updateStreaming(node);
  node.setDirtyCanvas?.(true, true);
});

function setup(node) {
  injectStyle();
  ["user_message", "history_json", "nonce", "approved"].forEach((n) => hideWidget(node, n));

  const wrap = document.createElement("div");
  wrap.className = "kb-lc-wrap";
  const view = document.createElement("div");
  view.className = "kb-lc-view";
  const input = document.createElement("textarea");
  input.className = "kb-lc-input";
  input.placeholder = "Type a message…  (Enter = send, Shift+Enter = new line)";
  wrap.append(view, input);
  node._kbView = view;
  node._kbInput = input;

  view.addEventListener("wheel", (e) => { view.scrollTop += e.deltaY; e.preventDefault(); e.stopPropagation(); }, { passive: false });
  input.addEventListener("pointerdown", (e) => e.stopPropagation());
  input.addEventListener("wheel", (e) => e.stopPropagation());
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doSend(node); }
  });

  node.addDOMWidget("kb_chat_view", "kinburg_llmchat", wrap, {
    serialize: false,
    getMinHeight: () => 160,
    getMaxHeight: () => 100000,
  });

  node.addWidget("button", "📨 Send", null, () => doSend(node), { serialize: false });
  node.addWidget("button", "✅ Approve", null, async () => {
    const aw = wv(node, "approved");
    if (aw) aw.value = true;
    try { await app.queuePrompt(0); } finally { if (aw) aw.value = false; }
  }, { serialize: false });
  node.addWidget("button", "🗑 Clear", null, () => {
    if (!confirm("Clear this chat?")) return;
    setHistory(node, []);
    node._kbPending = null; node._kbStream = "";
    if (node._kbInput) node._kbInput.value = "";
    render(node);
  }, { serialize: false });

  reorder(node, ["kb_chat_view", "📨 Send", "✅ Approve", "🗑 Clear"]);

  render(node);
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
        const user = this._kbPending != null ? this._kbPending : (wv(this, "user_message")?.value || "");
        if (user) h.push({ role: "user", content: user });
        const msg = { role: "assistant", content: p.reply };
        if (p.thoughts) msg.thoughts = p.thoughts;
        h.push(msg);
        setHistory(this, h);
        this._kbPending = null; this._kbStream = "";
        const um = wv(this, "user_message"); if (um) um.value = "";
        if (this._kbInput) this._kbInput.value = "";
        render(this);
      }
      this.setDirtyCanvas?.(true, true);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      render(this);
      return r;
    };
  },
});
