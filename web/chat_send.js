import { app } from "../../scripts/app.js";
import { chatNodes, sendToChat } from "./chat_llm.js";

// Send Image to Chat — the frontend half. The backend has already written the picture into
// input/kinburg_chat and handed us a file reference; all that is left is deciding WHICH chat node
// gets it and WHEN.
//
// The reference is kept in node.properties, so it survives a tab switch and the 📌 still works
// long after the branch that produced it ran. Nothing here talks to the graph — that is the whole
// point of the node: a picture reaching the chat must never make 📨 Send re-run a sampler.

const CLASS = "LocalLLMChatSendImage";
const AUTO = "(the only chat node)";

const lastSend = (node) => {
  const p = node.properties?._kbSend;
  return (p && typeof p === "object" && Array.isArray(p.refs) && p.refs.length) ? p : null;
};

// Chat nodes as picker entries. Titles can collide, so the id goes in front and is what we match
// on — a renamed chat node keeps working, a deleted one falls back to auto.
const label = (n) => "#" + n.id + " " + (n.title || "Local LLM Chat");
const idOf = (v) => {
  const m = /^#(\d+)\b/.exec(String(v || ""));
  return m ? m[1] : null;
};

function targetOf(node) {
  const id = idOf(node.properties?._kbTarget);
  if (!id) return null;                          // auto: sendToChat resolves it, or complains
  return chatNodes().find((n) => String(n.id) === id) || null;
}

function flash(node, w, text) {
  const was = w._kbLabel || (w._kbLabel = w.name);
  w.name = text;
  app.graph?.setDirtyCanvas(true, true);
  clearTimeout(w._kbTimer);
  w._kbTimer = setTimeout(() => {
    w.name = was;
    app.graph?.setDirtyCanvas(true, true);
  }, 2600);
}

const widgetVal = (node, name) => (node.widgets || []).find((w) => w.name === name)?.value;

// Re-derive the reference from the widgets AS THEY ARE NOW, keeping only the filename the backend
// produced. Otherwise 📌 would send whatever `send_as` said at the moment the branch executed —
// and the natural way to use this node is to look at the picture FIRST and only then decide who it
// comes from. Only `megapixels` still needs a re-run: it changes the file, not the reference.
//
// The ctx rule here MUST match send_image_node.run() — a persona's picture is invisible to the
// model unless note_in_context says otherwise; yours is always remembered.
function refsNow(node, p) {
  const as = widgetVal(node, "send_as") ?? p.as;
  const caption = String(widgetVal(node, "caption") ?? "").trim();
  const shot = String(widgetVal(node, "shot") ?? "").trim();
  const note = !!widgetVal(node, "note_in_context");
  const toUser = as === "me (user)";
  const refs = p.refs.map((r) => {
    const o = { name: r.name, subfolder: r.subfolder, type: r.type };
    if (caption) o.caption = caption;
    if (shot) o.shot = shot;
    if (!toUser && !note) o.ctx = false;
    return o;
  });
  return { refs, as };
}

function push(node, w) {
  const p = lastSend(node);
  if (!p) return flash(node, w, "📌 run this branch first");
  const { refs, as } = refsNow(node, p);
  const r = sendToChat({ refs, as, target: targetOf(node) });
  flash(node, w, (r.ok ? "📌 " : "⚠ ") + r.msg);
  if (!r.ok) console.warn("[Send Image to Chat]", r.msg);
}

app.registerExtension({
  name: "Kinburg.ChatSendImage",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const node = this;

      // Which chat node to feed. Deliberately NOT a backend input: ComfyUI validates a combo's
      // value against the list its INPUT_TYPES declared, and this list only exists at runtime.
      // properties carries it instead — serialized with the workflow, no validation, no extra row
      // beyond this one.
      const pick = node.addWidget("combo", "→ chat", node.properties?._kbTarget || AUTO,
        (v) => {
          node.properties = node.properties || {};
          node.properties._kbTarget = v;
        }, { values: [AUTO], serialize: false });
      pick.tooltip = "Which chat window the picture goes to. Leave on auto when there is only one.";
      // Refresh the list as the menu opens, so a chat node added since is there.
      const openPicker = () => {
        pick.options.values = [AUTO, ...chatNodes().map(label)];
      };
      node._kbRefreshTargets = openPicker;
      openPicker();

      const btn = node.addWidget("button", "📌 Send to chat", null, () => push(node, btn),
                                 { serialize: false });
      btn.tooltip = "Put the last generated picture in the chat. Works any time after this branch "
        + "has run — it does not re-run anything.";
      return r;
    };

    // The backend saved the file and told us where it is. Stash it, and push straight away when
    // the node is set to "every run" — the reference is idempotent, so a cached or repeated
    // execution cannot stack the same picture twice.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const raw = message?.kinburg_chatsend?.[0];
      if (raw == null) return;
      let p;
      try { p = JSON.parse(raw); } catch (e) { return; }
      if (p?.error) {
        console.error("[Send Image to Chat]", p.error);
        return;
      }
      if (!Array.isArray(p?.refs) || !p.refs.length) return;
      this.properties = this.properties || {};
      this.properties._kbSend = p;
      this._kbRefreshTargets?.();
      if (p.when === "every run") {
        // Same path as 📌 — at this instant the widgets still hold what the backend just used, so
        // it comes to the same thing, and there is only one rule to keep straight.
        const { refs, as } = refsNow(this, p);
        const r = sendToChat({ refs, as, target: targetOf(this) });
        if (!r.ok) console.warn("[Send Image to Chat]", r.msg);
      }
      this.setDirtyCanvas?.(true, true);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      // The picker's list is runtime-only; its VALUE came back in properties, so restore it.
      const w = (this.widgets || []).find((x) => x.name === "→ chat");
      if (w) {
        this._kbRefreshTargets?.();
        w.value = this.properties?._kbTarget || AUTO;
      }
      return r;
    };
  },
});
