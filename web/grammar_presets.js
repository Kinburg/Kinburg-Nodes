import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Grammar Presets — one dropdown of GBNF grammar templates (built-in + user-added). Values are
// served live from the backend store, so grammars you add appear without an object_info reload.
// Buttons add/edit a grammar and manage (delete) your custom ones.

const CLASS = "GrammarPresets";
const NONE = "🚫 None";

let STORE = { none: NONE, order: [NONE], grammars: {}, builtins: [] };

async function refreshStore() {
  try {
    const r = await api.fetchApi("/kinburg/grammars/data");
    const j = await r.json();
    if (j && j.ok) STORE = j;
  } catch (e) {
    console.error("[Kinburg] grammars: load failed", e);
  }
  return STORE;
}

async function postJSON(path, body) {
  const r = await api.fetchApi(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!j || !j.ok) throw new Error(j?.error || `request to ${path} failed`);
  STORE = j;
  return j;
}

const wv = (node, name) => node.widgets?.find((w) => w.name === name);

function syncNode(node) {
  const w = wv(node, "preset");
  if (!w) return;
  const opts = [...(STORE.order || [NONE])];
  if (w.value && !opts.includes(w.value)) opts.push(w.value); // keep a just-deleted selection visible
  if (!w.options) w.options = {};
  w.options.values = opts;
  node.setDirtyCanvas(true, true);
}

function refreshAllNodes() {
  for (const n of app.graph?._nodes || []) {
    if (n.comfyClass === CLASS || n.type === CLASS) syncNode(n);
  }
}

// ---------------------------------------------------------------- minimal modal helpers
function modal(title, build) {
  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", background: "rgba(0,0,0,0.55)", zIndex: 10000,
    display: "flex", alignItems: "center", justifyContent: "center",
  });
  const box = document.createElement("div");
  Object.assign(box.style, {
    background: "#222", color: "#eee", border: "1px solid #444", borderRadius: "8px",
    padding: "16px", minWidth: "420px", maxWidth: "640px", maxHeight: "80vh", overflow: "auto",
    font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
  });
  const h = document.createElement("div");
  h.textContent = title;
  Object.assign(h.style, { fontSize: "15px", fontWeight: "600", marginBottom: "12px" });
  box.appendChild(h);
  const close = () => overlay.remove();
  build(box, close);
  overlay.appendChild(box);
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
  document.body.appendChild(overlay);
  return close;
}

function row(parent, label) {
  const r = document.createElement("div");
  Object.assign(r.style, { display: "flex", flexDirection: "column", gap: "4px", marginBottom: "10px" });
  if (label) { const l = document.createElement("label"); l.textContent = label; l.style.opacity = "0.8"; r.appendChild(l); }
  parent.appendChild(r);
  return r;
}

function styledInput(el) {
  Object.assign(el.style, {
    background: "#111", color: "#eee", border: "1px solid #555", borderRadius: "4px",
    padding: "6px 8px", font: "13px monospace", width: "100%", boxSizing: "border-box",
  });
  return el;
}

function button(label, onClick, primary) {
  const b = document.createElement("button");
  b.textContent = label;
  Object.assign(b.style, {
    background: primary ? "#3b82f6" : "#333", color: "#eee", border: "1px solid #555",
    borderRadius: "4px", padding: "6px 12px", cursor: "pointer", font: "13px sans-serif",
  });
  b.addEventListener("click", onClick);
  return b;
}

function footer(parent, ...buttons) {
  const f = document.createElement("div");
  Object.assign(f.style, { display: "flex", gap: "8px", justifyContent: "flex-end", marginTop: "14px" });
  buttons.forEach((b) => f.appendChild(b));
  parent.appendChild(f);
}

// ---------------------------------------------------------------- dialogs
function addDialog(node) {
  modal("➕ Add / edit grammar", (box, close) => {
    const nameRow = row(box, "Name");
    const name = styledInput(document.createElement("input"));
    name.style.font = "13px sans-serif";
    name.placeholder = "e.g. Scene Card (JSON)";
    nameRow.appendChild(name);

    const textRow = row(box, "GBNF grammar");
    const text = styledInput(document.createElement("textarea"));
    text.rows = 12; text.placeholder = "root ::= ...";
    textRow.appendChild(text);

    // Pre-fill when an existing name is typed.
    const prefill = () => { const g = STORE.grammars?.[name.value.trim()]; if (g != null) text.value = g; };
    name.addEventListener("blur", prefill);

    footer(box,
      button("Cancel", close),
      button("Save", async () => {
        try {
          await postJSON("/kinburg/grammars/preset", { name: name.value, text: text.value });
          refreshAllNodes();
          const w = wv(node, "preset"); if (w) w.value = name.value.trim();
          close();
        } catch (e) { alert(e.message); }
      }, true),
    );
  });
}

function manageDialog(node) {
  const render = () => modal("🗑 Manage custom grammars", (box, close) => {
    const line = (label, onDel) => {
      const r = document.createElement("div");
      r.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;padding:4px 0;border-bottom:1px solid #333";
      const s = document.createElement("span"); s.textContent = label; s.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
      r.appendChild(s); r.appendChild(button("Delete", onDel)); box.appendChild(r);
    };
    const builtins = new Set(STORE.builtins || []);
    let any = false;
    for (const nm of Object.keys(STORE.grammars || {})) {
      if (builtins.has(nm)) continue; // built-ins can't be deleted
      any = true;
      line(nm, async () => {
        try { await postJSON("/kinburg/grammars/preset", { name: nm, delete: true }); refreshAllNodes(); close(); render(); }
        catch (e) { alert(e.message); }
      });
    }
    if (!any) { const e = document.createElement("div"); e.textContent = "(no custom grammars added)"; e.style.opacity = "0.6"; box.appendChild(e); }
    footer(box, button("Close", close, true));
  });
  render();
}

// ---------------------------------------------------------------- node wiring
app.registerExtension({
  name: "Kinburg.GrammarPresets",
  async setup() {
    await refreshStore();
    refreshAllNodes();
  },
  async nodeCreated(node) {
    if (node.comfyClass !== CLASS && node.type !== CLASS) return;
    node.addWidget("button", "➕ Add grammar", null, () => addDialog(node), { serialize: false });
    node.addWidget("button", "🗑 Manage", null, () => manageDialog(node), { serialize: false });
    if (!Object.keys(STORE.grammars || {}).length) refreshStore().then(() => syncNode(node));
    syncNode(node);
    node.setSize?.(node.computeSize?.());
  },
});
