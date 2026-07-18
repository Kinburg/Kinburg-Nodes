import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Card Presets reader — a dropdown of saved Character / Entity cards.
// Cards are SAVED on the backend: fill a Character Card / Entity Card, type a name in its
// `save_preset_as` field, and run — the node saves the resolved values (works whether fields are
// typed or wired in from outside). This node's dropdown is served live from the backend store.

const NONE = "🚫 None";
let STORE = { none: NONE, order: [NONE], presets: {} };

async function refreshStore() {
  try {
    const r = await api.fetchApi("/kinburg/cards/data");
    const j = await r.json();
    if (j && j.ok) STORE = j;
  } catch (e) { console.error("[Kinburg] cards: load failed", e); }
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

function syncReader(node) {
  const w = wv(node, "preset");
  if (!w) return;
  const opts = [...(STORE.order || [NONE])];
  if (w.value && !opts.includes(w.value)) opts.push(w.value);
  if (!w.options) w.options = {};
  w.options.values = opts;
  node.setDirtyCanvas(true, true);
}

function refreshReaders() {
  for (const n of app.graph?._nodes || []) {
    if (n.comfyClass === "CardPresets" || n.type === "CardPresets") syncReader(n);
  }
}

// Manage dialog — a plain DOM overlay (NOT window.prompt, which the desktop app forbids).
async function manageDialog() {
  await refreshStore();
  const overlay = document.createElement("div");
  Object.assign(overlay.style, { position: "fixed", inset: "0", background: "rgba(0,0,0,0.55)",
    zIndex: 10000, display: "flex", alignItems: "center", justifyContent: "center" });
  const box = document.createElement("div");
  Object.assign(box.style, { background: "#222", color: "#eee", border: "1px solid #444",
    borderRadius: "8px", padding: "16px", minWidth: "380px", maxWidth: "560px", maxHeight: "80vh",
    overflow: "auto", font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" });
  const close = () => overlay.remove();
  const render = () => {
    box.innerHTML = "";
    const h = document.createElement("div"); h.textContent = "🗑 Manage saved cards";
    Object.assign(h.style, { fontSize: "15px", fontWeight: "600", marginBottom: "12px" });
    box.appendChild(h);
    const presets = STORE.presets || {};
    const keys = Object.keys(presets).sort();
    if (!keys.length) { const e = document.createElement("div"); e.textContent = "(none saved)"; e.style.opacity = "0.6"; box.appendChild(e); }
    for (const nm of keys) {
      const r = document.createElement("div");
      r.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;padding:4px 0;border-bottom:1px solid #333";
      const icon = presets[nm].type === "entity" ? "📦" : "👤";
      const s = document.createElement("span"); s.textContent = `${icon} ${nm}`;
      s.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
      const del = document.createElement("button"); del.textContent = "Delete";
      Object.assign(del.style, { background: "#333", color: "#eee", border: "1px solid #555", borderRadius: "4px", padding: "4px 10px", cursor: "pointer" });
      del.onclick = async () => { try { await postJSON("/kinburg/cards/save", { name: nm, delete: true }); refreshReaders(); render(); } catch (e) { alert(e.message); } };
      r.appendChild(s); r.appendChild(del); box.appendChild(r);
    }
    const f = document.createElement("div"); f.style.cssText = "display:flex;justify-content:flex-end;margin-top:14px";
    const cl = document.createElement("button"); cl.textContent = "Close";
    Object.assign(cl.style, { background: "#3b82f6", color: "#fff", border: "1px solid #555", borderRadius: "4px", padding: "6px 12px", cursor: "pointer" });
    cl.onclick = close; f.appendChild(cl); box.appendChild(f);
  };
  render();
  overlay.appendChild(box);
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
  document.body.appendChild(overlay);
}

app.registerExtension({
  name: "Kinburg.CardPresets",
  async setup() { await refreshStore(); refreshReaders(); },
  async nodeCreated(node) {
    if (node.comfyClass !== "CardPresets" && node.type !== "CardPresets") return;
    node.addWidget("button", "🔄 Refresh", null, async () => { await refreshStore(); refreshReaders(); }, { serialize: false });
    node.addWidget("button", "🗑 Manage", null, () => manageDialog(), { serialize: false });
    if (!Object.keys(STORE.presets || {}).length) refreshStore().then(() => syncReader(node));
    syncReader(node);
  },
});
