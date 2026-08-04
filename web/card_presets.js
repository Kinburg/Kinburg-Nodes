import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Card Presets reader — a dropdown of saved Character / Entity cards, with an optional tag filter.
// Cards are SAVED on the backend: use Card Save (parses an LLM JSON card), or fill a Character /
// Entity Card, type a name in its `save_preset_as` field (+ optional `tags`), and run. This node's
// dropdown is served live from the backend store; the `filter` widget narrows it to one tag.

const NONE = "🚫 None";
const ALL_TAGS = "🏷 All";
let STORE = { none: NONE, all_tags: ALL_TAGS, order: [NONE], tags: [], presets: {} };

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

// Preset names available under a given tag filter (NONE always first).
function presetsForTag(tag) {
  const none = STORE.none || NONE;
  const allTag = STORE.all_tags || ALL_TAGS;
  if (!tag || tag === allTag) return [...(STORE.order || [none])];
  const lc = tag.toLowerCase();
  const names = Object.keys(STORE.presets || {})
    .filter((nm) => (STORE.presets[nm].tags || []).some((t) => String(t).toLowerCase() === lc))
    .sort();
  return [none, ...names];
}

function syncReader(node) {
  const allTag = STORE.all_tags || ALL_TAGS;
  const fw = wv(node, "filter");
  if (fw) {
    const tagOpts = [allTag, ...(STORE.tags || [])];
    if (fw.value && !tagOpts.includes(fw.value)) tagOpts.push(fw.value); // keep a now-gone tag selectable
    if (!fw.options) fw.options = {};
    fw.options.values = tagOpts;
    if (!tagOpts.includes(fw.value)) fw.value = allTag;
  }
  const pw = wv(node, "preset");
  if (pw) {
    const tag = fw ? fw.value : allTag;
    const opts = presetsForTag(tag);
    if (pw.value && !opts.includes(pw.value)) opts.push(pw.value); // never drop the current selection
    if (!pw.options) pw.options = {};
    pw.options.values = opts;
  }
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
    borderRadius: "8px", padding: "16px", minWidth: "420px", maxWidth: "620px", maxHeight: "80vh",
    overflow: "auto", font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" });
  const close = () => overlay.remove();
  const render = () => {
    box.innerHTML = "";
    const h = document.createElement("div"); h.textContent = "🗑 Manage saved cards";
    Object.assign(h.style, { fontSize: "15px", fontWeight: "600", marginBottom: "4px" });
    box.appendChild(h);
    const sub = document.createElement("div");
    sub.textContent = "Edit tags (comma-separated) to filter the library in Card Presets.";
    Object.assign(sub.style, { opacity: "0.6", marginBottom: "12px", fontSize: "12px" });
    box.appendChild(sub);

    const presets = STORE.presets || {};
    const keys = Object.keys(presets).sort();
    if (!keys.length) { const e = document.createElement("div"); e.textContent = "(none saved)"; e.style.opacity = "0.6"; box.appendChild(e); }
    for (const nm of keys) {
      const row = document.createElement("div");
      row.style.cssText = "display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #333";
      const icon = presets[nm].type === "entity" ? "📦" : "👤";
      const s = document.createElement("span"); s.textContent = `${icon} ${nm}`;
      s.style.cssText = "flex:0 0 34%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
      s.title = nm;

      const tagInput = document.createElement("input"); tagInput.type = "text";
      tagInput.value = (presets[nm].tags || []).join(", ");
      tagInput.placeholder = "tags…";
      tagInput.style.cssText = "flex:1;min-width:0;background:#1a1a1a;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 8px";

      const save = document.createElement("button"); save.textContent = "Save tags";
      Object.assign(save.style, { background: "#333", color: "#eee", border: "1px solid #555", borderRadius: "4px", padding: "4px 10px", cursor: "pointer", flex: "0 0 auto" });
      const flash = (btn, txt, ok = true) => { const o = btn.textContent; btn.textContent = txt; btn.style.color = ok ? "#7CFC7C" : "#ff6b6b"; setTimeout(() => { btn.textContent = o; btn.style.color = "#eee"; }, 1200); };
      const doSave = async () => { try { await postJSON("/kinburg/cards/tags", { name: nm, tags: tagInput.value }); refreshReaders(); flash(save, "✓ saved"); } catch (e) { flash(save, "✕ " + e.message, false); } };
      save.onclick = doSave;
      tagInput.addEventListener("keydown", (e) => { if (e.key === "Enter") doSave(); });

      const del = document.createElement("button"); del.textContent = "Delete";
      Object.assign(del.style, { background: "#3a2727", color: "#eee", border: "1px solid #663", borderRadius: "4px", padding: "4px 10px", cursor: "pointer", flex: "0 0 auto" });
      del.onclick = async () => { try { await postJSON("/kinburg/cards/save", { name: nm, delete: true }); refreshReaders(); render(); } catch (e) { alert(e.message); } };

      row.appendChild(s); row.appendChild(tagInput); row.appendChild(save); row.appendChild(del);
      box.appendChild(row);
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
    // Re-narrow the preset list whenever the tag filter changes.
    const fw = wv(node, "filter");
    if (fw) {
      const orig = fw.callback;
      fw.callback = function () { const r = orig ? orig.apply(this, arguments) : undefined; syncReader(node); return r; };
    }
    node.addWidget("button", "🔄 Refresh", null, async () => { await refreshStore(); refreshReaders(); }, { serialize: false });
    node.addWidget("button", "🗑 Manage", null, () => manageDialog(), { serialize: false });
    if (!Object.keys(STORE.presets || {}).length) refreshStore().then(() => syncReader(node));
    syncReader(node);
  },
});
