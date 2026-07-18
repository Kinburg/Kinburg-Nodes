import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Prompt Presets — five FLEXIBLE slots. Each slot has two dropdowns: a category selector
// (cat_i) and a preset selector (preset_i, the presets of the chosen category). Categories and
// presets are served live from the backend store, so edits appear without an object_info
// reload. Buttons add presets, manage categories, save the current slots as a "setup", and
// manage setups/custom presets. Output slot labels follow each slot's chosen category.

const CLASS = "PromptPresets";
const NONE = "🚫 None";

// One shared cache of the store; refreshed after every edit and reused by all node instances.
let STORE = { none: NONE, order: ["camera", "aesthetics", "light", "medium", "background"],
              categories: {}, builtins: {}, builtin_cats: [], setups: {}, n_slots: 5 };

const N = () => STORE.n_slots || 5;
const catList = () => STORE.order || [];
const presetOptions = (cat) => [NONE, ...Object.keys(STORE.categories?.[cat] || {})];
const setupNames = () => Object.keys(STORE.setups || {}).sort();
const isBuiltinCat = (c) => (STORE.builtin_cats || []).includes(c);
const displayCat = (c) => (c ? c.charAt(0).toUpperCase() + c.slice(1) : c);

const wv = (node, name) => node.widgets?.find((w) => w.name === name);
const catW = (node, i) => wv(node, `cat_${i}`);
const presetW = (node, i) => wv(node, `preset_${i}`);

async function refreshStore() {
  try {
    const r = await api.fetchApi("/kinburg/presets/data");
    const j = await r.json();
    if (j && j.ok) STORE = j;
  } catch (e) {
    console.error("[Kinburg] presets: load failed", e);
  }
  return STORE;
}

async function postJSON(path, body) {
  const r = await api.fetchApi(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!j || !j.ok) throw new Error(j?.error || `request to ${path} failed`);
  STORE = j; // every mutating route returns the fresh full store
  return j;
}

// Set the preset dropdown's options from its slot's current category. Keep an unknown selection
// (e.g. a preset just deleted) visible instead of dropping it.
function syncPreset(node, i) {
  const cw = catW(node, i), pw = presetW(node, i);
  if (!pw) return;
  const opts = presetOptions(cw?.value);
  if (pw.value && !opts.includes(pw.value)) opts.push(pw.value);
  if (!pw.options) pw.options = {};
  pw.options.values = opts;
}

// The output slot label follows the slot's chosen category.
function syncOutputLabel(node, i) {
  const out = node.outputs?.[i - 1];
  if (out) out.name = displayCat(catW(node, i)?.value) || `slot_${i}`;
}

// A slot's category changed: reset the preset if it no longer exists in the new category.
function onCatChanged(node, i) {
  const cw = catW(node, i), pw = presetW(node, i);
  if (pw) {
    const opts = presetOptions(cw?.value);
    if (!opts.includes(pw.value)) pw.value = NONE;
  }
  syncPreset(node, i);
  syncOutputLabel(node, i);
  node.setDirtyCanvas(true, true);
}

// Push the whole store into one node's dropdowns + output labels.
function syncNode(node) {
  for (let i = 1; i <= N(); i++) {
    const cw = catW(node, i);
    if (cw) {
      const opts = [...catList()];
      if (cw.value && !opts.includes(cw.value)) opts.push(cw.value); // keep a deleted cat visible
      if (!cw.options) cw.options = {};
      cw.options.values = opts;
    }
    syncPreset(node, i);
    syncOutputLabel(node, i);
  }
  const s = wv(node, "⚙ setup");
  if (s) {
    if (!s.options) s.options = {};
    s.options.values = ["—", ...setupNames()];
    if (!s.options.values.includes(s.value)) s.value = "—";
  }
  node.setDirtyCanvas(true, true);
}

function refreshAllNodes() {
  for (const n of app.graph?._nodes || []) {
    if (n.comfyClass === CLASS || n.type === CLASS) syncNode(n);
  }
}

// Apply a saved setup (a per-slot list of {cat, preset}) onto this node's slots.
function applySetup(node, name) {
  const setup = STORE.setups?.[name];
  if (!Array.isArray(setup)) return;
  for (let i = 1; i <= N(); i++) {
    const slot = setup[i - 1];
    if (!slot) continue;
    const cw = catW(node, i), pw = presetW(node, i);
    if (cw && catList().includes(slot.cat)) cw.value = slot.cat;
    if (pw) {
      const opts = presetOptions(cw?.value);
      pw.value = opts.includes(slot.preset) ? slot.preset : NONE;
    }
    syncPreset(node, i);
    syncOutputLabel(node, i);
  }
  node.setDirtyCanvas(true, true);
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
    padding: "16px", minWidth: "360px", maxWidth: "560px", maxHeight: "80vh", overflow: "auto",
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
    padding: "6px 8px", font: "13px sans-serif", width: "100%", boxSizing: "border-box",
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

function catSelect() {
  const sel = styledInput(document.createElement("select"));
  catList().forEach((c) => { const o = document.createElement("option"); o.value = c; o.textContent = displayCat(c); sel.appendChild(o); });
  return sel;
}

// ---------------------------------------------------------------- dialogs
function addPresetDialog(node) {
  modal("➕ Add / edit preset", (box, close) => {
    const catRow = row(box, "Category");
    const sel = catSelect();
    catRow.appendChild(sel);

    const nameRow = row(box, "Preset name");
    const name = styledInput(document.createElement("input"));
    name.placeholder = "e.g. Top Light";
    nameRow.appendChild(name);

    const textRow = row(box, "Prompt fragment");
    const text = styledInput(document.createElement("textarea"));
    text.rows = 4; text.placeholder = "text added to your prompt when this preset is chosen";
    textRow.appendChild(text);

    const prefill = () => {
      const t = STORE.categories?.[sel.value]?.[name.value.trim()];
      if (t != null) text.value = t;
    };
    name.addEventListener("blur", prefill);
    sel.addEventListener("change", prefill);

    footer(box,
      button("Cancel", close),
      button("Save", async () => {
        try {
          await postJSON("/kinburg/presets/preset", { category: sel.value, name: name.value, text: text.value });
          refreshAllNodes();
          close();
        } catch (e) { alert(e.message); }
      }, true),
    );
  });
}

function categoriesDialog(node) {
  const render = () => modal("🗂 Manage categories", (box, close) => {
    const addRow = document.createElement("div");
    addRow.style.cssText = "display:flex;gap:6px;margin-bottom:12px";
    const inp = styledInput(document.createElement("input"));
    inp.placeholder = "new category name";
    addRow.appendChild(inp);
    addRow.appendChild(button("Add", async () => {
      try { await postJSON("/kinburg/presets/category", { action: "add", name: inp.value }); refreshAllNodes(); close(); render(); }
      catch (e) { alert(e.message); }
    }, true));
    box.appendChild(addRow);

    catList().forEach((c) => {
      const r = document.createElement("div");
      r.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;padding:4px 0;border-bottom:1px solid #333";
      const s = document.createElement("span");
      s.textContent = isBuiltinCat(c) ? `🔒 ${displayCat(c)} (built-in)` : displayCat(c);
      s.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
      r.appendChild(s);
      if (!isBuiltinCat(c)) {
        r.appendChild(button("Rename", async () => {
          const nn = window.prompt(`Rename category "${c}" to:`, c);
          if (nn == null || !nn.trim() || nn.trim() === c) return;
          try { await postJSON("/kinburg/presets/category", { action: "rename", name: c, new_name: nn.trim() }); refreshAllNodes(); close(); render(); }
          catch (e) { alert(e.message); }
        }));
        r.appendChild(button("Delete", async () => {
          if (!window.confirm(`Delete category "${c}" and its custom presets?`)) return;
          try { await postJSON("/kinburg/presets/category", { action: "delete", name: c }); refreshAllNodes(); close(); render(); }
          catch (e) { alert(e.message); }
        }));
      }
      box.appendChild(r);
    });

    footer(box, button("Close", close, true));
  });
  render();
}

function saveSetupDialog(node) {
  modal("💾 Save current slots as a setup", (box, close) => {
    const nameRow = row(box, "Setup name");
    const name = styledInput(document.createElement("input"));
    name.placeholder = "e.g. Cinematic portrait";
    nameRow.appendChild(name);

    const slots = [];
    for (let i = 1; i <= N(); i++) slots.push({ cat: catW(node, i)?.value ?? "", preset: presetW(node, i)?.value ?? NONE });

    const preview = document.createElement("div");
    preview.style.cssText = "opacity:0.75;font-size:12px;line-height:1.6;margin-top:4px";
    preview.innerHTML = slots.map((s) => `<b>${displayCat(s.cat)}:</b> ${s.preset}`).join("<br>");
    box.appendChild(preview);

    footer(box,
      button("Cancel", close),
      button("Save", async () => {
        try {
          await postJSON("/kinburg/presets/setup", { name: name.value, slots });
          const s = wv(node, "⚙ setup"); if (s) s.value = name.value.trim();
          refreshAllNodes();
          close();
        } catch (e) { alert(e.message); }
      }, true),
    );
  });
}

function manageDialog(node) {
  const render = () => modal("🗑 Manage setups & custom presets", (box, close) => {
    const section = (title) => { const h = document.createElement("div"); h.textContent = title; h.style.cssText = "font-weight:600;margin:10px 0 6px"; box.appendChild(h); };
    const line = (label, onDel) => {
      const r = document.createElement("div");
      r.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;padding:4px 0;border-bottom:1px solid #333";
      const s = document.createElement("span"); s.textContent = label; s.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
      r.appendChild(s);
      r.appendChild(button("Delete", onDel));
      box.appendChild(r);
    };

    section("Setups");
    const setups = setupNames();
    if (!setups.length) { const e = document.createElement("div"); e.textContent = "(none saved)"; e.style.opacity = "0.6"; box.appendChild(e); }
    setups.forEach((nm) => line(nm, async () => {
      try { await postJSON("/kinburg/presets/setup", { name: nm, delete: true }); refreshAllNodes(); close(); render(); } catch (e) { alert(e.message); }
    }));

    section("Custom presets");
    let any = false;
    for (const cat of catList()) {
      const builtins = new Set(STORE.builtins?.[cat] || []);
      for (const nm of Object.keys(STORE.categories?.[cat] || {})) {
        if (builtins.has(nm)) continue; // built-ins can't be deleted
        any = true;
        line(`${displayCat(cat)} — ${nm}`, async () => {
          try { await postJSON("/kinburg/presets/preset", { category: cat, name: nm, delete: true }); refreshAllNodes(); close(); render(); } catch (e) { alert(e.message); }
        });
      }
    }
    if (!any) { const e = document.createElement("div"); e.textContent = "(none added)"; e.style.opacity = "0.6"; box.appendChild(e); }

    footer(box, button("Close", close, true));
  });
  render();
}

// ---------------------------------------------------------------- node wiring
app.registerExtension({
  name: "Kinburg.PromptPresets",
  async setup() {
    await refreshStore();
    refreshAllNodes();
  },
  async nodeCreated(node) {
    if (node.comfyClass !== CLASS && node.type !== CLASS) return;

    // Frontend-only "setup" selector at the very top. serialize:false keeps it out of the
    // prompt/workflow (it only drives the slot dropdowns).
    const setupW = node.addWidget("combo", "⚙ setup", "—",
      (v) => { if (v && v !== "—" && !app.configuringGraph) applySetup(node, v); },
      { values: ["—"], serialize: false });
    setupW.serialize = false;
    const i = node.widgets.indexOf(setupW);
    if (i > 0) { node.widgets.splice(i, 1); node.widgets.unshift(setupW); }

    node.addWidget("button", "➕ Add preset", null, () => addPresetDialog(node), { serialize: false });
    node.addWidget("button", "🗂 Categories", null, () => categoriesDialog(node), { serialize: false });
    node.addWidget("button", "💾 Save setup", null, () => saveSetupDialog(node), { serialize: false });
    node.addWidget("button", "🗑 Manage", null, () => manageDialog(node), { serialize: false });

    // Wrap each category dropdown's callback so picking a category refreshes its preset list
    // and the output label.
    for (let s = 1; s <= N(); s++) {
      const cw = catW(node, s);
      if (!cw) continue;
      const orig = cw.callback;
      cw.callback = function () { const r = orig?.apply(this, arguments); onCatChanged(node, s); return r; };
    }

    // Re-sync after a saved workflow restores widget values (nodeCreated runs before that).
    const origConfigure = node.onConfigure;
    node.onConfigure = function () { const r = origConfigure?.apply(this, arguments); syncNode(this); return r; };

    if (!Object.keys(STORE.categories || {}).length) refreshStore().then(() => syncNode(node));
    syncNode(node);

    node.setSize?.(node.computeSize?.());
  },
});
