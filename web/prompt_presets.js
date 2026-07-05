import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Prompt Presets — five preset dropdowns (camera/aesthetics/light/medium/background).
// Dropdown values are served live from the backend store, so presets the user adds appear
// without an object_info reload. A frontend-only "setup" selector applies a saved
// combination; buttons add presets, save the current selection as a setup, and manage both.

const CLASS = "PromptPresets";
const NONE = "🚫 None";

// One shared cache of the store; refreshed after every edit and reused by all node instances.
let STORE = { none: NONE, order: ["camera", "aesthetics", "light", "medium", "background"],
              categories: {}, builtins: {}, setups: {} };
const _label = { camera: "Camera", aesthetics: "Aesthetics", light: "Light", medium: "Medium", background: "Background" };

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

const catOptions = (cat) => [NONE, ...Object.keys(STORE.categories?.[cat] || {})];
const setupNames = () => Object.keys(STORE.setups || {}).sort();

const wv = (node, name) => node.widgets?.find((w) => w.name === name);

// Push the current store into one node's dropdowns. We overwrite `options.values` with a
// fresh array (rather than relying on a getter) so added/removed presets show up immediately,
// regardless of how the frontend reads combo values.
function syncNode(node) {
  for (const cat of STORE.order) {
    const w = wv(node, cat);
    if (!w) continue;
    const opts = catOptions(cat);
    // Keep an unknown selection (e.g. a preset just deleted) visible instead of dropping it.
    if (w.value && !opts.includes(w.value)) opts.push(w.value);
    if (!w.options) w.options = {};
    w.options.values = opts;
  }
  const s = wv(node, "⚙ setup");
  if (s) {
    if (!s.options) s.options = {};
    s.options.values = ["—", ...setupNames()];
    if (!s.options.values.includes(s.value)) s.value = "—";
  }
  node.setDirtyCanvas(true, true);
}

// Re-sync every PromptPresets node in the graph (after any store edit, here or elsewhere).
function refreshAllNodes() {
  for (const n of app.graph?._nodes || []) {
    if (n.comfyClass === CLASS || n.type === CLASS) syncNode(n);
  }
}

// Apply a saved setup's preset names onto this node's category dropdowns.
function applySetup(node, name) {
  const setup = STORE.setups?.[name];
  if (!setup) return;
  for (const cat of STORE.order) {
    const w = wv(node, cat);
    if (!w) continue;
    const want = setup[cat] || NONE;
    w.value = catOptions(cat).includes(want) ? want : NONE;
    w.callback?.(w.value);
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

// ---------------------------------------------------------------- dialogs
function addPresetDialog(node) {
  modal("➕ Add / edit preset", (box, close) => {
    const catRow = row(box, "Category");
    const sel = styledInput(document.createElement("select"));
    STORE.order.forEach((c) => { const o = document.createElement("option"); o.value = c; o.textContent = _label[c] || c; sel.appendChild(o); });
    catRow.appendChild(sel);

    const nameRow = row(box, "Preset name");
    const name = styledInput(document.createElement("input"));
    name.placeholder = "e.g. Top Light";
    nameRow.appendChild(name);

    const textRow = row(box, "Prompt fragment");
    const text = styledInput(document.createElement("textarea"));
    text.rows = 4; text.placeholder = "text added to your prompt when this preset is chosen";
    textRow.appendChild(text);

    // Pre-fill when an existing name is typed for the chosen category.
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

function saveSetupDialog(node) {
  modal("💾 Save current selection as a setup", (box, close) => {
    const nameRow = row(box, "Setup name");
    const name = styledInput(document.createElement("input"));
    name.placeholder = "e.g. Cinematic portrait";
    nameRow.appendChild(name);

    const preview = document.createElement("div");
    preview.style.cssText = "opacity:0.75;font-size:12px;line-height:1.6;margin-top:4px";
    preview.innerHTML = STORE.order.map((c) => `<b>${_label[c]}:</b> ${wv(node, c)?.value ?? NONE}`).join("<br>");
    box.appendChild(preview);

    footer(box,
      button("Cancel", close),
      button("Save", async () => {
        const values = {};
        for (const c of STORE.order) values[c] = wv(node, c)?.value ?? NONE;
        try {
          await postJSON("/kinburg/presets/setup", { name: name.value, values });
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
    for (const cat of STORE.order) {
      const builtins = new Set(STORE.builtins?.[cat] || []);
      for (const nm of Object.keys(STORE.categories?.[cat] || {})) {
        if (builtins.has(nm)) continue; // built-ins can't be deleted
        any = true;
        line(`${_label[cat]} — ${nm}`, async () => {
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

    // Frontend-only "setup" selector, placed at the very top. serialize:false keeps it out
    // of the prompt/workflow (it only drives the category dropdowns).
    const setupW = node.addWidget("combo", "⚙ setup", "—",
      (v) => { if (v && v !== "—" && !app.configuringGraph) applySetup(node, v); },
      { values: ["—"], serialize: false });
    setupW.serialize = false;
    const i = node.widgets.indexOf(setupW);
    if (i > 0) { node.widgets.splice(i, 1); node.widgets.unshift(setupW); }

    node.addWidget("button", "➕ Add preset", null, () => addPresetDialog(node), { serialize: false });
    node.addWidget("button", "💾 Save setup", null, () => saveSetupDialog(node), { serialize: false });
    node.addWidget("button", "🗑 Manage", null, () => manageDialog(node), { serialize: false });

    // Populate the dropdowns from the cache; fetch first only if it hasn't loaded yet.
    if (!Object.keys(STORE.categories || {}).length) refreshStore().then(() => syncNode(node));
    syncNode(node);

    node.setSize?.(node.computeSize?.());
  },
});
