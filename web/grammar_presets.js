import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Prompt Presets — a dropdown over a library of named prompts, each with an optional GBNF grammar.
// The library lives on the backend (grammar_presets/store.py) and is served live, so a preset saved
// here shows up in every Prompt Presets node without an object_info reload. The node keeps its old
// GrammarPresets class id so saved workflows still load.
//
// Everything is edited in the 🗂 Manage dialog — a plain DOM overlay, never window.prompt (the
// desktop app forbids it). Built-in grammars are read-only: saving one under its own name writes an
// edited copy over it, and ↺ drops the copy to bring the original back.

const CLASS = "GrammarPresets";
const NONE = "🚫 None";

let STORE = { none: NONE, order: [NONE], presets: {} };

async function refreshStore() {
  try {
    const r = await api.fetchApi("/kinburg/grammars/data");
    const j = await r.json();
    if (j && j.ok) STORE = j;
  } catch (e) { console.error("[Kinburg] prompt presets: load failed", e); }
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

// One preset's texts. Kept off /data: the nodes refresh that and have no use for a whole library.
async function fetchPreset(name) {
  const r = await api.fetchApi(`/kinburg/grammars/preset?name=${encodeURIComponent(name)}`);
  const j = await r.json();
  if (!j || !j.ok) throw new Error(j?.error || `preset '${name}' could not be read`);
  return j;
}

// ---------------------------------------------------------------------------------------- nodes
const wv = (node, name) => node.widgets?.find((w) => w.name === name);
const isPresets = (node) => node.comfyClass === CLASS || node.type === CLASS;

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
  for (const n of app.graph?._nodes || []) if (isPresets(n)) syncNode(n);
}

// A node holds its preset by NAME, so a rename that didn't follow the graph would leave it failing
// on the next run with "no preset named …".
function renameInGraph(oldName, newName) {
  let moved = 0;
  for (const n of app.graph?._nodes || []) {
    if (!isPresets(n)) continue;
    const w = wv(n, "preset");
    if (w && w.value === oldName) { w.value = newName; moved++; }
    syncNode(n);
  }
  return moved;
}

// ---------------------------------------------------------------------------------- DOM helpers
const css = (el, s) => Object.assign(el.style, s);
const mk = (tag, style, text) => {
  const e = document.createElement(tag);
  if (style) e.style.cssText = style;
  if (text != null) e.textContent = text;
  return e;
};
const BTN = "background:#333;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 10px;cursor:pointer;flex:0 0 auto";
const ICON = "background:#333;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 7px;cursor:pointer;flex:0 0 auto";
const DANGER = "background:#3a2727;color:#eee;border:1px solid #663;border-radius:4px;padding:4px 7px;cursor:pointer;flex:0 0 auto";
const PRIMARY = "background:#3b82f6;color:#fff;border:1px solid #555;border-radius:4px;padding:6px 12px;cursor:pointer;flex:0 0 auto";
const INPUT = "background:#1a1a1a;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 8px;min-width:0";
const TEXT = INPUT + ";font:12px/1.45 monospace;resize:vertical;width:100%;box-sizing:border-box";
const chip = (on) => `font-size:11px;padding:1px 6px;border-radius:8px;flex:0 0 auto;border:1px solid ${on ? "#3d6b3d" : "#383838"};color:${on ? "#9fd89f" : "#555"}`;

function flash(btn, txt, ok = true) {
  const o = btn.textContent;
  btn.textContent = txt;
  btn.style.color = ok ? "#7CFC7C" : "#ff6b6b";
  setTimeout(() => { btn.textContent = o; btn.style.color = "#eee"; }, 1400);
}

// Two-click delete: the first click arms the button for 3s. A system prompt can be a page of text
// that nothing else holds a copy of — the library is not in the workflow.
function arm(btn, label, run) {
  const idle = btn.textContent;
  btn.onclick = () => {
    if (btn.dataset.armed !== "1") {
      btn.dataset.armed = "1";
      btn.textContent = label;
      setTimeout(() => {
        if (btn.dataset.armed === "1") { btn.dataset.armed = "0"; btn.textContent = idle; }
      }, 3000);
      return;
    }
    btn.dataset.armed = "0";
    run();
  };
}

function modalShell(width = "640px", z = 10000) {
  const overlay = mk("div");
  css(overlay, { position: "fixed", inset: "0", background: "rgba(0,0,0,0.55)", zIndex: z,
    display: "flex", alignItems: "center", justifyContent: "center" });
  const box = mk("div");
  css(box, { background: "#222", color: "#eee", border: "1px solid #444", borderRadius: "8px",
    padding: "16px", minWidth: width, maxWidth: "900px", maxHeight: "88vh", overflow: "auto",
    font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" });
  const close = () => overlay.remove();
  overlay.appendChild(box);
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
  document.body.appendChild(overlay);
  return { overlay, box, close };
}

// --------------------------------------------------------------------------------- pure helpers
// Rows the library lists, in dropdown order (built-ins, then yours A→Z), narrowed by the search
// box. The preset the node that opened the dialog has picked goes first.
function libraryRows(query, focus) {
  const q = String(query || "").trim().toLowerCase();
  const none = STORE.none || NONE;
  const rows = (STORE.order || []).filter((nm) => nm !== none && (!q || nm.toLowerCase().includes(q)));
  const i = rows.indexOf(focus);
  if (i > 0) rows.unshift(rows.splice(i, 1)[0]);
  return rows;
}

// "Writer" -> "Writer copy" -> "Writer copy 2" … never an overwrite by accident.
function copyName(base) {
  const taken = new Set(Object.keys(STORE.presets || {}));
  let name = `${base} copy`;
  for (let i = 2; taken.has(name); i++) name = `${base} copy ${i}`;
  return name;
}

// The body a Save click posts. Only your own preset is renamed: a built-in saved under a new name
// is a new preset beside it, and an old_name there would drop its edited copy — a silent reset.
function editorPayload(entry, { key, prompt, grammar }) {
  const body = { name: String(key || "").trim(), prompt: prompt || "", grammar: grammar || "" };
  if (!entry.isNew && !entry.builtin && entry.key && entry.key !== body.name) body.old_name = entry.key;
  return body;
}

// The store overwrites whatever is filed under a name, so creating or renaming onto a name the
// library already holds would replace that preset without a word. The one check only the dialog
// can make — it knows what the user meant.
const nameTaken = (entry, name) =>
  (entry.isNew || name !== entry.key) && Object.prototype.hasOwnProperty.call(STORE.presets || {}, name);

// ------------------------------------------------------------------------------------- editor
// `entry` = {key, prompt, grammar, builtin, isNew}. `afterSave(name)` runs on a successful save, so
// the library behind it can re-render on the preset that was just written.
function presetEditor(entry, afterSave) {
  const { box, close } = modalShell("720px", 10001);

  box.appendChild(mk("div", "font-size:15px;font-weight:600;margin-bottom:2px",
    entry.isNew ? "➕ New preset" : `✏️ Edit preset — ${entry.key}`));
  box.appendChild(mk("div", "opacity:0.6;margin-bottom:12px;font-size:12px", entry.builtin
    ? "Built-in. Saving keeps your edit over it (↺ in the library restores the original); a new name saves a copy beside it."
    : "Either field may stay empty — it then emits \"\", which the Local LLM node reads as \"not set\"."));

  const keyRow = mk("div", "display:flex;align-items:center;gap:8px;margin-bottom:10px");
  const keyLabel = mk("div", "flex:0 0 auto;opacity:0.8", "Saved as");
  keyLabel.title = "The name in the Prompt Presets dropdown. Rename it and every node picking it follows.";
  keyRow.appendChild(keyLabel);
  const keyInput = mk("input", INPUT + ";flex:1");
  keyInput.value = entry.key || "";
  keyInput.placeholder = "library name…";
  keyRow.appendChild(keyInput);
  box.appendChild(keyRow);

  const area = (label, hint, value, placeholder) => {
    const row = mk("div", "display:flex;flex-direction:column;gap:4px;margin-bottom:10px");
    const l = mk("div", "opacity:0.8", label);
    l.title = hint;
    row.appendChild(l);
    const ta = mk("textarea", TEXT);
    ta.rows = 12;
    ta.value = value || "";
    ta.placeholder = placeholder;
    row.appendChild(ta);
    box.appendChild(row);
    return ta;
  };
  const promptArea = area("Prompt", "The `prompt` output — e.g. a system prompt for a Local LLM node's system_override.",
    entry.prompt, "system prompt, or any prompt worth keeping…");
  const grammarArea = area("GBNF grammar", "The `grammar` output — wire into a Local LLM node's grammar_override.",
    entry.grammar, "root ::= …");

  const foot = mk("div", "display:flex;justify-content:flex-end;gap:8px;margin-top:14px");
  const cancel = mk("button", BTN, "Cancel");
  cancel.onclick = close;
  const save = mk("button", PRIMARY, entry.isNew ? "Create" : "Save");
  save.onclick = async () => {
    const body = editorPayload(entry, { key: keyInput.value, prompt: promptArea.value, grammar: grammarArea.value });
    if (nameTaken(entry, body.name)) { flash(save, "✕ name taken", false); return; }
    try {
      await postJSON("/kinburg/grammars/preset", body);
      if (body.old_name) renameInGraph(body.old_name, body.name);
      refreshAllNodes();
      close();
      afterSave?.(body.name);
    } catch (e) { flash(save, "✕ " + e.message, false); }
  };
  foot.appendChild(cancel);
  foot.appendChild(save);
  box.appendChild(foot);
}

// ------------------------------------------------------------------------------- library dialog
async function libraryDialog(focusName) {
  await refreshStore();
  const { box, close } = modalShell("640px");
  let query = "";

  const openEditor = async (name, duplicate) => {
    try {
      const p = await fetchPreset(name);
      presetEditor({
        key: duplicate ? copyName(name) : name,
        prompt: p.prompt,
        grammar: p.grammar,
        builtin: !duplicate && !!(STORE.presets[name] || {}).builtin,
        isNew: !!duplicate,
      }, (savedAs) => { if (!duplicate && focusName === name) focusName = savedAs; render(); });
    } catch (e) { alert(e.message); }
  };

  const render = () => {
    box.innerHTML = "";
    box.appendChild(mk("div", "font-size:15px;font-weight:600;margin-bottom:2px", "🗂 Prompt library"));
    box.appendChild(mk("div", "opacity:0.6;margin-bottom:12px;font-size:12px",
      "A prompt, a GBNF grammar, or the pair that belongs together. Stored on disk, not in the workflow."));

    const bar = mk("div", "display:flex;align-items:center;gap:8px;margin-bottom:10px");
    const search = mk("input", INPUT + ";flex:1");
    search.value = query;
    search.placeholder = "search…";
    search.addEventListener("input", () => { query = search.value; renderRows(); });
    bar.appendChild(search);
    const add = mk("button", BTN, "➕ New");
    add.onclick = () => presetEditor({ key: "", prompt: "", grammar: "", builtin: false, isNew: true },
      (savedAs) => { focusName = savedAs; render(); });
    bar.appendChild(add);
    box.appendChild(bar);

    const list = mk("div");
    box.appendChild(list);

    const renderRows = () => {
      list.innerHTML = "";
      const rows = libraryRows(query, focusName);
      if (!rows.length) { list.appendChild(mk("div", "opacity:0.6;padding:8px 0", "(nothing matches)")); return; }
      for (const nm of rows) {
        const p = STORE.presets[nm] || {};
        const row = mk("div", "display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #333");
        if (nm === focusName) css(row, { borderLeft: "2px solid #3b82f6", paddingLeft: "6px", marginLeft: "-8px" });

        const s = mk("span", "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap",
          `${p.builtin ? "🔒 " : ""}${nm}${p.edited ? "  ✎ edited" : ""}`);
        s.title = p.builtin ? (p.edited ? "Built-in, with your edit over it" : "Built-in") : nm;
        row.appendChild(s);
        row.appendChild(mk("span", chip(p.prompt), "prompt"));
        row.appendChild(mk("span", chip(p.grammar), "grammar"));

        const edit = mk("button", ICON, "✏️");
        edit.title = "Edit";
        edit.onclick = () => openEditor(nm, false);
        row.appendChild(edit);

        const dup = mk("button", ICON, "⧉");
        dup.title = "Duplicate — opens a copy to rename and edit";
        dup.onclick = () => openEditor(nm, true);
        row.appendChild(dup);

        // A pristine built-in has nothing to delete; an edited one can only go back to the original.
        if (!p.builtin || p.edited) {
          const del = mk("button", DANGER, p.builtin ? "↺" : "🗑");
          del.title = p.builtin ? "Drop your edit and restore the built-in" : "Delete this preset";
          arm(del, p.builtin ? "reset?" : "sure?", async () => {
            try {
              await postJSON("/kinburg/grammars/preset", { name: nm, delete: true });
              refreshAllNodes();
              render();
            } catch (e) { alert(e.message); }
          });
          row.appendChild(del);
        }
        list.appendChild(row);
      }
    };
    renderRows();

    const foot = mk("div", "display:flex;justify-content:flex-end;margin-top:14px");
    const cl = mk("button", PRIMARY, "Close");
    cl.onclick = close;
    foot.appendChild(cl);
    box.appendChild(foot);
  };
  render();
}

app.registerExtension({
  name: "Kinburg.GrammarPresets",
  async setup() { await refreshStore(); refreshAllNodes(); },
  async nodeCreated(node) {
    if (!isPresets(node)) return;
    // Opens on the preset THIS node has picked.
    node.addWidget("button", "🗂 Manage", null, () => {
      const cur = wv(node, "preset")?.value;
      libraryDialog(cur && cur !== (STORE.none || NONE) ? cur : null);
    }, { serialize: false });
    if (!Object.keys(STORE.presets || {}).length) refreshStore().then(() => syncNode(node));
    syncNode(node);
    node.setSize?.(node.computeSize?.());
  },
});
