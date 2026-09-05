import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Card Presets reader — a dropdown of saved Character / Entity cards, with an optional tag filter.
// Cards are SAVED on the backend: use Card Save (parses an LLM JSON card), fill a Character /
// Entity Card and type a name in its `save_preset_as` field, or write one by hand in the library
// dialog below. This node's dropdown is served live from the backend store; the `filter` widget
// narrows it to one tag.
//
// The library dialog is a plain DOM overlay: never window.prompt — the desktop (Electron) app
// forbids it. Its editor draws the card's fields from a schema the backend reads off the card
// NODES' own INPUT_TYPES, so a field added to Character Card shows up here without touching this
// file, labelled exactly as the rendered block labels it.

const NONE = "🚫 None";
const ALL_TAGS = "🏷 All";
const READER = "CardPresets";

// Only reached if /kinburg/cards/data never answered — enough to still write a card by hand.
const FALLBACK_SCHEMA = {
  character: [{ key: "name", label: "Name" }, { key: "notes", label: "Notes", multiline: true }],
  entity: [{ key: "name", label: "Name" }, { key: "description", label: "Description", multiline: true }],
};

let STORE = { none: NONE, all_tags: ALL_TAGS, order: [NONE], tags: [], presets: {}, schema: null };

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
  if (j.presets) STORE = j;   // /render answers with a card, not a store — don't wipe the list
  return j;
}

// One card's full values. Kept off /data on purpose: the reader nodes refresh that constantly and
// have no use for a whole library's worth of field text.
async function fetchPreset(name) {
  const r = await api.fetchApi(`/kinburg/cards/preset?name=${encodeURIComponent(name)}`);
  const j = await r.json();
  if (!j || !j.ok) throw new Error(j?.error || `card '${name}' could not be read`);
  return j;
}

const fieldsFor = (type) => (STORE.schema || FALLBACK_SCHEMA)[type] || FALLBACK_SCHEMA[type] || [];

// --------------------------------------------------------------------------------- reader nodes
const wv = (node, name) => node.widgets?.find((w) => w.name === name);
const isReader = (node) => node.comfyClass === READER || node.type === READER;

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
  for (const n of app.graph?._nodes || []) if (isReader(n)) syncReader(n);
}

// A rename would otherwise leave every node that picked the card pointing at a name the store no
// longer has — silently emitting an empty block on the next run. Follow it instead.
function renameInGraph(oldName, newName) {
  let moved = 0;
  for (const n of app.graph?._nodes || []) {
    if (!isReader(n)) continue;
    const pw = wv(n, "preset");
    if (pw && pw.value === oldName) { pw.value = newName; moved++; }
    syncReader(n);
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

function flash(btn, txt, ok = true) {
  const o = btn.textContent;
  btn.textContent = txt;
  btn.style.color = ok ? "#7CFC7C" : "#ff6b6b";
  setTimeout(() => { btn.textContent = o; btn.style.color = "#eee"; }, 1400);
}

// Two-click delete: the first click arms the button for 3s. Same arming the Model Library uses —
// a card can be twenty lines of description that nothing else in the graph holds a copy of.
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
    padding: "16px", minWidth: width, maxWidth: "820px", maxHeight: "84vh", overflow: "auto",
    font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" });
  const close = () => overlay.remove();
  overlay.appendChild(box);
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
  document.body.appendChild(overlay);
  return { overlay, box, close };
}

const icon = (type) => (type === "entity" ? "📦" : "👤");

// --------------------------------------------------------------------------------- pure helpers
// Rows the library lists: the card the node that opened the dialog has selected goes first, the
// rest A→Z, narrowed by the search box and the tag select.
function libraryRows(query, tag, focus) {
  const presets = STORE.presets || {};
  const q = String(query || "").trim().toLowerCase();
  const allTag = STORE.all_tags || ALL_TAGS;
  const lc = tag && tag !== allTag ? String(tag).toLowerCase() : "";
  const rows = Object.keys(presets).filter((nm) => {
    const tags = ((presets[nm] || {}).tags || []).map((t) => String(t).toLowerCase());
    if (lc && !tags.includes(lc)) return false;
    if (q && !nm.toLowerCase().includes(q) && !tags.some((t) => t.includes(q))) return false;
    return true;
  }).sort((a, b) => a.localeCompare(b));
  const i = rows.indexOf(focus);
  if (i > 0) rows.unshift(rows.splice(i, 1)[0]);
  return rows;
}

// "Vasya" -> "Vasya copy" -> "Vasya copy 2" … never an overwrite by accident.
function copyName(base) {
  const taken = new Set(Object.keys(STORE.presets || {}));
  let name = `${base} copy`;
  for (let i = 2; taken.has(name); i++) name = `${base} copy ${i}`;
  return name;
}

// The body a Save click posts. `edited` is merged ONTO the stored values rather than replacing
// them, so switching a card's type in the editor doesn't throw away the fields of the type it came
// from. Pure, because rename is the one thing in this dialog that can quietly break a saved
// workflow, and it is worth asserting on directly.
function editorPayload(entry, { key, type, tags, edited }) {
  const body = {
    name: String(key || "").trim(),
    type: type === "entity" ? "entity" : "character",
    values: { ...(entry.values || {}), ...(edited || {}) },
    tags: tags == null ? "" : tags,
  };
  if (!entry.isNew && entry.key && entry.key !== body.name) body.old_name = entry.key;
  return body;
}

// ---------------------------------------------------------------------------------- card editor
// `entry` = {key, type, tags, values, isNew}. `afterSave(name)` runs on a successful save, so the
// library behind it can re-render on the card that was just written.
function cardEditor(entry, afterSave) {
  const { box, close } = modalShell("560px", 10001);
  let type = entry.type === "entity" ? "entity" : "character";
  let keyTouched = !entry.isNew;               // a new card takes its key from the Name field
  let previewTimer = null;
  const controls = new Map();                  // field key -> read its current value

  const edited = () => {
    const out = {};
    for (const [k, get] of controls) out[k] = get();
    return out;
  };

  const render = () => {
    if (controls.size) entry.values = { ...(entry.values || {}), ...edited() }; // keep typing across a type switch
    controls.clear();
    box.innerHTML = "";

    box.appendChild(mk("div", "font-size:15px;font-weight:600;margin-bottom:2px",
      entry.isNew ? "➕ New card" : `✏️ Edit card — ${entry.key}`));
    box.appendChild(mk("div", "opacity:0.6;margin-bottom:12px;font-size:12px",
      "Empty fields are left out of the block. Saving writes straight to the library."));

    // -- the two things that are about the PRESET, not about the card -------------------------
    const line = (label, hint) => {
      const row = mk("div", "display:flex;align-items:center;gap:8px;margin-bottom:6px");
      const l = mk("div", "flex:0 0 30%;opacity:0.8", label);
      if (hint) l.title = hint;
      row.appendChild(l);
      box.appendChild(row);
      return row;
    };

    const keyRow = line("Saved as",
      "The name in the Card Presets dropdown. Change it and every node picking this card follows.");
    const keyInput = mk("input", INPUT + ";flex:1");
    keyInput.value = entry.key || "";
    keyInput.placeholder = "library name…";
    keyInput.addEventListener("input", () => { keyTouched = true; });
    keyRow.appendChild(keyInput);

    const typeRow = line("Type",
      "Character has the attribute fields; Entity is one free-form description. Switching keeps both sets of fields — only the block changes.");
    const typeSel = mk("select", INPUT + ";flex:1");
    for (const [v, lbl] of [["character", "👤 Character"], ["entity", "📦 Entity"]]) {
      const o = mk("option", null, lbl);
      o.value = v;
      if (v === type) o.selected = true;
      typeSel.appendChild(o);
    }
    typeRow.appendChild(typeSel);

    const tagRow = line("Tags", "Comma-separated. This is what the reader's `filter` dropdown narrows by.");
    const tagInput = mk("input", INPUT + ";flex:1");
    tagInput.value = Array.isArray(entry.tags) ? entry.tags.join(", ") : (entry.tags || "");
    tagInput.placeholder = "heroes, medieval…";
    tagRow.appendChild(tagInput);

    typeSel.addEventListener("change", () => {
      type = typeSel.value;
      entry.key = keyInput.value;              // a re-render rebuilds these boxes; carry them over
      entry.tags = tagInput.value;
      render();
    });

    box.appendChild(mk("div", "border-top:1px solid #383838;margin:12px 0 10px"));

    // -- the card's own fields, straight off the card node's schema -----------------------------
    const values = entry.values || {};
    const preview = mk("pre", "background:#1a1a1a;border:1px solid #383838;border-radius:4px;padding:8px;margin:0;white-space:pre-wrap;font:12px/1.45 monospace;max-height:180px;overflow:auto;opacity:0.9");
    const refreshPreview = async () => {
      try {
        const j = await postJSON("/kinburg/cards/render", { type, values: { ...values, ...edited() } });
        preview.textContent = j.card || "(nothing — every field is empty)";
      } catch (e) { preview.textContent = `(preview failed: ${e.message})`; }
    };
    const schedulePreview = () => { clearTimeout(previewTimer); previewTimer = setTimeout(refreshPreview, 250); };

    for (const f of fieldsFor(type)) {
      const row = mk("div", `display:flex;align-items:${f.multiline ? "flex-start" : "center"};gap:8px;margin-bottom:6px`);
      const l = mk("div", "flex:0 0 30%;opacity:0.8;padding-top:4px", f.label || f.key);
      if (f.tooltip) l.title = f.tooltip;
      row.appendChild(l);
      const ctl = f.multiline
        ? mk("textarea", INPUT + ";flex:1;min-height:54px;resize:vertical")
        : mk("input", INPUT + ";flex:1");
      ctl.value = values[f.key] ?? "";
      if (f.tooltip) ctl.title = f.tooltip;
      ctl.addEventListener("input", () => {
        if (f.key === "name" && !keyTouched) keyInput.value = ctl.value;
        schedulePreview();
      });
      controls.set(f.key, () => ctl.value);
      row.appendChild(ctl);
      box.appendChild(row);
    }

    // -- what the `card` output will actually be ------------------------------------------------
    box.appendChild(mk("div", "opacity:0.6;font-size:12px;margin:12px 0 4px",
      "Preview — the block the `card` output emits:"));
    box.appendChild(preview);
    refreshPreview();

    // -- footer ---------------------------------------------------------------------------------
    const foot = mk("div", "display:flex;justify-content:flex-end;gap:8px;margin-top:14px");
    const cancel = mk("button", BTN, "Cancel");
    cancel.onclick = close;
    const save = mk("button", PRIMARY, entry.isNew ? "Create" : "Save");
    save.onclick = async () => {
      const body = editorPayload(entry, { key: keyInput.value, type, tags: tagInput.value, edited: edited() });
      if (!body.name) { flash(save, "✕ name required", false); return; }
      try {
        await postJSON("/kinburg/cards/save", body);
        if (body.old_name) renameInGraph(body.old_name, body.name);
        refreshReaders();
        close();
        afterSave?.(body.name);
      } catch (e) { flash(save, "✕ " + e.message, false); }
    };
    foot.appendChild(cancel);
    foot.appendChild(save);
    box.appendChild(foot);
  };
  render();
}

// ------------------------------------------------------------------------------- library dialog
async function libraryDialog(focusName) {
  await refreshStore();
  const { box, close } = modalShell("660px");
  let query = "";
  let tag = STORE.all_tags || ALL_TAGS;

  const openEditor = async (name, duplicate) => {
    try {
      const p = await fetchPreset(name);
      cardEditor({
        key: duplicate ? copyName(name) : name,
        type: p.type,
        tags: p.tags,
        values: p.values,
        isNew: !!duplicate,
      }, (savedAs) => { if (!duplicate && focusName === name) focusName = savedAs; render(); });
    } catch (e) { alert(e.message); }
  };

  const render = () => {
    box.innerHTML = "";
    box.appendChild(mk("div", "font-size:15px;font-weight:600;margin-bottom:2px", "🗂 Card library"));
    box.appendChild(mk("div", "opacity:0.6;margin-bottom:12px;font-size:12px",
      "Edit a saved card's fields, rename it, duplicate it as the base for the next one, or write one from scratch."));

    // -- toolbar --------------------------------------------------------------------------------
    const bar = mk("div", "display:flex;align-items:center;gap:8px;margin-bottom:10px");
    const search = mk("input", INPUT + ";flex:1");
    search.value = query;
    search.placeholder = "search name / tag…";
    search.addEventListener("input", () => { query = search.value; renderRows(); });
    bar.appendChild(search);

    const tagSel = mk("select", INPUT + ";flex:0 0 auto");
    for (const t of [STORE.all_tags || ALL_TAGS, ...(STORE.tags || [])]) {
      const o = mk("option", null, t);
      o.value = t;
      if (t === tag) o.selected = true;
      tagSel.appendChild(o);
    }
    tagSel.addEventListener("change", () => { tag = tagSel.value; renderRows(); });
    bar.appendChild(tagSel);

    for (const [t, label, title] of [["character", "➕ 👤", "New character card"],
                                     ["entity", "➕ 📦", "New entity card"]]) {
      const b = mk("button", BTN, label);
      b.title = title;
      b.onclick = () => cardEditor({ key: "", type: t, tags: "", values: {}, isNew: true },
        (savedAs) => { focusName = savedAs; render(); });
      bar.appendChild(b);
    }
    box.appendChild(bar);

    const list = mk("div");
    box.appendChild(list);

    // -- rows -----------------------------------------------------------------------------------
    const renderRows = () => {
      list.innerHTML = "";
      const presets = STORE.presets || {};
      const rows = libraryRows(query, tag, focusName);
      if (!rows.length) {
        list.appendChild(mk("div", "opacity:0.6;padding:8px 0", Object.keys(presets).length
          ? "(nothing matches)"
          : "(no cards saved yet — ➕ writes one, or save from a Character / Entity Card node)"));
        return;
      }
      for (const nm of rows) {
        const p = presets[nm] || {};
        const row = mk("div", "display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #333");
        if (nm === focusName) {
          css(row, { borderLeft: "2px solid #3b82f6", paddingLeft: "6px", marginLeft: "-8px" });
        }

        const s = mk("span", "flex:0 0 32%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap",
          `${icon(p.type)} ${nm}`);
        s.title = nm;
        row.appendChild(s);

        const tagInput = mk("input", INPUT + ";flex:1");
        tagInput.value = (p.tags || []).join(", ");
        tagInput.placeholder = "tags…";
        tagInput.title = "Comma-separated. Saved on Enter, or when you click away.";
        let sent = tagInput.value;
        const saveTags = async () => {
          if (tagInput.value === sent) return;
          try {
            await postJSON("/kinburg/cards/tags", { name: nm, tags: tagInput.value });
            sent = tagInput.value;
            refreshReaders();
            tagInput.style.borderColor = "#3d6b3d";
            setTimeout(() => { tagInput.style.borderColor = "#555"; }, 1200);
          } catch (e) { tagInput.style.borderColor = "#ff6b6b"; tagInput.title = e.message; }
        };
        tagInput.addEventListener("keydown", (e) => { if (e.key === "Enter") saveTags(); });
        tagInput.addEventListener("blur", saveTags);
        row.appendChild(tagInput);

        const edit = mk("button", ICON, "✏️");
        edit.title = "Edit this card's fields";
        edit.onclick = () => openEditor(nm, false);
        row.appendChild(edit);

        const dup = mk("button", ICON, "⧉");
        dup.title = "Duplicate — opens a copy to rename and edit";
        dup.onclick = () => openEditor(nm, true);
        row.appendChild(dup);

        const del = mk("button", DANGER, "🗑");
        del.title = "Delete this card";
        arm(del, "sure?", async () => {
          try {
            await postJSON("/kinburg/cards/save", { name: nm, delete: true });
            refreshReaders();
            render();
          } catch (e) { alert(e.message); }
        });
        row.appendChild(del);

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
  name: "Kinburg.CardPresets",
  async setup() { await refreshStore(); refreshReaders(); },
  async nodeCreated(node) {
    if (!isReader(node)) return;
    // Re-narrow the preset list whenever the tag filter changes.
    const fw = wv(node, "filter");
    if (fw) {
      const orig = fw.callback;
      fw.callback = function () { const r = orig ? orig.apply(this, arguments) : undefined; syncReader(node); return r; };
    }
    node.addWidget("button", "🔄 Refresh", null, async () => { await refreshStore(); refreshReaders(); }, { serialize: false });
    // Opens on the card THIS node has picked — one click from "wrong eye colour" to fixing it.
    node.addWidget("button", "🗂 Manage", null, () => {
      const cur = wv(node, "preset")?.value;
      libraryDialog(cur && cur !== (STORE.none || NONE) ? cur : null);
    }, { serialize: false });
    if (!Object.keys(STORE.presets || {}).length) refreshStore().then(() => syncReader(node));
    syncReader(node);
  },
});
