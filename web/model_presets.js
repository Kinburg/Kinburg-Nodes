import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Model Library — Model Select's two dropdowns, wired so the second can only offer settings that
// are valid for the first. The model list and the presets under it are served live from the backend
// (/kinburg/models/data), because a combo's values are baked when /object_info is served — long
// before anyone picks a model.
//
// Picking a model re-narrows the preset list to that model's own presets plus any shared with its
// families, and lands on the model's default preset when it has one. Ordering puts the default
// first and then the best measured score, so the useful choice is at the top.
//
// The Library dialog is a plain DOM overlay: never window.prompt/alert-for-input — the desktop
// (Electron) app forbids prompt(), which is how the card-preset Save button used to break.

const NONE = "🚫 None";
const SELECT = "KinburgModelSelect";
const SAVE = "KinburgSettingsSave";
const SETTINGS = "KinburgSettingsSelect";
const CAPTURE = "KinburgModelCapture";
const PICK_HINT = "(add existing…)";
const ALL_FAMILIES = "🏷 All";
const FAMILY_W = "🏷 family";

let STORE = { none: NONE, order: [NONE], families: [], models: {}, shared: {} };

const wv = (node, name) => node.widgets?.find((w) => w.name === name);
const isType = (node, t) => node.comfyClass === t || node.type === t;

async function refreshStore() {
  try {
    const r = await api.fetchApi("/kinburg/models/data");
    const j = await r.json();
    if (j && j.ok) STORE = j;
  } catch (e) {
    console.error("[Kinburg] model library: load failed", e);
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

// ---------------------------------------------------------------------------- preset narrowing
// Presets a model may use, default first, then by measured score (desc), then alphabetically.
// Values stay the exact stored names — the score is shown in the Library dialog, not folded into
// the option text, since the option text IS what gets sent to the backend.
function presetsForModel(modelId) {
  const none = STORE.none || NONE;
  const model = (STORE.models || {})[modelId];
  if (!model) return [none];
  const entries = Object.entries(model.presets || {});
  entries.sort((a, b) => {
    if (!!b[1].default !== !!a[1].default) return b[1].default ? 1 : -1;
    const sa = a[1].score, sb = b[1].score;
    if (sa != null && sb != null && sa !== sb) return sb - sa;
    if ((sa == null) !== (sb == null)) return sa == null ? 1 : -1;
    return a[0].localeCompare(b[0]);
  });
  return [none, ...entries.map(([name]) => name)];
}

function defaultPresetFor(modelId) {
  const model = (STORE.models || {})[modelId];
  if (!model) return null;
  for (const [name, p] of Object.entries(model.presets || {})) if (p.default) return name;
  return null;
}

function setValues(widget, values, keepCurrent = true) {
  if (!widget) return;
  const opts = [...values];
  // Never drop what's currently selected: a preset saved live (no /object_info reload) or one that
  // has since been deleted must stay visible rather than silently becoming something else.
  if (keepCurrent && widget.value && !opts.includes(widget.value)) opts.push(widget.value);
  if (!widget.options) widget.options = {};
  widget.options.values = opts;
}

// -------------------------------------------------------------------------- family filter (picker)
// With a dozen finetunes per base model the model dropdown gets unusable, so it's narrowed by
// family first. Purely a picker convenience: the chosen `model` alone decides what's loaded.
//
// It is deliberately NOT a node input. A real widget would land in the prompt, and widget values are
// part of ComfyUI's cache signature (comfy_execution/caching.py) — so flipping a cosmetic filter
// would invalidate this node AND everything downstream, i.e. re-sample the image. Instead it lives
// in `node.properties`, which litegraph serialises but never sends to the backend, so the filter
// survives a reload and costs nothing. (Same trick as the Image Compare link in web/compare.js.)
function familyOf(node) {
  const v = node.properties?._kbFamily;
  return typeof v === "string" && v ? v : ALL_FAMILIES;
}

// Any Settings Select / Settings Save reading a `model_id` wire has to be re-narrowed when the model
// it follows changes — including when it changes to nothing.
function resyncFollowers() {
  for (const n of app.graph?._nodes || []) {
    if (isType(n, SETTINGS)) syncSettings(n);
    else if (isType(n, SAVE)) syncSave(n);
  }
}

function modelsForFamily(family) {
  const none = STORE.none || NONE;
  if (!family || family === ALL_FAMILIES) return [...(STORE.order || [none])];
  const lc = family.toLowerCase();
  const ids = Object.keys(STORE.models || {})
    .filter((id) => (STORE.models[id].families || []).some((f) => String(f).toLowerCase() === lc))
    .sort();
  return [none, ...ids];
}

function installFamilyFilter(node, resync) {
  const opts = { serialize: false };
  Object.defineProperty(opts, "values", {
    get: () => {
      const vals = [ALL_FAMILIES, ...(STORE.families || [])];
      const cur = familyOf(node);
      if (!vals.includes(cur)) vals.push(cur);   // a family that has since been renamed away
      return vals;
    }, enumerable: true, configurable: true });
  const w = node.addWidget("combo", FAMILY_W, familyOf(node), (v) => {
    node.properties = node.properties || {};
    node.properties._kbFamily = v;
    // Switching family means you're on your way to a different model, so the old pick is cleared
    // rather than left showing under a filter it doesn't belong to. Cleared to None (not
    // auto-selected, even when the family holds exactly one model) so nothing loads until you
    // choose. This happens ONLY in this callback — the user's own click. syncSelect /
    // loadedGraphNode / Refresh must never reset a selection, or opening a saved workflow would
    // wipe it.
    const none = STORE.none || NONE;
    const mw = wv(node, "model");
    const pw = wv(node, "preset");
    if (mw) mw.value = none;
    if (pw) pw.value = none;          // a preset only means anything under its model
    resync(node);
    resyncFollowers();
  }, opts);
  // Top of the node — you filter, then pick.
  const at = node.widgets.indexOf(w);
  if (at > 0) {
    node.widgets.splice(at, 1);
    node.widgets.unshift(w);
  }
  return w;
}

// Properties are restored by configure(), which runs AFTER the widget was created, so the saved
// filter has to be pushed back into the widget once the graph has loaded.
function restoreFamilyFilter(node) {
  const fw = wv(node, FAMILY_W);
  if (fw) fw.value = familyOf(node);
}

function syncSelect(node) {
  const mw = wv(node, "model");
  setValues(mw, modelsForFamily(familyOf(node)));
  const pw = wv(node, "preset");
  setValues(pw, presetsForModel(mw?.value));
  node.setDirtyCanvas(true, true);
}

// ------------------------------------------------------------------------------- family picker
// `families` stays a plain comma-separated STRING (multi-value, and ComfyUI has no multi-select
// widget), but you shouldn't have to TYPE one: a typo doesn't error, it just quietly stops shared
// presets from ever appearing. This combo appends an existing family and resets itself.
function installFamilyPicker(node) {
  const fw = wv(node, "families");
  if (!fw) return;
  const opts = { serialize: false };
  Object.defineProperty(opts, "values", {
    get: () => [PICK_HINT, ...(STORE.families || [])], enumerable: true, configurable: true });
  const combo = node.addWidget("combo", "＋ family", PICK_HINT, (v) => {
    if (!v || v === PICK_HINT) return;
    const have = String(fw.value || "").split(",").map((s) => s.trim()).filter(Boolean);
    if (!have.some((h) => h.toLowerCase() === v.toLowerCase())) have.push(v);
    fw.value = have.join(", ");
    combo.value = PICK_HINT;              // reset, so the next pick is a fresh choice
    node.setDirtyCanvas(true, true);
  }, opts);
  // Sit directly under `families` instead of at the end of the widget list.
  const at = node.widgets.indexOf(fw);
  const now = node.widgets.indexOf(combo);
  if (at >= 0 && now !== at + 1) {
    node.widgets.splice(now, 1);
    node.widgets.splice(at + 1, 0, combo);
  }
}

// --------------------------------------------------------------------------- Settings Select
// Presets live under a model, so this node needs to know which one. A wired `model_id` (from Model
// Select) wins over its own dropdown — then the model is chosen in exactly one place — so the picker
// follows the link back to read whatever that node currently has selected.
function wiredModelId(node) {
  const idx = (node.inputs || []).findIndex((i) => i.name === "model_id");
  if (idx < 0 || node.inputs[idx].link == null) return null;
  const up = node.getInputNode?.(idx);
  const v = up?.widgets?.find((w) => w.name === "model")?.value;
  return typeof v === "string" && v && v !== (STORE.none || NONE) ? v : null;
}

function syncSettings(node) {
  const wired = wiredModelId(node);
  const mw = wv(node, "model");
  setValues(mw, modelsForFamily(familyOf(node)));
  if (mw) mw.label = wired ? `model → ${wired}` : undefined;   // say the dropdown is being ignored
  const pw = wv(node, "preset");
  const opts = presetsForModel(wired || mw?.value);
  // Unlike Model Select this does NOT jump to the model's default: picking presets is the whole
  // point of the node, so only an unavailable choice is corrected.
  if (pw && pw.value && !opts.includes(pw.value)) pw.value = STORE.none || NONE;
  setValues(pw, opts, false);
  node.setDirtyCanvas(true, true);
}

function onModelPicked(node) {
  const mw = wv(node, "model");
  const pw = wv(node, "preset");
  const opts = presetsForModel(mw?.value);
  if (pw) {
    // Landing on the model's default is the whole point — "don't make me remember which settings
    // go with this model". Otherwise fall back to None rather than keeping the previous model's
    // preset, which would resolve to nothing.
    const def = defaultPresetFor(mw?.value);
    pw.value = def && opts.includes(def) ? def : (STORE.none || NONE);
    setValues(pw, opts, false);
  }
  node.setDirtyCanvas(true, true);
  resyncFollowers();
}

// Settings Save gets no family filter on purpose: wiring `model_id` from Model Select is the right
// way to use it (the preset is then always filed under the model that actually ran), which leaves the
// dropdown a rarely-touched fallback — and a second family-ish combo next to `＋ family` would just
// be confusing.
function syncSave(node) {
  const wired = wiredModelId(node);
  const mw = wv(node, "model");
  setValues(mw, STORE.order || [NONE]);
  if (mw) mw.label = wired ? `model → ${wired}` : undefined;
  node.setDirtyCanvas(true, true);
}

function refreshAll() {
  for (const n of app.graph?._nodes || []) {
    if (isType(n, SELECT)) syncSelect(n);
    else if (isType(n, SETTINGS)) syncSettings(n);
    else if (isType(n, SAVE)) syncSave(n);
  }
}

// ------------------------------------------------------------------------------- Library dialog
const css = (el, s) => Object.assign(el.style, s);
const mk = (tag, style, text) => {
  const e = document.createElement(tag);
  if (style) e.style.cssText = style;
  if (text != null) e.textContent = text;
  return e;
};
const BTN = "background:#333;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 10px;cursor:pointer;flex:0 0 auto";
const DANGER = "background:#3a2727;color:#eee;border:1px solid #663;border-radius:4px;padding:4px 10px;cursor:pointer;flex:0 0 auto";
const INPUT = "background:#1a1a1a;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 8px;min-width:0";

function flash(btn, txt, ok = true) {
  const o = btn.textContent;
  btn.textContent = txt;
  btn.style.color = ok ? "#7CFC7C" : "#ff6b6b";
  setTimeout(() => { btn.textContent = o; btn.style.color = "#eee"; }, 1200);
}

function fmtScore(p) {
  const bits = [];
  if (p.score != null) bits.push(`★${Number(p.score).toFixed(2)}`);
  if (p.seconds != null) bits.push(`${Number(p.seconds).toFixed(1)}s`);
  if (p.width && p.height) bits.push(`${p.width}×${p.height}`);
  bits.push(`${p.stages || 0} stage${p.stages === 1 ? "" : "s"}`);
  return bits.join(" · ");
}

// ------------------------------------------------------------------------- schema-driven fields
// Controls are rendered from the node class's OWN schema, fetched from ComfyUI's /object_info. That
// is what keeps this editor honest for node types it has never heard of: a FLOAT gets its declared
// min/max/step, a combo gets its real option list — and for a loader that means the LIVE list of
// files on disk, so a bundle can be pointed at a different checkpoint from here.
const SCHEMAS = {};

async function fetchSchemas(classes) {
  await Promise.all([...new Set(classes)].filter((c) => c && !(c in SCHEMAS)).map(async (c) => {
    try {
      const r = await api.fetchApi(`/object_info/${encodeURIComponent(c)}`);
      const j = await r.json();
      SCHEMAS[c] = j?.[c] || null;
    } catch (e) {
      SCHEMAS[c] = null;                       // uninstalled / renamed: fall back to a text box
    }
  }));
  return SCHEMAS;
}

function inputSpec(classType, field) {
  const inp = SCHEMAS[classType]?.input || {};
  for (const cat of ["required", "optional"]) {
    const s = inp[cat]?.[field];
    if (s) return { type: s[0], opts: (s[1] && typeof s[1] === "object") ? s[1] : {} };
  }
  return null;
}

// Returns { el, get } — `get` reads the control's current value already in the right JS type, so
// nothing downstream has to guess whether "3" meant a string or a number.
function fieldControl(spec, value) {
  const type = spec?.type;
  const opts = spec?.opts || {};
  if (Array.isArray(type)) {                   // combo
    const sel = mk("select", INPUT + ";flex:1");
    const values = [...type.map(String)];
    if (value != null && !values.includes(String(value))) values.unshift(String(value));
    for (const v of values) {
      const o = mk("option", null, v);
      o.value = v;
      if (String(value) === v) o.selected = true;
      sel.appendChild(o);
    }
    if (value != null && !type.map(String).includes(String(value))) {
      sel.style.borderColor = "#b45309";       // the stored value is no longer on disk / valid
      sel.title = "this value is not in the node's current options";
    }
    return { el: sel, get: () => sel.value };
  }
  if (type === "BOOLEAN") {
    const cb = mk("input");
    cb.type = "checkbox";
    cb.checked = !!value;
    css(cb, { flex: "0 0 auto", width: "16px", height: "16px" });
    return { el: cb, get: () => cb.checked };
  }
  if (type === "INT" || type === "FLOAT") {
    const inp = mk("input", INPUT + ";flex:1");
    inp.type = "number";
    if (opts.min != null) inp.min = opts.min;
    if (opts.max != null) inp.max = opts.max;
    inp.step = opts.step != null ? opts.step : (type === "INT" ? 1 : 0.01);
    inp.value = value ?? "";
    return { el: inp, get: () => (inp.value === "" ? null : Number(inp.value)) };
  }
  if (opts.multiline) {
    const ta = mk("textarea", INPUT + ";flex:1;min-height:54px;resize:vertical");
    ta.value = value ?? "";
    return { el: ta, get: () => ta.value };
  }
  const inp = mk("input", INPUT + ";flex:1");
  inp.value = value ?? "";
  if (!spec) inp.title = "this node type isn't installed — edited as plain text";
  return { el: inp, get: () => inp.value };
}

function modalShell(width = "620px") {
  const overlay = mk("div");
  css(overlay, { position: "fixed", inset: "0", background: "rgba(0,0,0,0.55)", zIndex: 10001,
    display: "flex", alignItems: "center", justifyContent: "center" });
  const box = mk("div");
  css(box, { background: "#222", color: "#eee", border: "1px solid #444", borderRadius: "8px",
    padding: "16px", minWidth: width, maxWidth: "860px", maxHeight: "84vh", overflow: "auto",
    font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" });
  const close = () => overlay.remove();
  overlay.appendChild(box);
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
  document.body.appendChild(overlay);
  return { overlay, box, close };
}

// ------------------------------------------------------------------------------- recipe editor
async function recipeEditor(id, afterSave) {
  const { box, close } = modalShell();
  box.appendChild(mk("div", "opacity:0.7", "loading…"));
  let payload;
  try {
    const r = await api.fetchApi(`/kinburg/models/recipe?id=${encodeURIComponent(id)}`);
    payload = await r.json();
    if (!payload.ok) throw new Error(payload.error);
  } catch (e) {
    box.innerHTML = "";
    box.appendChild(mk("div", "color:#ff6b6b", `✕ ${e.message}`));
    return;
  }
  await fetchSchemas((payload.editable || []).map((n) => n.class_type));

  const render = () => {
    box.innerHTML = "";
    box.appendChild(mk("div", "font-size:15px;font-weight:600;margin-bottom:4px",
      `🔧 Recipe — ${id}`));
    box.appendChild(mk("div", "opacity:0.6;margin-bottom:12px;font-size:12px",
      "Settings of the captured assembly. Wired inputs are the shape of the bundle and can't be "
      + "edited here — re-capture to change the wiring. Changing a loader's file re-reads weights; "
      + "changing a patch value does not."));

    const getters = [];
    for (const node of payload.editable || []) {
      const card = mk("div", "border:1px solid #3a3a3a;border-radius:6px;padding:8px 10px;margin-bottom:8px");
      const title = mk("div", "font-weight:600;margin-bottom:6px", node.class_type);
      title.appendChild(mk("span", "opacity:0.45;font-weight:400;margin-left:6px", `#${node.id}`));
      card.appendChild(title);
      if (!SCHEMAS[node.class_type]) {
        card.appendChild(mk("div", "color:#b45309;font-size:12px;margin-bottom:6px",
          "⚠ this node type isn't installed right now — values shown as plain text"));
      }
      for (const [field, value] of Object.entries(node.values || {})) {
        const row = mk("div", "display:flex;align-items:center;gap:8px;margin:4px 0");
        const lab = mk("span", "flex:0 0 34%;opacity:0.8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap", field);
        lab.title = field;
        const ctl = fieldControl(inputSpec(node.class_type, field), value);
        row.append(lab, ctl.el);
        card.appendChild(row);
        getters.push({ key: `${node.id}.${field}`, get: ctl.get, was: value });
      }
      for (const field of node.linked || []) {
        const row = mk("div", "display:flex;align-items:center;gap:8px;margin:4px 0;opacity:0.45");
        row.append(mk("span", "flex:0 0 34%", field), mk("span", null, "⟵ wired (part of the assembly)"));
        card.appendChild(row);
      }
      box.appendChild(card);
    }

    const foot = mk("div", "display:flex;justify-content:flex-end;gap:8px;margin-top:12px");
    const status = mk("div", "flex:1;font-size:12px;opacity:0.75;align-self:center");
    const saveBtn = mk("button", "background:#3b82f6;color:#fff;border:1px solid #555;border-radius:4px;padding:6px 12px;cursor:pointer", "Save changes");
    saveBtn.onclick = async () => {
      const values = {};
      for (const g of getters) {
        const now = g.get();
        // Only send what actually moved — a no-op save shouldn't rewrite the bundle (and shouldn't
        // invalidate Model Select's IS_CHANGED for nothing).
        if (String(now) !== String(g.was)) values[g.key] = now;
      }
      if (!Object.keys(values).length) { status.textContent = "nothing changed"; return; }
      try {
        const r = await api.fetchApi("/kinburg/models/recipe", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id, values }),
        });
        const j = await r.json();
        if (!j.ok) throw new Error(j.error);
        STORE = j;
        refreshAll();
        const rr = await api.fetchApi(`/kinburg/models/recipe?id=${encodeURIComponent(id)}`);
        payload = await rr.json();
        render();
        const el = box.querySelector("[data-status]");
        if (el) {
          el.textContent = `✓ saved ${(j.applied || []).length} field(s)`
            + ((j.rejected || []).length ? ` · ✕ ${j.rejected.join("; ")}` : "");
          el.style.color = (j.rejected || []).length ? "#b45309" : "#7CFC7C";
        }
        afterSave?.();
      } catch (e) {
        status.textContent = `✕ ${e.message}`;
        status.style.color = "#ff6b6b";
      }
    };
    const revert = mk("button", BTN, "Revert");
    revert.onclick = render;
    const cl = mk("button", BTN, "Close");
    cl.onclick = close;
    status.dataset.status = "1";
    foot.append(status, revert, saveBtn, cl);
    box.appendChild(foot);
  };
  render();
}

// ---------------------------------------------------------------------------- overrides editor
// A preset may retune the bundle's patch values without cloning the bundle: "same model, shift 3 vs
// shift 5" is two presets over one recipe. Keys are class-scoped, not node-scoped, so an override
// survives re-capturing the bundle (which renumbers its nodes).
async function overridesEditor(modelId, presetName, shared, afterSave) {
  const { box, close } = modalShell("580px");
  box.appendChild(mk("div", "opacity:0.7", "loading…"));
  let editable = [];
  try {
    const r = await api.fetchApi(`/kinburg/models/recipe?id=${encodeURIComponent(modelId)}`);
    const j = await r.json();
    if (!j.ok) throw new Error(j.error);
    editable = j.editable || [];
  } catch (e) {
    box.innerHTML = "";
    box.appendChild(mk("div", "color:#ff6b6b", `✕ ${e.message}`));
    return;
  }
  await fetchSchemas(editable.map((n) => n.class_type));

  // `override_map` is the real map; the sibling `overrides` field is only the count the badge uses.
  const bucket = shared ? (STORE.shared || {}) : ((STORE.models || {})[modelId]?.presets || {});
  const current = bucket[presetName]?.override_map || {};
  // Candidates, deduped by class.field; a class appearing twice in one bundle (Ideogram's two UNET
  // loaders) is flagged, since an override there would hit both.
  const seen = new Map();
  for (const node of editable) {
    for (const [field, value] of Object.entries(node.values || {})) {
      const key = `${node.class_type}.${field}`;
      const prev = seen.get(key);
      if (prev) prev.count += 1;
      else seen.set(key, { key, class_type: node.class_type, field, value, count: 1 });
    }
  }

  box.innerHTML = "";
  box.appendChild(mk("div", "font-size:15px;font-weight:600;margin-bottom:4px",
    `🎚 Preset overrides — ${presetName}`));
  box.appendChild(mk("div", "opacity:0.6;margin-bottom:12px;font-size:12px",
    `Values this preset applies to ${modelId}'s bundle instead of the bundle's own. Tick a row to `
    + "override it; the bundle itself is untouched, so other presets keep the original value."));

  const rows = [];
  for (const cand of seen.values()) {
    const row = mk("div", "display:flex;align-items:center;gap:8px;margin:5px 0");
    const cb = mk("input");
    cb.type = "checkbox";
    cb.checked = cand.key in current;
    css(cb, { flex: "0 0 auto", width: "16px", height: "16px" });
    const lab = mk("span", "flex:0 0 40%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap",
      cand.key + (cand.count > 1 ? ` (×${cand.count})` : ""));
    lab.title = cand.count > 1
      ? `${cand.key} — ${cand.count} nodes of this class; an override hits all of them`
      : cand.key;
    const ctl = fieldControl(inputSpec(cand.class_type, cand.field),
      cand.key in current ? current[cand.key] : cand.value);
    const base = mk("span", "flex:0 0 auto;opacity:0.5;font-size:12px", `bundle: ${cand.value}`);
    const sync = () => { ctl.el.disabled = !cb.checked; ctl.el.style.opacity = cb.checked ? "1" : "0.45"; };
    cb.onchange = sync;
    sync();
    row.append(cb, lab, ctl.el, base);
    box.appendChild(row);
    rows.push({ cand, cb, ctl });
  }
  if (!rows.length) box.appendChild(mk("div", "opacity:0.6", "(this bundle has no editable values)"));

  const foot = mk("div", "display:flex;justify-content:flex-end;gap:8px;margin-top:12px");
  const status = mk("div", "flex:1;font-size:12px;align-self:center");
  const saveBtn = mk("button", "background:#3b82f6;color:#fff;border:1px solid #555;border-radius:4px;padding:6px 12px;cursor:pointer", "Save overrides");
  saveBtn.onclick = async () => {
    const overrides = {};
    for (const r of rows) if (r.cb.checked) overrides[r.cand.key] = r.ctl.get();
    try {
      await postJSON("/kinburg/models/overrides",
        { model: modelId, name: presetName, shared: !!shared, overrides });
      refreshAll();
      status.textContent = `✓ ${Object.keys(overrides).length} override(s) saved`;
      status.style.color = "#7CFC7C";
      afterSave?.();
    } catch (e) {
      status.textContent = `✕ ${e.message}`;
      status.style.color = "#ff6b6b";
    }
  };
  const cl = mk("button", BTN, "Close");
  cl.onclick = close;
  foot.append(status, saveBtn, cl);
  box.appendChild(foot);
}

async function libraryDialog() {
  await refreshStore();
  const overlay = mk("div");
  css(overlay, { position: "fixed", inset: "0", background: "rgba(0,0,0,0.55)", zIndex: 10000,
    display: "flex", alignItems: "center", justifyContent: "center" });
  const box = mk("div");
  css(box, { background: "#222", color: "#eee", border: "1px solid #444", borderRadius: "8px",
    padding: "16px", minWidth: "560px", maxWidth: "820px", maxHeight: "84vh", overflow: "auto",
    font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" });
  const close = () => overlay.remove();

  const render = () => {
    box.innerHTML = "";
    const h = mk("div", "font-size:15px;font-weight:600;margin-bottom:4px", "🗂 Model Library");
    box.appendChild(h);
    box.appendChild(mk("div", "opacity:0.6;margin-bottom:12px;font-size:12px",
      "Bundles are registered with Model Capture; presets are saved with Settings Save. " +
      "Here you can retag, set defaults, rename and delete."));

    const models = STORE.models || {};
    const ids = Object.keys(models).sort();
    if (!ids.length) {
      box.appendChild(mk("div", "opacity:0.6",
        "(empty — wire a working loader stack into Model Capture and register it)"));
    }

    for (const id of ids) {
      const m = models[id];
      const card = mk("div", "border:1px solid #3a3a3a;border-radius:6px;padding:10px;margin-bottom:10px");

      const top = mk("div", "display:flex;align-items:center;gap:8px;margin-bottom:6px");
      const title = mk("span", "flex:1;min-width:0;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap",
        `🎛 ${id}`);
      title.title = id;
      const recipeBtn = mk("button", BTN, "🔧 Recipe");
      recipeBtn.title = "Edit this bundle's settings — filenames, shift, cfg, dtype…";
      recipeBtn.onclick = () => recipeEditor(id, render);
      const shapeBtn = mk("button", BTN, "Shape");
      shapeBtn.title = "Show the captured assembly and its raw recipe";
      shapeBtn.onclick = async () => {
        const old = card.querySelector("pre");
        if (old) { old.remove(); return; }
        try {
          const r = await api.fetchApi(`/kinburg/models/recipe?id=${encodeURIComponent(id)}`);
          const j = await r.json();
          if (!j.ok) throw new Error(j.error);
          card.appendChild(mk("pre", "white-space:pre-wrap;background:#1a1a1a;border:1px solid #444;border-radius:4px;padding:8px;margin:6px 0 0;font-size:12px",
            `${j.summary}\n\n${JSON.stringify(j.recipe, null, 2)}`));
        } catch (e) { flash(shapeBtn, "✕ " + e.message, false); }
      };
      const renameBtn = mk("button", BTN, "Rename");
      const delBtn = mk("button", DANGER, "Delete");
      delBtn.onclick = async () => {
        if (delBtn.dataset.armed !== "1") {
          delBtn.dataset.armed = "1"; delBtn.textContent = "Delete — sure?";
          setTimeout(() => { delBtn.dataset.armed = "0"; delBtn.textContent = "Delete"; }, 3000);
          return;
        }
        try { await postJSON("/kinburg/models/model", { id, delete: true }); refreshAll(); render(); }
        catch (e) { flash(delBtn, "✕ " + e.message, false); }
      };
      top.append(title, recipeBtn, shapeBtn, renameBtn, delBtn);
      card.appendChild(top);

      const renameRow = mk("div", "display:none;gap:8px;margin-bottom:6px");
      const renameIn = mk("input", INPUT + ";flex:1"); renameIn.value = id;
      const renameGo = mk("button", BTN, "Save new id");
      const doRename = async () => {
        try {
          await postJSON("/kinburg/models/model", { id, rename: renameIn.value });
          refreshAll(); render();
        } catch (e) { flash(renameGo, "✕ " + e.message, false); }
      };
      renameGo.onclick = doRename;
      renameIn.addEventListener("keydown", (e) => { if (e.key === "Enter") doRename(); });
      renameRow.append(renameIn, renameGo);
      card.appendChild(renameRow);
      renameBtn.onclick = () => {
        renameRow.style.display = renameRow.style.display === "none" ? "flex" : "none";
      };

      const meta = mk("div", "display:flex;gap:8px;align-items:center;margin-bottom:8px;font-size:12px");
      const famIn = mk("input", INPUT + ";flex:1");
      famIn.placeholder = "new family…";
      const famGo = mk("button", BTN, "Add");
      const setFamilies = async (list, btn) => {
        try {
          await postJSON("/kinburg/models/model", { id, families: list.join(", ") });
          refreshAll(); render();
        } catch (e) { flash(btn, "✕ " + e.message, false); }
      };
      const doFam = () => {
        const v = famIn.value.trim();
        if (!v) return;
        setFamilies([...(m.families || []), v], famGo);
      };
      famGo.onclick = doFam;
      famIn.addEventListener("keydown", (e) => { if (e.key === "Enter") doFam(); });
      meta.append(mk("span", "opacity:0.6;flex:0 0 auto", `${m.node_count} node(s) · ${(m.slots || []).join(", ") || "no slots"}`), famIn, famGo);
      card.appendChild(meta);

      // Membership by clicking, not by retyping: a family is only useful when it matches EXACTLY
      // what other models and shared presets use, and typing it again is the one way to get that
      // wrong. The text box above is for creating a family that doesn't exist yet.
      const known = STORE.families || [];
      if (known.length) {
        const chips = mk("div", "display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px");
        const mine = new Set((m.families || []).map((f) => f.toLowerCase()));
        for (const f of known) {
          const on = mine.has(f.toLowerCase());
          const chip = mk("button",
            (on ? "background:#1d4ed8;border-color:#3b82f6;" : "background:#2a2a2a;")
            + "color:#eee;border:1px solid #555;border-radius:12px;padding:2px 10px;cursor:pointer;font-size:12px;flex:0 0 auto",
            (on ? "✓ " : "") + f);
          chip.title = on ? `remove '${f}' from ${id}` : `add '${f}' to ${id}`;
          chip.onclick = () => {
            const next = new Set(mine);
            if (on) next.delete(f.toLowerCase()); else next.add(f.toLowerCase());
            setFamilies(known.filter((k) => next.has(k.toLowerCase())), chip);
          };
          chips.appendChild(chip);
        }
        card.appendChild(chips);
      }

      if (m.classes?.length) {
        card.appendChild(mk("div", "opacity:0.5;font-size:11px;margin-bottom:8px",
          m.classes.join(" · ")));
      }

      const presets = Object.entries(m.presets || {}).sort((a, b) => a[0].localeCompare(b[0]));
      if (!presets.length) {
        card.appendChild(mk("div", "opacity:0.6;font-size:12px", "(no presets yet)"));
      }
      for (const [name, p] of presets) {
        const row = mk("div", "display:flex;align-items:center;gap:8px;padding:5px 0;border-top:1px solid #333");
        const label = mk("span", "flex:0 0 30%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap",
          `${p.default ? "⭐ " : ""}${p.shared ? "🌐 " : ""}${name}`);
        label.title = p.shared ? `${name} (shared preset)` : name;
        const stats = mk("span", "flex:0 0 auto;opacity:0.65;font-size:12px", fmtScore(p));
        const tagIn = mk("input", INPUT + ";flex:1");
        tagIn.value = (p.tags || []).join(", ");
        tagIn.placeholder = "tags…";
        const saveTags = mk("button", BTN, "Save");
        const doTags = async () => {
          try {
            await postJSON("/kinburg/models/preset",
              { model: id, name, tags: tagIn.value, shared: !!p.shared });
            refreshAll(); flash(saveTags, "✓");
          } catch (e) { flash(saveTags, "✕ " + e.message, false); }
        };
        saveTags.onclick = doTags;
        tagIn.addEventListener("keydown", (e) => { if (e.key === "Enter") doTags(); });
        const ovBtn = mk("button", BTN, p.overrides ? `🎚 ${p.overrides}` : "🎚");
        ovBtn.title = "Values this preset applies to the bundle instead of the bundle's own — "
          + "e.g. the same model at a different shift";
        ovBtn.onclick = () => overridesEditor(id, name, !!p.shared, render);
        const defBtn = mk("button", BTN, p.default ? "Unset default" : "Make default");
        defBtn.onclick = async () => {
          try {
            await postJSON("/kinburg/models/preset",
              { model: id, name, set_default: !p.default, shared: !!p.shared });
            refreshAll(); render();
          } catch (e) { flash(defBtn, "✕ " + e.message, false); }
        };
        const pdel = mk("button", DANGER, "✕");
        pdel.title = p.shared ? "Delete this shared preset (affects every model in its families)"
          : "Delete this preset";
        pdel.onclick = async () => {
          try {
            await postJSON("/kinburg/models/preset",
              { model: id, name, delete: true, shared: !!p.shared });
            refreshAll(); render();
          } catch (e) { flash(pdel, "✕ " + e.message, false); }
        };
        row.append(label, stats, tagIn, saveTags, ovBtn, defBtn, pdel);
        card.appendChild(row);
      }
      box.appendChild(card);
    }

    const foot = mk("div", "display:flex;justify-content:flex-end;gap:8px;margin-top:14px");
    const refreshBtn = mk("button", BTN, "🔄 Reload");
    refreshBtn.onclick = async () => { await refreshStore(); refreshAll(); render(); };
    const cl = mk("button", "background:#3b82f6;color:#fff;border:1px solid #555;border-radius:4px;padding:6px 12px;cursor:pointer", "Close");
    cl.onclick = close;
    foot.append(refreshBtn, cl);
    box.appendChild(foot);
  };

  render();
  overlay.appendChild(box);
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
  document.addEventListener("keydown", function esc(e) {
    if (e.key === "Escape") { close(); document.removeEventListener("keydown", esc); }
  });
  document.body.appendChild(overlay);
}

// -------------------------------------------------------------------------------- registration
app.registerExtension({
  name: "Kinburg.ModelLibrary",
  async setup() { await refreshStore(); refreshAll(); },
  async nodeCreated(node) {
    if (isType(node, SELECT)) {
      installFamilyFilter(node, syncSelect);
      const mw = wv(node, "model");
      if (mw) {
        const orig = mw.callback;
        mw.callback = function (...a) {
          const r = orig ? orig.apply(this, a) : undefined;
          // Loading a workflow replays widget values through the callbacks; re-picking the default
          // then would overwrite the saved preset choice.
          if (!app.configuringGraph) onModelPicked(node);
          return r;
        };
      }
      node.addWidget("button", "🗂 Library", null, () => libraryDialog(), { serialize: false });
      node.addWidget("button", "🔄 Refresh", null,
        async () => { await refreshStore(); refreshAll(); }, { serialize: false });
      if (!Object.keys(STORE.models || {}).length) refreshStore().then(() => syncSelect(node));
      syncSelect(node);
    } else if (isType(node, SETTINGS)) {
      installFamilyFilter(node, syncSettings);
      const mw = wv(node, "model");
      if (mw) {
        const orig = mw.callback;
        mw.callback = function (...a) {
          const r = orig ? orig.apply(this, a) : undefined;
          if (!app.configuringGraph) syncSettings(node);
          return r;
        };
      }
      // Connecting / disconnecting `model_id` changes which model's presets are valid.
      const origOCC = node.onConnectionsChange;
      node.onConnectionsChange = function (...a) {
        const r = origOCC ? origOCC.apply(this, a) : undefined;
        if (!app.configuringGraph) syncSettings(node);
        return r;
      };
      node.addWidget("button", "🔄 Refresh", null,
        async () => { await refreshStore(); refreshAll(); }, { serialize: false });
      syncSettings(node);
    } else if (isType(node, SAVE)) {
      installFamilyPicker(node);
      const origOCC = node.onConnectionsChange;
      node.onConnectionsChange = function (...a) {
        const r = origOCC ? origOCC.apply(this, a) : undefined;
        if (!app.configuringGraph) syncSave(node);
        return r;
      };
      node.addWidget("button", "🔄 Refresh", null,
        async () => { await refreshStore(); refreshAll(); }, { serialize: false });
      syncSave(node);
    } else if (isType(node, CAPTURE)) {
      installFamilyPicker(node);
    }
  },
  async loadedGraphNode(node) {
    // After a workflow load the saved values are in place — restore the filter (properties are
    // populated by now) and widen the option lists so the saved model/preset stay selectable
    // without needing a Refresh.
    if (isType(node, SELECT)) { restoreFamilyFilter(node); syncSelect(node); }
    else if (isType(node, SETTINGS)) { restoreFamilyFilter(node); syncSettings(node); }
    else if (isType(node, SAVE)) syncSave(node);
  },
});
