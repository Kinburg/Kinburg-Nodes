import { app } from "../../scripts/app.js";

// JSON Extract — self-labelling, auto-growing value outputs driven by the `paths` field,
// a 🔍 Explore JSON dialog to click fields out of the last run / a pasted sample, and a
// live in-node preview of the extracted values.
//
// Output layout is (found, report, value_1..value_N). Status is first so we can prune trailing
// unused value_* slots without disturbing the return-tuple index mapping (only trailing outputs
// are ever added/removed — the same invariant the dynamic-input helper keeps for inputs).

const CLASS = "KinburgJSONExtract";
const BASE_OUTPUTS = 2;          // found, report — never touched
const wv = (node, name) => node.widgets?.find((w) => w.name === name);

// ---------------------------------------------------------------- path → label (mirrors python)
function labelFromPath(path) {
  let p = String(path || "").trim();
  if (p.startsWith("$")) p = p.slice(1);
  p = p.replace(/\[(-?\d+|\*)\]/g, ".$1");
  const toks = p.split(".").map((s) => s.trim()).filter((s) => s !== "");
  if (!toks.length) return "root";
  const real = toks.filter((t) => t !== "*");
  if (!real.length) return "root";
  const last = real[real.length - 1];
  if (/^-?\d+$/.test(last)) {
    const prev = real.length >= 2 ? real[real.length - 2] : "item";
    return `${prev}_${last.replace("-", "")}`;
  }
  return last;
}

// Parse the `paths` textarea into [{path, label}], skipping blanks and `# comments`.
function parsePaths(text) {
  const out = [];
  for (const raw of String(text || "").split(/\r?\n/)) {
    const s = raw.trim();
    if (!s || s.startsWith("#")) continue;
    let path = s, alias = "";
    for (const sep of ["->", "=>"]) {
      const i = s.indexOf(sep);
      if (i >= 0) { path = s.slice(0, i).trim(); alias = s.slice(i + sep.length).trim(); break; }
    }
    if (!path) continue;
    out.push({ path, label: alias || labelFromPath(path) });
  }
  return out;
}

// Disambiguate duplicate output labels (name, name#2, name#3, …).
function dedupeLabels(labels) {
  const seen = {};
  return labels.map((lb) => {
    seen[lb] = (seen[lb] || 0) + 1;
    return seen[lb] === 1 ? lb : `${lb}#${seen[lb]}`;
  });
}

// ---------------------------------------------------------------- output sync
// Add/remove trailing value outputs to match the path count, then relabel. `relabelOnly` skips
// structural changes (used on workflow load so restored links are never dropped).
function syncOutputs(node, relabelOnly) {
  const max = node._kbMaxValues || 12;
  const entries = parsePaths(wv(node, "paths")?.value || "");
  const labels = dedupeLabels(entries.slice(0, max).map((e) => e.label));

  if (!node.outputs) node.outputs = [];
  if (!relabelOnly) {
    const want = BASE_OUTPUTS + labels.length;
    // Prune trailing slots, but never drop one that still has a link: this runs while the
    // user types, and a transient empty `paths` (select-all + retype) would otherwise
    // silently destroy their wiring. A linked orphan stays until it's disconnected.
    while (node.outputs.length > want) {
      const last = node.outputs[node.outputs.length - 1];
      if (last?.links?.length) break;
      node.removeOutput(node.outputs.length - 1);
    }
    while (node.outputs.length < want) node.addOutput(`value_${node.outputs.length - BASE_OUTPUTS + 1}`, "STRING");
  }
  // Slots past the path count (kept only because they're linked) revert to their generic
  // name, which reads as "no path drives this any more".
  for (let k = 0; k < node.outputs.length - BASE_OUTPUTS; k++) {
    const o = node.outputs[BASE_OUTPUTS + k];
    if (o) o.name = k < labels.length ? labels[k] : `value_${k + 1}`;
  }
  node.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------- `paths` edits → output sync
// Outputs are rebuilt when you finish editing — never mid-keystroke. A half-typed line would name
// a slot after itself ("name" → an output called "n"), and mid-edit slots are useless anyway: you
// can't drag a wire while the caret is in the field. So: type the whole list, click away, done.
//
// Deliberately NOT driven by the widget itself. Two dead ends learned the hard way:
//   • `widget.callback` fires once, on the FIRST keystroke — that's what produced the "n" output.
//   • there is no textarea to listen on: since frontend 1.45 the multiline STRING widget is a Vue
//     component (no `widget.inputEl`; the bundle has no `createElement("textarea")` at all), so a
//     blur/change listener had nothing to attach to and never fired.
// What IS stable across frontends is the widget's *value* and the document's focus. So: watch the
// value, and commit only while no text field has focus.

const POLL_MS = 400;

// True while the user is editing any text field — defer committing until they click away.
function isTyping() {
  const el = document.activeElement;
  if (!el) return false;
  return el.tagName === "TEXTAREA" || el.tagName === "INPUT" || el.isContentEditable === true;
}

const extractNodes = () =>
  (app.graph?._nodes || []).filter((n) => n.comfyClass === CLASS || n.type === CLASS);

// Rebuild this node's outputs if its `paths` changed since the last commit.
function checkNode(node, force) {
  const v = wv(node, "paths")?.value ?? "";
  if (v === node._kbLastPaths) return false;
  if (!force && isTyping()) return false;
  node._kbLastPaths = v;
  syncOutputs(node);
  return true;
}

function checkAll() { for (const n of extractNodes()) checkNode(n); }

// One document-level watcher for every JSON Extract node on the canvas.
let watching = false;
function startWatching() {
  if (watching) return;
  watching = true;
  // Clicking away fires focusout → commit right then, with no poll lag. The 0ms timeout lets
  // document.activeElement settle on whatever was clicked next before isTyping() reads it.
  document.addEventListener("focusout", () => setTimeout(checkAll, 0), true);
  // Backstop for value changes that arrive without a focus event (undo, programmatic edits, or a
  // frontend that only publishes widget.value late).
  setInterval(checkAll, POLL_MS);
}

// ---------------------------------------------------------------- live preview widget (Show Text recipe)
function buildPreview() {
  const root = document.createElement("div");
  root.style.cssText = "display:flex;flex-direction:column;width:100%;height:100%;box-sizing:border-box;";
  const box = document.createElement("div");
  box.style.cssText = "flex:1 1 auto;position:relative;min-height:60px;background:#181818;border:1px solid #2b2b2b;border-radius:6px;overflow:hidden;";
  const scroll = document.createElement("div");
  scroll.style.cssText = "position:absolute;inset:0;overflow:auto;padding:6px 8px;font:12px/1.5 ui-monospace,Consolas,monospace;color:#dfe7e2;white-space:pre-wrap;word-break:break-word;";
  scroll.textContent = "(run to preview extracted values)";
  box.appendChild(scroll);
  root.appendChild(box);
  root.addEventListener("wheel", (e) => e.stopPropagation());
  return { root, scroll };
}

function renderPreview(node, text) {
  const scroll = node._kbPreview;
  if (!scroll) return;
  scroll.textContent = "";
  const lines = String(text || "").split(/\r?\n/);
  for (const ln of lines) {
    const d = document.createElement("div");
    d.textContent = ln;
    if (ln.startsWith("✗") || ln.startsWith("⚠")) d.style.color = "#ff8f8f";
    else if (ln.startsWith("✔")) d.style.color = "#8be08b";
    scroll.appendChild(d);
  }
}

// ---------------------------------------------------------------- tolerant JSON parse (mirrors python)
function parseTolerant(text) {
  const s = String(text || "").trim();
  if (!s) return { obj: null, err: "empty input" };
  try { return { obj: JSON.parse(s), err: "" }; } catch (e) {}
  const m = s.match(/(\{[\s\S]*\}|\[[\s\S]*\])/);
  if (m) { try { return { obj: JSON.parse(m[1]), err: "" }; } catch (e) { return { obj: null, err: "invalid JSON" }; } }
  return { obj: null, err: "no JSON found" };
}

// ---------------------------------------------------------------- Explore JSON dialog
function joinSegs(segs) {
  return segs.join("").replace(/^\./, "");
}

function typeHint(v) {
  if (Array.isArray(v)) return `array[${v.length}]`;
  if (v === null) return "null";
  if (typeof v === "object") return `object{${Object.keys(v).length}}`;
  if (typeof v === "string") return `"${v.length > 40 ? v.slice(0, 40) + "…" : v}"`;
  return String(v);
}

function exploreDialog(node) {
  const overlay = document.createElement("div");
  Object.assign(overlay.style, { position: "fixed", inset: "0", background: "rgba(0,0,0,0.55)",
    zIndex: 10000, display: "flex", alignItems: "center", justifyContent: "center" });
  const boxEl = document.createElement("div");
  Object.assign(boxEl.style, { background: "#222", color: "#eee", border: "1px solid #444",
    borderRadius: "8px", padding: "16px", width: "640px", maxWidth: "92vw", maxHeight: "84vh",
    display: "flex", flexDirection: "column", font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" });
  const close = () => { document.removeEventListener("keydown", onKey); overlay.remove(); };
  const onKey = (e) => { if (e.key === "Escape") close(); };

  const h = document.createElement("div");
  h.textContent = "🔍 Explore JSON — click ＋ to add a path";
  Object.assign(h.style, { fontSize: "15px", fontWeight: "600", marginBottom: "8px" });

  const ta = document.createElement("textarea");
  ta.spellcheck = false;
  ta.placeholder = "Paste a sample JSON (or run the node once to load its last output), then Parse.";
  ta.style.cssText = "width:100%;box-sizing:border-box;height:110px;resize:vertical;background:#151515;color:#e6e6e6;border:1px solid #444;border-radius:6px;padding:8px;font:12px ui-monospace,Consolas,monospace;";
  ta.value = node._kbLastJson || wv(node, "json_string")?.value || "";

  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;align-items:center;gap:10px;margin:8px 0;";
  const parseBtn = document.createElement("button");
  parseBtn.textContent = "Parse ▸";
  Object.assign(parseBtn.style, { background: "#3b82f6", color: "#fff", border: "0", borderRadius: "4px", padding: "5px 12px", cursor: "pointer" });
  const status = document.createElement("span");
  status.style.cssText = "opacity:0.7;font-size:12px;";
  bar.append(parseBtn, status);

  const tree = document.createElement("div");
  tree.style.cssText = "flex:1 1 auto;overflow:auto;background:#151515;border:1px solid #333;border-radius:6px;padding:6px 8px;min-height:120px;font:12px ui-monospace,Consolas,monospace;";

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;justify-content:flex-end;margin-top:12px;";
  const closeBtn = document.createElement("button");
  closeBtn.textContent = "Close";
  Object.assign(closeBtn.style, { background: "#333", color: "#eee", border: "1px solid #555", borderRadius: "4px", padding: "6px 14px", cursor: "pointer" });
  closeBtn.onclick = close;
  footer.appendChild(closeBtn);

  boxEl.append(h, ta, bar, tree, footer);

  // Append a path line to the `paths` widget (deduped) and resync outputs.
  function addPath(pathStr) {
    const w = wv(node, "paths");
    if (!w) return;
    const cur = String(w.value || "");
    const lines = cur.split(/\r?\n/).map((l) => l.trim());
    if (lines.includes(pathStr)) { flash(status, `already added: ${pathStr}`); return; }
    w.value = (cur && !cur.endsWith("\n") ? cur + "\n" : cur) + pathStr;
    w.callback?.(w.value);
    node._kbLastPaths = w.value;      // already applied — don't let the watcher redo it
    syncOutputs(node);
    flash(status, `＋ ${pathStr}`);
  }

  function addBtn(label, title, pathStr, color) {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = title;
    Object.assign(b.style, { background: color || "#2c5", color: "#0a0a0a", border: "0", borderRadius: "3px",
      padding: "0 6px", marginLeft: "6px", cursor: "pointer", fontWeight: "700", lineHeight: "18px" });
    b.onclick = (e) => { e.stopPropagation(); addPath(pathStr); };
    return b;
  }

  // Recursive tree row. `concrete`/`wild` are segment arrays ([".key"] / ["[0]"] / ["[*]"]).
  function renderNode(container, keyLabel, value, concrete, wild, depth) {
    const path = joinSegs(concrete);
    const wildPath = joinSegs(wild);
    const isObj = value && typeof value === "object";
    const row = document.createElement("div");
    row.style.cssText = `display:flex;align-items:center;padding:2px 0;padding-left:${depth * 14}px;`;

    const key = document.createElement("span");
    key.textContent = keyLabel;
    key.style.cssText = "color:#9ecbff;";
    const hint = document.createElement("span");
    hint.textContent = "  " + typeHint(value);
    hint.style.cssText = "opacity:0.55;margin-left:6px;";
    row.append(key, hint);

    if (path) {
      row.append(addBtn("＋", `add  ${path}`, path));
      if (wildPath !== path) row.append(addBtn("[*]", `add all  ${wildPath}`, wildPath, "#e2a03f"));
    }
    container.appendChild(row);

    if (isObj && depth < 12) {
      const kids = document.createElement("div");
      const toggle = () => { kids.style.display = kids.style.display === "none" ? "" : "none"; };
      if (path) { key.style.cursor = "pointer"; key.onclick = toggle; }
      container.appendChild(kids);
      if (Array.isArray(value)) {
        value.slice(0, 200).forEach((v, i) =>
          renderNode(kids, `[${i}]`, v, [...concrete, `[${i}]`], [...wild, "[*]"], depth + 1));
      } else {
        for (const k of Object.keys(value))
          renderNode(kids, k, value[k], [...concrete, `.${k}`], [...wild, `.${k}`], depth + 1);
      }
    }
  }

  function doParse() {
    node._kbLastJson = ta.value;           // remember what the user is looking at
    const { obj, err } = parseTolerant(ta.value);
    tree.innerHTML = "";
    if (err) { status.textContent = "⚠ " + err; return; }
    status.textContent = "";
    renderNode(tree, "$ (root)", obj, [], [], 0);
  }
  parseBtn.onclick = doParse;

  overlay.appendChild(boxEl);
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
  document.addEventListener("keydown", onKey);
  document.body.appendChild(overlay);
  if (ta.value.trim()) doParse();
}

function flash(el, msg) {
  el.textContent = msg;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.textContent = ""; }, 1600);
}

// ---------------------------------------------------------------- node wiring
app.registerExtension({
  name: "Kinburg.JSONExtract",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;
    const names = nodeData.output_name || nodeData.output || [];
    const maxValues = names.filter((n) => String(n).startsWith("value_")).length || 12;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      this._kbMaxValues = maxValues;

      startWatching();

      this.addWidget("button", "🔍 Explore JSON", null, () => exploreDialog(this), { serialize: false });

      const { root, scroll } = buildPreview();
      this._kbPreview = scroll;
      this.addDOMWidget("kb_preview", "kinburg_json_preview", root, { serialize: false });

      this._kbLastPaths = wv(this, "paths")?.value ?? "";
      syncOutputs(this);
      const w = Math.max(this.size?.[0] || 0, 320);
      if ((this.size?.[1] || 0) < 300) this.setSize([w, 320]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const t = message?.kb_text?.[0];
      if (t != null) renderPreview(this, t);
      const j = message?.kb_json?.[0];
      if (j) this._kbLastJson = j;
    };

    // On load, links are restored by slot index after configure — relabel only, never restructure.
    // Baseline the watcher to the loaded value so it doesn't immediately rebuild what was saved.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      startWatching();
      this._kbLastPaths = wv(this, "paths")?.value ?? "";
      syncOutputs(this, true);
      return r;
    };
  },
});
