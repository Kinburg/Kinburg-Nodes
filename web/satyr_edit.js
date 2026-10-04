import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { openEditor } from "./satyr_roll.js";

// Satyr Edit (Plan → Plan) — the node side of the piano-roll editor in satyr_roll.js.
//
// The node carries two plans in one JSON string, `plan_state`: `upstream`, the last plan that came
// in on `abc` (sent back by the node after every run, and kept in the workflow so a reload does not
// force the upstream to run again), and `edited`, what was last saved in the editor. That string IS
// the backend input — the DOM widget below is named after it and serialises it — so the node has no
// hidden carrier rows (see web/chat_llm.js for why that matters).
//
// The editor opens on what the node will send out: the edited plan while 'use_edited' is on, the
// upstream plan otherwise. Saving stores the edit and turns 'use_edited' on.

const CLASS = "KinburgSatyrEdit";

const FIELDS = ["upstream", "upstreamSummary", "edited", "editedSummary", "editedReport", "lyrics"];

function ST(node) {
  if (!node._kbPlan) node._kbPlan = { ...Object.fromEntries(FIELDS.map((k) => [k, ""])), voices: [] };
  return node._kbPlan;
}

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function load(node, value) {
  let parsed = {};
  try { parsed = JSON.parse(value || "{}") || {}; } catch (e) { parsed = {}; }
  const st = ST(node);
  for (const k of FIELDS) st[k] = String(parsed[k] || "");
  st.voices = Array.isArray(parsed.voices) ? parsed.voices : [];
  render(node);
}

async function post(path, body) {
  const r = await api.fetchApi(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let j = null;
  try { j = await r.json(); } catch (e) { /* not JSON */ }
  if (!j || !j.ok) throw new Error(j?.error || `${r.status} ${r.statusText}`);
  return j;
}

function frozen(node) {
  return !!widget(node, "use_edited")?.value;
}

function render(node) {
  const ui = node._kbUi;
  if (!ui) return;
  const st = ST(node);
  const useEdit = frozen(node) && st.edited;
  ui.status.textContent = useEdit
    ? "🔒 sending the edited plan" + (st.editedSummary ? " — " + st.editedSummary : "")
    : st.upstream
      ? "sending the upstream plan" + (st.upstreamSummary ? " — " + st.upstreamSummary : "")
      : "no plan yet — run the graph up to this node";
  ui.note.textContent = st.edited && !useEdit ? "an edit is kept on the node — turn 'use_edited' on to send it" : "";
  ui.saved.textContent = st.edited && st.editedReport ? "last save: " + st.editedReport : "";
  const lines = st.lyrics.split("\n").filter((l) => l.trim() && !/^\s*\[.*\]\s*$/.test(l)).length;
  const names = st.voices.map((v) => v.name).filter(Boolean);
  ui.words.textContent = st.lyrics.trim()
    ? `lyrics: ${lines} line${lines === 1 ? "" : "s"}` + (names.length ? ` · voices: ${names.join(", ")}` : "")
    : "";
  ui.discard.style.display = st.edited ? "" : "none";
  ui.open.disabled = !(st.upstream || st.edited);
}

async function openFor(node) {
  const st = ST(node);
  const base = frozen(node) && st.edited ? st.edited : st.upstream || st.edited;
  if (!base) {
    alert("No plan on this node yet — run the graph up to it once, then open the editor.");
    return;
  }
  let model;
  try {
    ({ model } = await post("/kinburg/satyr/edit/load", { abc: base }));
  } catch (e) {
    alert("Satyr Edit could not read the plan: " + e.message);
    return;
  }
  openEditor({
    model,
    base,
    title: node.title || "Satyr Edit",
    lyrics: st.lyrics,
    words: (edited, from) => post("/kinburg/satyr/edit/words",
                                  { base: from, model: edited, lyrics: st.lyrics, voices: st.voices }),
    save: async (edited, from) => {
      const got = await post("/kinburg/satyr/edit/save", { base: from, model: edited });
      st.edited = got.abc;
      st.editedSummary = got.report[0] || "";
      st.editedReport = got.report.slice(1).join(" · ");
      const w = widget(node, "use_edited");
      if (w && !w.value) { w.value = true; w.callback?.(true); }
      render(node);
      node.setDirtyCanvas?.(true, true);
      node.graph?.change?.();
      return got;
    },
  });
}

function setup(node) {
  // Drop the auto-created text row for plan_state: the panel below carries the value itself.
  for (let i = node.widgets.length - 1; i >= 0; i--) {
    if (node.widgets[i].name === "plan_state") node.widgets.splice(i, 1);
  }
  ST(node);

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;padding:4px 2px;font:12px system-ui,sans-serif;color:#c9ced8;";
  const open = document.createElement("button");
  open.textContent = "✏ Open the editor";
  open.style.cssText = "background:#2b3140;color:#e3e7ee;border:1px solid #3d4556;border-radius:6px;padding:6px 10px;cursor:pointer;font:600 12px system-ui,sans-serif;";
  const status = document.createElement("div");
  status.style.cssText = "color:#9aa3b2;line-height:1.35;";
  const note = document.createElement("div");
  note.style.cssText = "color:#c9a45c;line-height:1.35;";
  const saved = document.createElement("div");
  saved.style.cssText = "color:#7fbf8f;line-height:1.35;font-size:11px;";
  const words = document.createElement("div");
  words.style.cssText = "color:#9fb4d8;line-height:1.35;font-size:11px;";
  const discard = document.createElement("a");
  discard.textContent = "discard the edit";
  discard.href = "#";
  discard.style.cssText = "color:#8e96a5;font-size:11px;align-self:flex-start;";
  wrap.append(open, status, note, saved, words, discard);
  for (const e of ["pointerdown", "mousedown", "wheel"]) wrap.addEventListener(e, (ev) => ev.stopPropagation());
  open.addEventListener("click", () => openFor(node));
  discard.addEventListener("click", (e) => {
    e.preventDefault();
    if (!confirm("Discard the edited plan kept on this node?")) return;
    const st = ST(node);
    st.edited = "";
    st.editedSummary = "";
    st.editedReport = "";
    const w = widget(node, "use_edited");
    if (w?.value) { w.value = false; w.callback?.(false); }
    render(node);
    node.graph?.change?.();
  });
  node._kbUi = { open, status, note, saved, words, discard };

  // Named after the backend input on purpose: graphToPrompt reads inputs[widget.name] = value, so
  // this panel is both the UI and the carrier of plan_state.
  node.addDOMWidget("plan_state", "kinburg_satyr_edit", wrap, {
    serialize: true,
    getValue: () => JSON.stringify(ST(node)),
    setValue: (v) => load(node, v),
    getMinHeight: () => 124,
    getMaxHeight: () => 180,
  });

  const w = widget(node, "use_edited");
  if (w) {
    const prev = w.callback;
    w.callback = function (...a) { const r = prev?.apply(this, a); render(node); return r; };
  }
  render(node);
}

app.registerExtension({
  name: "Kinburg.SatyrEdit",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      setup(this);
      return r;
    };

    // After a run the node sends the plan that came in (null when it was frozen and did not ask),
    // and the lyrics and voices wired to it (empty when they are not).
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const got = message?.satyr_plan?.[0];
      if (!got) return;
      const st = ST(this);
      if (got.upstream != null) {
        st.upstream = got.upstream;
        st.upstreamSummary = got.summary || "";
      }
      if (got.lyrics != null) {
        st.lyrics = String(got.lyrics);
        st.voices = Array.isArray(got.voices) ? got.voices : [];
      }
      render(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      render(this);
      return r;
    };
  },
});
