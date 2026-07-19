import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Ouroboros Live Log — a UI-only node that shows the loop's progress live. It has no connections;
// it just listens for the `kinburg.ouroboros` websocket events the Ouroboros node emits during
// run() (start / iteration / error / stopped / done) and appends a row per iteration with the
// image thumbnail, seed, score (+ per-criterion breakdown), full prompt, advice and negative adds.

const CLASS = "KinburgOuroborosLog";
const instances = new Set(); // live log-node instances to fan events out to

let _styled = false;
function injectStyle() {
  if (_styled) return;
  _styled = true;
  const s = document.createElement("style");
  s.textContent =
    ".ob-row{display:flex;flex-direction:column;gap:4px;padding:6px;border-radius:5px;background:#ffffff0d;}" +
    ".ob-row.ob-err{background:#8a1f1f44;}" +
    ".ob-imgwrap{position:relative;align-self:center;max-width:100%;display:inline-block;line-height:0;}" +
    ".ob-thumb{width:auto;max-width:100%;max-height:512px;object-fit:contain;border-radius:4px;" +
      "background:#000;border:1px solid #0006;}" +
    ".ob-copy{position:absolute;top:4px;right:4px;opacity:0;transition:opacity .12s;cursor:pointer;" +
      "width:24px;height:24px;padding:0;border-radius:4px;border:1px solid #0008;background:#000a;" +
      "color:#fff;font-size:13px;line-height:1;display:flex;align-items:center;justify-content:center;}" +
    ".ob-imgwrap:hover .ob-copy{opacity:1;}" +
    ".ob-copy:hover{background:#000d;}" +
    ".ob-head{color:#e5e5e5;font-weight:600;word-break:break-word;}" +
    ".ob-prompt{color:#b9b9c2;white-space:pre-wrap;word-break:break-word;}" +
    ".ob-advice{color:#8fd6c9;font-style:italic;}" +
    ".ob-neg{color:#c9b3ff;}";
  document.head.appendChild(s);
}

function setStatus(node, txt) {
  if (node._obEls) node._obEls.status.textContent = txt;
}

function clearLog(node) {
  if (node._obEls) node._obEls.list.innerHTML = "";
}

// Copy a data-URI image to the clipboard. The clipboard wants image/png, so we redraw the JPEG
// preview onto a canvas and export PNG. Needs a secure context + user gesture (the click covers it).
async function copyImageToClipboard(dataUri) {
  const img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = dataUri; });
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  canvas.getContext("2d").drawImage(img, 0, 0);
  const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
  if (!blob) throw new Error("PNG encode failed");
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}

// An image with a hover-reveal "copy to clipboard" button pinned to its top-right corner.
function makeThumb(src) {
  const wrap = document.createElement("div");
  wrap.className = "ob-imgwrap";
  const img = document.createElement("img");
  img.className = "ob-thumb";
  img.src = src;
  const btn = document.createElement("button");
  btn.className = "ob-copy";
  btn.title = "Copy image to clipboard";
  btn.textContent = "📋";
  btn.addEventListener("pointerdown", (e) => e.stopPropagation()); // don't start a node/canvas drag
  btn.onclick = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    const prev = btn.textContent;
    try {
      await copyImageToClipboard(src);
      btn.textContent = "✓";
    } catch (err) {
      console.error("[Ouroboros] copy image failed", err);
      btn.textContent = "✕";
    }
    setTimeout(() => { btn.textContent = prev; }, 1200);
  };
  wrap.append(img, btn);
  return wrap;
}

function addRow(node, d, isErr) {
  const els = node._obEls;
  if (!els) return;
  const row = document.createElement("div");
  row.className = "ob-row" + (isErr ? " ob-err" : "");

  // Image on top, full width (uncropped) — big enough to actually assess.
  if (d.thumb) row.appendChild(makeThumb(d.thumb));

  const head = document.createElement("div");
  head.className = "ob-head";
  const ts = d.ts ? `[${d.ts}] ` : "";
  if (isErr) {
    head.textContent = `${ts}#${d.i}  ⚠ critic failed — stopped   seed ${d.seed ?? "?"}`;
  } else {
    const scores = d.scores && typeof d.scores === "object" ? d.scores : {};
    const brk = Object.keys(scores).length
      ? "  (" + Object.entries(scores).map(([k, v]) => `${k} ${v}`).join(" · ") + ")"
      : "";
    head.textContent =
      `${ts}#${d.i}/${d.total}  ★${d.score}/${d.score_max}${brk}   seed ${d.seed}   ${d.gen_seconds}s` +
      (d.is_best ? "   ← best" : "");
  }
  row.appendChild(head);

  if (d.prompt) {
    const p = document.createElement("div");
    p.className = "ob-prompt";
    p.textContent = d.prompt;   // full prompt, wraps; the whole log scrolls (no cramped inner bar)
    row.appendChild(p);
  }
  if (isErr && d.message) {
    const m = document.createElement("div");
    m.className = "ob-advice";
    m.textContent = d.message;
    row.appendChild(m);
  }
  if (d.advice) {
    const a = document.createElement("div");
    a.className = "ob-advice";
    a.textContent = "↳ " + d.advice;
    row.appendChild(a);
  }
  if (Array.isArray(d.negative_add) && d.negative_add.length) {
    const ne = document.createElement("div");
    ne.className = "ob-neg";
    ne.textContent = "⊖ negative += " + d.negative_add.join(", ");
    row.appendChild(ne);
  }

  els.list.appendChild(row);
  els.scroll.scrollTop = els.scroll.scrollHeight; // stick to the latest
}

// Per-step mode: one compact, timestamped row per stage (prompt / image / verdict).
function addStageRow(node, d) {
  const els = node._obEls;
  if (!els) return;
  const row = document.createElement("div");
  row.className = "ob-row";
  const ts = d.ts ? `[${d.ts}] ` : "";
  const tag = `#${d.i}/${d.total}`;
  const head = document.createElement("div");
  head.className = "ob-head";

  if (d.stage === "prompt") {
    head.textContent = `${ts}${tag}  ✎ prompt   seed ${d.seed}`;
    row.appendChild(head);
    if (d.prompt) {
      const p = document.createElement("div");
      p.className = "ob-prompt";
      p.textContent = d.prompt;
      row.appendChild(p);
    }
  } else if (d.stage === "image") {
    head.textContent = `${ts}${tag}  🖼 image · ${d.gen_seconds}s`;
    row.appendChild(head);
    if (d.thumb) row.appendChild(makeThumb(d.thumb));
  } else if (d.stage === "verdict") {
    const scores = d.scores && typeof d.scores === "object" ? d.scores : {};
    const brk = Object.keys(scores).length
      ? "  (" + Object.entries(scores).map(([k, v]) => `${k} ${v}`).join(" · ") + ")"
      : "";
    head.textContent = `${ts}${tag}  ★${d.score}/${d.score_max}${brk}` + (d.is_best ? "   ← best" : "");
    row.appendChild(head);
    if (d.advice) {
      const a = document.createElement("div");
      a.className = "ob-advice";
      a.textContent = "↳ " + d.advice;
      row.appendChild(a);
    }
    if (Array.isArray(d.negative_add) && d.negative_add.length) {
      const ne = document.createElement("div");
      ne.className = "ob-neg";
      ne.textContent = "⊖ negative += " + d.negative_add.join(", ");
      row.appendChild(ne);
    }
  }

  els.list.appendChild(row);
  els.scroll.scrollTop = els.scroll.scrollHeight;
}

// Render one event into a node's DOM. Pure (no storage) so it can be replayed on restore.
function applyEvent(node, d) {
  if (!node._obEls) return;
  const ts = d.ts ? `[${d.ts}] ` : "";
  switch (d.type) {
    case "start":     clearLog(node); setStatus(node, `${ts}running… (${d.total} iterations)`); break;
    case "stage":     addStageRow(node, d); setStatus(node, `${ts}#${d.i}/${d.total} · ${d.stage}…`); break;
    case "iteration": addRow(node, d, false); setStatus(node, `${ts}iteration ${d.i}/${d.total}…`); break;
    case "error":     addRow(node, d, true); setStatus(node, `${ts}⚠ critic failed at #${d.i} — stopped`); break;
    case "stopped":   setStatus(node, `${ts}⏹ stopped at #${d.i}`); break;
    case "done":      setStatus(node, `${ts}done — ${d.iterations} iteration(s), best ★${d.best_score}`); break;
  }
}

// Per-node event history (keyed by node id), kept in memory only — NOT serialized into the
// workflow (thumbnails would bloat it). This lets the log survive ComfyUI Desktop tab switches
// (which destroy & recreate the node): on recreate we replay the stored events. Reset each run
// (on "start"). Lost on full app restart — acceptable; the concern is tab switching.
const logStore = new Map();

function restore(node) {
  const evs = logStore.get(node.id);
  if (!node._obEls || !evs || !evs.length) return;
  clearLog(node);
  for (const d of evs) applyEvent(node, d);
}

// One shared websocket listener fans out to every live log node on the canvas.
api.addEventListener("kinburg.ouroboros", (e) => {
  const d = (e && e.detail) || {};
  for (const node of instances) {
    if (!node._obEls) continue;
    applyEvent(node, d);
    // Record for restore-after-tab-switch. "start" resets the history for this run.
    if (d.type === "start") {
      logStore.set(node.id, [d]);
    } else {
      const arr = logStore.get(node.id);
      if (arr) arr.push(d); else logStore.set(node.id, [d]);
    }
  }
});

app.registerExtension({
  name: "Kinburg.OuroborosLog",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const node = this;
      injectStyle();

      // Mirror the Show Text (Markdown) node so the node stays a FIXED height and the content
      // scrolls (rather than the node growing per row): root is height:100% (fills the widget
      // area ComfyUI gives it), a flex:1 + min-height:0 wrapper scrolls, and addDOMWidget is
      // passed NO height options. (min-height:0 is the flexbox trick that lets the child scroll
      // instead of expanding to its content.)
      const root = document.createElement("div");
      root.className = "ob-log";
      root.style.cssText =
        "display:flex;flex-direction:column;width:100%;height:100%;box-sizing:border-box;" +
        "gap:4px;padding:4px;font-family:inherit;font-size:11px;background:#00000022;border-radius:6px;";
      // Let the canvas keep panning/zooming outside our interactive area.
      root.addEventListener("wheel", (e) => e.stopPropagation());
      root.addEventListener("pointerdown", (e) => e.stopPropagation());

      const status = document.createElement("div");
      status.style.cssText =
        "flex:0 0 auto;color:#9a9aa2;font-size:10px;padding:1px 4px;text-transform:uppercase;letter-spacing:.04em;";
      status.textContent = "waiting for a run…";

      // Show Text's exact trick: a flex:1 box (position:relative, overflow:hidden) whose scrolling
      // content sits in an ABSOLUTELY positioned layer (inset:0). Being out of flow, the rows never
      // inflate the root's measured offsetHeight — which is what ComfyUI reads to size the node — so
      // the node keeps its height and the content scrolls instead of growing it.
      const box = document.createElement("div");
      box.style.cssText = "flex:1 1 auto;position:relative;min-height:80px;overflow:hidden;";

      const scroll = document.createElement("div");
      scroll.style.cssText = "position:absolute;inset:0;overflow-y:auto;";

      const list = document.createElement("div");
      list.style.cssText = "display:flex;flex-direction:column;gap:6px;";

      scroll.appendChild(list);
      box.appendChild(scroll);
      root.append(status, box);
      node._obEls = { root, box, scroll, list, status };
      node.addDOMWidget("ob_log", "kinburg_ouroboros_log", root, { serialize: false });
      if ((node.size?.[1] || 0) < 300) {
        node.setSize([Math.max(node.size?.[0] || 0, 420), 560]);
      }
      instances.add(node);
      restore(node);   // best-effort (id may not be final yet); onConfigure restores reliably
      return r;
    };

    // Fires on workflow load / tab switch, AFTER the node's saved id is restored — replay the
    // in-memory event history so the log isn't blank when you switch back to this tab.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      restore(this);
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      instances.delete(this);
      return onRemoved?.apply(this, arguments);
    };
  },
});
