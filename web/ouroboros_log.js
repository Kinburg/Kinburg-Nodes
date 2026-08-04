import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Ouroboros Live Log — a UI-only node that shows the loop's progress live. It has no connections;
// it just listens for the `kinburg.ouroboros` websocket events the Ouroboros node emits during
// run() (start / iteration / error / stopped / done) and appends a row per iteration with the
// image thumbnail, seed, score (+ per-criterion breakdown), full prompt, advice and negative adds.
//
// Scrolling follows new rows ONLY while you are already parked at the bottom. Scroll up and the
// view stays put while the loop keeps running; a "↓ latest" pill takes you back.

const CLASS = "KinburgOuroborosLog";
const STICK_SLACK = 24;      // px from the bottom that still counts as "at the bottom"
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
    ".ob-head{display:flex;align-items:baseline;gap:6px;color:#e5e5e5;font-weight:600;}" +
    ".ob-htxt{flex:1 1 auto;min-width:0;word-break:break-word;}" +
    ".ob-copytxt{flex:0 0 auto;opacity:0;transition:opacity .12s;cursor:pointer;padding:1px 5px;" +
      "border-radius:3px;border:1px solid #0008;background:#0006;color:#dcdce4;font-size:10px;" +
      "line-height:1.3;font-weight:400;}" +
    ".ob-row:hover .ob-copytxt{opacity:1;}" +
    ".ob-copytxt:hover{background:#000a;}" +
    ".ob-prompt{color:#b9b9c2;white-space:pre-wrap;word-break:break-word;}" +
    ".ob-advice{color:#8fd6c9;font-style:italic;}" +
    ".ob-neg{color:#c9b3ff;}" +
    ".ob-ctx{color:#8a8a94;font-size:10px;font-family:ui-monospace,Consolas,monospace;word-break:break-word;}" +
    ".ob-ctx.warn{color:#e0a648;}" +
    ".ob-jump{position:absolute;right:12px;bottom:8px;z-index:5;cursor:pointer;padding:3px 9px;" +
      "border-radius:11px;border:1px solid #0008;background:#2b2b33f0;color:#dcdce4;font-size:10px;" +
      "line-height:1.3;box-shadow:0 2px 6px #0007;}" +
    ".ob-jump:hover{background:#3a3a44f0;}" +
    ".ob-jump.hot{border-color:#8fd6c999;color:#8fd6c9;}" +
    ".ob-clear{flex:0 0 auto;cursor:pointer;padding:1px 6px;border-radius:3px;border:1px solid #ffffff1f;" +
      "background:transparent;color:#9a9aa2;font-size:9px;text-transform:uppercase;letter-spacing:.04em;}" +
    ".ob-clear:hover{background:#ffffff14;color:#dcdce4;}";
  document.head.appendChild(s);
}

// A row header with a hover-reveal "copy" button. `textFn` produces the clipboard payload lazily,
// so a row that is still being written copies whatever it holds at the moment you click; pass
// nothing for a row with no text worth copying (the image stage).
function makeHead(textFn) {
  const head = document.createElement("div");
  head.className = "ob-head";
  const txt = document.createElement("span");
  txt.className = "ob-htxt";
  head.appendChild(txt);
  if (textFn) {
    const btn = document.createElement("button");
    btn.className = "ob-copytxt";
    btn.textContent = "copy";
    btn.title = "Copy this entry's text";
    btn.addEventListener("pointerdown", (e) => e.stopPropagation()); // don't start a node/canvas drag
    btn.onclick = async (e) => {
      e.preventDefault();
      e.stopPropagation();
      try {
        await navigator.clipboard.writeText(textFn() || "");
        btn.textContent = "✓";
      } catch (err) {
        console.error("[Ouroboros] copy text failed", err);
        btn.textContent = "✕";
      }
      setTimeout(() => { btn.textContent = "copy"; }, 1200);
    };
    head.appendChild(btn);
  }
  return { head, txt };
}

// ── scrolling ───────────────────────────────────────────────────────────────────────────────
// Auto-follow is opt-out by position: we only pin to the bottom while the view IS at the bottom,
// so scrolling up mid-run freezes it (nothing above moves — rows/text are appended below).

function atBottom(node) {
  const s = node._obEls?.scroll;
  if (!s) return true;
  return s.scrollHeight - s.scrollTop - s.clientHeight <= STICK_SLACK;
}

function showJump(node, visible, hot) {
  const e = node._obEls;
  if (!e || !e.jump) return;
  e.jump.style.display = visible ? "" : "none";
  if (!visible) e.jump.classList.remove("hot");
  else if (hot) e.jump.classList.add("hot");
}

// Follow new content only if the user hasn't scrolled away; otherwise flag the pill as "hot".
function stick(node) {
  const e = node._obEls;
  if (!e) return;
  if (node._obStick === false) showJump(node, true, true);
  else e.scroll.scrollTop = e.scroll.scrollHeight;
}

// Snap back to the bottom and resume following (pill click / fresh render).
function toBottom(node) {
  const e = node._obEls;
  if (!e) return;
  node._obStick = true;
  e.scroll.scrollTop = e.scroll.scrollHeight;
  showJump(node, false);
}

// One-line context-fill summary for an LLM call (enhancer / critic), or "" if no data.
// `c` is the compact dict the Ouroboros node emits: {prompt, output, used, n_ctx, pct, truncated, over}.
function fmtCtx(c, label) {
  if (!c || !c.n_ctx) return "";
  let s = `ⓘ ${label} ctx ${c.used}/${c.n_ctx} (${c.pct}%) · prompt ${c.prompt} + gen ${c.output}`;
  const warn = [];
  if (c.truncated) warn.push("output truncated — raise max_tokens");
  if (c.over || c.tight) warn.push("prompt ≈ context limit — raise n_ctx");
  if (warn.length) s += "  ⚠ " + warn.join("; ");
  return s;
}

function addCtx(row, c, label) {
  const txt = fmtCtx(c, label);
  if (!txt) return;
  const d = document.createElement("div");
  d.className = "ob-ctx" + (c.truncated || c.over || c.tight ? " warn" : "");
  d.textContent = txt;
  row.appendChild(d);
}

function setStatus(node, txt) {
  if (node._obEls) node._obEls.statusTxt.textContent = txt;
}

function clearLog(node) {
  if (node._obEls) node._obEls.list.innerHTML = "";
  node._obStream = null;   // drop any in-progress streaming prompt row
  toBottom(node);          // an emptied log follows again (a new run shouldn't stay frozen)
}

// 'streaming' mode: the enhancer prompt types out live into a single growing row. The head keeps
// a running token count against the enhancer's max_tokens ceiling (one delta == one token) plus
// the generation rate, so you can see how much of the budget the rewrite is eating as it happens.
function streamHead(st) {
  const bits = [`${st.ts}#${st.i}/${st.total}  ✎ prompt`, `seed ${st.seed}`];
  if (st.tokens) bits.push(st.max ? `${st.tokens}/${st.max} tok` : `${st.tokens} tok`);
  else if (st.max) bits.push(`0/${st.max} tok`);
  if (st.t0 && st.tokens > 2) {
    const secs = (performance.now() - st.t0) / 1000;
    if (secs > 0.2) bits.push(`${((st.tokens - 1) / secs).toFixed(1)} tok/s`);
  }
  st.head.textContent = bits.join("   ") + "  …";
}

function openPromptStream(node, d) {
  const els = node._obEls;
  if (!els) return;
  const row = document.createElement("div");
  row.className = "ob-row";
  const p = document.createElement("div");
  p.className = "ob-prompt";
  p.textContent = "";
  const { head, txt } = makeHead(() => p.textContent);   // copies the prompt as written so far
  row.append(head, p);
  els.list.appendChild(row);
  node._obStream = {
    i: d.i, row, head: txt, promptEl: p, seed: d.seed, total: d.total,
    ts: d.ts ? `[${d.ts}] ` : "", tokens: 0, max: Number(d.max_tokens) || 0, t0: 0,
  };
  streamHead(node._obStream);
  stick(node);
}

function appendPromptDelta(node, d) {
  const st = node._obStream;
  if (!st || st.i !== d.i || !node._obEls) return;
  st.promptEl.textContent += (d.delta || "");
  st.tokens += 1;
  if (!st.t0) st.t0 = performance.now();
  streamHead(st);
  stick(node);
}

// Finalize the streaming row: swap in the authoritative prompt (cleaned + triggers appended) and
// add the enhancer context line, matching a normal per-step prompt row.
function finalizePromptStream(node, d) {
  const st = node._obStream;
  if (!st) return;
  const ts = d.ts ? `[${d.ts}] ` : "";
  const gen = Number(d.ctx?.output) || st.tokens;   // worker's exact count, else the deltas we saw
  const tok = gen ? (st.max ? `   ${gen}/${st.max} tok` : `   ${gen} tok`) : "";
  st.head.textContent = `${ts}#${d.i}/${d.total}  ✎ prompt   seed ${d.seed ?? st.seed}${tok}`;
  if (d.prompt) st.promptEl.textContent = d.prompt;
  addCtx(st.row, d.ctx, "enhancer");
  node._obStream = null;
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

  const parts = [];   // what the row's copy button hands over (the text you can actually see)
  const { head, txt } = makeHead(() => parts.join("\n\n"));
  const ts = d.ts ? `[${d.ts}] ` : "";
  if (isErr) {
    txt.textContent = `${ts}#${d.i}  ⚠ critic failed — stopped   seed ${d.seed ?? "?"}`;
  } else {
    const scores = d.scores && typeof d.scores === "object" ? d.scores : {};
    const brk = Object.keys(scores).length
      ? "  (" + Object.entries(scores).map(([k, v]) => `${k} ${v}`).join(" · ") + ")"
      : "";
    txt.textContent =
      `${ts}#${d.i}/${d.total}  ★${d.score}/${d.score_max}${brk}   seed ${d.seed}   ${d.gen_seconds}s` +
      (d.is_best ? "   ← best" : "");
  }
  row.appendChild(head);

  if (d.prompt) {
    const p = document.createElement("div");
    p.className = "ob-prompt";
    p.textContent = d.prompt;   // full prompt, wraps; the whole log scrolls (no cramped inner bar)
    row.appendChild(p);
    parts.push(d.prompt);
  }
  if (isErr && d.message) {
    const m = document.createElement("div");
    m.className = "ob-advice";
    m.textContent = d.message;
    row.appendChild(m);
    parts.push(d.message);
  }
  if (d.advice) {
    const a = document.createElement("div");
    a.className = "ob-advice";
    a.textContent = "↳ " + d.advice;
    row.appendChild(a);
    parts.push("↳ " + d.advice);
  }
  if (Array.isArray(d.negative_add) && d.negative_add.length) {
    const ne = document.createElement("div");
    ne.className = "ob-neg";
    ne.textContent = "⊖ negative += " + d.negative_add.join(", ");
    row.appendChild(ne);
    parts.push(ne.textContent);
  }
  addCtx(row, d.ctx, "enhancer");        // per-iteration (and error) rows carry the enhancer fill
  addCtx(row, d.ctx_crit, "critic");     // and the critic fill (per-iteration mode only)

  els.list.appendChild(row);
  stick(node);                            // follow the latest only if the view is at the bottom
}

// Per-step mode: one compact, timestamped row per stage (prompt / image / verdict).
function addStageRow(node, d) {
  const els = node._obEls;
  if (!els) return;
  const row = document.createElement("div");
  row.className = "ob-row";
  const ts = d.ts ? `[${d.ts}] ` : "";
  const tag = `#${d.i}/${d.total}`;
  const parts = [];   // clipboard payload for this row's copy button (an image row gets none)
  const { head, txt } = makeHead(d.stage === "image" ? null : () => parts.join("\n\n"));

  if (d.stage === "prompt") {
    txt.textContent = `${ts}${tag}  ✎ prompt   seed ${d.seed}`;
    row.appendChild(head);
    if (d.prompt) {
      const p = document.createElement("div");
      p.className = "ob-prompt";
      p.textContent = d.prompt;
      row.appendChild(p);
      parts.push(d.prompt);
    }
    addCtx(row, d.ctx, "enhancer");
  } else if (d.stage === "image") {
    txt.textContent = `${ts}${tag}  🖼 image · ${d.gen_seconds}s`;
    row.appendChild(head);
    if (d.thumb) row.appendChild(makeThumb(d.thumb));
  } else if (d.stage === "verdict") {
    const scores = d.scores && typeof d.scores === "object" ? d.scores : {};
    const brk = Object.keys(scores).length
      ? "  (" + Object.entries(scores).map(([k, v]) => `${k} ${v}`).join(" · ") + ")"
      : "";
    txt.textContent = `${ts}${tag}  ★${d.score}/${d.score_max}${brk}` + (d.is_best ? "   ← best" : "");
    row.appendChild(head);
    if (d.advice) {
      const a = document.createElement("div");
      a.className = "ob-advice";
      a.textContent = "↳ " + d.advice;
      row.appendChild(a);
      parts.push("↳ " + d.advice);
    }
    if (Array.isArray(d.negative_add) && d.negative_add.length) {
      const ne = document.createElement("div");
      ne.className = "ob-neg";
      ne.textContent = "⊖ negative += " + d.negative_add.join(", ");
      row.appendChild(ne);
      parts.push(ne.textContent);
    }
    addCtx(row, d.ctx, "critic");
  }

  els.list.appendChild(row);
  stick(node);
}

// Render one event into a node's DOM. Pure (no storage) so it can be replayed on restore.
function applyEvent(node, d) {
  if (!node._obEls) return;
  const ts = d.ts ? `[${d.ts}] ` : "";
  switch (d.type) {
    case "start":     clearLog(node); setStatus(node, `${ts}running… (${d.total} iterations)`); break;
    case "prompt_delta": appendPromptDelta(node, d); break;
    case "stage":
      if (d.stage === "prompt" && d.open) openPromptStream(node, d);
      else if (d.stage === "prompt" && node._obStream && node._obStream.i === d.i) finalizePromptStream(node, d);
      else addStageRow(node, d);
      setStatus(node, `${ts}#${d.i}/${d.total} · ${d.stage}…`);
      break;
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
  toBottom(node);   // a fresh render starts parked at the newest row, following again
}

// One shared websocket listener fans out to every live log node on the canvas.
api.addEventListener("kinburg.ouroboros", (e) => {
  const d = (e && e.detail) || {};
  for (const node of instances) {
    if (!node._obEls) continue;
    applyEvent(node, d);
    // Record for restore-after-tab-switch. "start" resets the history for this run. The live-only
    // streaming events (per-token deltas + the empty 'open' prompt row) are NOT stored — on replay
    // the finalized prompt stage renders the full prompt row instead, so history stays compact.
    if (d.type === "start") {
      logStore.set(node.id, [d]);
    } else if (d.type === "prompt_delta" || (d.type === "stage" && d.stage === "prompt" && d.open)) {
      /* transient — skip */
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
      status.style.cssText = "flex:0 0 auto;display:flex;align-items:center;gap:6px;padding:1px 4px;";
      const statusTxt = document.createElement("span");
      statusTxt.style.cssText =
        "flex:1 1 auto;min-width:0;color:#9a9aa2;font-size:10px;text-transform:uppercase;letter-spacing:.04em;" +
        "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      statusTxt.textContent = "waiting for a run…";
      const clear = document.createElement("button");
      clear.className = "ob-clear";
      clear.textContent = "clear";
      clear.title = "Remove every entry from this log";
      clear.addEventListener("pointerdown", (e) => e.stopPropagation());
      clear.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        logStore.set(node.id, []);   // history too, so a tab switch doesn't replay it
        clearLog(node);
        setStatus(node, "waiting for a run…");
      };
      status.append(statusTxt, clear);

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

      // "Back to the newest row" pill — only visible once you've scrolled away from the bottom.
      const jump = document.createElement("button");
      jump.className = "ob-jump";
      jump.textContent = "↓ latest";
      jump.style.display = "none";
      jump.addEventListener("pointerdown", (e) => e.stopPropagation());
      jump.onclick = (e) => { e.preventDefault(); e.stopPropagation(); toBottom(node); };

      scroll.appendChild(list);
      box.append(scroll, jump);
      root.append(status, box);
      node._obEls = { root, box, scroll, list, status, statusTxt, jump };
      node._obStick = true;

      // Any scroll (wheel, drag, keyboard) re-decides whether we follow: parked at the bottom →
      // follow; anywhere above → freeze. Programmatic scrolls land at the bottom, so they keep it on.
      scroll.addEventListener("scroll", () => {
        const bottom = atBottom(node);
        node._obStick = bottom;
        showJump(node, !bottom, false);
      });

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
