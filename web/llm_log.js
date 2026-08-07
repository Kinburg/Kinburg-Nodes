import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Kinburg Live Log — a UI-only node that shows what the pack's LLM nodes are writing, as they
// write it. It has no connections; it listens on the `kinburg.llm` and `kinburg.chatllm` websocket
// channels (start / delta / done / frames) and grows a block per generation, token by token. One
// log shows every source on the canvas, each block labelled by its node and, when the source says
// so, by the individual call ("Morpheus Storyboard · shot 2/4 (2 keyframes)").
//
// Blocks can carry IMAGES: the frames a vision call was actually shown, or a `frames` event on its
// own (Morpheus sends each shot's last frame after decoding it, which makes the log a live
// storyboard of the run). Thumbnails have a hover copy-to-clipboard button.
//
// Scrolling follows the newest text ONLY while you are already parked at the bottom. Scroll up
// and the view stays put while generation continues; a "↓ latest" pill takes you back.

// `KinburgLLMLog` is the older, LLM-only id: the same renderer drives it, so graphs saved against
// it keep working and gain the images.
const CLASSES = new Set(["KinburgLiveLog", "KinburgLLMLog"]);
const MAX_BLOCKS = 25;                 // keep the last N generations; older ones drop off the top
const STICK_SLACK = 24;                // px from the bottom that still counts as "at the bottom"
const instances = new Set();           // live log-node instances to fan events out to

// Per-log-node block history, kept in memory only (NOT serialized). Survives ComfyUI Desktop tab
// switches (which destroy & recreate the node): on recreate we rebuild the DOM from these blocks.
// Lost on full app restart — acceptable, the concern is tab switching.
const snapStore = new Map();           // logNodeId -> [block]
const snapFor = (id) => { let a = snapStore.get(id); if (!a) { a = []; snapStore.set(id, a); } return a; };

let _styled = false;
function injectStyle() {
  if (_styled) return;
  _styled = true;
  const s = document.createElement("style");
  s.textContent =
    ".llm-block{display:flex;flex-direction:column;gap:3px;padding:6px;border-radius:5px;background:#ffffff0d;}" +
    ".llm-head{display:flex;align-items:baseline;gap:6px;color:#9ecbff;font-weight:600;font-size:10px;" +
      "text-transform:uppercase;letter-spacing:.03em;}" +
    ".llm-head.done{color:#7CFC7C;}" +
    ".llm-head.warn{color:#e0a648;}" +
    ".llm-htxt{flex:1 1 auto;min-width:0;word-break:break-word;}" +
    ".llm-copy{flex:0 0 auto;opacity:0;transition:opacity .12s;cursor:pointer;padding:1px 5px;" +
      "border-radius:3px;border:1px solid #0008;background:#0006;color:#dcdce4;font-size:10px;line-height:1.3;}" +
    ".llm-block:hover .llm-copy{opacity:1;}" +
    ".llm-copy:hover{background:#000a;}" +
    ".llm-body{color:#dcdce4;white-space:pre-wrap;word-break:break-word;user-select:text;" +
      "font-family:ui-monospace,Consolas,monospace;font-size:11px;line-height:1.4;}" +
    ".llm-bar{height:3px;border-radius:2px;background:#ffffff14;overflow:hidden;}" +
    ".llm-bar-fill{height:100%;width:0;background:#4b9fff;transition:width .12s linear;}" +
    ".llm-bar-fill.done{background:#7CFC7C;}" +
    ".llm-bar-fill.warn{background:#e0a648;}" +
    ".llm-think{display:flex;flex-direction:column;gap:2px;border-left:2px solid #ffffff1f;padding-left:6px;}" +
    ".llm-think-head{color:#8a8a94;font-size:10px;cursor:pointer;user-select:none;letter-spacing:.03em;}" +
    ".llm-think-head:hover{color:#b9b9c2;}" +
    ".llm-think-body{color:#8a8a94;white-space:pre-wrap;word-break:break-word;user-select:text;font-style:italic;" +
      "font-family:ui-monospace,Consolas,monospace;font-size:10px;line-height:1.35;}" +
    ".llm-ctx{color:#8a8a94;font-size:10px;font-family:ui-monospace,Consolas,monospace;word-break:break-word;}" +
    ".llm-ctx.warn{color:#e0a648;}" +
    ".llm-jump{position:absolute;right:12px;bottom:8px;z-index:5;cursor:pointer;padding:3px 9px;" +
      "border-radius:11px;border:1px solid #0008;background:#2b2b33f0;color:#dcdce4;font-size:10px;" +
      "line-height:1.3;box-shadow:0 2px 6px #0007;}" +
    ".llm-jump:hover{background:#3a3a44f0;}" +
    ".llm-jump.hot{border-color:#7CFC7C99;color:#7CFC7C;}" +
    ".llm-clear{flex:0 0 auto;cursor:pointer;padding:1px 6px;border-radius:3px;border:1px solid #ffffff1f;" +
      "background:transparent;color:#9a9aa2;font-size:9px;text-transform:uppercase;letter-spacing:.04em;}" +
    ".llm-clear:hover{background:#ffffff14;color:#dcdce4;}" +
    ".llm-imgs{display:flex;flex-wrap:wrap;gap:4px;}" +
    ".llm-imgwrap{position:relative;display:inline-block;line-height:0;}" +
    ".llm-thumb{max-width:100%;max-height:180px;border-radius:4px;display:block;}" +
    ".llm-imgcopy{position:absolute;top:3px;right:3px;opacity:0;transition:opacity .12s;cursor:pointer;" +
      "padding:1px 4px;border-radius:3px;border:1px solid #0008;background:#0009;color:#dcdce4;font-size:10px;" +
      "line-height:1.2;}" +
    ".llm-imgwrap:hover .llm-imgcopy{opacity:1;}" +
    ".llm-imgcopy:hover{background:#000d;}";
  document.head.appendChild(s);
}

// Copy a data-URI image to the clipboard. The clipboard wants image/png, so the JPEG preview is
// redrawn onto a canvas and exported as PNG. Needs a secure context + a user gesture (the click).
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

// A thumbnail with a hover-reveal copy button pinned to its corner.
function makeThumb(src) {
  const wrap = document.createElement("div");
  wrap.className = "llm-imgwrap";
  const img = document.createElement("img");
  img.className = "llm-thumb";
  img.src = src;
  const btn = document.createElement("button");
  btn.className = "llm-imgcopy";
  btn.title = "Copy image to clipboard";
  btn.textContent = "📋";
  btn.addEventListener("pointerdown", (e) => e.stopPropagation()); // don't start a node/canvas drag
  btn.onclick = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    try { await copyImageToClipboard(src); btn.textContent = "✓"; }
    catch (err) { console.error("[LiveLog] image copy failed", err); btn.textContent = "✕"; }
    setTimeout(() => { btn.textContent = "📋"; }, 1200);
  };
  wrap.append(img, btn);
  return wrap;
}

// Split reasoning from the answer exactly the way the backend's _split_reasoning does, so what the
// log calls "the answer" is what the node's `text` output will be. With an answer_marker the split
// only happens once the marker line has actually streamed in (before that the backend would treat
// everything as the answer too) — so mid-stream text can hop into the reasoning block when the
// marker lands. <think> tags split live, including an unclosed one left by truncation.
function splitReasoning(raw, marker) {
  const m = (marker || "").trim();
  if (m) {
    const lines = raw.split("\n");
    let last = -1;
    for (let i = 0; i < lines.length; i++) if (lines[i].trim() === m) last = i;
    if (last !== -1) {
      return [lines.slice(last + 1).join("\n").trim(), lines.slice(0, last).join("\n").trim()];
    }
  }
  const parts = [];
  let answer = raw.replace(/<think>([\s\S]*?)<\/think>/g, (_, p) => { parts.push(p); return ""; });
  const open = answer.indexOf("<think>");
  if (open !== -1) { parts.push(answer.slice(open + 7)); answer = answer.slice(0, open); }
  return [answer.trim(), parts.map((p) => p.trim()).join("\n\n").trim()];
}

// Resolve the source LLM node's current title (falls back to type, then id).
function titleFor(srcId) {
  const n = (app.graph?._nodes || []).find((x) => String(x.id) === String(srcId));
  return (n && (n.title || n.type)) || `LLM #${srcId}`;
}

// ── token accounting ────────────────────────────────────────────────────────────────────────
// While streaming, one delta == one generated token (the worker emits a token per chunk), so the
// live count is exact; 'done' then replaces it with the worker's own output_tokens.

function tokensOf(block) {
  return (block.done && block.outTokens != null) ? block.outTokens : block.tokens;
}

function fmtTokens(block) {
  const n = tokensOf(block);
  if (!n && !block.maxTokens) return "";
  return block.maxTokens ? `${n}/${block.maxTokens} tok` : `${n} tok`;
}

// Generation speed measured over the streamed token window (excludes model load + prefill, so it
// matches what llama.cpp calls the eval rate). Falls back to total tokens / total seconds for a
// non-streaming (grammar/JSON) run.
function rateOf(block) {
  if (block.t0 && block.tokens > 2) {
    const end = block.tEnd || performance.now();
    const secs = (end - block.t0) / 1000;
    if (secs > 0.2) return (block.tokens - 1) / secs;
  }
  if (block.done && block.seconds > 0 && tokensOf(block)) return tokensOf(block) / block.seconds;
  return 0;
}

// One-line context-fill summary, shown once the exact figures land on 'done'.
function ctxLine(block) {
  if (!block.nCtx) return null;
  const gen = tokensOf(block) || 0;
  const used = block.ctxUsed || (block.promptTokens ? block.promptTokens + gen : 0);
  if (!used) return null;
  const pct = Math.round((used * 1000) / block.nCtx) / 10;
  let txt = `ⓘ ctx ${used}/${block.nCtx} (${pct}%) · prompt ${block.promptTokens || "?"} + gen ${gen}`;
  const warn = [];
  if (block.finish === "length") warn.push("output hit max_tokens — raise it");
  if (pct >= 95) warn.push("context ≈ full — raise n_ctx");
  if (warn.length) txt += "  ⚠ " + warn.join("; ");
  return { txt, warn: warn.length > 0 };
}

function buildBlockDom(node, block) {
  const els = node._llmEls;
  if (!els) return;
  const wrap = document.createElement("div");
  wrap.className = "llm-block";
  const head = document.createElement("div");
  const htxt = document.createElement("span");
  htxt.className = "llm-htxt";
  const copy = document.createElement("button");
  copy.className = "llm-copy";
  copy.textContent = "copy";
  copy.title = "Copy the answer (without the reasoning)";
  copy.addEventListener("pointerdown", (e) => e.stopPropagation()); // don't start a node drag
  copy.onclick = async (e) => {
    e.preventDefault(); e.stopPropagation();
    const [answer] = splitReasoning(block.text, block.marker);
    try { await navigator.clipboard.writeText(answer || block.text || ""); copy.textContent = "✓"; }
    catch (err) { console.error("[LLMLog] copy failed", err); copy.textContent = "✕"; }
    setTimeout(() => { copy.textContent = "copy"; }, 1200);
  };
  head.append(htxt, copy);

  // Token budget bar — only meaningful when the run told us its max_tokens.
  const bar = document.createElement("div");
  bar.className = "llm-bar";
  const fill = document.createElement("div");
  fill.className = "llm-bar-fill";
  bar.appendChild(fill);

  const body = document.createElement("div");
  body.className = "llm-body";
  const imgs = document.createElement("div");
  imgs.className = "llm-imgs";
  imgs.style.display = "none";
  wrap.append(head, bar, body, imgs);
  els.list.appendChild(wrap);
  block._el = wrap; block._head = head; block._htxt = htxt; block._body = body;
  block._bar = bar; block._fill = fill; block._think = null; block._ctx = null;
  block._imgs = imgs; block._imgCount = 0;
  renderImages(block);
  updateBlockDom(block);
}

// Thumbnails are append-only: a block can be handed frames on 'start' and more on 'done'.
function renderImages(block) {
  if (!block._imgs) return;
  const list = block.images || [];
  for (let i = block._imgCount; i < list.length; i++) block._imgs.appendChild(makeThumb(list[i]));
  block._imgCount = list.length;
  block._imgs.style.display = list.length ? "" : "none";
}

function addImages(block, images) {
  if (!Array.isArray(images) || !images.length) return;
  block.images = (block.images || []).concat(images.filter((s) => typeof s === "string" && s));
  renderImages(block);
}

// The collapsible reasoning section, inserted above the answer the first time thoughts appear.
function ensureThink(block) {
  if (block._think) return block._think;
  const wrap = document.createElement("div");
  wrap.className = "llm-think";
  const head = document.createElement("div");
  head.className = "llm-think-head";
  head.addEventListener("pointerdown", (e) => e.stopPropagation());
  head.onclick = (e) => {
    e.preventDefault(); e.stopPropagation();
    block.thinkOpen = !block.thinkOpen;
    block.thinkTouched = true;   // you decided — stop auto-collapsing this block
    updateBlockDom(block);
  };
  const body = document.createElement("div");
  body.className = "llm-think-body";
  wrap.append(head, body);
  block._el.insertBefore(wrap, block._body);
  block._think = { wrap, head, body };
  return block._think;
}

function updateBlockDom(block) {
  if (!block._head) return;
  const [answer, thoughts] = splitReasoning(block.text, block.marker);

  const bits = [block.title];
  const tok = fmtTokens(block);
  if (tok) bits.push(tok);
  const rate = rateOf(block);
  if (rate) bits.push(`${rate.toFixed(1)} tok/s`);
  if (block.done) {
    bits.push(`${block.seconds ?? "?"}s`);
    if (block.finish && block.finish !== "stop") bits.push(block.finish);
  } else {
    bits.push("…");
  }
  block._htxt.textContent = `${block.done ? "✓" : "▶"} ${bits.join(" · ")}`;
  block._head.className = "llm-head" + (block.done ? " done" : "") + (block.finish === "length" ? " warn" : "");

  // Budget bar: how much of max_tokens this generation has eaten.
  if (block.maxTokens) {
    const ratio = Math.min(1, (tokensOf(block) || 0) / block.maxTokens);
    block._bar.style.display = "";
    block._fill.style.width = `${(ratio * 100).toFixed(1)}%`;
    block._fill.className = "llm-bar-fill" +
      (block.finish === "length" || ratio >= 1 ? " warn" : (block.done ? " done" : ""));
  } else {
    block._bar.style.display = "none";
  }

  // Reasoning: shown live, then folded away once the answer starts (unless you opened/closed it
  // yourself). The token figure is a char-length estimate — same approximation the node makes.
  if (thoughts) {
    if (!block.thinkStart) block.thinkStart = block.t0 || performance.now();
    if (!block.thinkEnd && (answer || block.done)) {
      block.thinkEnd = performance.now();
      if (!block.thinkTouched) block.thinkOpen = false;
    }
    const t = ensureThink(block);
    const secs = ((block.thinkEnd || performance.now()) - block.thinkStart) / 1000;
    const total = tokensOf(block) || 0;
    const est = total ? Math.round((total * thoughts.length) / (thoughts.length + answer.length)) : 0;
    const label = block.thinkEnd ? `thought for ${secs.toFixed(1)}s` : `thinking… ${secs.toFixed(1)}s`;
    t.head.textContent = `${block.thinkOpen ? "▾" : "▸"} ${label}${est ? ` · ~${est} tok` : ""}`;
    t.body.style.display = block.thinkOpen ? "" : "none";
    t.body.textContent = thoughts;
  } else if (block._think) {
    block._think.wrap.style.display = "none";
  }

  block._body.textContent = answer || (block.done ? "(no text)" : "");

  const ctx = ctxLine(block);
  if (ctx) {
    if (!block._ctx) {
      block._ctx = document.createElement("div");
      block._el.appendChild(block._ctx);
    }
    block._ctx.className = "llm-ctx" + (ctx.warn ? " warn" : "");
    block._ctx.textContent = ctx.txt;
  }
}

// ── scrolling ───────────────────────────────────────────────────────────────────────────────
// Auto-follow is opt-out by position: we only pin to the bottom while the view IS at the bottom,
// so scrolling up mid-stream freezes the view (nothing above it moves — text is appended below).

function atBottom(node) {
  const s = node._llmEls?.scroll;
  if (!s) return true;
  return s.scrollHeight - s.scrollTop - s.clientHeight <= STICK_SLACK;
}

function showJump(node, visible, hot) {
  const e = node._llmEls;
  if (!e) return;
  e.jump.style.display = visible ? "" : "none";
  if (!visible) e.jump.classList.remove("hot");
  else if (hot) e.jump.classList.add("hot");
}

// Follow new content only if the user hasn't scrolled away; otherwise flag the pill as "hot".
function stick(node) {
  const e = node._llmEls;
  if (!e) return;
  if (node._llmStick === false) showJump(node, true, true);
  else e.scroll.scrollTop = e.scroll.scrollHeight;
}

// Snap back to the bottom and resume following (pill click / fresh render).
function toBottom(node) {
  const e = node._llmEls;
  if (!e) return;
  node._llmStick = true;
  e.scroll.scrollTop = e.scroll.scrollHeight;
  showJump(node, false);
}

function setStatus(node, txt) {
  if (node._llmEls) node._llmEls.statusTxt.textContent = txt;
}

// Wipe this log node's blocks (history included, so a tab switch doesn't bring them back). An
// in-flight stream loses its current block and simply starts a new one on the next token.
function clearLog(node) {
  const els = node._llmEls;
  if (!els) return;
  snapFor(node.id).length = 0;
  els.list.innerHTML = "";
  node._llmCur = new Map();
  setStatus(node, "waiting for a run…");
  toBottom(node);
}

// New empty block for a source, appended to this node's history + DOM (with the top-drop cap).
function pushBlock(node, srcId, d) {
  const snap = snapFor(node.id);
  // A source may label its individual calls (Morpheus Storyboard sends "shot 2/4"); without a
  // label the block is just the node's title, exactly as before.
  const label = typeof d?.label === "string" && d.label ? ` · ${d.label}` : "";
  const block = {
    srcId: String(srcId), title: titleFor(srcId) + label, text: "", done: false, seconds: null, finish: "",
    hadDelta: false, tokens: 0, maxTokens: Number(d?.max_tokens) || 0, outTokens: null,
    promptTokens: 0, ctxUsed: 0, nCtx: Number(d?.n_ctx) || 0, t0: 0, tEnd: 0,
    marker: d?.answer_marker || "", thinkOpen: true, thinkTouched: false, thinkStart: 0, thinkEnd: 0,
    images: [],
  };
  snap.push(block);
  while (snap.length > MAX_BLOCKS) { const rm = snap.shift(); if (rm._el) rm._el.remove(); }
  buildBlockDom(node, block);
  return block;
}

function handle(node, d) {
  if (!node._llmEls) return;
  const src = String(d.id);
  // The Chat node streams bare {id, delta} with no event name — normalise so every LLM source in
  // the pack lands in the same block model.
  const ev = d.event || (d.delta != null ? "delta" : null);
  if (ev === "frames") {
    // Images with no generation behind them: a finished block that is only pictures.
    const block = pushBlock(node, src, d);
    block.done = true;
    addImages(block, d.images);
    updateBlockDom(block);
    setStatus(node, block.title);
    stick(node);
  } else if (ev === "start") {
    const block = pushBlock(node, src, d);
    node._llmCur.set(src, block);
    addImages(block, d.images);   // what this call was actually shown
    setStatus(node, `${block.title} — generating…`);
    stick(node);
  } else if (ev === "delta") {
    let block = node._llmCur.get(src);
    if (!block) { block = pushBlock(node, src, d); node._llmCur.set(src, block); }  // delta with no start
    block.text += (d.delta || "");
    block.hadDelta = true;
    block.tokens += 1;
    if (!block.t0) block.t0 = performance.now();
    updateBlockDom(block);
    setStatus(node, `${block.title} — generating… ${fmtTokens(block)}`);
    stick(node);
  } else if (ev === "done") {
    let block = node._llmCur.get(src);
    if (!block) block = pushBlock(node, src, d);   // grammar/JSON run: no deltas streamed
    if (!block.hadDelta && d.text) block.text = d.text;
    addImages(block, d.images);
    block.done = true;
    block.tEnd = performance.now();
    block.seconds = d.gen_seconds;
    block.finish = d.finish_reason || "";
    if (d.max_tokens) block.maxTokens = Number(d.max_tokens) || block.maxTokens;
    if (d.output_tokens != null) block.outTokens = Number(d.output_tokens) || 0;
    block.promptTokens = Number(d.prompt_tokens) || 0;
    block.ctxUsed = Number(d.context_used) || 0;
    block.nCtx = Number(d.n_ctx) || block.nCtx;
    updateBlockDom(block);
    node._llmCur.delete(src);
    setStatus(node, `${block.title} — done · ${fmtTokens(block) || "?"} · ${block.seconds ?? "?"}s`);
    stick(node);
  }
}

// Rebuild the DOM from the stored blocks (workflow load / tab switch).
function restore(node) {
  const els = node._llmEls;
  if (!els) return;
  els.list.innerHTML = "";
  node._llmCur = new Map();
  const snap = snapFor(node.id);
  for (const block of snap) {
    buildBlockDom(node, block);
    if (!block.done) node._llmCur.set(block.srcId, block);  // let an in-flight stream keep landing
  }
  setStatus(node, snap.length ? "" : "waiting for a run…");
  toBottom(node);
}

// One shared listener per channel fans out to every live log node on the canvas.
for (const channel of ["kinburg.llm", "kinburg.chatllm"]) {
  api.addEventListener(channel, (e) => {
    const d = (e && e.detail) || {};
    for (const node of instances) handle(node, d);
  });
}

app.registerExtension({
  name: "Kinburg.LiveLog",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!CLASSES.has(nodeData.name)) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const node = this;
      injectStyle();

      // Fixed-height node whose content scrolls (mirrors Show Text / Ouroboros Log): root fills
      // the widget area, a flex:1 box (relative, overflow:hidden) holds an ABSOLUTELY positioned
      // scroll layer so the rows never inflate the node's measured height.
      const root = document.createElement("div");
      root.style.cssText =
        "display:flex;flex-direction:column;width:100%;height:100%;box-sizing:border-box;" +
        "gap:4px;padding:4px;font-family:inherit;font-size:11px;background:#00000022;border-radius:6px;";
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
      clear.className = "llm-clear";
      clear.textContent = "clear";
      clear.title = "Remove every block from this log";
      clear.addEventListener("pointerdown", (e) => e.stopPropagation());
      clear.onclick = (e) => { e.preventDefault(); e.stopPropagation(); clearLog(node); };
      status.append(statusTxt, clear);

      const box = document.createElement("div");
      box.style.cssText = "flex:1 1 auto;position:relative;min-height:80px;overflow:hidden;";
      const scroll = document.createElement("div");
      scroll.style.cssText = "position:absolute;inset:0;overflow-y:auto;";
      const list = document.createElement("div");
      list.style.cssText = "display:flex;flex-direction:column;gap:6px;";

      // "Back to the newest text" pill — only visible once you've scrolled away from the bottom.
      const jump = document.createElement("button");
      jump.className = "llm-jump";
      jump.textContent = "↓ latest";
      jump.style.display = "none";
      jump.addEventListener("pointerdown", (e) => e.stopPropagation());
      jump.onclick = (e) => { e.preventDefault(); e.stopPropagation(); toBottom(node); };

      scroll.appendChild(list);
      box.append(scroll, jump);
      root.append(status, box);
      node._llmEls = { root, box, scroll, list, status, statusTxt, jump };
      node._llmCur = new Map();
      node._llmStick = true;

      // Any scroll (wheel, drag, keyboard) re-decides whether we follow: parked at the bottom →
      // follow; anywhere above → freeze. Programmatic scrolls land at the bottom, so they keep it on.
      scroll.addEventListener("scroll", () => {
        const bottom = atBottom(node);
        node._llmStick = bottom;
        showJump(node, !bottom, false);
      });

      node.addDOMWidget("llm_log", "kinburg_llm_log", root, { serialize: false });
      if ((node.size?.[1] || 0) < 260) {
        node.setSize([Math.max(node.size?.[0] || 0, 400), 420]);
      }
      instances.add(node);
      restore(node);   // best-effort; onConfigure restores reliably once the id is final
      return r;
    };

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
