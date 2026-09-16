import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// LLM Server Live Log — a UI-only node showing what the managed llama.cpp server is doing: every
// request the chat client makes, what it sent with it, what it cost, plus loads, unloads, errors
// and the parameter repairs on the way through.
//
// Two things make it different from the pack's other live logs. The events do NOT belong to a
// ComfyUI run — SillyTavern talks to the gateway whenever it likes, quite possibly with no browser
// open — so on appearing the node asks /kinburg/llm_server/log for the backlog instead of starting
// blank. And a request is TWO events: it opens a row the moment it arrives (so a cold start reads
// as a load rather than a hang) and fills that same row in when it finishes.
//
// Scrolling follows new rows ONLY while you are already parked at the bottom. Scroll up and the
// view stays put while the chat keeps going; a "↓ latest" pill takes you back.

const CLASS = "LLMServerLog";
const CHANNEL = "kinburg.llmserver";
const STICK_SLACK = 24;      // px from the bottom that still counts as "at the bottom"
const MAX_ROWS = 400;        // matches the backend ring; older rows are dropped
const instances = new Set(); // live log-node instances to fan events out to

let _styled = false;
function injectStyle() {
  if (_styled) return;
  _styled = true;
  const s = document.createElement("style");
  s.textContent =
    ".ls-row{display:flex;flex-direction:column;gap:2px;padding:3px 5px;border-radius:4px;" +
      "background:#ffffff0a;border-left:2px solid transparent;}" +
    ".ls-row.ls-err{background:#8a1f1f33;border-left-color:#d46a6a;}" +
    ".ls-row.ls-model{background:#1f4a8a26;border-left-color:#6a9fd4;}" +
    ".ls-row.ls-note{background:transparent;}" +
    ".ls-row.ls-probe{background:transparent;padding:1px 5px;}" +
    ".ls-row.ls-out{background:transparent;padding:0 5px;}" +
    ".ls-line{display:flex;align-items:baseline;gap:6px;color:#dcdce4;}" +
    ".ls-t{flex:0 0 auto;color:#70707a;font-size:9px;font-family:ui-monospace,Consolas,monospace;}" +
    ".ls-main{flex:1 1 auto;min-width:0;word-break:break-word;}" +
    ".ls-tag{flex:0 0 auto;font-size:9px;text-transform:uppercase;letter-spacing:.04em;color:#8a8a94;}" +
    ".ls-pending{color:#e0a648;}" +
    ".ls-ok{color:#8fd6c9;}" +
    ".ls-bad{color:#e08a8a;}" +
    ".ls-dim{color:#8a8a94;font-size:10px;}" +
    ".ls-samp{color:#9a9ac2;font-size:10px;font-family:ui-monospace,Consolas,monospace;" +
      "word-break:break-word;}" +
    ".ls-ctx{color:#8a8a94;font-size:10px;font-family:ui-monospace,Consolas,monospace;}" +
    ".ls-ctx.warn{color:#e0a648;}" +
    ".ls-txt{color:#b9b9c2;white-space:pre-wrap;word-break:break-word;font-size:10px;}" +
    ".ls-txt.ls-reply{color:#8fd6c9;}" +
    ".ls-out .ls-main{color:#7a7a84;font-size:9px;font-family:ui-monospace,Consolas,monospace;" +
      "white-space:pre-wrap;word-break:break-all;}" +
    ".ls-jump{position:absolute;right:12px;bottom:8px;z-index:5;cursor:pointer;padding:3px 9px;" +
      "border-radius:11px;border:1px solid #0008;background:#2b2b33f0;color:#dcdce4;font-size:10px;" +
      "line-height:1.3;box-shadow:0 2px 6px #0007;}" +
    ".ls-jump:hover{background:#3a3a44f0;}" +
    ".ls-btn{flex:0 0 auto;cursor:pointer;padding:1px 6px;border-radius:3px;border:1px solid #ffffff1f;" +
      "background:transparent;color:#9a9aa2;font-size:9px;text-transform:uppercase;letter-spacing:.04em;}" +
    ".ls-btn:hover{background:#ffffff14;color:#dcdce4;}";
  document.head.appendChild(s);
}

// ── formatting ────────────────────────────────────────────────────────────────────────────────
function fmtMs(ms) {
  if (!(ms > 0)) return "";
  return ms < 1000 ? Math.round(ms) + "ms" : (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + "s";
}

function fmtCount(n) {
  if (!(n > 0)) return "0";
  return n >= 10000 ? (n / 1000).toFixed(1) + "k" : String(n);
}

// The sampler settings the client actually sent, on one line. This is the answer to "what is my
// preset really doing?", which is otherwise guesswork on the SillyTavern side.
function fmtSampler(s) {
  if (!s) return "";
  const bits = [];
  const put = (label, key, digits) => {
    const v = s[key];
    if (v === undefined || v === null) return;
    bits.push(label + " " + (digits === 0 ? v : Number(v).toFixed(digits ?? 2)));
  };
  put("temp", "temperature");
  put("top_p", "top_p");
  put("top_k", "top_k", 0);
  put("min_p", "min_p");
  put("typ", "typical_p");
  if (s.top_n_sigma > 0 || s.nsigma > 0) put("nsigma", s.top_n_sigma > 0 ? "top_n_sigma" : "nsigma");
  const rep = s.repetition_penalty ?? s.repeat_penalty;
  if (rep !== undefined) bits.push("rep " + Number(rep).toFixed(2));
  if (s.frequency_penalty) bits.push("freq " + Number(s.frequency_penalty).toFixed(2));
  if (s.presence_penalty) bits.push("pres " + Number(s.presence_penalty).toFixed(2));
  if (s.dry_multiplier) {
    bits.push("dry " + Number(s.dry_multiplier).toFixed(2)
      + (s.dry_base ? "/" + Number(s.dry_base).toFixed(2) : "")
      + (s.dry_allowed_length ? "/" + s.dry_allowed_length : "")
      + (s.dry_breakers ? " ×" + s.dry_breakers : ""));
  }
  if (s.xtc_probability) bits.push("xtc " + Number(s.xtc_probability).toFixed(2));
  if (s.mirostat) bits.push("mirostat " + s.mirostat);
  if (s.max_tokens ?? s.n_predict) bits.push("max " + (s.max_tokens ?? s.n_predict));
  if (s.seed !== undefined && s.seed >= 0) bits.push("seed " + s.seed);
  if (s.constrained) bits.push("grammar");
  if (s.stops) bits.push("stops ×" + s.stops);
  return bits.join(" · ");
}

// prompt tokens against the context the server was started with — the number that says when the
// chat is about to start dropping its own beginning.
function fmtCtx(d) {
  const used = d.prompt_tokens, lim = d.n_ctx;
  if (!(used > 0)) return { text: "", warn: false };
  if (!(lim > 0)) return { text: "ctx " + fmtCount(used), warn: false };
  const pct = Math.round((used / lim) * 100);
  return { text: "ctx " + fmtCount(used) + "/" + fmtCount(lim) + " (" + pct + "%)", warn: pct >= 85 };
}

function el(cls, text) {
  const d = document.createElement("div");
  if (cls) d.className = cls;
  if (text !== undefined) d.textContent = text;
  return d;
}

function span(cls, text) {
  const s = document.createElement("span");
  if (cls) s.className = cls;
  if (text !== undefined) s.textContent = text;
  return s;
}

// ── scrolling ─────────────────────────────────────────────────────────────────────────────────
function atBottom(node) {
  const sc = node._lsEls?.scroll;
  if (!sc) return true;
  return sc.scrollHeight - sc.scrollTop - sc.clientHeight <= STICK_SLACK;
}

function showJump(node, visible) {
  const j = node._lsEls?.jump;
  if (j) j.style.display = visible ? "" : "none";
}

function stick(node) {
  if (node._lsStick) toBottom(node);
  else showJump(node, true);
}

function toBottom(node) {
  const sc = node._lsEls?.scroll;
  if (!sc) return;
  sc.scrollTop = sc.scrollHeight;
  node._lsStick = true;
  showJump(node, false);
}

function setStatus(node, txt) {
  if (node._lsEls) node._lsEls.statusTxt.textContent = txt;
}

function statusFrom(s) {
  if (!s || !s.listening) return "gateway not started — run the Local LLM Server node";
  const model = s.model_loaded ? "loaded: " + (s.model || "?") : "unloaded";
  const emb = s.embed ? " · embeddings " + (s.embed.in_chat_process ? "inline"
    : (s.embed.loaded ? "loaded" : "unloaded")) : "";
  return s.listening + " · " + model + emb + " · " + (s.requests || 0) + " requests";
}

function clearLog(node) {
  const els = node._lsEls;
  if (!els) return;
  els.list.innerHTML = "";
  node._lsRows = new Map();
  node._lsSeen = new Set();
  node._lsProbe = null;
  toBottom(node);
}

function trimRows(node) {
  const list = node._lsEls?.list;
  while (list && list.children.length > MAX_ROWS) list.removeChild(list.children[0]);
}

// ── rows ──────────────────────────────────────────────────────────────────────────────────────
function addRow(node, cls) {
  const row = el("ls-row" + (cls ? " " + cls : ""));
  node._lsEls.list.appendChild(row);
  trimRows(node);
  return row;
}

function headLine(row, time, main, tag) {
  const line = el("ls-line");
  line.append(span("ls-t", time || ""), span("ls-main", main || ""));
  if (tag) line.append(span("ls-tag", tag));
  row.appendChild(line);
  return line;
}

function modelRow(node, d) {
  const what = {
    loading: "⬆ loading " + (d.model || ""),
    ready: "✓ ready" + (d.ms ? " in " + fmtMs(d.ms) : ""),
    unloaded: "⬇ unloaded" + (d.reason ? " — " + d.reason : ""),
    failed: "✗ " + (d.error || "failed to start"),
  }[d.event] || d.event;
  const row = addRow(node, d.event === "failed" ? "ls-err" : "ls-model");
  headLine(row, d.t, what, d.role === "embed" ? "embed" : "");
  return row;
}

function gatewayRow(node, d) {
  const row = addRow(node, "ls-model");
  headLine(row, d.t, d.event === "listening" ? "◉ listening on " + d.addr : "○ stopped listening");
  return row;
}

// A run of model-list / health polls collapses into one row with a counter, so a client that
// checks every few seconds cannot bury the thing you are trying to read.
function probeRow(node, d) {
  const prev = node._lsProbe;
  if (prev && prev.path === d.path && node._lsEls.list.children[node._lsEls.list.children.length - 1] === prev.row) {
    prev.n += 1;
    prev.main.textContent = "· " + d.path + " ×" + prev.n;
    return prev.row;
  }
  const row = addRow(node, "ls-probe");
  const line = el("ls-line");
  const main = span("ls-main ls-dim", "· " + d.path);
  line.append(span("ls-t", d.t), main);
  row.appendChild(line);
  node._lsProbe = { path: d.path, n: 1, row, main };
  return row;
}

function noteRow(node, d) {
  const row = addRow(node, "ls-note");
  const line = el("ls-line");
  line.append(span("ls-t", d.t), span("ls-main ls-dim", "⚙ " + (d.text || "")));
  row.appendChild(line);
  return row;
}

function stdoutRow(node, d) {
  const row = addRow(node, "ls-out");
  const line = el("ls-line");
  line.append(span("ls-main", d.line || ""));
  row.appendChild(line);
  return row;
}

// A request opens its row on arrival and is filled in when it finishes — so the gap between the
// two IS the wait, visibly.
function requestRow(node, d) {
  const row = addRow(node, "");
  const shape = [];
  if (d.messages) shape.push(d.messages + " msg");
  if (d.images) shape.push(d.images + " img");
  if (d.chars) shape.push(fmtCount(d.chars) + " ch");
  if (d.inputs) shape.push(d.inputs + " input" + (d.inputs > 1 ? "s" : ""));
  if (d.stream) shape.push("stream");
  const head = headLine(row, d.t, (d.method || "") + " " + (d.path || ""),
                        d.role === "embed" ? "embed" : "");
  const result = span("ls-pending", " …" + (shape.length ? "  " + shape.join(" · ") : ""));
  head.appendChild(result);

  const samp = fmtSampler(d.sampler);
  if (samp) row.appendChild(el("ls-samp", samp));
  if (d.prompt) row.appendChild(el("ls-txt", "▸ " + d.prompt));

  node._lsRows.set(d.seq, { row, result, n_ctx: d.n_ctx });
  return row;
}

function finishRequest(node, d) {
  const held = node._lsRows.get(d.id);
  const row = held ? held.row : addRow(node, "");
  const result = held ? held.result : span("", "");
  if (!held) headLine(row, d.t, (d.path || "")).appendChild(result);
  node._lsRows.delete(d.id);

  const bad = d.status >= 400 || d.error;
  const bits = [String(d.status || "?")];
  if (d.ms) bits.push(fmtMs(d.ms));
  if (d.gen_tokens) {
    bits.push(fmtCount(d.prompt_tokens || 0) + "+" + fmtCount(d.gen_tokens)
              + " tok" + (d.estimated ? "~" : ""));
  }
  if (d.tps) bits.push(Number(d.tps).toFixed(1) + " tok/s");
  else if (d.gen_tokens && d.ms > 0) bits.push((d.gen_tokens / (d.ms / 1000)).toFixed(1) + " tok/s");
  if (d.aborted) bits.push("aborted");
  result.className = bad ? "ls-bad" : "ls-ok";
  result.textContent = "  " + bits.join(" · ");

  const ctx = fmtCtx({ prompt_tokens: d.prompt_tokens, n_ctx: held ? held.n_ctx : 0 });
  if (ctx.text) row.appendChild(el("ls-ctx" + (ctx.warn ? " warn" : ""), ctx.text));
  if (d.error) {
    row.classList.add("ls-err");
    row.appendChild(el("ls-txt ls-bad", d.error));
  }
  if (d.reply) row.appendChild(el("ls-txt ls-reply", "◂ " + d.reply));
  return row;
}

function applyEvent(node, d) {
  if (!node._lsEls || !d) return;
  // The websocket and the backlog fetch overlap by design, so the same event can arrive twice.
  if (d.seq) {
    if (node._lsSeen.has(d.seq)) return;
    node._lsSeen.add(d.seq);
  }
  if (d.kind !== "probe") node._lsProbe = null;   // a real row breaks a run of polls

  switch (d.kind) {
    case "request": requestRow(node, d); break;
    case "request_done": finishRequest(node, d); break;
    case "model": modelRow(node, d); break;
    case "gateway": gatewayRow(node, d); break;
    case "note": noteRow(node, d); break;
    case "probe": probeRow(node, d); break;
    case "stdout": stdoutRow(node, d); break;
    default: return;
  }
  // The status line is a one-glance answer to "is it up, is a model in?" — the newest lifecycle
  // event is a better answer than whatever the backlog fetch said a minute ago.
  if (d.kind === "gateway") {
    setStatus(node, d.event === "listening" ? d.addr + " · waiting for requests"
                                            : "gateway stopped listening");
  } else if (d.kind === "model" && d.role !== "embed") {
    if (d.event === "loading") setStatus(node, "loading " + (d.model || "the model") + "…");
    else if (d.event === "ready") setStatus(node, "model loaded: " + (d.model || ""));
    else if (d.event === "unloaded") setStatus(node, "model unloaded — loads on the next request");
    else if (d.event === "failed") setStatus(node, "model failed to start");
  }
  stick(node);
}

// ── history ───────────────────────────────────────────────────────────────────────────────────
// Per-node event history (keyed by node id), in memory only. Lets the log survive a ComfyUI
// Desktop tab switch, which destroys and recreates the node. The backend ring is the real history
// — this is only so a rebuild does not have to go and ask for it again.
const logStore = new Map();

function record(d, nodes) {
  for (const id of new Set(nodes.map((n) => n.id))) {
    let arr = logStore.get(id);
    if (!arr) logStore.set(id, (arr = []));
    arr.push(d);
    if (arr.length > MAX_ROWS) arr.splice(0, arr.length - MAX_ROWS);
  }
}

function restore(node) {
  const evs = logStore.get(node.id);
  if (!node._lsEls || !evs || !evs.length) return false;
  clearLog(node);
  for (const d of evs) applyEvent(node, d);
  toBottom(node);
  return true;
}

// The backlog: what happened before this node existed — including while ComfyUI had no browser
// attached at all, which for this log is the normal case rather than the exotic one.
async function loadBacklog(node) {
  try {
    const r = await api.fetchApi("/kinburg/llm_server/log?after=0");
    if (!r || r.status !== 200) return;
    const data = await r.json();
    if (!node._lsEls) return;
    for (const d of data.events || []) {
      applyEvent(node, d);
      record(d, [node]);
    }
    // After the replay, not before: the events would otherwise overwrite it with a line from
    // whenever the backlog happens to end.
    setStatus(node, statusFrom(data.status));
    toBottom(node);
  } catch (err) {
    console.warn("[LLM server log] could not fetch the backlog", err);
  }
}

// Which log nodes are actually on a canvas right now. `instances` is a CACHE, not the truth: the
// frontend rebuilds nodes behind our back (tab switch, undo, reloaded graph) without always
// running both onNodeCreated and onRemoved, and a stale copy left in the set takes events meant
// for the live one and records the history twice under the same id.
function liveNodes() {
  for (const n of [...instances]) {
    if (n.graph) n._lsLive = true;                          // it is on a canvas
    else if (n._lsLive || !n._lsEls) instances.delete(n);   // …and has since left one
  }
  for (const n of app.graph?._nodes || []) {
    if ((n.comfyClass || n.type) === CLASS && n._lsEls) instances.add(n);
  }
  return [...instances].filter((n) => n._lsEls);
}

// One shared websocket listener fans out to every live log node on the canvas.
api.addEventListener(CHANNEL, (e) => {
  const d = (e && e.detail) || {};
  const nodes = liveNodes();
  for (const node of nodes) {
    // One node's row blowing up must not cost the others theirs.
    try {
      applyEvent(node, d);
    } catch (err) {
      console.error("[LLM server log] could not render an event on node", node.id, d.kind, err);
    }
  }
  record(d, nodes);
});

app.registerExtension({
  name: "Kinburg.LLMServerLog",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const node = this;
      injectStyle();

      // Mirror Show Text (Markdown): root fills the widget area, a flex:1 box scrolls, and the
      // scrolling layer is absolutely positioned so its rows never inflate the height ComfyUI
      // measures — the node keeps its size and the content scrolls instead of growing it.
      const root = document.createElement("div");
      root.className = "ls-log";
      root.style.cssText =
        "display:flex;flex-direction:column;width:100%;height:100%;box-sizing:border-box;" +
        "gap:4px;padding:4px;font-family:inherit;font-size:11px;background:#00000022;border-radius:6px;";
      root.addEventListener("wheel", (e) => e.stopPropagation());
      root.addEventListener("pointerdown", (e) => e.stopPropagation());

      const status = document.createElement("div");
      status.style.cssText = "flex:0 0 auto;display:flex;align-items:center;gap:6px;padding:1px 4px;";
      const statusTxt = document.createElement("span");
      statusTxt.style.cssText =
        "flex:1 1 auto;min-width:0;color:#9a9aa2;font-size:10px;overflow:hidden;" +
        "text-overflow:ellipsis;white-space:nowrap;";
      statusTxt.textContent = "asking the gateway…";
      const clear = document.createElement("button");
      clear.className = "ls-btn";
      clear.textContent = "clear";
      clear.title = "Empty this log — and the backlog the gateway is keeping";
      clear.addEventListener("pointerdown", (e) => e.stopPropagation());
      clear.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        logStore.set(node.id, []);
        clearLog(node);
        api.fetchApi("/kinburg/llm_server/log/clear", { method: "POST" }).catch(() => {});
      };
      status.append(statusTxt, clear);

      const box = document.createElement("div");
      box.style.cssText = "flex:1 1 auto;position:relative;min-height:80px;overflow:hidden;";
      const scroll = document.createElement("div");
      scroll.style.cssText = "position:absolute;inset:0;overflow-y:auto;";
      const list = document.createElement("div");
      list.style.cssText = "display:flex;flex-direction:column;gap:3px;";

      const jump = document.createElement("button");
      jump.className = "ls-jump";
      jump.textContent = "↓ latest";
      jump.style.display = "none";
      jump.addEventListener("pointerdown", (e) => e.stopPropagation());
      jump.onclick = (e) => { e.preventDefault(); e.stopPropagation(); toBottom(node); };

      scroll.appendChild(list);
      box.append(scroll, jump);
      root.append(status, box);
      node._lsEls = { root, box, scroll, list, status, statusTxt, jump };
      node._lsStick = true;
      node._lsRows = new Map();
      node._lsSeen = new Set();
      node._lsProbe = null;

      scroll.addEventListener("scroll", () => {
        const bottom = atBottom(node);
        node._lsStick = bottom;
        showJump(node, !bottom);
      });

      node.addDOMWidget("ls_log", "kinburg_llm_server_log", root, { serialize: false });
      if ((node.size?.[1] || 0) < 300) {
        node.setSize([Math.max(node.size?.[0] || 0, 460), 520]);
      }
      instances.add(node);
      if (!restore(node)) loadBacklog(node);
      return r;
    };

    // Fires on workflow load / tab switch, AFTER the saved id is restored — replay the history so
    // the log isn't blank when you switch back, and fall back to the backend's own ring.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      if (!restore(this)) loadBacklog(this);
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      instances.delete(this);
      return onRemoved?.apply(this, arguments);
    };
  },
});
