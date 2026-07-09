import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Show Text (Markdown) — display any input as text.
//   • markdown toggle flips between a rendered preview and an editable raw textarea,
//     inside a fixed-size scroll box, so the node never resizes when you toggle.
//   • the shown text lives in node.properties (serialized into the workflow), so it
//     survives switching between workflow tabs — unlike the core Preview Text node.
//   • 💾 Save posts the text to /kinburg/showtext/save (writes a .md file); 📋 Copy
//     copies it to the clipboard; a header shows a char/line counter.

const CLASS = "KinburgShowText";
const wv = (node, name) => node.widgets?.find((w) => w.name === name);

// ---------------------------------------------------------------- minimal markdown renderer
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Inline spans, run on already-escaped text. Split on code spans so their contents are
// emitted verbatim (no bold/italic/link processing inside `code`).
function inline(s) {
  return s.split(/(`[^`]+`)/g).map((seg) => {
    if (seg.length >= 2 && seg.startsWith("`") && seg.endsWith("`")) {
      return `<code>${seg.slice(1, -1)}</code>`;
    }
    return seg
      .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, '<img alt="$1" src="$2" style="max-width:100%">')
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>");
  }).join("");
}

// Runs on already-escaped text, so blockquote markers appear as "&gt;", not ">".
const BLOCK_START = /^\s*(#{1,6}\s|&gt;|[-*+]\s|\d+[.)]\s|```)/;
const HR = /^\s*([-*_])\1\1+\s*$/;

function renderMarkdown(src) {
  const lines = escapeHtml(src || "").split(/\r?\n/);
  const out = [];
  let i = 0, list = null; // list: "ul" | "ol" | null
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };

  while (i < lines.length) {
    const line = lines[i];

    const fence = line.match(/^\s*```/);
    if (fence) {
      closeList();
      const buf = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
      i++; // closing fence
      out.push(`<pre class="kb-code"><code>${buf.join("\n")}</code></pre>`);
      continue;
    }
    if (HR.test(line)) { closeList(); out.push("<hr>"); i++; continue; }

    const h = line.match(/^\s*(#{1,6})\s+(.*)$/);
    if (h) { closeList(); const lvl = h[1].length; out.push(`<h${lvl}>${inline(h[2])}</h${lvl}>`); i++; continue; }

    if (/^\s*&gt;\s?/.test(line)) {
      closeList();
      const buf = [];
      while (i < lines.length && /^\s*&gt;\s?/.test(lines[i])) buf.push(lines[i++].replace(/^\s*&gt;\s?/, ""));
      out.push(`<blockquote>${inline(buf.join("<br>"))}</blockquote>`);
      continue;
    }

    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    if (ul) { if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inline(ul[1])}</li>`); i++; continue; }

    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ol) { if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inline(ol[1])}</li>`); i++; continue; }

    if (/^\s*$/.test(line)) { closeList(); i++; continue; }

    // paragraph: gather consecutive plain lines
    closeList();
    const buf = [line];
    i++;
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !BLOCK_START.test(lines[i]) && !HR.test(lines[i])) buf.push(lines[i++]);
    out.push(`<p>${inline(buf.join("<br>"))}</p>`);
  }
  closeList();
  return out.join("\n");
}

// ---------------------------------------------------------------- styles (injected once)
function injectStyle() {
  if (document.getElementById("kb-showtext-style")) return;
  const s = document.createElement("style");
  s.id = "kb-showtext-style";
  s.textContent = `
  .kb-md{color:#dfe7e2}
  .kb-md h1,.kb-md h2,.kb-md h3,.kb-md h4,.kb-md h5,.kb-md h6{margin:.6em 0 .3em;line-height:1.25;color:#fff}
  .kb-md h1{font-size:1.5em;border-bottom:1px solid #333;padding-bottom:.2em}
  .kb-md h2{font-size:1.3em;border-bottom:1px solid #2a2a2a;padding-bottom:.15em}
  .kb-md h3{font-size:1.13em}.kb-md h4{font-size:1em}
  .kb-md p{margin:.5em 0}
  .kb-md ul,.kb-md ol{margin:.4em 0;padding-left:1.5em}
  .kb-md li{margin:.15em 0}
  .kb-md code{background:#2b2b2b;border-radius:4px;padding:.1em .35em;font:12px ui-monospace,Consolas,monospace}
  .kb-md pre.kb-code{background:#111;border:1px solid #2b2b2b;border-radius:6px;padding:8px 10px;overflow:auto;margin:.5em 0}
  .kb-md pre.kb-code code{background:none;padding:0;color:#cfe6da}
  .kb-md blockquote{margin:.5em 0;padding:.2em .8em;border-left:3px solid #4a4a52;color:#b8b8be}
  .kb-md a{color:#5aa9ff;text-decoration:none}
  .kb-md a:hover{text-decoration:underline}
  .kb-md hr{border:0;border-top:1px solid #333;margin:.8em 0}
  .kb-md img{border-radius:4px}
  `;
  document.head.appendChild(s);
}

// ---------------------------------------------------------------- DOM
function buildDisplay() {
  const root = document.createElement("div");
  root.style.cssText = "display:flex;flex-direction:column;width:100%;height:100%;box-sizing:border-box;";

  const bar = document.createElement("div");
  bar.style.cssText = "flex:0 0 auto;display:flex;justify-content:flex-end;color:#8a8a92;font:11px ui-monospace,Consolas,monospace;padding:1px 4px 3px;user-select:none;";
  const counter = document.createElement("span");
  bar.appendChild(counter);

  const box = document.createElement("div");
  box.style.cssText = "flex:1 1 auto;position:relative;min-height:80px;background:#181818;border:1px solid #2b2b2b;border-radius:6px;overflow:hidden;";

  const ta = document.createElement("textarea");
  ta.spellcheck = false;
  ta.placeholder = "(no text — connect an input and run, or type here)";
  ta.style.cssText = "position:absolute;inset:0;width:100%;height:100%;box-sizing:border-box;resize:none;border:0;outline:0;background:transparent;color:#e6e6e6;font:12px/1.5 ui-monospace,Consolas,monospace;padding:8px;white-space:pre-wrap;word-break:break-word;overflow:auto;";

  const pv = document.createElement("div");
  pv.className = "kb-md";
  pv.style.cssText = "position:absolute;inset:0;overflow:auto;font:13px/1.55 -apple-system,Segoe UI,sans-serif;padding:8px 10px;word-break:break-word;";

  box.append(ta, pv);
  root.append(bar, box);
  return { root, ta, pv, counter };
}

// ---------------------------------------------------------------- actions
function copyText(node) {
  const text = node.properties?.kb_text || "";
  if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).catch(() => {});
}

async function saveToDisk(node) {
  const text = node.properties?.kb_text || "";
  const path = (wv(node, "save_path")?.value || "").trim();
  if (!path) { alert("Set a save_path first (e.g. notes/report — it's saved as .md under ComfyUI/output)."); return; }
  try {
    const r = await api.fetchApi("/kinburg/showtext/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, text }),
    });
    const j = await r.json();
    if (!j || !j.ok) throw new Error(j?.error || "save failed");
    alert("Saved:\n" + j.path);
  } catch (e) {
    alert("Save failed: " + e.message);
  }
}

// ---------------------------------------------------------------- node wiring
function setupNode(node) {
  injectStyle();
  node.properties = node.properties || {};
  if (node.properties.kb_text == null) node.properties.kb_text = "";

  const { root, ta, pv, counter } = buildDisplay();

  const setCounter = (text) => {
    const n = text.length;
    const lines = text ? text.split(/\r\n|\r|\n/).length : 0;
    counter.textContent = `${n} chars · ${lines} lines`;
  };

  const render = () => {
    const text = node.properties.kb_text || "";
    const isMd = wv(node, "markdown")?.value ?? true;
    if (isMd) {
      pv.innerHTML = renderMarkdown(text);
      pv.style.display = "";
      ta.style.display = "none";
    } else {
      if (ta.value !== text) ta.value = text;
      ta.style.display = "";
      pv.style.display = "none";
    }
    setCounter(text);
  };
  node._kbRender = render;

  // Editing in raw mode updates the stored text (no re-render, so typing isn't interrupted).
  ta.addEventListener("input", () => { node.properties.kb_text = ta.value; setCounter(ta.value); });
  // Keep canvas pan/zoom from eating interactions with the field.
  ta.addEventListener("pointerdown", (e) => e.stopPropagation());
  root.addEventListener("wheel", (e) => e.stopPropagation());

  // Frontend-only controls, placed just above the display.
  node.addWidget("button", "💾 Save .md", null, () => saveToDisk(node), { serialize: false });
  node.addWidget("button", "📋 Copy", null, () => copyText(node), { serialize: false });
  node.addDOMWidget("kb_display", "kinburg_showtext", root, { serialize: false });

  // Re-render when the markdown toggle flips.
  const md = wv(node, "markdown");
  if (md) { const prev = md.callback; md.callback = function (...a) { const r = prev?.apply(this, a); render(); return r; }; }

  render();
  const w = Math.max(node.size?.[0] || 0, 340);
  if ((node.size?.[1] || 0) < 340) node.setSize([w, 360]);
}

app.registerExtension({
  name: "Kinburg.ShowText",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      setupNode(this);
      return r;
    };

    // Fresh execution output → store it and re-render.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const t = message?.kinburg_showtext?.[0];
      if (t == null) return;
      this.properties = this.properties || {};
      this.properties.kb_text = String(t);
      this._kbRender?.();
    };

    // Workflow load / tab switch restores properties → re-render from them.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      this._kbRender?.();
      return r;
    };
  },
});
