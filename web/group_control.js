import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Group Control — a client-side panel for the KinburgGroupControl node.
//
// Lists every *unique* group name in the workflow and lets you flip all groups with that name
// between Always (active), Bypass (skipped) and Never (muted). "Mode" of a group means the `mode`
// of the nodes inside it — exactly what ComfyUI's own "Set Group Nodes to …" menu does — so the
// effect is baked into the target nodes and travels with the workflow. Nesting is resolved by
// bounding box: an outer group covers the nodes of any group nested inside it, and nested names
// are shown indented. The panel polls the graph so the list grows/rebuilds as groups change.
//
// Per row:
//   * left-click the switch  → toggle Always / Bypass (the fast on/off).
//   * ▶ button               → run ONLY this group: queues its output nodes as ComfyUI
//                              "partial execution targets", so just this group and the nodes it
//                              depends on run (like the core "Queue Selected Output Nodes").
//   * ⋯ button / right-click → a menu with Run · Always · Bypass · Never (the discoverable way
//                              to set Never / mute a group).
//
// The rows can be reordered: "sort" sorts by name (toggling A–Z / Z–A; right-click restores the
// original order), or drag a row by its ⠿ handle. The chosen order is saved on the node
// (node.properties._kbOrder) so it travels with the workflow; it's purely cosmetic.

const CLASS = "KinburgGroupControl";

const LG = () => window.LiteGraph || {};
const ALWAYS = () => LG().ALWAYS ?? 0;
const MUTE   = () => LG().NEVER  ?? 2;   // ComfyUI's "Never"
const BYPASS = () => LG().BYPASS ?? 4;

// ---------------------------------------------------------------- graph / group introspection
function getGroups() {
  const g = app.graph;
  if (!g) return [];
  return g._groups || g.groups || [];
}

function bboxOf(group) {
  const b = group._bounding || group.bounding;
  if (b && b.length >= 4) return [b[0], b[1], b[2], b[3]];
  if (group.pos && group.size) return [group.pos[0], group.pos[1], group.size[0], group.size[1]];
  return null;
}

// Nodes belonging to a group. Prefer LiteGraph's own computation; fall back to bbox containment
// (which also picks up nodes of nested groups, so an outer group naturally covers inner ones).
function groupNodes(group) {
  try { group.recomputeInsideNodes?.(); } catch (e) { /* older/newer API — fall through */ }
  let ns = group._nodes || group.nodes;
  if (Array.isArray(ns) && ns.length) return ns;
  const b = bboxOf(group);
  if (!b) return [];
  const [x, y, w, h] = b;
  return (app.graph?._nodes || []).filter((n) => {
    const p = n.pos, s = n.size;
    if (!p || !s) return false;
    const cx = p[0] + s[0] / 2, cy = p[1] + s[1] / 2;
    return cx >= x && cx <= x + w && cy >= y && cy <= y + h;
  });
}

// How many other groups fully contain this one (→ nesting depth for indentation).
function depthOf(group, all) {
  const b = bboxOf(group);
  if (!b) return 0;
  const [x, y, w, h] = b, area = w * h;
  let d = 0;
  for (const o of all) {
    if (o === group) continue;
    const ob = bboxOf(o);
    if (!ob) continue;
    const [ox, oy, ow, oh] = ob;
    if (x >= ox && y >= oy && x + w <= ox + ow && y + h <= oy + oh && ow * oh > area) d++;
  }
  return d;
}

// Collapse groups to one entry per unique (trimmed) title, in first-appearance order.
function collectEntries() {
  const groups = getGroups();
  const byName = new Map();
  for (const g of groups) {
    const name = String(g.title ?? "").trim();
    if (!name) continue;
    let e = byName.get(name);
    if (!e) { e = { name, groups: [], depth: depthOf(g, groups), color: g.color }; byName.set(name, e); }
    e.groups.push(g);
  }
  return [...byName.values()];
}

// -------------------------------------------------------------------- ordering (sort / drag)
// The desired row order is a list of group names saved on the node (node.properties._kbOrder),
// so it serializes with the workflow and survives a reload. It's UI-only and never touches the
// prompt. Both the "sort" button and drag-to-reorder write into this same list.
function setOrder(node, names) {
  node.properties = node.properties || {};
  node.properties._kbOrder = names;
}

// Reorder entries to match the saved order. Names not in the list keep their natural
// first-appearance order, placed after the ones that are (so new groups show up at the bottom).
function applyOrder(entries, node) {
  const order = node.properties && node.properties._kbOrder;
  if (!Array.isArray(order) || !order.length) return entries;
  const pos = new Map(order.map((n, i) => [n, i]));
  const orig = new Map(entries.map((e, i) => [e.name, i]));
  const big = order.length + entries.length;
  return entries.slice().sort((a, b) => {
    const pa = pos.has(a.name) ? pos.get(a.name) : big + orig.get(a.name);
    const pb = pos.has(b.name) ? pos.get(b.name) : big + orig.get(b.name);
    return pa - pb;
  });
}

// Sort by name; each click flips A→Z / Z→A. Natural (numeric-aware) collation.
function sortByName(node) {
  const asc = !(node.properties && node.properties._kbSortAsc === true);
  const names = collectEntries()
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }));
  if (!asc) names.reverse();
  setOrder(node, names);
  node.properties._kbSortAsc = asc;
  rebuild(node);
}

// Drop the custom order — back to first-appearance order.
function clearOrder(node) {
  if (node.properties) {
    delete node.properties._kbOrder;
    delete node.properties._kbSortAsc;
  }
  rebuild(node);
}

// Pointer-driven drag reorder from a row's grip handle. Reorders the DOM live as the pointer
// moves, then commits the new row order to node.properties on release. paint()/rebuild() are
// held off while dragging (see refresh) so the list DOM isn't rebuilt mid-drag.
//
// Pointer capture and the move/up listeners live on the LIST, never on the row/handle: the
// dragged row is reparented by insertBefore on every step, which momentarily detaches it and
// would drop capture held on it — the drag would then stall after one move and never get its
// pointerup (so the highlight would stick). The list stays put, so capture survives the drag.
function attachDrag(node, row, handle) {
  handle.addEventListener("pointerdown", (downEvt) => {
    if (downEvt.button !== 0) return; // left button only
    if (node._kbDragging) return;     // ignore a second pointer mid-drag
    downEvt.preventDefault();
    downEvt.stopPropagation();

    const list = node._kbEls && node._kbEls.list;
    if (!list) return;
    node._kbDragging = true;
    handle.style.cursor = "grabbing";
    const prevBg = row.style.background;
    const prevOpacity = row.style.opacity;
    row.style.opacity = "0.55";
    row.style.background = "#2b6cb055";
    const pid = downEvt.pointerId;
    try { list.setPointerCapture(pid); } catch (e) { /* not fatal */ }

    const onMove = (e) => {
      const y = e.clientY;
      let before = null;
      for (const sib of list.children) {
        if (sib === row) continue;
        const r = sib.getBoundingClientRect();
        if (y < r.top + r.height / 2) { before = sib; break; }
      }
      if (before) {
        if (before !== row.nextElementSibling) list.insertBefore(row, before);
      } else if (list.lastElementChild !== row) {
        list.appendChild(row);
      }
    };

    const finish = () => {
      list.removeEventListener("pointermove", onMove);
      list.removeEventListener("pointerup", finish);
      list.removeEventListener("pointercancel", finish);
      try { list.releasePointerCapture(pid); } catch (e) { /* already released */ }
      handle.style.cursor = "grab";
      row.style.opacity = prevOpacity;
      row.style.background = prevBg;
      node._kbDragging = false;
      const names = [...list.children].map((c) => c._kbEntry && c._kbEntry.name).filter(Boolean);
      setOrder(node, names);
      app.graph?.setDirtyCanvas(true, true);
    };

    list.addEventListener("pointermove", onMove);
    list.addEventListener("pointerup", finish);
    list.addEventListener("pointercancel", finish);
  });
}

// Aggregate on/off state across every node of every same-named group.
function stateOf(entry) {
  const seen = new Set();
  let count = 0, always = true, bypass = true, mute = true;
  for (const g of entry.groups) {
    for (const n of groupNodes(g)) {
      if (seen.has(n.id)) continue;
      seen.add(n.id);
      count++;
      const m = n.mode;
      if (m === BYPASS()) { always = false; mute = false; }
      else if (m === MUTE()) { always = false; bypass = false; }
      else { bypass = false; mute = false; } // ALWAYS / anything else counts as active
    }
  }
  if (count === 0) return "empty";
  if (bypass) return "bypass";
  if (mute) return "mute";
  if (always) return "always";
  return "mixed";
}

// Rewrite the mode of every node in every same-named group.
function applyMode(entry, mode) {
  const seen = new Set();
  for (const g of entry.groups) {
    for (const n of groupNodes(g)) {
      if (seen.has(n.id)) continue;
      seen.add(n.id);
      n.mode = mode;
    }
  }
  app.graph?.setDirtyCanvas(true, true);
  app.graph?.change?.();
}

// ---------------------------------------------------------------------- run a single group
// A node is an "output node" (SaveImage, PreviewImage, …) when its backend class declares
// OUTPUT_NODE — the frontend carries that on constructor.nodeData.output_node.
function isOutputNode(n) {
  return !!(n && n.constructor && n.constructor.nodeData && n.constructor.nodeData.output_node);
}

// ComfyUI "partial execution target" id for a node. For the root graph this is just the node id;
// nested subgraphs would need a "parent:child" path (not built here — the panel operates on the
// currently-open graph, which is the root in virtually every workflow).
function execIdOf(n) {
  return String(n.id);
}

// De-duplicated nodes across every same-named group of an entry.
function uniqueNodes(entry) {
  const seen = new Set(), out = [];
  for (const g of entry.groups) {
    for (const n of groupNodes(g)) {
      if (seen.has(n.id)) continue;
      seen.add(n.id);
      out.push(n);
    }
  }
  return out;
}

// Run ONLY this group. We queue the group's *active* output nodes as ComfyUI partial-execution
// targets: the backend then runs only those outputs and whatever they depend on, skipping every
// unrelated branch — exactly like the core "Queue Selected Output Nodes" command. Nothing here
// changes node modes, so the run reflects the group's current on/off state.
async function runGroup(node, entry) {
  const nodes = uniqueNodes(entry);
  const outputs = nodes.filter(isOutputNode);
  if (!outputs.length) {
    flash(node, `“${entry.name}” has no output node (Save/Preview) to run`, true);
    return;
  }
  const active = outputs.filter((n) => n.mode === ALWAYS());
  if (!active.length) {
    flash(node, `“${entry.name}” is off — switch it on to run`, true);
    return;
  }
  try {
    const prompt = await app.graphToPrompt();
    await api.queuePrompt(0, prompt, { partialExecutionTargets: active.map(execIdOf) });
    flash(node, `▶ queued “${entry.name}”`);
  } catch (e) {
    console.error("[GroupControl] run failed", e);
    const msg = (e && (e.response?.error?.message || e.message)) || String(e);
    flash(node, `run failed: ${msg}`, true);
  }
}

// ------------------------------------------------------------------ focus a group on the canvas
// Pan/zoom the canvas onto the union bounding box of every same-named group in this entry, using
// LiteGraph's own animated helper (the same one the core "focus node/group" uses). Falls back to
// a manual offset/scale for older builds. Purely a view change — never touches node modes.
function focusEntry(entry) {
  const canvas = app.canvas;
  if (!canvas) return;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const g of entry.groups) {
    const b = bboxOf(g);
    if (!b) continue;
    x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]);
    x1 = Math.max(x1, b[0] + b[2]); y1 = Math.max(y1, b[1] + b[3]);
  }
  if (!isFinite(x0)) return;
  const w = x1 - x0, h = y1 - y0, bounds = [x0, y0, w, h];
  try {
    if (typeof canvas.animateToBounds === "function") { canvas.animateToBounds(bounds, { zoom: 0.7 }); return; }
    if (canvas.ds && typeof canvas.ds.animateToBounds === "function") {
      canvas.ds.animateToBounds(bounds, () => canvas.setDirty(true, true), { zoom: 0.7 });
      return;
    }
  } catch (e) { /* fall through to manual centering */ }
  const ds = canvas.ds, el = canvas.canvas;
  if (!ds || !el || w <= 0 || h <= 0) return;
  const vw = el.clientWidth || el.width, vh = el.clientHeight || el.height;
  let scale = Math.min(vw / w, vh / h) * 0.7;
  scale = Math.max(0.1, Math.min(scale, 1.2));
  ds.scale = scale;
  ds.offset[0] = vw / (2 * scale) - (x0 + w / 2);
  ds.offset[1] = vh / (2 * scale) - (y0 + h / 2);
  canvas.setDirty(true, true);
}

// ---------------------------------------------------------------------------- solo (isolate)
// Persistent isolate: set THIS group to Always and every other listed group to Bypass, so a run
// executes only this branch. Snapshots the prior modes (in memory) so "Undo solo" can restore
// them. Only nodes that belong to groups are touched; ungrouped nodes are left alone. Others are
// bypassed first and the target set Always last, so the soloed group wins even under nesting.
function soloEntry(node, entry) {
  const entries = node._kbEntries || collectEntries();
  const snap = new Map();
  for (const e of entries) for (const g of e.groups) for (const n of groupNodes(g)) {
    if (!snap.has(n.id)) snap.set(n.id, n.mode);
  }
  node._kbSolo = { name: entry.name, modes: snap };
  for (const e of entries) if (e.name !== entry.name) applyMode(e, BYPASS());
  applyMode(entry, ALWAYS());
  refresh(node);
  flash(node, `◎ solo “${entry.name}” — rest bypassed`);
}

// Restore the modes captured by the last solo.
function undoSolo(node) {
  const solo = node._kbSolo;
  if (!solo) return;
  const byId = new Map();
  for (const n of (app.graph?._nodes || [])) byId.set(n.id, n);
  for (const [id, mode] of solo.modes) { const n = byId.get(id); if (n) n.mode = mode; }
  node._kbSolo = null;
  app.graph?.setDirtyCanvas(true, true);
  app.graph?.change?.();
  refresh(node);
  flash(node, "↩ solo undone");
}

// Per-row menu (⋯ button or right-click): the discoverable home for every per-group action,
// including Never, Focus and Solo. Uses LiteGraph's own context menu so it looks native.
function openRowMenu(node, entry, evt) {
  const CM = LG().ContextMenu;
  if (!CM) return;
  const items = [
    { content: "▶ Run this group", callback: () => runGroup(node, entry) },
    { content: "🔍 Focus on canvas", callback: () => focusEntry(entry) },
    null,
    { content: "● Always (on)", callback: () => { applyMode(entry, ALWAYS()); refresh(node); } },
    { content: "⇄ Bypass",      callback: () => { applyMode(entry, BYPASS()); refresh(node); } },
    { content: "✕ Never (mute)", callback: () => { applyMode(entry, MUTE());  refresh(node); } },
    null,
    { content: "◎ Solo — only this group", callback: () => soloEntry(node, entry) },
  ];
  if (node._kbSolo) {
    items.push({ content: `↩ Undo solo (${node._kbSolo.name})`, callback: () => undoSolo(node) });
  }
  new CM(items, { event: evt, title: entry.name });
}

// A short-lived status line under the header — feedback for Run without blocking dialogs
// (window.alert/prompt are unavailable in the desktop app, so we never use them).
function flash(node, msg, isError) {
  const el = node._kbEls && node._kbEls.status;
  if (!el) return;
  el.textContent = msg;
  el.style.color = isError ? "#e06a6a" : "#66bb6a";
  el.style.display = "";
  if (node._kbFlashTimer) clearTimeout(node._kbFlashTimer);
  node._kbFlashTimer = setTimeout(() => { el.style.display = "none"; el.textContent = ""; }, 3500);
}

// ------------------------------------------------------------------------------- name filter
// The filter is a live, view-only text match on the group name (node._kbFilter). It hides
// non-matching rows without rebuilding, and updates the count + empty message. It never touches
// the graph and isn't serialized — purely a way to find a row when there are many groups.
function applyFilter(node) {
  const els = node._kbEls;
  if (!els) return;
  const total = (node._kbEntries || []).length;
  const q = node._kbFilter || "";
  let shown = 0;
  for (const row of els.list.children) {
    const e = row._kbEntry;
    const ok = !q || (e && e.name.toLowerCase().includes(q));
    // Show with "grid" (the row's real display), NOT "" — an empty string deletes the inline
    // display and the row falls back to block, which would kill the grid column alignment.
    row.style.display = ok ? "grid" : "none";
    if (ok) shown++;
  }
  els.count.textContent = total ? (q ? `${shown}/${total}` : `${total}`) : "";
  if (!total) { els.empty.textContent = "No named groups in this workflow."; els.empty.style.display = ""; }
  else if (shown === 0) { els.empty.textContent = "No groups match the filter."; els.empty.style.display = ""; }
  else { els.empty.style.display = "none"; }
}

// Which entries the header bulk buttons act on: the filtered (visible) set when a filter is
// active, otherwise every entry.
function bulkEntries(node) {
  const all = node._kbEntries || [];
  const q = node._kbFilter || "";
  return q ? all.filter((e) => e.name.toLowerCase().includes(q)) : all;
}

// A stable signature of the group layout; a change means the list must be rebuilt.
function signature() {
  return getGroups()
    .map((g) => {
      const b = bboxOf(g) || [0, 0, 0, 0];
      return `${String(g.title ?? "").trim()}@${b.map((v) => Math.round(v)).join(",")}`;
    })
    .sort()
    .join("|");
}

// ------------------------------------------------------------------------------------- the UI
function buildPanel(node) {
  const root = document.createElement("div");
  root.className = "kb-gc";
  root.style.cssText =
    "display:flex;flex-direction:column;gap:4px;padding:4px 2px;overflow:auto;" +
    "font-family:inherit;font-size:12px;box-sizing:border-box;";
  // Let the canvas keep panning/zooming everywhere except our interactive controls.
  root.addEventListener("wheel", (e) => e.stopPropagation());
  root.addEventListener("pointerdown", (e) => e.stopPropagation());

  const header = document.createElement("div");
  header.style.cssText =
    "display:flex;align-items:center;gap:8px;padding:0 4px 2px;color:#9a9aa2;" +
    "text-transform:uppercase;letter-spacing:.04em;font-size:10px;";
  const title = document.createElement("span");
  title.textContent = "Groups";
  title.style.flex = "1";
  const count = document.createElement("span");
  const mkBtn = (label, tip, fn) => {
    const b = document.createElement("button");
    b.textContent = label; b.title = tip;
    b.style.cssText =
      "cursor:pointer;border:1px solid #3a3a44;background:#2a2a32;color:#ccc;" +
      "border-radius:4px;font-size:10px;padding:1px 6px;";
    b.onclick = (e) => { e.preventDefault(); fn(); };
    return b;
  };
  const sortBtn = mkBtn(
    "sort ⇅",
    "Sort groups by name — click toggles A–Z / Z–A. Right-click: restore original order. " +
    "You can also drag rows by the ⠿ handle to reorder manually.",
    () => {
      sortByName(node);
      sortBtn.textContent = node.properties && node.properties._kbSortAsc ? "A–Z" : "Z–A";
    });
  sortBtn.oncontextmenu = (e) => {
    e.preventDefault();
    clearOrder(node);
    sortBtn.textContent = "sort ⇅";
  };
  // Bulk buttons act on the filtered set when a filter is active (else everything).
  const allOn = mkBtn("all on", "Set matching groups to Always (all when no filter)", () => {
    for (const e of bulkEntries(node)) applyMode(e, ALWAYS());
    refresh(node);
  });
  const allOff = mkBtn("all off", "Bypass matching groups (all when no filter)", () => {
    for (const e of bulkEntries(node)) applyMode(e, BYPASS());
    refresh(node);
  });
  const allNever = mkBtn("all ✕", "Set matching groups to Never/mute (all when no filter)", () => {
    for (const e of bulkEntries(node)) applyMode(e, MUTE());
    refresh(node);
  });
  header.append(title, count, sortBtn, allOn, allOff, allNever);

  // Live name filter — hides non-matching rows (view only; never touches the graph).
  const filter = document.createElement("input");
  filter.type = "text";
  filter.placeholder = "filter groups…";
  filter.style.cssText =
    "margin:0 4px;padding:2px 6px;border:1px solid #3a3a44;background:#1e1e24;color:#ddd;" +
    "border-radius:4px;font-size:11px;box-sizing:border-box;outline:none;";
  filter.addEventListener("input", () => {
    node._kbFilter = filter.value.trim().toLowerCase();
    applyFilter(node);
  });
  // Keep keystrokes/clicks inside the box — don't trigger canvas shortcuts or node drag.
  filter.addEventListener("keydown", (e) => e.stopPropagation());
  filter.addEventListener("pointerdown", (e) => e.stopPropagation());

  // Transient feedback line for Run/Solo (queued / errors); hidden until something happens.
  const status = document.createElement("div");
  status.style.cssText =
    "display:none;padding:1px 6px 2px;font-size:10px;line-height:1.3;" +
    "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";

  const list = document.createElement("div");
  list.style.cssText = "display:flex;flex-direction:column;gap:3px;";

  const empty = document.createElement("div");
  empty.textContent = "No named groups in this workflow.";
  empty.style.cssText = "color:#777;padding:8px 6px;font-style:italic;";

  root.append(header, filter, status, list, empty);
  node._kbEls = { root, list, count, empty, status, filter };
  return root;
}

// Rebuild the rows from scratch (structure changed).
function rebuild(node) {
  const els = node._kbEls;
  if (!els) return;
  const entries = applyOrder(collectEntries(), node);
  node._kbEntries = entries;
  els.list.innerHTML = "";

  for (const entry of entries) {
    // Fixed grid columns keep the controls in the same place on every row, whatever the name
    // length: [grip][dot][name grows][▶][switch][⋯]. NOTE: applyFilter() must re-show rows with
    // display:"grid" (not ""), or it wipes this inline display and the row falls back to block.
    const row = document.createElement("div");
    row.style.cssText =
      "display:grid;grid-template-columns:auto auto minmax(0,1fr) auto auto auto;" +
      "align-items:center;column-gap:6px;padding:3px 6px;border-radius:5px;" +
      `background:#00000022;margin-left:${entry.depth * 14}px;`;

    const grip = document.createElement("span");
    grip.textContent = "⠿";
    grip.title = "Drag to reorder";
    grip.style.cssText =
      "cursor:grab;color:#666;font-size:13px;line-height:1;padding:0 1px;" +
      "user-select:none;touch-action:none;";

    // Dot + name focus the group on the canvas (pan/zoom to it).
    const dot = document.createElement("span");
    dot.title = "Focus this group on the canvas";
    dot.style.cssText =
      `width:9px;height:9px;border-radius:50%;cursor:pointer;` +
      `background:${entry.color || "#666"};border:1px solid #0006;`;
    dot.onclick = (e) => { e.preventDefault(); focusEntry(entry); };

    const name = document.createElement("span");
    name.textContent = entry.name;
    if (entry.groups.length > 1) name.textContent += `  ×${entry.groups.length}`;
    name.title = entry.name + (entry.groups.length > 1 ? ` (${entry.groups.length} groups)` : "") +
      " — click to focus on canvas";
    name.style.cssText =
      "min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#ddd;cursor:pointer;";
    name.onclick = (e) => { e.preventDefault(); focusEntry(entry); };

    // ▶ run only this group.
    const run = document.createElement("button");
    run.textContent = "▶";
    run.title = "Run only this group (queue its output nodes + what they depend on)";
    run.style.cssText =
      "cursor:pointer;border:1px solid #2e5d34;background:#1f3a24;color:#8fd69b;" +
      "border-radius:4px;width:22px;height:20px;font-size:10px;line-height:1;padding:0;";
    run.onclick = (e) => { e.preventDefault(); runGroup(node, entry); };

    const sw = document.createElement("button");
    sw.style.cssText =
      "cursor:pointer;border:none;border-radius:10px;width:64px;height:20px;" +
      "font-size:10px;font-weight:600;letter-spacing:.03em;color:#fff;padding:0;";
    sw.onclick = (e) => {
      e.preventDefault();
      const on = stateOf(entry) === "always";
      applyMode(entry, on ? BYPASS() : ALWAYS());
      refresh(node);
    };
    // Right-click the switch → the full per-group menu (Run / Always / Bypass / Never).
    sw.oncontextmenu = (e) => { e.preventDefault(); openRowMenu(node, entry, e); };

    // ⋯ menu: the discoverable home for Always / Bypass / Never (+ Run).
    const menu = document.createElement("button");
    menu.textContent = "⋯";
    menu.title = "More: Run · Focus · Always / Bypass / Never · Solo";
    menu.style.cssText =
      "cursor:pointer;border:1px solid #3a3a44;background:#2a2a32;color:#bbb;" +
      "border-radius:4px;width:20px;height:20px;font-size:12px;line-height:1;padding:0;";
    menu.onclick = (e) => { e.preventDefault(); openRowMenu(node, entry, e); };

    row._kbEntry = entry;
    row._kbSwitch = sw;
    // Right-click anywhere on the row also opens the menu (except the switch, handled above).
    row.oncontextmenu = (e) => { e.preventDefault(); openRowMenu(node, entry, e); };
    row.append(grip, dot, name, run, sw, menu);
    attachDrag(node, row, grip);
    els.list.appendChild(row);
  }
  applyFilter(node); // re-apply the name filter + refresh count/empty for the new rows
  paint(node);
}

// Repaint the switches from the live node modes (cheap; runs every tick).
function paint(node) {
  const els = node._kbEls;
  if (!els) return;
  for (const row of els.list.children) {
    const entry = row._kbEntry, sw = row._kbSwitch;
    if (!entry || !sw) continue;
    const st = stateOf(entry);
    if (st === "always") { sw.textContent = "ALWAYS"; sw.style.background = "#2e7d32"; }
    else if (st === "bypass") { sw.textContent = "BYPASS"; sw.style.background = "#6a3ea1"; }
    else if (st === "mute") { sw.textContent = "MUTE"; sw.style.background = "#8a1f1f"; }
    else if (st === "mixed") { sw.textContent = "MIXED"; sw.style.background = "#7a6a1f"; }
    else { sw.textContent = "empty"; sw.style.background = "#444"; sw.style.opacity = ".6"; }
  }
}

// Rebuild if the layout changed, otherwise just repaint the switches.
function refresh(node) {
  if (node._kbDragging) return; // don't rebuild/repaint the list out from under a drag
  const sig = signature();
  if (sig !== node._kbSig) { node._kbSig = sig; rebuild(node); }
  else paint(node);
}

app.registerExtension({
  name: "Kinburg.GroupControl",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const node = this;
      const root = buildPanel(node);
      node.addDOMWidget("kb_group_control", "kinburg_groupctl", root, {
        serialize: false,
        getMinHeight: () => 120,
        getMaxHeight: () => 100000,
      });
      node._kbSig = null;
      // Defer the first build a tick so a freshly-loaded workflow's groups exist.
      const tick = () => { if (!node.graph) return; refresh(node); };
      node._kbTimer = setInterval(tick, 500);
      setTimeout(tick, 50);

      if ((node.size?.[1] || 0) < 200) node.setSize([Math.max(node.size?.[0] || 0, 300), 260]);
      return r;
    };

    // A saved workflow restores node.properties (incl. _kbOrder) via configure(), which may run
    // AFTER the first poll tick has already locked _kbSig to the current group layout — and since
    // the order isn't part of the signature, no later rebuild would re-apply it. Reset _kbSig here
    // (properties are populated by now) so the next tick rebuilds and honors the saved order.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      this._kbSig = null;
      if (this.graph) refresh(this);
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this._kbTimer) { clearInterval(this._kbTimer); this._kbTimer = null; }
      if (this._kbFlashTimer) { clearTimeout(this._kbFlashTimer); this._kbFlashTimer = null; }
      return onRemoved?.apply(this, arguments);
    };
  },
});
