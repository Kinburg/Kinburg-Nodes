import { app } from "../../scripts/app.js";

// Group Control — a client-side panel for the KinburgGroupControl node.
//
// Lists every *unique* group name in the workflow and lets you flip all groups with that name
// between Always (active) and Bypass (skipped). "Mode" of a group means the `mode` of the nodes
// inside it — exactly what ComfyUI's own "Set Group Nodes to …" menu does — so the effect is
// baked into the target nodes and travels with the workflow. Nesting is resolved by bounding
// box: an outer group covers the nodes of any group nested inside it, and nested names are
// shown indented. The panel polls the graph so the list grows/rebuilds as groups change.
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
  const allOn = mkBtn("all on", "Set every group to Always", () => {
    for (const e of node._kbEntries || []) applyMode(e, ALWAYS());
    refresh(node);
  });
  const allOff = mkBtn("all off", "Bypass every group", () => {
    for (const e of node._kbEntries || []) applyMode(e, BYPASS());
    refresh(node);
  });
  header.append(title, count, sortBtn, allOn, allOff);

  const list = document.createElement("div");
  list.style.cssText = "display:flex;flex-direction:column;gap:3px;";

  const empty = document.createElement("div");
  empty.textContent = "No named groups in this workflow.";
  empty.style.cssText = "color:#777;padding:8px 6px;font-style:italic;";

  root.append(header, list, empty);
  node._kbEls = { root, list, count, empty };
  return root;
}

// Rebuild the rows from scratch (structure changed).
function rebuild(node) {
  const els = node._kbEls;
  if (!els) return;
  const entries = applyOrder(collectEntries(), node);
  node._kbEntries = entries;
  els.list.innerHTML = "";
  els.count.textContent = entries.length ? `${entries.length}` : "";
  els.empty.style.display = entries.length ? "none" : "";

  for (const entry of entries) {
    const row = document.createElement("div");
    row.style.cssText =
      "display:flex;align-items:center;gap:8px;padding:3px 6px;border-radius:5px;" +
      `background:#00000022;margin-left:${entry.depth * 14}px;`;

    const grip = document.createElement("span");
    grip.textContent = "⠿";
    grip.title = "Drag to reorder";
    grip.style.cssText =
      "flex:0 0 auto;cursor:grab;color:#666;font-size:13px;line-height:1;padding:0 1px;" +
      "user-select:none;touch-action:none;";

    const dot = document.createElement("span");
    dot.style.cssText =
      `width:9px;height:9px;border-radius:50%;flex:0 0 auto;` +
      `background:${entry.color || "#666"};border:1px solid #0006;`;

    const name = document.createElement("span");
    name.textContent = entry.name;
    if (entry.groups.length > 1) name.textContent += `  ×${entry.groups.length}`;
    name.title = entry.name + (entry.groups.length > 1 ? ` (${entry.groups.length} groups)` : "");
    name.style.cssText =
      "flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#ddd;";

    const sw = document.createElement("button");
    sw.style.cssText =
      "cursor:pointer;border:none;border-radius:10px;width:64px;height:20px;flex:0 0 auto;" +
      "font-size:10px;font-weight:600;letter-spacing:.03em;color:#fff;";
    sw.onclick = (e) => {
      e.preventDefault();
      const on = stateOf(entry) === "always";
      applyMode(entry, on ? BYPASS() : ALWAYS());
      refresh(node);
    };
    // Right-click a switch to Mute (Never) instead of Bypass.
    sw.oncontextmenu = (e) => {
      e.preventDefault();
      applyMode(entry, stateOf(entry) === "mute" ? ALWAYS() : MUTE());
      refresh(node);
    };

    row._kbEntry = entry;
    row._kbSwitch = sw;
    row.append(grip, dot, name, sw);
    attachDrag(node, row, grip);
    els.list.appendChild(row);
  }
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
      return onRemoved?.apply(this, arguments);
    };
  },
});
