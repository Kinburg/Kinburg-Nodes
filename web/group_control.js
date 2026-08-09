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
// Rows can also be HIDDEN (⋯ → "Hide from this list") — for set-and-forget groups (base loaders,
// VAE, …) you never toggle and don't want in the way. Hidden names are saved on the node
// (node.properties._kbHidden) so they travel with the workflow, and hidden groups are excluded
// from the bulk buttons and from Solo, so "all off" can never switch off your loaders. The header
// 👁 button reveals hidden rows (dimmed) so they can still be used or unhidden.
//
// Groups can also be LINKED, so flipping one flips the others (⋯ → "Toggle with…", or the header
// 🔗 button for the full editor). A link is a set of group names with a rule:
//   * "follow" — polarity. Each member is + (same state as whichever group just moved) or − (the
//     opposite). "A+ B−" is the classic toggle; "A+ B− C−" means switching A off brings B and C on
//     and switching A on takes them both off; "A+ B+" is a mirror.
//   * "oneof"  — radio. Switching a member on switches every other member off (a 3-way choice can't
//     be written with polarities, which is why this is its own type).
// Links live on the node (node.properties._kbLinks) so they travel with the workflow, and they fire
// on ANY change — this panel, Ctrl+B on the canvas or ComfyUI's own group menu — because the poll
// tick diffs the members' states. Propagation is breadth-first with first-assignment-wins, so a
// contradictory set (A⇄B, B⇄C, C⇄A) settles and reports instead of oscillating.
//
// The rows can be reordered: "sort" sorts by name (toggling A–Z / Z–A; right-click restores the
// original order), or drag a row by its ⠿ handle. Reordering is TREE AWARE and respects nesting:
// a group can only be moved among its siblings (same parent), and dragging a parent carries its
// whole subtree — so the panel can't be left showing a child under an unrelated parent. The chosen
// order is saved on the node (node.properties._kbOrder) so it travels with the workflow; it's
// purely cosmetic (never changes the actual group nesting, which is defined by canvas geometry).

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

// Geometric containment between two bounding boxes (outer strictly bigger). This is how the group
// tree is derived — a group whose bbox sits inside another's is that other's child.
function areaOf(b) { return b ? b[2] * b[3] : 0; }
function strictlyContains(outer, inner) {
  if (!outer || !inner) return false;
  const [x, y, w, h] = inner, [ox, oy, ow, oh] = outer;
  return x >= ox && y >= oy && x + w <= ox + ow && y + h <= oy + oh && ow * oh > w * h;
}

// Collapse groups to one entry per unique (trimmed) title, in first-appearance order.
function collectEntries() {
  const groups = getGroups();
  const byName = new Map();
  for (const g of groups) {
    const name = String(g.title ?? "").trim();
    if (!name) continue;
    let e = byName.get(name);
    if (!e) { e = { name, groups: [], color: g.color }; byName.set(name, e); }
    e.groups.push(g);
  }
  return [...byName.values()];
}

// Build a parent/child tree over the entries from geometric containment (each entry represented by
// its first group's bbox). The immediate parent is the TIGHTEST (smallest-area) group that contains
// it. Sets e.parent (entry|null) and e.depth on every entry, and returns { roots, childrenOf } for
// pre-order traversal. This tree is what pins a group under its real parent in the panel — all
// reordering below is constrained to siblings of the same parent, so subtrees can't be interleaved.
function buildTree(entries) {
  const bb = new Map(entries.map((e) => [e, bboxOf(e.groups[0])]));
  for (const e of entries) {
    let parent = null, parentArea = Infinity;
    const be = bb.get(e);
    if (be) {
      for (const o of entries) {
        if (o === e) continue;
        const bo = bb.get(o);
        if (strictlyContains(bo, be) && areaOf(bo) < parentArea) { parent = o; parentArea = areaOf(bo); }
      }
    }
    e.parent = parent;
  }
  const childrenOf = new Map();
  const roots = [];
  for (const e of entries) {
    if (e.parent) {
      if (!childrenOf.has(e.parent)) childrenOf.set(e.parent, []);
      childrenOf.get(e.parent).push(e);
    } else {
      roots.push(e);
    }
  }
  for (const e of entries) { let d = 0, p = e.parent; while (p) { d++; p = p.parent; } e.depth = d; }
  return { roots, childrenOf };
}

// Every entry with .parent/.depth filled in. rebuild() works from this, and so does the links
// engine when the panel hasn't drawn yet (the first tick, or a headless test).
function treeEntries() {
  const all = collectEntries();
  buildTree(all);
  return all;
}

// -------------------------------------------------------------------- ordering (sort / drag)
// The desired row order is a list of group names saved on the node (node.properties._kbOrder),
// so it serializes with the workflow and survives a reload. It's UI-only and never touches the
// prompt. Both the "sort" button and drag-to-reorder write into this same list.
function setOrder(node, names) {
  node.properties = node.properties || {};
  node.properties._kbOrder = names;
}

// Produce the pre-order render list from the tree, ordering SIBLINGS (not the whole flat list) by
// the saved order. Names not listed keep first-appearance order, after the listed ones. Applying
// the order per-sibling is the core of the fix: the saved order can only shuffle children within
// one parent, never lift a group out of its subtree or interleave two subtrees.
function orderEntries(node, entries, tree) {
  const order = (node.properties && node.properties._kbOrder) || [];
  const pos = new Map(order.map((n, i) => [n, i]));
  const firstIdx = new Map(entries.map((e, i) => [e, i]));
  const big = order.length + entries.length;
  const rank = (e) => (pos.has(e.name) ? pos.get(e.name) : big + firstIdx.get(e));
  const sortSibs = (list) => list.slice().sort((a, b) => rank(a) - rank(b));
  const out = [];
  const walk = (e) => {
    out.push(e);
    const kids = tree.childrenOf.get(e);
    if (kids) for (const c of sortSibs(kids)) walk(c);
  };
  for (const r of sortSibs(tree.roots)) walk(r);
  return out;
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

// Pointer-driven drag reorder from a row's grip handle — TREE AWARE.
//   * The dragged unit is the group's whole SUBTREE (its row + all descendant rows, which are
//     contiguous in the pre-order render), so a parent always moves together with its children.
//   * It can only be dropped among its SIBLINGS (same parent, same depth) — never lifted out of
//     its subtree — so the panel can't end up showing a child under an unrelated parent.
// A thin drop-line marks the target gap; the rows aren't shuffled live (only the indicator moves),
// and the new order is committed + rebuilt on release. paint()/rebuild() are held off mid-drag
// (see refresh). Pointer capture + listeners live on the LIST so they survive DOM changes.
function attachDrag(node, row, handle) {
  handle.addEventListener("pointerdown", (downEvt) => {
    if (downEvt.button !== 0) return; // left button only
    if (node._kbDragging) return;     // ignore a second pointer mid-drag
    downEvt.preventDefault();
    downEvt.stopPropagation();

    const list = node._kbEls && node._kbEls.list;
    if (!list) return;

    const rows = [...list.children].filter((c) => c._kbEntry);
    const startIdx = rows.indexOf(row);
    if (startIdx < 0) return;
    const depth = row._kbDepth || 0;
    const parentKey = row._kbParentKey || "";

    // Dragged subtree = this row + the following rows of greater depth (contiguous in pre-order).
    let blockEnd = startIdx + 1;
    while (blockEnd < rows.length && (rows[blockEnd]._kbDepth || 0) > depth) blockEnd++;
    const block = rows.slice(startIdx, blockEnd);

    // Sibling blocks (same parent + depth), excluding the dragged one, in display order.
    const sibs = [];
    for (let i = 0; i < rows.length; i++) {
      if (i === startIdx) continue;
      if ((rows[i]._kbDepth || 0) === depth && (rows[i]._kbParentKey || "") === parentKey) {
        let e = i + 1;
        while (e < rows.length && (rows[e]._kbDepth || 0) > depth) e++;
        sibs.push({ first: rows[i], lastIdx: e - 1, firstIdx: i });
        i = e - 1;
      }
    }
    if (!sibs.length) return; // only child under its parent — nowhere to reorder to

    node._kbDragging = true;
    handle.style.cursor = "grabbing";
    const savedOpacity = block.map((r) => r.style.opacity);
    block.forEach((r) => (r.style.opacity = "0.5"));

    const line = document.createElement("div");
    line.style.cssText =
      "height:2px;background:#3b82f6;border-radius:1px;margin:0 6px;pointer-events:none;" +
      "box-shadow:0 0 4px #3b82f6;";

    // gap = insertion index among the sibling blocks [0..sibs.length]; start at the dragged row's
    // own current sibling position so the indicator doesn't jump before the first move.
    let gap = sibs.filter((s) => s.firstIdx < startIdx).length;

    const positionLine = () => {
      if (line.parentNode) line.remove();
      if (gap < sibs.length) {
        list.insertBefore(line, sibs[gap].first);
      } else {
        const after = rows[sibs[sibs.length - 1].lastIdx].nextElementSibling;
        if (after) list.insertBefore(line, after); else list.appendChild(line);
      }
    };

    const pid = downEvt.pointerId;
    try { list.setPointerCapture(pid); } catch (e) { /* not fatal */ }

    const onMove = (e) => {
      if (line.parentNode) line.remove(); // measure layout without the indicator shifting it
      const y = e.clientY;
      let k = 0;
      for (const sb of sibs) {
        const top = sb.first.getBoundingClientRect().top;
        const bottom = rows[sb.lastIdx].getBoundingClientRect().bottom;
        if (y > (top + bottom) / 2) k++; else break;
      }
      gap = k;
      positionLine();
    };

    const finish = () => {
      list.removeEventListener("pointermove", onMove);
      list.removeEventListener("pointerup", finish);
      list.removeEventListener("pointercancel", finish);
      try { list.releasePointerCapture(pid); } catch (e) { /* already released */ }
      if (line.parentNode) line.remove();
      handle.style.cursor = "grab";
      block.forEach((r, i) => (r.style.opacity = savedOpacity[i]));
      node._kbDragging = false;

      // Rebuild the full pre-order name list with the dragged block moved to `gap` among siblings.
      const rest = rows.filter((r) => !block.includes(r));
      let insertAt;
      if (gap < sibs.length) insertAt = rest.indexOf(sibs[gap].first);
      else insertAt = rest.indexOf(rows[sibs[sibs.length - 1].lastIdx]) + 1;
      const newRows = rest.slice();
      newRows.splice(insertAt, 0, ...block);
      setOrder(node, newRows.map((r) => r._kbEntry.name));
      rebuild(node);                      // re-render from the tree so depth/indent stay correct
      app.graph?.setDirtyCanvas(true, true);
    };

    list.addEventListener("pointermove", onMove);
    list.addEventListener("pointerup", finish);
    list.addEventListener("pointercancel", finish);
    positionLine();
  });
}

// ------------------------------------------------------------------------------ hidden groups
// Some groups are set-and-forget (base model loaders, VAE, LoRAs you never disable) — they only
// clutter the panel. Hiding one drops its row from the list. The hidden names live on the node
// (node.properties._kbHidden), so they serialize with the workflow just like the row order.
//
// Hiding is a PANEL-level thing only: the group keeps whatever mode it has, and hidden groups are
// deliberately EXCLUDED from the header bulk buttons and from Solo (see activeEntries) — so "all
// off" / "all ✕" / Solo can never silently switch off the groups you hid because they must stay on.
// Nothing is lost: the header 👁 button reveals hidden rows (dimmed) so they can be used again or
// unhidden, and right-clicking 👁 unhides everything at once.
function hiddenNames(node) {
  return (node.properties && node.properties._kbHidden) || [];
}

function isHidden(node, entry) {
  return hiddenNames(node).includes(entry.name);
}

// Hidden names that actually exist as rows right now (stale names from renamed/deleted groups are
// kept in properties — harmless, same as _kbOrder — but never counted).
function hiddenCount(node) {
  return (node._kbAllEntries || []).filter((e) => isHidden(node, e)).length;
}

function setHidden(node, name, hide) {
  node.properties = node.properties || {};
  const list = hiddenNames(node).filter((n) => n !== name);
  if (hide) list.push(name);
  if (list.length) node.properties._kbHidden = list;
  else delete node.properties._kbHidden;
  rebuild(node);
  app.graph?.setDirtyCanvas(true, true);
}

function unhideAll(node) {
  if (!hiddenNames(node).length) return;
  if (node.properties) delete node.properties._kbHidden;
  rebuild(node);
  flash(node, "👁 all groups unhidden");
}

function toggleShowHidden(node) {
  node._kbShowHidden = !node._kbShowHidden;
  rebuild(node);
}

// Entries the bulk buttons and Solo may touch: everything except the hidden ones — regardless of
// whether hidden rows are currently revealed. Hidden means protected.
function activeEntries(node) {
  const all = node._kbAllEntries || collectEntries();
  return all.filter((e) => !isHidden(node, e));
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

// ------------------------------------------------------------------------------------- links
// A LINK ties several groups' on/off states together, so flipping one flips the others. Links are
// stored on the node (node.properties._kbLinks), which means they serialize with the workflow just
// like _kbOrder / _kbHidden, and they reference groups BY NAME — the same unit the whole panel
// works in, so one link covers every group carrying that name.
//
//   { id, name, type: "follow" | "oneof", off: "bypass" | "never", enabled, requireOne,
//     members: [{ name, pol: 1 | -1 }, …] }
//
// See the file header for what the two types mean. Everything below splits into a PURE resolver
// (resolveLinks — no DOM, no graph, unit-tested) and the graph-facing part that applies its answer.
const LINK_COLORS = ["#4f8ef7", "#e0883a", "#38a169", "#b45ec8", "#d05c5c", "#3aa8a8"];

function linksOf(node) {
  return (node.properties && node.properties._kbLinks) || [];
}

// The links the engine may act on: enabled, and with something to propagate to.
function activeLinks(node) {
  if (node._kbLinksPaused) return [];
  return linksOf(node).filter((l) => l && l.enabled !== false && (l.members || []).length > 1);
}

function saveLinks(node, list) {
  node.properties = node.properties || {};
  if (list && list.length) node.properties._kbLinks = list;
  else delete node.properties._kbLinks;
  node._kbLinkSnap = null;   // re-learn the states before firing anything at the new set of links
  rebuild(node);
  app.graph?.setDirtyCanvas(true, true);
  app.graph?.change?.();
}

function newLinkId() {
  return "l" + Date.now().toString(36) + Math.floor(Math.random() * 46656).toString(36);
}

function linksFor(node, name) {
  return linksOf(node).filter((l) => (l.members || []).some((m) => m.name === name));
}

function linkColor(node, link) {
  const i = linksOf(node).indexOf(link);
  return LINK_COLORS[(i < 0 ? 0 : i) % LINK_COLORS.length];
}

function linkLabel(link) {
  if (link && link.name) return link.name;
  const names = ((link && link.members) || []).map((m) => m.name);
  if (!names.length) return "empty link";
  return (link && link.type) === "oneof" ? `one of: ${names.join(", ")}` : names.join(" ⇄ ");
}

// What a row's 🔗 badge says on hover — the rule spelled out from THIS group's point of view.
function describeLinks(node, name) {
  const out = [];
  for (const l of linksFor(node, name)) {
    const others = (l.members || []).filter((m) => m.name !== name);
    let line;
    if (l.type === "oneof") {
      line = `Only one of: ${[name, ...others.map((m) => m.name)].join(", ")} — switching this on ` +
             "switches the others off" + (l.requireOne ? "; one of them is always on." : ".");
    } else {
      const me = (l.members || []).find((m) => m.name === name) || { pol: 1 };
      const flip = (m) => (m.pol < 0) !== (me.pol < 0);
      const opp = others.filter(flip).map((m) => m.name);
      const same = others.filter((m) => !flip(m)).map((m) => m.name);
      const parts = [];
      if (opp.length) parts.push(`switches OFF ${opp.join(", ")}`);
      if (same.length) parts.push(`switches ON ${same.join(", ")}`);
      line = `Switching this on ${parts.join(" and ")} — and switching it off does the reverse.`;
    }
    if (l.enabled === false) line += "  (link disabled)";
    out.push(line);
  }
  return out.join("\n");
}

// Boolean view of a group's state: ALWAYS is on, Bypass/Never are off, MIXED/empty is unknown
// (null). An unknown state never drives a link and is never diffed against.
function onOf(state) {
  if (state === "always") return true;
  if (state === "bypass" || state === "mute") return false;
  return null;
}

// PURE resolver — the heart of the feature, deliberately free of the DOM and of the graph so it can
// be unit-tested. Walks out from the group(s) that just moved (`drivers`) and assigns a target
// state to every group reachable through the links. FIRST ASSIGNMENT WINS and the drivers are
// pinned first, which is what makes a contradictory set (A⇄B, B⇄C, C⇄A) settle with a reported
// conflict instead of oscillating for ever. `stateAt(name)` reads the current state (true/false/
// null) — only "oneof + requireOne" needs it.
// Returns { assign: Map<name, {on, off, link}>, conflicts: [{link, name, want, kept}] }.
function resolveLinks(links, drivers, stateAt) {
  const assign = new Map();
  const conflicts = [];
  const queue = [];
  for (const d of drivers) {
    if (assign.has(d.name)) continue;
    assign.set(d.name, { on: d.on, off: null, link: null, driver: true });
    queue.push(d.name);
  }
  let guard = 0;
  while (queue.length && guard++ < 5000) {
    const name = queue.shift();
    const on = assign.get(name).on;
    for (const link of links) {
      const me = (link.members || []).find((m) => m.name === name);
      if (!me) continue;
      const off = link.off === "never" ? "never" : "bypass";
      const put = (target, want) => {
        const prev = assign.get(target);
        if (prev) {
          if (prev.on !== want) conflicts.push({ link, name: target, want, kept: prev.on });
          return;
        }
        assign.set(target, { on: want, off, link: link.id });
        queue.push(target);
      };
      if (link.type === "oneof") {
        if (on) {
          for (const o of link.members) if (o.name !== name) put(o.name, false);
        } else if (link.requireOne) {
          // The set must never be all-off: bring in the first member that nothing else has claimed.
          const anyOn = link.members.some((m) => {
            const a = assign.get(m.name);
            return a ? a.on : stateAt(m.name) === true;
          });
          if (!anyOn) {
            const pick = link.members.find((m) => m.name !== name && !assign.has(m.name));
            if (pick) put(pick.name, true);
          }
        }
      } else {
        const S = me.pol < 0 ? !on : on;   // the link's own axis, read through this member's sign
        for (const o of link.members) {
          if (o.name === name) continue;
          put(o.name, o.pol < 0 ? !S : S);
        }
      }
    }
  }
  return { assign, conflicts };
}

// ---------------------------------------------------------------- links: the graph-facing part
function entriesNow(node) {
  const cached = node._kbAllEntries;
  return cached && cached.length ? cached : treeEntries();
}

function entryByName(node, name) {
  return entriesNow(node).find((e) => e.name === name) || null;
}

function stateOfName(node, name) {
  const e = entryByName(node, name);
  return e ? onOf(stateOf(e)) : null;
}

// What "off" means for this group: the first enabled link that has an opinion, else Bypass (the
// panel's own long-standing default).
function offForEntry(node, name) {
  const l = linksFor(node, name).find((x) => x.enabled !== false);
  return l && l.off === "never" ? "never" : "bypass";
}

// Compared by NAME, not by object identity: entries are rebuilt from the graph on every poll, and
// links (like the rest of the panel) address a group by its name anyway.
function descendantsOf(node, entry) {
  return entriesNow(node).filter((e) => {
    for (let p = e.parent; p; p = p.parent) if (p.name === entry.name) return true;
    return false;
  });
}

// Switching a parent group off flattens everything inside it — every node in its bounding box gets
// the same mode — which would silently destroy the arrangement of the groups nested in it. So the
// descendants' states are snapshotted on the way down and put back when the parent comes up again.
// The snapshot lives on the node (properties._kbInner) rather than in memory because once the
// parent is off the arrangement can no longer be recovered from the nodes themselves, and a save/
// reload in that state would lose it.
function innerStore(node) {
  node.properties = node.properties || {};
  return (node.properties._kbInner = node.properties._kbInner || {});
}

function clearInner(node) {
  if (node.properties) delete node.properties._kbInner;
}

// Set one group on/off, preserving the arrangement of any groups nested inside it. This is the
// single door every deliberate on/off goes through — the panel switch, the ⋯ menu and the links
// engine — so the behaviour is the same whoever asked. The bulk buttons deliberately do NOT use it
// (see bulkApply): "all on" means all on, inner arrangement included.
function setEntryState(node, entry, on, off) {
  const kids = descendantsOf(node, entry);
  if (kids.length && !on) {
    const snap = {};
    for (const k of kids) snap[k.name] = stateOf(k);
    innerStore(node)[entry.name] = snap;
  }
  applyMode(entry, on ? ALWAYS() : (off === "never" ? MUTE() : BYPASS()));
  if (!on) return;
  const store = (node.properties && node.properties._kbInner) || null;
  const snap = store && store[entry.name];
  if (store) delete store[entry.name];
  if (!snap) return;
  // Shallow first: a nested group's own state is re-applied AFTER the parent's blanket sweep.
  for (const k of kids.slice().sort((a, b) => (a.depth || 0) - (b.depth || 0))) {
    const st = snap[k.name];
    if (st === "bypass") applyMode(k, BYPASS());
    else if (st === "mute") applyMode(k, MUTE());
  }
}

// The states the engine last saw, so the next tick can tell what the user moved.
function snapshotLinks(node) {
  const links = activeLinks(node);
  if (!links.length) { node._kbLinkSnap = null; return; }
  const names = new Set();
  for (const l of links) for (const m of l.members) names.add(m.name);
  const snap = {};
  for (const e of entriesNow(node)) if (names.has(e.name)) snap[e.name] = onOf(stateOf(e));
  node._kbLinkSnap = snap;
}

// Apply a set of drivers through the links. Everything written here happens with the engine marked
// busy and ends with a fresh snapshot, so the poll tick that follows recognises its own work
// instead of mistaking it for a new user action.
function applyLinks(node, drivers) {
  const links = activeLinks(node);
  if (!links.length || !drivers.length) return 0;
  const changed = [];
  node._kbLinksBusy = true;
  try {
    const { assign, conflicts } = resolveLinks(links, drivers, (n) => stateOfName(node, n));
    const pinned = new Set(drivers.map((d) => d.name));
    const targets = [];
    for (const [name, want] of assign) {
      if (pinned.has(name)) continue;
      const e = entryByName(node, name);
      if (e) targets.push({ entry: e, want });   // a renamed/deleted group is simply skipped
    }
    // Parents before children, so a nested group's own rule survives its parent's blanket sweep.
    targets.sort((a, b) => (a.entry.depth || 0) - (b.entry.depth || 0));
    for (const t of targets) {
      const st = stateOf(t.entry);
      if (st === "empty") continue;              // nothing in the group to switch
      if (onOf(st) === t.want.on) continue;      // re-read here: a parent applied a moment ago may
      setEntryState(node, t.entry, t.want.on, t.want.off);   // already have moved this one
      changed.push(`${t.entry.name} ${t.want.on ? "on" : "off"}`);
    }
    node._kbConflicts = new Set(conflicts.map((c) => c.name));
    if (conflicts.length) {
      flash(node, `🔗 contradiction on ${[...node._kbConflicts].join(", ")} — check the links`, true);
    } else if (changed.length) {
      flash(node, `🔗 ${drivers.map((d) => d.name).join(", ")} → ${changed.join(", ")}`);
    }
  } finally {
    node._kbLinksBusy = false;
  }
  snapshotLinks(node);
  if (changed.length) paint(node);
  return changed.length;
}

// Every poll tick: did any linked group move since we last looked? This is what lets a Ctrl+B on
// the canvas, or ComfyUI's own group menu, drive the links exactly like a click in this panel.
function pollLinks(node) {
  if (node._kbLinksBusy || node._kbDragging) return;
  const prev = node._kbLinkSnap;
  snapshotLinks(node);
  if (!prev) return;                    // first look (or just-edited links) — only learn, never fire
  const cur = node._kbLinkSnap || {};
  const drivers = [];
  for (const name of Object.keys(cur)) {
    if (!(name in prev)) continue;      // a group that only just appeared isn't a user action
    if (cur[name] === null || prev[name] === cur[name]) continue;
    drivers.push({ name, on: cur[name] });
  }
  if (drivers.length) applyLinks(node, drivers);
}

// The panel's own on/off path: flip the group, then let the links follow at once rather than on the
// next poll tick.
function toggleEntry(node, entry, on, off) {
  node._kbLinksBusy = true;
  try { setEntryState(node, entry, on, off || offForEntry(node, entry.name)); }
  finally { node._kbLinksBusy = false; }
  applyLinks(node, [{ name: entry.name, on }]);
  refresh(node);
}

// A bulk switch is an explicit "make it so": the engine stays out of it (otherwise "all on" would
// fight every toggle pair the moment it ran) and simply re-learns the states afterwards.
function bulkApply(node, mode) {
  node._kbLinksBusy = true;
  try { for (const e of bulkEntries(node)) applyMode(e, mode); }
  finally { node._kbLinksBusy = false; }
  clearInner(node);
  snapshotLinks(node);
  refresh(node);
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
// HIDDEN groups are left completely alone — a hidden "load the models" group must survive a solo.
function soloEntry(node, entry) {
  const entries = activeEntries(node);
  const snap = new Map();
  for (const e of [...entries, entry]) for (const g of e.groups) for (const n of groupNodes(g)) {
    if (!snap.has(n.id)) snap.set(n.id, n.mode);
  }
  node._kbSolo = { name: entry.name, modes: snap };
  // Solo is an explicit override of the whole board, so the links engine sits it out and re-learns
  // the states afterwards — otherwise a toggle pair would immediately undo the isolation.
  node._kbLinksBusy = true;
  try {
    for (const e of entries) if (e.name !== entry.name) applyMode(e, BYPASS());
    applyMode(entry, ALWAYS());
  } finally { node._kbLinksBusy = false; }
  clearInner(node);
  snapshotLinks(node);
  refresh(node);
  flash(node, `◎ solo “${entry.name}” — rest bypassed`);
}

// Restore the modes captured by the last solo.
function undoSolo(node) {
  const solo = node._kbSolo;
  if (!solo) return;
  const byId = new Map();
  for (const n of (app.graph?._nodes || [])) byId.set(n.id, n);
  node._kbLinksBusy = true;
  try { for (const [id, mode] of solo.modes) { const n = byId.get(id); if (n) n.mode = mode; } }
  finally { node._kbLinksBusy = false; }
  node._kbSolo = null;
  clearInner(node);
  snapshotLinks(node);
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
    { content: "● Always (on)", callback: () => toggleEntry(node, entry, true) },
    { content: "⇄ Bypass",      callback: () => toggleEntry(node, entry, false, "bypass") },
    { content: "✕ Never (mute)", callback: () => toggleEntry(node, entry, false, "never") },
    null,
    { content: "◎ Solo — only this group", callback: () => soloEntry(node, entry) },
  ];
  if (node._kbSolo) {
    items.push({ content: `↩ Undo solo (${node._kbSolo.name})`, callback: () => undoSolo(node) });
  }
  // Hide / unhide this row, plus the reveal toggle when anything is hidden.
  items.push(null);
  if (isHidden(node, entry)) {
    items.push({
      content: "👁 Unhide — keep in the list",
      callback: () => { setHidden(node, entry.name, false); flash(node, `👁 “${entry.name}” unhidden`); },
    });
  } else {
    items.push({
      content: "🚫 Hide from this list",
      callback: () => {
        setHidden(node, entry.name, true);
        flash(node, `🚫 “${entry.name}” hidden — 👁 in the header shows it again`);
      },
    });
  }
  const hn = hiddenCount(node);
  if (hn) {
    items.push({
      content: node._kbShowHidden ? `🙈 Stop showing hidden (${hn})` : `👁 Show hidden groups (${hn})`,
      callback: () => toggleShowHidden(node),
    });
  }
  // Links: the one-click "these two take turns" plus the way into the full editor.
  items.push(null);
  const mine = linksFor(node, entry.name);
  const total = linksOf(node).length;
  items.push({ content: "🔗 Toggle with…", callback: () => pickToggleWith(node, entry, evt) });
  if (total) items.push({ content: "🔗 Add to a link…", callback: () => pickLinkToJoin(node, entry, evt) });
  if (mine.length) {
    items.push({
      content: `🔗 Unlink this group (${mine.length})`,
      callback: () => removeFromLinks(node, entry.name),
    });
  }
  items.push({ content: total ? `🔗 Links… (${total})` : "🔗 Links…", callback: () => openLinksDialog(node) });
  new CM(items, { event: evt, title: entry.name });
}

// One click for the common case: "this group and that one take turns". Builds a two-member polarity
// link (this +, the other −), so flipping either one flips the other.
function pickToggleWith(node, entry, evt) {
  const CM = LG().ContextMenu;
  if (!CM) return;
  const others = entriesNow(node).filter((e) => e.name !== entry.name);
  if (!others.length) { flash(node, "there is no second group to toggle with", true); return; }
  new CM(
    others.map((o) => ({
      content: o.name,
      callback: () => {
        saveLinks(node, [...linksOf(node), {
          id: newLinkId(), name: "", type: "follow", off: "bypass", enabled: true,
          members: [{ name: entry.name, pol: 1 }, { name: o.name, pol: -1 }],
        }]);
        flash(node, `🔗 “${entry.name}” ⇄ “${o.name}” — one on, the other off`);
      },
    })),
    { event: evt, title: `Toggle “${entry.name}” with…` },
  );
}

// Add this group to a link that already exists (as a "+" member — flip the sign in the editor).
function pickLinkToJoin(node, entry, evt) {
  const CM = LG().ContextMenu;
  if (!CM) return;
  const cand = linksOf(node).filter((l) => !(l.members || []).some((m) => m.name === entry.name));
  if (!cand.length) { flash(node, `“${entry.name}” is already in every link`, true); return; }
  new CM(
    cand.map((l) => ({
      content: linkLabel(l),
      callback: () => {
        const list = linksOf(node).map((x) =>
          x === l ? { ...x, members: [...(x.members || []), { name: entry.name, pol: 1 }] } : x);
        saveLinks(node, list);
        flash(node, `🔗 “${entry.name}” added to ${linkLabel(l)}`);
      },
    })),
    { event: evt, title: `Add “${entry.name}” to…` },
  );
}

// Drop a group from every link it is in. A link left with a single member has nothing to propagate
// to, so it goes as well.
function removeFromLinks(node, name) {
  const list = linksOf(node)
    .map((l) => ({ ...l, members: (l.members || []).filter((m) => m.name !== name) }))
    .filter((l) => l.members.length > 1);
  saveLinks(node, list);
  flash(node, `🔗 “${name}” unlinked`);
}

// ------------------------------------------------------------------------------ links editor
// A plain DOM overlay (window.prompt doesn't exist in the desktop app, so it is never an option).
// Every edit writes straight through to node.properties and re-renders — there is no OK/Cancel,
// which keeps the panel and the dialog from ever disagreeing about what the links are.
const DLG_BTN = "cursor:pointer;border:1px solid #3a3a44;background:#2a2a32;color:#ccc;" +
                "border-radius:4px;font-size:11px;padding:2px 8px;";
const DLG_IN = "border:1px solid #3a3a44;background:#1e1e24;color:#ddd;border-radius:4px;" +
               "font-size:12px;padding:2px 6px;outline:none;";

function mkEl(tag, cssText, text) {
  const e = document.createElement(tag);
  if (cssText) e.style.cssText = cssText;
  if (text != null) e.textContent = text;
  return e;
}

function openLinksDialog(node) {
  if (node._kbDlg) return;
  const overlay = mkEl("div",
    "position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:10002;display:flex;" +
    "align-items:center;justify-content:center;");
  const box = mkEl("div",
    "background:#222;color:#eee;border:1px solid #444;border-radius:8px;padding:16px;" +
    "min-width:560px;max-width:820px;max-height:84vh;overflow:auto;font:13px sans-serif;" +
    "box-shadow:0 8px 32px rgba(0,0,0,0.5);");
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  const close = () => {
    overlay.remove();
    document.removeEventListener("keydown", onKey, true);
    node._kbDlg = null;
  };
  document.addEventListener("keydown", onKey, true);
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
  // Keep the canvas from zooming/panning under the dialog.
  overlay.addEventListener("wheel", (e) => e.stopPropagation());
  overlay.addEventListener("pointerdown", (e) => e.stopPropagation());
  overlay.appendChild(box);
  document.body.appendChild(overlay);
  node._kbDlg = { close, box };
  renderDialog(node, box);
}

function renderDialog(node, box) {
  box.innerHTML = "";
  const links = linksOf(node);
  const names = entriesNow(node).map((e) => e.name);

  box.appendChild(mkEl("div", "font-size:15px;font-weight:600;margin-bottom:3px", "🔗 Group links"));
  box.appendChild(mkEl("div", "opacity:.6;font-size:11px;line-height:1.5;margin-bottom:12px",
    "Tie groups together so switching one switches the others. A link fires whichever way the " +
    "change came — this panel, Ctrl+B on the canvas, or ComfyUI's own group menu."));

  if (node._kbLinksPaused) {
    const warn = mkEl("div",
      "background:#4a2a2a;border:1px solid #7a3a3a;border-radius:5px;padding:6px 8px;" +
      "margin-bottom:10px;font-size:11px;display:flex;align-items:center;gap:8px;");
    warn.appendChild(mkEl("span", "flex:1", "⏸ Links are paused for this session — nothing follows anything."));
    const b = mkEl("button", DLG_BTN, "resume");
    b.onclick = () => {
      node._kbLinksPaused = false;
      node._kbLinkSnap = null;
      paintLinkBtn(node);
      renderDialog(node, box);
    };
    warn.appendChild(b);
    box.appendChild(warn);
  }

  if (!links.length) {
    box.appendChild(mkEl("div", "opacity:.6;font-style:italic;padding:6px 0 2px",
      "No links yet — add one below, or use “🔗 Toggle with…” in a row's ⋯ menu."));
  }
  links.forEach((link) => box.appendChild(linkCard(node, box, link, names)));

  const foot = mkEl("div", "display:flex;gap:8px;margin-top:14px;align-items:center");
  const add = mkEl("button", DLG_BTN, "+ New link");
  add.onclick = () => {
    saveLinks(node, [...linksOf(node),
      { id: newLinkId(), name: "", type: "follow", off: "bypass", enabled: true, members: [] }]);
    renderDialog(node, box);
  };
  const done = mkEl("button", DLG_BTN + "margin-left:auto", "Close");
  done.onclick = () => node._kbDlg && node._kbDlg.close();
  foot.append(add, done);
  box.appendChild(foot);
}

function linkCard(node, box, link, names) {
  const card = mkEl("div",
    "border:1px solid #3a3a44;border-radius:6px;padding:8px 10px;margin-bottom:10px;" +
    `border-left:3px solid ${linkColor(node, link)};` +
    (link.enabled === false ? "opacity:.55;" : ""));

  // Every control edits a fresh copy of the list and saves it, so the node's properties are the one
  // source of truth and the panel rebuilds from them on the spot.
  const write = (fn) => {
    const idx = linksOf(node).indexOf(link);
    if (idx < 0) return;                      // the card is stale — a re-render is already coming
    const list = linksOf(node).slice();
    const copy = { ...link, members: [...(link.members || [])] };
    list[idx] = copy;
    fn(copy);
    saveLinks(node, list);
    renderDialog(node, box);
  };
  const type = link.type === "oneof" ? "oneof" : "follow";

  const head = mkEl("div", "display:flex;align-items:center;gap:6px;margin-bottom:7px");
  const nameIn = mkEl("input", DLG_IN + "flex:1;min-width:0");
  nameIn.value = link.name || "";
  nameIn.placeholder = linkLabel({ ...link, name: "" });
  nameIn.title = "A name for this link (optional)";
  nameIn.addEventListener("keydown", (e) => e.stopPropagation());
  nameIn.addEventListener("change", () => write((t) => { t.name = nameIn.value.trim(); }));

  const mkSel = (opts, cur, tip, apply) => {
    const s = mkEl("select", DLG_IN);
    s.title = tip;
    for (const [v, label] of opts) {
      const o = mkEl("option", "", label);
      o.value = v;
      if (v === cur) o.selected = true;
      s.appendChild(o);
    }
    s.onchange = () => apply(s.value);
    return s;
  };
  const typeSel = mkSel(
    [["follow", "follow / opposite"], ["oneof", "only one of"]], type,
    "follow: each group is + (same) or − (opposite).  only one of: switching one on switches the rest off.",
    (v) => write((t) => { t.type = v; }));
  const offSel = mkSel(
    [["bypass", "off = Bypass"], ["never", "off = Never"]], link.off === "never" ? "never" : "bypass",
    "Which off this link uses when it switches a group off.",
    (v) => write((t) => { t.off = v; }));

  const en = mkEl("input", "width:14px;height:14px;");
  en.type = "checkbox";
  en.checked = link.enabled !== false;
  en.title = "Enable this link";
  en.onchange = () => write((t) => { t.enabled = en.checked; });

  const del = mkEl("button", DLG_BTN + "color:#e08a8a", "🗑");
  del.title = "Delete this link";
  del.onclick = () => { saveLinks(node, linksOf(node).filter((x) => x !== link)); renderDialog(node, box); };
  head.append(nameIn, typeSel, offSel, en, del);
  card.appendChild(head);

  // Members as chips. On a "follow" link each chip carries its +/− sign, which is the whole rule.
  const mem = mkEl("div", "display:flex;flex-wrap:wrap;gap:6px;align-items:center");
  const known = new Set(names);
  for (const m of (link.members || [])) {
    const stale = !known.has(m.name);
    const chip = mkEl("span",
      "display:inline-flex;align-items:center;gap:5px;padding:2px 7px;border-radius:11px;" +
      "background:#2a2a32;border:1px solid " + (stale ? "#b45309" : "#3a3a44") + ";font-size:11px;");
    if (stale) chip.title = "no group with this name in the workflow right now";
    if (type === "follow") {
      const pol = mkEl("button",
        "cursor:pointer;border:none;border-radius:8px;width:18px;height:16px;font-size:11px;" +
        "font-weight:700;color:#fff;padding:0;line-height:1;" +
        (m.pol < 0 ? "background:#8a4a1f;" : "background:#2e7d32;"), m.pol < 0 ? "−" : "+");
      pol.title = m.pol < 0
        ? "opposite — this group goes the other way from whichever member you flip"
        : "same — this group follows whichever member you flip";
      pol.onclick = () => write((t) => {
        const mm = t.members.find((x) => x.name === m.name);
        if (mm) t.members = t.members.map((x) => (x === mm ? { ...x, pol: mm.pol < 0 ? 1 : -1 } : x));
      });
      chip.appendChild(pol);
    }
    chip.appendChild(mkEl("span", stale ? "text-decoration:line-through" : "", m.name));
    const x = mkEl("span", "cursor:pointer;color:#8a8a92;padding-left:1px", "✕");
    x.title = "Remove from this link";
    x.onclick = () => write((t) => { t.members = t.members.filter((y) => y.name !== m.name); });
    chip.appendChild(x);
    mem.appendChild(chip);
  }
  const avail = names.filter((n) => !(link.members || []).some((m) => m.name === n));
  if (avail.length) {
    const addSel = mkEl("select", DLG_IN);
    const ph = mkEl("option", "", "+ add group");
    ph.value = "";
    addSel.appendChild(ph);
    for (const n of avail) { const o = mkEl("option", "", n); o.value = n; addSel.appendChild(o); }
    addSel.onchange = () => {
      const v = addSel.value;
      if (v) write((t) => { t.members = [...t.members, { name: v, pol: 1 }]; });
    };
    mem.appendChild(addSel);
  }
  card.appendChild(mem);

  if (type === "oneof") {
    const row = mkEl("label",
      "display:flex;align-items:center;gap:6px;margin-top:7px;font-size:11px;opacity:.85;cursor:pointer");
    const cb = mkEl("input", "width:14px;height:14px;");
    cb.type = "checkbox";
    cb.checked = !!link.requireOne;
    cb.onchange = () => write((t) => { t.requireOne = cb.checked; });
    row.append(cb, mkEl("span", "", "keep one always on — switching the last one off brings the next in"));
    card.appendChild(row);
  }
  card.appendChild(mkEl("div", "margin-top:7px;font-size:11px;opacity:.6;line-height:1.5", explainLink(link)));
  return card;
}

// The rule in words, under each card — the fastest way to spot a link wired the wrong way round.
function explainLink(link) {
  const ms = link.members || [];
  if (ms.length < 2) return "Add at least two groups — a link with fewer does nothing.";
  const q = (m) => `“${m.name}”`;
  if (link.type === "oneof") {
    return `Switching any of ${ms.map(q).join(", ")} on switches the others off` +
      (link.requireOne ? "; one of them is always on." : "; switching one off leaves the rest alone.");
  }
  const plus = ms.filter((m) => m.pol >= 0), minus = ms.filter((m) => m.pol < 0);
  if (!minus.length || !plus.length) return `${ms.map(q).join(", ")} always share one state (mirror).`;
  return `${plus.map(q).join(", ")} on ⇒ ${minus.map(q).join(", ")} off, and off ⇒ on. ` +
         "Any member can be the one you flip.";
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
  const q = node._kbFilter || "";
  let total = 0, shown = 0;
  for (const row of els.list.children) {
    const e = row._kbEntry;
    if (!e) continue; // e.g. the drag drop-line — not a row
    total++;
    const ok = !q || e.name.toLowerCase().includes(q);
    // Show with "grid" (the row's real display), NOT "" — an empty string deletes the inline
    // display and the row falls back to block, which would kill the grid column alignment.
    row.style.display = ok ? "grid" : "none";
    if (ok) shown++;
  }
  // Count reads "<shown>/<total>" under a filter, and appends "+N🚫" for rows kept out of sight.
  const hn = hiddenCount(node);
  const extra = !node._kbShowHidden && hn ? ` +${hn}🚫` : "";
  els.count.textContent = total ? `${q ? `${shown}/${total}` : total}${extra}` : extra.trim();
  if (!total) {
    els.empty.textContent = hn
      ? `All ${hn} group(s) are hidden — click 👁 in the header to show them.`
      : "No named groups in this workflow.";
    els.empty.style.display = "";
  } else if (shown === 0) { els.empty.textContent = "No groups match the filter."; els.empty.style.display = ""; }
  else { els.empty.style.display = "none"; }
  paintEye(node);
  paintLinkBtn(node);
}

// Keep the header 🔗 button in sync: how many links exist, and whether they are paused.
function paintLinkBtn(node) {
  const els = node._kbEls;
  if (!els || !els.link) return;
  const n = linksOf(node).length, paused = !!node._kbLinksPaused;
  els.link.textContent = n ? (paused ? `🔗⏸ ${n}` : `🔗 ${n}`) : "🔗";
  els.link.style.background = paused ? "#4a2a2a" : (n ? "#2f3a4a" : "#2a2a32");
  els.link.style.color = n ? "#ddd" : "#6d6d76";
  els.link.title = (n
    ? `${n} link(s) between groups${paused ? " — PAUSED, nothing follows anything" : ""}. ` +
      "Click to edit them."
    : "No links. Click to tie groups together — switching one group on can switch another off " +
      "(also on a row's ⋯ menu → “Toggle with…”).") +
    "  Right-click: pause/resume every link for this session.";
}

// Keep the header 👁 button in sync: it shows how many groups are hidden and whether they're
// currently revealed.
function paintEye(node) {
  const els = node._kbEls;
  if (!els || !els.eye) return;
  const n = hiddenCount(node), on = !!node._kbShowHidden;
  els.eye.textContent = n ? (on ? `👁 ${n}` : `🚫 ${n}`) : "👁";
  els.eye.style.background = on ? "#3b3b5c" : "#2a2a32";
  els.eye.style.color = n ? "#ddd" : "#6d6d76";
  els.eye.title = n
    ? (on
        ? `Showing ${n} hidden group(s) as dimmed rows — click to put them away again. ` +
          "Right-click: unhide all."
        : `${n} group(s) hidden from this list — click to reveal them (dimmed). Right-click: unhide all.`)
    : "No hidden groups. Use a row's ⋯ menu → “Hide from this list” to keep set-and-forget " +
      "groups (loaders, VAE, …) out of the way. Hidden groups are also skipped by all on/off/✕ and Solo.";
}

// Which entries the header bulk buttons act on: never the hidden ones, and only the filtered
// (visible) set when a filter is active.
function bulkEntries(node) {
  const all = activeEntries(node);
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
  // 👁 reveals the hidden rows (dimmed) so they can be used or unhidden; right-click unhides all.
  const eye = mkBtn("👁", "", () => toggleShowHidden(node));
  eye.oncontextmenu = (e) => { e.preventDefault(); unhideAll(node); };
  // 🔗 opens the links editor; right-click pauses every link for this session (the escape hatch
  // when you want to set the board by hand without anything following anything).
  const link = mkBtn("🔗", "", () => openLinksDialog(node));
  link.oncontextmenu = (e) => {
    e.preventDefault();
    node._kbLinksPaused = !node._kbLinksPaused;
    node._kbLinkSnap = null;
    paintLinkBtn(node);
    flash(node, node._kbLinksPaused ? "🔗 links paused for this session" : "🔗 links active again");
  };
  // Bulk buttons act on the filtered set when a filter is active (else everything), and never on
  // hidden groups — that's the point of hiding a "load the base models" group. They also bypass the
  // links engine: "all on" means all on, whatever the links would otherwise have to say.
  const allOn = mkBtn("all on", "Set matching groups to Always (all when no filter; hidden groups untouched; links don't fire)",
    () => bulkApply(node, ALWAYS()));
  const allOff = mkBtn("all off", "Bypass matching groups (all when no filter; hidden groups untouched; links don't fire)",
    () => bulkApply(node, BYPASS()));
  const allNever = mkBtn("all ✕", "Set matching groups to Never/mute (all when no filter; hidden groups untouched; links don't fire)",
    () => bulkApply(node, MUTE()));
  header.append(title, count, sortBtn, eye, link, allOn, allOff, allNever);

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
  node._kbEls = { root, list, count, empty, status, filter, eye, link };
  paintEye(node);
  paintLinkBtn(node);
  return root;
}

// Rebuild the rows from scratch (structure changed).
function rebuild(node) {
  const els = node._kbEls;
  if (!els) return;
  const all = collectEntries();
  const tree = buildTree(all);
  const ordered = orderEntries(node, all, tree);
  node._kbAllEntries = ordered;   // every entry — what activeEntries()/hiddenCount() work from
  // Hidden entries get no row unless the 👁 toggle is on (then they render dimmed).
  const entries = ordered.filter((e) => node._kbShowHidden || !isHidden(node, e));
  const shownSet = new Set(entries);
  node._kbEntries = entries;
  els.list.innerHTML = "";

  for (const entry of entries) {
    const hidden = isHidden(node, entry);
    // A hidden parent leaves no row, so indentation and the drag sibling-grouping below follow the
    // nearest VISIBLE ancestor: children of a hidden group are promoted a level instead of being
    // indented under a row that isn't there.
    let vparent = entry.parent;
    while (vparent && !shownSet.has(vparent)) vparent = vparent.parent;
    let depth = 0;
    for (let p = vparent; p; p = p.parent) if (shownSet.has(p)) depth++;

    // Fixed grid columns keep the controls in the same place on every row, whatever the name
    // length: [grip][dot][name grows][🔗][▶][switch][⋯]. NOTE: applyFilter() must re-show rows with
    // display:"grid" (not ""), or it wipes this inline display and the row falls back to block.
    const row = document.createElement("div");
    row.style.cssText =
      "display:grid;grid-template-columns:auto auto minmax(0,1fr) auto auto auto auto;" +
      "align-items:center;column-gap:6px;padding:3px 6px;border-radius:5px;" +
      `background:#00000022;margin-left:${depth * 14}px;` +
      (hidden ? "opacity:.5;" : "");

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
    if (hidden) name.textContent += "  🚫";
    name.title = entry.name + (entry.groups.length > 1 ? ` (${entry.groups.length} groups)` : "") +
      (hidden ? " — HIDDEN (skipped by all on/off/✕ and Solo; ⋯ → Unhide to keep it)" : "") +
      " — click to focus on canvas";
    name.style.cssText =
      "min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#ddd;cursor:pointer;";
    name.onclick = (e) => { e.preventDefault(); focusEntry(entry); };

    // 🔗 this group is tied to others — the tooltip spells out what follows what. The column is
    // always there (just invisible when unlinked) so the rows stay aligned.
    const ls = linksFor(node, entry.name);
    const badge = document.createElement("span");
    badge.textContent = "🔗";
    badge.style.cssText =
      "font-size:9px;line-height:1;cursor:pointer;padding:2px 3px;border-radius:3px;" +
      (ls.length ? `background:${linkColor(node, ls[0])}44;` : "visibility:hidden;");
    badge.title = ls.length ? describeLinks(node, entry.name) : "";
    badge.onclick = (e) => { e.preventDefault(); openLinksDialog(node); };

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
      toggleEntry(node, entry, stateOf(entry) !== "always");
    };
    // Right-click the switch → the full per-group menu (Run / Always / Bypass / Never).
    sw.oncontextmenu = (e) => { e.preventDefault(); openRowMenu(node, entry, e); };

    // ⋯ menu: the discoverable home for Always / Bypass / Never (+ Run).
    const menu = document.createElement("button");
    menu.textContent = "⋯";
    menu.title = "More: Run · Focus · Always / Bypass / Never · Solo · Hide from list";
    menu.style.cssText =
      "cursor:pointer;border:1px solid #3a3a44;background:#2a2a32;color:#bbb;" +
      "border-radius:4px;width:20px;height:20px;font-size:12px;line-height:1;padding:0;";
    menu.onclick = (e) => { e.preventDefault(); openRowMenu(node, entry, e); };

    row._kbEntry = entry;
    row._kbHidden = hidden;
    row._kbDepth = depth;
    row._kbParentKey = vparent ? vparent.name : "";
    row._kbSwitch = sw;
    row._kbBadge = ls.length ? badge : null;
    // Right-click anywhere on the row also opens the menu (except the switch, handled above).
    row.oncontextmenu = (e) => { e.preventDefault(); openRowMenu(node, entry, e); };
    row.append(grip, dot, name, badge, run, sw, menu);
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
    // A group two links can't agree on is ringed in red — the engine kept the first answer.
    if (row._kbBadge) {
      const bad = node._kbConflicts && node._kbConflicts.has(entry.name);
      row._kbBadge.style.outline = bad ? "1px solid #e06a6a" : "none";
    }
  }
}

// Rebuild if the layout changed, otherwise just repaint the switches; then let the links follow
// anything that moved since the last tick (a Ctrl+B on the canvas, the core group menu, …).
function refresh(node) {
  if (node._kbDragging) return; // don't rebuild/repaint the list out from under a drag
  const sig = signature();
  if (sig !== node._kbSig) { node._kbSig = sig; rebuild(node); }
  else paint(node);
  pollLinks(node);
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
      // The links arrive with the same properties, and the modes they describe are already baked
      // into the loaded nodes — so the engine must LEARN this state, not treat it as a change.
      this._kbLinkSnap = null;
      if (this.graph) refresh(this);
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this._kbTimer) { clearInterval(this._kbTimer); this._kbTimer = null; }
      if (this._kbFlashTimer) { clearTimeout(this._kbFlashTimer); this._kbFlashTimer = null; }
      if (this._kbDlg) this._kbDlg.close();   // the editor is on document.body — take it with us
      return onRemoved?.apply(this, arguments);
    };
  },
});
