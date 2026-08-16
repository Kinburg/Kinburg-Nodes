import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { chatNodes, chatSnapshot, registerRefHolder } from "./chat_llm.js";

// Morpheus Dream Board 🌙 — pick a conversation apart into shots.
//   • ⟳ Update History snapshots the chosen chat node CLIENT-SIDE. No graph link on purpose: a link
//     would sit below the chat's blocked output, so rendering a video would need the chat to run
//     first. It also means this node works when you run it by itself, which is how the author works.
//   • The pictures ARE the shot boundaries (picture i ends shot i and starts shot i+1), so the shot
//     list is DERIVED, live, from what is ticked — there is nothing to "create" and no state that
//     can contradict itself. planBoard() below MUST match plan_board() in morpheus/dream_board.py.
//   • Each message has a tick (does its text go into that shot's direction) and each picture has one
//     (is it a boundary). They are independent: a persona's bubble is usually text you don't want
//     and a picture you do.

const CLASS = "KinburgDreamBoard";
const AUTO = "(the only chat node)";
const VIEW_MIN = 140, SHOTS_MIN = 44, GAP = 6;
const DEFAULT_DURATION = 5.17;
const LINKS = ["continue", "cut"];

const instances = new Set();

// Every picture any board on the canvas still names, so the chat's cleanup spares those files. A
// board holds a SNAPSHOT, so without this a picture removed from the chat after Update History
// takes its file with it and the board's shot list points at nothing — which is exactly what
// happened the first time this node met a real conversation.
function heldRefs() {
  const out = [];
  for (const n of instances) {
    for (const m of (n._dbState?.msgs || [])) {
      for (const a of (m.img || [])) if (a?.name) out.push(a.name);
    }
  }
  return out;
}
registerRefHolder(heldRefs);

// ── state ───────────────────────────────────────────────────────────────────────────────────
// One JSON string, carried by the board's own DOM widget — the same trick the chat uses, and for
// the same reason: every entry in node.widgets costs a 24px row whether or not it draws anything.

function ST(node) {
  if (!node._dbState || typeof node._dbState !== "object") {
    node._dbState = { v: 1, src: "", msgs: [], skip: [], noimg: [], breaks: [], dur: [], lnk: [] };
  }
  return node._dbState;
}
const stateJSON = (node) => JSON.stringify(ST(node));

function loadState(node, raw) {
  let s = null;
  try { s = (typeof raw === "string" && raw) ? JSON.parse(raw) : raw; } catch (e) { s = null; }
  if (!s || typeof s !== "object") return;
  const st = ST(node);
  const arr = (v) => (Array.isArray(v) ? v : []);
  st.src = typeof s.src === "string" ? s.src : "";
  st.msgs = arr(s.msgs).filter((m) => m && typeof m === "object");
  st.skip = arr(s.skip); st.noimg = arr(s.noimg); st.breaks = arr(s.breaks);
  st.dur = arr(s.dur); st.lnk = arr(s.lnk);
}

// Names whose thumbnail failed to load, i.e. whose file has gone. Kept on the node and reset only
// by a fresh pull, so a redraw doesn't forget what the browser already told us.
const gone = (node) => (node._dbGone || (node._dbGone = new Set()));

const has = (list, v) => (list || []).some((x) => x === v);
function toggle(list, v) {
  const i = (list || []).findIndex((x) => x === v);
  if (i >= 0) list.splice(i, 1); else list.push(v);
  return list;
}

// ── the rule, mirrored from python ──────────────────────────────────────────────────────────
// MUST match plan_board() in morpheus/dream_board.py — the panel would otherwise promise a
// storyboard the backend doesn't build. Same fixtures test both.
function planBoard(msgs, skip, noimg, breaks) {
  const dead = new Set((noimg || []).map(String));
  const out = new Set((skip || []).map(Number));
  const brk = new Set((breaks || []).map(Number));

  const bounds = [];
  msgs.forEach((m, i) => {
    for (const a of (m.img || [])) {
      if (a && a.name && !dead.has(String(a.name))) bounds.push([i, a]);
    }
  });

  const line = (i) => textOf(msgs[i]);
  const kept = (lo, hi) => {
    const r = [];
    for (let i = lo; i < hi; i++) if (!out.has(i) && line(i)) r.push(i);
    return r;
  };

  const shots = [], notes = [];
  for (let k = 0; k < Math.max(0, bounds.length - 1); k++) {
    const lo = k === 0 ? 0 : bounds[k][0];
    shots.push({ msgs: kept(lo, bounds[k + 1][0]),
                 start: bounds[k][1], end: bounds[k + 1][1], tail: false });
  }

  const tailFrom = bounds.length >= 2 ? bounds[bounds.length - 1][0] : 0;
  const tail = kept(tailFrom, msgs.length);
  if (tail.length) {
    const cuts = new Set([...brk].filter((b) => b > tail[0] && b <= tail[tail.length - 1]));
    let group = [];
    for (const i of tail) {
      if (cuts.has(i) && group.length) {
        shots.push({ msgs: group, start: null, end: null, tail: true });
        group = [];
      }
      group.push(i);
    }
    shots.push({ msgs: group, start: null, end: null, tail: true });
  }

  const ignored = [...brk].filter((b) => !(tail.length && b > tail[0] && b <= tail[tail.length - 1]))
    .sort((a, b) => a - b);
  if (ignored.length) notes.push("break inside a keyframed span, ignored: " + ignored.map((b) => "#" + b).join(", "));
  return { shots, bounds: bounds.map(([, a]) => a), notes, tailFrom, tail };
}

const speaker = (m) => (m.r === "u" ? "User" : (m.p || "Assistant"));
const textOf = (m) => String((m && m.t) || "").replace(/\s+/g, " ").trim();
const nth = (list, i, dflt) => {
  const v = (list || []).filter((x) => x !== null && x !== "" && x !== undefined);
  if (!v.length) return dflt;
  return i < v.length ? v[i] : v[v.length - 1];
};

const attUrl = (a) => {
  const q = "/view?filename=" + encodeURIComponent(a?.name || "")
    + "&subfolder=" + encodeURIComponent(a?.subfolder || "")
    + "&type=" + encodeURIComponent(a?.type || "input");
  return api.apiURL ? api.apiURL(q) : q;
};

// ── snapshot ────────────────────────────────────────────────────────────────────────────────

const label = (n) => "#" + n.id + " " + (n.title || "Local LLM Chat");
const idOf = (v) => (/^#(\d+)\b/.exec(String(v || "")) || [])[1] || null;

function targetChat(node) {
  const id = idOf(node.properties?._dbTarget);
  const all = chatNodes();
  if (id) return all.find((n) => String(n.id) === id) || null;
  return all.length === 1 ? all[0] : null;
}

function updateHistory(node) {
  const chat = targetChat(node);
  if (!chat) {
    node._dbNote = chatNodes().length ? "several chat nodes — pick one above" : "no chat node here";
    return render(node);
  }
  const snap = chatSnapshot(chat);
  const st = ST(node);
  const grew = st.msgs.length;
  gone(node).clear();               // a fresh pull re-asks the browser about every picture
  st.src = String(chat.id);
  st.msgs = snap.msgs;
  // Picks survive a re-pull: the chat only ever grows at the end, so existing indices still point
  // at the same messages. Newly arrived turns of a PRIVATE persona start unticked — that is the
  // "camera" pattern, whose whole job is writing image prompts rather than story.
  //
  // By persona, NOT by role. Your own messages are tagged with whoever was selected, so "make it
  // tighter, more bokeh" typed at a camera carries the camera's name and is no more story than its
  // reply is — while your dialogue with the character carries the character's name and stays, which
  // it must, or the character ends up talking to itself. This mirrors the chat's own privacy rule:
  // _private_out withholds a reply AND the instruction that produced it, for the same reason.
  const priv = new Set(snap.private || []);
  for (let i = grew; i < st.msgs.length; i++) {
    if (priv.has(st.msgs[i].p) && !has(st.skip, i)) st.skip.push(i);
  }
  st.noimg = st.noimg.filter((nm) => st.msgs.some((m) => (m.img || []).some((a) => a.name === nm)));
  node._dbNote = st.msgs.length ? "" : "that chat is empty";
  render(node);
  node.setDirtyCanvas?.(true, true);
}

// ── styles ──────────────────────────────────────────────────────────────────────────────────

function injectStyle() {
  if (document.getElementById("kb-board-style")) return;
  const s = document.createElement("style");
  s.id = "kb-board-style";
  s.textContent = `
  .kb-db-hide{display:none !important;}
  .kb-db-wrap{position:relative;display:flex;flex-direction:column;gap:6px;width:100%;height:100%;box-sizing:border-box;overflow:hidden;font:12px/1.45 -apple-system,Segoe UI,sans-serif;}
  .kb-db-box{flex:1 1 auto;position:relative;min-height:0;overflow:hidden;}
  .kb-db-view{position:absolute;inset:0;overflow-y:auto;padding:4px;box-sizing:border-box;background:#181818;border:1px solid #2b2b2b;border-radius:6px;}
  .kb-db-row{display:flex;align-items:flex-start;gap:6px;padding:2px 3px;border-radius:4px;}
  .kb-db-row:hover{background:#22222a;}
  .kb-db-row.off .kb-db-txt{opacity:.38;text-decoration:line-through;}
  .kb-db-tick{flex:0 0 auto;margin:2px 0 0;cursor:pointer;}
  .kb-db-who{flex:0 0 auto;font-size:10px;opacity:.6;min-width:52px;max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .kb-db-txt{flex:1 1 auto;min-width:0;color:#dcdce2;word-break:break-word;}
  .kb-db-pics{display:flex;flex-wrap:wrap;gap:3px;margin:3px 0 1px;}
  .kb-db-pic{position:relative;width:46px;height:46px;border:2px solid #4a86c4;border-radius:5px;overflow:hidden;background:#101014;cursor:pointer;}
  .kb-db-pic img{display:block;width:100%;height:100%;object-fit:cover;}
  .kb-db-pic.off{border-color:#3a3a44;}
  .kb-db-pic.off img{opacity:.3;}
  .kb-db-pic.gone{border-color:#c4564a;background:#2a1414;}
  .kb-db-pic.gone img{opacity:.12;}
  .kb-db-pic.gone .kb-db-pin{background:#c4564ad0;color:#fff;}
  .kb-db-pin{position:absolute;left:0;bottom:0;background:#000000b0;color:#cfe3f7;font:9px/1 sans-serif;padding:2px 3px;border-radius:0 4px 0 0;}
  .kb-db-brk{display:flex;align-items:center;gap:6px;margin:3px 0;cursor:pointer;opacity:.32;}
  .kb-db-brk:hover{opacity:.85;}
  .kb-db-brk i{flex:1 1 auto;height:0;border-top:1px dashed #6a6a78;}
  .kb-db-brk b{flex:0 0 auto;font:9px/1.3 sans-serif;font-weight:400;color:#9a9aa8;}
  .kb-db-brk.on{opacity:1;}
  .kb-db-brk.on i{border-top:1px dashed #c9a227;}
  .kb-db-brk.on b{color:#e6cf8a;}
  .kb-db-brk.dead{cursor:default;}
  .kb-db-shots{flex:0 0 auto;max-height:40%;overflow-y:auto;background:#15151a;border:1px solid #2b2b2b;border-radius:6px;padding:3px 4px;}
  .kb-db-shot{display:flex;align-items:center;gap:5px;padding:2px 2px;font-size:11px;}
  .kb-db-shot b{flex:0 0 auto;width:16px;color:#8fb0c8;font-weight:600;}
  .kb-db-kf{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;opacity:.75;}
  .kb-db-num{flex:0 0 auto;width:46px;background:#1a1a1a;color:#e6e6ea;border:1px solid #4a4a52;border-radius:4px;padding:1px 4px;font:11px/1.3 inherit;text-align:right;}
  .kb-db-sel{flex:0 0 auto;background:#1a1a1a;color:#e6e6ea;border:1px solid #4a4a52;border-radius:4px;padding:1px 2px;font:10px/1.3 inherit;}
  .kb-db-msgs{flex:0 0 auto;font-size:10px;opacity:.5;min-width:34px;text-align:right;}
  .kb-db-note{flex:0 0 auto;font-size:10px;color:#d7a55a;padding:0 2px;}
  .kb-db-empty{margin:auto;color:#6d6d75;font-style:italic;text-align:center;padding:16px 8px;}
  `;
  document.head.appendChild(s);
}

// ── render ──────────────────────────────────────────────────────────────────────────────────

function tickEl(on, title, onChange) {
  const c = document.createElement("input");
  c.type = "checkbox";
  c.className = "kb-db-tick";
  c.checked = on;
  c.title = title;
  c.addEventListener("pointerdown", (e) => e.stopPropagation());
  c.addEventListener("change", (e) => { e.stopPropagation(); onChange(c.checked); });
  return c;
}

function breakEl(node, i, on, live) {
  const b = document.createElement("div");
  b.className = "kb-db-brk" + (on ? " on" : "") + (live ? "" : " dead");
  const l = document.createElement("i"), r = document.createElement("i");
  const t = document.createElement("b");
  t.textContent = on ? "shot break" : (live ? "+ break" : "break needs a keyframe here");
  b.append(l, t, r);
  b.title = live
    ? "Start a new shot here. Only the stretch after the last picture can be cut this way."
    : "A boundary inside a keyframed span would have no keyframe, which Morpheus can't express — "
      + "put another picture in the chat instead.";
  if (live) {
    b.addEventListener("pointerdown", (e) => e.stopPropagation());
    b.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation();
      toggle(ST(node).breaks, i);
      render(node);
      node.setDirtyCanvas?.(true, true);
    });
  }
  return b;
}

function render(node) {
  const view = node._dbView;
  if (!view) return;
  const st = ST(node);
  const keep = view.scrollTop;
  view.innerHTML = "";

  if (!st.msgs.length) {
    const e = document.createElement("div");
    e.className = "kb-db-empty";
    e.textContent = node._dbNote || "(press ⟳ Update History to pull a conversation in)";
    view.appendChild(e);
    renderShots(node);
    return;
  }

  const plan = planBoard(st.msgs, st.skip, st.noimg, st.breaks);
  const tailSet = new Set(plan.tail);

  st.msgs.forEach((m, i) => {
    // A break can only live where a boundary needs no keyframe: after the last picture.
    if (i > 0 && tailSet.has(i)) {
      const live = plan.tail.length > 0 && i > plan.tail[0];
      const on = has(st.breaks, i);
      if (live || on) view.appendChild(breakEl(node, i, on, live));
    }

    const row = document.createElement("div");
    const off = has(st.skip, i);
    row.className = "kb-db-row" + (off ? " off" : "");
    row.appendChild(tickEl(!off, "Use this message as direction for its shot", () => {
      toggle(st.skip, i);
      render(node);
      node.setDirtyCanvas?.(true, true);
    }));
    const who = document.createElement("span");
    who.className = "kb-db-who";
    who.textContent = speaker(m);
    who.title = speaker(m);
    row.appendChild(who);

    const right = document.createElement("div");
    right.style.cssText = "flex:1 1 auto;min-width:0;";
    const txt = document.createElement("div");
    txt.className = "kb-db-txt";
    txt.textContent = textOf(m) || "—";
    right.appendChild(txt);

    if ((m.img || []).length) {
      const pics = document.createElement("div");
      pics.className = "kb-db-pics";
      for (const a of m.img) {
        if (!a?.name) continue;
        const used = !has(st.noimg, a.name);
        const lost = gone(node).has(a.name);
        const cell = document.createElement("div");
        cell.className = "kb-db-pic" + (used ? "" : " off") + (lost ? " gone" : "");
        const img = document.createElement("img");
        img.src = attUrl(a);
        img.alt = a.caption || a.name;
        // A snapshot can outlive its pictures — the file may have been removed from the chat since.
        // The browser telling us the thumbnail 404s is the only way this side can know, and knowing
        // BEFORE the run is worth a lot: otherwise the first sign is a failed graph.
        img.addEventListener("error", () => {
          if (gone(node).has(a.name)) return;
          gone(node).add(a.name);
          cell.classList.add("gone");
          cell.title = a.name + "\nThis file is no longer on disk — press ⟳ Update History.";
          renderShots(node);
          node.setDirtyCanvas?.(true, true);
        });
        cell.appendChild(img);
        const pin = document.createElement("span");
        pin.className = "kb-db-pin";
        if (lost) {
          pin.textContent = "gone";
          cell.appendChild(pin);
        } else if (used) {
          const n = plan.bounds.findIndex((b) => b.name === a.name);
          if (n >= 0) {
            pin.textContent = "K" + (n + 1);
            cell.appendChild(pin);
          }
        }
        cell.title = (a.caption ? a.caption + "\n" : "") + a.name + "\n"
          + (lost ? "This file is no longer on disk — press ⟳ Update History."
             : used ? "A shot boundary — click to leave it out."
             : "Not used — click to make it a boundary.");
        cell.addEventListener("pointerdown", (e) => e.stopPropagation());
        cell.addEventListener("click", (e) => {
          e.preventDefault(); e.stopPropagation();
          toggle(st.noimg, a.name);
          render(node);
          node.setDirtyCanvas?.(true, true);
        });
        pics.appendChild(cell);
      }
      right.appendChild(pics);
    }
    row.appendChild(right);
    view.appendChild(row);
  });

  view.scrollTop = keep;
  renderShots(node);
}

function renderShots(node) {
  const el = node._dbShots;
  if (!el) return;
  const st = ST(node);
  el.innerHTML = "";
  const { shots, notes } = planBoard(st.msgs, st.skip, st.noimg, st.breaks);

  if (!shots.length) {
    const e = document.createElement("div");
    e.className = "kb-db-note";
    e.textContent = "no shots yet — tick two pictures, or some messages";
    el.appendChild(e);
    return;
  }

  shots.forEach((s, i) => {
    const row = document.createElement("div");
    row.className = "kb-db-shot";
    const n = document.createElement("b");
    n.textContent = String(i + 1);
    const kf = document.createElement("span");
    kf.className = "kb-db-kf";
    kf.textContent = s.tail ? "text only"
      : (s.start.name.replace(/\.\w+$/, "") + " → " + s.end.name.replace(/\.\w+$/, ""));
    kf.title = kf.textContent;

    const dur = document.createElement("input");
    dur.className = "kb-db-num";
    dur.value = String(nth(st.dur, i, DEFAULT_DURATION));
    dur.title = "Seconds. Snapped up to H3's 0.71 s grid when it runs.";
    dur.addEventListener("pointerdown", (e) => e.stopPropagation());
    dur.addEventListener("keydown", (e) => e.stopPropagation());
    dur.addEventListener("change", () => {
      const v = parseFloat(dur.value);
      const list = st.dur.slice();
      while (list.length < shots.length) list.push(nth(st.dur, list.length, DEFAULT_DURATION));
      list[i] = Number.isFinite(v) && v > 0 ? v : DEFAULT_DURATION;
      st.dur = list.slice(0, shots.length);
      renderShots(node);
      node.setDirtyCanvas?.(true, true);
    });

    row.append(n, kf, dur);

    // `link` only speaks for a shot that has no start keyframe and isn't the opening one.
    if (s.tail && i > 0) {
      const sel = document.createElement("select");
      sel.className = "kb-db-sel";
      for (const v of LINKS) {
        const o = document.createElement("option");
        o.value = v; o.textContent = v;
        if (nth(st.lnk, i, "continue") === v) o.selected = true;
        sel.appendChild(o);
      }
      sel.title = "continue = carry on from the previous shot's last frame; cut = start fresh.";
      sel.addEventListener("pointerdown", (e) => e.stopPropagation());
      sel.addEventListener("change", () => {
        const list = st.lnk.slice();
        while (list.length < shots.length) list.push(nth(st.lnk, list.length, "continue"));
        list[i] = sel.value;
        st.lnk = list.slice(0, shots.length);
        node.setDirtyCanvas?.(true, true);
      });
      row.appendChild(sel);
    } else {
      const dash = document.createElement("span");
      dash.className = "kb-db-sel";
      dash.style.cssText += "opacity:.35;border-color:transparent;background:transparent;";
      dash.textContent = i === 0 ? "opens" : "keyframed";
      row.appendChild(dash);
    }

    const c = document.createElement("span");
    c.className = "kb-db-msgs";
    c.textContent = s.msgs.length + " msg";
    row.appendChild(c);
    el.appendChild(row);
  });

  const lost = gone(node).size;
  if (lost) {
    notes.unshift(lost + " picture(s) no longer on disk — press ⟳ Update History before running, "
                  + "or the run will stop on the first one");
  }
  for (const t of notes) {
    const w = document.createElement("div");
    w.className = "kb-db-note";
    w.textContent = "⚠ " + t;
    el.appendChild(w);
  }
}

// ── setup ───────────────────────────────────────────────────────────────────────────────────

function setup(node) {
  injectStyle();
  for (let i = node.widgets.length - 1; i >= 0; i--) {
    if (node.widgets[i].name === "board_state") node.widgets.splice(i, 1);
  }
  ST(node);
  node._dbNote = "";

  const wrap = document.createElement("div");
  wrap.className = "kb-db-wrap";
  const box = document.createElement("div");
  box.className = "kb-db-box";
  const view = document.createElement("div");
  view.className = "kb-db-view";
  box.appendChild(view);
  const shots = document.createElement("div");
  shots.className = "kb-db-shots";
  wrap.append(box, shots);

  view.addEventListener("wheel", (e) => { view.scrollTop += e.deltaY; e.preventDefault(); e.stopPropagation(); }, { passive: false });
  shots.addEventListener("wheel", (e) => { shots.scrollTop += e.deltaY; e.preventDefault(); e.stopPropagation(); }, { passive: false });

  node._dbView = view;
  node._dbShots = shots;

  node.addDOMWidget("board_state", "kinburg_dreamboard", wrap, {
    serialize: true,
    getValue: () => stateJSON(node),
    setValue: (v) => loadState(node, v),
    getMinHeight: () => VIEW_MIN + GAP + SHOTS_MIN,
    getMaxHeight: () => 100000,
  });

  const pick = node.addWidget("combo", "→ chat", node.properties?._dbTarget || AUTO, (v) => {
    node.properties = node.properties || {};
    node.properties._dbTarget = v;
  }, { values: [AUTO], serialize: false });
  pick.tooltip = "Which chat window to pull from. Leave on auto when there is only one.";
  node._dbRefresh = () => { pick.options.values = [AUTO, ...chatNodes().map(label)]; };
  node._dbRefresh();

  const btn = node.addWidget("button", "⟳ Update History", null, () => {
    node._dbRefresh();
    updateHistory(node);
  }, { serialize: false });
  btn.tooltip = "Pull the conversation in (or refresh it after chatting some more). Your ticks are "
    + "kept — the chat only grows at the end.";

  render(node);
  if ((node.size?.[1] || 0) < 420) node.setSize([Math.max(node.size?.[0] || 0, 420), 460]);
}

app.registerExtension({
  name: "Kinburg.DreamBoard",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      setup(this);
      instances.add(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      const w = (this.widgets || []).find((x) => x.name === "→ chat");
      if (w) { this._dbRefresh?.(); w.value = this.properties?._dbTarget || AUTO; }
      render(this);
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      instances.delete(this);
      return onRemoved?.apply(this, arguments);
    };
  },
});
