import { app } from "../../scripts/app.js";

// Set Results / Get Results — name-based accumulators (image + text + gen-info pairs).
//   Set Results: labelled pass-through; index auto-increments on copy; title shows name+index.
//   Get Results: a name dropdown (live list of Set names) + a "Collect" button that physically
//                wires every matching Set's output into it (real links, in index order).
const PAIRS = [
  { set: "SetAccumImages",   get: "GetAccumImages",   slot: "image_", type: "IMAGE" },
  { set: "SetAccumTexts",    get: "GetAccumTexts",    slot: "text_",  type: "STRING" },
  // Compare-tuned text pairs: same wiring as texts, but the Get joins with a fixed
  // separator (prompts: a '---' line; captions: a newline) to match Image Compare.
  { set: "SetAccumPrompts",  get: "GetAccumPrompts",  slot: "text_",  type: "STRING" },
  { set: "SetAccumCaptions", get: "GetAccumCaptions", slot: "text_",  type: "STRING" },
  { set: "SetAccumGenInfo",  get: "GetAccumGenInfo",  slot: "data_",  type: "GEN_INFO" },
];

const wv = (node, name) => node.widgets?.find((w) => w.name === name);
const isType = (node, t) => node.comfyClass === t || node.type === t;

function retitle(node, pair) {
  const nm = (wv(node, "name")?.value ?? "").trim();
  if (isType(node, pair.set)) {
    node.title = `Set ${nm || "?"} #${wv(node, "index")?.value ?? 0}`;
  } else {
    node.title = nm ? `Get ${nm}` : "Get Results";
  }
}

// Distinct accumulator names currently defined by Set nodes of this pair — drives the dropdown.
function setNames(graph, setClass) {
  if (!graph) return [];
  const names = new Set();
  for (const n of graph._nodes || []) {
    if (isType(n, setClass)) { const v = (wv(n, "name")?.value || "").trim(); if (v) names.add(v); }
  }
  return [...names].sort();
}

// Set index = (max index among same-name Sets) + 1, so a fresh/pasted Set is unique.
function autoIndexSet(node, pair) {
  if (!node.graph) return;
  const nameW = wv(node, "name"), idxW = wv(node, "index");
  if (!nameW || !idxW) return;
  const others = node.graph._nodes.filter(
    (n) => n !== node && isType(n, pair.set) && wv(n, "name")?.value === nameW.value);
  if (others.length) idxW.value = Math.max(...others.map((n) => wv(n, "index")?.value ?? 0)) + 1;
}

// Re-collect every Get accumulator in the graph (any pair). Returns how many were wired.
function collectAll(graph) {
  if (!graph) return 0;
  let n = 0;
  for (const node of graph._nodes || []) {
    const pair = PAIRS.find((p) => isType(node, p.get));
    if (pair) { collect(node, pair); n++; }
  }
  return n;
}

function collect(getNode, pair) {
  const graph = getNode.graph;
  if (!graph) return;
  const name = (wv(getNode, "name")?.value || "").trim();
  // Drop any previously-collected inputs (and their links).
  for (let i = (getNode.inputs?.length || 0) - 1; i >= 0; i--) {
    if (getNode.inputs[i].name.startsWith(pair.slot)) getNode.removeInput(i);
  }
  let count = 0;
  if (name) {
    const sets = graph._nodes
      .filter((n) => isType(n, pair.set) && (wv(n, "name")?.value || "").trim() === name && n.inputs?.[0]?.link != null)
      .sort((a, b) => (wv(a, "index")?.value ?? 0) - (wv(b, "index")?.value ?? 0));
    sets.forEach((s, i) => {
      getNode.addInput(`${pair.slot}${i + 1}`, pair.type);
      s.connect(0, getNode, getNode.inputs.length - 1);
    });
    count = sets.length;
  }
  getNode.title = name ? `Get ${name} (${count})` : "Get Results";
  getNode.setDirtyCanvas(true, true);
}

function registerPair(pair) {
  // ---- Set node: auto-index (menu-add + paste) and live title ----
  app.registerExtension({
    name: `Kinburg.${pair.set}`,
    async beforeRegisterNodeDef(nodeType, nodeData) {
      if (nodeData.name !== pair.set) return;
      const onAdded = nodeType.prototype.onAdded;
      nodeType.prototype.onAdded = function () {
        const r = onAdded?.apply(this, arguments);
        this._justAdded = true;
        const retit = () => retitle(this, pair);
        for (const wn of ["name", "index"]) {
          const wd = wv(this, wn);
          if (wd) { const cb = wd.callback; wd.callback = function (...a) { const rr = cb?.apply(this, a); retit(); return rr; }; }
        }
        if (this.graph && !app.configuringGraph) autoIndexSet(this, pair); // menu-add
        retit();
        return r;
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        // Paste: configure() just restored the COPIED index — re-increment over it.
        // (Workflow load also calls onConfigure but with app.configuringGraph true → preserved.)
        if (this._justAdded && this.graph && !app.configuringGraph) autoIndexSet(this, pair);
        this._justAdded = false;
        retitle(this, pair);
        return r;
      };
    },
  });

  // ---- Get node: dropdown of Set names (auto-collects on pick) + Collect button ----
  app.registerExtension({
    name: `Kinburg.${pair.get}`,
    async beforeRegisterNodeDef(nodeType, nodeData) {
      if (nodeData.name !== pair.get) return;
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        const idx = this.widgets.findIndex((w) => w.name === "name");
        const cur = idx >= 0 ? this.widgets[idx].value : "";
        if (idx >= 0) this.widgets.splice(idx, 1);
        const opts = {};
        Object.defineProperty(opts, "values", { get: () => setNames(this.graph, pair.set), enumerable: true, configurable: true });
        const combo = this.addWidget("combo", "name", cur, () => { if (!app.configuringGraph) collect(this, pair); }, opts);
        const ni = this.widgets.indexOf(combo);
        if (idx >= 0 && ni !== idx) { this.widgets.splice(ni, 1); this.widgets.splice(idx, 0, combo); }
        this.addWidget("button", "🔌 Collect", null, () => collect(this, pair));
        retitle(this, pair);
        return r;
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () { const r = onConfigure?.apply(this, arguments); retitle(this, pair); return r; };
    },
  });
}

for (const pair of PAIRS) registerPair(pair);

// ---- Collect All: one button that (re)wires every Get accumulator in the graph ----
app.registerExtension({
  name: "Kinburg.CollectAllAccumulators",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CollectAllAccumulators") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      this.addWidget("button", "🔌 Collect All", null, () => {
        const n = collectAll(this.graph);
        this.title = `Collect All (${n})`;
        this.setDirtyCanvas(true, true);
      });
      return r;
    };
  },
});
