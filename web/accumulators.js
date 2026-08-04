import { app } from "../../scripts/app.js";

// Set Results / Get Results — name-based accumulators (image + text + gen-info pairs).
//   Set Results: labelled pass-through; index auto-increments on copy; title shows name+index.
//   Get Results: a name dropdown (live list of Set names) + a "Collect" button that physically
//                wires every matching Set's output into it (real links, in index order).
const PAIRS = [
  { set: "SetAccumImages",   get: "GetAccumImages",     slot: "image_", type: "IMAGE" },
  // Same Set, a second Get that returns a LIST instead of a batch (different sizes coexist).
  { set: "SetAccumImages",   get: "GetAccumImagesList", slot: "image_", type: "IMAGE" },
  { set: "SetAccumAudio",    get: "GetAccumAudio",    slot: "audio_", type: "AUDIO"  },
  { set: "SetAccumTexts",    get: "GetAccumTexts",    slot: "text_",  type: "STRING" },
  // Compare-tuned text pairs: same wiring as texts, but the Get joins with a fixed
  // separator (prompts: a '---' line; captions: a newline) to match Image Compare.
  { set: "SetAccumPrompts",  get: "GetAccumPrompts",  slot: "text_",  type: "STRING" },
  { set: "SetAccumCaptions", get: "GetAccumCaptions", slot: "text_",  type: "STRING" },
  { set: "SetAccumGenInfo",  get: "GetAccumGenInfo",  slot: "data_",  type: "GEN_INFO" },
];

const wv = (node, name) => node.widgets?.find((w) => w.name === name);
const isType = (node, t) => node.comfyClass === t || node.type === t;
// Only active nodes are collected; Bypass (mode 4) and Mute (mode 2) are skipped, so toggling a
// Set's mode and re-collecting wires it in or drops it out.
const isActive = (node) => (node.mode ?? 0) === 0;

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
      .filter((n) => isType(n, pair.set) && isActive(n) && (wv(n, "name")?.value || "").trim() === name && n.inputs?.[0]?.link != null)
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

// ---- Set node: auto-index (menu-add + paste) and live title. Registered once per Set class,
// since one Set can now feed several Get types (e.g. images batch + images list). ----
function registerSet(setClass) {
  const pair = { set: setClass };  // Set-side helpers only need the set class
  app.registerExtension({
    name: `Kinburg.${setClass}`,
    async beforeRegisterNodeDef(nodeType, nodeData) {
      if (nodeData.name !== setClass) return;
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
}

// ---- Get node: dropdown of Set names (auto-collects on pick) + Collect button ----
function registerGet(pair) {
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

for (const setClass of new Set(PAIRS.map((p) => p.set))) registerSet(setClass);
for (const pair of PAIRS) registerGet(pair);

// ---- Collect All, hosted on Image Compare ----
// Collecting exists to feed a comparison, so the one button that (re)wires EVERY Get accumulator in
// the graph lives on the node that consumes them, rather than on a separate helper node.
// Every node that consumes accumulated results and therefore wants the button. Collecting exists
// to feed a comparison, so it lives on the nodes that consume them rather than a helper node.
const COLLECT_HOSTS = new Set(["ImageCompareHTML", "KinburgSirenCompare"]);

app.registerExtension({
  name: "Kinburg.CollectAllOnCompare",
  async setup() {
    // Auto-collect before the prompt is queued, whenever an (active) Image Compare has the toggle
    // on. Runs synchronously before the original queuePrompt builds the prompt, so the freshly
    // (re)wired links are the ones that get queued. Idempotent — patched only once.
    const orig = app.queuePrompt;
    if (typeof orig === "function" && !orig.__kinburgAutoCollect) {
      const patched = async function (...args) {
        try {
          const g = app.graph;
          const on = (g?._nodes || []).some(
            (n) => [...COLLECT_HOSTS].some((h) => isType(n, h)) && isActive(n)
                 && wv(n, "auto_collect")?.value);
          if (on) collectAll(g);
        } catch (e) {
          console.error("[Kinburg] auto-collect before queue failed:", e);
        }
        return orig.apply(this, args);
      };
      patched.__kinburgAutoCollect = true;
      app.queuePrompt = patched;
    }
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!COLLECT_HOSTS.has(nodeData.name)) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      // The label keeps the last count, so pressing it tells you how many Gets were re-wired
      // without stealing the node's title (which Image Compare uses for its own name).
      const btn = this.addWidget("button", "🔌 Collect All", null, () => {
        btn.label = `🔌 Collect All (${collectAll(this.graph)})`;
        this.setDirtyCanvas(true, true);
      });
      return r;
    };
  },
});
