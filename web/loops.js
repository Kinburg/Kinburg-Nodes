import { app } from "../../scripts/app.js";

// Auto-growing wildcard slots + Open/Close auto-pairing for the loop nodes.
//
// Each node grows its input slots on demand (connect the last one, a new spare appears) and
// keeps its output slots in lockstep, so the node shows only what's used plus one spare. Slot
// names stay sequential so the Python side lines them up. Prefixes differ per node:
//   Repeat/While Open & Close : value_*  in/out
//   For Each (Open)           : value_*  in   ->  element_*  out
//   For Each (Collect)        : result_* in   ->  list_*     out
const SLOTS = {
  KinburgRepeatOpen:    { in: "value_",  out: "value_" },
  KinburgRepeatClose:   { in: "value_",  out: "value_" },
  KinburgWhileOpen:     { in: "value_",  out: "value_" },
  KinburgWhileClose:    { in: "value_",  out: "value_" },
  KinburgForEachOpen:   { in: "value_",  out: "element_" },
  KinburgForEachCollect:{ in: "result_", out: "collected_" },
  KinburgListEmit:      { in: "value_",  out: "item_" },
};

// open class -> { close class, output names auto-wired into the same-named close inputs,
// and optionally `emit`: a List Output node chained after Collect (collected_0 -> value_0) so
// the accumulated python list comes out as a real per-item ComfyUI list. }
const PAIRS = {
  KinburgRepeatOpen:  { close: "KinburgRepeatClose",   links: ["flow", "index"] },
  KinburgWhileOpen:   { close: "KinburgWhileClose",    links: ["flow", "index"] },
  KinburgForEachOpen: { close: "KinburgForEachCollect", links: ["flow"], emit: "KinburgListEmit" },
};

function update(node, cfg) {
  const inP = cfg.in, outP = cfg.out;

  // Inputs: keep every connected one + a single spare.
  let ins = (node.inputs || []).filter((s) => s.name.startsWith(inP));
  let lastIn = 0;
  ins.forEach((s, k) => { if (s.link != null) lastIn = k + 1; });
  const wantIn = Math.max(1, lastIn + 1);
  while (ins.length > wantIn) { node.removeInput(node.inputs.indexOf(ins[ins.length - 1])); ins.pop(); }
  while (ins.length < wantIn) { node.addInput(`${inP}${ins.length}`, "*"); ins = (node.inputs || []).filter((s) => s.name.startsWith(inP)); }

  // Outputs follow the input count (but never drop a connected one).
  let outs = (node.outputs || []).filter((s) => s.name.startsWith(outP));
  let lastOut = 0;
  outs.forEach((s, k) => { if (s.links && s.links.length) lastOut = k + 1; });
  const wantOut = Math.max(wantIn, lastOut + 1);
  while (outs.length > wantOut) { node.removeOutput(node.outputs.indexOf(outs[outs.length - 1])); outs.pop(); }
  while (outs.length < wantOut) { node.addOutput(`${outP}${outs.length}`, "*"); outs = (node.outputs || []).filter((s) => s.name.startsWith(outP)); }

  // Renumber sequentially so names never have gaps.
  let n = 0;
  for (const s of node.inputs || []) if (s.name.startsWith(inP)) s.name = `${inP}${n++}`;
  n = 0;
  for (const s of node.outputs || []) if (s.name.startsWith(outP)) s.name = `${outP}${n++}`;
  node.setDirtyCanvas?.(true, true);
}

function addCloseAndLink(openNode, pair) {
  const graph = openNode.graph;
  if (!graph || !window.LiteGraph) return;
  const close = window.LiteGraph.createNode(pair.close);
  if (!close) return;
  graph.add(close);
  close.pos = [openNode.pos[0] + (openNode.size?.[0] || 240) + 80, openNode.pos[1]];
  for (const name of pair.links) {
    const o = (openNode.outputs || []).findIndex((s) => s.name === name);
    const i = (close.inputs || []).findIndex((s) => s.name === name);
    if (o >= 0 && i >= 0) openNode.connect(o, close, i);
  }
  close.setDirtyCanvas?.(true, true);

  // For Each: chain a List Output so the accumulated python list comes out as a real per-item
  // list (collected_0 -> value_0). Wire extra slots by hand if you accumulate more than one.
  if (pair.emit) {
    const emit = window.LiteGraph.createNode(pair.emit);
    if (emit) {
      graph.add(emit);
      emit.pos = [close.pos[0] + (close.size?.[0] || 240) + 60, close.pos[1]];
      const o = (close.outputs || []).findIndex((s) => s.name === "collected_0");
      const i = (emit.inputs || []).findIndex((s) => s.name === "value_0");
      if (o >= 0 && i >= 0) close.connect(o, emit, i);
      emit.setDirtyCanvas?.(true, true);
    }
  }
}

app.registerExtension({
  name: "Kinburg.Loops",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const name = nodeData?.name;
    const cfg = SLOTS[name];
    if (!cfg) return;
    const pair = PAIRS[name];

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      update(this, cfg);
      if (pair) this.addWidget("button", "🔗 Add / link Close", null, () => addCloseAndLink(this, pair));
      return r;
    };

    const onConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type_, index, connected, link_info, ioSlot) {
      const res = onConn?.apply(this, arguments);
      update(this, cfg);
      return res;
    };
  },
});
