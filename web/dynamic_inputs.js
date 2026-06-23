// Shared helper for auto-growing, same-type input slots (text_1.., image_1.., ...).
//
// Invariant kept on every connection change: the node shows every connected `${prefix}N`
// slot plus exactly one trailing empty one, and never fewer than two. Slot names stay
// sequential (no gaps), so the Python side can join/stack them by index.
export function installDynamicInputs(nodeType, prefix, type) {
  const dynOf = (node) => (node.inputs || []).filter(s => s.name.startsWith(prefix));

  function update(node) {
    let slots = dynOf(node);

    // 1-based position of the last connected dynamic input (0 if none).
    let lastConnected = 0;
    slots.forEach((s, k) => { if (s.link != null) lastConnected = k + 1; });
    const desired = Math.max(2, lastConnected + 1);

    // Trailing slots past `desired` are always empty — safe to drop.
    while (slots.length > desired) {
      node.removeInput(node.inputs.indexOf(slots[slots.length - 1]));
      slots.pop();
    }
    while (slots.length < desired) {
      node.addInput(`${prefix}${slots.length + 1}`, type);
      slots = dynOf(node);
    }
    // Renumber by position so names never have gaps.
    let n = 1;
    for (const s of node.inputs) {
      if (s.name.startsWith(prefix)) s.name = `${prefix}${n++}`;
    }
    node.setDirtyCanvas?.(true, true);
  }

  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);
    update(this);
    return r;
  };

  const onConnectionsChange = nodeType.prototype.onConnectionsChange;
  nodeType.prototype.onConnectionsChange = function (type_, index, connected, link_info, ioSlot) {
    const r = onConnectionsChange?.apply(this, arguments);
    // LiteGraph.INPUT === 1; guard in case the global is unavailable.
    if (type_ === (window.LiteGraph?.INPUT ?? 1)) update(this);
    return r;
  };
}
