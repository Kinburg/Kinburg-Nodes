import { app } from "../../scripts/app.js";

// Combo to String — let the core `Primitive` node (in COMBO mode) connect into our
// ComboToString node, which core ComfyUI otherwise blocks (a COMBO output only links to a
// COMBO input whose value list overlaps). We patch PrimitiveNode.onConnectOutput so that a
// link to ComboToString is always allowed, and we push the Primitive's current value into our
// node's `value` widget (the same mechanism core uses for real combo inputs).

const TARGET = "ComboToString";

function isTarget(node) {
  return node && (node.comfyClass === TARGET || node.type === TARGET);
}

app.registerExtension({
  name: "Kinburg.ComboToString",
  async setup() {
    const LG = globalThis.LiteGraph || window.LiteGraph;
    const ctor = LG?.registered_node_types?.["PrimitiveNode"];
    const proto = ctor?.prototype;
    if (!proto || typeof proto.onConnectOutput !== "function") {
      console.warn("[Kinburg] ComboToString: PrimitiveNode not found — combo→string wiring won't be patched.");
      return;
    }
    if (proto.__kinburgComboPatched) return;

    const orig = proto.onConnectOutput;
    proto.onConnectOutput = function (slot, inputType, input, targetNode, targetSlot) {
      if (isTarget(targetNode)) {
        // Deliver the Primitive's current value into our node's `value` widget, exactly like
        // core does for a valid combo connection, then allow the link.
        try {
          this.applyToGraph?.([{ target_id: targetNode.id, target_slot: targetSlot }]);
        } catch (e) {
          console.error("[Kinburg] ComboToString: applyToGraph failed", e);
        }
        return true;
      }
      return orig.apply(this, arguments);
    };
    proto.__kinburgComboPatched = true;
  },
});
