import { app } from "../../scripts/app.js";
import { buildColorControl } from "./color_control.js";

// Color Picker — the swatch/native-picker control as a standalone node, bound to `color`.
app.registerExtension({
  name: "Kinburg.ColorPicker",
  async nodeCreated(node) {
    if (node.comfyClass !== "ColorPicker" && node.type !== "ColorPicker") return;
    const w = node.widgets?.find((x) => x.name === "color");
    if (!w) return;

    const root = buildColorControl(node, w, "Color", "#FFFFFF");
    node.addDOMWidget("color_picker", "colorpicker", root, { serialize: false, getMinHeight: () => 96 });

    // After a saved workflow loads, re-sync the swatches/picker to the restored value.
    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      for (const wd of (this.widgets || [])) {
        if (typeof wd._knReflect === "function") wd._knReflect();
      }
      return r;
    };
  },
});
