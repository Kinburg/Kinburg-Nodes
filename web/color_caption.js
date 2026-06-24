import { app } from "../../scripts/app.js";
import { buildColorControl } from "./color_control.js";

app.registerExtension({
  name: "Kinburg.ColorCaption",
  async nodeCreated(node) {
    if (node.comfyClass !== "ColorCaption" && node.type !== "ColorCaption") return;
    const textW = node.widgets?.find(w => w.name === "color");
    const bandW = node.widgets?.find(w => w.name === "band_color");
    if (!textW || !bandW) return;

    const root = document.createElement("div");
    root.style.cssText = "display:flex;flex-direction:column;gap:10px;";
    root.append(
      buildColorControl(node, textW, "Text color", "#FFFFFF"),
      buildColorControl(node, bandW, "Band color", "#000000"),
    );

    node.addDOMWidget("color_pickers", "colorpickers", root, {
      serialize: false,
      getMinHeight: () => 220,
    });

    // After a saved workflow loads, configure() restores color/band_color values but
    // doesn't fire the visual update — re-sync the swatches/picker to the restored values.
    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      for (const w of (this.widgets || [])) {
        if (typeof w._knReflect === "function") w._knReflect();
      }
      return r;
    };
  },
});
