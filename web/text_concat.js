import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Auto-growing string inputs for the Unlim Text Concat node.
app.registerExtension({
  name: "Kinburg.UnlimTextConcat",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "UnlimTextConcat") return;
    installDynamicInputs(nodeType, "text_", "STRING");
  },
});
