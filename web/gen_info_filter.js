import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Auto-growing GEN_INFO inputs for the Generation Info Filter node.
app.registerExtension({
  name: "Kinburg.GenerationInfoFilter",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "GenerationInfoFilter") return;
    installDynamicInputs(nodeType, "data_", "GEN_INFO");
  },
});
