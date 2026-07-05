import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Auto-growing item inputs for the Context Collector node.
app.registerExtension({
  name: "Kinburg.ContextCollector",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ContextCollector") return;
    installDynamicInputs(nodeType, "item_", "STRING");
  },
});
