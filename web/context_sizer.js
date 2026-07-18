import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Auto-growing text inputs for the Context Sizer node.
app.registerExtension({
  name: "Kinburg.ContextSizer",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KinburgContextSizer") return;
    installDynamicInputs(nodeType, "text_", "STRING");
  },
});
