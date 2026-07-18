import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Auto-growing wildcard inputs for the Any Switch node.
app.registerExtension({
  name: "Kinburg.AnySwitch",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KinburgAnySwitch") return;
    installDynamicInputs(nodeType, "input_", "*");
  },
});
