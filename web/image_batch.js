import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Auto-growing IMAGE inputs for the Unlim Image Batch node.
app.registerExtension({
  name: "Kinburg.UnlimImageBatch",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "UnlimImageBatch") return;
    installDynamicInputs(nodeType, "image_", "IMAGE");
  },
});
