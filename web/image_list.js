import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Auto-growing IMAGE inputs for the Unlim Image List node.
app.registerExtension({
  name: "Kinburg.UnlimImageList",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "UnlimImageList") return;
    installDynamicInputs(nodeType, "image_", "IMAGE");
  },
});
