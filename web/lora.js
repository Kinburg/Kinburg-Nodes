import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Auto-growing lora_* inputs for the Lora Unlim Accumulator (same helper as the other
// accumulator-style nodes): connect the last slot and a new one appears.
app.registerExtension({
  name: "Kinburg.LoraUnlimAccumulator",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LoraUnlimAccumulator") return;
    installDynamicInputs(nodeType, "lora_", "KINBURG_LORA");
  },
});
