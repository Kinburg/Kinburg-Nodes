import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Start Timer: auto-growing wildcard inputs (any_1, any_2, …) so you can tap every branch
// that feeds your sampler. The node executes only after all are ready, so the timer starts
// right before the sampler instead of as soon as the first branch resolves.
app.registerExtension({
  name: "Kinburg.StartTimer",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "StartTimer") return;
    installDynamicInputs(nodeType, "any_", "*");
  },
});
