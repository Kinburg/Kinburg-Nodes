import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Auto-growing cast inputs on the Morpheus sampler and Storyboard — one slot per Card Presets card
// (a band member or a location), plus a spare. A shot gets a card when its prompt names it.
app.registerExtension({
  name: "Kinburg.MorpheusCast",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KinburgMorpheus" && nodeData.name !== "KinburgMorpheusStoryboard") return;
    installDynamicInputs(nodeType, "cast_", "KINBURG_CAST");
    if (nodeData.name === "KinburgMorpheusStoryboard") installDynamicInputs(nodeType, "location_", "KINBURG_CAST");
  },
});
