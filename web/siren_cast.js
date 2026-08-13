import { app } from "../../scripts/app.js";
import { installDynamicInputs } from "./dynamic_inputs.js";

// Auto-growing voice inputs — one slot per band member, plus a spare. Siren Cast pastes a voice's
// tags onto its section's caption; Siren Score matches marker names against the same cards.
app.registerExtension({
  name: "Kinburg.SirenCast",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KinburgSirenCast" && nodeData.name !== "KinburgSirenScore") return;
    installDynamicInputs(nodeType, "voice_", "KINBURG_VOICE");
  },
});
