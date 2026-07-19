import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Adds a "⏹ Stop loop" button to the Ouroboros node. Clicking it POSTs the node id to a backend
// route that sets a stop flag; the node polls that flag between iterations and returns gracefully
// with everything generated so far (unlike ComfyUI's Cancel, which aborts and discards the run).

app.registerExtension({
  name: "Kinburg.Ouroboros",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KinburgOuroboros") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const node = this;
      const btn = this.addWidget("button", "⏹ Stop loop", null, () => {
        const prev = btn.name;
        api.fetchApi("/kinburg/ouroboros/stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: String(node.id) }),
        })
          .then(() => {
            // brief confirmation, then restore the label
            btn.name = "⏹ stopping after this iteration…";
            app.graph?.setDirtyCanvas(true, true);
            setTimeout(() => { btn.name = prev; app.graph?.setDirtyCanvas(true, true); }, 2500);
          })
          .catch((e) => console.error("[Ouroboros] stop request failed", e));
      });
      btn.serialize = false; // it's an action, not saved state
      return r;
    };
  },
});
