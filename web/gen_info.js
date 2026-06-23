import { app } from "../../scripts/app.js";

// Generation Info: a collapsed-by-default settings dump on the node (click to expand).
app.registerExtension({
  name: "Kinburg.GenerationInfo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "GenerationInfo") return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const t = message?.kinburg_geninfo?.[0];
      if (t == null) return;
      if (!this._giEl) {
        const root = document.createElement("div");
        const hdr = document.createElement("div");
        hdr.style.cssText = "cursor:pointer;color:#9a9aa2;font-size:11px;padding:2px;user-select:none;";
        const pre = document.createElement("pre");
        pre.style.cssText =
          "display:none;white-space:pre-wrap;word-break:break-word;font:12px/1.45 ui-monospace,Consolas,monospace;" +
          "max-height:320px;overflow:auto;background:#181818;color:#cfe6da;border-radius:6px;padding:6px 8px;margin:2px;";
        let open = false;
        const sync = () => { pre.style.display = open ? "" : "none"; hdr.textContent = (open ? "▾" : "▸") + " settings dump"; };
        hdr.onclick = () => { open = !open; sync(); this.setDirtyCanvas(true, true); };
        sync();
        root.append(hdr, pre);
        this._giEl = pre;
        this.addDOMWidget("geninfo_display", "info", root, { serialize: false });
      }
      this._giEl.textContent = t;
      this.setDirtyCanvas(true, true);
    };
  },
});
