import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "ImageCompare.OpenLink",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ImageCompareHTML") return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const url = message?.compare_url?.[0];
      if (!url) return;
      const full = /^(https?|file):/.test(url) ? url : window.location.origin + url;
      if (!this._cmpLink) {
        const a = document.createElement("a");
        a.target = "_blank";
        a.rel = "noopener";
        a.style.cssText =
          "display:block;text-align:center;color:#fff;background:#4f8cff;" +
          "border-radius:6px;padding:6px;margin:4px 2px;text-decoration:none;font-size:12px;";
        this._cmpLink = a;
        this.addDOMWidget("compare_link", "link", a, {});
      }
      this._cmpLink.href = full;
      this._cmpLink.textContent = "🔗 Open comparison";
      this.setDirtyCanvas(true, true);
    };
  },
});
