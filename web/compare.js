import { app } from "../../scripts/app.js";

// (Re)build the "Open comparison" link from the URL saved in node.properties, so it survives
// tab switches and workflow reloads (onExecuted only fires on a run, not on restore).
function ensureLink(node) {
  const url = node.properties?._cmpUrl;
  if (!url) return;
  const full = /^(https?|file):/.test(url) ? url : window.location.origin + url;
  if (!node._cmpLink) {
    // The widget element is a wrapper, not the anchor itself: ComfyUI stretches the widget
    // element to fill whatever vertical space the node gives it, so if the <a> *were* the
    // element it would grow with the node. Instead the wrapper takes the space (its empty
    // area just shows the node's grey background) and the anchor keeps its natural size,
    // pinned to the top by normal block flow.
    const wrap = document.createElement("div");
    wrap.style.cssText = "box-sizing:border-box;overflow:hidden;";
    const a = document.createElement("a");
    a.target = "_blank";
    a.rel = "noopener";
    a.style.cssText =
      "display:block;text-align:center;color:#fff;background:#4f8cff;" +
      "border-radius:6px;padding:6px;margin:4px 2px;text-decoration:none;font-size:12px;box-sizing:border-box;";
    wrap.appendChild(a);
    node._cmpLink = a;
    node.addDOMWidget("compare_link", "link", wrap, {});
  }
  node._cmpLink.href = full;
  node._cmpLink.textContent = "🔗 Open comparison";
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "ImageCompare.OpenLink",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ImageCompareHTML") return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const url = message?.compare_url?.[0];
      if (!url) return;
      this.properties = this.properties || {};
      this.properties._cmpUrl = url; // persisted -> restored on tab switch / reload
      ensureLink(this);
    };

    // Restore the link when the node is (re)created or a workflow is loaded.
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      ensureLink(this);
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      ensureLink(this); // properties are populated by now
      return r;
    };
  },
});
