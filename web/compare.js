import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Every node that publishes a `compare_url` and wants the "Open comparison" link widget. The link
// logic below is entirely about the URL, not about what the page contains, so audio rides along.
const LINK_NODES = new Set(["ImageCompareHTML", "KinburgSirenCompare"]);

const LINK_LIVE = "display:block;text-align:center;color:#fff;background:#4f8cff;" +
  "border-radius:6px;padding:6px;margin:4px 2px;text-decoration:none;font-size:12px;box-sizing:border-box;";
const LINK_DEAD = "display:block;text-align:center;color:#e9b8ab;background:#5b3a34;" +
  "border-radius:6px;padding:6px;margin:4px 2px;text-decoration:none;font-size:12px;box-sizing:border-box;cursor:default;";

// (Re)build the "Open comparison" link from the URL saved in node.properties, so it survives
// tab switches and workflow reloads (a run only refreshes it, it doesn't create it).
// `fresh` = this URL came from a run that just finished; otherwise it was RESTORED, and may well
// point at a run whose folder has since been deleted (or that predates a restart old enough to
// have fallen out of the server's folder registry) — so it is labelled as such and verified,
// rather than letting the click land on a blank 404.
function ensureLink(node, fresh) {
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
    wrap.appendChild(a);
    node._cmpLink = a;
    node.addDOMWidget("compare_link", "link", wrap, {});
  }
  const a = node._cmpLink;
  a.href = full;
  a.style.cssText = LINK_LIVE;
  a.textContent = fresh ? "🔗 Open comparison" : "🔗 Open comparison (previous run)";
  node.setDirtyCanvas(true, true);
  if (fresh || !/^https?:/.test(full)) return;
  // HEAD, so a self-contained multi-MB page isn't downloaded just to test it.
  const token = (node._cmpProbe = (node._cmpProbe || 0) + 1);
  fetch(full, { method: "HEAD" })
    .then((r) => {
      if (r.ok || token !== node._cmpProbe || node._cmpLink !== a) return;
      a.removeAttribute("href");        // dead: don't open a blank tab on click
      a.textContent = "⚠ Comparison expired — run to rebuild";
      a.style.cssText = LINK_DEAD;
      node.setDirtyCanvas(true, true);
    })
    .catch(() => {});                   // server unreachable: leave the link as it is
}

// Store the URL a run just produced and repaint the link.
function applyUrl(node, url) {
  if (!node || !url) return;
  node.properties = node.properties || {};
  node.properties._cmpUrl = url; // persisted -> restored on tab switch / reload
  ensureLink(node, true);
}

// Primary hook: the api event, NOT the onExecuted prototype chain. Every extension that patches
// onExecuted sits on that one chain, so a single one of them throwing (or forgetting to call
// through) silently leaves this node showing the URL of a PREVIOUS run — a link that 404s once the
// server has been restarted, because the token registry it was minted in is gone.
api.addEventListener("executed", ({ detail }) => {
  try {
    const url = detail?.output?.compare_url?.[0];
    if (!url) return;
    const node = app.graph?.getNodeById?.(detail.node);
    if (node && (LINK_NODES.has(node.comfyClass) || LINK_NODES.has(node.type))) {
      applyUrl(node, url);
    }
  } catch (e) {
    console.error("[Kinburg] Image Compare: could not refresh the link:", e);
  }
});

app.registerExtension({
  name: "ImageCompare.OpenLink",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!LINK_NODES.has(nodeData.name)) return;

    // Kept as a fallback for frontends that don't emit the api event; applyUrl is idempotent.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      applyUrl(this, message?.compare_url?.[0]);
    };

    // Restore the link when the node is (re)created or a workflow is loaded.
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      ensureLink(this, false);
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      ensureLink(this, false); // properties are populated by now
      return r;
    };
  },
});
