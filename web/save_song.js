import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Save Song: show an <audio> player on the node. ComfyUI only auto-attaches its native player to
// a fixed list of core audio node types, so we render our own from the node's ui.audio result
// (the file still lands in Media Assets via ui.audio, and the cover shows via ui.images). The
// player lives in a sized container with an explicit widget height so it's always visible.

const CLASS = "KinburgSaveSong";

function viewUrl(ref) {
  const qs = [
    "filename=" + encodeURIComponent(ref.filename || ""),
    "type=" + encodeURIComponent(ref.type || "output"),
    "subfolder=" + encodeURIComponent(ref.subfolder || ""),
  ].join("&");
  return api.apiURL ? api.apiURL("/view?" + qs) : "/view?" + qs;
}

function build() {
  const root = document.createElement("div");
  root.style.cssText = "display:flex;flex-direction:column;justify-content:center;width:100%;height:100%;box-sizing:border-box;padding:4px 6px;";
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.style.cssText = "width:100%;display:none;";
  const empty = document.createElement("div");
  empty.textContent = "(run to save & play)";
  empty.style.cssText = "color:#6d6d75;font:12px sans-serif;font-style:italic;text-align:center;";
  root.append(empty, audio);
  return { root, audio, empty };
}

function setSrc(node, ref) {
  if (!ref?.filename || !node._kbAudio) return;
  node._kbAudio.src = viewUrl(ref);
  node._kbAudio.style.display = "";
  if (node._kbEmpty) node._kbEmpty.style.display = "none";
}

app.registerExtension({
  name: "Kinburg.SaveSong",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const { root, audio, empty } = build();
      this._kbAudio = audio;
      this._kbEmpty = empty;
      const w = this.addDOMWidget("kb_player", "kinburg_player", root, { serialize: false });
      // Reserve a fixed height so the player is always visible (a bare DOM widget can collapse).
      w.computeSize = () => [this.size?.[0] || 300, 60];
      if (this.properties?.kb_audio) setSrc(this, this.properties.kb_audio);
      const width = Math.max(this.size?.[0] || 0, 300);
      if ((this.size?.[0] || 0) < 300) this.setSize([width, this.size?.[1] || 200]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const a = message?.audio?.[0];
      if (!a) return;
      this.properties = this.properties || {};
      this.properties.kb_audio = a;
      setSrc(this, a);
    };

    // Restore the player after a workflow load / tab switch.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      if (this.properties?.kb_audio) setSrc(this, this.properties.kb_audio);
      // Repair image_quality: older workflows (when `lyrics` was a widget, not a socket) have an
      // extra widgets_values entry that shifts this INT onto a stray string. Coerce it back.
      const iq = this.widgets?.find((w) => w.name === "image_quality");
      if (iq && typeof iq.value !== "number") {
        const n = parseInt(iq.value, 10);
        iq.value = Number.isFinite(n) ? n : 90;
      }
      return r;
    };
  },
});
