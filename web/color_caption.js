import { app } from "../../scripts/app.js";

// Keep in sync with PALETTE in image_compare/color_caption.py
const PALETTE = [
  ["White", "#FFFFFF"], ["Black", "#000000"], ["Red", "#E53935"],
  ["Orange", "#FB8C00"], ["Yellow", "#FDD835"], ["Green", "#43A047"],
  ["Teal", "#00ACC1"], ["Blue", "#1E88E5"], ["Purple", "#8E24AA"],
  ["Pink", "#D81B60"],
];

function normalizeHex(v, fallback = "#FFFFFF") {
  if (!v) return fallback;
  let s = String(v).trim();
  if (s[0] !== "#") s = "#" + s;
  if (/^#[0-9a-fA-F]{3}$/.test(s)) s = "#" + s.slice(1).split("").map(c => c + c).join("");
  return /^#[0-9a-fA-F]{6}$/.test(s) ? s.toUpperCase() : fallback;
}

// Build a palette + native-picker control bound to a string widget. Returns its
// DOM root; `node.setDirtyCanvas` is called on every change.
function buildColorControl(node, widget, title, fallback) {
  const root = document.createElement("div");
  root.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:2px 0;";

  const heading = document.createElement("div");
  heading.textContent = title;
  heading.style.cssText = "font-size:11px;color:#9a9aa2;text-transform:uppercase;letter-spacing:.04em;";

  const swatches = document.createElement("div");
  swatches.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;";
  const buttons = [];
  for (const [name, hex] of PALETTE) {
    const b = document.createElement("button");
    b.title = `${name} ${hex}`;
    b.style.cssText =
      `width:22px;height:22px;border-radius:4px;cursor:pointer;background:${hex};` +
      `border:2px solid transparent;padding:0;`;
    b.onclick = (e) => { e.preventDefault(); setColor(hex); };
    swatches.appendChild(b);
    buttons.push([hex.toUpperCase(), b]);
  }

  const row = document.createElement("div");
  row.style.cssText = "display:flex;align-items:center;gap:8px;";
  const tag = document.createElement("span");
  tag.textContent = "Custom:";
  tag.style.cssText = "font-size:12px;color:#999;";
  const picker = document.createElement("input");
  picker.type = "color";
  picker.style.cssText = "width:36px;height:24px;padding:0;border:none;background:none;cursor:pointer;";
  picker.oninput = () => setColor(picker.value);
  const label = document.createElement("span");
  label.style.cssText = "font-family:monospace;font-size:12px;color:#ccc;";
  row.append(tag, picker, label);

  root.append(heading, swatches, row);

  function reflect(hex) {
    picker.value = hex;
    label.textContent = hex;
    for (const [h, b] of buttons) b.style.borderColor = (h === hex) ? "#4f8cff" : "transparent";
  }
  function setColor(v) {
    const hex = normalizeHex(v, fallback);
    widget.value = hex;
    reflect(hex);
    node.setDirtyCanvas(true, true);
  }

  reflect(normalizeHex(widget.value, fallback));
  const origCb = widget.callback;
  widget.callback = function (v) {
    origCb?.apply(this, arguments);
    reflect(normalizeHex(v, fallback));
  };
  // Used to re-sync the swatch/picker after a workflow load restores widget.value.
  widget._knReflect = () => reflect(normalizeHex(widget.value, fallback));
  return root;
}

app.registerExtension({
  name: "Kinburg.ColorCaption",
  async nodeCreated(node) {
    if (node.comfyClass !== "ColorCaption" && node.type !== "ColorCaption") return;
    const textW = node.widgets?.find(w => w.name === "color");
    const bandW = node.widgets?.find(w => w.name === "band_color");
    if (!textW || !bandW) return;

    const root = document.createElement("div");
    root.style.cssText = "display:flex;flex-direction:column;gap:10px;";
    root.append(
      buildColorControl(node, textW, "Text color", "#FFFFFF"),
      buildColorControl(node, bandW, "Band color", "#000000"),
    );

    node.addDOMWidget("color_pickers", "colorpickers", root, {
      serialize: false,
      getMinHeight: () => 220,
    });

    // After a saved workflow loads, configure() restores color/band_color values but
    // doesn't fire the visual update — re-sync the swatches/picker to the restored values.
    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      for (const w of (this.widgets || [])) {
        if (typeof w._knReflect === "function") w._knReflect();
      }
      return r;
    };
  },
});
