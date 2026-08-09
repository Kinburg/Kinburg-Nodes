// The environment web/*.js expects, minus a browser.
//
// Exported as a STRING rather than as real modules: a suite is assembled by concatenating the
// stubs, the extension file(s) with their `import` lines stripped, and the assertions, then running
// the lot as one .mjs. That is the only way to reach the module-private functions these files are
// mostly made of, and it is what lets a test call the real setup() / onExecuted / widget callbacks.
//
// Deliberately a superset: one stub for every suite beats two that drift apart.
export const STUBS = String.raw`
class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = []; this.style = { cssText: "" }; this._cls = new Set();
    this.textContent = ""; this.value = ""; this.placeholder = ""; this.title = "";
    this.alt = ""; this.src = ""; this.disabled = false; this.type = "";
    this.checked = false; this.selected = false;
    this.scrollTop = 0; this.scrollHeight = 0; this.clientHeight = 0; this.offsetHeight = 20;
    this.classList = {
      add: (...c) => c.forEach((x) => this._cls.add(x)),
      remove: (...c) => c.forEach((x) => this._cls.delete(x)),
      contains: (c) => this._cls.has(c),
      toggle: (c, on) => {
        const v = on === undefined ? !this._cls.has(c) : !!on;
        if (v) this._cls.add(c); else this._cls.delete(c);
        return v;
      },
    };
  }
  get className() { return [...this._cls].join(" "); }
  set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get innerHTML() { return ""; }
  set innerHTML(v) { if (!v) this.children = []; }
  append(...k) { this.children.push(...k); }
  appendChild(k) { this.children.push(k); return k; }
  addEventListener(t, f) { (this._ls || (this._ls = {}))[t] = f; }
  // How a test raises an event the browser would have raised.
  fire(t, ev) { this._ls?.[t]?.(ev || { preventDefault() {}, stopPropagation() {} }); }
  removeEventListener() {} remove() {} focus() {} blur() {} setSelectionRange() {}
}
globalThis.document = {
  createElement: (t) => new El(t),
  getElementById: () => ({}),            // injectStyle sees the sheet already there and bails
  head: { appendChild() {} }, body: { appendChild() {} },
  addEventListener() {}, removeEventListener() {}, activeElement: null,
};
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.confirm = () => true;
globalThis.FormData = class { constructor() { this.d = []; } append(k, v, n) { this.d.push([k, v, n]); } };
const OPENED = [];
globalThis.window = { open: (u) => OPENED.push(u) };

const QUEUED = [];      // app.queuePrompt calls
const UPLOADS = [];     // [route, options] of every api.fetchApi call
let UPLOAD_FAILS = false;
const EXTS = [];
const extBy = (n) => EXTS.find((e) => e.name === n);
const app = {
  queuePrompt: (n) => { QUEUED.push(n); return Promise.resolve(); },
  graph: { _nodes: [], setDirtyCanvas() {} },
  registerExtension: (e) => { EXTS.push(e); },
};
const api = {
  addEventListener() {},
  apiURL: (r) => "/api" + r,
  fetchApi: (route, opt) => {
    UPLOADS.push([route, opt]);
    if (UPLOAD_FAILS) return Promise.resolve({ status: 500 });
    const n = UPLOADS.length;
    return Promise.resolve({
      status: 200,
      json: () => Promise.resolve({ name: "up" + n + ".png", subfolder: "kinburg_chat", type: "input" }),
    });
  },
};
const mkFile = (name) => ({ name, type: "image/png" });
const tick = () => new Promise((r) => setTimeout(r, 0));
`;
