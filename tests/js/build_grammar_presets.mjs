import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STUBS } from "./stubs.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "..", "web") + path.sep;
const OUT = path.join(HERE, "run_grammar_presets.mjs");
// Harness for web/grammar_presets.js — the Prompt Presets library dialog. What it keeps honest:
//
//   * RENAME. A Prompt Presets node holds its preset by NAME, so a rename that doesn't follow the
//     graph leaves the node failing on the next run.
//   * BUILT-INS. A built-in renamed in the editor is a copy beside it; an old_name there would make
//     the store drop the user's edit over it — a silent reset.
//   * NAME CLASHES. The store overwrites by name, so creating onto a taken name must stop in the
//     dialog, before anything is posted.
//   * the two-click delete, and the reset button standing in for it on an edited built-in.
//
// The backend is stubbed as grammar_presets/routes.py behaves, so the dialog is driven end to end:
// open, edit, save, and the store the rows re-render from.

const strip = (p) => fs.readFileSync(p, "utf8")
  .split("\n").filter((l) => !/^import\s/.test(l)).join("\n")
  .replace(/^export\s+(function|const|async function)/gm, "$1");

const EXPOSE = `
globalThis.GP = { libraryRows, copyName, editorPayload, nameTaken, renameInGraph, syncNode,
                  presetEditor, libraryDialog, refreshStore,
                  setStore: (s) => { STORE = s; }, getStore: () => STORE,
                  posted: () => POSTED, lib: () => LIB, boxes: () => BOXES };
`;

// A backend that behaves like grammar_presets/routes.py over an in-memory library.
const EXTRA = String.raw`
globalThis.alert = () => {};
const BOXES = [];
document.body.appendChild = (el) => { BOXES.push(el); return el; };

const BUILTIN = { "Character Card (JSON)": "root ::= card", "Siren Voice Plan (table)": "root ::= row+" };
let LIB = {
  Writer: { prompt: "write ===PROMPT===", grammar: "root ::= prompt" },
  Critic: { prompt: "be harsh", grammar: "" },
  "Character Card (JSON)": { prompt: "fill the card", grammar: "root ::= card" },
};
const merged = () => ({ ...Object.fromEntries(Object.entries(BUILTIN).map(([k, g]) => [k, { prompt: "", grammar: g }])), ...LIB });
const storeData = () => ({
  ok: true, none: "\u{1F6AB} None",
  order: ["\u{1F6AB} None", ...Object.keys(BUILTIN),
          ...Object.keys(LIB).filter((k) => !(k in BUILTIN)).sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()))],
  presets: Object.fromEntries(Object.entries(merged()).map(([k, p]) => [k, {
    builtin: k in BUILTIN, edited: k in BUILTIN && k in LIB, prompt: !!p.prompt.trim(), grammar: !!p.grammar.trim() }])),
});

const POSTED = [];
const reply = (j) => ({ status: 200, json: async () => j });
api.fetchApi = async (route, opt) => {
  if (opt && opt.method === "POST") {
    const body = JSON.parse((opt && opt.body) || "{}");
    POSTED.push({ path: route, body });
    if (body.delete) {
      if (body.name in BUILTIN && !(body.name in LIB)) return reply({ ok: false, error: "cannot delete a built-in preset" });
      delete LIB[body.name];
    } else {
      if (!String(body.prompt).trim() && !String(body.grammar).trim()) return reply({ ok: false, error: "a preset needs a prompt, a grammar, or both" });
      if (body.old_name && body.old_name !== body.name) delete LIB[body.old_name];
      LIB[body.name] = { prompt: body.prompt, grammar: body.grammar };
    }
    return reply(storeData());
  }
  if (route.startsWith("/kinburg/grammars/preset")) {
    const nm = decodeURIComponent(route.split("name=")[1] || "");
    const p = merged()[nm];
    return reply(p ? { ok: true, name: nm, ...p } : { ok: false, error: "no preset named '" + nm + "'" });
  }
  return reply(storeData());
};
`;

const TESTS = String.raw`
const fails = [];
const check = (label, cond, extra) => {
  console.log((cond ? "  ok   " : "  FAIL ") + label + (extra !== undefined ? "  " + extra : ""));
  if (!cond) fails.push(label);
};
const GP = globalThis.GP;      // tick() comes from the shared stubs

// ---- DOM spelunking: the dialogs build their own trees, nothing here has ids ------------------
const walk = (el, fn) => { fn(el); for (const c of el.children || []) walk(c, fn); };
const allOf = (root, tag) => { const out = []; walk(root, (e) => { if (e.tagName === tag) out.push(e); }); return out; };
const btn = (root, text) => allOf(root, "BUTTON").find((b) => String(b.textContent).includes(text));
// The control sitting next to (or under) a labelled row: "Saved as", "Prompt", "GBNF grammar".
const field = (root, label) => {
  let hit = null;
  walk(root, (e) => {
    if (hit) return;
    const [l, c] = e.children || [];
    if (l && c && String(l.textContent) === label) hit = c;
  });
  return hit;
};
const lastBox = () => { const b = GP.boxes(); return b[b.length - 1].children[0]; };
const posts = () => GP.posted().filter((x) => x.path === "/kinburg/grammars/preset");
const rowOf = (box, name) => allOf(box, "DIV").find((d) => (d.children || []).some((c) => c.tagName === "SPAN" && String(c.textContent).includes(name)) && allOf(d, "BUTTON").length);

await GP.refreshStore();

// ---- the pure parts ---------------------------------------------------------------------------
check("the library lists in dropdown order, without the None option",
      JSON.stringify(GP.libraryRows("", null)) === JSON.stringify(["Character Card (JSON)", "Siren Voice Plan (table)", "Critic", "Writer"]),
      JSON.stringify(GP.libraryRows("", null)));
check("...the preset the node has picked goes first", GP.libraryRows("", "Writer")[0] === "Writer");
check("...search narrows it, ignoring case", JSON.stringify(GP.libraryRows("SIREN", null)) === JSON.stringify(["Siren Voice Plan (table)"]));

check("a duplicate is named beside the original", GP.copyName("Writer") === "Writer copy");
GP.setStore({ ...GP.getStore(), presets: { ...GP.getStore().presets, "Writer copy": {} } });
check("...and again for the copy of the copy", GP.copyName("Writer") === "Writer copy 2");
await GP.refreshStore();

const mine = { key: "Writer", prompt: "", grammar: "", builtin: false, isNew: false };
const builtin = { key: "Character Card (JSON)", prompt: "", grammar: "", builtin: true, isNew: false };
check("a plain edit sends no old_name", GP.editorPayload(mine, { key: "Writer", prompt: "p", grammar: "" }).old_name === undefined);
let body = GP.editorPayload(mine, { key: "  Novelist  ", prompt: "p", grammar: "g" });
check("a changed name renames your preset", body.old_name === "Writer" && body.name === "Novelist", JSON.stringify(body));
body = GP.editorPayload(builtin, { key: "My card", prompt: "p", grammar: "g" });
check("...but a renamed built-in is a copy beside it, never an old_name that would reset it",
      body.old_name === undefined && body.name === "My card", JSON.stringify(body));
body = GP.editorPayload({ ...mine, key: "Writer copy", isNew: true }, { key: "Writer copy", prompt: "p", grammar: "" });
check("a duplicate never renames its original", body.old_name === undefined);

check("creating onto a name the library holds is caught", GP.nameTaken({ isNew: true, key: "" }, "Critic") === true);
check("...and so is renaming onto one", GP.nameTaken(mine, "Critic") === true);
check("...but saving a preset under its own name is not a clash", GP.nameTaken(mine, "Writer") === false);
check("...nor is saving a built-in under its own name — that is the edit over it", GP.nameTaken(builtin, "Character Card (JSON)") === false);
check("...nor is a fresh name", GP.nameTaken(mine, "Novelist") === false);

// ---- rename follows the graph -----------------------------------------------------------------
const mkNode = (value) => ({ comfyClass: "GrammarPresets", setDirtyCanvas() {},
                             widgets: [{ name: "preset", value, options: { values: [] } }] });
const picked = mkNode("Writer");
const other = mkNode("Critic");
const foreign = { comfyClass: "CardPresets", setDirtyCanvas() {}, widgets: [{ name: "preset", value: "Writer" }] };
app.graph._nodes = [picked, other, foreign];
check("a rename re-points every node that picked the preset", GP.renameInGraph("Writer", "Novelist") === 1);
check("...to the new name", picked.widgets[0].value === "Novelist");
check("...which the dropdown keeps offering even before the store has it",
      picked.widgets[0].options.values.includes("Novelist"));
check("...leaving the other presets alone", other.widgets[0].value === "Critic");
check("...and never touching another node's preset widget", foreign.widgets[0].value === "Writer");
picked.widgets[0].value = "Writer";

// ---- the editor, end to end -------------------------------------------------------------------
GP.presetEditor({ key: "Writer", ...GP.lib().Writer, builtin: false, isNew: false });
let box = lastBox();
check("the editor opens on the preset's own texts",
      field(box, "Prompt").value === "write ===PROMPT===" && field(box, "GBNF grammar").value === "root ::= prompt");
check("...filed under its library name", field(box, "Saved as").value === "Writer");
field(box, "Prompt").value = "write ===PROMPT=== and ===NOTES===";
field(box, "Saved as").value = "Novelist";
await btn(box, "Save").onclick();
let sent = posts().pop();
check("saving posts both fields", sent.body.prompt.includes("===NOTES===") && sent.body.grammar === "root ::= prompt");
check("...as a rename", sent.body.old_name === "Writer");
check("...the store holds it under the new name only", !!GP.lib().Novelist && !GP.lib().Writer, Object.keys(GP.lib()).join(","));
check("...and the node that picked it followed", picked.widgets[0].value === "Novelist");

// a new preset onto a taken name never reaches the store
const before = posts().length;
GP.presetEditor({ key: "", prompt: "", grammar: "", builtin: false, isNew: true });
box = lastBox();
field(box, "Saved as").value = "Critic";
field(box, "Prompt").value = "overwrite me";
await btn(box, "Create").onclick();
check("creating onto a taken name posts nothing", posts().length === before);
check("...and says why on the button", String(btn(box, "name taken")?.textContent || "").includes("name taken"));
check("...leaving the original intact", GP.lib().Critic.prompt === "be harsh");

// a grammar alone is a whole preset
field(box, "Saved as").value = "Sections";
field(box, "Prompt").value = "";
field(box, "GBNF grammar").value = "root ::= sections";
await btn(box, "name taken").onclick();
check("a grammar with no prompt saves", GP.lib().Sections && GP.lib().Sections.grammar === "root ::= sections");

// a built-in edited under its own name is the edit over it, not a rename
GP.presetEditor({ key: "Siren Voice Plan (table)", prompt: "", grammar: "root ::= row+", builtin: true, isNew: false });
box = lastBox();
field(box, "Prompt").value = "plan the voices";
await btn(box, "Save").onclick();
sent = posts().pop();
check("editing a built-in posts no old_name", sent.body.old_name === undefined && sent.body.name === "Siren Voice Plan (table)");
check("...and the list marks it edited", GP.getStore().presets["Siren Voice Plan (table)"].edited === true);

// ---- the library dialog -----------------------------------------------------------------------
await GP.libraryDialog("Critic");
box = lastBox();
const rowCard = rowOf(box, "Character Card (JSON)");
check("an edited built-in offers reset instead of delete",
      !!btn(rowCard, "↺") && !btn(rowCard, "\u{1F5D1}"), allOf(rowCard, "BUTTON").map((b) => b.textContent).join(" "));
const rowCritic = rowOf(box, "Critic");
const del = btn(rowCritic, "\u{1F5D1}");
const n = posts().length;
del.onclick();
check("the first delete click only arms the button", del.textContent === "sure?" && posts().length === n, del.textContent);
del.onclick();
await tick();
check("...the second one deletes", posts().pop().body.delete === true && !GP.lib().Critic);

const reset = btn(rowOf(lastBox(), "Character Card (JSON)"), "↺");
reset.onclick();
reset.onclick();
await tick();
check("reset drops the edit, and the built-in is back to pristine",
      !GP.lib()["Character Card (JSON)"] && GP.getStore().presets["Character Card (JSON)"].edited === false);
const rowPristine = rowOf(lastBox(), "Character Card (JSON)");
check("...after which it offers neither delete nor reset",
      !btn(rowPristine, "↺") && !btn(rowPristine, "\u{1F5D1}"),
      allOf(rowPristine, "BUTTON").map((b) => b.textContent).join(" "));

console.log("\n" + (fails.length ? "FAILED: " + fails.join(", ") : "ALL PASS"));
process.exit(fails.length ? 1 : 0);
`;

fs.writeFileSync(OUT, STUBS + EXTRA + strip(WEB + "grammar_presets.js") + EXPOSE + TESTS, "utf8");
console.log("wrote " + path.basename(OUT));
