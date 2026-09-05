import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STUBS } from "./stubs.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "..", "web") + path.sep;
const OUT = path.join(HERE, "run_card_presets.mjs");
// Harness for web/card_presets.js — the Card library dialog. What it keeps honest:
//
//   * RENAME. It is the one edit in there that reaches outside the store: a Card Presets node
//     holds the preset by NAME, so a rename that doesn't follow the graph leaves the node quietly
//     emitting an empty block on the next run.
//   * the values merge. The editor draws the fields of ONE card type, and a card carries the keys
//     of both after a type switch — a save that posted only the drawn fields would drop the rest.
//   * the two-click delete, and that a preview render never overwrites the cached library.
//
// The backend is stubbed as the real routes behave (including 404 for an unknown card), so the
// dialog is driven end to end: open → edit → save → the store the rows re-render from.

const strip = (p) => fs.readFileSync(p, "utf8")
  .split("\n").filter((l) => !/^import\s/.test(l)).join("\n")
  .replace(/^export\s+(function|const|async function)/gm, "$1");

const EXPOSE = `
globalThis.CP = { presetsForTag, libraryRows, copyName, editorPayload, renameInGraph, syncReader,
                  cardEditor, libraryDialog, fieldsFor, refreshStore,
                  setStore: (s) => { STORE = s; }, getStore: () => STORE,
                  posted: () => POSTED, cards: () => CARDS, boxes: () => BOXES };
`;

// A backend that behaves like card_presets/routes.py, over an in-memory library.
const EXTRA = String.raw`
globalThis.alert = () => {};
const BOXES = [];
document.body.appendChild = (el) => { BOXES.push(el); return el; };

const SCHEMA = {
  character: [{ key: "name", label: "Name" }, { key: "gender", label: "Gender" },
              { key: "eye_color", label: "Eyes" }, { key: "voice_tags", label: "Voice (music tags)" },
              { key: "notes", label: "Notes", multiline: true }],
  entity: [{ key: "name", label: "Name" }, { key: "description", label: "Description", multiline: true }],
};
let CARDS = {
  Vasya: { type: "character", tags: ["heroes"],
           values: { name: "Vasya", gender: "male", eye_color: "brown", notes: "quiet" } },
  Zoya:  { type: "character", tags: ["heroes", "band"],
           values: { name: "Zoya", gender: "female", voice_tags: "female lead vocal" } },
  Cafe:  { type: "entity", tags: ["places"],
           values: { name: "Cafe", description: "bronze pitchers" } },
};
const storeData = () => ({
  ok: true, none: "\u{1F6AB} None", all_tags: "\u{1F3F7} All",
  order: ["\u{1F6AB} None", ...Object.keys(CARDS).sort()],
  tags: [...new Set(Object.values(CARDS).flatMap((c) => c.tags))].sort(),
  presets: Object.fromEntries(Object.entries(CARDS).map(([k, v]) => [k, { type: v.type, tags: v.tags }])),
  schema: SCHEMA,
});

const POSTED = [];
const reply = (j) => ({ status: 200, json: async () => j });
api.fetchApi = async (route, opt) => {
  if (opt && opt.method === "POST") {
    const body = JSON.parse((opt && opt.body) || "{}");
    POSTED.push({ path: route, body });
    if (route === "/kinburg/cards/render") {
      const v = body.values || {};
      const bits = Object.keys(v).filter((k) => k !== "name" && v[k]).map((k) => "- " + k + ": " + v[k]);
      return reply({ ok: true, card: bits.length ? ["### " + (v.name || ""), ...bits].join("\n") : "" });
    }
    if (route === "/kinburg/cards/tags") {
      if (CARDS[body.name]) CARDS[body.name].tags = String(body.tags).split(",").map((t) => t.trim()).filter(Boolean);
      return reply(storeData());
    }
    if (route === "/kinburg/cards/save") {
      if (body.delete) delete CARDS[body.name];
      else {
        const prev = CARDS[body.name] || (body.old_name ? CARDS[body.old_name] : null) || {};
        if (body.old_name && body.old_name !== body.name) delete CARDS[body.old_name];
        const tags = String(body.tags || "").split(",").map((t) => t.trim()).filter(Boolean);
        CARDS[body.name] = { type: body.type, values: body.values,
                             tags: tags.length ? tags : (prev.tags || []) };
      }
      return reply(storeData());
    }
    return reply({ ok: true });
  }
  if (route.startsWith("/kinburg/cards/preset")) {
    const nm = decodeURIComponent(route.split("name=")[1] || "");
    const c = CARDS[nm];
    return reply(c ? { ok: true, name: nm, type: c.type, tags: c.tags, values: { ...c.values } }
                   : { ok: false, error: "no saved card named '" + nm + "'" });
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
const CP = globalThis.CP;      // tick() comes from the shared stubs

// ---- DOM spelunking: the dialogs build their own trees, nothing here has ids ------------------
const walk = (el, fn) => { fn(el); for (const c of el.children || []) walk(c, fn); };
const allOf = (root, tag) => { const out = []; walk(root, (e) => { if (e.tagName === tag) out.push(e); }); return out; };
const btn = (root, text) => allOf(root, "BUTTON").find((b) => String(b.textContent).includes(text));
// The control sitting next to a labelled row ("Saved as", "Eyes", …).
const field = (root, label) => {
  let hit = null;
  walk(root, (e) => {
    if (hit) return;
    const [l, c] = e.children || [];
    if (l && c && String(l.textContent) === label) hit = c;
  });
  return hit;
};
const lastBox = () => { const b = CP.boxes(); return b[b.length - 1].children[0]; };
const postsTo = (p) => CP.posted().filter((x) => x.path === p);

await CP.refreshStore();

// ---- the pure parts ---------------------------------------------------------------------------
check("the tag filter still narrows the reader's dropdown",
      JSON.stringify(CP.presetsForTag("band")) === JSON.stringify(["\u{1F6AB} None", "Zoya"]),
      JSON.stringify(CP.presetsForTag("band")));

check("the library lists every card A-Z by default",
      JSON.stringify(CP.libraryRows("", "\u{1F3F7} All", null)) === JSON.stringify(["Cafe", "Vasya", "Zoya"]));
check("...the card the node has picked goes first",
      CP.libraryRows("", "\u{1F3F7} All", "Zoya")[0] === "Zoya");
check("...a focus that is filtered out doesn't sneak back in",
      JSON.stringify(CP.libraryRows("", "places", "Zoya")) === JSON.stringify(["Cafe"]));
check("...search matches the name", JSON.stringify(CP.libraryRows("vas", "\u{1F3F7} All")) === JSON.stringify(["Vasya"]));
check("...and the tags too, which is how you find a card you named badly",
      JSON.stringify(CP.libraryRows("band", "\u{1F3F7} All")) === JSON.stringify(["Zoya"]));

check("a duplicate is named beside the original, never onto it", CP.copyName("Vasya") === "Vasya copy");
CP.setStore({ ...CP.getStore(), presets: { ...CP.getStore().presets, "Vasya copy": { type: "character", tags: [] } } });
check("...and again for the copy of the copy", CP.copyName("Vasya") === "Vasya copy 2");
await CP.refreshStore();

// ---- the save payload ---------------------------------------------------------------------
const entry = { key: "Vasya", type: "character", tags: ["heroes"],
                values: { name: "Vasya", gender: "male", description: "left over from an entity" },
                isNew: false };
let body = CP.editorPayload(entry, { key: "Vasya", type: "character", tags: "heroes",
                                     edited: { name: "Vasya", gender: "female" } });
check("a plain edit sends no old_name", body.old_name === undefined);
check("...the edited field wins", body.values.gender === "female");
check("...and a field the editor never drew survives the save",
      body.values.description === "left over from an entity");

body = CP.editorPayload(entry, { key: "  Vasiliy  ", type: "character", tags: "", edited: {} });
check("a changed name renames rather than forking the card", body.old_name === "Vasya" && body.name === "Vasiliy",
      JSON.stringify(body));
body = CP.editorPayload({ ...entry, key: "Vasya copy", isNew: true }, { key: "Vasya copy", type: "entity", tags: "", edited: {} });
check("a duplicate never renames its original", body.old_name === undefined && body.type === "entity");

// ---- rename follows the graph -----------------------------------------------------------------
const mkNode = (value) => ({
  comfyClass: "CardPresets", setDirtyCanvas() {},
  widgets: [{ name: "preset", value, options: { values: [] } },
            { name: "filter", value: "\u{1F3F7} All", options: { values: [] } }],
});
const picked = mkNode("Vasya");
const other = mkNode("Cafe");
const foreign = { comfyClass: "KSampler", setDirtyCanvas() {}, widgets: [{ name: "preset", value: "Vasya" }] };
app.graph._nodes = [picked, other, foreign];
check("a rename re-points every node that picked the card", CP.renameInGraph("Vasya", "Vasiliy") === 1);
check("...to the new name", picked.widgets[0].value === "Vasiliy");
check("...which the dropdown then offers", picked.widgets[0].options.values.includes("Vasiliy"));
check("...leaving the other cards alone", other.widgets[0].value === "Cafe");
check("...and never touching a node that only happens to have a preset widget of its own",
      foreign.widgets[0].value === "Vasya");

// ---- the editor, end to end -------------------------------------------------------------------
app.graph._nodes = [];
CP.cardEditor({ key: "Vasya", type: "character", tags: ["heroes"],
                values: { ...CP.cards().Vasya.values }, isNew: false });
let box = lastBox();
check("the editor opens on the card's own values", field(box, "Eyes").value === "brown");
check("...with the library name it is filed under", field(box, "Saved as").value === "Vasya");
check("...and its tags", field(box, "Tags").value === "heroes");
await tick();
check("...previewing the block the node will emit, before anything is saved",
      (allOf(box, "PRE")[0].textContent || "").includes("- eye_color: brown"),
      JSON.stringify(allOf(box, "PRE")[0].textContent));

field(box, "Eyes").value = "green";
field(box, "Saved as").value = "Vasiliy";
await btn(box, "Save").onclick();
let saved = postsTo("/kinburg/cards/save").pop();
check("saving posts the edit", saved.body.values.eye_color === "green");
check("...as a rename, since the library name changed", saved.body.old_name === "Vasya");
check("...and the store now holds it under the new name",
      !!CP.cards().Vasiliy && !CP.cards().Vasya, Object.keys(CP.cards()).join(","));
check("...keeping the fields the editor never showed", CP.cards().Vasiliy.values.notes === "quiet");

// a new card takes its library name from the Name field until you type one yourself
CP.cardEditor({ key: "", type: "character", tags: "", values: {}, isNew: true });
box = lastBox();
const nameCtl = field(box, "Name");
nameCtl.value = "Grisha";
nameCtl.fire("input");
check("a new card files itself under the name you type", field(box, "Saved as").value === "Grisha");
field(box, "Saved as").value = "grisha-v2";
field(box, "Saved as").fire("input");
nameCtl.value = "Grisha Ivanov";
nameCtl.fire("input");
check("...but stops once you name it yourself", field(box, "Saved as").value === "grisha-v2");
field(box, "Gender").value = "male";
await btn(box, "Create").onclick();
saved = postsTo("/kinburg/cards/save").pop();
check("...and creating never sends a rename", saved.body.old_name === undefined && saved.body.name === "grisha-v2");
check("...with the card's own name kept apart from the library key",
      CP.cards()["grisha-v2"].values.name === "Grisha Ivanov");

// the name alone is not a card: the preview says so rather than showing a bare heading
CP.cardEditor({ key: "", type: "character", tags: "", values: { name: "Empty" }, isNew: true });
await tick();
check("an all-but-empty card previews as nothing, not as a heading",
      allOf(lastBox(), "PRE")[0].textContent.startsWith("(nothing"),
      allOf(lastBox(), "PRE")[0].textContent);

// a preview is not a store read — it must not blank the cached library
const before = Object.keys(CP.getStore().presets || {}).length;
check("rendering a preview leaves the cached library intact", before > 0, before);

// ---- the library dialog -----------------------------------------------------------------------
await CP.libraryDialog("Cafe");
box = lastBox();
check("the dialog lists a row per card", allOf(box, "INPUT").length >= Object.keys(CP.cards()).length);
const del = allOf(box, "BUTTON").find((b) => b.textContent === "\u{1F5D1}");
const postsBefore = postsTo("/kinburg/cards/save").length;
del.onclick();
check("the first delete click only arms the button",
      del.textContent === "sure?" && postsTo("/kinburg/cards/save").length === postsBefore, del.textContent);
del.onclick();
await tick();
check("...the second one actually deletes",
      (postsTo("/kinburg/cards/save").pop() || {}).body.delete === true);

console.log("\n" + (fails.length ? "FAILED: " + fails.join(", ") : "ALL PASS"));
process.exit(fails.length ? 1 : 0);
`;

fs.writeFileSync(OUT, STUBS + EXTRA + strip(WEB + "card_presets.js") + EXPOSE + TESTS, "utf8");
console.log("wrote " + path.basename(OUT));
