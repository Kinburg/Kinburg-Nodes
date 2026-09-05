import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STUBS } from "./stubs.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "..", "web") + path.sep;
const OUT = path.join(HERE, "run_model_presets.mjs");
// Harness for web/model_presets.js. Two things it keeps honest:
//
//   * the saved-workflow migration, which rewrites the user's own files and therefore has to be
//     exactly right. Model Select's outputs were re-ordered and its width/height dropped, and
//     Settings Select lost the same pair; a workflow stores links by SLOT INDEX, so every wire off
//     an older node has to be re-pointed by NAME or it silently lands on a wrongly-typed output.
//   * the Library dialog's two sections that exist so nothing in the store is unreachable — shared
//     presets (listed whether a model matches them or not) and families (renamed / deleted across
//     every holder at once, since a family is stored nowhere on its own).

const strip = (p) => fs.readFileSync(p, "utf8")
  .split("\n").filter((l) => !/^import\s/.test(l)).join("\n")
  .replace(/^export\s+(function|const|async function)/gm, "$1");

const EXPOSE = `
globalThis.MP = { migrateSelectNodes, nodeIsCurrent, MIGRATIONS, SELECT_OUTPUTS, SETTINGS_OUTPUTS,
                  SELECT, SETTINGS, orderModelIds, effectiveModel, presetRow, sharedSection,
                  familySection, setStore: (s) => { STORE = s; }, posted: () => POSTED };
`;

// postJSON goes through api.fetchApi; record the bodies so a click can be checked for what it sends.
const EXTRA = `
const POSTED = [];
api.fetchApi = async (path, opts) => {
  POSTED.push({ path, body: JSON.parse((opts && opts.body) || "{}") });
  return { json: async () => ({ ok: true }) };
};
`;

const TESTS = String.raw`
const fails = [];
const check = (label, cond, extra) => {
  console.log((cond ? "  ok   " : "  FAIL ") + label + (extra !== undefined ? "  " + extra : ""));
  if (!cond) fails.push(label);
};
const MP = globalThis.MP;

// The output layout Model Select shipped with, straight out of a workflow saved by it.
const OLD = ["model", "model_negative", "clip", "vae", "sampler_settings", "width", "height",
             "info", "gen_extra_info", "model_id"];
// …and the one it had for the few minutes conditioning was appended rather than ordered.
const MID = [...OLD, "prompt", "triggers", "positive", "negative"];
// Settings Select's own old layout.
const OLD_SET = ["sampler_settings", "width", "height", "label", "info", "gen_extra_info"];

// [id, origin_node, origin_slot, target_node, target_slot, type]
const link = (id, slot, type) => [id, 47, slot, 90, 0, type];

function graph(names, links, widgets, type) {
  const byId = new Map();
  for (const l of links) {
    if (!byId.has(l[2])) byId.set(l[2], []);
    byId.get(l[2]).push(l[0]);
  }
  return {
    links,
    nodes: [
      { id: 47, type: type || MP.SELECT,
        widgets_values: widgets || ["krea2", "Krea2_turbo_bf16", "\u{1F6AB} None", -1, 1024, 1024,
                                    true, null, null],
        outputs: names.map((name, i) => ({ name, type: "X", links: byId.get(i) || null })) },
      { id: 90, type: "KSampler", inputs: links.map((l) => ({ name: "in" + l[0], link: l[0] })) },
    ],
  };
}

const slotOf = (name) => MP.SELECT_OUTPUTS.indexOf(name);

// -- the output remap -------------------------------------------------------------------------
let g = graph(OLD, [link(1, 0, "MODEL"), link(2, 2, "CLIP"), link(3, 4, "SAMPLER_CFG"),
                    link(4, 9, "STRING")]);
let n = MP.migrateSelectNodes(g);
check("an old Model Select is migrated", n === 1, n);
check("model stays at 0", g.links[0][2] === slotOf("model"), g.links[0][2]);
check("clip moves 2 -> 4", g.links[1][2] === slotOf("clip"), g.links[1][2]);
check("sampler_settings moves 4 -> 6", g.links[2][2] === slotOf("sampler_settings"), g.links[2][2]);
check("model_id moves 9 -> 11", g.links[3][2] === slotOf("model_id"), g.links[3][2]);
const outs = g.nodes[0].outputs.map((o) => o.name);
check("the slots are rebuilt in the new order",
      outs.join(",") === MP.SELECT_OUTPUTS.join(","), outs.join(","));
check("...and each keeps its own wire",
      JSON.stringify(g.nodes[0].outputs[slotOf("clip")].links) === "[2]",
      JSON.stringify(g.nodes[0].outputs[slotOf("clip")].links));
check("...and the empty ones stay empty",
      (g.nodes[0].outputs[slotOf("positive")].links || []).length === 0);
check("the slots carry their real types now",
      g.nodes[0].outputs[slotOf("positive")].type === "CONDITIONING",
      g.nodes[0].outputs[slotOf("positive")].type);

// -- width / height wires have nowhere to go and must be removed everywhere ---------------------
g = graph(OLD, [link(1, 0, "MODEL"), link(7, 5, "INT"), link(8, 6, "INT"), link(9, 7, "STRING")]);
MP.migrateSelectNodes(g);
check("a width wire is dropped from the link table",
      g.links.map((l) => l[0]).join(",") === "1,9", g.links.map((l) => l[0]).join(","));
check("...and from the target node's input",
      g.nodes[1].inputs.filter((i) => i.link !== null).map((i) => i.link).join(",") === "1,9",
      JSON.stringify(g.nodes[1].inputs));
check("...and no output slot still claims it",
      !JSON.stringify(g.nodes[0].outputs).includes('"links":[7'), JSON.stringify(g.nodes[0].outputs));
check("the surviving info wire moved 7 -> 9", g.links[1][2] === slotOf("info"), g.links[1][2]);

// -- the intermediate layout (conditioning appended) --------------------------------------------
g = graph(MID, [link(1, 12, "CONDITIONING"), link(2, 13, "CONDITIONING"), link(3, 10, "STRING")]);
MP.migrateSelectNodes(g);
check("positive moves 12 -> 2", g.links[0][2] === slotOf("positive"), g.links[0][2]);
check("negative moves 13 -> 3", g.links[1][2] === slotOf("negative"), g.links[1][2]);
check("prompt moves 10 -> 7", g.links[2][2] === slotOf("prompt"), g.links[2][2]);

// -- Settings Select lost the same pair, in different company -----------------------------------
const sSlot = (name) => MP.SETTINGS_OUTPUTS.indexOf(name);
g = graph(OLD_SET, [link(1, 0, "SAMPLER_CFG"), link(2, 1, "INT"), link(3, 3, "STRING"),
                    link(4, 5, "GEN_INFO")],
          ["krea2_real", "\u{1F6AB} None", "\u{1F6AB} None", -1, 1024, 1024, null],
          MP.SETTINGS);
MP.migrateSelectNodes(g);
check("Settings Select is migrated too",
      g.nodes[0].outputs.map((o) => o.name).join(",") === MP.SETTINGS_OUTPUTS.join(","),
      g.nodes[0].outputs.map((o) => o.name).join(","));
check("sampler_settings stays at 0", g.links[0][2] === sSlot("sampler_settings"), g.links[0][2]);
check("label moves 3 -> 1", g.links[1][2] === sSlot("label"), g.links[1][2]);
check("gen_extra_info moves 5 -> 3", g.links[2][2] === sSlot("gen_extra_info"), g.links[2][2]);
check("its width wire is gone", g.links.length === 3, g.links.map((l) => l[0]).join(","));
check("its widgets lose the two sizes — anchored on the last number, it has no boolean",
      JSON.stringify(g.nodes[0].widgets_values)
      === JSON.stringify(["krea2_real", "\u{1F6AB} None", "\u{1F6AB} None", -1, null]),
      JSON.stringify(g.nodes[0].widgets_values));

// -- widgets ------------------------------------------------------------------------------------
g = graph(OLD, []);
MP.migrateSelectNodes(g);
check("width/height leave widgets_values, anchored on the boolean",
      JSON.stringify(g.nodes[0].widgets_values)
      === JSON.stringify(["krea2", "Krea2_turbo_bf16", "\u{1F6AB} None", -1, true, null, null]),
      JSON.stringify(g.nodes[0].widgets_values));
// A file written before the 🏷 filter existed has no leading family value — anchoring from the end
// is what makes both shapes work without knowing which one this is.
g = graph(OLD, [], ["Krea2_turbo_bf16", "\u{1F6AB} None", -1, 1024, 1024, false]);
MP.migrateSelectNodes(g);
check("...with or without the family filter in front",
      JSON.stringify(g.nodes[0].widgets_values)
      === JSON.stringify(["Krea2_turbo_bf16", "\u{1F6AB} None", -1, false]),
      JSON.stringify(g.nodes[0].widgets_values));
check("...and unload_others keeps the value it was saved with",
      g.nodes[0].widgets_values[3] === false);

// -- idempotence: the one property a migration must have ----------------------------------------
g = graph(OLD, [link(1, 0, "MODEL"), link(2, 2, "CLIP"), link(7, 5, "INT")]);
MP.migrateSelectNodes(g);
const once = JSON.stringify(g);
check("a second pass changes nothing", MP.migrateSelectNodes(g) === 0 && JSON.stringify(g) === once);
check("...because a current node is recognised as current",
      MP.nodeIsCurrent(g.nodes[0], MP.MIGRATIONS[MP.SELECT]));

// -- everything else is left alone ---------------------------------------------------------------
g = { links: [[1, 5, 3, 6, 0, "MODEL"]],
      nodes: [{ id: 5, type: "CheckpointLoaderSimple", widgets_values: ["a.safetensors", 1024, 1024, true],
                outputs: [{ name: "MODEL", links: [1] }] }] };
const before = JSON.stringify(g);
check("a graph with no library node is untouched",
      MP.migrateSelectNodes(g) === 0 && JSON.stringify(g) === before);
check("a graph with no links at all does not throw",
      MP.migrateSelectNodes({ nodes: [{ id: 1, type: MP.SELECT, outputs: [{ name: "model" }] }] }) === 1);
check("an empty payload does not throw", MP.migrateSelectNodes({}) === 0);
check("undefined does not throw", MP.migrateSelectNodes(undefined) === 0);

// -- the Library dialog's card order ------------------------------------------------------------
const IDS = ["Krea2_turbo", "Flux2_dev", "ZIT_turbo"];
check("cards are alphabetical with no model in effect",
      MP.orderModelIds(IDS).join(",") === "Flux2_dev,Krea2_turbo,ZIT_turbo",
      MP.orderModelIds(IDS).join(","));
check("the node's own model goes on top",
      MP.orderModelIds(IDS, "ZIT_turbo").join(",") === "ZIT_turbo,Flux2_dev,Krea2_turbo",
      MP.orderModelIds(IDS, "ZIT_turbo").join(","));
check("...and nothing is filtered out", MP.orderModelIds(IDS, "ZIT_turbo").length === 3);
check("a model that is not in the library is ignored",
      MP.orderModelIds(IDS, "gone").join(",") === "Flux2_dev,Krea2_turbo,ZIT_turbo");
check("an empty library is still an empty list", MP.orderModelIds([], "x").length === 0);

// -- which model a preset node is working with ---------------------------------------------------
const settings = (val, wiredTo) => ({
  inputs: [{ name: "model_id", link: wiredTo ? 5 : null }],
  widgets: [{ name: "model", value: val }],
  getInputNode: () => (wiredTo ? { widgets: [{ name: "model", value: wiredTo }] } : null),
});
check("the dropdown is used when nothing is wired",
      MP.effectiveModel(settings("Krea2_turbo")) === "Krea2_turbo");
check("a wired model_id wins over the dropdown",
      MP.effectiveModel(settings("Krea2_turbo", "ZIT_turbo")) === "ZIT_turbo");
check("'None' is no model at all",
      MP.effectiveModel(settings("\u{1F6AB} None")) === null,
      MP.effectiveModel(settings("\u{1F6AB} None")));

// -- shared presets and families: nothing in the store may be unreachable -----------------------
// The state that exposed the gap: no models at all, so no model card, so the shared preset and the
// family it declares could be neither edited nor deleted from anywhere.
MP.setStore({
  none: "\u{1F6AB} None", order: ["\u{1F6AB} None"], models: {},
  families: ["krea2_real"],
  family_usage: { krea2_real: { models: [], shared: ["Krea2_seeds2"] } },
  shared: {
    Krea2_seeds2: { families: ["krea2_real"], tags: ["phi"], stages: 1, overrides: 0,
                    override_map: {}, score: null, seconds: null, default: false },
    Nofamily: { families: [], tags: [], stages: 1, overrides: 0, override_map: {},
                score: null, seconds: null, default: false },
  },
});
const textOf = (el) => {
  const out = [];
  (function walk(e) {
    if (e.textContent) out.push(e.textContent);
    if (e.placeholder) out.push(e.placeholder);
    for (const c of e.children || []) walk(c);
  })(el);
  return out.join(" | ");
};

let box = document.createElement("div");
MP.sharedSection(box, () => {});
let txt = textOf(box);
check("shared presets are listed with no models in the library at all",
      txt.includes("Shared presets") && txt.includes("Krea2_seeds2"), txt.slice(0, 80));
check("...including one no model can see", txt.includes("Nofamily"));
check("...which is called out rather than left looking fine",
      txt.includes("no families"), txt.slice(0, 200));
check("...and each row offers its families for editing", txt.includes("families…"));

box = document.createElement("div");
MP.familySection(box, () => {});
txt = textOf(box);
check("families get a section of their own", txt.includes("Families") && txt.includes("krea2_real"));
check("...saying what declares each one", txt.includes("1 shared preset(s)"), txt.slice(0, 240));
check("...with a rename field", txt.includes("rename to…"));

// A delete is armed, never one click — and it says what it is about to touch.
const findBtn = (el, label) => {
  let hit = null;
  (function walk(e) {
    if (e.textContent === label) hit = e;
    for (const c of e.children || []) walk(c);
  })(el);
  return hit;
};
const del = findBtn(box, "Delete");
check("the family delete button exists", !!del);
del.onclick?.();
check("...and the first click only arms it, sending nothing",
      del.textContent.includes("sure?") && MP.posted().length === 0, del.textContent);
del.onclick?.();
const last = MP.posted().pop() || {};
check("...the second click posts the delete for that family",
      last.path === "/kinburg/models/family" && last.body.name === "krea2_real"
      && last.body.delete === true, JSON.stringify(last));

const shBox = document.createElement("div");
MP.sharedSection(shBox, () => {});
const sdel = findBtn(shBox, "✕");
sdel.onclick?.();
const sposted = MP.posted().pop() || {};
check("a shared preset can be deleted straight from its own row",
      sposted.path === "/kinburg/models/preset" && sposted.body.shared === true
      && sposted.body.delete === true, JSON.stringify(sposted));

console.log("\n" + (fails.length ? "FAILED: " + fails.join(", ") : "ALL PASS"));
process.exit(fails.length ? 1 : 0);
`;

fs.writeFileSync(OUT, STUBS + EXTRA + strip(WEB + "model_presets.js") + EXPOSE + TESTS, "utf8");
console.log("wrote " + path.basename(OUT));
