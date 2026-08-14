# 🛠️ Workflow Control & Utilities

[← back to the node index](../README.md#-node-index)

---

## 🛠️ `util/` — General Workflow Utilities

> **System Purpose & Overview**  
> Essential general-purpose utility nodes: Date string, dynamic string concat, color picker, type converters, text transform, routing switches, and JSON extraction.

**`Date String`** appends the current **date** (and optionally **time**) to a string, with
selectable formats. Handy for building save paths: e.g. `project/2026-06-20` (a folder per
day) or `.../2026-06-20/17-05` (per minute). The `/` separator creates subfolders; `_`/`-`
make a flat name. Presets plus a `custom` strftime field. The node is re-evaluated on every
run so the date never freezes from caching. Category `Kinburg-Nodes/util`.

**`Unlim Text Concat`** joins an unlimited number of string inputs with a `separator`
(multi-line allowed; default is a newline). The input list grows on its own — it starts
with `text_1` (required) and `text_2`, and each time you connect the last slot a new empty
one appears (disconnect and trailing empties collapse). `skip_empty` drops empty/unconnected
inputs so they don't leave stray separators. Pairs naturally with **Color Caption** →
Concat (newline) → the compare node's `captions`. Category `Kinburg-Nodes/util`.

**`Color Picker`** is the handy color control from Color Caption (a 10-swatch palette + a
native color picker, or type a HEX) as a standalone node. Outputs the normalized `#RRGGBB`
string plus its `R` / `G` / `B` components. Category `Kinburg-Nodes/util`.

**`Any to String`** / **`Combo to String`** convert a value into a real `STRING` you can feed
into text/preview/prompt nodes. **`Any to String`** has a wildcard `*` input for values that
already connect loosely. **`Combo to String`** solves the specific case ComfyUI blocks by
design: a **COMBO** output (e.g. the core `Primitive` node in combo mode) refuses to wire into
a `STRING` input. Connect the Primitive's COMBO output into `Combo to String` (a small frontend
patch in `web/combo_to_string.js` allows the link and pushes the selected value in), and its
`string` output carries the value — e.g. `ru` — wherever you need it. Category
`Kinburg-Nodes/util`.

**`Text Transform`** does string find/replace, regex, trim and case in one node —
`replace`, `regex_replace`, `regex_extract`, `regex_findall`, `strip`, `collapse_whitespace`,
`lower`/`upper`/`title`, with `ignorecase` / `multiline` / `dotall` flags. The regex is
**validated**: a bad pattern (or replacement backreference) is reported on the **`error`** output
while the input text passes through unchanged, so a typo never crashes the run. Outputs `text` /
`count` / `error`. Category `Kinburg-Nodes/util`.

**`Any Switch`** forwards one of several inputs, chosen by `select` (1-based). The `input_*`
slots take any type and auto-grow (the node shows the connected ones plus a spare); outputs the
selected `output` and its `selected` slot number. It routes a value — ComfyUI still computes the
other branches (it's not a lazy gate). Category `Kinburg-Nodes/util`.

**`JSON Extract`** pulls fields out of a JSON string by path into separate `STRING` outputs. You
write **one path per line** in the `paths` field (each line either a bare path or `path -> alias`;
blank lines and `# comments` are ignored) and the node grows one **auto-labelled `value_*` output
per line** — named by the alias, or the path's last key — so the graph is self-documenting; no
fixed slot count and no more guessing which `value_3` is which. Outputs are rebuilt **when you
click away from the field**, not while you type, so they appear complete and ready to wire. Don't want to type paths? Hit
**🔍 Explore JSON** to paste a sample (or reuse the node's last run), then click ＋ on any field to
add its path — arrays also offer a `[*]` button that grabs every element. A **live preview** on the
node shows the extracted values after each run. Outputs are `found` (True when the JSON parsed and
every non-empty path resolved) and `report` (per-path hit/miss) first, then the labelled values.
Path syntax: dot keys, `[n]` / `.n` indices (negative allowed), **`[*]` / `*` wildcard** (grabs
every element/value and joins them with `array_join`, e.g. `elements[*].desc`), `$` (or empty) for
the whole document; an object/array value comes out as compact JSON, and prose-wrapped JSON is
tolerated (the first `{…}`/`[…]` is parsed). Missing paths return `default`. Up to 12 paths map to
outputs (extras are noted in `report`). Pairs with the structured-output LLM nodes (e.g.
`ideogram4_json`) to route sub-fields into different prompt inputs. Category `Kinburg-Nodes/util`.

---

## ⏱️ `timer/` — Execution Timer

> **System Purpose & Overview**  
> Execution timing profile nodes for workflow benchmarking.

**`Start Timer`** / **`Stop Timer`** measure the wall-clock time of a slice of a workflow.
Any value (MODEL, LATENT, IMAGE, …) passes through unchanged — wire Start at the beginning of
the slice and Stop at the end, and that data dependency forces ComfyUI to run Start → slice →
Stop in order. **Start Timer has unlimited inputs** (`any_1` passes through; the rest are
dependency taps that grow as you connect): it starts only after *all* connected inputs are
ready, so tap every branch feeding your sampler (noise / guider / sampler / sigmas / latent)
to start the clock right before the sampler runs, not as soon as one branch resolves. Start also outputs the start time as epoch
seconds (feed it into Stop) and as a formatted string; Stop outputs the `elapsed` time
(format: `auto` / `seconds` / `milliseconds` / `HH:MM:SS` / `human`) and the raw seconds —
wire `elapsed` to any text preview to see it. Both nodes always re-execute (so the timing is
real), which means the wrapped slice is recomputed every run while the timers are active —
mute/bypass them when you're not measuring. Category `Kinburg-Nodes/util`.

---

## ℹ️ `gen_info/` — Generation Info & Filter

> **System Purpose & Overview**  
> Generation metadata extraction and metadata filtering nodes.

**`Generation Info`** lists the settings of the branch that produced an output. Pass your
**`LATENT`** through its `passthrough` slot (tap it downstream of where your branches
converge — e.g. the sampler's latent output — so the upstream walk reaches them all);
the node reads ComfyUI's hidden `PROMPT`, walks upstream, and lists the upstream nodes'
widget values (`[RandomNoise] noise_seed: …`, `[KSamplerSelect] sampler_name: …`, etc.). The
dump shows on the node, collapsed by default — click to expand. Outputs a human dump (`info`)
and machine-readable `data` (GEN_INFO). Category `Kinburg-Nodes/util`.
<br>An optional **`extra`** input takes `GEN_INFO` from a node that reports its **own runtime facts**
— e.g. **Chimera**'s `gen_extra_info`, which knows the resolved step split, the sigma the handoff
landed on and the per-stage times, none of which a walk over widget literals can see. It is merged
**in front of** the walked entries into a **single** dump (per-class `ord`s are renumbered so every
entry stays addressable), so the branch still feeds one `Set Accumulator (gen info)` — one image
downstream, not a phantom extra one. Such an input is an *addition* to a dump: send it to the
accumulator directly and that branch carries none of the shared settings, which makes the Filter's
`differences` mode keep everything (a field present in some branches and absent in others counts as
a difference).

**`Generation Info Filter`** takes a single `data` bundle (`GEN_INFO_LIST`) from a
**Get Accumulator (gen info)** — which collects one Generation Info `data` per branch — and
emits a per-image `settings` string (one block per dump, in accumulator `index` order).
`mode`: `all`, `differences` (only the
fields that vary across the inputs — each block then shows them all, so the images line
up), `custom` (fields named in `custom_fields`, one selector per line: `ClassType`,
`ClassType[n]`, `ClassType[n].param`, `ClassType.param`; `[n]` is the 1-based occurrence),
or `differences + custom`. A `help` output prints the selector syntax. `skip_empty` (on by
default) drops empty dumps so a bypassed branch keeps the blocks aligned.
Outputs: `settings_data`
(`GEN_SETTINGS`, structured per-image `{key, value}` with class-qualified keys) — wire it into
the compare node's **`settings_data`** input, which both renders the settings under each image
(one `[Class] param: value` line per field, toggleable) and stores them by field in a saved
report for filtering/sorting. A plain-text `settings` output is also provided for standalone
use (saving to a file, feeding a text node, etc.). Category `Kinburg-Nodes/util`.

---

## 🎚️ `group_control/` — Group Control 🎚️

> **System Purpose & Overview**  
> Fast bulk bypass and mute controller for ComfyUI node groups.

**`Group Control 🎚️`** is a client-side control panel for enabling/bypassing workflow **groups by
name**. It lists every *unique* group title in the graph, one row each, with a switch that flips
all groups carrying that name between **`ALWAYS`** (active) and **`BYPASS`** (skipped) at once —
so naming three groups `Upscale` and toggling one row turns the whole set off together (the row
shows a `×N` badge for how many groups share the name). "Toggling a group" rewrites the `mode` of
the nodes inside it, exactly like ComfyUI's own *Set Group Nodes to…* menu, so the state is baked
into the target nodes and travels with the workflow — no extra serialization, survives a reload.
**Nesting is supported**: membership is resolved by bounding box, so an outer group automatically
covers the nodes of any group nested inside it, and nested names are shown **indented** by depth.
The list **grows and rebuilds itself** as you add, rename or delete groups (it polls the graph),
and a mixed selection (some nodes in a group active, some not) shows a `MIXED` marker. Each row
also has a **`▶` button that runs *only that group*** — it queues the group's output nodes as
ComfyUI **partial-execution targets**, so just this group and the nodes it depends on run while
every unrelated branch is skipped (the same mechanism as the core *Queue Selected Output Nodes*).
It respects the current on/off state (won't run a bypassed group) and tells you if the group has
no output node to run. A **`⋯` button** (or **right-click** the row / switch) opens a small menu
with **Run · Focus · `ALWAYS` · `BYPASS` · `NEVER` · Solo** — the discoverable way to **mute a
group to `NEVER`**, and to **Solo** it (set this group `ALWAYS` and every other group `BYPASS` as a
*persistent* state, with an **Undo solo** entry that restores the previous modes). **Click a row's
name or colour dot to _focus_ it** — the canvas pans/zooms to that group. A **filter box** hides
non-matching rows by name (handy with many groups; view-only, never touches the graph), and while
a filter is active the header **`all on`** / **`all off`** / **`all ✕`** (set to Never) buttons act
only on the matching groups (otherwise on every group).
**Link groups so one switch drives another.** Two variants of a branch, three upscalers, a draft
path and a final path — normally you switch one on and remember to switch the other off. A **link**
does it for you. The quickest way in is a row's **`⋯` → `🔗 Toggle with…`**: pick a second group and
the two now take turns — switch either one on and the other goes off. The header **`🔗`** button
opens the full editor, where a link has a type:
* **`follow / opposite`** — every member carries a sign: **`+`** means "same state as whichever
  member you just moved", **`−`** means "the opposite". `A+ B−` is the plain toggle; **`A+ B− C−`**
  means *switch `A` off and `B` and `C` both come on, switch `A` back on and both go off*; `A+ B+`
  is a mirror, two groups that are always in the same state.
* **`only one of`** — a radio: switching any member on switches every other member off. Tick
  **keep one always on** and switching the last one off brings the next one in instead of leaving
  the set empty. (Three-way switching can't be written with signs, which is why it's its own type.)

Each link chooses whether its **off** means `BYPASS` or `NEVER`, can be **disabled** without being
deleted, and links **chain** — if `A` drives `B` and `B` drives `C`, moving `A` carries through to
`C`. A link fires **however the change arrived**: a click in this panel, `Ctrl+B` on the canvas, or
ComfyUI's own group menu — the panel notices and follows. Linked rows carry a **`🔗` badge** whose
tooltip spells out what will happen ("switching this on switches OFF …"). Links are saved on the
node (in `properties`), so they travel with the workflow. A set that contradicts itself (`A` off
means `B` on means `C` off means `A` on…) is **resolved, not oscillated**: the group you actually
moved wins, the rest follow in order, and the badges of the groups it couldn't satisfy are ringed in
red. The **`all on` / `all off` / `all ✕` buttons and Solo deliberately skip the engine** — a bulk
switch means exactly what it says — and **right-clicking `🔗` pauses every link** for the session
when you want to arrange the board by hand.
**Nested groups keep their arrangement.** Switching a parent group off sets every node inside it —
including the nested groups' — to the same mode, which would otherwise wipe out how the inside was
set up. So the nested groups' states are remembered on the way down and put back when the parent
comes on again: a parent holding an `ESRGAN` / `LDSR` pair comes back with the same one of the two
selected, not with both on. When a link drives a parent *and* one of its children, the parent is
applied first, so the child's own rule survives its parent's blanket sweep.
**Hide the groups you never touch.** A group that just loads the base models, VAE or LoRAs is
set-and-forget — it only clutters the panel. Pick **`🚫 Hide from this list`** in a row's **`⋯`**
menu and the row disappears; the header count shows how many are put away (`7 +2🚫`). Nothing is
lost and nothing is destructive: the group keeps its current mode, and hidden groups are
deliberately **skipped by `all on` / `all off` / `all ✕` and by Solo**, so a bulk switch-off can
never disable the loaders your workflow needs. To get one back, click the header **`👁`** button —
hidden rows reappear **dimmed** (marked `🚫`), fully usable, with **`👁 Unhide`** in their `⋯` menu;
click `👁` again to put them away, or **right-click `👁` to unhide everything at once**. The hidden
set is saved on the node (in `properties`), so it travels with the workflow and survives a reload,
while the reveal toggle itself is per-session. If you hide a group that has nested groups inside it,
the children stay listed and simply shift one level left.
**Reorder the rows** to taste: **`sort ⇅`** sorts by name (click toggles A–Z / Z–A; **right-click**
restores the original order), or **drag a row by its `⠿` handle**. Reordering is **nesting-aware**:
a group can only be moved **among its siblings** (groups sharing the same parent), and dragging a
parent **carries its whole subtree** — so a nested group can't be stranded under an unrelated parent
in the list. The chosen order is saved on the node (in `properties`), so it travels with the
workflow and survives a reload — it's purely cosmetic and never affects the prompt (or the actual
group nesting, which is defined by the groups' positions on the canvas). The node itself has no inputs or outputs and never runs on
the backend — it's excluded from the prompt and only manipulates other nodes' modes on the client
before the run. Category `Kinburg-Nodes/util`.

---

## 🗄️ `accumulators/` — Name-Based Accumulators

> **System Purpose & Overview**  
> Named global accumulators across node executions for images, audio, texts, prompts, captions, and generation info.

For collecting parallel branches without manual batch wiring. **`Set Accumulator (images)`** is a
labelled pass-through: connect a flow's final image and give it an accumulator `name` (e.g.
`IMG_RESULTS`); copying the flow auto-increments its `index`. **`Get Accumulator (images)`** has a
**dropdown** of the defined accumulator names — pick one (it auto-collects) — plus a
**Collect** button that physically wires every matching Set's output into it (real links, in
`index` order) and batches them (reusing Unlim Image Batch: `mode` /
`pad_color` / `skip_empty`). Plug the Get node where an image batch used to go (e.g. Image
Compare). A sibling **`Get Accumulator (images list)`** collects from the *same* Set nodes but
returns a **list** instead of a batch, so accumulated images of different sizes coexist (feed it
straight into Image Compare, which now takes a batch or a list). Press Collect again after adding/removing Set nodes — it rebuilds the links from
scratch and **only wires active Sets** (a Set in Bypass or Mute is skipped, so it drops out on
re-collect). A **text pair** —
**`Set Accumulator (texts)`** / **`Get Accumulator (texts)`** — works the same way and joins the
collected texts with a `separator` (reusing Unlim Text Concat) in index order. Two
compare-tuned twins of the text pair drop the separator field entirely and hardcode the
separator Image Compare expects, so the two ends can't mismatch: **`Set/Get Accumulator
(prompts)`** joins blocks with a `---` line (feed `Get` into the compare node's `prompts`),
and **`Set/Get Accumulator (captions)`** joins with a newline, one caption per line (feed into
`captions`). A **gen-info pair** — **`Set Accumulator (gen info)`** / **`Get Accumulator (gen info)`** — works the same
way for the `data` (GEN_INFO) of **Generation Info** nodes: Get collects every matching Set's
dump (in index order) and outputs a single `data` bundle (`GEN_INFO_LIST`) — wire that one
output straight into **Generation Info Filter**, so its settings no longer need wiring branch
by branch. And an **audio pair** — **`Set Accumulator (audio)`** / **`Get Accumulator (audio)`** —
does it for `AUDIO`, which is what feeds **Siren Compare**'s single `audios` input. The wiring is
plain links, so execution and caching are completely standard.

**Collect All lives on `Image Compare (HTML)`** — collecting only ever serves a comparison, so the
button sits on the node that consumes the accumulators rather than on a helper node of its own. Its
**🔌 Collect All** button (re)wires *every* Get Accumulator in the graph at once (the label then shows
how many were wired), so you don't have to visit each one after scaling the workflow. Like the
per-node Collect it rebuilds links from scratch and wires only **active** Sets — bypass or mute a Set
and re-collect to drop it from every accumulator. The compare node's **`auto_collect`** toggle (on by
default) does it automatically right before the workflow is queued, so a run always reflects the
current, non-bypassed Sets; turn it off to collect only on the button press. *(This replaces the old
standalone `Collect All Accumulators` node — delete it from older workflows, where it will show up as
missing.)*

---

## 📑 `list_ops/` — List & Batch Operations

> **System Purpose & Overview**  
> Element insertion and removal operations for image batches and lists.

Edit collections by position without rewiring. Two families:

**Image batch** (a single `[B,H,W,C]` IMAGE tensor, frame-wise): **`Image Batch Insert`** puts an
`image` (itself possibly a batch) into `batch` at a chosen spot — `position` = *at start / at end
/ at index / after index* with a 0-based `index` (negative counts from the end, `-1` = last).
Frames of a different size are reconciled losslessly like Unlim Image Batch (`mode` /
`pad_color`). **`Image Batch Remove`** drops `count` frame(s) starting at `index` and returns both
the `batch` remainder and the `removed` frames (`index` may be negative). Category
`Kinburg-Nodes/image`.

**Generic list** (a ComfyUI list of ANY type — images, strings, latents, ints…): **`List
Insert`** inserts the `item` input into `list` at the chosen `position` / `index` (feed `item` a
multi-item list to insert several at once); **`List Remove`** drops `count` item(s) at `index`
and returns the `list` remainder plus the `removed` items. An item is one list element (a single
value counts as a one-item list). Use these when items aren't same-size images — for frame-level
edits of a same-size IMAGE batch, use the Image Batch nodes above. Category `Kinburg-Nodes/list`.

---

## 📊 `report/` — Investigation Report DB

> **System Purpose & Overview**  
> Investigation report database behind Image Compare (work in progress).

### ℹ️ No nodes of its own

**This package registers no nodes** — it is the SQLite report DB and the `PromptServer` routes that
**`Image Compare (HTML)`** uses, so everything below is driven from that node and its served page.

The Image Compare node carries a `report_db` input (default `<output>/kinburg/reports.db`,
editable on the page) and saves clean per-image PNGs to a run-scoped folder. The served
comparison page has a **"Save run to report"** button that POSTs the run — images, caption,
prompt, settings, and your per-result verdict/rating/comment — to a local **SQLite** DB
(`/kinburg/report/save`). Settings are stored both as text and expanded into a key/value
table so the report browser can filter/sort by any setting field, including ones that didn't
exist before. Re-saving the same run updates it in place (no duplicates). A **📊 Report**
button (also at `/kinburg/report`) opens a browser page over the whole DB — a
sortable/filterable table (thumbnail, run, caption, status, ★ rating, tags, settings, prompt,
comment) with free-text search and a by-setting-field filter; rows link back to their
comparison. You can **edit in place** (toggle status, set rating, add/remove tags, edit the
comment — saved straight to the DB), **export** the filtered view to CSV / Markdown / HTML,
and **delete a run** (removing its rows and image files).

---

[← back to the node index](../README.md#-node-index)
