# 🖼️ Image Processing & Visualizers

<!-- index-order: 6 -->

[← back to the node index](../README.md#-node-index)

---

## ⚖️ `image_compare/` — Image Compare (HTML) & Color Caption

> **System Purpose & Overview**  
> Side-by-side interactive HTML image comparison viewer with zoom, slider, and vision judge ratings section.

Takes a batch **or an image list** (different sizes are fine — e.g. from Get Accumulator (images
list)) + short labels (`captions`) + full generation prompts (`prompts`)
and produces an interactive HTML comparison page — grid (columns + max row height),
before/after slider, opacity overlay, A/B flip, pixel difference, **synced zoom & pan**, synced loupe, lightbox,
drag-to-reorder, and per-result **review** controls — a **hide**/reject button (with a
toolbar toggle to show/hide hidden results), a **star rating** (1–5), **tags**, and a
**comment** box — plus a "Save page" button and an `images_captioned` output (a *list*, so
mixed-size inputs stay separate) with the captions drawn on each image. The review state (hide / rating / tags / comment) persists across reloads of the served
page (browser localStorage). Save
options: `output_dir` (any folder; the page is served straight from there, no copies),
`save_captioned_images`, `save_prompts_txt`. Only `images` is required; the text inputs are
all sockets (no inline fields): `captions` (one per line, e.g. from **Get Accumulator
(captions)**), `prompts` (full per-image blocks separated by a `---` line, e.g. from **Get
Accumulator (prompts)**), and `settings_data` (structured per-image settings from
**Generation Info Filter**) — rendered under each image as one `[Class] param: value` line
per field, with its own page toggle. The `url` output is a clickable http
link. Node: **`Image Compare (HTML)`** (category `Kinburg-Nodes/image/compare`).

The node carries no *view* settings: what's visible — captions, prompts, settings, times, metrics,
judge, hidden results — is toggled **on the page**, where you can see the effect. It does carry the
accumulator **🔌 Collect All** button and its **`auto_collect`** toggle (see the accumulators section),
since gathering the branches is what feeds this node in the first place.

**The page header is three fixed rows** instead of a toolbar that wrapped into five, and it's grouped
by purpose rather than by accident of order:

- the **title** on its own thin line, never shortened;
- the **tools** line, read left to right: the five **modes** pinned left (also keys **1…5**), the
  viewing controls centred on the page — **👁 Panels**, **Zoom**, **⤢ Fit**, **🔍 Loupe** — and
  **⬇ Save page** / **📊 Report** / **❔ Help** pinned right;
- a **mode line** that always belongs to what you're doing: **Columns / Rows / Sort** in Grid, and the
  **A** / **B** selectors in the pair modes, where they meet at the centre of the page — the same seam
  the wipe divider sits on — so each selector is on the side of the image it controls and long captions
  grow **outward** instead of being cropped (the full label is in the option list and the tooltip).

The two panels you touch least often stand behind one button each: **👁 Panels** (the six visibility
toggles + hidden results, with a badge counting them) and **📊 Report** (the DB path, *Save run*,
*Open report browser*). The **Opacity** slider isn't in the header at all — it appears right above the
image it fades, as `A ──○── B` with a live percentage. The Kinburg-Nodes credit sits in a thin footer.
Result: **146 px → 109 px** of header at 1266 px wide (11.5% → 8.6% of the window), with a layout that
no longer reshuffles itself as the window changes.

**Rows on screen, not pixels.** Grid's height control is **Rows** — how many rows of results you want
to see at once (default **1**) — instead of a **Max h** in pixels you had to guess and re-guess. One row
fills the window, scrolling **snaps row by row**, and the page becomes a flipper: one comparison at a
time, always the same size. **Max h** is still there for when you want a specific scale (set **Rows** to
**0**), and while Rows is on it just reports the height that was worked out.

The mechanism is deliberately dumb, because the obvious implementation isn't: rather than measure the
caption/prompt/settings/review chrome and subtract it — a two-pass layout that has to re-run every time
you toggle a panel, change **Columns**, type in a comment or load a lazy image — a row is simply given a
**fixed height of one N-th of the visible area**, and each card's own flex layout hands the picture
whatever its panels leave over. So a card with a long prompt shows a slightly smaller image than its
neighbour, with no arithmetic anywhere; when the panels would crowd the picture out altogether **they**
scroll inside the card instead, and the image never drops below a third of it. Exactly **one** number is
ever measured — the height the stage can show — and only when the window (or the header's own wrapping)
actually changes it. The page is an app shell now: fixed header and footer, one scrolling stage, which
is also what lets the pair modes fill the window instead of the old magic `78vh`.

**Synced zoom & pan.** The **Zoom** slider (×1–×8) magnifies **every** image at once, and they all
show the **same region**: drag any one of them and the visible area moves in all of them together.
**Ctrl + wheel** over an image zooms **at the cursor** (the bit you point at stays put); **double-click**,
**0**, or **⤢ Fit** returns to fit. Nothing is re-laid-out — each image is scaled by a CSS transform
inside a clipping frame, so the grid never moves and only the visible region does. The region is held
in *picture-relative* coordinates, so it survives switching modes and changing **Columns** / **Rows**,
and lands on the same content even when the compared images differ in size or aspect. Made for judging
fine detail — texture, small faces, foliage — without hunting for the same spot in each picture.

The **Loupe stacks on top of it**: the lens shows **zoom × lens** while its surroundings stay at the
current zoom, so you can park every image on a region and then peek closer still (plain **wheel** =
lens zoom, **Alt/Shift + wheel** = lens diameter, **Ctrl + wheel** = image zoom, dragging still pans).
Reordering goes through a **⠿ grip** in each card's top-left corner rather than the card itself, so all
of a card's text stays selectable and reordering keeps working at any zoom. In **Slider** mode the
divider's round handle always moves the wipe — which is what makes it reachable when the pointer is
busy panning or driving the lens — and **the wipe stays where you put it**: switching A/B, toggling a
panel, hiding a result or stepping through other modes no longer snaps it back to the middle.

The **🔗 Open comparison** link is stored on the node, so it survives tab switches and workflow
reloads. Three things keep it honest. The token → folder map behind the served URL is **persisted to
disk** (`<output>/kinburg/compare_dirs.json`, last 300 runs), so a link minted before a ComfyUI
restart still opens instead of 404-ing. The link is refreshed from the API's `executed` event rather
than from the `onExecuted` prototype chain, which any other installed extension can break by patching
it without calling through. And a link that was **restored** rather than produced by a run you just
did says so — `🔗 Open comparison (previous run)` — and is quietly checked against the server; if that
run's folder is gone it turns into a dull `⚠ Comparison expired — run to rebuild` and stops being
clickable, instead of opening a blank 404 tab.

Two more controls. **`embed_images`** chooses how the comparison is saved: **off** (default) writes
a **portable folder** `<prefix>_<datetime>/` — a light `index.html` + an `images/` subfolder with
relative links — so you can open it offline, zip and share it, or open it in-app; **on** writes a
single self-contained `.html` with every image inlined as base64 (one bigger file). Both open from
the node's `url` output. And an optional **`reference`** image (or
**`reference_index`**, a 0-based index into the batch; `-1` = off) enables **similarity
metrics** — **SSIM** and **PSNR** of every image vs the reference — shown under each image (a
**Metrics** page toggle) with a **Similarity** sort in the grid. Great for upscale / img2img /
restoration, or for checking how far a quantized model's output drifts from its fp16 baseline.
Note these measure *closeness to the reference*, not aesthetic quality (that's the Vision LLM
Judge's job).

An optional **`judge_data`** input takes the **Vision LLM Judge**'s `results_json` and renders
each image's AI verdict — stars (proportional to its score), tags and a comment — as a
**read-only** section under it, with its own **🤖 Judge** page toggle, kept separate from your
own hide / rating / tags / comment review controls.

Also in this package: **`Color Caption`** — write a caption and type two colors, the
**text** color and the **band** color behind it (each as a HEX `#RRGGBB` value). It outputs a one-line JSON
`{"caption": "...", "color": "#RRGGBB", "band_color": "#RRGGBB"}`. Wire it into the compare
node's `captions` input (one caption per line) to style that label — both on the page and
on the drawn `images_captioned`. The defaults (white text on a black band) reproduce the
classic look. The `captions` input still accepts plain text lines exactly as before; each
line is treated independently, so you can mix plain and styled captions. Category
`Kinburg-Nodes/image/compare`.

---

## 🖼️ `image_batch/` — Unlim Image Batch & List

> **System Purpose & Overview**  
> Dynamic unlimited image batching and list creation utilities.

**`Unlim Image Batch`** concatenates an unlimited number of IMAGE inputs into a single
batch. The input list grows on its own (like Unlim Text Concat): `image_1` (required) +
`image_2`, and a new empty slot appears whenever you connect the last one. A single batch
tensor needs every frame at the same size, so when inputs differ `mode` reconciles them
**without resampling** (no quality loss): `as is` stacks pixels untouched and errors on a
size mismatch; `crop to smallest` center-crops every input down to the smallest size;
`pad to largest` center-pads every input up to the largest, filling the borders with
`pad_color` (HEX). Mismatched channel counts (RGB vs RGBA) are padded with opaque alpha so
everything stacks. Each input may itself be a batch. `skip_empty` (on by default) drops
empty / unconnected inputs so bypassing a branch doesn't break the batch. Category
`Kinburg-Nodes/image`.

**`Unlim Image List`** is the sibling for when the sizes *shouldn't* be reconciled: it
returns a ComfyUI **image list** (`OUTPUT_IS_LIST`) instead of a stacked tensor, so frames
of different sizes can travel together. The growing inputs work the same way; there are no
options. Every input is split into single frames, in slot order, so the list length is the
total image count and each index is exactly one image — convenient for loop / iterator
nodes (read the length, take an item by index, process it inside the loop). Note that
downstream nodes then run once per item rather than on a single batch. Category
`Kinburg-Nodes/image`.

---

## 🎨 `collage/` — Collage Layout Builder

> **System Purpose & Overview**  
> Custom grid layout collage builder for images.

**`Collage`** arranges images into a grid on a single output canvas (e.g. an A4-ish
2480×3508). Source is the wired `input_images` batch, or — if nothing is connected —
every image in `folder_path` (natural-sorted, so `img2` < `img10`). `cols` sets the column
count and the rows follow from the image count; `gap` spaces the cells and `margin` frames
the grid, with the cell size derived so the whole grid fits the canvas. Each image is fit
into its cell with its aspect ratio preserved; the letterbox border is filled with the
image's own top-left pixel color so it blends in. The background is `background_color`
(HEX), or a connected `background_image` (stretched to the output size, first frame).
Category `Kinburg-Nodes/image`.

---

[← back to the node index](../README.md#-node-index)
