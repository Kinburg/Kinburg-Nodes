# Vendored third-party code

`audiosr/` is the model implementation of **AudioSR / Versatile Audio Super Resolution**, copied here
so that this pack's `Audio SR` node does not depend on another custom-node pack being installed. It is
third-party code under the **MIT licence** (`setup.py` in the upstream repository declares
`license="MIT"`).

Copy this pack's own licence terms as they apply to `kinburg-nodes`; the files under `audiosr/`
remain under their own. Two files carry their own upstream licences and are untouched:
`audiosr/hifigan/LICENSE` and `audiosr/latent_diffusion/modules/phoneme_encoder/text/LICENSE`.

**Note on attribution.** The `LICENSE` file shipped alongside the upstream package contains the MIT
text with GitHub's default template copyright line (`Copyright (c) 2012-2023 Scott Chacon and
others`), which is boilerplate rather than the project's actual attribution — so it was not copied
here verbatim. Before publishing this pack, replace the placeholder below with the copyright line
from the upstream repository itself:

    Copyright (c) <year> <upstream AudioSR authors>

The route it came by, for anyone tracing it: upstream AudioSR → the `ComfyUI-AudioSR` wrapper (MIT,
Copyright (c) 2025 Saganaki22), whose vendored copy of the model was the immediate source. None of
that wrapper's own node code is here; the node in `../nodes.py` is this pack's.

## What was changed

Three edits, each marked in place with a `# --- kinburg-nodes:` comment:

1. **Import prefix.** `versatile_audio_super_resolution.audiosr.…` → `audiosr.…` throughout (21
   files), so the package is importable as a plain top-level `audiosr` from this `vendor/`
   directory. No other code was altered by that rewrite.
2. **`audiosr/clap/` was removed** (56 files, 3.25 MB — three quarters of the package). Its only
   construction site was `self.clap = CLAPAudioEmbeddingClassifierFreev2(...)` in
   `latent_diffusion/models/ddpm.py`, and that attribute was never read anywhere in the package — a
   leftover of the AudioLDM lineage this model grew from. Super-resolution conditions on
   `VAEFeatureExtract` (the shipped config's `cond_stage_config`), not on CLAP, so the module cost
   0.80 GB of the checkpoint's 6.18 GB and a HuggingFace round-trip at import time
   (`clap/open_clip/model.py` called `BertModel.from_pretrained("bert-base-uncased")` while merely
   being imported).

   Verified rather than assumed: with `clap/` gone, `LatentDiffusion` built from the shipped config
   has **0 missing** parameters against both real checkpoints (basic and speech) and 507 unexpected
   ones, all of them `clap.*`. So the model is fully satisfied by the weights it is given, and the
   only tensors now ignored are ones that were never used. It builds in 7 s at 1085.8 M parameters,
   4.34 GB fp32, down from 5.14 GB.

   `CLAPAudioEmbeddingClassifierFreev2` itself was left in `encoders/modules.py` as upstream wrote
   it, with its two CLAP imports moved from module scope into the constructor — so the file still
   imports, and building that conditioner raises a plain `ImportError` naming the absent package
   instead of a missing import breaking every import in the tree.
3. **A progress hook** in `audiosr/latent_diffusion/models/ddim.py`: a module-level `STEP_HOOK`,
   `None` by default, called once per sampling step. It exists because the diffusion loop is where
   the time goes, so it is the only place a host can report honest progress or honour a cancel
   part-way through a chunk. With `STEP_HOOK` left at `None` the file behaves exactly as upstream,
   tqdm and all.
