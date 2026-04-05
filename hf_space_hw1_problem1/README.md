---
title: HW1 QML Viewer
emoji: 📈
colorFrom: yellow
colorTo: green
sdk: static
pinned: false
---

# HW1 QML Viewer

Static Hugging Face Space for visualizing the current `HW1 Problem 1` export.

Workflow:

- Keep this folder in the repository as the static app scaffold.
- Let `gx10` training write runtime files into `hf_space_hw1_problem1/runtime/`.
- The viewer first looks for `runtime/viewer_manifest.json`.
- If no runtime manifest exists, it falls back to `data/viewer_manifest.template.json`.
- Runtime exports are batch-based by default, so the slider advances one
  optimizer step at a time.
- The run selector lets you switch between different hyperparameter results.
- When you are ready to publish, build a clean bundle from `gx10` and push that
  bundle to a Hugging Face Static Space.

## Local preview

Open `index.html` in a browser, or serve the folder with a tiny static server:

```bash
cd hf_space_hw1_problem1
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Publish to Hugging Face

On `gx10`:

```bash
cd ~/quantum_computing
./scripts/gx10_prepare_hf_space.sh
```

That creates a clean bundle under `.out/hf_space_hw1_problem1_publish/` with the latest
runtime export included. Push the contents of that directory to a Hugging Face
Static Space repository.
