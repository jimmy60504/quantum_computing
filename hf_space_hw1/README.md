---
title: HW1 QML Viewer
emoji: 📈
colorFrom: amber
colorTo: teal
sdk: static
pinned: false
---

# HW1 QML Viewer

Static Hugging Face Space for visualizing the current `HW1 Problem 1` export.

This first export is intentionally training-free:

- It reuses the latest saved repository artifacts.
- It uses Plotly for the heatmap and loss panels.
- It shows the current circuit diagram and data overview.
- The target panel is drawn from numeric grid data in the browser.
- The prediction and error panels currently use raster fallback images because
  the training script has not exported raw grid tensors yet.
- The viewer already supports a step slider and timeline panel.
- When step-by-step snapshots are exported later, the same UI can replay them
  without changing the Space structure.

## Local preview

Open `index.html` in a browser, or serve the folder with a tiny static server:

```bash
cd hf_space_hw1
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Publish to Hugging Face

Create a new Static HTML Space, then copy the contents of this folder into the
Space repository and push.
