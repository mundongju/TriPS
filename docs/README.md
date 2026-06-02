# TriPS Project Page (`docs/`)

Static project page for **Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse
Problems: Optimizing Guidance and Stochasticity Schedules (ICML 2026)**, served by GitHub Pages
from this `docs/` folder. Layout is inspired by the
[FLAIR](https://inverseflair.github.io/) page and the
[Nerfies](https://github.com/nerfies/nerfies.github.io) template.

Live URL once Pages is enabled: **https://mundongju.github.io/TriPS/**

## Layout

```
docs/
├── index.html                 # the page (sections: Key Idea, Qualitative Samples,
│                              #  Analysis, Method, Quantitative, Reward-guided PD Control, BibTeX)
├── static/
│   ├── css/index.css
│   ├── js/index.js            # hero animation, before/after sliders, gallery nav
│   ├── hero/                  # 4 alternating measurement/result tiles for the hero
│   ├── quali/<task>/          # 15 qualitative samples x {meas, flow, ours}
│   ├── images/                # Key_Idea_triadic.jpg, Table1.png, method/analysis/ablation figures
│   └── pdfs/                  # source figure PDFs + the paper PDF (used as image fallback)
├── ref_fig_main/ ref_figs_quali/ ref_figs_rest/   # original source figures (not used at runtime)
└── convert_figures.sh / .ps1  # optional: re-render diagram PDFs to PNG
```

All runtime images are pre-generated, so the page is ready to view and deploy as is.

## Editing

- **Hero**: the four alternating tiles are `static/hero/*.jpg`; CSS keyframes (`@keyframes swap`
  in `index.css`) cross-fade measurement and result. Title, authors and buttons sit on the
  translucent `.hero2-card`.
- **Qualitative Samples**: edit the `SAMPLES` array in `static/js/index.js`
  (task, label, index, ours-label). Images load from `static/quali/<task>/<idx>_{meas,flow,ours}.jpg`.
- **Quantitative**: `static/images/Table1.png`. To regenerate the table image, rerun
  `make_table1.ps1` (kept alongside this README) or drop in a crop of Table 1 from the paper.
- **Numbers/links/authors**: top of `index.html`.

## Preview locally

```bash
cd docs
python -m http.server 8080      # open http://localhost:8080
```

## Deploy (GitHub Pages)

Push the repo, then in **Settings -> Pages** set **Source = `main` branch, `/docs` folder**.
The site publishes at `https://mundongju.github.io/TriPS/`.
