# TriPS — Project Page

Static academic project page for **“Triadic Dynamics Aware Diffusion Posterior Sampling
for Inverse Problems: Optimizing Guidance and Stochasticity Schedules” (ICML 2026)**.
Built on the [Nerfies](https://github.com/nerfies/nerfies.github.io) template and benchmarked
against the [FLAIR](https://inverseflair.github.io/) page.

```
TriPS_page/
├── index.html              # the page (self-contained; styles/JS via CDN)
├── convert_figures.sh/.ps1 # render figure PDFs -> static/images/*.png (optional, recommended)
└── static/
    ├── css/index.css
    ├── js/index.js
    ├── pdfs/               # all figure PDFs + TriPS_arxiv.pdf (shipped)
    └── images/             # rendered figures (photos=.jpg, diagrams=.png) — pre-generated
```

## 1) Figures — already rendered

All figures are **pre-rendered** into `static/images/` (photographic comparison grids as
`.jpg`, diagram/plot figures as `.png`). The page is ready to view and deploy as-is — you do
**not** need to run any conversion script.

> Optional: if you replace a figure PDF in `static/pdfs/` and want to re-render, run
> `bash convert_figures.sh` (Linux/macOS) or `convert_figures.ps1` (Windows). The page also
> falls back to embedding the source PDF automatically if an image is ever missing.

## 2) Preview locally

```bash
cd TriPS_page
python -m http.server 8080      # then open http://localhost:8080
```
(Opening `index.html` directly also works; a local server just makes the PDF-embed fallback
and relative paths behave consistently.)

## 3) Deploy to GitHub Pages → get the link

**Recommended (serve from the code repo):** copy this folder into the
[`mundongju/TriPS`](https://github.com/mundongju/TriPS) repo under `docs/`, then in
*Settings → Pages* set **Source = `main` branch / `/docs` folder**.

```bash
# from a clone of github.com/mundongju/TriPS
mkdir -p docs
cp -r /path/to/TriPS_page/* docs/
git add docs && git commit -m "Add project page" && git push
```

➡️ **Live URL:** `https://mundongju.github.io/TriPS/`

**Alternative (dedicated repo):** push the *contents* of `TriPS_page/` to a new repo
`mundongju/TriPS-page` (root), enable Pages on `main` → `https://mundongju.github.io/TriPS-page/`.

> Note: `static/pdfs/TriPS_arxiv.pdf` is ~17 MB. If you prefer a lighter repo, replace the
> **Paper** button `href` in `index.html` with your arXiv/OpenReview link and delete that PDF.

## Editing

- Authors / links / venue: top of `index.html` (hero section).
- Quantitative numbers: the `TRIPS_TABLE` object near the bottom of `index.html`
  (`#` = best/bold, `~` = second/underline).
- Colors / layout: `static/css/index.css` (the triad palette `--trips-dc/-cfg/-sto`).
- The arXiv button is currently disabled (`btn-pill disabled`); enable it by adding the URL.
