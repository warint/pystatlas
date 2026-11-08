<p align="center">
  <img src="site_src/favicon.svg" width="72" alt="PyStatlas logo">
</p>

<h1 align="center">PyStatlas</h1>

<p align="center"><b>A Python gallery of statistical graphics for the social sciences.</b><br>
{{N}} charts · {{K}} families · copy‑paste Python code · inspired by the R Graph Gallery, <code>ggstats</code> and <code>ggstatsplot</code>.</p>

<p align="center">
  <a href="{{SITE}}"><b>Browse the gallery →</b></a><br>
  <a href="{{SITE}}">English</a> · <a href="{{SITE}}fr/">Français</a> · <a href="{{SITE}}ja/">日本語</a> · <a href="{{SITE}}zh/">中文</a><br>
  Created by <a href="https://www.warin.ca"><b>Thierry Warin</b></a> · <a href="https://www.warin.ca">www.warin.ca</a>
</p>

---

PyStatlas is a curated atlas of **statistical** charts for Python: coefficient plots, odds‑ratio forests, Likert scales, Kaplan–Meier curves with risk tables, regression diagnostics, SHAP explanations, spatial autocorrelation maps, meta‑analysis forests and more. Each chart is a single, self‑contained script built on **NumPy, pandas, Matplotlib, seaborn, SciPy, statsmodels, scikit‑learn, lifelines, SHAP, GeoPandas, Shapely, libpysal and esda** — no downloads, no hidden helpers.

### The examples are set in published research

Nearly every chart takes its setting from one of Thierry Warin's papers — party manifestos, central-bank speeches, fake news on Facebook, patent grants in China's pharmaceutical industry, the Mundellian trilemma, waste reduction in manufacturing — and the spatial charts from the CIRANO GVCdtLab's digital twin of the Great Lakes–St. Lawrence corridor. The data are **small seeded simulations**, just large enough to show the method: they are labelled as simulated on every figure and are **not** the papers' data or results. The full reference is printed under each chart. The exceptions: the power curve uses no data, and the meta-analysis and funnel plots use real data — a published table by Raudenbush (1984), typed in.

## The repository *is* the gallery

```
i18n/                     website translations (en, fr, ja, zh)
gallery/
  <NN-family>/
    _category.yaml        family title + summary
    <NN-chart>/
      plot.py             the code (metadata in the docstring)
      plot.png            the rendered figure
      README.md           browsable on GitHub
```

Browse the folders right here on GitHub, or visit the [website]({{SITE}}).

## Families

| | |
|:--|:--|
{{FAMILIES}}

## Research behind the examples

{{PAPERS}}

## Use a chart

```bash
pip install -r requirements.txt
python gallery/05-regression-models/01-coefficient-plot/plot.py
```

For the exact look of the gallery, drop [`pystatlas.mplstyle`](pystatlas.mplstyle) next to your script and add `plt.style.use("pystatlas.mplstyle")`.

## Build the site

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build.py            # renders new/changed charts, READMEs and docs/
```

The website in `docs/` is served by GitHub Pages.

## How to cite

If PyStatlas helped your research, teaching or work, please cite it:

> Warin, T. (2026). *PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences*. https://warint.github.io/pystatlas/

```bibtex
@misc{warin2026statlas,
  author       = {Warin, Thierry},
  title        = {PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences},
  year         = {2026},
  howpublished = {\url{https://warint.github.io/pystatlas/}},
  note         = {Source code: https://github.com/warint/pystatlas}
}
```

GitHub's **"Cite this repository"** button (right sidebar) gives APA and BibTeX from [`CITATION.cff`](CITATION.cff).

## Author

**Thierry Warin** — [www.warin.ca](https://www.warin.ca)

## Contributing

New charts are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT © [Thierry Warin](https://www.warin.ca)
