<p align="center">
  <img src="site_src/favicon.svg" width="72" alt="Statlas logo">
</p>

<h1 align="center">Statlas</h1>

<p align="center"><b>The Python gallery of statistical graphics.</b><br>
{{N}} charts · {{K}} families · copy‑paste Python code · inspired by the R Graph Gallery, <code>ggstats</code> and <code>ggstatsplot</code>.</p>

<p align="center">
  <a href="{{SITE}}"><b>Browse the gallery →</b></a><br>
  Created by <a href="https://www.warin.ca"><b>Thierry Warin</b></a> · <a href="https://www.warin.ca">www.warin.ca</a>
</p>

---

Statlas is a curated atlas of **statistical** charts for Python: coefficient plots, odds‑ratio forests, Likert scales, Kaplan–Meier curves with risk tables, regression diagnostics, Bland–Altman, meta‑analysis forests and more. Each chart is a single, self‑contained script built on **NumPy, pandas, Matplotlib, seaborn, SciPy, statsmodels, scikit‑learn and lifelines** — no downloads, no hidden helpers.

## The repository *is* the gallery

```
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

## Use a chart

```bash
pip install -r requirements.txt
python gallery/05-regression-models/01-coefficient-plot/plot.py
```

For the exact look of the gallery, drop [`statlas.mplstyle`](statlas.mplstyle) next to your script and add `plt.style.use("statlas.mplstyle")`.

## Build the site

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build.py            # renders new/changed charts, READMEs and docs/
```

The website in `docs/` is served by GitHub Pages.

## How to cite

If Statlas helped your research, teaching or work, please cite it:

> Warin, T. (2026). *Statlas: The Python Gallery of Statistical Graphics*. https://warint.github.io/statlas/

```bibtex
@misc{warin2026statlas,
  author       = {Warin, Thierry},
  title        = {Statlas: The Python Gallery of Statistical Graphics},
  year         = {2026},
  howpublished = {\url{https://warint.github.io/statlas/}},
  note         = {Source code: https://github.com/warint/statlas}
}
```

GitHub's **"Cite this repository"** button (right sidebar) gives APA and BibTeX from [`CITATION.cff`](CITATION.cff).

## Author

**Thierry Warin** — [www.warin.ca](https://www.warin.ca)

## Contributing

New charts are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT © [Thierry Warin](https://www.warin.ca)
