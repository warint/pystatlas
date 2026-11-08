<p align="center">
  <img src="site_src/favicon.svg" width="72" alt="PyStatlas logo">
</p>

<h1 align="center">PyStatlas</h1>

<p align="center"><b>A Python gallery of statistical graphics for the social sciences.</b><br>
61 charts · 12 families · copy‑paste Python code · inspired by the R Graph Gallery, <code>ggstats</code> and <code>ggstatsplot</code>.</p>

<p align="center">
  <a href="https://warint.github.io/pystatlas/"><b>Browse the gallery →</b></a><br>
  <a href="https://warint.github.io/pystatlas/">English</a> · <a href="https://warint.github.io/pystatlas/fr/">Français</a> · <a href="https://warint.github.io/pystatlas/ja/">日本語</a> · <a href="https://warint.github.io/pystatlas/zh/">中文</a><br>
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

Browse the folders right here on GitHub, or visit the [website](https://warint.github.io/pystatlas/).

## Families

| | |
|:--|:--|
| [<img src="gallery/11-machine-learning/01-confusion-matrix/plot.png" width="220">](gallery/11-machine-learning/) | **[Machine learning](gallery/11-machine-learning/)** — How well does a model predict on new data, and why? Out-of-fold evaluation, learning curves and model-agnostic explanations — each with its uncertainty.<br><sub>[Confusion matrix with metrics](gallery/11-machine-learning/01-confusion-matrix/) · [Precision–recall curves](gallery/11-machine-learning/02-precision-recall/) · [Learning curve](gallery/11-machine-learning/03-learning-curve/) · [Cross-validated model comparison](gallery/11-machine-learning/04-model-comparison/) · [Permutation importance](gallery/11-machine-learning/05-permutation-importance/) · [Partial dependence with ICE curves](gallery/11-machine-learning/06-partial-dependence/) · [SHAP summary (beeswarm)](gallery/11-machine-learning/07-shap-beeswarm/) · [SHAP dependence plot](gallery/11-machine-learning/08-shap-dependence/) · [SHAP waterfall for one prediction](gallery/11-machine-learning/09-shap-waterfall/)</sub> |
| [<img src="gallery/12-spatial-data/01-spatial-weights/plot.png" width="220">](gallery/12-spatial-data/) | **[Spatial data](gallery/12-spatial-data/)** — Is a value higher where its neighbours' values are higher? Spatial weights, global and local spatial autocorrelation, and how far clusters reach — along a schematic Great Lakes–St. Lawrence corridor.<br><sub>[Choropleth with spatial weights](gallery/12-spatial-data/01-spatial-weights/) · [Moran scatterplot (global autocorrelation)](gallery/12-spatial-data/02-moran-scatterplot/) · [LISA cluster map (local autocorrelation)](gallery/12-spatial-data/03-lisa-cluster-map/) · [Spatial correlogram](gallery/12-spatial-data/04-spatial-correlogram/)</sub> |
| [<img src="gallery/01-distributions/01-histogram-density/plot.png" width="220">](gallery/01-distributions/) | **[Distributions](gallery/01-distributions/)** — See the shape of your data: histograms, densities, ridgelines, ECDFs and Q–Q plots.<br><sub>[Histogram with density](gallery/01-distributions/01-histogram-density/) · [Raincloud plot](gallery/01-distributions/02-raincloud/) · [Ridgeline plot](gallery/01-distributions/03-ridgeline/) · [ECDF comparison](gallery/01-distributions/04-ecdf-comparison/) · [Q–Q plot with confidence band](gallery/01-distributions/05-qq-plot/)</sub> |
| [<img src="gallery/02-group-comparisons/01-between-groups-violin/plot.png" width="220">](gallery/02-group-comparisons/) | **[Group comparisons](gallery/02-group-comparisons/)** — Compare groups with the test, effect size and p-values on the chart — the ggstatsplot way.<br><sub>[Between-groups comparison](gallery/02-group-comparisons/01-between-groups-violin/) · [Within-subjects comparison](gallery/02-group-comparisons/02-within-subjects/) · [Estimation plot](gallery/02-group-comparisons/03-estimation-plot/) · [Interaction plot](gallery/02-group-comparisons/04-means-with-ci/) · [Tukey HSD intervals](gallery/02-group-comparisons/05-tukey-hsd/)</sub> |
| [<img src="gallery/03-proportions-surveys/01-likert-diverging/plot.png" width="220">](gallery/03-proportions-surveys/) | **[Proportions & surveys](gallery/03-proportions-surveys/)** — Proportions, cross-tabs and Likert scales, in the spirit of ggstats::gglikert and stat_prop.<br><sub>[Diverging Likert chart](gallery/03-proportions-surveys/01-likert-diverging/) · [Stacked proportions by group](gallery/03-proportions-surveys/02-stacked-proportions/) · [Mosaic plot with Pearson residuals](gallery/03-proportions-surveys/03-mosaic-plot/) · [Proportions with confidence intervals](gallery/03-proportions-surveys/04-proportions-with-ci/) · [Cross-table with standardized residuals](gallery/03-proportions-surveys/05-crosstab-residuals/)</sub> |
| [<img src="gallery/04-correlation/01-correlation-matrix/plot.png" width="220">](gallery/04-correlation/) | **[Correlation & association](gallery/04-correlation/)** — Correlation matrices, scatterplots with statistics and pairwise relationships.<br><sub>[Correlation matrix](gallery/04-correlation/01-correlation-matrix/) · [Scatterplot with statistics](gallery/04-correlation/02-scatter-with-stats/) · [Pairs plot with correlations](gallery/04-correlation/03-pairs-plot/) · [Correlogram with clustering](gallery/04-correlation/04-correlogram-bubble/)</sub> |
| [<img src="gallery/05-regression-models/01-coefficient-plot/plot.png" width="220">](gallery/05-regression-models/) | **[Regression models](gallery/05-regression-models/)** — Coefficient plots, odds ratios, model comparison and marginal effects — ggcoef_model for Python.<br><sub>[Coefficient plot](gallery/05-regression-models/01-coefficient-plot/) · [Odds-ratio forest plot](gallery/05-regression-models/02-odds-ratio-forest/) · [Comparing nested models](gallery/05-regression-models/03-compare-models/) · [Predicted probabilities](gallery/05-regression-models/04-predicted-probabilities/) · [Interaction effects](gallery/05-regression-models/05-interaction-effects/) · [Quantile regression process](gallery/05-regression-models/06-quantile-regression/)</sub> |
| [<img src="gallery/06-model-diagnostics/01-residual-diagnostics/plot.png" width="220">](gallery/06-model-diagnostics/) | **[Model diagnostics](gallery/06-model-diagnostics/)** — Residuals, leverage, calibration and ROC curves: check before you trust a model.<br><sub>[Residual diagnostics](gallery/06-model-diagnostics/01-residual-diagnostics/) · [Influence plot](gallery/06-model-diagnostics/02-influence-plot/) · [ROC curves](gallery/06-model-diagnostics/03-roc-curve/) · [Calibration plot](gallery/06-model-diagnostics/04-calibration-plot/) · [Added-variable plots](gallery/06-model-diagnostics/05-partial-regression/)</sub> |
| [<img src="gallery/07-survival/01-kaplan-meier/plot.png" width="220">](gallery/07-survival/) | **[Survival analysis](gallery/07-survival/)** — Kaplan–Meier curves with risk tables, cumulative hazards and Cox model hazard ratios.<br><sub>[Kaplan–Meier curves](gallery/07-survival/01-kaplan-meier/) · [Cox hazard ratios](gallery/07-survival/02-cox-hazard-ratios/) · [Cumulative hazard](gallery/07-survival/03-cumulative-hazard/) · [Adjusted survival curves](gallery/07-survival/04-adjusted-survival/)</sub> |
| [<img src="gallery/08-time-series/01-stl-decomposition/plot.png" width="220">](gallery/08-time-series/) | **[Time series](gallery/08-time-series/)** — Decomposition, autocorrelation, forecasts with prediction intervals and change over time.<br><sub>[STL decomposition](gallery/08-time-series/01-stl-decomposition/) · [ACF and PACF](gallery/08-time-series/02-acf-pacf/) · [SARIMA forecast](gallery/08-time-series/03-arima-forecast/) · [Seasonal subseries plot](gallery/08-time-series/04-seasonal-subseries/)</sub> |
| [<img src="gallery/09-multivariate/01-pca-biplot/plot.png" width="220">](gallery/09-multivariate/) | **[Multivariate & clustering](gallery/09-multivariate/)** — PCA biplots, scree plots, clustering and heatmaps for high-dimensional data.<br><sub>[PCA biplot](gallery/09-multivariate/01-pca-biplot/) · [Scree plot](gallery/09-multivariate/02-scree-plot/) · [Silhouette plot for k-means](gallery/09-multivariate/03-silhouette-clusters/) · [Clustered heatmap](gallery/09-multivariate/04-clustered-heatmap/)</sub> |
| [<img src="gallery/10-inference-uncertainty/01-bootstrap-distribution/plot.png" width="220">](gallery/10-inference-uncertainty/) | **[Inference & uncertainty](gallery/10-inference-uncertainty/)** — Bootstrap, confidence intervals, power, agreement and meta-analysis forest plots.<br><sub>[Bootstrap distribution](gallery/10-inference-uncertainty/01-bootstrap-distribution/) · [What a 95% confidence interval means](gallery/10-inference-uncertainty/02-confidence-intervals/) · [Power curve](gallery/10-inference-uncertainty/03-power-curve/) · [Bland–Altman plot](gallery/10-inference-uncertainty/04-bland-altman/) · [Meta-analysis forest plot](gallery/10-inference-uncertainty/05-meta-analysis-forest/) · [Funnel plot](gallery/10-inference-uncertainty/06-funnel-plot/)</sub> |

## Research behind the examples

| Reference | Charts set in it |
|:--|:--|
| [Kouloukoui et al. (2023), global conversations on ESG topics](https://doi.org/10.1016/j.jclepro.2023.137369) | [Confusion matrix with metrics](gallery/11-machine-learning/01-confusion-matrix/) · [Cross-table with standardized residuals](gallery/03-proportions-surveys/05-crosstab-residuals/) |
| [Melchior, Warin & Oliveira (2025), COVID-19 fake-news sharing on Facebook](https://doi.org/10.1016/j.techfore.2024.123969) | [Precision–recall curves](gallery/11-machine-learning/02-precision-recall/) · [Learning curve](gallery/11-machine-learning/03-learning-curve/) |
| [Warin, d'Anglade & de Marcellis-Warin (2025), data science for waste reduction in manufacturing](https://doi.org/10.1080/19397038.2025.2527300) | [Cross-validated model comparison](gallery/11-machine-learning/04-model-comparison/) · [Permutation importance](gallery/11-machine-learning/05-permutation-importance/) · [Partial dependence with ICE curves](gallery/11-machine-learning/06-partial-dependence/) · [SHAP summary (beeswarm)](gallery/11-machine-learning/07-shap-beeswarm/) · [SHAP dependence plot](gallery/11-machine-learning/08-shap-dependence/) · [SHAP waterfall for one prediction](gallery/11-machine-learning/09-shap-waterfall/) · [Within-subjects comparison](gallery/02-group-comparisons/02-within-subjects/) |
| [CIRANO GVCdtLab, digital twin of the Great Lakes–St. Lawrence corridor](https://cirano.qc.ca/en/research/pole/Transport) | [Choropleth with spatial weights](gallery/12-spatial-data/01-spatial-weights/) · [Moran scatterplot (global autocorrelation)](gallery/12-spatial-data/02-moran-scatterplot/) · [LISA cluster map (local autocorrelation)](gallery/12-spatial-data/03-lisa-cluster-map/) · [Spatial correlogram](gallery/12-spatial-data/04-spatial-correlogram/) |
| [Sanger & Warin (2019), similarity of European party manifestos](https://doi.org/10.1016/j.dib.2019.103907) | [Histogram with density](gallery/01-distributions/01-histogram-density/) · [Clustered heatmap](gallery/09-multivariate/04-clustered-heatmap/) |
| [de Marcellis-Warin, Kouloukoui & Warin (2025), AI-related tweets](https://doi.org/10.1016/j.dib.2025.111960) | [Raincloud plot](gallery/01-distributions/02-raincloud/) |
| [Stojkov, Warin & Kjosevski (2026), current accounts of OECD countries](https://doi.org/10.1556/032.2026.00539) | [Ridgeline plot](gallery/01-distributions/03-ridgeline/) |
| [Pastoriza Rivas & Warin (2022), Tiger Woods press conferences and performance](https://doi.org/10.1016/j.dib.2022.107955) | [ECDF comparison](gallery/01-distributions/04-ecdf-comparison/) |
| [Warin (2021), global research on coronaviruses](https://doi.org/10.2196/31510) | [Q–Q plot with confidence band](gallery/01-distributions/05-qq-plot/) |
| [Kouloukoui et al. (2021), perceived exposure to climate risks](https://doi.org/10.1016/j.jclepro.2021.127160) | [Between-groups comparison](gallery/02-group-comparisons/01-between-groups-violin/) · [What a 95% confidence interval means](gallery/10-inference-uncertainty/02-confidence-intervals/) |
| [Warin, Melchior & de Marcellis-Warin (2025), public engagement with the WHO on X](https://doi.org/10.2196/69959) | [Estimation plot](gallery/02-group-comparisons/03-estimation-plot/) |
| [de Marcellis-Warin et al. (2022), AI and consumer manipulation](https://doi.org/10.1007/s43681-022-00149-5) | [Interaction plot](gallery/02-group-comparisons/04-means-with-ci/) |
| [Sanger & Warin (2020), the speeches of the ECB's presidents](https://doi.org/10.1142/s2194565920500098) | [Tukey HSD intervals](gallery/02-group-comparisons/05-tukey-hsd/) |
| [Melchior, Warin & Oliveira (2026), survey on COVID-19 fake-news sharing](https://doi.org/10.1016/j.dib.2026.112533) | [Diverging Likert chart](gallery/03-proportions-surveys/01-likert-diverging/) |
| [Sanger & Warin (2018), the 2014 Québec election on Twitter](https://doi.org/10.22230/cjc.2018v43n2a3251) | [Stacked proportions by group](gallery/03-proportions-surveys/02-stacked-proportions/) |
| [Stojkov & Warin (2023), the refugee crisis on Twitter](https://doi.org/10.1177/17504813231155739) | [Mosaic plot with Pearson residuals](gallery/03-proportions-surveys/03-mosaic-plot/) |
| [Kouloukoui, de Marcellis-Warin & Warin (2025), corporate climate transition plans](https://doi.org/10.3390/cli13050088) | [Proportions with confidence intervals](gallery/03-proportions-surveys/04-proportions-with-ci/) |
| [Stojkov & Warin (2019), the Mundellian trilemma dataset](https://doi.org/10.1016/j.dib.2019.104677) | [Correlation matrix](gallery/04-correlation/01-correlation-matrix/) |
| [Janicki, Warin & Wunnava (2009), endogenous optimum currency areas](https://doi.org/10.1111/j.1467-9396.2008.00802.x) | [Scatterplot with statistics](gallery/04-correlation/02-scatter-with-stats/) |
| [Hadengue & Warin (2014), specialisation and convergence in Brazil, China and India](https://doi.org/10.7202/1027869ar) | [Pairs plot with correlations](gallery/04-correlation/03-pairs-plot/) |
| Dedinec, Stojkov, Warin & Kocarev (2026), the development compression trap | [Correlogram with clustering](gallery/04-correlation/04-correlogram-bubble/) |
| [Stojkov, Veljanovski & Warin (2024), European integration and growth in emerging Europe](https://doi.org/10.1080/14631377.2024.2376974) | [Coefficient plot](gallery/05-regression-models/01-coefficient-plot/) |
| [Blakely & Warin (2012), the decision to migrate in Europe](https://doi.org/10.1515/1524-5861.1818) | [Odds-ratio forest plot](gallery/05-regression-models/02-odds-ratio-forest/) |
| [Stojkov & Warin (2018), EU membership and FDI](https://doi.org/10.1080/10669868.2018.1431579) | [Comparing nested models](gallery/05-regression-models/03-compare-models/) |
| [Kouloukoui, de Marcellis-Warin & Warin (2025), public perceptions of AI](https://doi.org/10.1007/s00146-025-02232-x) | [Predicted probabilities](gallery/05-regression-models/04-predicted-probabilities/) |
| [Stojkov & Warin (2021), foreign capital and domestic funding conditions](https://doi.org/10.1142/s2194565921500081) | [Interaction effects](gallery/05-regression-models/05-interaction-effects/) |
| Troadec & Warin (2015), Canadian firms in global R&D | [Quantile regression process](gallery/05-regression-models/06-quantile-regression/) |
| Tengia, Wandschneider, Warin & Wunnava (2011), Southern African trade integration | [Residual diagnostics](gallery/06-model-diagnostics/01-residual-diagnostics/) · [Added-variable plots](gallery/06-model-diagnostics/05-partial-regression/) |
| [Warin & Stojkov (2021), banks' foreign claims after the 2008 crisis](https://doi.org/10.3390/jrfm14020061) | [Influence plot](gallery/06-model-diagnostics/02-influence-plot/) |
| [Marty & Warin (2025), algorithmic collusion and antitrust](https://doi.org/10.1016/j.ject.2024.10.001) | [ROC curves](gallery/06-model-diagnostics/03-roc-curve/) |
| [Zarea et al. (2026), supply-chain resilience and digitalisation](https://jitm.ut.ac.ir/article_108005.html) | [Calibration plot](gallery/06-model-diagnostics/04-calibration-plot/) |
| [Vorreuther & Warin (2021), patent velocity in China's pharmaceutical industry](https://doi.org/10.1016/j.dib.2021.106814) | [Kaplan–Meier curves](gallery/07-survival/01-kaplan-meier/) · [Cox hazard ratios](gallery/07-survival/02-cox-hazard-ratios/) · [Adjusted survival curves](gallery/07-survival/04-adjusted-survival/) |
| [Hadengue, de Marcellis-Warin & Warin (2015), reverse innovation in pharmaceuticals](https://doi.org/10.7202/1043076ar) | [Cumulative hazard](gallery/07-survival/03-cumulative-hazard/) |
| [Warin (2024), Statistics Canada's open economic data for teaching](https://doi.org/10.5070/t5.1868) | [STL decomposition](gallery/08-time-series/01-stl-decomposition/) · [SARIMA forecast](gallery/08-time-series/03-arima-forecast/) · [Seasonal subseries plot](gallery/08-time-series/04-seasonal-subseries/) |
| [Warin & Stojkov (2023), sovereign bond yield differentials in Europe](https://doi.org/10.3390/e25040630) | [ACF and PACF](gallery/08-time-series/02-acf-pacf/) |
| [Stojkov & Warin (2025), the Western Balkans and the future of Europe](https://doi.org/10.1007/978-3-031-86465-0) | [PCA biplot](gallery/09-multivariate/01-pca-biplot/) |
| [Aubert et al. (2026), digital skills in Canadian jobs](https://doi.org/10.1186/s12651-026-00440-6) | [Scree plot](gallery/09-multivariate/02-scree-plot/) |
| [Chaffa & Warin (2026), industrial clusters from growth co-movement](https://doi.org/10.1007/s10708-025-11564-6) | [Silhouette plot for k-means](gallery/09-multivariate/03-silhouette-clusters/) |
| [Leiter & Warin (2012), price dispersion on the internet](https://doi.org/10.1504/ijebr.2012.048776) | [Bootstrap distribution](gallery/10-inference-uncertainty/01-bootstrap-distribution/) |
| [Bisson & Warin (2026), AI-assisted reading of IPCC synthesis reports](https://doi.org/10.1371/journal.pclm.0000965) | [Bland–Altman plot](gallery/10-inference-uncertainty/04-bland-altman/) |

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

> Warin, T. (2026). *PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences* (Version 1.1.0) [Computer software]. https://warint.github.io/pystatlas/

```bibtex
@misc{warin2026statlas,
  author       = {Warin, Thierry},
  title        = {PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences},
  year         = {2026},
  version      = {1.1.0},
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
