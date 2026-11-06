# Regression models

Coefficient plots, odds ratios, model comparison and marginal effects — ggcoef_model for Python.

| Chart | Description |
|:--|:--|
| [<img src="01-coefficient-plot/plot.png" width="260">](01-coefficient-plot/) | **[Coefficient plot](01-coefficient-plot/)**<br>Regression estimates with 95% confidence intervals, one row per term.<br>`ggstats::ggcoef_model()` |
| [<img src="02-odds-ratio-forest/plot.png" width="260">](02-odds-ratio-forest/) | **[Odds-ratio forest plot](02-odds-ratio-forest/)**<br>Logistic regression odds ratios on a log scale, with a table of OR [95% CI] alongside.<br>`ggstats::ggcoef_model(exponentiate = TRUE)` |
| [<img src="03-compare-models/plot.png" width="260">](03-compare-models/) | **[Comparing nested models](03-compare-models/)**<br>Coefficients of three nested regressions side by side, to see how estimates shift.<br>`ggstats::ggcoef_compare()` |
| [<img src="04-predicted-probabilities/plot.png" width="260">](04-predicted-probabilities/) | **[Predicted probabilities](04-predicted-probabilities/)**<br>Model-based probability curves with 95% CI ribbons across a predictor, by group.<br>`ggeffects::ggpredict() / marginaleffects::plot_predictions()` |
| [<img src="05-interaction-effects/plot.png" width="260">](05-interaction-effects/) | **[Interaction effects](05-interaction-effects/)**<br>Simple slopes at mean ± 1 SD and a Johnson–Neyman plot for two continuous predictors.<br>`interactions::interact_plot() / interactions::johnson_neyman()` |
| [<img src="06-quantile-regression/plot.png" width="260">](06-quantile-regression/) | **[Quantile regression process](06-quantile-regression/)**<br>How a slope changes across the conditional distribution, against the single OLS estimate.<br>`quantreg::plot(summary(rq(tau = 1:19 / 20)))` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/regression-models/)
