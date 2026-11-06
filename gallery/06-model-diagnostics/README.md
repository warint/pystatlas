# Model diagnostics

Residuals, leverage, calibration and ROC curves: check before you trust a model.

| Chart | Description |
|:--|:--|
| [<img src="01-residual-diagnostics/plot.png" width="260">](01-residual-diagnostics/) | **[Residual diagnostics](01-residual-diagnostics/)**<br>The classic 2 × 2 panel for a linear model — linearity, normality, variance, leverage.<br>`plot(lm) / performance::check_model()` |
| [<img src="02-influence-plot/plot.png" width="260">](02-influence-plot/) | **[Influence plot](02-influence-plot/)**<br>Studentized residuals vs leverage, with bubbles sized by Cook's distance.<br>`car::influencePlot()` |
| [<img src="03-roc-curve/plot.png" width="260">](03-roc-curve/) | **[ROC curves](03-roc-curve/)**<br>Three classifiers compared by ROC curve, AUC with bootstrap 95% CI and Youden point.<br>`pROC::ggroc() / pROC::ci.auc()` |
| [<img src="04-calibration-plot/plot.png" width="260">](04-calibration-plot/) | **[Calibration plot](04-calibration-plot/)**<br>Predicted vs observed risk per decile, with Wilson CIs, prediction histograms and Brier score.<br>`rms::calibrate() / probably::cal_plot_breaks()` |
| [<img src="05-partial-regression/plot.png" width="260">](05-partial-regression/) | **[Added-variable plots](05-partial-regression/)**<br>One partial-regression panel per predictor; each fitted slope equals its coefficient.<br>`car::avPlots()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/model-diagnostics/)
