# Machine learning

How well does a model predict on new data, and why? Out-of-fold evaluation, learning curves and model-agnostic explanations — each with its uncertainty.

| Chart | Description |
|:--|:--|
| [<img src="01-confusion-matrix/plot.png" width="260">](01-confusion-matrix/) | **[Confusion matrix with metrics](01-confusion-matrix/)**<br>How well a logistic regression predicts who voted Dole in 1996 — out-of-fold confusion matrix with counts and row shares, and bootstrap intervals for the headline metrics.<br>`yardstick::conf_mat() + autoplot()` |
| [<img src="02-precision-recall/plot.png" width="260">](02-precision-recall/) | **[Precision–recall curves](02-precision-recall/)**<br>Out-of-fold precision–recall curves for two classifiers predicting which married women report an affair, with average precision and the no-skill baseline.<br>`yardstick::pr_curve() + autoplot()` |
| [<img src="03-learning-curve/plot.png" width="260">](03-learning-curve/) | **[Learning curve](03-learning-curve/)**<br>Training and cross-validated accuracy as the training set grows, for two models predicting which married women report an affair.<br>`caret::learning_curve_dat() + ggplot()` |
| [<img src="04-model-comparison/plot.png" width="260">](04-model-comparison/) | **[Cross-validated model comparison](04-model-comparison/)**<br>Fold-by-fold R² of six models predicting California school districts' maths results, with means and corrected 95% intervals.<br>`caret::resamples() + bwplot() / tune::autoplot()` |
| [<img src="05-permutation-importance/plot.png" width="260">](05-permutation-importance/) | **[Permutation importance](05-permutation-importance/)**<br>Which district characteristics a random forest relies on to predict maths results — drop in test-set R² when each one is shuffled.<br>`vip::vip(method = "permute")` |
| [<img src="06-partial-dependence/plot.png" width="260">](06-partial-dependence/) | **[Partial dependence with ICE curves](06-partial-dependence/)**<br>How predicted maths results change with poverty and with spending per pupil — the average (PDP) over one curve per school district (ICE).<br>`pdp::partial(ice = TRUE) + autoplot() / iml::FeatureEffect` |
| [<img src="07-shap-beeswarm/plot.png" width="260">](07-shap-beeswarm/) | **[SHAP summary (beeswarm)](07-shap-beeswarm/)**<br>Every school district's SHAP value for every characteristic, coloured by the characteristic's value, with mean |SHAP| importance.<br>`shapviz::sv_importance(kind = "beeswarm")` |
| [<img src="08-shap-dependence/plot.png" width="260">](08-shap-dependence/) | **[SHAP dependence plot](08-shap-dependence/)**<br>How the share of low-income students feeds into each district's predicted maths results, coloured by its strongest interaction partner, with a smooth trend.<br>`shapviz::sv_dependence()` |
| [<img src="09-shap-waterfall/plot.png" width="260">](09-shap-waterfall/) | **[SHAP waterfall for one prediction](09-shap-waterfall/)**<br>How each characteristic moves one struggling school district's predicted maths results from the average to the model's output.<br>`shapviz::sv_waterfall()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/machine-learning/)
