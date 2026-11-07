# Machine learning

How well does a model predict on new data, and why? Out-of-fold evaluation, learning curves and model-agnostic explanations — each with its uncertainty.

| Chart | Description |
|:--|:--|
| [<img src="01-confusion-matrix/plot.png" width="260">](01-confusion-matrix/) | **[Confusion matrix with metrics](01-confusion-matrix/)**<br>Out-of-fold confusion matrix with counts and row shares, and bootstrap intervals for the headline metrics.<br>`yardstick::conf_mat() + autoplot()` |
| [<img src="02-precision-recall/plot.png" width="260">](02-precision-recall/) | **[Precision–recall curves](02-precision-recall/)**<br>Out-of-fold precision–recall curves for two classifiers, with average precision and the no-skill baseline.<br>`yardstick::pr_curve() + autoplot()` |
| [<img src="03-learning-curve/plot.png" width="260">](03-learning-curve/) | **[Learning curve](03-learning-curve/)**<br>Training and cross-validated score as the training set grows, with fold-to-fold spread.<br>`caret::learning_curve_dat() + ggplot()` |
| [<img src="04-model-comparison/plot.png" width="260">](04-model-comparison/) | **[Cross-validated model comparison](04-model-comparison/)**<br>Fold-by-fold scores of several models, with means and corrected 95% intervals.<br>`caret::resamples() + bwplot() / tune::autoplot()` |
| [<img src="05-permutation-importance/plot.png" width="260">](05-permutation-importance/) | **[Permutation importance](05-permutation-importance/)**<br>Drop in test-set R² when each feature is shuffled, over 50 repeats per feature.<br>`vip::vip(method = "permute")` |
| [<img src="06-partial-dependence/plot.png" width="260">](06-partial-dependence/) | **[Partial dependence with ICE curves](06-partial-dependence/)**<br>Average model response to one feature (PDP) over the individual curves (ICE) of every patient.<br>`pdp::partial(ice = TRUE) + autoplot() / iml::FeatureEffect` |
| [<img src="07-shap-beeswarm/plot.png" width="260">](07-shap-beeswarm/) | **[SHAP summary (beeswarm)](07-shap-beeswarm/)**<br>Every patient's SHAP value for every feature, coloured by the feature's value, with mean |SHAP| importance.<br>`shapviz::sv_importance(kind = "beeswarm")` |
| [<img src="08-shap-dependence/plot.png" width="260">](08-shap-dependence/) | **[SHAP dependence plot](08-shap-dependence/)**<br>SHAP value of one feature against its value, coloured by its strongest interaction partner, with a smooth trend.<br>`shapviz::sv_dependence()` |
| [<img src="09-shap-waterfall/plot.png" width="260">](09-shap-waterfall/) | **[SHAP waterfall for one prediction](09-shap-waterfall/)**<br>How each feature moves one patient's prediction from the average to the model's output.<br>`shapviz::sv_waterfall()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/machine-learning/)
