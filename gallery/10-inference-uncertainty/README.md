# Inference & uncertainty

Bootstrap, confidence intervals, power, agreement and meta-analysis forest plots.

| Chart | Description |
|:--|:--|
| [<img src="01-bootstrap-distribution/plot.png" width="260">](01-bootstrap-distribution/) | **[Bootstrap distribution](01-bootstrap-distribution/)**<br>Resampled correlation with percentile and BCa 95% confidence intervals.<br>`boot::boot() + boot::boot.ci()` |
| [<img src="02-confidence-intervals/plot.png" width="260">](02-confidence-intervals/) | **[What a 95% confidence interval means](02-confidence-intervals/)**<br>100 surveys of 30 people drawn from 944 ANES respondents; the 95% CIs for mean age that miss the true value are in red.<br>`teaching classic (e.g. TeachingDemos::ci.examp())` |
| [<img src="03-power-curve/plot.png" width="260">](03-power-curve/) | **[Power curve](03-power-curve/)**<br>Power of a two-sample t-test against sample size for small, medium and large effects.<br>`pwr::pwr.t.test() + plot()` |
| [<img src="04-bland-altman/plot.png" width="260">](04-bland-altman/) | **[Bland–Altman plot](04-bland-altman/)**<br>Do two coders place party manifestos on the left–right scale the same way? Bias and 95% limits of agreement.<br>`BlandAltmanLeh::bland.altman.plot() / blandr::blandr.draw()` |
| [<img src="05-meta-analysis-forest/plot.png" width="260">](05-meta-analysis-forest/) | **[Meta-analysis forest plot](05-meta-analysis-forest/)**<br>Do teacher expectations raise pupils' IQ? 19 "Pygmalion" experiments, a random-effects pooled diamond and heterogeneity statistics.<br>`meta::forest() / metafor::forest()` |
| [<img src="06-funnel-plot/plot.png" width="260">](06-funnel-plot/) | **[Funnel plot](06-funnel-plot/)**<br>Are small "Pygmalion" studies more optimistic? 19 teacher-expectancy effects against their standard error, with a pseudo 95% funnel and Egger's test.<br>`metafor::funnel() + metafor::regtest()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/inference-uncertainty/)
