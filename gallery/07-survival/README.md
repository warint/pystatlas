# Survival analysis

Kaplan–Meier curves with risk tables, cumulative hazards and Cox model hazard ratios.

| Chart | Description |
|:--|:--|
| [<img src="01-kaplan-meier/plot.png" width="260">](01-kaplan-meier/) | **[Kaplan–Meier curves](01-kaplan-meier/)**<br>Time from patent filing to grant, domestic vs foreign applicants, with log-rank test and risk table.<br>`survminer::ggsurvplot(risk.table = TRUE)` |
| [<img src="02-cox-hazard-ratios/plot.png" width="260">](02-cox-hazard-ratios/) | **[Cox hazard ratios](02-cox-hazard-ratios/)**<br>Forest plot of a Cox model of time to patent grant, with HR, 95% CI and p-values.<br>`survminer::ggforest()` |
| [<img src="03-cumulative-hazard/plot.png" width="260">](03-cumulative-hazard/) | **[Cumulative hazard](03-cumulative-hazard/)**<br>Nelson–Aalen cumulative hazard of a drug's launch, with 95% bands, in four market groups.<br>`survminer::ggsurvplot(fun = "cumhaz")` |
| [<img src="04-adjusted-survival/plot.png" width="260">](04-adjusted-survival/) | **[Adjusted survival curves](04-adjusted-survival/)**<br>Cox-predicted time to patent grant at four levels of relatedness, other covariates at their mean.<br>`survminer::ggadjustedcurves()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/survival/)
