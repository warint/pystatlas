# Group comparisons

Compare groups with the test, effect size and p-values on the chart — the ggstatsplot way.

| Chart | Description |
|:--|:--|
| [<img src="01-between-groups-violin/plot.png" width="260">](01-between-groups-violin/) | **[Between-groups comparison](01-between-groups-violin/)**<br>Violin + box + points with Welch's ANOVA, effect size and pairwise tests.<br>`ggstatsplot::ggbetweenstats()` |
| [<img src="02-within-subjects/plot.png" width="260">](02-within-subjects/) | **[Within-subjects comparison](02-within-subjects/)**<br>Repeated measures with one line per subject, means, rm-ANOVA and paired tests.<br>`ggstatsplot::ggwithinstats()` |
| [<img src="03-estimation-plot/plot.png" width="260">](03-estimation-plot/) | **[Estimation plot](03-estimation-plot/)**<br>Raw data of two groups plus the mean difference and its bootstrap 95% CI.<br>`dabestr::dabest() + plot()` |
| [<img src="04-means-with-ci/plot.png" width="260">](04-means-with-ci/) | **[Interaction plot](04-means-with-ci/)**<br>Group means with 95% confidence intervals for two factors, with a two-way ANOVA.<br>`ggpubr::ggline(add = "mean_ci") / ggstats` |
| [<img src="05-tukey-hsd/plot.png" width="260">](05-tukey-hsd/) | **[Tukey HSD intervals](05-tukey-hsd/)**<br>Simultaneous 95% confidence intervals for every pairwise difference in means.<br>`TukeyHSD() + plot()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/statlas/group-comparisons/)
