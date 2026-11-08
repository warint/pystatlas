# Proportions & surveys

Proportions, cross-tabs and Likert scales, in the spirit of ggstats::gglikert and stat_prop.

| Chart | Description |
|:--|:--|
| [<img src="01-likert-diverging/plot.png" width="260">](01-likert-diverging/) | **[Diverging Likert chart](01-likert-diverging/)**<br>Five-point Likert items as diverging stacked bars centred on the neutral answer.<br>`ggstats::gglikert()` |
| [<img src="02-stacked-proportions/plot.png" width="260">](02-stacked-proportions/) | **[Stacked proportions by group](02-stacked-proportions/)**<br>100% stacked bars with percentage labels, n per bar, chi-squared test and Cramér's V.<br>`ggstatsplot::ggbarstats()` |
| [<img src="03-mosaic-plot/plot.png" width="260">](03-mosaic-plot/) | **[Mosaic plot with Pearson residuals](03-mosaic-plot/)**<br>Tile areas proportional to cell counts, shaded by Pearson residuals from independence.<br>`vcd::mosaic(shade = TRUE)` |
| [<img src="04-proportions-with-ci/plot.png" width="260">](04-proportions-with-ci/) | **[Proportions with confidence intervals](04-proportions-with-ci/)**<br>A sorted dot plot of proportions with Wilson 95% confidence intervals.<br>`ggstats::stat_prop() + geom_errorbar()` |
| [<img src="05-crosstab-residuals/plot.png" width="260">](05-crosstab-residuals/) | **[Cross-table with standardized residuals](05-crosstab-residuals/)**<br>A contingency table as a heatmap, coloured by standardized residuals from independence.<br>`ggstats::ggtable() / stat_cross()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/proportions-surveys/)
