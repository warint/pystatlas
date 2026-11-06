# Correlation & association

Correlation matrices, scatterplots with statistics and pairwise relationships.

| Chart | Description |
|:--|:--|
| [<img src="01-correlation-matrix/plot.png" width="260">](01-correlation-matrix/) | **[Correlation matrix](01-correlation-matrix/)**<br>Lower-triangle Pearson correlation heatmap; non-significant cells (Holm) are crossed out.<br>`ggstatsplot::ggcorrmat()` |
| [<img src="02-scatter-with-stats/plot.png" width="260">](02-scatter-with-stats/) | **[Scatterplot with statistics](02-scatter-with-stats/)**<br>Scatter, OLS line with 95% band and marginal histograms, with Pearson's r in the subtitle.<br>`ggstatsplot::ggscatterstats()` |
| [<img src="03-pairs-plot/plot.png" width="260">](03-pairs-plot/) | **[Pairs plot with correlations](03-pairs-plot/)**<br>Scatter matrix with group KDEs on the diagonal and correlation coefficients above it.<br>`GGally::ggpairs()` |
| [<img src="04-correlogram-bubble/plot.png" width="260">](04-correlogram-bubble/) | **[Correlogram with clustering](04-correlogram-bubble/)**<br>Corrplot-style circles sized and coloured by r, ordered by hierarchical clustering.<br>`corrplot::corrplot(order = "hclust", addrect = 3)` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/correlation/)
