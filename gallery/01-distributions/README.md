# Distributions

See the shape of your data: histograms, densities, ridgelines, ECDFs and Q–Q plots.

| Chart | Description |
|:--|:--|
| [<img src="01-histogram-density/plot.png" width="260">](01-histogram-density/) | **[Histogram with density](01-histogram-density/)**<br>Histogram, kernel density and fitted normal curve, with shape statistics.<br>`ggplot2::geom_histogram() + geom_density()` |
| [<img src="02-raincloud/plot.png" width="260">](02-raincloud/) | **[Raincloud plot](02-raincloud/)**<br>Half violin, box plot and raw points per group, drawn horizontally.<br>`ggdist::stat_halfeye() / ggrain::geom_rain()` |
| [<img src="03-ridgeline/plot.png" width="260">](03-ridgeline/) | **[Ridgeline plot](03-ridgeline/)**<br>Overlapping density curves to compare one distribution across many groups.<br>`ggridges::geom_density_ridges()` |
| [<img src="04-ecdf-comparison/plot.png" width="260">](04-ecdf-comparison/) | **[ECDF comparison](04-ecdf-comparison/)**<br>Empirical cumulative distributions of two groups, with the Kolmogorov–Smirnov gap.<br>`ggplot2::stat_ecdf()` |
| [<img src="05-qq-plot/plot.png" width="260">](05-qq-plot/) | **[Q–Q plot with confidence band](05-qq-plot/)**<br>Normal quantile–quantile plots with a 95% envelope, for a raw and a logged variable.<br>`qqplotr::stat_qq_band()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/statlas/distributions/)
