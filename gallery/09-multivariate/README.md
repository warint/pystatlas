# Multivariate & clustering

PCA biplots, scree plots, clustering and heatmaps for high-dimensional data.

| Chart | Description |
|:--|:--|
| [<img src="01-pca-biplot/plot.png" width="260">](01-pca-biplot/) | **[PCA biplot](01-pca-biplot/)**<br>US states and their social indicators on the first two principal components.<br>`factoextra::fviz_pca_biplot()` |
| [<img src="02-scree-plot/plot.png" width="260">](02-scree-plot/) | **[Scree plot](02-scree-plot/)**<br>How many dimensions describe 303 California school districts? Variance per component and a parallel-analysis cut-off.<br>`factoextra::fviz_eig()` |
| [<img src="03-silhouette-clusters/plot.png" width="260">](03-silhouette-clusters/) | **[Silhouette plot for k-means](03-silhouette-clusters/)**<br>Do US states fall into distinct social profiles? k-means clusters next to their silhouette widths.<br>`factoextra::fviz_silhouette()` |
| [<img src="04-clustered-heatmap/plot.png" width="260">](04-clustered-heatmap/) | **[Clustered heatmap](04-clustered-heatmap/)**<br>Z-scored heatmap of US states' social indicators, with row and column dendrograms and a Census-region colour bar.<br>`pheatmap::pheatmap() / ComplexHeatmap::Heatmap()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/multivariate/)
