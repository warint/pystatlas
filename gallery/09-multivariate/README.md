# Multivariate & clustering

PCA biplots, scree plots, clustering and heatmaps for high-dimensional data.

| Chart | Description |
|:--|:--|
| [<img src="01-pca-biplot/plot.png" width="260">](01-pca-biplot/) | **[PCA biplot](01-pca-biplot/)**<br>Observations and variable loadings on the first two principal components.<br>`factoextra::fviz_pca_biplot()` |
| [<img src="02-scree-plot/plot.png" width="260">](02-scree-plot/) | **[Scree plot](02-scree-plot/)**<br>Variance explained per component, cumulative share and a parallel-analysis cut-off.<br>`factoextra::fviz_eig()` |
| [<img src="03-silhouette-clusters/plot.png" width="260">](03-silhouette-clusters/) | **[Silhouette plot for k-means](03-silhouette-clusters/)**<br>k-means clusters in PCA space next to the per-observation silhouette widths.<br>`factoextra::fviz_silhouette()` |
| [<img src="04-clustered-heatmap/plot.png" width="260">](04-clustered-heatmap/) | **[Clustered heatmap](04-clustered-heatmap/)**<br>Z-scored heatmap with row and column dendrograms and a group colour bar.<br>`pheatmap::pheatmap() / ComplexHeatmap::Heatmap()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/multivariate/)
