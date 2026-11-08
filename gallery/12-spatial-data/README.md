# Spatial data

Is a value higher where its neighbours' values are higher? Spatial weights, global and local spatial autocorrelation, and how far clusters reach — along a schematic Great Lakes–St. Lawrence corridor.

| Chart | Description |
|:--|:--|
| [<img src="01-spatial-weights/plot.png" width="260">](01-spatial-weights/) | **[Choropleth with spatial weights](01-spatial-weights/)**<br>A map of one variable with the neighbour network drawn on top — the structure every spatial statistic relies on.<br>`spdep::poly2nb() + plot(nb, coords)` |
| [<img src="02-moran-scatterplot/plot.png" width="260">](02-moran-scatterplot/) | **[Moran scatterplot (global autocorrelation)](02-moran-scatterplot/)**<br>Each area against the average of its neighbours, with Moran's I, its permutation test and the reference distribution.<br>`spdep::moran.plot() + spdep::moran.mc()` |
| [<img src="03-lisa-cluster-map/plot.png" width="260">](03-lisa-cluster-map/) | **[LISA cluster map (local autocorrelation)](03-lisa-cluster-map/)**<br>Where the clusters and spatial outliers are — local Moran's I for each area, mapped with its permutation significance.<br>`spdep::localmoran_perm() / rgeoda::local_moran()` |
| [<img src="04-spatial-correlogram/plot.png" width="260">](04-spatial-correlogram/) | **[Spatial correlogram](04-spatial-correlogram/)**<br>Moran's I at increasing neighbour orders, with a permutation envelope — how far spatial clustering reaches.<br>`spdep::sp.correlogram(method = "I")` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/spatial-data/)
