# Clustered heatmap

> Z-scored heatmap with row and column dendrograms and a group colour bar.

![Clustered heatmap](plot.png)

Each row is an observation, each column a variable, and each cell its z-score
(blue below the variable's mean, red above). Rows and columns are reordered by
**hierarchical clustering** (Ward linkage on Euclidean distance) and the
dendrograms show the merge history. The colour bar on the left gives the known
group of each row, so you can see at a glance whether the unsupervised tree
rediscovers it — and which variables drive the split.

Data are the 178 wines of the **scikit-learn** wine dataset (three cultivars,
13 measurements). `seaborn.clustermap` does the clustering and drawing;
`scipy.cluster.hierarchy` cuts the row tree into three clusters to compute
the adjusted Rand index against the true cultivars.

**R equivalent:** `pheatmap::pheatmap() / ComplexHeatmap::Heatmap()`  
**Libraries:** Matplotlib, NumPy, pandas, seaborn, SciPy, scikit-learn

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import fcluster
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score

# Data: wine measurements, z-scored per column; cultivar as row annotation
wine = load_wine(as_frame=True)
Z = (wine.data - wine.data.mean()) / wine.data.std()
Z.columns = [c.replace("_", " ").capitalize().replace("Od280/od315 of diluted wines", "OD280/OD315")
             for c in Z.columns]
names = np.array(["Barolo", "Grignolino", "Barbera"])
palette = dict(zip(names, ["#007AFF", "#FF9500", "#34C759"]))
cultivar = pd.Series(names[wine.target], name="Cultivar")

with plt.rc_context({"figure.constrained_layout.use": False}):
    g = sns.clustermap(Z, method="ward", metric="euclidean", cmap="RdBu_r", center=0,
                       vmin=-3, vmax=3, row_colors=cultivar.map(palette), yticklabels=False,
                       figsize=(8.5, 8), dendrogram_ratio=(0.14, 0.12), colors_ratio=0.025,
                       cbar_pos=(0.02, 0.83, 0.025, 0.11),
                       cbar_kws=dict(label="z-score", ticks=[-3, 0, 3]),
                       tree_kws=dict(colors="#8E8E93", linewidths=0.8))

# How well does the row tree, cut into three clusters, recover the cultivars?
ari = adjusted_rand_score(wine.target, fcluster(g.dendrogram_row.linkage, 3, "maxclust"))

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=40, ha="right", fontsize=9.5)
g.ax_heatmap.set(xlabel="", ylabel="")
g.ax_heatmap.tick_params(axis="x", length=0)
g.ax_row_colors.set_xticklabels(["Cultivar"], rotation=40, ha="right", fontsize=9.5)
g.ax_row_colors.tick_params(axis="x", length=0)
g.ax_cbar.tick_params(labelsize=8.5)
g.ax_cbar.yaxis.label.set_size(9)
g.ax_col_dendrogram.legend(handles=[Patch(color=c, label=n) for n, c in palette.items()],
                           loc="upper left", bbox_to_anchor=(1.0, 1.05), fontsize=9.5,
                           title="Cultivar", frameon=False, handlelength=1, handleheight=1)

g.figure.text(0.02, 1.04, "Hierarchical clustering largely rediscovers the wine cultivars",
              fontsize=13, fontweight="semibold", color="#1D1D1F")
g.figure.text(0.02, 1.01, f"Ward linkage on z-scores · n = {len(Z)} wines × {Z.shape[1]} "
              f"variables · 3-cluster cut vs cultivar: ARI = {ari:.2f}",
              fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**Statlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/statlas/multivariate/clustered-heatmap/) · [← Multivariate & clustering](../README.md)

**Cite:** Warin, T. (2026). Clustered heatmap. In Statlas: The Python Gallery of Statistical Graphics. https://warint.github.io/statlas/multivariate/clustered-heatmap/
