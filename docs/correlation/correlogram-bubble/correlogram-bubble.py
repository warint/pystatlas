"""
title: Correlogram with clustering
summary: Corrplot-style circles sized and coloured by r, ordered by hierarchical clustering.
r: corrplot::corrplot(order = "hclust", addrect = 3)
tags: [correlation, correlogram, hierarchical clustering, heatmap, corrplot]
description: |
  The Python counterpart of `corrplot(method = "circle", order = "hclust")`. Each
  circle's **area** grows with |*r*| and its colour gives the sign, so strong
  correlations pop out and weak ones fade. The variables are reordered by
  **hierarchical clustering** (complete linkage on the distance 1 − *r*), which
  pulls positively correlated variables into blocks along the diagonal; the boxes
  outline the three clusters, like `addrect = 3`.

  Data are the 13 chemical measurements of 178 Italian wines bundled with
  **scikit-learn** (`load_wine`). The ordering comes from
  `scipy.cluster.hierarchy.linkage` and `leaves_list`, the boxes from `fcluster`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.datasets import load_wine

# Data: 13 chemical measurements of 178 wines (bundled with scikit-learn)
df = load_wine(as_frame=True).data
df.columns = [c.replace("_", " ").capitalize() for c in df.columns]
df = df.rename(columns={"Od280/od315 of diluted wines": "OD280/OD315"})
corr = df.corr()

# Hierarchical clustering on the distance 1 - r, then reorder the matrix
Z = linkage(squareform(1 - corr.to_numpy(), checks=False), method="complete")
order = leaves_list(Z)
corr = corr.iloc[order, order]
clusters = fcluster(Z, t=3, criterion="maxclust")[order]
k = len(corr)

fig, ax = plt.subplots(figsize=(8.4, 7.2))
cmap, norm = plt.get_cmap("RdBu"), plt.Normalize(-1, 1)
for i in range(k):
    for j in range(k):
        r = corr.iat[i, j]
        if i == j:
            continue
        ax.add_patch(Circle((j, i), radius=0.47 * np.sqrt(abs(r)), color=cmap(norm(r)), lw=0))

# Boxes around the clusters (consecutive runs along the diagonal)
start = 0
for idx in range(1, k + 1):
    if idx == k or clusters[idx] != clusters[start]:
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), idx - start, idx - start,
                               fill=False, ec="#1D1D1F", lw=1.6))
        start = idx

ax.set_xticks(range(k), corr.columns, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(k), corr.index)
ax.set_xticks(np.arange(-0.5, k), minor=True)
ax.set_yticks(np.arange(-0.5, k), minor=True)
ax.grid(which="minor", color="#E5E5EA", lw=0.8)
ax.grid(which="major", visible=False)
ax.tick_params(which="minor", length=0)
ax.set_xlim(-0.5, k - 0.5)
ax.set_ylim(k - 0.5, -0.5)
ax.set_aspect("equal")
for s in ax.spines.values():
    s.set_visible(False)

cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.02,
                  ticks=[-1, -0.5, 0, 0.5, 1])
cb.set_label("Pearson r", color="#3A3A3C")
cb.outline.set_visible(False)
ax.set_title("Phenolic compounds form the tightest block in wine chemistry", pad=26)
ax.text(0, 1.02, f"Pearson r · n = {len(df)} wines · complete-linkage clustering on 1 − r · "
        "circle area grows with |r|", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
