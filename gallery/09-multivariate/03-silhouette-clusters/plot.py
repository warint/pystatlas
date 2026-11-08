"""
title: Silhouette plot for k-means
summary: Do industries fall into clusters of growth co-movement? k-means clusters next to their silhouette widths.
r: factoextra::fviz_silhouette()
tags: [clustering, k-means, silhouette, PCA, cluster validation, industrial clusters]
description: |
  The silhouette width compares how close each observation is to its own cluster
  with how close it is to the nearest other cluster: near 1 means well placed,
  near 0 means on the border, negative means probably misassigned. The number of
  clusters k is the one with the highest average silhouette (2 to 8 tried).
  The left panel shows the clusters on the first two principal components; the
  right panel sorts the silhouette widths within each cluster, with the dashed
  line at the overall average.

  The setting is Chaffa & Warin (2026), "Identifying industrial clusters through
  growth co-movement with unsupervised learning methods" (*GeoJournal*): 90
  industries, each described by 5 standardised co-movement features — the
  correlation of its output growth with national GDP, US demand, commodity
  prices and housing starts, and its growth volatility. The data are
  **simulated** with `numpy` (seeded) around three profiles (domestic cycle,
  export-oriented, resource-based) for illustration; they are *not* the paper's
  data or results. An average width near 0.5 sits at the border between a
  weak and a reasonable structure by Kaufman & Rousseeuw's rule of thumb.
  `KMeans`, `silhouette_score` and `silhouette_samples` from
  `sklearn` do the work; `PCA` is only used to draw the clusters.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

# Simulated data, set in the context of Chaffa & Warin (2026) [chaffa2026]
# 90 industries x 5 growth co-movement features around three profiles
rng = np.random.default_rng(91)
centres = np.array([[0.7, 0.2, 0.0, 0.6, 0.04],    # domestic cycle
                    [0.4, 0.7, 0.1, 0.1, 0.06],    # export-oriented
                    [0.2, 0.3, 0.7, 0.0, 0.10]])   # resource-based
profile = rng.choice(3, 90, p=[0.45, 0.33, 0.22])
X = StandardScaler().fit_transform(centres[profile] + rng.normal(0, [0.15, 0.15, 0.15, 0.15, 0.02],
                                                                 (90, 5)))

# Choose k by the average silhouette width
fits = {k: KMeans(k, n_init=20, random_state=1).fit(X) for k in range(2, 9)}
avg = {k: silhouette_score(X, m.labels_) for k, m in fits.items()}
k = max(avg, key=avg.get)
labels = fits[k].labels_
sil = silhouette_samples(X, labels)
colors = ["#007AFF", "#FF9500", "#34C759", "#FF3B30", "#AF52DE", "#5AC8FA", "#8E8E93", "#A2845E"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), width_ratios=[1, 1.15])

# Left: clusters projected on the first two principal components
pca = PCA(2).fit(X)
S, C = pca.transform(X), pca.transform(fits[k].cluster_centers_)
for j in range(k):
    ax1.scatter(*S[labels == j].T, s=22, color=colors[j], alpha=0.6, lw=0)
    ax1.scatter(*C[j], s=150, marker="X", color=colors[j], edgecolor="white", lw=1.5, zorder=3)
ax1.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
ax1.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
ax1.grid(axis="x")

# Right: silhouette widths, sorted within cluster
start = 0
for j in range(k):
    s = np.sort(sil[labels == j])[::-1]
    x = np.arange(start, start + len(s))
    ax2.fill_between(x, 0, s, color=colors[j], step="mid", lw=0, alpha=0.85)
    ax2.text(x.mean(), -0.05, f"Cluster {j + 1}\nn = {len(s)}\nmean {s.mean():.2f}",
             ha="center", va="top", fontsize=9, color=colors[j], fontweight="semibold")
    start += len(s) + 6
ax2.axhline(avg[k], color="#3A3A3C", ls="--", lw=1)
ax2.text(start - 6, avg[k] + 0.015, f"average {avg[k]:.2f}", ha="right", va="bottom",
         fontsize=9, color="#3A3A3C")
ax2.axhline(0, color="#8E8E93", lw=0.8)
ax2.set_xticks([])
ax2.set_xlim(-3, start - 3)
ax2.set_ylim(-0.18, max(sil) + 0.08)
ax2.set_ylabel("Silhouette width")
ax2.spines["bottom"].set_visible(False)

ax1.set_title(f"Silhouette picks k = {k} co-movement clusters of industries",
              pad=26)
ax1.text(0, 1.02, "Average silhouette for k = 2…8: " + ", ".join(f"{v:.2f}" for v in avg.values())
         + f" · {(sil < 0).sum()} industries with width < 0 · n = {len(X)} industries (simulated), "
         f"{X.shape[1]} features", transform=ax1.transAxes, fontsize=9.5,
         color="#6E6E73", in_layout=False)
plt.show()
