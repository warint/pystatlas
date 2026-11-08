# Correlogram with clustering

> Corrplot-style circles sized and coloured by r for 12 development indicators, ordered by hierarchical clustering.

![Correlogram with clustering](plot.png)

The Python counterpart of `corrplot(method = "circle", order = "hclust")`. Each
circle's **area** grows with |*r*| and its colour gives the sign, so strong
correlations pop out and weak ones fade. The variables are reordered by
**hierarchical clustering** (complete linkage on the distance 1 − *r*), which
pulls positively correlated variables into blocks along the diagonal; the boxes
outline the three clusters, like `addrect = 3`.

The setting is Dedinec, Stojkov, Warin & Kocarev (2026, *Journal of Sustainable
Development of Energy, Water and Environment Systems*) on uneven development
trajectories. The data are **simulated** for illustration and are *not* the
paper's data or results: 150 countries scored on 12 indicators of income,
energy, emissions, health, education and economic structure, generated from
three correlated latent factors so that blocks appear. The ordering comes from
`scipy.cluster.hierarchy.linkage` and `leaves_list`, the boxes from `fcluster`.

**R equivalent:** `corrplot::corrplot(order = "hclust", addrect = 3)`  
**Libraries:** Matplotlib, NumPy, pandas, SciPy

**Data:** Simulated data with a fixed random seed, created for this example. — No real respondents, documents or measurements: the numbers only illustrate how the chart and its statistics work.

**Data:** Dedinec, A., Stojkov, A., Warin, T., & Kocarev, L. (2026). The development compression trap: A framework for understanding uneven development trajectories. Journal of Sustainable Development of Energy, Water and Environment Systems, 15(1). — The example is set in the context of this paper; unless its description says otherwise, the numbers are simulated and are not the paper's data or results.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

# Simulated data, set in the context of Dedinec, Stojkov, Warin & Kocarev (2026) [dedinec2026]

# Simulated countries: 12 indicators driven by three correlated latent factors
rng = np.random.default_rng(2026)
n = 150
income = rng.normal(0, 1, n)
human = 0.35 * income + 0.94 * rng.normal(0, 1, n)
green = -0.5 * income + 0.85 * rng.normal(0, 1, n)
loads = {"GDP per head (log)": (income, .9), "Energy use per head": (income, .8),
         "CO₂ per head": (income, .75), "Urban population": (income, .65),
         "Internet users": (income, .7), "Life expectancy": (human, .85),
         "Years of schooling": (human, .8), "Electricity access": (human, .7),
         "Child survival": (human, .8), "Renewables share": (green, .8),
         "Agriculture (% GDP)": (green, .7), "Forest cover": (green, .55)}
df = pd.DataFrame({k: w * f + np.sqrt(1 - w**2) * rng.normal(0, 1, n) for k, (f, w) in loads.items()})
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
r_ge = corr.loc["GDP per head (log)", "Energy use per head"]
ax.set_title(f"Income, energy and emissions form one block (r = {r_ge:.2f}, GDP–energy)", pad=26)
ax.text(0, 1.02, f"Pearson r · n = {len(df)} countries (simulated) · complete linkage clustering on 1 − r · "
        "circle area grows with |r|", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/correlation/correlogram-bubble/) · [← Correlation & association](../README.md)

**Cite:** Warin, T. (2026). Correlogram with clustering. In PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences. https://warint.github.io/pystatlas/correlation/correlogram-bubble/
