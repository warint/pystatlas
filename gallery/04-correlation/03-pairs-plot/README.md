# Pairs plot with correlations

> Scatter matrix with group KDEs on the diagonal and correlation coefficients above it.

![Pairs plot with correlations](plot.png)

The Python counterpart of `GGally::ggpairs()`. Every pair of variables gets a
scatterplot below the diagonal, coloured by group; the diagonal shows each
variable's density per group; and the mirror cell above the diagonal prints
Pearson's *r*, overall (bold) and within each group, with significance stars.
Comparing the two is the point: pooled and within-group correlations can differ
sharply — even in sign — when groups sit in different regions of the plot.

Data are Fisher's iris measurements bundled with **scikit-learn** (`load_iris`),
150 flowers from three species. Densities use `scipy.stats.gaussian_kde` and
correlations `scipy.stats.pearsonr`; the grid is a plain `plt.subplots(4, 4)`.

**R equivalent:** `GGally::ggpairs()`  
**Libraries:** Matplotlib, NumPy, SciPy, scikit-learn

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris

# Data: Fisher's iris (bundled with scikit-learn)
iris = load_iris(as_frame=True)
df = iris.frame.rename(columns=lambda c: c.replace(" (cm)", "").capitalize())
df["species"] = iris.target_names[iris.target]
cols = df.columns[:4]
species, palette = ["setosa", "versicolor", "virginica"], ["#007AFF", "#FF9500", "#34C759"]
stars = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

fig, axes = plt.subplots(4, 4, figsize=(9.5, 8.6), sharex="col")
lims = {c: (df[c].min() - 0.3, df[c].max() + 0.3) for c in cols}
fig.get_layout_engine().set(rect=(0, 0, 1, 0.92), w_pad=0.04, h_pad=0.04)
for i, yv in enumerate(cols):
    for j, xv in enumerate(cols):
        ax = axes[i, j]
        ax.grid(False)
        if i == j:                                       # diagonal: KDE per species
            grid = np.linspace(*lims[xv], 200)
            for s, c in zip(species, palette):
                dens = stats.gaussian_kde(df.loc[df.species == s, xv])(grid)
                ax.fill_between(grid, dens, color=c, alpha=0.25, lw=0)
                ax.plot(grid, dens, color=c, lw=1.3)
            ax.set_yticks([])
        elif i > j:                                      # lower: scatter by species
            for s, c in zip(species, palette):
                d = df[df.species == s]
                ax.scatter(d[xv], d[yv], s=12, color=c, alpha=0.65, lw=0)
            ax.set_ylim(lims[yv])
        else:                                            # upper: correlation coefficients
            r, p = stats.pearsonr(df[xv], df[yv])
            ax.text(0.5, 0.78, f"r = {r:.2f}{stars(p)}", ha="center", fontsize=11,
                    fontweight="semibold", transform=ax.transAxes)
            for k, (s, c) in enumerate(zip(species, palette)):
                d = df[df.species == s]
                rs, ps = stats.pearsonr(d[xv], d[yv])
                ax.text(0.5, 0.52 - k * 0.17, f"{s}: {rs:.2f}{stars(ps)}", ha="center",
                        fontsize=9, color=c, transform=ax.transAxes)
            ax.tick_params(labelbottom=False, labelleft=False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        if i < 3:
            ax.tick_params(labelbottom=False)
        if j > 0 or i == 0:
            ax.tick_params(labelleft=False)
        if i == 3:
            ax.set_xlabel(xv)
            ax.set_xlim(lims[xv])
        if j == 0:
            ax.set_ylabel(yv)

fig.text(0.01, 0.985, "Pooled correlations hide what happens within species", fontsize=13,
         fontweight="semibold", va="top")
fig.text(0.01, 0.945, f"Pearson r, overall (bold) and within species · n = {len(df)} flowers, 50 per "
         "species · * p<.05  ** p<.01  *** p<.001", fontsize=9.5, color="#6E6E73", va="top")
plt.show()
```

---

**Statlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/correlation/pairs-plot/) · [← Correlation & association](../README.md)

**Cite:** Warin, T. (2026). Pairs plot with correlations. In Statlas: The Python Gallery of Statistical Graphics. https://warint.github.io/pystatlas/correlation/pairs-plot/
