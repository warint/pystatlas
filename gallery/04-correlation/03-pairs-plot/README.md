# Pairs plot with correlations

> Scatter matrix of four regional growth variables, with group KDEs on the diagonal and pooled vs within-group correlations above it.

![Pairs plot with correlations](plot.png)

The Python counterpart of `GGally::ggpairs()`. Every pair of variables gets a
scatterplot below the diagonal, coloured by group; the diagonal shows each
variable's density per group; and the mirror cell above the diagonal prints
Pearson's *r*, overall (bold) and within each group, with significance stars.
Comparing the two is the point: pooled and within-group correlations can differ
sharply — even in sign — when groups sit in different regions of the plot.

The setting is Hadengue & Warin (2014, *Management international*) on
specialisation and (un)conditional convergence in Brazil, China and India. The
data are **simulated** for illustration and are *not* the paper's data or
results: 86 regions (27 Brazilian states, 31 Chinese provinces, 28 Indian
states) with average growth, log initial income per head, a specialisation index
and an export-diversification index. Within each country poorer regions grow
faster (conditional convergence), while the pooled correlation mixes in the gaps
between countries. Densities use `scipy.stats.gaussian_kde` and correlations
`scipy.stats.pearsonr`; the grid is a plain `plt.subplots(4, 4)`.

**R equivalent:** `GGally::ggpairs()`  
**Libraries:** Matplotlib, NumPy, pandas, SciPy

**Data:** Simulated data with a fixed random seed, created for this example. — No real respondents, documents or measurements: the numbers only illustrate how the chart and its statistics work.

**Data:** Hadengue, M., & Warin, T. (2014). Patterns of specialization and (un)conditional convergence: The cases of Brazil, China and India. Management international, 18, 123–141. https://doi.org/10.7202/1027869ar — The example is set in the context of this paper; unless its description says otherwise, the numbers are simulated and are not the paper's data or results.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Simulated data, set in the context of Hadengue & Warin (2014) [hadengue2014]

# Simulated regions: country means differ; within a country, poorer regions grow faster
rng = np.random.default_rng(2014)
groups = ["Brazil", "China", "India"]
setup = {"Brazil": (27, 9.2, 2.5), "China": (31, 9.0, 9.0), "India": (28, 7.4, 6.0)}
parts = []
for g, (n, inc0, gr0) in setup.items():
    dev = rng.normal(0, 0.45, n)                       # region's income gap to its country mean
    spec = 0.30 - 0.12 * dev + rng.normal(0, 0.05, n)
    parts.append(pd.DataFrame({"Growth (% per yr)": gr0 - 2.2 * dev + rng.normal(0, 0.7, n),
                               "Initial income (log)": inc0 + dev,
                               "Specialisation index": spec,
                               "Export diversification": 3.4 - 4 * spec + rng.normal(0, 0.25, n),
                               "group": g}))
df = pd.concat(parts, ignore_index=True)
cols = df.columns[:4]
palette = ["#007AFF", "#FF9500", "#34C759"]
stars = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

fig, axes = plt.subplots(4, 4, figsize=(9.5, 8.6), sharex="col")
lims = {c: (df[c].min() - 0.05 * np.ptp(df[c]), df[c].max() + 0.05 * np.ptp(df[c])) for c in cols}
fig.get_layout_engine().set(rect=(0, 0, 1, 0.92), w_pad=0.04, h_pad=0.04)
for i, yv in enumerate(cols):
    for j, xv in enumerate(cols):
        ax = axes[i, j]
        ax.grid(False)
        if i == j:                                       # diagonal: KDE per group
            grid = np.linspace(*lims[xv], 200)
            for s, c in zip(groups, palette):
                dens = stats.gaussian_kde(df.loc[df.group == s, xv])(grid)
                ax.fill_between(grid, dens, color=c, alpha=0.25, lw=0)
                ax.plot(grid, dens, color=c, lw=1.3)
            ax.set_yticks([])
        elif i > j:                                      # lower: scatter by group
            for s, c in zip(groups, palette):
                d = df[df.group == s]
                ax.scatter(d[xv], d[yv], s=12, color=c, alpha=0.65, lw=0)
            ax.set_ylim(lims[yv])
        else:                                            # upper: correlation coefficients
            r, p = stats.pearsonr(df[xv], df[yv])
            ax.text(0.5, 0.78, f"r = {r:.2f}{stars(p)}", ha="center", fontsize=11,
                    fontweight="semibold", transform=ax.transAxes)
            for k, (s, c) in enumerate(zip(groups, palette)):
                d = df[df.group == s]
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

fig.text(0.01, 0.985, "Within each country, poorer regions grow faster", fontsize=13,
         fontweight="semibold", va="top")
fig.text(0.01, 0.945, f"Pearson r, overall (bold) and within country · n = {len(df)} regions "
         "(simulated) · * p<.05  ** p<.01  *** p<.001", fontsize=9.5, color="#6E6E73", va="top")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/correlation/pairs-plot/) · [← Correlation & association](../README.md)

**Cite:** Warin, T. (2026). Pairs plot with correlations. In PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences. https://warint.github.io/pystatlas/correlation/pairs-plot/
