# Raincloud plot

> Half violin, box plot and raw points per group, drawn horizontally.

![Raincloud plot](plot.png)

A raincloud plot stacks three views of the same data: a half violin (the
"cloud") shows the density, a slim box plot gives the median and quartiles,
and the jittered points underneath (the "rain") show every single
observation. Nothing is hidden, so bimodality, outliers and small samples are
all visible.

Data are the iris flowers bundled with **scikit-learn**: sepal length for three
species. The densities come from `scipy.stats.gaussian_kde`, and the subtitle
reports a Kruskal–Wallis test (`scipy.stats.kruskal`) with ε² as effect size.

**R equivalent:** `ggdist::stat_halfeye() / ggrain::geom_rain()`  
**Libraries:** Matplotlib, NumPy, SciPy, scikit-learn

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris

# Data: sepal length of three iris species
iris = load_iris(as_frame=True)
df = iris.frame.assign(species=iris.target.map(dict(enumerate(iris.target_names))))
groups = ["setosa", "versicolor", "virginica"]
data = [df.loc[df.species == g, "sepal length (cm)"].to_numpy() for g in groups]
palette = ["#007AFF", "#FF9500", "#34C759"]

# Kruskal–Wallis test + epsilon squared
H, p = stats.kruskal(*data)
n = sum(len(d) for d in data)
eps2 = H / (n - 1)

fig, ax = plt.subplots(figsize=(8, 5.2))
rng = np.random.default_rng(7)
for i, (d, c) in enumerate(zip(data, palette)):
    # Cloud: half violin above the baseline, scaled to a fixed height
    grid = np.linspace(d.min() - 0.2, d.max() + 0.2, 300)  # trim the cloud near the data
    dens = stats.gaussian_kde(d)(grid)
    ax.fill_between(grid, i, i + 0.55 * dens / dens.max(), color=c, alpha=0.3, lw=0)
    ax.plot(grid, i + 0.55 * dens / dens.max(), color=c, lw=1.4)
    # Box: median and quartiles just under the cloud
    ax.boxplot(d, positions=[i - 0.1], widths=0.09, vert=False, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor="white", edgecolor="#3A3A3C", lw=1),
               medianprops=dict(color=c, lw=2.2), whiskerprops=dict(color="#3A3A3C"),
               capprops=dict(visible=False))
    # Rain: jittered raw points below the box
    ax.scatter(d, i - 0.3 + rng.uniform(-0.08, 0.08, len(d)),
               s=12, color=c, alpha=0.55, lw=0)

ax.set_yticks(range(len(groups)), [f"I. {g}\nmedian {np.median(d):.1f} · n = {len(d)}"
                                   for g, d in zip(groups, data)])
ax.set_ylim(-0.5, len(groups) - 0.3)
ax.set_xlim(4, 8.2)
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Sepal length (cm)")
ax.set_title("Virginica irises have the longest sepals", pad=26)
ax.text(0, 1.02, f"Kruskal–Wallis H(2) = {H:.1f}, p = {p:.1e}, ε² = {eps2:.2f} · "
        "cloud = density, box = quartiles, rain = flowers", transform=ax.transAxes,
        fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**Statlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/distributions/raincloud/) · [← Distributions](../README.md)

**Cite:** Warin, T. (2026). Raincloud plot. In Statlas: The Python Gallery of Statistical Graphics. https://warint.github.io/pystatlas/distributions/raincloud/
