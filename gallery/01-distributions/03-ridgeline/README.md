# Ridgeline plot

> Overlapping density curves to compare one distribution across many groups.

![Ridgeline plot](plot.png)

A ridgeline (or "joy") plot stacks one density curve per group, letting them
overlap slightly so that a dozen distributions fit in one compact figure.
Here each ridge is a calendar month and shows how sea-surface temperature
varied over 61 years; the white tick marks the median and the colour encodes
it, so the seasonal cycle and the year-to-year spread read together.

Data are the monthly El Niño sea-surface temperatures (1950–2010) bundled with
**statsmodels** (`sm.datasets.elnino`). Densities come from
`scipy.stats.gaussian_kde`; because every year contributes one value per
month, the subtitle reports a Friedman test (`scipy.stats.friedmanchisquare`)
with Kendall's W as effect size.

**R equivalent:** `ggridges::geom_density_ridges()`  
**Libraries:** Matplotlib, NumPy, statsmodels, SciPy

```python
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Data: one row per year, one column per month
sst = sm.datasets.elnino.load_pandas().data.drop(columns="YEAR")
months = list(sst.columns)

# Friedman test (years are blocks) + Kendall's W
chi2, p = stats.friedmanchisquare(*[sst[m] for m in months])
W = chi2 / (len(sst) * (len(months) - 1))

grid = np.linspace(sst.values.min() - 1, sst.values.max() + 1, 400)
medians = sst.median()
cmap, norm = plt.get_cmap("RdBu_r"), plt.Normalize(medians.min() - 0.6, medians.max() + 0.6)

fig, ax = plt.subplots(figsize=(8, 6.4))
spacing, height = 1.0, 2.2  # vertical gap between ridges, peak height of a ridge
for i, m in enumerate(months):
    base = (len(months) - 1 - i) * spacing  # January on top
    dens = stats.gaussian_kde(sst[m])(grid)
    curve = base + height * dens / dens.max() * 0.6
    color = cmap(norm(medians[m]))
    ax.fill_between(grid, base, curve, color=color, alpha=0.9, lw=0, zorder=2 * i)
    ax.plot(grid, curve, color="white", lw=1.2, zorder=2 * i + 1)
    ax.axhline(base, color="#C7C7CC", lw=0.6, zorder=2 * i)
    # White tick at the median
    y_med = base + height * stats.gaussian_kde(sst[m])(medians[m])[0] / dens.max() * 0.6
    ax.plot([medians[m]] * 2, [base, y_med], color="white", lw=1.6, zorder=2 * i + 1)

ax.set_yticks([(len(months) - 1 - i) * spacing for i in range(len(months))],
              [m.capitalize() for m in months])
ax.set_ylim(-0.2, (len(months) - 1) * spacing + height * 0.6 + 0.1)
ax.set_xlim(grid[0], grid[-1])
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Sea-surface temperature (°C)")
ax.set_title("El Niño waters peak in March and bottom out in September", pad=26)
ax.text(0, 1.02, f"Friedman χ²({len(months) - 1}) = {chi2:.0f}, p = {p:.1e}, Kendall's W = {W:.2f}"
        f" · {len(sst)} years (1950–2010) · tick = median", transform=ax.transAxes,
        fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**Statlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/distributions/ridgeline/) · [← Distributions](../README.md)

**Cite:** Warin, T. (2026). Ridgeline plot. In Statlas: The Python Gallery of Statistical Graphics. https://warint.github.io/pystatlas/distributions/ridgeline/
