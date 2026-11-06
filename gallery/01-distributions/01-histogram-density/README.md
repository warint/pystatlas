# Histogram with density

> Histogram, kernel density and fitted normal curve, with shape statistics.

![Histogram with density](plot.png)

The first look at any continuous variable. Bars show the empirical
distribution on a density scale, the solid curve is a kernel density estimate
(KDE) and the dashed curve is the normal distribution with the same mean and
standard deviation. When the two curves disagree, the data are not normal —
here the long right tail pulls the **mean** above the **median**.

Data are the mean tumour area of 569 breast-cancer biopsies bundled with
**scikit-learn**. `scipy.stats.gaussian_kde` draws the density,
`scipy.stats.skew` / `kurtosis` describe the shape and `scipy.stats.shapiro`
tests normality.

**R equivalent:** `ggplot2::geom_histogram() + geom_density()`  
**Libraries:** Matplotlib, NumPy, SciPy, scikit-learn

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer

# Data: mean area of the cell nuclei (in pixels²) for each biopsy
x = load_breast_cancer(as_frame=True).frame["mean area"].to_numpy()
mean, median, sd = x.mean(), np.median(x), x.std(ddof=1)

# Shape statistics and normality test
skew, kurt = stats.skew(x), stats.kurtosis(x)  # kurtosis: excess, 0 for a normal
W, p = stats.shapiro(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(x, bins=40, density=True, color="#007AFF", alpha=0.25, edgecolor="white")

# Kernel density estimate and the normal curve with the same mean / SD
grid = np.linspace(0, x.max() * 1.05, 400)
ax.plot(grid, stats.gaussian_kde(x)(grid), color="#007AFF", lw=2.2, label="Kernel density")
ax.plot(grid, stats.norm.pdf(grid, mean, sd), color="#8E8E93", lw=1.6, ls="--",
        label="Normal fit (same mean, SD)")

# Mean and median markers
ymax = ax.get_ylim()[1]
for value, name, color, ha, dx in [(mean, "Mean", "#FF9500", "left", 6),
                                   (median, "Median", "#34C759", "right", -6)]:
    ax.axvline(value, color=color, lw=1.6, ymax=0.9)
    ax.annotate(f"{name}\n{value:,.0f}", (value, ymax * 0.9), xytext=(dx, -2),
                textcoords="offset points", ha=ha, va="top", fontsize=9.5,
                fontweight="semibold", color=color)

ax.set_xlim(0, grid[-1])
ax.set_yticks([])
ax.set_ylabel("Density")
ax.set_xlabel("Mean nucleus area (pixels²)")
ax.legend(loc="upper right")
ax.set_title("Tumour area is strongly right-skewed", pad=26)
ax.text(0, 1.02, f"n = {len(x)} · skewness = {skew:.2f} · excess kurtosis = {kurt:.2f} · "
        f"Shapiro–Wilk W = {W:.3f}, p = {p:.1e}", transform=ax.transAxes,
        fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/distributions/histogram-density/) · [← Distributions](../README.md)

**Cite:** Warin, T. (2026). Histogram with density. In PyStatlas: The Python Gallery of Statistical Graphics. https://warint.github.io/pystatlas/distributions/histogram-density/
