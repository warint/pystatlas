# Scatterplot with statistics

> Scatter, OLS line with 95% band and marginal histograms, with Pearson's r in the subtitle.

![Scatterplot with statistics](plot.png)

The Python counterpart of `ggscatterstats()`. The central panel shows every
observation with the least-squares line and its 95% confidence band (uncertainty
about the mean of *y* at each *x*, not about individual points). Marginal
histograms on the top and right reveal the shape of each variable, which a
correlation coefficient silently assumes is reasonable. The subtitle reports
Pearson's *r* with its 95% confidence interval, the t-test and the sample size.

Data are the diabetes study bundled with **scikit-learn**
(`load_diabetes(scaled=False)`): body-mass index against a quantitative measure
of disease progression one year later, for 442 patients. The test comes from
`scipy.stats.pearsonr`, the line and band from **statsmodels** OLS
`get_prediction().conf_int()`.

**R equivalent:** `ggstatsplot::ggscatterstats()`  
**Libraries:** Matplotlib, NumPy, statsmodels, SciPy, scikit-learn

```python
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.datasets import load_diabetes

# Data: BMI and disease progression for 442 diabetes patients
data = load_diabetes(as_frame=True, scaled=False).frame
x, y = data["bmi"].to_numpy(), data["target"].to_numpy()
n = len(x)

# Pearson correlation with its 95% CI
res = stats.pearsonr(x, y)
ci = res.confidence_interval(0.95)

# OLS fit and 95% confidence band for the mean
fit = sm.OLS(y, sm.add_constant(x)).fit()
grid = np.linspace(x.min(), x.max(), 200)
band = fit.get_prediction(sm.add_constant(grid)).conf_int(alpha=0.05)

fig = plt.figure(figsize=(8, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5), wspace=0.03, hspace=0.03)
ax = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

ax.scatter(x, y, s=22, color="#007AFF", alpha=0.45, lw=0)
ax.fill_between(grid, band[:, 0], band[:, 1], color="#FF9500", alpha=0.25, lw=0)
ax.plot(grid, fit.params[0] + fit.params[1] * grid, color="#FF9500", lw=2.2)
ax.set_xlabel("Body-mass index (kg/m²)")
ax.set_ylabel("Disease progression after one year")
ax.grid(axis="x")

# Marginal histograms
ax_top.hist(x, bins=30, color="#007AFF", alpha=0.35, edgecolor="white")
ax_right.hist(y, bins=30, color="#007AFF", alpha=0.35, edgecolor="white", orientation="horizontal")
for a in (ax_top, ax_right):
    a.axis("off")

ax_top.set_title("Heavier patients progress faster", pad=26)
p_txt = "p < .001" if res.pvalue < 0.001 else f"p = {res.pvalue:.3f}"
ax_top.text(0, 1.06, f"Pearson r = {res.statistic:.2f}, 95% CI [{ci.low:.2f}, {ci.high:.2f}], "
            f"t({n - 2}) = {fit.tvalues[1]:.1f}, {p_txt} · n = {n}",
            transform=ax_top.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**Statlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/statlas/correlation/scatter-with-stats/) · [← Correlation & association](../README.md)

**Cite:** Warin, T. (2026). Scatterplot with statistics. In Statlas: The Python Gallery of Statistical Graphics. https://warint.github.io/statlas/correlation/scatter-with-stats/
