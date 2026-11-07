# What a 95% confidence interval means

> 100 surveys of 30 people drawn from 944 ANES respondents; the 95% CIs for mean age that miss the true value are in red.

![What a 95% confidence interval means](plot.png)

"95% confidence" is a property of the *procedure*, not of one interval. Here
the 944 respondents of the 1996 American National Election Study (**statsmodels**
`anes96`) are treated as the whole population, so their mean age is the known
true value. The same small survey — 30 respondents drawn at random, without
replacement — is repeated 100 times, and each horizontal line is the resulting
95% t-interval around the sample mean age. About 95 of them cover the true
mean (dashed line); the few that miss it are drawn in red. Interval widths
vary too, because each sample estimates its own standard deviation.

The draws are seeded with `numpy` and the intervals use
`scipy.stats.t.interval`. Age is right-skewed (19 to 91), and one run of 100
surveys can land a few points above or below 95% by chance alone.

**R equivalent:** `teaching classic (e.g. TeachingDemos::ci.examp())`  
**Libraries:** Matplotlib, NumPy, statsmodels, SciPy

**Data:** Rosenstone, S. J., Kinder, D. R., Miller, W. E., & the National Election Studies. (2016). ANES 1996 Time Series Study (ICPSR 6896; Version 6) [Data set]. Inter-university Consortium for Political and Social Research. https://doi.org/10.3886/ICPSR06896.v6 — Subset of 944 respondents distributed with statsmodels (sm.datasets.anes96); public domain.


```python
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Population: ages of the 944 ANES 1996 respondents; 100 surveys of n = 30 each
age = sm.datasets.anes96.load_pandas().data["age"].to_numpy()
rng = np.random.default_rng(2024)
mu, n, reps = age.mean(), 30, 100
samples = np.array([rng.choice(age, n, replace=False) for _ in range(reps)])
means = samples.mean(axis=1)
se = samples.std(axis=1, ddof=1) / np.sqrt(n)
lo, hi = stats.t.interval(0.95, df=n - 1, loc=means, scale=se)
miss = (lo > mu) | (hi < mu)

fig, ax = plt.subplots(figsize=(8, 6.6))
y = np.arange(1, reps + 1)
colors = np.where(miss, "#FF3B30", "#7FB8FF")
ax.hlines(y, lo, hi, color=colors, lw=np.where(miss, 2.2, 1.2))
ax.scatter(means, y, s=np.where(miss, 16, 9), color=colors, zorder=3, lw=0)
ax.vlines(mu, 0, reps + 1.5, color="#1D1D1F", ls="--", lw=1.2, zorder=4)
ax.text(mu, reps + 2.5, f"population mean age μ = {mu:.1f}", ha="center", va="bottom", fontsize=9.5,
        color="#1D1D1F")

ax.set_ylim(0, reps + 6)
ax.set_yticks([1, 25, 50, 75, 100])
ax.set_ylabel("Survey (random sample)")
ax.set_xlabel("Mean age with 95% confidence interval (years)")
ax.grid(axis="y", visible=False); ax.grid(axis="x")
ax.margins(x=0.08)
ax.set_title(f"{reps - miss.sum()} of {reps} surveys' intervals capture the true mean age", pad=26)
ax.text(0, 1.02, f"Observed coverage {100 * (1 - miss.mean()):.0f}% (nominal 95%) · "
        f"each survey n = {n} of {len(age)} ANES 1996 respondents · t-intervals, {miss.sum()} misses in red",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/inference-uncertainty/confidence-intervals/) · [← Inference & uncertainty](../README.md)

**Cite:** Warin, T. (2026). What a 95% confidence interval means. In PyStatlas: The Python Gallery of Statistical Graphics. https://warint.github.io/pystatlas/inference-uncertainty/confidence-intervals/
