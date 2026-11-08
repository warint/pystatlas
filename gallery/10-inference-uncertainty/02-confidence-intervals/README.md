# What a 95% confidence interval means

> 100 surveys of 30 firms out of 2,000; the 95% CIs for the mean climate-risk score that miss are in red.

![What a 95% confidence interval means](plot.png)

"95% confidence" is a property of the *procedure*, not of one interval. Here
a population of 2,000 firms is fully known, so its mean score is the known
true value. The same small survey — 30 firms drawn at random, without
replacement — is repeated 100 times, and each horizontal line is the resulting
95% t-interval around the sample mean. About 95 of them cover the true
mean (dashed line); the few that miss it are drawn in red. Interval widths
vary too, because each sample estimates its own standard deviation.

The setting is Kouloukoui, de Marcellis-Warin, Armellini, Warin & Andrade
Torres (2021), *Journal of Cleaner Production*, on how firms in carbon-intensive
industries perceive their exposure to climate risks. The population of
perception scores (0–100, right-skewed) is **simulated** with `numpy` (seeded)
for illustration; it is *not* the paper's data or results. The intervals use
`scipy.stats.t.interval`. Because the scores are skewed, samples with a low
mean also tend to have a small spread, so misses fall mostly below the true
mean; and one run of 100 surveys can land a few points
above or below 95% by chance alone.

**R equivalent:** `teaching classic (e.g. TeachingDemos::ci.examp())`  
**Libraries:** Matplotlib, NumPy, SciPy

**Data:** Simulated data with a fixed random seed, created for this example. — No real respondents, documents or measurements: the numbers only illustrate how the chart and its statistics work.

**Data:** Kouloukoui, D., de Marcellis-Warin, N., Armellini, F., Warin, T., & Andrade Torres, E. (2021). Factors influencing the perception of exposure to climate risks: Evidence from the world's largest carbon-intensive industries. Journal of Cleaner Production, 306, 127160. https://doi.org/10.1016/j.jclepro.2021.127160 — The example is set in the context of this paper; unless its description says otherwise, the numbers are simulated and are not the paper's data or results.


```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Simulated data, set in the context of Kouloukoui et al. (2021) [kouloukoui2021]
# Population: climate-risk perception scores (0–100) of 2,000 firms; 100 surveys of n = 30
rng = np.random.default_rng(2021)
score = 100 * rng.beta(2, 4.5, 2000)
mu, n, reps = score.mean(), 30, 100
samples = np.array([rng.choice(score, n, replace=False) for _ in range(reps)])
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
ax.text(mu, reps + 2.5, f"population mean score μ = {mu:.1f}", ha="center", va="bottom", fontsize=9.5,
        color="#1D1D1F")

ax.set_ylim(0, reps + 6)
ax.set_yticks([1, 25, 50, 75, 100])
ax.set_ylabel("Survey (random sample of firms)")
ax.set_xlabel("Mean climate-risk perception score with 95% CI (0–100)")
ax.grid(axis="y", visible=False); ax.grid(axis="x")
ax.margins(x=0.08)
ax.set_title(f"{reps - miss.sum()} of {reps} surveys' intervals capture the true mean score", pad=26)
ax.text(0, 1.02, f"Observed coverage {100 * (1 - miss.mean()):.0f}% (nominal 95%) · "
        f"each survey n = {n} of {len(score):,} firms (simulated) · t-intervals, {miss.sum()} misses in red",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/inference-uncertainty/confidence-intervals/) · [← Inference & uncertainty](../README.md)

**Cite:** Warin, T. (2026). What a 95% confidence interval means. In PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences (Version 1.1.0) [Computer software]. https://warint.github.io/pystatlas/inference-uncertainty/confidence-intervals/
