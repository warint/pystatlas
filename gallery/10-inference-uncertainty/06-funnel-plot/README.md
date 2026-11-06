# Funnel plot

> Study effects against their standard error, with a pseudo 95% funnel and Egger's test.

![Funnel plot](plot.png)

Each dot is a study: its effect size on the x-axis, its standard error on the
(inverted) y-axis, so large precise studies sit at the top. Without bias the
dots form a symmetric funnel around the pooled effect, 95% of them inside the
pseudo-confidence region (± 1.96 SE). Here the bottom-left is empty: small
studies with null results are missing. **Egger's regression test** (standard
normal deviate on precision) quantifies that asymmetry; its fitted line is
dashed.

The meta-analysis is simulated with `numpy` (seeded): 100 trials of a true
standardised effect of 0.2, where small non-significant trials mostly stay in
the file drawer (hollow grey — never observable in real life). Egger's test is
an OLS fit from **statsmodels**.

**R equivalent:** `metafor::funnel() + metafor::regtest()`  
**Libraries:** Matplotlib, NumPy, statsmodels, SciPy

```python
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Simulate 100 trials; small, non-significant ones are rarely published
rng = np.random.default_rng(2)
k = 100
n = np.round(np.exp(rng.uniform(np.log(10), np.log(300), k)))  # participants per arm
se = np.sqrt(2 / n)
d = rng.normal(0.2, 0.05, k) + rng.normal(0, se)
published = (stats.norm.sf(d / se) < 0.05) | (n > 120) | (rng.uniform(size=k) < 0.15)
dp, sp = d[published], se[published]

# Fixed-effect pooled estimate and Egger's test: z = b0 + b1 · precision
w = 1 / sp**2
pooled = np.sum(w * dp) / w.sum()
egger = sm.OLS(dp / sp, sm.add_constant(1 / sp)).fit()
b0, b1 = egger.params

fig, ax = plt.subplots(figsize=(9, 5.6))
s_grid = np.linspace(0, se.max() * 1.05, 100)
ax.fill_betweenx(s_grid, pooled - 1.96 * s_grid, pooled + 1.96 * s_grid, color="#007AFF",
                 alpha=0.07, lw=0)
for sign in (-1, 1):
    ax.plot(pooled + sign * 1.96 * s_grid, s_grid, color="#8E8E93", lw=1, ls="--",
            label="Pseudo 95% CI\n(pooled ± 1.96 SE)" if sign > 0 else None)
ax.axvline(pooled, color="#007AFF", lw=1.2, label=f"Pooled (fixed effect)\nd = {pooled:.2f}")
ax.axvline(0, color="#C7C7CC", lw=0.8)
ax.plot(b1 + b0 * s_grid, s_grid, color="#FF9500", lw=1.8, ls="--", label="Egger regression line")

ax.scatter(d[~published], se[~published], s=30, facecolor="none", edgecolor="#AEAEB2", lw=1,
           label=f"Unpublished, file\ndrawer ({(~published).sum()})")
ax.scatter(dp, sp, s=34, color="#007AFF", edgecolor="white", lw=0.7, zorder=3,
           label=f"Published studies ({published.sum()})")

ax.invert_yaxis()
ax.set_ylim(se.max() * 1.05, 0)
ax.set_xlim(-0.9, 1.45)
ax.set_xlabel("Standardised mean difference (Cohen's d)")
ax.set_ylabel("Standard error")
ax.grid(axis="y", visible=False)

handles, labels = ax.get_legend_handles_labels()
order = [4, 3, 1, 2, 0]
ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
          bbox_to_anchor=(1.01, 1), fontsize=9, labelspacing=1)

ax.set_title("Small null studies are missing: a sign of publication bias", pad=26)
ax.text(0, 1.02, f"Egger's test: intercept = {b0:.2f}, t({egger.df_resid:.0f}) = "
        f"{egger.tvalues[0]:.2f}, p = {egger.pvalues[0]:.3f} · k = {published.sum()} published "
        f"studies · simulated, true d = 0.20", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/inference-uncertainty/funnel-plot/) · [← Inference & uncertainty](../README.md)

**Cite:** Warin, T. (2026). Funnel plot. In PyStatlas: The Python Gallery of Statistical Graphics. https://warint.github.io/pystatlas/inference-uncertainty/funnel-plot/
