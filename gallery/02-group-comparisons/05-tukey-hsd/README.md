# Tukey HSD intervals

> Simultaneous 95% confidence intervals for every pairwise difference in means.

![Tukey HSD intervals](plot.png)

After an ANOVA says "the groups differ", Tukey's honestly significant
difference (HSD) says *which* pairs differ. Each row is one pairwise
difference in means with a confidence interval that is widened so that all
intervals hold **simultaneously** at 95%. Intervals that do not cross zero are
coloured: blue when the first group's mean is higher, orange when it is lower.

The setting is Sanger & Warin (2020, *Global Economy Journal*), an NLP study
of the speeches of the European Central Bank's presidents. The numbers here
are **simulated** for illustration — the share of each speech devoted to
price stability for n = 290 speeches by Duisenberg, Trichet, Draghi and
Lagarde — and are not the paper's data or results. The intervals come from
`statsmodels.stats.multicomp.pairwise_tukeyhsd`, the omnibus test from
`scipy.stats.f_oneway`.

**R equivalent:** `TukeyHSD() + plot()`  
**Libraries:** Matplotlib, NumPy, pandas, SciPy, statsmodels

**Data:** Simulated data with a fixed random seed, created for this example. — No real respondents, documents or measurements: the numbers only illustrate how the chart and its statistics work.

**Data:** Sanger, W., & Warin, T. (2020). The speeches of the European Central Bank's presidents: An NLP study. Global Economy Journal, 20(2), 2050009. https://doi.org/10.1142/s2194565920500098 — The example is set in the context of this paper; unless its description says otherwise, the numbers are simulated and are not the paper's data or results.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Simulated data, set in the context of Sanger & Warin (2020) [sanger2020ecb]

# Data: % of each speech on price stability, by ECB president (mean, SD, number of speeches)
rng = np.random.default_rng(2020)
spec = {"Duisenberg": (22, 7, 60), "Trichet": (19, 7, 90), "Draghi": (14, 6, 90),
        "Lagarde": (16, 7, 50)}
names = list(spec)
df = pd.DataFrame({"president": np.repeat(range(4), [n for _, _, n in spec.values()]),
                   "share": np.concatenate([np.clip(rng.normal(m, s, n), 0, 100)
                                            for m, s, n in spec.values()])})

# One-way ANOVA, then Tukey HSD on all 6 pairs
F, p = stats.f_oneway(*[g.share for _, g in df.groupby("president")])
k, n = len(names), len(df)
eta2 = F * (k - 1) / (F * (k - 1) + n - k)
tukey = pairwise_tukeyhsd(df.share, df.president)
g1, g2 = np.triu_indices(k, 1)                    # pairs in the order statsmodels reports them
res = pd.DataFrame({"group1": g1, "group2": g2, "meandiff": tukey.meandiffs,
                    "lower": tukey.confint[:, 0], "upper": tukey.confint[:, 1],
                    "p-adj": tukey.pvalues, "reject": tukey.reject})
# statsmodels reports group2 − group1; flip to read "group1 − group2"
res = res.assign(diff=-res.meandiff, low=-res.upper, high=-res.lower,
                 label=[f"{names[a]} − {names[b]}" for a, b in zip(res.group1, res.group2)])
res = res.iloc[::-1].reset_index(drop=True)  # first pair on top

colors = np.where(~res.reject, "#AEAEB2", np.where(res["diff"] > 0, "#007AFF", "#FF9500"))
fig, ax = plt.subplots(figsize=(8, 4.8))
y = np.arange(len(res))
ax.axvline(0, color="#8E8E93", lw=1, ls="--", zorder=1)
ax.hlines(y, res.low, res.high, color=colors, lw=2.5, zorder=2)
ax.scatter(res["diff"], y, s=55, color=colors, edgecolor="white", lw=1.2, zorder=3)
for yi, row in res[res.reject].iterrows():
    pv = "p < .001" if row["p-adj"] < 0.001 else f"p = {row['p-adj']:.3f}"
    ax.annotate(f"{row['diff']:+.1f} pts, {pv}", (row.high, yi), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color="#3A3A3C")

ax.set_yticks(y, res.label, fontsize=9.5)
ax.set_ylim(-0.6, len(res) - 0.4)
ax.set_xlim(right=res.high.max() + 6)
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Difference in mean share on price stability (percentage points), Tukey 95% CI")
ax.set_title("Tukey intervals for every pair of presidents in a simulated speech sample", pad=26)
ax.text(0, 1.02, f"n = {n} speeches (simulated) · one-way ANOVA F({k - 1}, {n - k}) = {F:.1f}, "
        f"p = {p:.1e}, η² = {eta2:.2f} · {res.reject.sum()} of {len(res)} pairs differ",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/group-comparisons/tukey-hsd/) · [← Group comparisons](../README.md)

**Cite:** Warin, T. (2026). Tukey HSD intervals. In PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences (Version 1.1.0) [Computer software]. https://warint.github.io/pystatlas/group-comparisons/tukey-hsd/
