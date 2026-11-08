# Proportions with confidence intervals

> A sorted dot plot of proportions with Wilson 95% confidence intervals.

![Proportions with confidence intervals](plot.png)

When the question is "what share of each group…?", a dot per group with its
confidence interval beats a bar chart: the eye compares positions, and the
interval width shows at once how precise each share is. The **Wilson score
interval** is used instead of the textbook Wald interval because it keeps its
95% coverage for small samples and proportions near 0 or 1. The dashed line is
the average share across elements.

The setting is Kouloukoui, de Marcellis-Warin & Warin (2025, *Climate*) on
corporate climate transition plans. The data are **simulated** for illustration
and are *not* the paper's data or results: 240 firms, each coded for whether its
plan includes seven elements (targets, capex, governance, scope 3 …). Intervals
come from `statsmodels.stats.proportion.proportion_confint(method="wilson")`.
Because every firm is coded on every element, equality of the seven shares is
tested with **Cochran's Q** (`statsmodels.stats.contingency_tables.cochrans_q`).

**R equivalent:** `ggstats::stat_prop() + geom_errorbar()`  
**Libraries:** Matplotlib, NumPy, pandas, statsmodels

**Data:** Simulated data with a fixed random seed, created for this example. — No real respondents, documents or measurements: the numbers only illustrate how the chart and its statistics work.

**Data:** Kouloukoui, D., de Marcellis-Warin, N., & Warin, T. (2025). Barriers, opportunities and best practices for corporate climate transition plans: A literature review. Climate, 13(5), 88. https://doi.org/10.3390/cli13050088 — The example is set in the context of this paper; unless its description says otherwise, the numbers are simulated and are not the paper's data or results.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.proportion import proportion_confint
# Simulated data, set in the context of Kouloukoui, de Marcellis-Warin & Warin (2025) [kouloukoui2025climate]

# Simulated coding: 240 firms × 7 plan elements (1 = included), linked by a firm-level ambition
rng = np.random.default_rng(88)
elements = {"Emission-reduction targets": 1.3, "Board oversight of climate": 0.7,
            "Interim (2030) milestones": 0.2, "Capex aligned with the plan": -0.6,
            "Scope 3 emissions covered": -0.9, "Executive pay linked to climate": -1.3,
            "Just-transition measures": -1.9}
ambition = rng.normal(0, 1, 240)
plans = pd.DataFrame({e: (b + ambition + rng.normal(0, 1, 240) > 0).astype(int)
                      for e, b in elements.items()})

# Proportion + Wilson 95% CI per element, sorted
g = pd.DataFrame({"k": plans.sum(), "n": len(plans)})
g["p"] = g.k / g.n
g["low"], g["high"] = proportion_confint(g.k, g.n, alpha=0.05, method="wilson")
g = g.sort_values("p")

# Cochran's Q: do the seven elements appear equally often in the same firms' plans?
q = cochrans_q(plans.to_numpy())
overall = g.p.mean()

fig, ax = plt.subplots(figsize=(8, 4.8))
y = np.arange(len(g))
ax.axvline(overall, color="#8E8E93", lw=1, ls="--", zorder=1)
ax.text(overall, len(g) - 0.45, f" average {overall:.0%}", fontsize=9, color="#6E6E73", va="bottom")
ax.hlines(y, g.low, g.high, color="#007AFF", lw=2.5, alpha=0.45, zorder=2)
ax.scatter(g.p, y, s=80, color="#007AFF", edgecolor="white", lw=1.5, zorder=3)

for yi, (_, row) in zip(y, g.iterrows()):
    ax.text(1.04, yi, f"{row.p:.0%}", va="center", fontsize=10, fontweight="semibold", color="#007AFF")
    ax.text(1.12, yi, f"{row.k:.0f} / {row.n:.0f}", va="center", fontsize=9.5, color="#6E6E73")

ax.set_yticks(y, g.index)
ax.set_ylim(-0.6, len(g) - 0.2)
ax.set_xlim(0, 1.26)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Firms whose transition plan includes the element (Wilson 95% CI)")
ax.set_title("Targets are common in transition plans; just-transition measures are rare", pad=26)
ax.text(0, 1.02, f"Cochran's Q({len(g) - 1}) = {q.statistic:.0f}, p < .001 for equal shares · "
        f"n = {len(plans)} firms (simulated)", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/proportions-surveys/proportions-with-ci/) · [← Proportions & surveys](../README.md)

**Cite:** Warin, T. (2026). Proportions with confidence intervals. In PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences (Version 1.1.0) [Computer software]. https://warint.github.io/pystatlas/proportions-surveys/proportions-with-ci/
