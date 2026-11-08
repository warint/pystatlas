# Cumulative hazard

> Nelson–Aalen cumulative hazard of a drug's launch, with 95% bands, in four market groups.

![Cumulative hazard](plot.png)

The cumulative hazard H(t) adds up the instantaneous risk of the event over
time; it is the "risk accumulated so far" and relates to survival through
S(t) = exp(−H(t)). Unlike a survival curve it is not bounded by 1, and its
**slope** is the hazard itself — a straight line means a constant risk, a
curve that bends upward means risk that grows with time.

The setting is Hadengue, de Marcellis-Warin & Warin (2015, *Management
international*), on reverse innovation and technology transfer in the
pharmaceutical sector. The data here are **simulated** for illustration and are
not the paper's data or results: 400 drug–market pairs, timed in years from a
drug's first approval anywhere to its launch in a given market, in four groups
of markets. Drugs not yet launched when observation stops are censored. Curves
come from `lifelines.NelsonAalenFitter`, and the subtitle reports a log-rank
test across the four groups (`multivariate_logrank_test`). The x-axis stops at
six years, after which too few drugs remain at risk in the fastest groups.

**R equivalent:** `survminer::ggsurvplot(fun = "cumhaz")`  
**Libraries:** Matplotlib, NumPy, pandas, lifelines

**Data:** Simulated data with a fixed random seed, created for this example. — No real respondents, documents or measurements: the numbers only illustrate how the chart and its statistics work.

**Data:** Hadengue, M., de Marcellis-Warin, N., & Warin, T. (2015). Reverse innovation and reverse technology transfer: From made in China to discovered in China in the pharmaceutical sector. Management international, 19(4), 49–69. https://doi.org/10.7202/1043076ar — The example is set in the context of this paper; unless its description says otherwise, the numbers are simulated and are not the paper's data or results.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import NelsonAalenFitter
from lifelines.statistics import multivariate_logrank_test
# Simulated data, set in the context of Hadengue, de Marcellis-Warin & Warin (2015) [hadengue2015]

# Data: years from first approval to launch, 100 drugs in each of four market groups
rng = np.random.default_rng(2015)
rates = {"United States": 0.5, "Western Europe": 0.33, "Japan": 0.22, "China": 0.12}  # launches per year
df = pd.concat([pd.DataFrame({"market": m, "launch": rng.weibull(1.3, 100) / r}) for m, r in rates.items()])
cens = rng.uniform(2, 10, len(df))                      # end of observation for each drug
df["years"], df["launched"] = np.minimum(df.launch, cens), (df.launch <= cens).astype(int)
colors = ["#007AFF", "#34C759", "#FF9500", "#FF3B30"]

fig, ax = plt.subplots(figsize=(8, 5))

# Nelson–Aalen estimate with 95% CI, one curve per market group, first 6 years, while enough drugs remain at risk
for g, c in zip(rates, colors):
    d = df[df.market == g]
    naf = NelsonAalenFitter().fit(d.years, d.launched, label=g)
    H, ci = naf.cumulative_hazard_[g].loc[:6], naf.confidence_interval_.loc[:6]
    H.loc[6], ci.loc[6] = H.iloc[-1], ci.iloc[-1]
    ax.step(H.index, H, where="post", color=c, lw=2, label=f"{g}  (n = {len(d)}, {d.launched.sum()} launches)")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post", color=c, alpha=0.13, lw=0)

# Log-rank test across market groups
lr = multivariate_logrank_test(df.years, df.market, df.launched)
p_txt = "p < .001" if lr.p_value < 0.001 else f"p = {lr.p_value:.3f}"
ax.legend(loc="upper left")
ax.set(xlabel="Years since first approval anywhere", ylabel="Cumulative hazard of launch  H(t)",
       xlim=(0, 6), xticks=range(7), ylim=(0, None))
ax.set_title("Nelson–Aalen cumulative hazard of launch for four market groups", pad=26)
ax.text(0, 1.02, f"Nelson–Aalen with 95% CI  ·  log-rank χ²({lr.degrees_of_freedom}) = "
        f"{lr.test_statistic:.1f}, {p_txt}  ·  n = {len(df)} drug–market pairs (simulated), "
        f"{df.launched.sum()} launches", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/survival/cumulative-hazard/) · [← Survival analysis](../README.md)

**Cite:** Warin, T. (2026). Cumulative hazard. In PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences. https://warint.github.io/pystatlas/survival/cumulative-hazard/
