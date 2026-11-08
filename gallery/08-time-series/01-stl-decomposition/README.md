# STL decomposition

> Monthly manufacturing sales split into trend, seasonal and remainder components.

![STL decomposition](plot.png)

STL (Seasonal–Trend decomposition using Loess) writes a series as
**observed = trend + seasonal + remainder**. Each panel shares the time axis but
has its own vertical scale, so the grey bar on the left of every panel — all of
the same length in data units — shows how large each component really is:
a short bar means a component that dominates, a long one a component that is small.

The setting is Warin (2024, *Technology Innovations in Statistics Education*),
on using Statistics Canada's open economic data in statistics courses, where
monthly manufacturing sales are a typical teaching series. The data here are
**simulated** for illustration and are not Statistics Canada's figures: a
manufacturing-sales-like series in billions of dollars, January 2005 to
December 2023 (228 months), built from a trend with two slumps (2009 and
spring 2020), a fixed monthly pattern and noise. The same simulated series is
used in the STL, SARIMA and seasonal subseries examples. The decomposition uses
`statsmodels.tsa.seasonal.STL` with a 12-month period and robust fitting; the
subtitle gives the strength of trend and seasonality as defined by Wang,
Smith & Hyndman (2006).

**R equivalent:** `feasts::STL() + autoplot()`  
**Libraries:** Matplotlib, NumPy, pandas, statsmodels

**Data:** Simulated data with a fixed random seed, created for this example. — No real respondents, documents or measurements: the numbers only illustrate how the chart and its statistics work.

**Data:** Warin, T. (2024). Access Statistics Canada's open economic data for statistics and data science courses. Technology Innovations in Statistics Education, 15(1). https://doi.org/10.5070/t5.1868 — The example is set in the context of this paper; unless its description says otherwise, the numbers are simulated and are not Statistics Canada's figures.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# Simulated data, set in the context of Warin (2024) [warin2024statcan]

# Data: simulated monthly manufacturing sales, billions of dollars, 2005–2023 (shared generator)
rng = np.random.default_rng(2024)
idx = pd.date_range("2005-01-01", "2023-12-01", freq="MS")
t = np.arange(len(idx))
trend = 48 + 0.06 * t - 7 * np.exp(-((t - 54) / 9) ** 2) - 8 * np.exp(-((t - 183) / 2.5) ** 2)
season = np.tile([-3.0, -0.8, 2.6, 0.6, 1.2, 2.2, -3.4, -1.2, 1.6, 2.0, 0.4, -2.2], len(idx) // 12)
sales = pd.Series(trend + season + rng.normal(0, 0.7, len(idx)), index=idx)

res = STL(sales, period=12, robust=True).fit()
parts = {"Observed": sales, "Trend": res.trend, "Seasonal": res.seasonal, "Remainder": res.resid}
colors = ["#1D1D1F", "#007AFF", "#FF9500", "#8E8E93"]

# Strength of trend and seasonality (0 = none, 1 = all the variation)
f_trend = max(0, 1 - res.resid.var() / (res.trend + res.resid).var())
f_seas = max(0, 1 - res.resid.var() / (res.seasonal + res.resid).var())

fig, axes = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True,
                         gridspec_kw=dict(height_ratios=[1.4, 1, 1, 1]))
bar = 2 * res.resid.abs().max()  # common scale bar, in billions of dollars
x0 = sales.index[0] - np.timedelta64(400, "D")

for ax, (name, s), c in zip(axes, parts.items(), colors):
    if name == "Remainder":
        ax.vlines(s.index, 0, s, color=c, lw=0.8)
        ax.axhline(0, color="#C7C7CC", lw=0.8)
    else:
        ax.plot(s.index, s, color=c, lw=2 if name == "Trend" else 1)
    ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=14)
    # Scale bar: same height in billions in every panel
    mid = (s.max() + s.min()) / 2
    ax.plot([x0, x0], [mid - bar / 2, mid + bar / 2], color="#C7C7CC", lw=5, solid_capstyle="butt",
            clip_on=False)

axes[-1].set_xlabel("Year")
fig.align_ylabels(axes)
axes[0].set_xlim(x0 - np.timedelta64(200, "D"), sales.index[-1])
slump = res.trend.diff(12).idxmin().year                                  # steepest yearly fall
low = res.seasonal.groupby(res.seasonal.index.month_name()).mean().idxmin()  # weakest month
axes[0].set_title(f"Manufacturing sales: the {slump} slump, and a dip every {low}", pad=26)
axes[0].text(0, 1.04, f"STL, period 12, robust · n = 228 months (simulated) · strength: trend = "
             f"{f_trend:.2f}, seasonality = {f_seas:.2f} · grey bars = {bar:.1f} bn",
             transform=axes[0].transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/time-series/stl-decomposition/) · [← Time series](../README.md)

**Cite:** Warin, T. (2026). STL decomposition. In PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences. https://warint.github.io/pystatlas/time-series/stl-decomposition/
