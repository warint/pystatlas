"""
title: Seasonal subseries plot
summary: Each calendar month's manufacturing sales across the years, with the monthly mean.
r: feasts::gg_subseries()
tags: [seasonality, subseries, month plot, ANOVA, time series, Statistics Canada]
description: |
  A seasonal subseries plot (or month plot) cuts a monthly series into twelve
  mini time series, one per calendar month, each running across all the years.
  The horizontal bar is that month's mean, so the **shape of the season** is read
  from bar to bar, while the line within each panel shows whether that month is
  drifting or swinging from year to year.

  The setting is Warin (2024, *Technology Innovations in Statistics Education*),
  on using Statistics Canada's open economic data in statistics courses, where
  monthly manufacturing sales are a typical teaching series. The data here are
  **simulated** for illustration and are not Statistics Canada's figures: a
  manufacturing-sales-like series in billions of dollars, January 2005 to
  December 2023 (228 months), built from a trend with two slumps (2009 and
  spring 2020), a fixed monthly pattern and noise. The same simulated series is
  used in the STL, SARIMA and seasonal subseries examples. The subtitle reports a
  one-way ANOVA of sales on month (`scipy.stats.f_oneway`) with η², the share of
  all variation explained by the season; the long-run trend inside each panel
  accounts for much of the rest. The 2020 slump is highlighted.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Simulated data, set in the context of Warin (2024) [warin2024statcan]

# Data: simulated monthly manufacturing sales, billions of dollars, 2005–2023 (shared generator)
rng = np.random.default_rng(2024)
idx = pd.date_range("2005-01-01", "2023-12-01", freq="MS")
t = np.arange(len(idx))
trend = 48 + 0.06 * t - 7 * np.exp(-((t - 54) / 9) ** 2) - 8 * np.exp(-((t - 183) / 2.5) ** 2)
season = np.tile([-3.0, -0.8, 2.6, 0.6, 1.2, 2.2, -3.4, -1.2, 1.6, 2.0, 0.4, -2.2], len(idx) // 12)
sales = pd.Series(trend + season + rng.normal(0, 0.7, len(idx)), index=idx)

# One row per year, one column per month
years = sales.index.year.unique().to_numpy()
months = sales.index[:12].strftime("%b")
values = sales.to_numpy().reshape(len(years), 12)

# Seasonal effect: one-way ANOVA of sales on month, with eta squared
F, p = stats.f_oneway(*values.T)
eta2 = ((values.mean(0) - values.mean()) ** 2).sum() * len(years) / ((values - values.mean()) ** 2).sum()

fig, ax = plt.subplots(figsize=(10, 5.2))
width = 0.8
x_in = (years - years.min()) / (years.max() - years.min()) * width - width / 2
highlight = {2020: list(months)}

# One mini series per month, plus its mean
for i, m in enumerate(months):
    v = values[:, i]
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#F2F2F7", lw=0, zorder=0)
    ax.plot(i + x_in, v, color="#007AFF", lw=0.9, alpha=0.7)
    ax.hlines(v.mean(), i - width / 2, i + width / 2, color="#FF9500", lw=2.5, zorder=3)
    for yr, ms in highlight.items():
        if m in ms:
            j = np.where(years == yr)[0][0]
            ax.scatter(i + x_in[j], v[j], s=26, color="#FF3B30", edgecolor="white", lw=0.8, zorder=4)

ax.set_xticks(range(12), months)
ax.set_xlim(-0.5, 11.5)
ax.grid(axis="y")
ax.set_ylabel("Manufacturing sales (billions of dollars)")
ax.plot([], [], color="#FF9500", lw=2.5, label="Monthly mean")
ax.plot([], [], color="#007AFF", lw=1, label=f"{years.min()}–{years.max()}, left to right")
ax.scatter([], [], s=26, color="#FF3B30", label="2020 (spring slump)")
ax.legend(loc="upper left", ncols=3)
ax.set_ylim(top=values.max() + 6)
lo_m = sales.index[values.mean(0).argmin()].strftime("%B")   # first year runs Jan–Dec
hi_m = sales.index[values.mean(0).argmax()].strftime("%B")
p_txt = "p < .001" if p < 0.001 else f"p = {p:.3f}"
ax.set_title(f"Manufacturing sales bottom out in {lo_m} and peak in {hi_m}", pad=26)
ax.text(0, 1.02, f"Monthly, billions of dollars (simulated) · one-way ANOVA F(11, {values.size - 12}) = {F:.1f}, "
        f"{p_txt}, η² = {eta2:.2f} · n = {len(years)} years",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
