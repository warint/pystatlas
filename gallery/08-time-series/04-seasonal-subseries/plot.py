"""
title: Seasonal subseries plot
summary: Each calendar month's values across the years, with the monthly mean as a reference.
r: feasts::gg_subseries()
tags: [seasonality, subseries, month plot, ANOVA, time series]
description: |
  A seasonal subseries plot (or month plot) cuts a monthly series into twelve
  mini time series, one per calendar month, each running across all the years.
  The horizontal bar is that month's mean, so the **shape of the season** is read
  from bar to bar, while the line within each panel shows whether that month is
  drifting or swinging from year to year.

  Data are monthly sea-surface temperatures in the Niño 1+2 region of the eastern
  Pacific, 1950–2010 (`sm.datasets.elnino`, bundled with **statsmodels**). The
  subtitle reports a one-way ANOVA of temperature on month (`scipy.stats.f_oneway`)
  with η², the share of all variation explained by the season. The El Niño winter
  of 1997–98 is highlighted.
"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Data: wide table, one row per year, one column per month
df = sm.datasets.elnino.load_pandas().data
years = df.YEAR.astype(int).to_numpy()
months = df.columns[1:]
values = df[months].to_numpy()

# Seasonal effect: one-way ANOVA of temperature on month, with eta squared
F, p = stats.f_oneway(*values.T)
eta2 = ((values.mean(0) - values.mean()) ** 2).sum() * len(years) / ((values - values.mean()) ** 2).sum()

fig, ax = plt.subplots(figsize=(10, 5.2))
width = 0.8
x_in = (years - years.min()) / (years.max() - years.min()) * width - width / 2
highlight = {1997: ["DEC"], 1998: ["JAN", "FEB", "MAR", "APR", "MAY"]}

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

dec97 = (11 + x_in[years == 1997][0], values[years == 1997, 11][0])
ax.annotate("1997–98 El Niño", dec97, xytext=(-30, 28), textcoords="offset points",
            ha="right", va="center", fontsize=9.5,
            color="#FF3B30", fontweight="semibold",
            arrowprops=dict(arrowstyle="-", color="#FF3B30", lw=0.8))
ax.set_xticks(range(12), [m.title() for m in months])
ax.set_xlim(-0.5, 11.5)
ax.grid(axis="y")
ax.set_ylabel("Sea-surface temperature (°C)")
ax.plot([], [], color="#FF9500", lw=2.5, label="Monthly mean")
ax.plot([], [], color="#007AFF", lw=1, label=f"{years.min()}–{years.max()}, left to right")
ax.legend(loc="upper right", ncols=2)
ax.set_title("Eastern Pacific waters peak in March and bottom out in September", pad=26)
ax.text(0, 1.02, f"Niño 1+2 SST, monthly · one-way ANOVA F(11, {values.size - 12}) = {F:.0f}, "
        f"p < .001, η² = {eta2:.2f} · n = {len(years)} years",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
