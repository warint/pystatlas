"""
title: Mosaic plot with Pearson residuals
summary: Tile areas proportional to cell counts, shaded by Pearson residuals from independence.
r: vcd::mosaic(shade = TRUE)
tags: [mosaic, contingency table, chi-squared, Pearson residuals, discourse, Twitter]
description: |
  A mosaic plot draws a two-way table as a square cut into tiles. Column **widths**
  are the marginal shares of the first variable, tile **heights** the conditional
  shares of the second, so every tile's area is proportional to its count. As in
  `vcd::mosaic(shade = TRUE)`, the colour is the **Pearson residual**
  (observed − expected) / √expected: blue tiles hold more tweets than independence
  predicts, red tiles fewer, and residuals beyond ±2 or ±4 flag the cells that drive χ².

  The setting is Stojkov & Warin (2023, *Discourse & Communication*) on Twitter
  discourse about the refugee crisis in Europe. The data are **simulated** for
  illustration and are *not* the paper's data or results: 3,000 refugee-related
  tweets, each with a country of origin and one dominant discourse frame
  (humanitarian, security, economic, political). The tiles are drawn by hand with `ax.bar`;
  expected counts come from `scipy.stats.chi2_contingency`.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
# Simulated data, set in the context of Stojkov & Warin (2023) [stojkov2023refugee]

# Simulated tweets: country of origin × dominant discourse frame, frame shares vary by country
rng = np.random.default_rng(2015)
frames = ["Humanitarian", "Security", "Economic", "Political"]
mix = {"Germany": ([0.46, 0.18, 0.18, 0.18], 1000), "UK": ([0.30, 0.26, 0.26, 0.18], 700),
       "France": ([0.36, 0.26, 0.14, 0.24], 550), "Italy": ([0.32, 0.22, 0.16, 0.30], 450),
       "Hungary": ([0.16, 0.44, 0.10, 0.30], 300)}
df = pd.DataFrame([(c, f) for c, (pr, n) in mix.items() for f in rng.choice(frames, n, p=pr)],
                  columns=["country", "frame"])
table = pd.crosstab(pd.Categorical(df.country, list(mix)), pd.Categorical(df.frame, frames))
obs = table.to_numpy()

# Chi-squared test and Pearson residuals
chi2, p, dof, expected = stats.chi2_contingency(obs, correction=False)
resid = (obs - expected) / np.sqrt(expected)

fig, ax = plt.subplots(figsize=(8.5, 5.6))
cmap = plt.get_cmap("RdBu")                         # red = fewer than expected, blue = more
norm = TwoSlopeNorm(0, vmin=-8, vmax=8)
gap = 0.012
widths = obs.sum(axis=1) / obs.sum()
x0 = 0
for i, w in enumerate(widths):
    heights = obs[i] / obs[i].sum()
    y0 = 0
    for j, h in enumerate(heights):
        ax.bar(x0, h - gap, width=w - gap, bottom=y0, align="edge",
               color=cmap(norm(resid[i, j])), edgecolor="none")
        sep = "  " if h < 0.12 else "\n"              # thin tiles get a one-line label
        ax.text(x0 + (w - gap) / 2, y0 + (h - gap) / 2, f"{obs[i, j]}{sep}{resid[i, j].round(1) + 0:+.1f}", ha="center", va="center",
                fontsize=9, linespacing=1.3, color="white" if abs(resid[i, j]) > 5 else "#1D1D1F")
        y0 += h
    ax.text(x0 + (w - gap) / 2, -0.03, f"{table.index[i]}\n(n = {obs[i].sum()})", ha="center", va="top",
            fontsize=10, color="#3A3A3C")
    x0 += w

# Row labels on the left, placed at the heights of the first column
y0 = 0
for j, h in enumerate(obs[0] / obs[0].sum()):
    ax.text(-0.015, y0 + (h - gap) / 2, table.columns[j], ha="right", va="center", fontsize=10, color="#3A3A3C")
    y0 += h

sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm_, ax=ax, shrink=0.8, pad=0.02, ticks=[-8, -4, 0, 4, 8])
cb.set_label("Pearson residual", color="#3A3A3C")
cb.outline.set_visible(False)
ax.set_xlim(0, 1 - gap); ax.set_ylim(0, 1 - gap)
ax.axis("off")
ax.text(0.5, -0.15, "Country of the tweet", transform=ax.transAxes, ha="center", fontsize=11)
ax.text(-0.19, 0.5, "Discourse frame", transform=ax.transAxes, rotation=90, va="center", fontsize=11)
ax.set_title("The mix of refugee frames changes from one country to the next", pad=26)
ax.text(0, 1.02, f"Pearson χ²({dof}) = {chi2:.0f}, p < .001 · n = {obs.sum():,} tweets (simulated) · "
        "tile labels: count and residual", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
