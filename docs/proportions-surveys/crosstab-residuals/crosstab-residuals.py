"""
title: Cross-table with standardized residuals
summary: A contingency table as a heatmap, coloured by standardized residuals from independence.
r: ggstats::ggtable() / stat_cross()
tags: [contingency table, chi-squared, standardized residuals, heatmap, cross-tab, ESG]
description: |
  The Python counterpart of `ggstats::ggtable()`: the familiar cross-tab, but each
  cell is coloured by its **standardized (adjusted) residual** — how many standard
  errors the observed count sits above or below what independence would predict.
  Under independence these residuals are roughly standard normal, so cells beyond
  ±2 (marked in bold) are the ones that make the χ² test significant. Each cell
  prints the count and its row percentage; the margins give the totals.

  The setting is Kouloukoui, de Marcellis-Warin, Gomes & Warin (2023, *Journal of
  Cleaner Production*) on Twitter conversations about ESG topics. The data are
  **simulated** for illustration and are *not* the paper's data or results: 4,000
  tweets, each with one ESG sub-topic and the world region of its author. Expected
  counts come from `scipy.stats.chi2_contingency`; the residual is
  (O − E) / √(E · (1 − row share) · (1 − column share)).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
# Simulated data, set in the context of Kouloukoui et al. (2023) [kouloukoui2023esg]

# Simulated tweets: ESG sub-topic (rows) × world region (columns), topic mix varies by region
rng = np.random.default_rng(2023)
topics = ["E · Climate change", "E · Pollution & waste", "S · Labour & human rights",
          "S · Diversity & inclusion", "G · Board & executive pay", "G · Corruption & ethics"]
mix = {"North America": ([.30, .10, .12, .22, .18, .08], 1400), "Europe": ([.38, .14, .12, .14, .14, .08], 1200),
       "Asia-Pacific": ([.28, .24, .14, .10, .14, .10], 800), "Latin America": ([.24, .16, .20, .10, .08, .22], 350),
       "Africa": ([.22, .14, .24, .10, .08, .22], 250)}
df = pd.DataFrame([(t, r) for r, (pr, n) in mix.items() for t in rng.choice(topics, n, p=pr)],
                  columns=["topic", "region"])
table = pd.crosstab(pd.Categorical(df.topic, topics), pd.Categorical(df.region, list(mix)))
obs = table.to_numpy()
n = obs.sum()

# Chi-squared test and adjusted (standardized) residuals
chi2, p, dof, exp = stats.chi2_contingency(obs, correction=False)
row_share, col_share = obs.sum(1, keepdims=True) / n, obs.sum(0, keepdims=True) / n
std_resid = (obs - exp) / np.sqrt(exp * (1 - row_share) * (1 - col_share))
row_pct = obs / obs.sum(1, keepdims=True) * 100
v = np.sqrt(chi2 / (n * (min(obs.shape) - 1)))

fig, ax = plt.subplots(figsize=(9, 5))
norm = TwoSlopeNorm(0, vmin=-6, vmax=6)
im = ax.imshow(std_resid, cmap="RdBu_r", norm=norm, aspect="auto")
for i in range(obs.shape[0]):
    for j in range(obs.shape[1]):
        strong = abs(std_resid[i, j]) >= 2
        ax.text(j, i - 0.12, f"{obs[i, j]:,}", ha="center", va="center", fontsize=11,
                fontweight="bold" if strong else "normal",
                color="white" if abs(std_resid[i, j]) > 3 else "#1D1D1F")
        ax.text(j, i + 0.2, f"{row_pct[i, j]:.0f}%", ha="center", va="center", fontsize=8.5,
                color="white" if abs(std_resid[i, j]) > 3 else "#3A3A3C")

# White gutters between cells and totals in the margins
ax.set_xticks(np.arange(-0.5, obs.shape[1]), minor=True)
ax.set_yticks(np.arange(-0.5, obs.shape[0]), minor=True)
ax.grid(which="minor", color="white", lw=3)
ax.grid(which="major", visible=False)
ax.tick_params(which="minor", length=0)
regions = [c.replace(" ", "\n") for c in table.columns]          # two-word regions on two lines
ax.set_xticks(range(obs.shape[1]), [f"{c}\n(n = {t:,})" for c, t in zip(regions, obs.sum(0))])
ax.set_yticks(range(obs.shape[0]), [f"{r}  (n = {t:,})" for r, t in zip(table.index, obs.sum(1))])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("World region of the tweet")
ax.set_ylabel("ESG sub-topic")

cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, ticks=[-6, -4, -2, 0, 2, 4, 6])
cb.set_label("Standardized residual", color="#3A3A3C")
cb.outline.set_visible(False)
ax.set_title("Each region tweets about a different side of ESG", pad=26)
ax.text(0, 1.02, f"Pearson χ²({dof}) = {chi2:.0f}, p < .001, V = {v:.2f} · n = {n:,} tweets (simulated) · "
        "count & row % · bold: |resid.| ≥ 2", transform=ax.transAxes,
        fontsize=9.5, color="#6E6E73")
plt.show()
