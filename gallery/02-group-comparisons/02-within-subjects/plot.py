"""
title: Within-subjects comparison
summary: Repeated measures with one line per subject, means, rm-ANOVA and paired tests.
r: ggstatsplot::ggwithinstats()
tags: [repeated measures, paired t-test, rm-ANOVA, effect size, spaghetti plot, manufacturing]
description: |
  When the same units are measured under every condition, the interesting
  variation is *within* each unit. Thin grey lines join each production line's
  measurements, so you can see whether almost every line moves in the same
  direction; violins show the distribution per stage and the large dots
  (joined by a dark line) are the stage means.

  The setting is Warin, d'Anglade & de Marcellis-Warin (2025, *International
  Journal of Sustainable Engineering*), which combines k-means clustering and
  data science to reduce waste in manufacturing. The numbers here are
  **simulated** for illustration — the waste rate of the same n = 24
  production lines at baseline, after a clustering-based reorganisation and
  at follow-up — and are not the paper's data or results. The subtitle
  reports a one-way repeated-measures ANOVA
  (`statsmodels.stats.anova.AnovaRM`) with partial η², and the brackets show
  Holm-corrected paired t-tests (`scipy.stats.ttest_rel`) with Cohen's d_z
  (negative = waste fell).
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

# Simulated data, set in the context of Warin, d'Anglade & de Marcellis-Warin (2025) [warin2025waste]

# Data: waste rate (% of output) of the same 24 lines at three stages
rng = np.random.default_rng(2025)
conditions = ["Baseline", "After\nreorganisation", "Follow-up"]
line = rng.normal(8, 1.8, 24)                          # each line's own level
wide = pd.DataFrame({c: line + shift + rng.normal(0, 0.9, 24)
                     for c, shift in zip(conditions, [0, -1.8, -1.4])})
n = len(wide)
long = wide.rename_axis("id").reset_index().melt("id", var_name="condition", value_name="waste")

# Repeated-measures ANOVA + partial eta squared
aov = AnovaRM(long, "waste", "id", within=["condition"]).fit().anova_table.iloc[0]
eta_p = aov["F Value"] * aov["Num DF"] / (aov["F Value"] * aov["Num DF"] + aov["Den DF"])

# Pairwise paired t-tests (Holm) with Cohen's d_z
pairs = list(combinations(range(3), 2))
tests = [stats.ttest_rel(wide.iloc[:, j], wide.iloc[:, i]) for i, j in pairs]
adj_p = multipletests([t.pvalue for t in tests], method="holm")[1]
dz = [(wide.iloc[:, j] - wide.iloc[:, i]).mean() / (wide.iloc[:, j] - wide.iloc[:, i]).std()
      for i, j in pairs]

fig, ax = plt.subplots(figsize=(8, 5.6))
palette = ["#FF9500", "#AF52DE", "#007AFF"]
x = np.arange(3)
for _, row in wide.iterrows():                      # one line per subject
    ax.plot(x, row.values, color="#C7C7CC", lw=0.8, alpha=0.8, zorder=1)
for i, c in enumerate(palette):
    v = ax.violinplot(wide.iloc[:, i], positions=[i], widths=0.55, showextrema=False)
    v["bodies"][0].set(facecolor=c, alpha=0.15, edgecolor="none")
    ax.scatter(np.full(n, i), wide.iloc[:, i], s=18, color=c, alpha=0.7, lw=0, zorder=2)
means = wide.mean()
ax.plot(x, means, color="#1D1D1F", lw=2.5, zorder=3)
ax.scatter(x, means, s=110, color="#1D1D1F", edgecolor="white", lw=2, zorder=4)
for i, m in enumerate(means):
    ax.annotate(f"μ = {m:.1f}%", (i, m), xytext=(14, 0), textcoords="offset points", va="center",
                fontsize=9.5, fontweight="semibold", color="#1D1D1F",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=5)

# Significance brackets
top, step = wide.values.max() + 0.4, 0.8
for h, ((i, j), p, d) in enumerate(sorted(zip(pairs, adj_p, dz), key=lambda t: t[0][1] - t[0][0])):
    y = top + h * step
    ax.plot([i, i, j, j], [y - 0.2, y, y, y - 0.2], color="#8E8E93", lw=1)
    label = "p < .001" if p < 0.001 else f"p = {p:.3f}"
    ax.text((i + j) / 2, y + 0.07, f"{label}, d_z = {d:.2f}", ha="center", va="bottom",
            fontsize=9, color="#3A3A3C")

ax.set_xticks(x, [f"{c}\n(n = {n})" for c in conditions])
ax.set_xlim(-0.5, 2.6)
ax.set_ylim(0, top + 3 * step + 0.2)
ax.set_ylabel("Waste rate (% of output)")
fell = (wide.iloc[:, 1] < wide.iloc[:, 0]).sum()
ax.set_title(f"Waste fell on {fell} of {n} production lines after the reorganisation", pad=26)
ax.text(0, 1.02, f"n = {n} lines (simulated) · repeated-measures ANOVA F({aov['Num DF']:.0f}, "
        f"{aov['Den DF']:.0f}) = {aov['F Value']:.1f}, p = {aov['Pr > F']:.1e}, η²p = {eta_p:.2f}"
        " · pairwise: paired t, Holm", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
