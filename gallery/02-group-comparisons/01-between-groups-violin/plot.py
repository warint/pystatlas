"""
title: Between-groups comparison
summary: Violin + box + points with Welch's ANOVA, effect size and pairwise tests.
r: ggstatsplot::ggbetweenstats()
tags: [ANOVA, Welch, Holm, effect size, violin, climate risk]
description: |
  The Python counterpart of `ggbetweenstats()`: every observation is shown,
  wrapped in a violin (distribution) and a slim box (quartiles), with the group
  mean highlighted. The subtitle reports Welch's one-way ANOVA — robust to
  unequal variances — together with ω² as effect size, and brackets show the
  Holm-corrected pairwise Welch t-tests.

  The setting is Kouloukoui et al. (2021, *Journal of Cleaner Production*), on
  the factors behind firms' perceived exposure to climate risks in the world's
  largest carbon-intensive industries. The numbers here are **simulated** for
  illustration — a 0–100 perceived-exposure score for n = 125 firms in three
  industries — and are not the paper's data or results. The ANOVA comes from
  `statsmodels.stats.oneway.anova_oneway`, the pairwise tests from
  `scipy.stats.ttest_ind` and `statsmodels.stats.multitest.multipletests`.
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

# Simulated data, set in the context of Kouloukoui et al. (2021) [kouloukoui2021]

# Data: perceived climate-risk exposure score (0–100) of firms, by industry
rng = np.random.default_rng(2021)
groups = {"Oil and gas": (68, 12, 45), "Mining and metals": (61, 14, 38),
          "Electric utilities": (53, 10, 42)}  # mean, SD, number of firms
data = [np.clip(rng.normal(m, s, n), 0, 100) for m, s, n in groups.values()]
palette = ["#007AFF", "#FF9500", "#34C759"]

# Welch's ANOVA + omega squared
res = anova_oneway(data, use_var="unequal")
k, n = len(data), sum(len(d) for d in data)
omega2 = (res.statistic - 1) * (k - 1) / ((res.statistic - 1) * (k - 1) + n)

# Pairwise Welch t-tests, Holm-corrected
pairs = list(combinations(range(k), 2))
raw_p = [stats.ttest_ind(data[i], data[j], equal_var=False).pvalue for i, j in pairs]
adj_p = multipletests(raw_p, method="holm")[1]

fig, ax = plt.subplots(figsize=(8, 5.4))
for i, (d, c) in enumerate(zip(data, palette)):
    v = ax.violinplot(d, positions=[i], widths=0.8, showextrema=False)
    for body in v["bodies"]:
        body.set(facecolor=c, alpha=0.15, edgecolor="none")
    ax.boxplot(d, positions=[i], widths=0.12, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="#3A3A3C", lw=1),
               medianprops=dict(color="#3A3A3C", lw=1.5),
               whiskerprops=dict(color="#3A3A3C"), capprops=dict(visible=False))
    ax.scatter(i + rng.uniform(-0.22, 0.22, len(d)) * 0.9, d, s=16, color=c, alpha=0.6, lw=0)
    ax.scatter(i, d.mean(), s=90, color=c, edgecolor="white", lw=2, zorder=5)
    ax.annotate(f"μ = {d.mean():.0f}", (i + 0.09, d.mean()), xytext=(10, 0), textcoords="offset points",
                va="center", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), fontsize=9.5, fontweight="semibold", color=c)

# Significance brackets
top, step = max(map(np.max, data)), 8
for h, ((i, j), p) in enumerate(sorted(zip(pairs, adj_p), key=lambda t: t[0][1] - t[0][0])):
    y = top + 4 + h * step
    ax.plot([i, i, j, j], [y - 1.5, y, y, y - 1.5], color="#8E8E93", lw=1)
    label = "p < .001" if p < 0.001 else f"p = {p:.3f}"
    ax.text((i + j) / 2, y + 0.8, label, ha="center", va="bottom", fontsize=9, color="#3A3A3C")

ax.set_xticks(range(k), [f"{g}\n(n = {len(d)} firms)" for g, d in zip(groups, data)])
ax.set_ylabel("Perceived climate-risk exposure\n(score, 0–100)")
ax.set_ylim(top=top + 4 + len(pairs) * step + 2)
ax.set_yticks(range(20, 101, 20))
ax.set_title(f"Oil and gas firms perceive the most climate-risk exposure "
             f"({data[0].mean():.0f} vs {data[-1].mean():.0f} for utilities)", pad=26)
ax.text(0, 1.02, f"n = {n} firms (simulated) · Welch's F({res.df_num:.0f}, {res.df_denom:.1f}) = "
        f"{res.statistic:.1f}, p = {res.pvalue:.1e}, ω² = {omega2:.2f} · pairwise: Welch t, Holm",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
