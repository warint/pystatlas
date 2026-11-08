"""
title: ECDF comparison
summary: Empirical cumulative distributions of two groups, with the Kolmogorov–Smirnov gap.
r: ggplot2::stat_ecdf()
tags: [ECDF, Kolmogorov-Smirnov, cumulative distribution, two-sample test, sport, golf]
description: |
  An empirical cumulative distribution function (ECDF) plots, for every value
  on the x-axis, the share of observations at or below it. Unlike a histogram
  it needs no bins, and every quantile can be read off directly. When two
  ECDFs are drawn together, the largest vertical distance between them is
  exactly the two-sample **Kolmogorov–Smirnov** statistic D, drawn here as a
  red segment.

  The setting is Pastoriza Rivas & Warin (2022, *Data in Brief*), a dataset of
  two decades of Tiger Woods' press conferences and tournament performance.
  The numbers here are **simulated** for illustration — round scores relative
  to par (whole strokes, lower is better) for n = 520 rounds, 1997–2009 versus
  2010–2019 — and are not the paper's data or results. `scipy.stats.ecdf`
  builds the curves and `scipy.stats.ks_2samp` returns D, its location and the
  p-value (approximate here, because whole-stroke scores have many ties).
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Simulated data, set in the context of Pastoriza Rivas & Warin (2022) [pastoriza2022]

# Data: round score to par (strokes) in two periods of the career
rng = np.random.default_rng(2022)
groups = {"1997–2009": np.round(rng.normal(-2.4, 2.9, 320)),
          "2010–2019": np.round(rng.normal(-0.6, 3.1, 200))}
palette = ["#007AFF", "#FF9500"]

# Two-sample Kolmogorov–Smirnov test: D is the largest vertical gap
ks = stats.ks_2samp(*groups.values())
x_gap = ks.statistic_location
F = [stats.ecdf(d).cdf for d in groups.values()]
y_gap = [f.evaluate(x_gap) for f in F]

allx = np.concatenate(list(groups.values()))
lo, hi = allx.min() - 1, allx.max() + 1  # common x-range for both curves
fig, ax = plt.subplots(figsize=(8, 5))
for (name, d), f, c in zip(groups.items(), F, palette):
    ax.step(np.r_[lo, f.quantiles, hi], np.r_[0, f.probabilities, 1], where="post",
            color=c, lw=2, label=f"Rounds {name} (n = {len(d)}, median {np.median(d):+.0f})")

# Mark the KS distance
ax.plot([x_gap, x_gap], y_gap, color="#FF3B30", lw=2.5, solid_capstyle="round", zorder=5)
ax.scatter([x_gap, x_gap], y_gap, color="#FF3B30", s=30, zorder=6)
ax.annotate(f"D = {ks.statistic:.2f}\nat {x_gap:+.0f}", (x_gap + 0.1, min(y_gap)),
            xytext=(x_gap + 0.6, min(y_gap) - 0.14), ha="left", va="center", fontsize=10,
            fontweight="semibold", color="#FF3B30",
            arrowprops=dict(arrowstyle="-", color="#FF3B30", lw=0.8, shrinkB=2))

ax.set_xlim(lo, hi)
ax.set_xticks(np.arange(-10, hi, 2))
ax.set_ylim(0, 1.02)
ax.set_yticks(np.linspace(0, 1, 5), [f"{v:.0%}" for v in np.linspace(0, 1, 5)])
ax.set_xlabel("Round score relative to par (strokes; lower is better)")
ax.set_ylabel("Cumulative share of rounds")
ax.legend(loc="lower right")
ax.set_title(f"The KS statistic is the widest vertical gap between the two ECDFs: D = {ks.statistic:.2f}", pad=26)
ax.text(0, 1.02, f"n = {len(allx)} rounds (simulated) · two-sample Kolmogorov–Smirnov "
        f"D = {ks.statistic:.3f}, p = {ks.pvalue:.1e}", transform=ax.transAxes,
        fontsize=9.5, color="#6E6E73")
plt.show()
