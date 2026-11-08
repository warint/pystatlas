"""
title: Influence plot
summary: Studentized residuals vs leverage, with bubbles sized by Cook's distance.
r: car::influencePlot()
tags: [influence, Cook's distance, leverage, outliers, studentized residuals, banking]
description: |
  An observation is **influential** when it is both unusual in its predictors
  (high leverage, x-axis) and badly fitted (large studentized residual, y-axis).
  Bubble area grows with Cook's distance (∝ √D, so extremes stay readable),
  which combines the two. Dashed lines mark the usual rules of thumb — leverage of 2p/n and 3p/n, studentized
  residuals of ±2 — and observations with Cook's D above 4/n are named and coloured.

  The setting is Warin & Stojkov (2021, *Journal of Risk and Financial
  Management*): banks' foreign claims after the 2008 crisis. The data are
  **simulated** for illustration — they are not the paper's data or results:
  n = 36 European and OECD countries, with foreign bank claims on each country
  (log US$ billions) explained by log GDP, financial depth, trade openness and
  euro-area membership. A few financial centres are given unusual predictors or
  claims so that the plot has something to find. Studentized (deleted) residuals,
  hat values and Cook's distances come from `model.get_influence()` in
  **statsmodels**; the subtitle checks the largest residual with a Bonferroni
  outlier test, as `car::outlierTest()` does.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
# Simulated data, set in the context of Warin & Stojkov (2021) [warin2021banks]

# Data: 36 countries; foreign bank claims from size, depth, openness and the euro
rng = np.random.default_rng(2021)
names = ["Austria", "Belgium", "Cyprus", "Estonia", "Finland", "France", "Germany", "Greece",
         "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands",
         "Portugal", "Slovakia", "Slovenia", "Spain", "Bulgaria", "Croatia", "Czechia",
         "Denmark", "Hungary", "Poland", "Romania", "Sweden", "Norway", "Switzerland",
         "United Kingdom", "Iceland", "Canada", "United States", "Japan", "Australia", "Turkey"]
n = len(names)
df = pd.DataFrame({"log_gdp": rng.normal(6, 1.2, n), "depth": rng.normal(100, 30, n),
                   "openness": rng.normal(90, 30, n),
                   "euro": [1] * 19 + [0] * 17}, index=names)
df.loc["Luxembourg", ["depth", "openness"]] = [300, 260]        # tiny economy, huge banks
df.loc["Malta", "openness"] = 230
df["claims"] = (-2 + 0.9 * df.log_gdp + 0.004 * df.depth + 0.003 * df.openness
                + 0.5 * df.euro + rng.normal(0, 0.45, n))
df.loc[["Ireland", "Cyprus", "Luxembourg"], "claims"] += [1.6, 2.2, -1.6]
model = smf.ols("claims ~ log_gdp + depth + openness + euro", data=df).fit()
infl = model.get_influence()
rstud, lev, cook = infl.resid_studentized_external, infl.hat_matrix_diag, infl.cooks_distance[0]
n, p = int(model.nobs), int(model.df_model + 1)
flag = cook > 4 / n

fig, ax = plt.subplots(figsize=(8.5, 5.6))

# Rule-of-thumb thresholds
for h in (2 * p / n, 3 * p / n):
    ax.axvline(h, color="#AEAEB2", lw=1, ls="--", zorder=0)
    ax.text(h, 1.0, f" {round(h * n / p)}p/n", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color="#8E8E93", va="top")
for r in (-2, 0, 2):
    ax.axhline(r, color="#AEAEB2" if r else "#C7C7CC", lw=1, ls="--" if r else "-", zorder=0)

# Bubbles: area proportional to √Cook's D (keeps extreme points readable)
area = lambda d: 20 + 700 * np.sqrt(d)
size = area(cook)
ax.scatter(lev[~flag], rstud[~flag], s=size[~flag], color="#007AFF", alpha=0.35,
           edgecolor="#007AFF", lw=0.8)
ax.scatter(lev[flag], rstud[flag], s=size[flag], color="#FF3B30", alpha=0.35,
           edgecolor="#FF3B30", lw=1.2)
for i in np.flatnonzero(flag):
    right = lev[i] > 0.7 * lev.max()              # far-right points: label on the left
    ax.annotate(f"{df.index[i]}\nD = {cook[i]:.2f}", (lev[i], rstud[i]),
                xytext=((-1 if right else 1) * (np.sqrt(size[i]) / 2 + 6), 0),
                textcoords="offset points", ha="right" if right else "left", va="center",
                fontsize=8.5, color="#3A3A3C", linespacing=1.1)

# Bubble-size key
for d, x in zip((0.1, 0.5, 1, 2), (0.745, 0.79, 0.84, 0.90)):
    ax.scatter(x, 0.86, s=area(d), transform=ax.transAxes, color="none",
               edgecolor="#8E8E93", lw=0.8)
    ax.text(x, 0.77, f"{d:g}", transform=ax.transAxes, ha="center", fontsize=8.5, color="#6E6E73")
ax.text(0.82, 0.97, "Cook's D", transform=ax.transAxes, ha="center", va="top",
        fontsize=8.5, color="#6E6E73")

ax.set_ylim(rstud.min() - 0.8, rstud.max() + 1.2)
ax.set_xlim(0, lev.max() * 1.15)
ax.set_xlabel("Leverage (hat value)")
ax.set_ylabel("Studentized residual")
ax.set_title(f"{flag.sum()} countries cross the Cook's D > 4/n line in a simulated model of bank claims", pad=26)

# Bonferroni outlier test on the largest |studentized residual|
k = np.argmax(np.abs(rstud))
p_bonf = min(1, n * 2 * stats.t.sf(abs(rstud[k]), model.df_resid - 1))
ax.text(0, 1.02, f"OLS log claims ~ 4 predictors · n = {n} countries (simulated) · {flag.sum()} with Cook's D > 4/n · "
        f"largest |t*| = {abs(rstud[k]):.2f} ({df.index[k]}), Bonferroni p = {p_bonf:.3f}",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
