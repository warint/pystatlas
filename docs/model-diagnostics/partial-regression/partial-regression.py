"""
title: Added-variable plots
summary: One partial-regression panel per predictor; each fitted slope equals its coefficient.
r: car::avPlots()
tags: [partial regression, added-variable, Frisch-Waugh-Lovell, multiple regression, gravity model]
description: |
  In a multiple regression, a raw scatter of y against one predictor mixes its
  effect with everything correlated with it. An **added-variable plot** removes
  that: it plots the residuals of y on all *other* predictors against the
  residuals of the focal predictor on those same predictors. By the
  Frisch–Waugh–Lovell theorem, the slope of the line through the cloud is exactly
  the multiple-regression coefficient, and points far out on the x-axis are the
  ones that drive that coefficient.

  The setting is Tengia, Wandschneider, Warin & Wunnava (2011, *African Finance
  Journal*): an augmented gravity model of Southern African trade. The data are
  **simulated** for illustration — they are not the paper's data or results — and
  come from the same generator as the residual-diagnostics example: n = 210
  directed pairs among 15 African economies, 10 of them SADC members. Log exports
  are regressed on the log GDPs of exporter and importer, log distance, a common
  border and joint SADC membership with **statsmodels** `smf.ols()`. Residuals are
  computed with one auxiliary OLS fit per axis; each panel reports β, its t and p.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
# Simulated data, set in the context of Tengia, Wandschneider, Warin & Wunnava (2011) [tengia2011]

# Data: 15 African economies (10 SADC members), all 210 directed trade pairs
rng = np.random.default_rng(2011)
codes = ["AGO", "BWA", "LSO", "MWI", "MOZ", "NAM", "ZAF", "SWZ", "ZMB", "ZWE",   # SADC
         "KEN", "UGA", "ETH", "NGA", "GHA"]                                       # non-SADC
gdp = rng.normal(23, 1.3, 15)                                    # log GDP (US$)
xy = np.r_[rng.normal([0, -2000], 700, (10, 2)), rng.normal([0, 1500], 1500, (5, 2))]  # km
o, d = np.nonzero(~np.eye(15, dtype=bool))                       # exporter, importer
dist = np.hypot(*(xy[o] - xy[d]).T) + 200
df = pd.DataFrame({"ln_gdp_o": gdp[o], "ln_gdp_d": gdp[d], "ln_dist": np.log(dist),
                   "sadc": ((o < 10) & (d < 10)).astype(int), "border": (dist < 800).astype(int)},
                  index=[f"{codes[a]}-{codes[b]}" for a, b in zip(o, d)])
df["ln_trade"] = (-12 + 0.9 * df.ln_gdp_o + 0.8 * df.ln_gdp_d - 1.2 * df.ln_dist
                  + 0.7 * df.sadc + 0.5 * df.border + rng.standard_t(4, len(df)))

predictors = ["ln_gdp_o", "ln_gdp_d", "ln_dist", "border", "sadc"]
labels = {"ln_gdp_o": "Exporter log GDP", "ln_gdp_d": "Importer log GDP",
          "ln_dist": "Log distance", "border": "Common border (0/1)",
          "sadc": "Both in SADC (0/1)"}
full = smf.ols("ln_trade ~ " + " + ".join(predictors), data=df).fit()

fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, x in zip(axes.flat, predictors):
    others = " + ".join(p for p in predictors if p != x)
    ey = smf.ols(f"ln_trade ~ {others}", data=df).fit().resid     # y | others
    ex = smf.ols(f"{x} ~ {others}", data=df).fit().resid          # x | others
    sig = full.pvalues[x] < 0.05
    c = "#007AFF" if sig else "#8E8E93"

    ax.axhline(0, color="#E5E5EA", lw=1); ax.axvline(0, color="#E5E5EA", lw=1)
    ax.scatter(ex, ey, s=20, color=c, alpha=0.6, lw=0, zorder=2)
    grid = np.linspace(ex.min(), ex.max(), 2)
    ax.plot(grid, full.params[x] * grid, color="#FF3B30" if sig else "#AEAEB2", lw=2)
    # Label the observation with the most extreme partial x
    for i in np.argsort(-np.abs(ex.to_numpy()))[:1]:
        right = ex.iloc[i] < 0
        ax.annotate(df.index[i], (ex.iloc[i], ey.iloc[i]),
                    xytext=(6 if right else -6, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center", fontsize=8.5, color="#3A3A3C",
                    zorder=3, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    p = full.pvalues[x]
    ax.set_title(f"β = {full.params[x]:.2f} · t = {full.tvalues[x]:.2f} · "
                 + ("p < .001" if p < .001 else f"p = {p:.3f}"), fontsize=10.5, color=c,
                 fontweight="semibold" if sig else "normal", pad=8)
    ax.set_xlabel(f"{labels[x]} | others", fontsize=10)
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Log exports | others", fontsize=10)
    ax.margins(x=0.12, y=0.15)

# Last cell: reading guide instead of an empty panel
axes.flat[-1].axis("off")
axes.flat[-1].text(0.05, 0.5, "How to read\n\nEach slope is the coefficient of the\n"
                   "full model, net of the other four\npredictors. Red line: p < .05;\n"
                   "grey line: not significant.\nPoints far left or right move\n"
                   "the slope most.", transform=axes.flat[-1].transAxes, va="center",
                   fontsize=10, color="#6E6E73", linespacing=1.5)

fig.text(0.01, 1.065, "Net of the other terms, GDPs, distance and SADC membership shape trade",
         fontsize=15, fontweight="semibold", va="bottom")
fig.text(0.01, 1.02, f"OLS log exports ~ 5 predictors · n = {int(full.nobs)} pairs (simulated) · "
         f"R² = {full.rsquared:.2f} · F({full.df_model:.0f}, {full.df_resid:.0f}) = "
         f"{full.fvalue:.1f}, p < .001", fontsize=9.5, color="#6E6E73", va="bottom")
plt.show()
