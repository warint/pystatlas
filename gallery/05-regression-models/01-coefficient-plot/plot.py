"""
title: Coefficient plot
summary: Regression estimates with 95% confidence intervals, one row per term.
r: ggstats::ggcoef_model()
tags: [regression, OLS, confidence interval, dot-and-whisker]
description: |
  A coefficient plot (a.k.a. dot-and-whisker plot) replaces the regression table.
  Each point is an estimate, each line its 95% confidence interval, and the dashed
  line marks zero. Terms whose interval crosses zero are drawn in grey, so the
  significant effects stand out at a glance.

  The model explains the violent-crime rate of the 50 US states (+ DC) with the
  **statsmodels** formula API. Predictors are standardised first, so the
  coefficients are directly comparable: change in crimes per 100,000 people for a
  one-standard-deviation change in the predictor.
"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Data: US state crime rates (bundled with statsmodels)
df = sm.datasets.statecrime.load_pandas().data
predictors = ["poverty", "single", "urban", "hs_grad", "white"]
df[predictors] = (df[predictors] - df[predictors].mean()) / df[predictors].std()

model = smf.ols("violent ~ " + " + ".join(predictors), data=df).fit()

# Tidy the results: estimate, 95% CI and p-value per term (intercept dropped)
ci = model.conf_int().drop("Intercept")
coefs = (
    ci.assign(estimate=model.params, p=model.pvalues)
      .rename(columns={0: "low", 1: "high"})
      .sort_values("estimate")
)
labels = {"poverty": "Poverty rate", "single": "Single-parent households",
          "urban": "Urban population", "hs_grad": "High-school graduates",
          "white": "White population"}

fig, ax = plt.subplots(figsize=(8, 4.6))
y = np.arange(len(coefs))
colors = np.where(coefs["p"] < 0.05, "#007AFF", "#AEAEB2")

ax.axvline(0, color="#8E8E93", lw=1, ls="--", zorder=1)
ax.hlines(y, coefs["low"], coefs["high"], color=colors, lw=2.5, zorder=2)
ax.scatter(coefs["estimate"], y, s=80, color=colors, edgecolor="white", lw=1.5, zorder=3)

for yi, (_, row) in zip(y, coefs.iterrows()):
    stars = "***" if row.p < .001 else "**" if row.p < .01 else "*" if row.p < .05 else ""
    ax.annotate(f"{row.estimate:+.0f}{stars}", (row.high, yi), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=10, color="#3A3A3C")

ax.set_yticks(y, [labels[t] for t in coefs.index])
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Effect on violent crimes per 100k (per 1 SD), 95% CI")
ax.set_title("What predicts violent crime across US states?", pad=26)
ax.text(0, 1.02, f"OLS · n = {int(model.nobs)} · R² = {model.rsquared:.2f} · "
        "* p<.05  ** p<.01  *** p<.001", transform=ax.transAxes,
        fontsize=9.5, color="#6E6E73")
ax.margins(x=0.15)
plt.show()
