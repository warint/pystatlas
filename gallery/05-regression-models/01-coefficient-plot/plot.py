"""
title: Coefficient plot
summary: Regression estimates with 95% confidence intervals, one row per term.
r: ggstats::ggcoef_model()
tags: [regression, OLS, confidence interval, dot-and-whisker, economic growth]
description: |
  A coefficient plot (a.k.a. dot-and-whisker plot) replaces the regression table.
  Each point is an estimate, each line its 95% confidence interval, and the dashed
  line marks zero. Terms whose interval crosses zero are drawn in grey, so the
  significant effects stand out at a glance.

  The setting is Stojkov, Veljanovski & Warin (2024, *Post-Communist Economies*):
  growth in emerging Europe, institutions and policy factors. The data are
  **simulated** for illustration — they are not the paper's data or results:
  n = 220 country-years (11 economies × 20 years), with annual growth of GDP per
  capita explained by six predictors in the **statsmodels** formula API
  (`smf.ols()`). Predictors are standardised first, so the coefficients are
  directly comparable: percentage points of growth for a one-standard-deviation
  change in the predictor.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
# Simulated data, set in the context of Stojkov, Veljanovski & Warin (2024) [stojkov2024growth]

# Data: 11 emerging-European economies × 20 years, six institutional and policy factors
rng = np.random.default_rng(2024)
n = 220
labels = {"rule_law": "Rule of law", "reg_quality": "Regulatory quality",
          "investment": "Investment (% of GDP)", "openness": "Trade openness",
          "inflation": "Inflation", "gov_size": "Government consumption"}
predictors = list(labels)
df = pd.DataFrame(rng.normal(size=(n, 6)), columns=predictors)
df["growth"] = (3 + 0.9 * df.rule_law + 0.45 * df.reg_quality + 0.8 * df.investment
                + 0.15 * df.openness - 0.7 * df.inflation - 0.05 * df.gov_size
                + rng.normal(0, 2.5, n))
df[predictors] = (df[predictors] - df[predictors].mean()) / df[predictors].std()

model = smf.ols("growth ~ " + " + ".join(predictors), data=df).fit()

# Tidy the results: estimate, 95% CI and p-value per term (intercept dropped)
ci = model.conf_int().drop("Intercept")
coefs = (
    ci.assign(estimate=model.params, p=model.pvalues)
      .rename(columns={0: "low", 1: "high"})
      .sort_values("estimate")
)

fig, ax = plt.subplots(figsize=(8, 4.6))
y = np.arange(len(coefs))
colors = np.where(coefs["p"] < 0.05, "#007AFF", "#AEAEB2")

ax.axvline(0, color="#8E8E93", lw=1, ls="--", zorder=1)
ax.hlines(y, coefs["low"], coefs["high"], color=colors, lw=2.5, zorder=2)
ax.scatter(coefs["estimate"], y, s=80, color=colors, edgecolor="white", lw=1.5, zorder=3)

for yi, (_, row) in zip(y, coefs.iterrows()):
    stars = "***" if row.p < .001 else "**" if row.p < .01 else "*" if row.p < .05 else ""
    ax.annotate(f"{row.estimate:+.2f}{stars}", (row.high, yi), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=10, color="#3A3A3C")

ax.set_yticks(y, [labels[t] for t in coefs.index])
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Effect on GDP-per-capita growth (pp per 1 SD), 95% CI")
ax.set_title("Rule of law and investment go with faster growth", pad=26)
ax.text(0, 1.02, f"OLS · n = {int(model.nobs)} country-years (simulated) · R² = {model.rsquared:.2f} · "
        "* p<.05  ** p<.01  *** p<.001", transform=ax.transAxes,
        fontsize=9.5, color="#6E6E73")
ax.margins(x=0.15)
plt.show()
