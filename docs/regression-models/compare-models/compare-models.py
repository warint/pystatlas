"""
title: Comparing nested models
summary: Coefficients of three nested regressions side by side, to see how estimates shift.
r: ggstats::ggcoef_compare()
tags: [regression, model comparison, nested models, confidence interval, OLS]
description: |
  When covariates are added one block at a time, the interesting question is often
  not *which* terms are significant but *how the estimates move*. Here each term
  gets one row and each model one dodged dot-and-whisker, so you can follow a
  coefficient as the model grows. Hollow markers are not significant at 5%. The
  subtitle compares the models with R² and F-tests of the added blocks.

  Data are the 442 diabetes patients bundled with **scikit-learn**
  (`load_diabetes`): one-year disease progression explained by body mass index,
  blood pressure and serum measurements. Predictors are standardised, so every
  coefficient is the change in progression per standard deviation. Models are fitted
  with `smf.ols()` and compared with `anova_lm()`.
"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from sklearn.datasets import load_diabetes
from statsmodels.stats.anova import anova_lm

# Data: diabetes progression; standardise predictors (z-scores)
df = load_diabetes(as_frame=True, scaled=False).frame
X = ["bmi", "bp", "s5", "s3", "age", "sex"]
df[X] = (df[X] - df[X].mean()) / df[X].std()

# Three nested models, each adding a block of predictors
formulas = {"M1 · body": "target ~ bmi + bp",
            "M2 · + serum": "target ~ bmi + bp + s5 + s3",
            "M3 · + demographics": "target ~ bmi + bp + s5 + s3 + age + sex"}
fits = {name: smf.ols(f, data=df).fit() for name, f in formulas.items()}
f_tests = anova_lm(*fits.values())

labels = {"bmi": "Body mass index", "bp": "Blood pressure", "s5": "Triglycerides (log)",
          "s3": "HDL cholesterol", "age": "Age", "sex": "Sex"}
terms = list(labels)[::-1]
palette = ["#007AFF", "#FF9500", "#34C759"]

fig, ax = plt.subplots(figsize=(8, 5.4))
ax.axvline(0, color="#8E8E93", lw=1, ls="--", zorder=1)
for k, ((name, m), c) in enumerate(zip(fits.items(), palette)):
    offset = (1 - k) * 0.22                      # dodge: first model on top
    ci = m.conf_int()
    for i, t in enumerate(terms):
        if t not in m.params:
            continue
        sig = m.pvalues[t] < 0.05
        ax.hlines(i + offset, ci.loc[t, 0], ci.loc[t, 1], color=c, lw=2.2, zorder=2)
        ax.scatter(m.params[t], i + offset, s=55, zorder=3, lw=1.8,
                   color=c if sig else "white", edgecolor=c if not sig else "white")
    ax.plot([], [], "o-", color=c, lw=2.2, ms=7,
            label=f"{name}  (R² = {m.rsquared:.2f})")

for i in range(len(terms) - 1):                  # thin separators between terms
    ax.axhline(i + 0.5, color="#F2F2F7", lw=1, zorder=0)
ax.set_yticks(range(len(terms)), [labels[t] for t in terms])
ax.set_ylim(-0.6, len(terms) - 0.4)
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Effect on disease progression per 1 SD (95% CI)")
ax.legend(loc="lower right", handlelength=1.6, borderaxespad=0.2)
ax.set_title("Adding serum markers shrinks the BMI effect by a third", pad=26)
fmt_p = lambda p: "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")
blocks = [f"M{j} vs M{j + 1}: F({f_tests.df_diff[j]:.0f}, {f_tests.df_resid[j]:.0f}) = "
          f"{f_tests.F[j]:.1f}, {fmt_p(f_tests['Pr(>F)'][j])}" for j in (1, 2)]
ax.text(0, 1.02, f"OLS · n = {int(fits['M1 · body'].nobs)} · " + " · ".join(blocks)
        + " · hollow = p ≥ .05", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
