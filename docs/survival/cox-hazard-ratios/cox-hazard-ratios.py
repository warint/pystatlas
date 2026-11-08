"""
title: Cox hazard ratios
summary: Forest plot of a Cox model of time to patent grant, with HR, 95% CI and p-values.
r: survminer::ggforest()
tags: [Cox model, hazard ratio, forest plot, proportional hazards, survival, patents]
description: |
  A Cox proportional hazards model estimates how each covariate multiplies the
  instantaneous risk of the event — here the "risk" of being granted, so a hazard
  ratio above 1 means a **faster** grant. Each row shows a **hazard ratio** (HR)
  with its 95% confidence interval on a log scale, so that halving and doubling
  sit at equal distances from the reference line at HR = 1. The table on the right
  repeats the numbers, as in a published forest plot.

  The setting is Vorreuther & Warin (2021, *Data in Brief*), a dataset on patent
  relatedness and velocity in the Chinese pharmaceutical industry. The data here
  are **simulated** for illustration and are not the paper's data or results: 400
  pharmaceutical patent applications followed in months from filing to grant, with
  the number of claims, the applicant's type and the Jaccard relatedness of the
  application to the applicant's earlier patents. The model is fitted with
  `lifelines.CoxPHFitter`; the subtitle reports the concordance index and the
  likelihood-ratio test of the whole model.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
# Simulated data, set in the context of Vorreuther & Warin (2021) [vorreuther2021]

# Data: months from filing to grant of 400 patent applications (shared generator)
rng = np.random.default_rng(2021)
n = 400
df = pd.DataFrame({"foreign": rng.integers(0, 2, n), "university": (rng.random(n) < 0.25).astype(int),
                   "claims": rng.poisson(12, n), "relatedness": rng.beta(2, 4, n).round(2)})
lp = 0.6 * df.foreign - 0.04 * (df.claims - 12) + 2.0 * (df.relatedness - 0.33) - 0.35 * df.university
grant = 40 * rng.weibull(2, n) * np.exp(-lp / 2)        # Weibull PH: higher lp, faster grant
cens = rng.uniform(24, 96, n)                           # end of observation window
df["months"], df["granted"] = np.minimum(grant, cens).round(1), (grant <= cens).astype(int)

# Cox model: claims per 10, relatedness per 0.1 so that hazard ratios are readable
X = df.assign(claims=df.claims / 10, relatedness=df.relatedness * 10)
cph = CoxPHFitter().fit(X, duration_col="months", event_col="granted")
lr = cph.log_likelihood_ratio_test()

# Tidy the summary: HR, 95% CI, p-value
labels = {"foreign": "Foreign applicant", "university": "University applicant",
          "claims": "Claims (per 10)", "relatedness": "Relatedness (per 0.1 Jaccard)"}
res = cph.summary.rename(columns={"exp(coef)": "hr", "exp(coef) lower 95%": "low",
                                  "exp(coef) upper 95%": "high"})
res = res.sort_values("hr")
y = np.arange(len(res))
colors = np.where(res.p < 0.05, np.where(res.hr > 1, "#FF3B30", "#007AFF"), "#AEAEB2")

fig, (ax, tab) = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True,
                              gridspec_kw=dict(width_ratios=[3, 1.3]))

# Forest: CI whiskers and point estimates on a log axis
ax.axvline(1, color="#8E8E93", lw=1, ls="--", zorder=1)
ax.hlines(y, res.low, res.high, color=colors, lw=2.5, zorder=2)
ax.scatter(res.hr, y, s=80, marker="s", color=colors, edgecolor="white", lw=1.2, zorder=3)
ax.set_xscale("log")
ax.set_xticks([0.4, 0.6, 1, 1.5, 2.5], ["0.4", "0.6", "1", "1.5", "2.5"])
ax.minorticks_off()
ax.set_xlim(0.26, 3)
ax.set_yticks(y, [labels[t] for t in res.index])
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Hazard ratio (log scale)")
ax.text(0.98, -0.6, "faster grant", ha="right", color="#FF3B30", fontsize=9,
        transform=ax.get_yaxis_transform())
ax.text(0.02, -0.6, "slower grant", ha="left", color="#007AFF", fontsize=9,
        transform=ax.get_yaxis_transform())

# Table: HR [95% CI] and p-value
for yi, (_, r) in zip(y, res.iterrows()):
    p = "< .001" if r.p < 0.001 else f"{r.p:.3f}".lstrip("0")
    tab.text(0.22, yi, f"{r.hr:.2f}", va="center", ha="right", fontsize=10, color="#1D1D1F")
    tab.text(0.27, yi, f"[{r.low:.2f}, {r.high:.2f}]", va="center", fontsize=10, color="#6E6E73")
    tab.text(1, yi, p, va="center", ha="right", fontsize=10,
             color="#1D1D1F" if r.p < 0.05 else "#8E8E93",
             fontweight="semibold" if r.p < 0.05 else "normal")
tab.text(0.22, len(res) - 0.3, "HR", ha="right", fontsize=9.5, color="#6E6E73", fontweight="semibold")
tab.text(0.27, len(res) - 0.3, "95% CI", fontsize=9.5, color="#6E6E73", fontweight="semibold")
tab.text(1, len(res) - 0.3, "p", ha="right", fontsize=9.5, color="#6E6E73", fontweight="semibold")
tab.set(xlim=(0, 1), ylim=(-1, len(res)))
tab.axis("off")

ax.set_title("What speeds up or slows down a patent grant?", pad=26)
ax.text(0, 1.02, f"Cox PH · n = {n} applications (simulated), {df.granted.sum()} grants · C = "
        f"{cph.concordance_index_:.2f} · LR χ²({lr.degrees_freedom:.0f}) = {lr.test_statistic:.1f}, "
        f"p < .001", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
