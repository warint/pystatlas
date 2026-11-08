"""
title: Adjusted survival curves
summary: Cox-predicted time to patent grant at four levels of relatedness, other covariates at their mean.
r: survminer::ggadjustedcurves()
tags: [Cox model, adjusted survival, prediction, survival, covariates, patents]
description: |
  A hazard ratio is hard to feel; a survival curve is not. Here a fitted Cox
  proportional hazards model is used to **predict** the whole curve for
  hypothetical applications that differ only in their relatedness to the
  applicant's earlier patents, while every other covariate is fixed at its sample
  mean. The horizontal spread between the curves is the effect of relatedness
  translated into months of waiting for a grant.

  The setting is Vorreuther & Warin (2021, *Data in Brief*), a dataset on patent
  relatedness and velocity in the Chinese pharmaceutical industry. The data here
  are **simulated** for illustration and are not the paper's data or results: 400
  pharmaceutical patent applications, with relatedness measured as a Jaccard
  similarity between 0 and 1. The model is fitted with `lifelines.CoxPHFitter` on
  all four covariates and the curves come from `predict_survival_function()`; the
  observed Kaplan–Meier curve of the whole sample is drawn in grey for reference.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
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

cph = CoxPHFitter().fit(df, duration_col="months", event_col="granted")

# Hypothetical applications: everything at the mean, relatedness varies
levels = [0.1, 0.3, 0.5, 0.7]
profiles = pd.DataFrame([df.drop(columns=["months", "granted"]).mean()] * len(levels))
profiles["relatedness"] = levels
surv = cph.predict_survival_function(profiles, times=np.arange(0, 73))
colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(levels)))

fig, ax = plt.subplots(figsize=(8, 5))

# Observed Kaplan–Meier curve of the whole sample, for reference
km = KaplanMeierFitter().fit(df.months, df.granted)
ax.step(km.survival_function_.index, km.survival_function_.iloc[:, 0], where="post",
        color="#AEAEB2", lw=1.5, ls="--", label=f"Observed Kaplan–Meier, all {n} applications")

# Model-predicted curves, with the predicted median time to grant in the legend
for col, r, c in zip(surv.columns, levels, colors):
    s = surv[col]
    med = s.index[(s <= 0.5).argmax()]
    ax.step(s.index, s, where="post", color=c, lw=2.2,
            label=f"relatedness = {r:.1f}  ·  median {med:.0f} months")
ax.axhline(0.5, color="#C7C7CC", lw=1, zorder=0)
ax.legend(loc="upper right")

hr = np.exp(0.1 * cph.params_["relatedness"])
lo, hi = np.exp(0.1 * cph.confidence_intervals_.loc["relatedness"])
p = cph.summary.loc["relatedness", "p"]
p_txt = "p < .001" if p < 0.001 else f"p = {p:.3f}"
ax.set(xlim=(0, 72), ylim=(0, 1.01), xticks=range(0, 73, 12),
       xlabel="Months since filing", ylabel="Predicted probability still pending")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.set_title("Applications related to earlier patents are granted sooner, all else equal", pad=26)
ax.text(0, 1.02, f"Cox PH, other covariates at their mean  ·  HR per 0.1 relatedness = "
        f"{hr:.2f} [{lo:.2f}, {hi:.2f}], {p_txt}  ·  n = {n} (simulated)",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
