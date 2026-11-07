"""
title: Partial dependence with ICE curves
summary: Average model response to one feature (PDP) over the individual curves (ICE) of every patient.
r: pdp::partial(ice = TRUE) + autoplot() / iml::FeatureEffect
tags: [partial dependence, ICE, interpretability, non-linearity, gradient boosting]
description: |
  A partial dependence plot (PDP) shows how the model's prediction changes, on
  average, as one feature varies while all others keep their observed values.
  Behind the average sit the **individual conditional expectation** (ICE)
  curves — one per observation. Parallel ICE curves mean the effect is the same
  for everyone; curves that fan out or cross reveal **interactions** that the
  average hides. ICE curves are centred at the left edge so that their shapes,
  not their levels, can be compared; the rug marks where the data actually are,
  since the model extrapolates elsewhere.

  Data are the 442 diabetes patients bundled with **scikit-learn**; the model is
  a gradient-boosting regressor, and the curves come from
  `sklearn.inspection.partial_dependence(kind="both")`. A PDP describes the
  model, not the world: it is causal only if the model is.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence

data = load_diabetes(as_frame=True, scaled=False)
X, y = data.data, data.target
model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, max_depth=2,
                                  subsample=0.8, random_state=0).fit(X, y)

features = {"bmi": "Body-mass index (kg/m²)", "s5": "Triglycerides (log scale)"}
plt.rcParams["figure.constrained_layout.use"] = False   # manual margins leave room for the subtitle
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.12, wspace=0.08)
for ax, (f, label) in zip(axes, features.items()):
    pd_res = partial_dependence(model, X, [f], kind="both", grid_resolution=40)
    grid = pd_res["grid_values"][0]
    ice = pd_res["individual"][0]
    ice_c = ice - ice[:, [0]]                      # centred ICE
    pdp_c = pd_res["average"][0] - pd_res["average"][0][0]
    ax.plot(grid, ice_c.T, color="#007AFF", alpha=0.07, lw=0.8)
    ax.plot(grid, pdp_c, color="#FF9500", lw=3, label="Average (PDP)")
    lo, hi = np.percentile(ice_c, [10, 90], axis=0)
    ax.plot(grid, lo, color="#007AFF", lw=1, ls="--")
    ax.plot(grid, hi, color="#007AFF", lw=1, ls="--", label="10th / 90th percentile of ICE")
    ax.plot(X[f], np.full(len(X), -16), "|", color="#3A3A3C", ms=8, alpha=0.4)   # rug
    ax.set_xlabel(label)
    ax.set_title(f"Effect of {label.split(' (')[0].lower()}")
axes[0].set_ylim(-20, 140)
axes[0].set_ylabel("Change in predicted progression\n(relative to the lowest value)")
axes[1].legend(loc="upper left", fontsize=9)
fig.suptitle("Both effects rise in steps — and the triglyceride effect fans out across patients",
             x=0.02, y=0.97, ha="left")
fig.text(0.02, 0.885, f"Gradient boosting · n = {len(X)} patients · thin lines = one patient (centred ICE) · "
         "ticks = observed values", fontsize=9.5, color="#6E6E73")
plt.show()
