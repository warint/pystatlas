"""
title: SHAP summary (beeswarm)
summary: Every production batch's SHAP value for every process feature, coloured by the feature's value, with mean |SHAP| importance.
r: shapviz::sv_importance(kind = "beeswarm")
subcategory: advanced
tags: [SHAP, Shapley values, beeswarm, feature importance, interpretability, manufacturing]
description: |
  SHAP (SHapley Additive exPlanations; Lundberg and Lee, 2017) splits each
  prediction into additive contributions, one per feature, so that
  **base value + sum of contributions = prediction**. The contributions are
  Shapley values from cooperative game theory: each feature's average marginal
  effect over all orders in which features could be added.

  The beeswarm shows one dot per batch and feature. Position is the SHAP value —
  how far that feature pushed this batch's predicted waste up or down — and
  colour is the feature's own value, from low (blue) to high (red). Features are
  sorted by mean |SHAP|, printed on the right as a global importance. A clean
  red-right / blue-left split means a monotone effect; mixed colours signal
  non-linearity (temperature, whose effect is U-shaped) or interactions.

  The example is set in the context of Warin, d'Anglade and de Marcellis-Warin
  (2025, *International Journal of Sustainable Engineering*) on data science for
  waste reduction in manufacturing. The data are **simulated** for illustration
  and are *not* the paper's data or results: the same n = 600 production
  batches as in the model comparison, with the waste rate in percentage points.
  The model is a gradient-boosting regressor and the exact tree SHAP values come
  from `shap.TreeExplainer`. The figure is drawn with Matplotlib so it matches
  the gallery style; `shap.plots.beeswarm` gives a similar view in one line.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
# Simulated data, set in the context of Warin, d'Anglade & de Marcellis-Warin (2025) [warin2025waste]

# Data: 600 production batches; outcome = waste rate (% of material scrapped)
rng = np.random.default_rng(2025)
n = 600
names = {"speed": "Machine speed (units/min)", "temperature": "Temperature (°C)",
         "humidity": "Humidity (%)", "experience": "Operator experience (years)",
         "maintenance": "Days since maintenance", "grade": "Material grade (1 = best)",
         "batch_size": "Batch size (kg)"}
X = pd.DataFrame({"speed": rng.uniform(80, 160, n), "temperature": rng.uniform(195, 225, n),
                  "humidity": rng.uniform(30, 75, n), "experience": rng.gamma(2, 4, n),
                  "maintenance": rng.uniform(0, 120, n), "grade": rng.integers(1, 4, n),
                  "batch_size": rng.uniform(200, 1000, n)})       # batch size has no effect
y = (2 + 0.002 * np.clip(X.speed - 110, 0, None) ** 2            # waste climbs above 110 units/min
     + 0.025 * (X.temperature - 210) ** 2                        # U-shape around the 210 °C set point
     + 3 * np.exp(-X.experience / 3)                             # novices waste more, gains fade
     + 4 * (X.humidity - 30) / 45 * X.maintenance / 120          # humidity hurts worn machines most
     + 0.6 * (X.grade - 1) + rng.normal(0, 0.7, n))
model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, max_depth=2,
                                  subsample=0.8, random_state=0).fit(X, y)

# Exact SHAP values for tree ensembles
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X)                           # shape: (batches, features)
importance = np.abs(sv).mean(axis=0)
order = np.argsort(importance)                          # least important at the bottom

fig, ax = plt.subplots(figsize=(8, 5.2))
jit = np.random.default_rng(0)
cmap = plt.get_cmap("coolwarm")
for i, j in enumerate(order):
    v = sv[:, j]
    xv = X.iloc[:, j].rank(pct=True).to_numpy()         # colour = percentile of the feature value
    # Beeswarm-style vertical jitter: wider where the points are denser
    hist, edges = np.histogram(v, bins=60)
    dens = hist[np.clip(np.digitize(v, edges) - 1, 0, len(hist) - 1)] / hist.max()
    ax.scatter(v, i + jit.uniform(-1, 1, len(v)) * 0.38 * np.sqrt(dens), c=xv, cmap=cmap, s=9,
               alpha=0.85, lw=0, vmin=0, vmax=1)
    ax.text(1.01, i, f"{importance[j]:.2f}", transform=ax.get_yaxis_transform(), va="center",
            fontsize=9.5, color="#3A3A3C")
ax.axvline(0, color="#3A3A3C", lw=0.8)
ax.set_yticks(range(len(order)), [names[X.columns[j]] for j in order])
ax.set_xlabel("SHAP value (percentage points of waste)")
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.text(1.01, -0.65, "mean\n|SHAP|", transform=ax.get_yaxis_transform(),
        fontsize=8.5, color="#6E6E73", va="top")

cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, pad=0.09, aspect=40, ticks=[0, 1])
cb.ax.set_yticklabels(["low", "high"])
cb.set_label("Feature value (percentile)", fontsize=9.5)
cb.outline.set_visible(False)

j = order[-1]
ax.set_title(f"{names[X.columns[j]].split(' (')[0]} moves predicted waste from {sv[:, j].min():+.1f} "
             f"to {sv[:, j].max():+.1f} points", pad=26)
ax.text(0, 1.02, f"Gradient boosting · exact tree SHAP · n = {n} batches (simulated) · "
        f"base value {float(np.ravel(explainer.expected_value)[0]):.1f}%", transform=ax.transAxes, fontsize=9.5,
        color="#6E6E73")
plt.show()
