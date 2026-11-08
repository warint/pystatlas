"""
title: Permutation importance
summary: Which process features a random forest relies on to predict a batch's waste rate — drop in test-set R² when each one is shuffled.
r: vip::vip(method = "permute")
tags: [feature importance, permutation, model-agnostic, interpretability, random forest, manufacturing]
description: |
  Permutation importance asks a simple, model-agnostic question: how much worse
  does the model predict on **held-out** data when one feature is shuffled,
  breaking its link with the outcome? A large drop means the model relies on
  that feature. Shuffling is random, so it is repeated 50 times per feature; the
  box shows the spread and the dot the mean. Features whose box straddles zero
  carry no usable signal for this model.

  Unlike the impurity-based importances built into tree models, permutation
  importance is computed on test data, is comparable across model types and is
  not biased towards features with many distinct values. It measures what the
  **model** uses, not causal effects. The example is set in the context of Warin,
  d'Anglade and de Marcellis-Warin (2025, *International Journal of Sustainable
  Engineering*) on data science for waste reduction in manufacturing. The data
  are **simulated** for illustration and are *not* the paper's data or results:
  n = 600 production batches (the same as in the model comparison), 180 of them
  held out, with seven process features, one of which — batch size — was built
  to have no effect. The model is a random forest; importances come from
  `sklearn.inspection.permutation_importance`.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
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
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0).fit(X_tr, y_tr)
r2 = model.score(X_te, y_te)
imp = permutation_importance(model, X_te, y_te, n_repeats=50, random_state=0, scoring="r2")

order = np.argsort(imp.importances_mean)
fig, ax = plt.subplots(figsize=(8, 5.2))
for i, j in enumerate(order):
    v = imp.importances[j]
    c = "#007AFF" if np.percentile(v, 2.5) > 0 else "#8E8E93"
    ax.boxplot(v, positions=[i], widths=0.5, vert=False, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=c, alpha=0.25, edgecolor=c), medianprops=dict(color=c, lw=1.5),
               whiskerprops=dict(color=c), capprops=dict(visible=False))
    ax.scatter(v.mean(), i, s=40, color=c, zorder=3, edgecolor="white", lw=1)
ax.axvline(0, color="#3A3A3C", lw=0.8)
ax.set_yticks(range(len(order)), [names[X.columns[j]] for j in order])
ax.set_xlabel("Drop in test-set R² when the feature is shuffled")
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
top = X.columns[order[-1]]
ax.set_title(f"{names[top].split(' (')[0]} matters most: shuffling it costs "
             f"{imp.importances_mean[order[-1]]:.2f} of R²; batch size adds nothing", pad=26)
ax.text(0, 1.02, f"Random forest · test R² = {r2:.2f} · n = {len(X_te)} held-out batches (simulated) · "
        "50 shuffles per feature · grey = interval includes 0", transform=ax.transAxes,
        fontsize=9.5, color="#6E6E73")
plt.show()
