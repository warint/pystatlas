"""
title: SHAP waterfall for one prediction
summary: How each feature moves one patient's prediction from the average to the model's output.
r: shapviz::sv_waterfall()
subcategory: advanced
tags: [SHAP, waterfall, local explanation, individual prediction, interpretability]
description: |
  Global charts describe the model; a waterfall explains **one** prediction.
  It starts from the base value — the model's average prediction over the
  data — and adds each feature's SHAP value in turn, largest first, until it
  reaches this patient's prediction. Red bars push the prediction up, blue bars
  pull it down; the smallest contributions are pooled into one bar so the chart
  stays readable. Because SHAP values are additive, the bars always sum exactly
  to the gap between the base value and the prediction.

  This is the chart to show when a single decision must be justified — a loan,
  a diagnosis, a flagged transaction. It explains what the **model** did, not
  why the outcome happens. Data are the 442 diabetes patients bundled with
  **scikit-learn**; the model is a gradient-boosting regressor; SHAP values come
  from `shap.TreeExplainer`. The patient shown is a high-risk case (more than
  60 points above the base value) chosen because its explanation pulls in both
  directions.
"""
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor

data = load_diabetes(as_frame=True, scaled=False)
X, y = data.data, data.target
names = {"age": "Age", "sex": "Sex", "bmi": "Body-mass index", "bp": "Blood pressure",
         "s1": "Total cholesterol", "s2": "LDL", "s3": "HDL", "s4": "Chol. / HDL",
         "s5": "Triglycerides (log)", "s6": "Blood sugar"}
fmt = {"age": "{:.0f} y", "sex": "{:.0f}", "bmi": "{:.1f}", "bp": "{:.0f}", "s1": "{:.0f}",
       "s2": "{:.0f}", "s3": "{:.0f}", "s4": "{:.1f}", "s5": "{:.2f}", "s6": "{:.0f}"}
model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, max_depth=2,
                                  subsample=0.8, random_state=0).fit(X, y)

explainer = shap.TreeExplainer(model)
base = float(np.ravel(explainer.expected_value)[0])
all_sv = explainer.shap_values(X)
# A high-risk patient whose explanation has pushes in both directions
high = np.where(all_sv.sum(axis=1) > 60)[0]
p = int(high[np.argmin(all_sv[high].min(axis=1))])
sv = all_sv[p]

# Largest contributions first; pool the rest
k = 6
idx = np.argsort(np.abs(sv))[::-1]
top, rest = idx[:k], idx[k:]
labels = [f"{names[X.columns[j]]} = {fmt[X.columns[j]].format(X.iloc[p, j])}" for j in top]
values = list(sv[top])
if len(rest):
    labels.append(f"{len(rest)} other features")
    values.append(sv[rest].sum())

fig, ax = plt.subplots(figsize=(8, 5.2))
start = base
for i, (lab, v) in enumerate(zip(labels, values)):
    c = "#FF3B30" if v > 0 else "#007AFF"
    ax.barh(i, v, left=start, color=c, alpha=0.85, height=0.62)
    ax.text(start + v + (2 if v > 0 else -2), i, f"{v:+.0f}", va="center",
            ha="left" if v > 0 else "right", fontsize=9.5, color=c, fontweight="semibold")
    if i < len(values) - 1:
        ax.plot([start + v] * 2, [i + 0.31, i + 0.69], color="#8E8E93", lw=0.8)
    start += v
ax.axvline(base, color="#8E8E93", ls="--", lw=1)
ax.axvline(start, color="#1d1d1f", lw=1.2)
bottom = len(labels) - 0.45
ax.text(base, bottom, f"base value {base:.0f}", ha="center", va="top", fontsize=9, color="#6E6E73",
        bbox=dict(fc="white", ec="none", pad=1))
ax.text(start, bottom, f"prediction {start:.0f}", ha="center", va="top", fontsize=9,
        color="#1d1d1f", fontweight="semibold", bbox=dict(fc="white", ec="none", pad=1))
ax.set_yticks(range(len(labels)), labels)
ax.set_ylim(len(labels) + 0.2, -0.6)
lo_x = min(base, *(base + np.cumsum(values)))
ax.set_xlim(lo_x - 30, max(start, base + np.cumsum(values).max()) + 40)
ax.set_xlabel("Predicted disease progression")
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_title(f"Why this patient's predicted progression is {start - base:.0f} points above average", pad=26)
ax.text(0, 1.02, f"Patient #{p} · gradient boosting · SHAP values sum exactly from base value to prediction "
        f"(observed: {y.iloc[p]:.0f})", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
