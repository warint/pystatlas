"""
title: SHAP dependence plot
summary: SHAP value of one feature against its value, coloured by its strongest interaction partner, with a smooth trend.
r: shapviz::sv_dependence()
subcategory: advanced
tags: [SHAP, dependence plot, interaction, non-linearity, LOWESS, interpretability]
description: |
  A SHAP dependence plot zooms in on one feature. Each dot is a patient: the
  horizontal axis is the feature's value, the vertical axis its SHAP value —
  its contribution to that patient's prediction. The trend line (a LOWESS
  smooth) is close to a partial-dependence curve, but the cloud around it is
  informative: at the same body-mass index, contributions differ from patient
  to patient because of **interactions** with other features.

  Colour shows the feature that best explains that vertical spread, chosen here
  as the feature whose SHAP **interaction** values with body-mass index are
  largest on average (`shap.TreeExplainer.shap_interaction_values`). Data are the
  442 diabetes patients bundled with **scikit-learn**; the model is a
  gradient-boosting regressor; the smooth comes from
  `statsmodels.nonparametric.lowess`.
"""
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess

data = load_diabetes(as_frame=True, scaled=False)
X, y = data.data, data.target
names = {"age": "Age", "sex": "Sex", "bmi": "Body-mass index", "bp": "Blood pressure",
         "s1": "Total cholesterol", "s2": "LDL", "s3": "HDL", "s4": "Chol. / HDL",
         "s5": "Triglycerides (log)", "s6": "Blood sugar"}
model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, max_depth=2,
                                  subsample=0.8, random_state=0).fit(X, y)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X)
inter = explainer.shap_interaction_values(X)            # (patients, features, features)

f = list(X.columns).index("bmi")
strength = np.abs(inter[:, f, :]).mean(axis=0)
strength[f] = 0
partner = int(np.argmax(strength))                       # strongest interaction partner

x, s = X.iloc[:, f].to_numpy(), sv[:, f]
col = X.iloc[:, partner].rank(pct=True).to_numpy()
trend = lowess(s, x, frac=0.35)

fig, ax = plt.subplots(figsize=(8, 5.2))
sc = ax.scatter(x, s, c=col, cmap="coolwarm", s=22, alpha=0.8, lw=0, vmin=0, vmax=1)
ax.plot(trend[:, 0], trend[:, 1], color="#1d1d1f", lw=2.2, label="LOWESS trend")
ax.axhline(0, color="#8E8E93", lw=0.8)
ax.set_xlabel("Body-mass index (kg/m²)")
ax.set_ylabel("SHAP value of body-mass index")
ax.grid(axis="x")
ax.legend(loc="upper left")
cb = fig.colorbar(sc, ax=ax, pad=0.02, aspect=40, ticks=[0, 1])
cb.ax.set_yticklabels(["low", "high"])
cb.set_label(f"{names[X.columns[partner]]} (percentile)", fontsize=9.5)
cb.outline.set_visible(False)

above = s[x > 30].mean()
ax.set_title(f"Above a BMI of 30, body-mass index adds {above:.0f} points on average", pad=26)
ax.text(0, 1.02, f"Gradient boosting · n = {len(X)} patients · colour = strongest interaction "
        f"partner ({names[X.columns[partner]]})", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
