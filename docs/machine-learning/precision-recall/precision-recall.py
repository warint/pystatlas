"""
title: Precision–recall curves
summary: Out-of-fold precision–recall curves for two classifiers predicting which Facebook users share COVID-19 fake news, with average precision and the no-skill baseline.
r: yardstick::pr_curve() + autoplot()
tags: [classification, precision, recall, average precision, imbalanced data, misinformation]
description: |
  When the positive class is a minority, a ROC curve can look excellent while
  many alarms are false. The precision–recall curve answers the operational question
  instead: if I want to catch this share of positives (**recall**), how many of
  my alerts will be right (**precision**)? Each point is one decision threshold.
  The dashed line is the no-skill baseline — the prevalence of the positive
  class — and the area under each curve is summarised by the **average
  precision** (AP).

  The example is set in the context of Melchior, Warin and Oliveira (2025,
  *Technological Forecasting and Social Change*), who study the sharing of
  COVID-19 fake news on Facebook with a mixed-methods approach. The data are
  **simulated** for illustration and are *not* the paper's data or results:
  n = 3,000 users, of whom about one in eight shared a fake-news post, described
  by age, daily hours on Facebook, COVID-19 groups joined, trust in mainstream
  media, belief in conspiracy theories and the share of friends who shared fake
  news. Scores are out-of-fold probabilities from a 5-fold cross-validation
  (`cross_val_predict(method="predict_proba")`); curves and AP come from
  `precision_recall_curve` and `average_precision_score`. The marker shows the
  default 0.5 threshold on each curve: with a rare class, it flags only a
  minority of the sharers.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Simulated data, set in the context of Melchior, Warin & Oliveira (2025) [melchior2025fake]

# Data: 3,000 Facebook users; 1 = shared a COVID-19 fake-news post (a minority class)
rng = np.random.default_rng(2024)
n = 3000
X = pd.DataFrame({"age": rng.integers(18, 80, n),            # years
                  "hours": rng.gamma(2, 0.9, n),             # daily hours on Facebook
                  "groups": rng.poisson(3, n),               # COVID-19 groups joined
                  "trust_media": rng.integers(1, 8, n),      # 1-7 scale
                  "conspiracy": rng.integers(1, 8, n),       # 1-7 scale
                  "friends_shared": rng.beta(1, 8, n)})      # share of friends who shared
logit = (-4.3 + 0.025 * (X.age - 45) + 0.35 * X.hours + 0.15 * X.groups
         - 0.3 * (X.trust_media - 4) + 0.45 * (X.conspiracy - 4) + 6 * X.friends_shared)
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
prevalence = y.mean()

models = {
    "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0),
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}
palette = ["#007AFF", "#FF9500"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
recall = {}

fig, ax = plt.subplots(figsize=(8, 5.2))
for (name, model), c in zip(models.items(), palette):
    score = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y, score)
    ap = average_precision_score(y, score)
    ax.step(rec, prec, where="post", color=c, lw=2.2, label=f"{name}  (AP = {ap:.3f})")
    k = np.argmin(np.abs(thr - 0.5))  # the default threshold
    recall[name] = rec[k]
    ax.scatter(rec[k], prec[k], s=70, color=c, edgecolor="white", lw=1.5, zorder=4)
    ax.annotate(f"threshold 0.5\nP = {prec[k]:.2f}, R = {rec[k]:.2f}", (rec[k], prec[k]),
                xytext=(40, -110) if c == palette[0] else (14, 26), textcoords="offset points",
                ha="left", fontsize=9, color=c,
                arrowprops=dict(arrowstyle="-", color=c, lw=0.8))

ax.axhline(prevalence, color="#8E8E93", ls="--", lw=1.4)
ax.text(0.01, prevalence + 0.015, f"no skill = prevalence ({prevalence:.1%})", fontsize=9, color="#6E6E73")
ax.set_xlim(0, 1.01)
ax.set_ylim(0, 1.03)
ax.set_xlabel("Recall (share of fake-news sharers who are flagged)")
ax.set_ylabel("Precision (share of flagged users who shared)")
ax.grid(axis="x")
ax.legend(loc="upper right")
ax.set_title(f"At the default 0.5 threshold, logistic regression flags only {recall['Logistic regression']:.0%} "
             "of fake-news sharers", pad=26)
ax.text(0, 1.02, f"Out-of-fold scores, 5-fold CV · n = {n:,} users (simulated), {y.sum():,} shared fake news · "
        "AP = average precision", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
