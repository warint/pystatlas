# Cross-validated model comparison

> Fold-by-fold R² of six models predicting the waste rate of production batches, with means and corrected 95% intervals.

![Cross-validated model comparison](plot.png)

"Model B scores 0.47, model A 0.46" says nothing until we know how much the
score moves from one split to the next. Here every model is evaluated on the
**same** 10 × 5 repeated folds; each dot is one fold, the large marker is the
mean and the bar its 95% interval. Fold scores are not independent — the
training sets overlap — so the interval uses the **Nadeau–Bengio corrected**
variance rather than the naive standard error, which would be far too narrow.

The example is set in the context of Warin, d'Anglade and de Marcellis-Warin
(2025, *International Journal of Sustainable Engineering*), who combine k-means
clustering and data science to reduce waste in manufacturing. The data are
**simulated** for illustration and are *not* the paper's data or results:
n = 600 production batches, whose waste rate (% of material scrapped) depends
non-linearly on machine speed, temperature and operator experience, on
material grade, and on an interaction between humidity and days since
maintenance; batch size has no effect. The same simulated batches are used in
the next five charts. Scores come from `cross_val_score` with `RepeatedKFold`;
the linear models are standardised in a `Pipeline`. Because the effects are
curved and interact, the tree ensembles beat the linear models by a wide margin.

**R equivalent:** `caret::resamples() + bwplot() / tune::autoplot()`  
**Libraries:** Matplotlib, NumPy, pandas, SciPy, scikit-learn

**Data:** Simulated data with a fixed random seed, created for this example. — No real respondents, documents or measurements: the numbers only illustrate how the chart and its statistics work.

**Data:** Warin, T., d'Anglade, P. M., & de Marcellis-Warin, N. (2025). Optimising industrial efficiency: Integrating K-means clustering and data science for sustainable manufacturing and waste reduction. International Journal of Sustainable Engineering, 18(1), 2527300. https://doi.org/10.1080/19397038.2025.2527300 — The example is set in the context of this paper; unless its description says otherwise, the numbers are simulated and are not the paper's data or results.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
k, reps = 5, 10
cv = RepeatedKFold(n_splits=k, n_repeats=reps, random_state=0)
models = {
    "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
    "Lasso": make_pipeline(StandardScaler(), LassoCV(random_state=0)),
    "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0),
    "Gradient boosting": GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, max_depth=2,
                                                   random_state=0),
    "k-nearest neighbours": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
}
scores = {name: cross_val_score(m, X, y, cv=cv, scoring="r2", n_jobs=-1) for name, m in models.items()}

# Nadeau–Bengio corrected 95% interval for a mean of k-fold scores
def corrected_ci(s):
    J = len(s)
    se = np.sqrt((1 / J + (1 / k) / (1 - 1 / k)) * s.var(ddof=1))
    t = stats.t.ppf(0.975, J - 1)
    return s.mean() - t * se, s.mean() + t * se

order = sorted(scores, key=lambda n: scores[n].mean())
best = order[-1]
fig, ax = plt.subplots(figsize=(8, 5.2))
jit = np.random.default_rng(3)
for i, name in enumerate(order):
    s = scores[name]
    c = "#007AFF" if name == best else "#8E8E93"
    ax.scatter(s, i + jit.uniform(-0.18, 0.18, len(s)), s=12, color=c, alpha=0.35, lw=0)
    lo, hi = corrected_ci(s)
    ax.plot([lo, hi], [i, i], color=c, lw=3, solid_capstyle="round")
    ax.scatter(s.mean(), i, s=80, color=c, edgecolor="white", lw=1.5, zorder=4)
    ax.text(hi + 0.012, i, f"{s.mean():.3f}", va="center", fontsize=9.5, color=c, fontweight="semibold",
            bbox=dict(fc="white", ec="none", pad=1), zorder=5)
ax.set_yticks(range(len(order)), order)
ax.set_xlabel("Out-of-sample R² (one dot per fold)")
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
lin = scores["Linear regression"].mean()
ax.set_title(f"{best} explains {scores[best].mean():.0%} of waste variance; linear regression only {lin:.0%}",
             pad=26)
ax.text(0, 1.02, f"Waste rate of production batches · n = {n} (simulated) · {k}-fold CV × {reps} repeats · "
        "bars = Nadeau–Bengio corrected 95% CI", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/machine-learning/model-comparison/) · [← Machine learning](../README.md)

**Cite:** Warin, T. (2026). Cross-validated model comparison. In PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences. https://warint.github.io/pystatlas/machine-learning/model-comparison/
