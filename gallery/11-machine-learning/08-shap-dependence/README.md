# SHAP dependence plot

> How days since maintenance feed into each batch's predicted waste rate, coloured by its strongest interaction partner, with a smooth trend.

![SHAP dependence plot](plot.png)

A SHAP dependence plot zooms in on one feature. Each dot is a production
batch: the horizontal axis is the feature's value, the vertical axis its SHAP
value — its contribution to that batch's predicted waste. The trend line (a
LOWESS smooth) is close to a partial-dependence curve, but the cloud around it
is informative: at the same number of days since maintenance, contributions
differ from batch to batch because of **interactions** with other features.

Colour shows the feature that best explains that vertical spread, chosen here
as the feature whose SHAP **interaction** values with days since maintenance
are largest on average (`shap.TreeExplainer.shap_interaction_values`). The
example is set in the context of Warin, d'Anglade and de Marcellis-Warin
(2025, *International Journal of Sustainable Engineering*) on data science for
waste reduction in manufacturing. The data are **simulated** for illustration
and are *not* the paper's data or results: the same n = 600 production
batches as in the model comparison, built so that a worn machine wastes more
when the workshop is humid. The model is a gradient-boosting regressor; the
smooth comes from `statsmodels.nonparametric.lowess`.

**R equivalent:** `shapviz::sv_dependence()`  
**Libraries:** Matplotlib, NumPy, pandas, SHAP, scikit-learn, statsmodels

**Data:** Simulated data with a fixed random seed, created for this example. — No real respondents, documents or measurements: the numbers only illustrate how the chart and its statistics work.

**Data:** Warin, T., d'Anglade, P. M., & de Marcellis-Warin, N. (2025). Optimising industrial efficiency: Integrating K-means clustering and data science for sustainable manufacturing and waste reduction. International Journal of Sustainable Engineering, 18(1), 2527300. https://doi.org/10.1080/19397038.2025.2527300 — The example is set in the context of this paper; unless its description says otherwise, the numbers are simulated and are not the paper's data or results.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
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

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X)
inter = explainer.shap_interaction_values(X)            # (batches, features, features)

f = list(X.columns).index("maintenance")
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
ax.set_xlabel(names["maintenance"])
ax.set_ylabel("SHAP value of days since maintenance\n(percentage points of waste)")
ax.grid(axis="x")
ax.legend(loc="upper left")
cb = fig.colorbar(sc, ax=ax, pad=0.02, aspect=40, ticks=[0, 1])
cb.ax.set_yticklabels(["low", "high"])
cb.set_label(f"{names[X.columns[partner]]} (percentile)", fontsize=9.5)
cb.outline.set_visible(False)

old = x > 90
wet, dry = s[old & (col > 2 / 3)].mean(), s[old & (col < 1 / 3)].mean()
ax.set_title(f"Beyond 90 days without maintenance: {wet:+.1f} points of waste when humid, "
             f"{dry:+.1f} when dry", pad=26)
ax.text(0, 1.02, f"Gradient boosting · n = {n} batches (simulated) · colour = strongest interaction "
        f"partner: {names[X.columns[partner]].split(' (')[0].lower()}", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/machine-learning/shap-dependence/) · [← Machine learning](../README.md)

**Cite:** Warin, T. (2026). SHAP dependence plot. In PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences (Version 1.1.0) [Computer software]. https://warint.github.io/pystatlas/machine-learning/shap-dependence/
