# Interaction effects

> Simple slopes at mean ± 1 SD and a Johnson–Neyman plot for two continuous predictors.

![Interaction effects](plot.png)

An interaction term says that the slope of one predictor depends on another.
The **left panel** draws that slope at three values of the moderator — its mean
and ±1 standard deviation — each with a 95% confidence band. The **right panel**
is the Johnson–Neyman view: the marginal effect of the focal predictor,
β₁ + β₃·moderator, across the whole range of the moderator, with its 95% CI.
Where the band excludes zero the effect is significant; the switch-over is the
Johnson–Neyman point.

Data are the 442 diabetes patients bundled with **scikit-learn**
(`load_diabetes`): disease progression after one year, modelled with
`smf.ols("target ~ age * bmi")`. The marginal-effect variance is computed by hand
from `cov_params()`: Var(β₁) + 2m·Cov(β₁, β₃) + m²·Var(β₃).

**R equivalent:** `interactions::interact_plot() / interactions::johnson_neyman()`  
**Libraries:** Matplotlib, NumPy, pandas, statsmodels, SciPy, scikit-learn

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.datasets import load_diabetes

df = load_diabetes(as_frame=True, scaled=False).frame
model = smf.ols("target ~ age * bmi", data=df).fit()
b, V = model.params, model.cov_params()
t_crit = stats.t.ppf(0.975, model.df_resid)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Left: simple slopes of age at BMI = mean − 1 SD, mean, mean + 1 SD
ax1.scatter(df.age, df.target, s=12, color="#C7C7CC", alpha=0.6, lw=0)
age = np.linspace(df.age.min(), df.age.max(), 100)
m, s = df.bmi.mean(), df.bmi.std()
for bmi, lab, c in [(m - s, "−1 SD", "#5AC8FA"), (m, "Mean", "#007AFF"), (m + s, "+1 SD", "#AF52DE")]:
    pred = model.get_prediction(pd.DataFrame({"age": age, "bmi": bmi})).summary_frame()
    ax1.fill_between(age, pred.mean_ci_lower, pred.mean_ci_upper, color=c, alpha=0.15, lw=0)
    ax1.plot(age, pred["mean"], color=c, lw=2.2,
             label=f"{lab} ({bmi:.1f}): slope {b['age'] + b['age:bmi'] * bmi:+.2f}")
ax1.legend(title="BMI (kg/m²)", loc="upper left", alignment="left")
ax1.set_ylim(top=420)
ax1.set_xlabel("Age (years)")
ax1.set_ylabel("Disease progression after one year")
ax1.set_title("Simple slopes of age at three BMI levels", fontsize=12)

# Right: Johnson–Neyman — marginal effect of age across BMI, with 95% CI
bmi = np.linspace(df.bmi.min(), df.bmi.max(), 400)
eff = b["age"] + b["age:bmi"] * bmi
se = np.sqrt(V.loc["age", "age"] + 2 * bmi * V.loc["age", "age:bmi"]
             + bmi**2 * V.loc["age:bmi", "age:bmi"])
lo, hi = eff - t_crit * se, eff + t_crit * se
sig = (lo > 0) | (hi < 0)
ax2.axhline(0, color="#8E8E93", lw=1, ls="--")
ax2.fill_between(bmi, lo, hi, where=sig, color="#007AFF", alpha=0.18, lw=0, label="p < .05")
ax2.fill_between(bmi, lo, hi, where=~sig, color="#AEAEB2", alpha=0.3, lw=0, label="n.s.")
ax2.plot(bmi, eff, color="#1D1D1F", lw=2)
for jn in bmi[1:][np.diff(sig.astype(int)) != 0]:   # where significance switches
    ax2.axvline(jn, color="#007AFF", lw=1, ls=":")
    ax2.annotate(f"Johnson–Neyman point\nBMI = {jn:.1f}", (jn, 0.97), xytext=(8, 0),
                 xycoords=ax2.get_xaxis_transform(), textcoords="offset points",
                 fontsize=9.5, color="#007AFF", va="top")
# Rug of the observed moderator values
ax2.plot(df.bmi, np.full(len(df), 0.015), "|", ms=7, color="#8E8E93", alpha=0.4,
         transform=ax2.get_xaxis_transform())
share = (df.bmi > jn).mean()
ax2.legend(loc="lower right", bbox_to_anchor=(1, 0.06))
ax2.set_xlabel("Body mass index (moderator, kg/m²)")
ax2.set_ylabel("Marginal effect of one year of age")
ax2.set_title(f"Age matters only above BMI ≈ {jn:.0f} ({share:.0%} of patients)", fontsize=12)

# Figure title and statistical subtitle above both panels
fig.text(0.01, 1.07, "Ageing speeds diabetes progression, but only in heavier patients",
         fontsize=15, fontweight="semibold", va="bottom")
fig.text(0.01, 1.02, f"OLS · n = {int(model.nobs)} · age × BMI: β = {b['age:bmi']:.3f}, "
         f"t({model.df_resid:.0f}) = {model.tvalues['age:bmi']:.2f}, "
         f"p = {model.pvalues['age:bmi']:.3f} · R² = {model.rsquared:.2f} · bands = 95% CI",
         fontsize=9.5, color="#6E6E73", va="bottom")
plt.show()
```

---

**Statlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/statlas/regression-models/interaction-effects/) · [← Regression models](../README.md)

**Cite:** Warin, T. (2026). Interaction effects. In Statlas: The Python Gallery of Statistical Graphics. https://warint.github.io/statlas/regression-models/interaction-effects/
