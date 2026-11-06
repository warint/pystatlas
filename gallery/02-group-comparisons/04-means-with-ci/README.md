# Interaction plot

> Group means with 95% confidence intervals for two factors, with a two-way ANOVA.

![Interaction plot](plot.png)

An interaction plot shows the mean outcome for every combination of two
factors: one factor on the x-axis, the other as coloured, slightly dodged
lines. Each mean carries its 95% confidence interval. Parallel lines mean
the two factors act independently; lines that converge, diverge or cross
signal an interaction.

Data are the 442 patients of the diabetes study bundled with
**scikit-learn** (`load_diabetes(scaled=False)`): disease progression after
one year, by BMI class and sex. The two-way ANOVA (type II sums of squares)
comes from `statsmodels.formula.api.ols` and `statsmodels.stats.anova.anova_lm`,
with partial η² for each term; the intervals use the t distribution.

**R equivalent:** `ggpubr::ggline(add = "mean_ci") / ggstats`  
**Libraries:** Matplotlib, NumPy, pandas, statsmodels, SciPy, scikit-learn

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.datasets import load_diabetes
from statsmodels.stats.anova import anova_lm

# Data: disease progression by BMI class and sex
df = load_diabetes(as_frame=True, scaled=False).frame
classes = ["Normal", "Overweight", "Obese"]
df["bmi_class"] = pd.cut(df.bmi, [0, 25, 30, np.inf], right=False, labels=classes)
df["sex"] = df.sex.map({1: "Sex 1", 2: "Sex 2"})

# Two-way ANOVA with interaction (type II) + partial eta squared
aov = anova_lm(smf.ols("target ~ C(bmi_class) * C(sex)", data=df).fit(), typ=2)
ss_res = aov.loc["Residual", "sum_sq"]
aov["eta_p"] = aov.sum_sq / (aov.sum_sq + ss_res)
df_res = int(aov.loc["Residual", "df"])

# Means and t-based 95% CIs per cell
cells = df.groupby(["sex", "bmi_class"], observed=True).target.agg(["mean", "sem", "count"])
cells["half"] = cells["sem"] * stats.t.ppf(0.975, cells["count"] - 1)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(classes))
for (sex, c, dx) in [("Sex 1", "#007AFF", -0.06), ("Sex 2", "#FF9500", 0.06)]:
    m = cells.loc[sex].reindex(classes)
    ax.errorbar(x + dx, m["mean"], yerr=m["half"], color=c, lw=2, capsize=0,
                elinewidth=2, marker="o", ms=9, mec="white", mew=1.8, label=sex)
    ax.annotate(sex, (x[-1] + dx, m["mean"].iloc[-1]), xytext=(14, 0),
                textcoords="offset points", va="center", fontsize=10.5,
                fontweight="semibold", color=c)

# Cell sizes under the axis labels
n = cells["count"].unstack(0).reindex(classes)
ax.set_xticks(x, [f"{k}\n(n = {n.loc[k, 'Sex 1']} / {n.loc[k, 'Sex 2']})" for k in classes])
ax.set_xlim(-0.3, len(classes) - 0.55)
ax.set_xlabel("BMI class (kg/m²: < 25, 25–30, ≥ 30)")
ax.set_ylabel("Disease progression after 1 year\n(mean ± 95% CI)")

# One "F(df1, df2) = …, p …, η²p = …" string per ANOVA term
stat = {}
for term, label in [("C(bmi_class)", "BMI"), ("C(sex)", "sex"), ("C(bmi_class):C(sex)", "BMI × sex")]:
    r = aov.loc[term]
    pval = "< .001" if r["PR(>F)"] < .001 else f"= {r['PR(>F)']:.2f}"
    stat[label] = f"{label}: F({r.df:.0f}, {df_res}) = {r.F:.1f}, p {pval}, η²p = {r.eta_p:.2f}"

ax.set_title("Progression rises with BMI, and the obese gap is wider for sex 2", pad=40)
ax.text(0, 1.075, f"Two-way ANOVA (type II) · {stat['BMI']} · {stat['sex']}",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
ax.text(0, 1.02, stat["BMI × sex"], transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/group-comparisons/means-with-ci/) · [← Group comparisons](../README.md)

**Cite:** Warin, T. (2026). Interaction plot. In PyStatlas: The Python Gallery of Statistical Graphics. https://warint.github.io/pystatlas/group-comparisons/means-with-ci/
