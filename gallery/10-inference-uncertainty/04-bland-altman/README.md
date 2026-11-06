# Bland–Altman plot

> Agreement between two measurement methods, with bias and 95% limits of agreement.

![Bland–Altman plot](plot.png)

When two instruments measure the same thing, a high correlation says nothing
about whether they **agree**. The Bland–Altman plot puts the difference
between methods on the y-axis and their average on the x-axis. The middle
line is the mean difference (systematic bias), the outer lines the 95%
**limits of agreement** (bias ± 1.96 SD) within which most disagreements
fall. Shaded bands are the 95% confidence intervals of these three lines,
using the Bland & Altman (1986) standard errors; a slope in the cloud would
signal a proportional bias.

Data are simulated with `numpy` (seeded): systolic blood pressure of 80
patients read by a reference cuff and a new wrist device. Bias and
intervals use `scipy.stats.t`; the slope test is an OLS fit from **statsmodels**.

**R equivalent:** `BlandAltmanLeh::bland.altman.plot() / blandr::blandr.draw()`  
**Libraries:** Matplotlib, NumPy, statsmodels, SciPy

```python
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Simulate paired readings: the device reads ~3 mmHg high with extra noise
rng = np.random.default_rng(11)
n = 80
true_bp = rng.normal(130, 18, n)
cuff = true_bp + rng.normal(0, 4, n)
device = true_bp + 3 + rng.normal(0, 5, n)

mean, diff = (cuff + device) / 2, device - cuff
bias, sd = diff.mean(), diff.std(ddof=1)
loa = bias + np.array([-1.96, 1.96]) * sd
t = stats.t.ppf(0.975, n - 1)
ci_bias = t * sd / np.sqrt(n)            # half-width of the CI for the bias
ci_loa = t * np.sqrt(3 * sd**2 / n)      # Bland & Altman (1986) SE of a limit
slope = sm.OLS(diff, sm.add_constant(mean)).fit()

fig, ax = plt.subplots(figsize=(8, 5))
for y0, hw, c in [(bias, ci_bias, "#007AFF"), (loa[0], ci_loa, "#FF9500"), (loa[1], ci_loa, "#FF9500")]:
    ax.axhspan(y0 - hw, y0 + hw, color=c, alpha=0.12, lw=0)
    ax.axhline(y0, color=c, lw=1.6, ls="-" if c == "#007AFF" else "--")
ax.axhline(0, color="#8E8E93", lw=0.8)
ax.scatter(mean, diff, s=30, color="#3A3A3C", alpha=0.7, edgecolor="white", lw=0.6, zorder=3)

# Labels on the right margin
x_lab = mean.max() + 2
for y0, name, c in [(loa[1], "+1.96 SD", "#FF9500"), (bias, "Bias", "#007AFF"),
                    (loa[0], "−1.96 SD", "#FF9500")]:
    ax.text(x_lab, y0, f"{name}\n{y0:+.1f} mmHg".replace("-", "−"), va="center", ha="left", fontsize=9,
            color=c, fontweight="semibold", linespacing=1.3)

ax.set_xlim(mean.min() - 3, mean.max() + 14)
ax.set_xlabel("Mean of the two methods (mmHg)")
ax.set_ylabel("Device − cuff (mmHg)")
ax.set_title(f"The wrist device reads {bias:.1f} mmHg high, but individual readings "
             f"differ by up to ±{1.96 * sd:.0f}", pad=26)
ax.text(0, 1.02, f"n = {n} · bias {bias:.1f} [95% CI {bias - ci_bias:.1f}, {bias + ci_bias:.1f}] · "
        f"LoA {loa[0]:.1f} to {loa[1]:.1f} · proportional bias: slope = {slope.params[1]:.3f}, "
        f"p = {slope.pvalues[1]:.2f}", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/inference-uncertainty/bland-altman/) · [← Inference & uncertainty](../README.md)

**Cite:** Warin, T. (2026). Bland–Altman plot. In PyStatlas: The Python Gallery of Statistical Graphics. https://warint.github.io/pystatlas/inference-uncertainty/bland-altman/
