"""
title: Bland–Altman plot
summary: Do AI-assisted and expert scores of IPCC scenario passages agree? Bias and 95% limits of agreement.
r: BlandAltmanLeh::bland.altman.plot() / blandr::blandr.draw()
tags: [agreement, Bland-Altman, limits of agreement, method comparison, bias, AI-assisted coding]
description: |
  When two instruments — or two coders — measure the same thing, a high
  correlation says nothing about whether they **agree**. The Bland–Altman plot puts the difference
  between methods on the y-axis and their average on the x-axis. The middle
  line is the mean difference (systematic bias), the outer lines the 95%
  **limits of agreement** (bias ± 1.96 SD) within which most disagreements
  fall. Shaded bands are the 95% confidence intervals of these three lines,
  using the Bland & Altman (1986) standard errors; a slope in the cloud would
  signal a proportional bias.

  The setting is Bisson & Warin (2026), "AI-assisted longitudinal comparison of
  scenario knowledge representation in IPCC synthesis reports" (*PLOS Climate*):
  90 scenario passages from IPCC synthesis reports, each scored on a 0–10 scale
  by a human expert and by an AI-assisted pipeline. The scores are **simulated**
  with `numpy` (seeded), with the AI built to score about 0.3 points higher and
  a little noisier; they are *not* the paper's data or results. Bias and
  intervals use `scipy.stats.t`; the slope test is an OLS fit from **statsmodels**.
"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Simulated data, set in the context of Bisson & Warin (2026) [bisson2026]
# 90 IPCC scenario passages scored 0–10 by a human expert and an AI-assisted pipeline
rng = np.random.default_rng(65)
n = 90
detail = rng.uniform(1.5, 8.5, n)
expert = detail + rng.normal(0, 0.35, n)
ai = detail + 0.3 + rng.normal(0, 0.5, n)

mean, diff = (expert + ai) / 2, ai - expert
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
x_lab = mean.max() + 0.3
for y0, name, c in [(loa[1], "+1.96 SD", "#FF9500"), (bias, "Bias", "#007AFF"),
                    (loa[0], "−1.96 SD", "#FF9500")]:
    ax.text(x_lab, y0, f"{name}\n{y0:+.2f} pts".replace("-", "−"), va="center", ha="left", fontsize=9,
            color=c, fontweight="semibold", linespacing=1.3)

ax.set_xlim(mean.min() - 0.2, mean.max() + 1.3)
ax.set_xlabel("Mean of AI-assisted and expert scores (0–10 scale)")
ax.set_ylabel("AI-assisted − expert score (points)")
ax.set_title(f"AI-assisted scores run {bias:.2f} points higher; single passages differ "
             f"by up to ±{1.96 * sd:.1f}", pad=26)
ax.text(0, 1.02, f"n = {n} passages (simulated) · bias {bias:.2f} [95% CI {bias - ci_bias:.2f}, "
        f"{bias + ci_bias:.2f}] · LoA {loa[0]:.2f} to {loa[1]:.2f} · slope test: b = {slope.params[1]:.3f}, "
        f"p = {slope.pvalues[1]:.2f}", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
