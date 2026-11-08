"""
title: Scatterplot with statistics
summary: Trade intensity vs business-cycle correlation for country pairs — scatter, OLS line with 95% band, marginal histograms and Pearson's r.
r: ggstatsplot::ggscatterstats()
tags: [correlation, Pearson, scatterplot, regression line, marginal histogram, optimum currency area]
description: |
  The Python counterpart of `ggscatterstats()`. The central panel shows every
  observation with the least-squares line and its 95% confidence band (uncertainty
  about the mean of *y* at each *x*, not about individual points). Marginal
  histograms on the top and right reveal the shape of each variable, which a
  correlation coefficient silently assumes is reasonable. The subtitle reports
  Pearson's *r* with its 95% confidence interval, the t-test, the OLS slope and
  the sample size.

  The setting is Janicki, Warin & Wunnava (2009, *Review of International
  Economics*), which tests Mundell's intuition that trade integration makes
  business cycles move together. The data are **simulated** for illustration and
  are *not* the paper's data or results: 190 country pairs (all pairs of 20
  countries), with the log of bilateral trade intensity and the correlation of the
  two countries' detrended output. The fitted line describes an association between
  pairs, not an effect. The test comes from `scipy.stats.pearsonr`, the line and
  band from **statsmodels** OLS `get_prediction().conf_int()`.
"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Simulated data, set in the context of Janicki, Warin & Wunnava (2009) [janicki2009]

# Simulated country pairs: 20 countries -> 190 pairs; cycles co-move more where trade is denser
rng = np.random.default_rng(2009)
x = rng.normal(-5.5, 1.1, 190)                        # log(bilateral trade / combined GDP)
y = np.tanh(0.35 + 0.13 * (x + 5.5) + rng.normal(0, 0.28, 190))   # business-cycle correlation
n = len(x)

# Pearson correlation with its 95% CI
res = stats.pearsonr(x, y)
ci = res.confidence_interval(0.95)

# OLS fit and 95% confidence band for the mean
fit = sm.OLS(y, sm.add_constant(x)).fit()
grid = np.linspace(x.min(), x.max(), 200)
band = fit.get_prediction(sm.add_constant(grid)).conf_int(alpha=0.05)

fig = plt.figure(figsize=(8, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5), wspace=0.03, hspace=0.03)
ax = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

ax.scatter(x, y, s=22, color="#007AFF", alpha=0.45, lw=0)
ax.fill_between(grid, band[:, 0], band[:, 1], color="#FF9500", alpha=0.25, lw=0)
ax.plot(grid, fit.params[0] + fit.params[1] * grid, color="#FF9500", lw=2.2)
ax.set_xlabel("Bilateral trade intensity (log of trade / combined GDP)")
ax.set_ylabel("Business-cycle correlation")
ax.grid(axis="x")

# Marginal histograms
ax_top.hist(x, bins=30, color="#007AFF", alpha=0.35, edgecolor="white")
ax_right.hist(y, bins=30, color="#007AFF", alpha=0.35, edgecolor="white", orientation="horizontal")
for a in (ax_top, ax_right):
    a.axis("off")

ax_top.set_title("Pairs that trade more have more synchronised business cycles", pad=26)
p_txt = "p < .001" if res.pvalue < 0.001 else f"p = {res.pvalue:.3f}"
ax_top.text(0, 1.06, f"Pearson r = {res.statistic:.2f}, 95% CI [{ci.low:.2f}, {ci.high:.2f}], "
            f"t({n - 2}) = {fit.tvalues[1]:.1f}, {p_txt} · slope = {fit.params[1]:.2f} · n = {n} pairs (simulated)",
            transform=ax_top.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
