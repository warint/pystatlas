"""
title: ACF and PACF
summary: ACF and PACF of a monthly sovereign bond spread over the Bund, with 95% significance bands.
r: forecast::ggtsdisplay()
tags: [autocorrelation, ACF, PACF, Ljung-Box, time series, sovereign bonds]
description: |
  The autocorrelation function (ACF) measures how strongly a series correlates
  with itself k steps earlier; the partial autocorrelation (PACF) does the same
  after removing the effect of the shorter lags in between. Stems that leave the
  grey band (±1.96/√n) are larger than white noise would produce. Together the two
  panels suggest a model: here the ACF decays slowly over many months — a wide
  spread this month means a wide spread for a while — while the PACF cuts off
  after **two lags**, the signature of an AR(2) process.

  The setting is Warin & Stojkov (2023, *Entropy*), on sovereign bond yield
  differentials across Europe. The data here are **simulated** for illustration
  and are not the paper's data or results: the 10-year yield spread of one
  euro-area sovereign over the German Bund, in percentage points, monthly from
  January 2003 to December 2022 (n = 240), generated as an AR(2) process.
  Correlations come from `statsmodels.tsa.stattools.acf` and `pacf`, and the
  subtitle reports a Ljung–Box test of no autocorrelation.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

# Simulated data, set in the context of Warin & Stojkov (2023) [warin2023entropy]

# Data: monthly 10-year spread over the Bund (percentage points), simulated AR(2)
rng = np.random.default_rng(2024)
n, nlags = 240, 30
e, s = rng.normal(0, 0.12, n + 100), np.zeros(n + 100)
for t in range(2, n + 100):
    s[t] = 1.3 * s[t - 1] - 0.4 * s[t - 2] + e[t]      # AR(2): persistent, mean-reverting
y = pd.Series(1.5 + s[100:], index=pd.date_range("2003-01-01", periods=n, freq="MS"))
band = 1.96 / np.sqrt(n)
lb = acorr_ljungbox(y, lags=[10]).iloc[0]

fig = plt.figure(figsize=(10, 6.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
ax_ts = fig.add_subplot(gs[0, :])
axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

# Top: the series itself
ax_ts.plot(y.index, y, color="#1D1D1F", lw=1.2)
ax_ts.axhline(y.mean(), color="#AEAEB2", lw=1, ls="--")
ax_ts.set(ylabel="Spread vs Bund (pp)", xlim=(y.index[0], y.index[-1]))

# Bottom: ACF and PACF as lollipops with the white-noise band
values = [acf(y, nlags=nlags)[1:], pacf(y, nlags=nlags, method="ywm")[1:]]
for ax, v, name in zip(axes, values, ["ACF", "PACF"]):
    lags = np.arange(1, nlags + 1)
    sig = np.abs(v) > band
    colors = np.where(sig, "#007AFF", "#AEAEB2")
    ax.axhspan(-band, band, color="#E5E5EA", alpha=0.7, lw=0)
    ax.axhline(0, color="#8E8E93", lw=0.8)
    ax.vlines(lags, 0, v, color=colors, lw=1.6)
    ax.scatter(lags, v, s=28, color=colors, edgecolor="white", lw=0.8, zorder=3)
    ax.set(xlabel="Lag (months)", ylabel=name, ylim=(-1, 1), xlim=(0, nlags + 1))
    ax.set_xticks(range(0, nlags + 1, 5))
    ax.set_yticks([-1, -0.5, 0, 0.5, 1], ["−1", "−0.5", "0", "0.5", "1"])
last = np.flatnonzero(np.abs(values[1]) > band)[-1] + 1     # last significant PACF lag
for lag in (1, last):
    axes[1].annotate(f"{values[1][lag - 1]:+.2f}", (lag, values[1][lag - 1]),
                     xytext=(10, 0), textcoords="offset points", va="center",
                     fontsize=9, color="#007AFF", fontweight="semibold")

ax_ts.set_title(f"The bond spread is persistent: the PACF cuts off after lag {last}", pad=26)
p_txt = "p < .001" if lb.lb_pvalue < 0.001 else f"p = {lb.lb_pvalue:.3f}"
ax_ts.text(0, 1.04, f"Monthly, 2003–2022, n = {n} (simulated) · Ljung–Box Q(10) = {lb.lb_stat:.0f}, "
           f"{p_txt} · grey band = ±1.96/√n", transform=ax_ts.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
