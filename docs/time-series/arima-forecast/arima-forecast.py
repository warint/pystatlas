"""
title: SARIMA forecast
summary: Seasonal ARIMA forecast of monthly manufacturing sales, with 80% and 95% intervals vs held-out data.
r: forecast::autoplot(forecast())
tags: [ARIMA, SARIMA, forecast, prediction interval, RMSE, Statistics Canada]
description: |
  A forecast is only honest with its uncertainty. The dark fan is the 80%
  prediction interval, the light fan the 95% interval; both widen as the horizon
  grows. The last two years were **held out** from the fit, so the dots show what
  actually happened and the subtitle reports the out-of-sample RMSE and how many
  of those months fell inside the 95% interval.

  The setting is Warin (2024, *Technology Innovations in Statistics Education*),
  on using Statistics Canada's open economic data in statistics courses, where
  monthly manufacturing sales are a typical teaching series. The data here are
  **simulated** for illustration and are not Statistics Canada's figures: a
  manufacturing-sales-like series in billions of dollars, January 2005 to
  December 2023 (228 months), built from a trend with two slumps (2009 and
  spring 2020), a fixed monthly pattern and noise. The same simulated series is
  used in the STL, SARIMA and seasonal subseries examples. The model is fitted on
  2005–2021 and forecasts the 24 months of 2022–2023. It is a seasonal
  ARIMA(1,1,1)(0,1,1)₁₂ — the classic "airline" structure — fitted with
  `statsmodels.tsa.statespace.SARIMAX`; intervals come from
  `get_forecast().conf_int()`.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Simulated data, set in the context of Warin (2024) [warin2024statcan]

# Data: simulated monthly manufacturing sales, billions of dollars, 2005–2023 (shared generator)
rng = np.random.default_rng(2024)
idx = pd.date_range("2005-01-01", "2023-12-01", freq="MS")
t = np.arange(len(idx))
trend = 48 + 0.06 * t - 7 * np.exp(-((t - 54) / 9) ** 2) - 8 * np.exp(-((t - 183) / 2.5) ** 2)
season = np.tile([-3.0, -0.8, 2.6, 0.6, 1.2, 2.2, -3.4, -1.2, 1.6, 2.0, 0.4, -2.2], len(idx) // 12)
sales = pd.Series(trend + season + rng.normal(0, 0.7, len(idx)), index=idx)
h = 24                                                  # hold out the last two years
train, test = sales[:-h], sales[-h:]

# Fit a seasonal ARIMA and forecast the held-out horizon
fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
fc = fit.get_forecast(h)
mean, ci95, ci80 = fc.predicted_mean, fc.conf_int(alpha=0.05), fc.conf_int(alpha=0.20)

rmse = np.sqrt(((test - mean) ** 2).mean())
bias = (test - mean).mean()                       # > 0: actuals above the forecast
covered = ((test >= ci95.iloc[:, 0]) & (test <= ci95.iloc[:, 1])).sum()

fig, ax = plt.subplots(figsize=(9, 5))
hist = train["2014":]
ax.plot(hist.index, hist, color="#1D1D1F", lw=1.4, label="Observed (training)")
ax.fill_between(mean.index, ci95.iloc[:, 0], ci95.iloc[:, 1], color="#007AFF", alpha=0.15, lw=0,
                label="95% prediction interval")
ax.fill_between(mean.index, ci80.iloc[:, 0], ci80.iloc[:, 1], color="#007AFF", alpha=0.28, lw=0,
                label="80% prediction interval")
ax.plot(mean.index, mean, color="#007AFF", lw=2, label="Forecast")
ax.scatter(test.index, test, s=16, color="#FF9500", edgecolor="white", lw=0.6, zorder=4,
           label="Held-out actuals")

# Mark the forecast origin
ax.axvline(test.index[0], color="#AEAEB2", lw=1, ls="--")
ax.text(test.index[0], 0.97, "  forecast origin", transform=ax.get_xaxis_transform(),
        fontsize=9, color="#8E8E93", va="top")

ax.set(ylabel="Manufacturing sales (billions of dollars)", xlim=(hist.index[0], test.index[-1]))
ax.legend(loc="upper left", ncols=3)
ax.set_ylim(top=mean.max() + 12)
ax.set_title(f"A seasonal ARIMA tracks sales two years ahead, off by {abs(bias):.1f} bn on average", pad=26)
ax.text(0, 1.02, f"SARIMA(1,1,1)(0,1,1)₁₂ · AIC = {fit.aic:.0f} · RMSE = {rmse:.2f} bn over {h} held-out "
        f"months (simulated) · {covered}/{h} actuals inside the 95% PI",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
