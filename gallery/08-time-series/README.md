# Time series

Decomposition, autocorrelation, forecasts with prediction intervals and change over time.

| Chart | Description |
|:--|:--|
| [<img src="01-stl-decomposition/plot.png" width="260">](01-stl-decomposition/) | **[STL decomposition](01-stl-decomposition/)**<br>Monthly manufacturing sales split into trend, seasonal and remainder components.<br>`feasts::STL() + autoplot()` |
| [<img src="02-acf-pacf/plot.png" width="260">](02-acf-pacf/) | **[ACF and PACF](02-acf-pacf/)**<br>ACF and PACF of a monthly sovereign bond spread over the Bund, with 95% significance bands.<br>`forecast::ggtsdisplay()` |
| [<img src="03-arima-forecast/plot.png" width="260">](03-arima-forecast/) | **[SARIMA forecast](03-arima-forecast/)**<br>Seasonal ARIMA forecast of monthly manufacturing sales, with 80% and 95% intervals vs held-out data.<br>`forecast::autoplot(forecast())` |
| [<img src="04-seasonal-subseries/plot.png" width="260">](04-seasonal-subseries/) | **[Seasonal subseries plot](04-seasonal-subseries/)**<br>Each calendar month's manufacturing sales across the years, with the monthly mean.<br>`feasts::gg_subseries()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/time-series/)
