# Time series

Decomposition, autocorrelation, forecasts with prediction intervals and change over time.

| Chart | Description |
|:--|:--|
| [<img src="01-stl-decomposition/plot.png" width="260">](01-stl-decomposition/) | **[STL decomposition](01-stl-decomposition/)**<br>A monthly industrial turnover series split into trend, seasonal and remainder components in stacked panels.<br>`feasts::STL() + autoplot()` |
| [<img src="02-acf-pacf/plot.png" width="260">](02-acf-pacf/) | **[ACF and PACF](02-acf-pacf/)**<br>Autocorrelation and partial autocorrelation of US quarterly unemployment, as lollipops with 95% significance bands.<br>`forecast::ggtsdisplay()` |
| [<img src="03-arima-forecast/plot.png" width="260">](03-arima-forecast/) | **[SARIMA forecast](03-arima-forecast/)**<br>Seasonal ARIMA forecast of industrial turnover with 80% and 95% prediction intervals against held-out data.<br>`forecast::autoplot(forecast())` |
| [<img src="04-seasonal-subseries/plot.png" width="260">](04-seasonal-subseries/) | **[Seasonal subseries plot](04-seasonal-subseries/)**<br>Each calendar month's industrial turnover across the years, with the monthly mean as a reference.<br>`feasts::gg_subseries()` |

[← All families](../../README.md) · [Browse on the website](https://warint.github.io/pystatlas/time-series/)
