# Ridgeline plot: the R equivalent of plot.py
# Simulated data, set in the context of Stojkov, Warin & Kjosevski (2026) [stojkov2026ca]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2026)

# Data: current-account balance (% of GDP), one column per country, one row per year
means <- c(Norway = 8, Switzerland = 7, Netherlands = 5.5, Germany = 4, Korea = 3, Japan = 2.5,
           France = -0.5, Canada = -1.5, "United Kingdom" = -3, "United States" = -3,
           Australia = -3.5, Greece = -6)
years <- 1995:2023
ca <- sapply(means, function(mu) {              # AR(1) swings around the country mean
  shocks <- c(0, rnorm(length(years) - 1, 0, 1.1 + 0.08 * abs(mu)))
  mu + as.numeric(stats::filter(shocks, 0.6, method = "recursive"))
})
rownames(ca) <- years

# Friedman test (years are blocks) + Kendall's W
ft <- friedman.test(ca)
chi2 <- unname(ft$statistic)
W <- chi2 / (nrow(ca) * (ncol(ca) - 1))

medians <- sort(apply(ca, 2, median), decreasing = TRUE)   # largest surplus on top
spacing <- 1.0                                  # vertical gap between ridges
height <- 1.9 * 0.6                             # peak height of a ridge
base <- setNames((length(medians) - seq_along(medians)) * spacing, names(medians))

# One density per country, scaled to the same peak height and lifted to its baseline
ridges <- do.call(rbind, lapply(names(medians), function(country) {
  k <- density(ca[, country], from = -13, to = 21, n = 400)
  data.frame(country = country, median = medians[[country]], x = k$x, base = base[[country]],
             y = base[[country]] + height * k$y / max(k$y))
}))
ticks <- do.call(rbind, lapply(split(ridges, ridges$country), function(r)
  data.frame(country = r$country[1], x = r$median[1], base = r$base[1],
             y = approx(r$x, r$y, r$median[1])$y)))        # top of the dark tick at the median

# Drawn from the top ridge down, so that each ridge overlaps the one above it
ridge_layers <- lapply(names(medians), function(country) list(
  geom_hline(yintercept = base[[country]], colour = "#C7C7CC", linewidth = 0.3),
  geom_ribbon(data = ridges[ridges$country == country, ], aes(x, ymin = base, ymax = y, fill = median),
              alpha = 0.9, colour = "white", linewidth = 0.6, outline.type = "upper"),
  geom_segment(data = ticks[ticks$country == country, ], aes(x = x, xend = x, y = base, yend = y),
               colour = "#1D1D1F", linewidth = 0.8)))

p <- ggplot() +
  geom_vline(xintercept = 0, colour = "#8E8E93", linewidth = 0.4, linetype = "dashed") +
  ridge_layers +
  scale_fill_gradient2(low = "#D8434E", mid = "#F2F2F2", high = "#3B7EC0", midpoint = 0,
                       limits = c(-10, 10), oob = scales::squish, guide = "none") +   # red = deficit, blue = surplus
  scale_x_continuous(limits = c(-13, 21), expand = c(0, 0)) +
  scale_y_continuous(breaks = base, labels = names(base), expand = c(0, 0),
                     limits = c(-0.2, max(base) + height + 0.1)) +
  labs(title = "One ridge per country, sorted by its median current account (simulated)",
       subtitle = sprintf(paste("n = %d country-years (simulated) · Friedman χ²(%d) = %.0f, p = %.1e,",
                                "Kendall's W = %.2f · tick = median"),
                          length(ca), ncol(ca) - 1, chi2, ft$p.value, W),
       x = sprintf("Current-account balance (%% of GDP), one value per year, %d–%d",
                   min(years), max(years)), y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.y = element_text(colour = "#1D1D1F", vjust = 0),
        panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        plot.margin = margin(8, 14, 8, 8))
print(p)
