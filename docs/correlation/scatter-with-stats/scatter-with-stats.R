# Scatterplot with statistics: the R equivalent of plot.py
# Simulated data, set in the context of Janicki, Warin & Wunnava (2009) [janicki2009]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2009)

# Simulated country pairs: 20 countries -> 190 pairs; cycles co-move more where trade is denser
x <- rnorm(190, -5.5, 1.1)                      # log(bilateral trade / combined GDP)
y <- tanh(0.35 + 0.13 * (x + 5.5) + rnorm(190, 0, 0.28))    # business-cycle correlation
n <- length(x)
d <- data.frame(x, y)

# Pearson correlation with its 95% CI
ct <- cor.test(x, y)

# OLS fit and 95% confidence band for the mean
fit <- lm(y ~ x, data = d)
grid <- data.frame(x = seq(min(x), max(x), length.out = 200))
grid <- cbind(grid, predict(fit, grid, interval = "confidence"))

# Marginal histograms, drawn in strips above (x) and to the right (y) of the scatter
xr <- range(x); yr <- range(y)
top <- yr[2] + 0.06 * diff(yr); top_h <- 0.22 * diff(yr)
right <- xr[2] + 0.04 * diff(xr); right_w <- 0.2 * diff(xr)
bins <- function(v, edges) {
  h <- hist(v, breaks = edges, plot = FALSE)
  data.frame(lo = head(h$breaks, -1), hi = tail(h$breaks, -1), share = h$counts / max(h$counts))
}
hx <- bins(x, seq(xr[1], xr[2], length.out = 31))
hy <- bins(y, seq(yr[1], yr[2], length.out = 31))
xbreaks <- pretty(xr)
xbreaks <- xbreaks[xbreaks >= xr[1] & xbreaks <= xr[2]]
ybreaks <- pretty(yr)
ybreaks <- ybreaks[ybreaks >= yr[1] & ybreaks <= yr[2]]

p_txt <- if (ct$p.value < .001) "p < .001" else sprintf("p = %.3f", ct$p.value)
p <- ggplot(d, aes(x, y)) +
  annotate("segment", x = xbreaks, xend = xbreaks, y = yr[1], yend = yr[2], colour = "#E5E5EA",
           linewidth = 0.4) +                   # vertical grid inside the scatter only
  geom_point(size = 1.8, colour = "#007AFF", alpha = 0.45, stroke = 0) +
  geom_ribbon(data = grid, aes(x, ymin = lwr, ymax = upr), inherit.aes = FALSE, fill = "#FF9500",
              alpha = 0.25) +
  geom_line(data = grid, aes(x, fit), colour = "#FF9500", linewidth = 1) +
  geom_rect(data = hx, aes(xmin = lo, xmax = hi, ymin = top, ymax = top + share * top_h),
            inherit.aes = FALSE, fill = "#007AFF", alpha = 0.35, colour = "white", linewidth = 0.3) +
  geom_rect(data = hy, aes(ymin = lo, ymax = hi, xmin = right, xmax = right + share * right_w),
            inherit.aes = FALSE, fill = "#007AFF", alpha = 0.35, colour = "white", linewidth = 0.3) +
  scale_x_continuous(breaks = xbreaks, expand = c(0.01, 0)) +
  scale_y_continuous(breaks = ybreaks, expand = c(0.01, 0)) +
  labs(title = "Pairs that trade more have more synchronised business cycles",
       subtitle = sprintf("Pearson r = %.2f, 95%% CI [%.2f, %.2f], t(%d) = %.1f, %s · slope = %.2f · n = %d pairs (simulated)",
                          ct$estimate, ct$conf.int[1], ct$conf.int[2], ct$parameter, ct$statistic,
                          p_txt, coef(fit)[2], n),
       x = "Bilateral trade intensity (log of trade / combined GDP)",
       y = "Business-cycle correlation") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.title.x = element_text(hjust = 0.4),
        axis.title.y = element_text(hjust = 0.4),
        panel.grid = element_blank())
print(p)
