# Q–Q plot with confidence band: the R equivalent of plot.py
# Simulated data, set in the context of Warin (2021) [warin2021corona]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2021)

# Data: citations per paper, raw and log(1 + x)
citations <- rpois(300, rlnorm(300, 2.5, 1.0))
samples <- list("Raw citations" = citations, "log(1 + citations)" = log1p(citations))

# Normal Q–Q coordinates of one sample, with a 95% pointwise envelope
qq <- function(name) {
  x <- samples[[name]]
  n <- length(x)
  i <- seq_len(n)
  d <- data.frame(panel = name,
                  z = sort((x - mean(x)) / sd(x)),          # standardised sample quantiles
                  theo = qnorm((i - 0.5) / n),              # theoretical normal quantiles
                  # order statistic i of a uniform sample is Beta(i, n - i + 1)
                  lo = qnorm(qbeta(0.025, i, n - i + 1)),
                  hi = qnorm(qbeta(0.975, i, n - i + 1)))
  d$group <- ifelse(d$z < d$lo | d$z > d$hi, "outside", name)
  d
}
d <- do.call(rbind, lapply(names(samples), qq))
d$panel <- factor(d$panel, levels = names(samples))

# One line of statistics per panel
notes <- do.call(rbind, lapply(names(samples), function(name) {
  sw <- shapiro.test(samples[[name]])
  data.frame(panel = factor(name, levels = names(samples)),
             label = sprintf("Shapiro–Wilk W = %.3f, p = %.1e\n%d of %d points outside the band",
                             sw$statistic, sw$p.value,
                             sum(d$group[d$panel == name] == "outside"), length(samples[[name]])))
}))

p <- ggplot(d, aes(theo, z)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = "#8E8E93", alpha = 0.18) +
  geom_abline(intercept = 0, slope = 1, colour = "#8E8E93", linewidth = 0.4, linetype = "dashed") +
  geom_point(aes(colour = group, size = group), alpha = 0.75, stroke = 0) +
  scale_colour_manual(values = c("Raw citations" = "#FF9500", "log(1 + citations)" = "#007AFF",
                                 outside = "#FF3B30"), guide = "none") +
  scale_size_manual(values = c("Raw citations" = 1.5, "log(1 + citations)" = 1.5, outside = 1.9),
                    guide = "none") +
  geom_text(data = notes, aes(x = -3.05, y = 7.9, label = label), hjust = 0, vjust = 1,
            size = 3.1, colour = "#6E6E73", lineheight = 1.05) +
  facet_wrap(~panel) +
  coord_cartesian(xlim = c(-3.2, 3.2), ylim = c(-3.3, 8.3), expand = FALSE) +   # room for the most-cited papers
  labs(title = "Citation counts are heavily right-skewed; log(1 + x) brings them close to normal",
       subtitle = sprintf("n = %d papers (simulated) · shaded band = 95%% pointwise envelope · red = points outside the band",
                          length(citations)),
       x = "Theoretical normal quantiles", y = "Standardised sample quantiles") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        strip.text = element_text(size = 10.5, face = "bold", hjust = 0),
        panel.grid.minor = element_blank(),
        panel.spacing = unit(1.4, "lines"),
        plot.margin = margin(8, 14, 8, 8))
print(p)
