# ECDF comparison: the R equivalent of plot.py
# Simulated data, set in the context of Pastoriza Rivas & Warin (2022) [pastoriza2022]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2022)

# Data: round score to par (strokes) in two periods of the career
early <- round(rnorm(320, -2.4, 2.9))
late <- round(rnorm(200, -0.6, 3.1))
label <- function(name, x) sprintf("Rounds %s (n = %d, median %+.0f)", name, length(x), median(x))
d <- data.frame(score = c(early, late),
                period = rep(c(label("1997–2009", early), label("2010–2019", late)),
                             c(length(early), length(late))))

# Two-sample Kolmogorov–Smirnov test: D is the largest vertical gap
# (the p-value is approximate, because whole-stroke scores have many ties)
ks <- suppressWarnings(ks.test(early, late, exact = FALSE))
D <- unname(ks$statistic)
values <- sort(unique(d$score))
x_gap <- values[which.max(abs(ecdf(early)(values) - ecdf(late)(values)))]
gap <- data.frame(x = x_gap, y = c(ecdf(early)(x_gap), ecdf(late)(x_gap)))

lo <- min(d$score) - 1                          # common x-range for both curves
hi <- max(d$score) + 1
p <- ggplot(d, aes(score, colour = period)) +
  stat_ecdf(geom = "step", pad = TRUE, linewidth = 0.9) +
  scale_colour_manual(NULL, values = c("#007AFF", "#FF9500")) +
  # Mark the KS distance
  geom_line(data = gap, aes(x, y), colour = "#FF3B30", linewidth = 1.2, lineend = "round") +
  geom_point(data = gap, aes(x, y), colour = "#FF3B30", size = 2) +
  annotate("segment", x = x_gap + 0.1, y = min(gap$y), xend = x_gap + 0.55, yend = min(gap$y) - 0.11,
           colour = "#FF3B30", linewidth = 0.35) +
  annotate("text", x = x_gap + 0.6, y = min(gap$y) - 0.14, hjust = 0, lineheight = 0.95,
           label = sprintf("D = %.2f\nat %+.0f", D, x_gap), colour = "#FF3B30", fontface = "bold",
           size = 3.5) +
  scale_x_continuous(breaks = seq(-10, hi, 2)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25), labels = scales::percent) +
  coord_cartesian(xlim = c(lo, hi), ylim = c(0, 1.02), expand = FALSE) +
  labs(title = sprintf("The KS statistic is the widest vertical gap between the two ECDFs: D = %.2f", D),
       subtitle = sprintf("n = %d rounds (simulated) · two-sample Kolmogorov–Smirnov D = %.3f, p = %.1e",
                          nrow(d), D, ks$p.value),
       x = "Round score relative to par (strokes; lower is better)",
       y = "Cumulative share of rounds") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11.5),
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        panel.grid.minor = element_blank(),
        legend.position = "inside", legend.position.inside = c(0.98, 0.04),
        legend.justification = c(1, 0),
        legend.background = element_rect(fill = "white", colour = NA),
        plot.margin = margin(8, 14, 8, 8))
print(p)
