# Raincloud plot: the R equivalent of plot.py
# Simulated data, set in the context of de Marcellis-Warin, Kouloukoui & Warin (2025) [dmw2025aitweets]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2025)

# Data: sentiment score (-1 to +1) of 120 tweets per AI topic (Beta draws rescaled)
topics <- c("AI in healthcare", "Generative AI tools", "AI regulation", "AI and jobs")
shape1 <- c(5.0, 4.0, 3.5, 2.6)
shape2 <- c(3.5, 3.5, 4.0, 4.5)
palette <- setNames(c("#007AFF", "#FF9500", "#34C759", "#FF3B30"), topics)
d <- do.call(rbind, lapply(1:4, function(i)
  data.frame(topic = topics[i], pos = i - 1, score = 2 * rbeta(120, shape1[i], shape2[i]) - 1)))
n <- nrow(d)

# Kruskal-Wallis test + epsilon squared
kw <- kruskal.test(score ~ topic, data = d)
H <- unname(kw$statistic)
eps2 <- H / (n - 1)

# Cloud: half violin above the baseline, trimmed near the data and scaled to a fixed height
cloud <- do.call(rbind, lapply(split(d, d$topic), function(g) {
  k <- density(g$score, from = min(g$score) - 0.05, to = max(g$score) + 0.05, n = 300)
  data.frame(topic = g$topic[1], pos = g$pos[1], score = k$x, height = 0.55 * k$y / max(k$y))
}))
# Rain: jittered raw points below the box
d$rain <- d$pos - 0.3 + runif(n, -0.08, 0.08)
# Group summaries: median for the box and the axis labels, share of negative tweets for the title
med <- aggregate(score ~ topic + pos, data = d, FUN = median)
med <- med[order(med$pos), ]
share <- tapply(d$score < 0, d$topic, mean)

p <- ggplot(d, aes(x = score, colour = topic, fill = topic)) +
  geom_vline(xintercept = 0, colour = "#8E8E93", linewidth = 0.4, linetype = "dashed") +
  geom_ribbon(data = cloud, aes(ymin = pos, ymax = pos + height), alpha = 0.3, linewidth = 0) +
  geom_line(data = cloud, aes(y = pos + height), linewidth = 0.6) +
  # Box: median and quartiles just under the cloud
  geom_boxplot(aes(y = pos - 0.1, group = topic), orientation = "y", width = 0.09,
               outlier.shape = NA, fill = "white", colour = "#3A3A3C", linewidth = 0.4,
               fatten = 0) +
  geom_segment(data = med, aes(xend = score, y = pos - 0.145, yend = pos - 0.055),
               linewidth = 1, lineend = "butt") +
  geom_point(aes(y = rain), size = 1, alpha = 0.55, stroke = 0) +
  scale_colour_manual(values = palette, aesthetics = c("colour", "fill"), guide = "none") +
  scale_x_continuous(limits = c(-1, 1), expand = c(0, 0)) +
  scale_y_continuous(breaks = med$pos, limits = c(-0.5, 3.7), expand = c(0, 0),
                     labels = sprintf("%s\nmedian %+.2f · n = 120", med$topic, med$score)) +
  labs(title = sprintf("%.0f%% of tweets on AI and jobs are negative, %.0f%% of those on AI in healthcare",
                       100 * share[["AI and jobs"]], 100 * share[["AI in healthcare"]]),
       subtitle = sprintf(paste("n = %d tweets (simulated) · Kruskal–Wallis H(3) = %.1f, p = %.1e,",
                                "ε² = %.2f · box = quartiles, rain = tweets"),
                          n, H, kw$p.value, eps2),
       x = "Tweet sentiment score (−1 = negative, +1 = positive)", y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11.5),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.y = element_text(colour = "#1D1D1F", lineheight = 0.95),
        panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        plot.margin = margin(8, 14, 8, 8))
print(p)
