# Between-groups comparison: the R equivalent of plot.py
# Simulated data, set in the context of Kouloukoui et al. (2021) [kouloukoui2021]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2021)

# Data: perceived climate-risk exposure score (0–100) of firms, by industry
industries <- c("Oil and gas", "Mining and metals", "Electric utilities")
design <- data.frame(mean = c(68, 61, 53), sd = c(12, 14, 10), firms = c(45, 38, 42))
d <- do.call(rbind, lapply(1:3, function(i)
  data.frame(industry = industries[i],
             score = pmin(pmax(rnorm(design$firms[i], design$mean[i], design$sd[i]), 0), 100))))
d$industry <- factor(d$industry, levels = industries)
palette <- setNames(c("#007AFF", "#FF9500", "#34C759"), industries)
k <- nlevels(d$industry)
n <- nrow(d)

# Welch's ANOVA + omega squared
welch <- oneway.test(score ~ industry, data = d, var.equal = FALSE)
Fw <- unname(welch$statistic)
omega2 <- (Fw - 1) * (k - 1) / ((Fw - 1) * (k - 1) + n)

# Pairwise Welch t-tests, Holm-corrected; brackets stacked from the shortest to the longest
pairs <- as.data.frame(t(combn(k, 2)))
names(pairs) <- c("i", "j")
pairs$p <- p.adjust(mapply(function(i, j)
  t.test(d$score[as.integer(d$industry) == i], d$score[as.integer(d$industry) == j])$p.value,
  pairs$i, pairs$j), method = "holm")
pairs <- pairs[order(pairs$j - pairs$i), ]
top <- max(d$score)
pairs$y <- top + 4 + (seq_len(nrow(pairs)) - 1) * 8
pairs$label <- ifelse(pairs$p < 0.001, "p < .001", sprintf("p = %.3f", pairs$p))

means <- aggregate(score ~ industry, data = d, FUN = mean)
p <- ggplot(d, aes(industry, score, colour = industry, fill = industry)) +
  geom_violin(alpha = 0.15, colour = NA, width = 0.8, trim = TRUE) +
  geom_boxplot(width = 0.12, outlier.shape = NA, fill = "white", colour = "#3A3A3C",
               linewidth = 0.4) +
  geom_jitter(width = 0.2, height = 0, size = 1.4, alpha = 0.6, stroke = 0) +
  geom_point(data = means, shape = 21, size = 4, colour = "white", stroke = 1.2) +
  geom_label(data = means, aes(label = sprintf("μ = %.0f", score)), fill = "white", alpha = 0.9,
             label.size = 0, fontface = "bold", size = 3.3, hjust = 0,
             position = position_nudge(x = 0.16)) +
  # Significance brackets
  geom_segment(data = pairs, aes(x = i, xend = j, y = y, yend = y), inherit.aes = FALSE,
               colour = "#8E8E93", linewidth = 0.4) +
  geom_segment(data = pairs, aes(x = i, xend = i, y = y - 1.5, yend = y), inherit.aes = FALSE,
               colour = "#8E8E93", linewidth = 0.4) +
  geom_segment(data = pairs, aes(x = j, xend = j, y = y - 1.5, yend = y), inherit.aes = FALSE,
               colour = "#8E8E93", linewidth = 0.4) +
  geom_text(data = pairs, aes(x = (i + j) / 2, y = y + 0.8, label = label), inherit.aes = FALSE,
            vjust = 0, size = 3.1, colour = "#3A3A3C") +
  scale_colour_manual(values = palette, aesthetics = c("colour", "fill"), guide = "none") +
  scale_x_discrete(labels = sprintf("%s\n(n = %d firms)", industries, table(d$industry))) +
  scale_y_continuous(breaks = seq(20, 100, 20)) +
  coord_cartesian(ylim = c(NA, top + 4 + nrow(pairs) * 8 + 2)) +
  labs(title = "Three industries compared: Welch's ANOVA with Holm-adjusted pairwise tests",
       subtitle = sprintf(paste("n = %d firms (simulated) · Welch's F(%.0f, %.1f) = %.1f, p = %.1e,",
                                "ω² = %.2f · pairwise: Welch t, Holm"),
                          n, welch$parameter[1], welch$parameter[2], Fw, welch$p.value, omega2),
       x = NULL, y = "Perceived climate-risk exposure\n(score, 0–100)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11.5),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.x = element_text(colour = "#1D1D1F", size = 10),
        panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
print(p)
