# Estimation plot: the R equivalent of plot.py
# Simulated data, set in the context of Warin, Melchior & de Marcellis-Warin (2025) [warin2025who]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(boot)
library(ggplot2)

set.seed(2025)

# Data: log10 of engagements per WHO post, before and after the start of COVID-19
pre <- rnorm(50, 2.6, 0.38)
post <- rnorm(50, 3.15, 0.45)
d <- data.frame(group = rep(c("pre", "post"), each = 50), x = rep(1:2, each = 50),
                log_eng = c(pre, post))

# Mean difference with a BCa bootstrap 95% CI, and the bootstrap distribution
mean_diff <- function(data, i) {
  b <- data[i, ]
  mean(b$log_eng[b$group == "post"]) - mean(b$log_eng[b$group == "pre"])
}
bs <- boot(d, mean_diff, R = 5000, strata = factor(d$group))   # resample within each group
ci <- boot.ci(bs, type = "bca")$bca[4:5]
delta <- mean(post) - mean(pre)
tt <- t.test(post, pre)                                         # Welch's t-test

# Mean ± SD of each group, drawn as a line beside its swarm
stats <- data.frame(x = 1:2 + 0.32, m = c(mean(pre), mean(post)), s = c(sd(pre), sd(post)))

# Bootstrap distribution as a half violin on the floating difference axis,
# whose zero sits at the pre-COVID mean
k <- density(bs$t)
violin <- data.frame(y = k$x + mean(pre), width = 0.45 * k$y / max(k$y))

p <- ggplot(d, aes(x, log_eng)) +
  geom_dotplot(aes(group = group, fill = group), binaxis = "y", stackdir = "center",
               binwidth = 0.045, colour = NA, show.legend = FALSE) +
  scale_fill_manual(values = c(pre = "#8E8E93", post = "#007AFF")) +
  geom_linerange(data = stats, aes(x = x, ymin = m - s, ymax = m + s), inherit.aes = FALSE,
                 colour = "#3A3A3C", linewidth = 0.7) +
  geom_point(data = stats, aes(x, m), shape = 21, size = 2.2, fill = "white", colour = "#3A3A3C",
             stroke = 0.8) +
  # Reference lines from the group means to the difference axis
  annotate("segment", x = 1.32, xend = 3.6, y = mean(pre), yend = mean(pre), colour = "#3A3A3C",
           linewidth = 0.4) +
  annotate("segment", x = 2.32, xend = 3.6, y = mean(post), yend = mean(post), colour = "#3A3A3C",
           linewidth = 0.4, linetype = "dotted") +
  # Difference: bootstrap distribution, 95% CI and point estimate
  geom_ribbon(data = violin, aes(y = y, xmin = 3, xmax = 3 + width), inherit.aes = FALSE,
              fill = "#007AFF", alpha = 0.3) +
  annotate("linerange", x = 3, ymin = mean(pre) + ci[1], ymax = mean(pre) + ci[2],
           colour = "#1D1D1F", linewidth = 1.2) +
  annotate("point", x = 3, y = mean(post), size = 3.2, colour = "#1D1D1F") +
  annotate("text", x = 2.95, y = mean(pre) + ci[2] + 0.04, hjust = 1, vjust = 0, lineheight = 0.95,
           label = sprintf("Δ = %+.2f\n[%.2f, %.2f]", delta, ci[1], ci[2]), colour = "#007AFF",
           fontface = "bold", size = 3.3) +
  scale_x_continuous(breaks = 1:3, limits = c(0.45, 3.6), expand = c(0, 0),
                     labels = c(sprintf("Before COVID-19\n(n = %d posts)", length(pre)),
                                sprintf("After COVID-19\n(n = %d posts)", length(post)),
                                "After − Before")) +
  scale_y_continuous(breaks = seq(2, 4, 0.5), labels = c("100", "316", "1,000", "3,162", "10,000"),
                     sec.axis = sec_axis(~ . - mean(pre), name = "Mean difference (log10 engagements)")) +
  labs(title = sprintf("Post minus pre: a mean difference of %+.2f on the log10 scale, with its bootstrap CI",
                       delta),
       subtitle = sprintf(paste("n = %d posts (simulated) · BCa bootstrap 95%% CI [%.2f, %.2f] ·",
                                "Welch's t(%.1f) = %.2f, p = %.1e"),
                          nrow(d), ci[1], ci[2], tt$parameter, tt$statistic, tt$p.value),
       x = NULL, y = "Engagements per post (log10 scale)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.x = element_text(colour = "#1D1D1F", size = 10),
        panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
print(p)
