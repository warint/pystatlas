# Within-subjects comparison: the R equivalent of plot.py
# Simulated data, set in the context of Warin, d'Anglade & de Marcellis-Warin (2025) [warin2025waste]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2025)

# Data: waste rate (% of output) of the same 24 lines at three stages
conditions <- c("Baseline", "After\nreorganisation", "Follow-up")
n <- 24
level <- rnorm(n, 8, 1.8)                       # each line's own level
wide <- sapply(c(0, -1.8, -1.4), function(shift) level + shift + rnorm(n, 0, 0.9))
long <- data.frame(id = factor(rep(1:n, 3)),
                   condition = factor(rep(conditions, each = n), levels = conditions),
                   waste = as.vector(wide))

# Repeated-measures ANOVA + partial eta squared
aov_table <- summary(aov(waste ~ condition + Error(id / condition), data = long))[["Error: id:condition"]][[1]]
Fv <- aov_table[["F value"]][1]
df <- aov_table[["Df"]]
eta_p <- Fv * df[1] / (Fv * df[1] + df[2])

# Pairwise paired t-tests (Holm) with Cohen's d_z; brackets stacked from the shortest to the longest
pairs <- as.data.frame(t(combn(3, 2)))
names(pairs) <- c("i", "j")
pairs$p <- p.adjust(mapply(function(i, j) t.test(wide[, j], wide[, i], paired = TRUE)$p.value,
                           pairs$i, pairs$j), method = "holm")
pairs$dz <- mapply(function(i, j) mean(wide[, j] - wide[, i]) / sd(wide[, j] - wide[, i]),
                   pairs$i, pairs$j)
pairs <- pairs[order(pairs$j - pairs$i), ]
top <- max(wide) + 0.4
pairs$y <- top + (seq_len(nrow(pairs)) - 1) * 0.8
pairs$label <- sprintf("%s, d_z = %.2f",
                       ifelse(pairs$p < 0.001, "p < .001", sprintf("p = %.3f", pairs$p)), pairs$dz)

means <- aggregate(waste ~ condition, data = long, FUN = mean)
fell <- sum(wide[, 2] < wide[, 1])
p <- ggplot(long, aes(condition, waste)) +
  geom_line(aes(group = id), colour = "#C7C7CC", linewidth = 0.35, alpha = 0.8) +   # one line per subject
  geom_violin(aes(fill = condition), alpha = 0.15, colour = NA, width = 0.55) +
  geom_point(aes(colour = condition), size = 1.6, alpha = 0.7, stroke = 0) +
  geom_line(data = means, aes(group = 1), colour = "#1D1D1F", linewidth = 1.1) +
  geom_point(data = means, shape = 21, size = 4.5, fill = "#1D1D1F", colour = "white", stroke = 1.2) +
  geom_label(data = means, aes(label = sprintf("μ = %.1f%%", waste)), fill = "white", alpha = 0.9,
             label.size = 0, fontface = "bold", size = 3.3, hjust = 0, colour = "#1D1D1F",
             position = position_nudge(x = 0.09)) +
  # Significance brackets
  geom_segment(data = pairs, aes(x = i, xend = j, y = y, yend = y), colour = "#8E8E93", linewidth = 0.4) +
  geom_segment(data = pairs, aes(x = i, xend = i, y = y - 0.2, yend = y), colour = "#8E8E93",
               linewidth = 0.4) +
  geom_segment(data = pairs, aes(x = j, xend = j, y = y - 0.2, yend = y), colour = "#8E8E93",
               linewidth = 0.4) +
  geom_text(data = pairs, aes(x = (i + j) / 2, y = y + 0.07, label = label), vjust = 0, size = 3.1,
            colour = "#3A3A3C") +
  scale_colour_manual(values = c("#FF9500", "#AF52DE", "#007AFF"), aesthetics = c("colour", "fill"),
                      guide = "none") +
  scale_x_discrete(labels = sprintf("%s\n(n = %d)", conditions, n)) +
  scale_y_continuous(limits = c(0, top + 3 * 0.8 + 0.2), expand = c(0, 0)) +
  labs(title = sprintf("Waste fell on %d of %d production lines after the reorganisation", fell, n),
       subtitle = sprintf(paste("n = %d lines (simulated) · repeated-measures ANOVA F(%.0f, %.0f) = %.1f,",
                                "p = %.1e, η²p = %.2f · pairwise: paired t, Holm"),
                          n, df[1], df[2], Fv, aov_table[["Pr(>F)"]][1], eta_p),
       x = NULL, y = "Waste rate (% of output)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.x = element_text(colour = "#1D1D1F", size = 10),
        panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
print(p)
