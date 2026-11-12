# Histogram with density: the R equivalent of plot.py
# Simulated data, set in the context of Sanger & Warin (2019) [sanger2019]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2019)

# Data: Jaccard similarity (0 = no shared words, 1 = identical) of 600 manifesto pairs
x <- rbeta(600, 1.6, 7.5)
n <- length(x)
m <- mean(x)
med <- median(x)
s <- sd(x)

# Shape statistics and normality test
dev <- x - m
skew <- mean(dev^3) / mean(dev^2)^1.5
kurt <- mean(dev^4) / mean(dev^2)^2 - 3         # excess kurtosis, 0 for a normal
sw <- shapiro.test(x)

# Mean and median markers, labelled on either side of their line
marks <- data.frame(value = c(m, med), name = c("Mean", "Median"), hjust = c(-0.15, 1.15))
marks$label <- sprintf("%s\n%.3f", marks$name, marks$value)

p <- ggplot(data.frame(x = x), aes(x)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, boundary = 0,
                 fill = "#007AFF", alpha = 0.25, colour = "white") +
  # Kernel density estimate and the normal curve with the same mean / SD
  geom_density(aes(colour = "Kernel density", linetype = "Kernel density"), linewidth = 1,
               key_glyph = "path") +
  stat_function(aes(colour = "Normal fit (same mean, SD)", linetype = "Normal fit (same mean, SD)"),
                fun = dnorm, args = list(mean = m, sd = s), linewidth = 0.7) +
  scale_colour_manual(NULL, values = c("Kernel density" = "#007AFF",
                                       "Normal fit (same mean, SD)" = "#8E8E93")) +
  scale_linetype_manual(NULL, values = c("Kernel density" = "solid",
                                         "Normal fit (same mean, SD)" = "dashed")) +
  geom_vline(data = marks, aes(xintercept = value), colour = c("#FF9500", "#34C759"),
             linewidth = 0.7) +
  geom_text(data = marks, aes(x = value, y = Inf, label = label, hjust = hjust), vjust = 1.2,
            colour = c("#FF9500", "#34C759"), fontface = "bold", size = 3.3, lineheight = 0.95) +
  scale_x_continuous(limits = c(0, 0.7), expand = c(0, 0)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +     # headroom for the labels
  labs(title = sprintf("Most manifesto pairs share little vocabulary: median Jaccard %.2f, %d pairs above 0.4",
                       med, sum(x > 0.4)),
       subtitle = sprintf(paste("n = %d manifesto pairs (simulated) · skewness = %.2f ·",
                                "excess kurtosis = %.2f · Shapiro–Wilk W = %.3f, p = %.1e"),
                          n, skew, kurt, unname(sw$statistic), sw$p.value),
       x = "Jaccard similarity between two party manifestos", y = "Density") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11.5),
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.y = element_blank(),
        panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        legend.position = "inside", legend.position.inside = c(0.98, 0.72),
        legend.justification = c(1, 1))
print(p)
