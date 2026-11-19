# Tukey HSD intervals: the R equivalent of plot.py
# Simulated data, set in the context of Sanger & Warin (2020) [sanger2020ecb]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2020)

# Data: % of each speech on price stability, by ECB president (mean, SD, number of speeches)
spec <- data.frame(president = c("Duisenberg", "Trichet", "Draghi", "Lagarde"),
                   mean = c(22, 19, 14, 16), sd = c(7, 7, 6, 7), speeches = c(60, 90, 90, 50))
d <- data.frame(president = factor(rep(spec$president, spec$speeches), levels = spec$president),
                share = pmin(pmax(rnorm(sum(spec$speeches), rep(spec$mean, spec$speeches),
                                        rep(spec$sd, spec$speeches)), 0), 100))

# One-way ANOVA, then Tukey HSD on all 6 pairs
fit <- aov(share ~ president, data = d)
tab <- summary(fit)[[1]]
Fv <- tab[["F value"]][1]
k <- nlevels(d$president)
n <- nrow(d)
eta2 <- Fv * (k - 1) / (Fv * (k - 1) + n - k)
res <- as.data.frame(TukeyHSD(fit)$president)
# TukeyHSD() reports "later − earlier" president; flip to read "earlier − later"
pair <- do.call(rbind, strsplit(rownames(res), "-"))
res <- data.frame(label = paste(pair[, 2], "−", pair[, 1]), diff = -res$diff, low = -res$upr,
                  high = -res$lwr, p_adj = res[["p adj"]])
res$label <- factor(res$label, levels = rev(res$label))        # first pair on top
res$reject <- res$p_adj < 0.05
res$direction <- ifelse(!res$reject, "none", ifelse(res$diff > 0, "higher", "lower"))
res$note <- sprintf("%+.1f pts, %s", res$diff,
                    ifelse(res$p_adj < 0.001, "p < .001", sprintf("p = %.3f", res$p_adj)))

p <- ggplot(res, aes(diff, label, colour = direction)) +
  geom_vline(xintercept = 0, colour = "#8E8E93", linewidth = 0.45, linetype = "dashed") +
  geom_linerange(aes(xmin = low, xmax = high), linewidth = 1.2) +
  geom_point(aes(fill = direction), shape = 21, size = 3, colour = "white", stroke = 0.8) +
  geom_text(data = res[res$reject, ], aes(x = high, label = note), hjust = 0, nudge_x = 0.4,
            size = 3.1, colour = "#3A3A3C") +
  scale_colour_manual(values = c(none = "#AEAEB2", higher = "#007AFF", lower = "#FF9500"),
                      aesthetics = c("colour", "fill"), guide = "none") +
  scale_x_continuous(limits = c(NA, max(res$high) + 6)) +
  labs(title = "Tukey intervals for every pair of presidents in a simulated speech sample",
       subtitle = sprintf(paste("n = %d speeches (simulated) · one-way ANOVA F(%d, %d) = %.1f, p = %.1e,",
                                "η² = %.2f · %d of %d pairs differ"),
                          n, k - 1, n - k, Fv, tab[["Pr(>F)"]][1], eta2, sum(res$reject), nrow(res)),
       x = "Difference in mean share on price stability (percentage points), Tukey 95% CI",
       y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.y = element_text(colour = "#1D1D1F", size = 9.5),
        panel.grid.minor = element_blank(), panel.grid.major.y = element_blank())
print(p)
