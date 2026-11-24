# Correlation matrix: the R equivalent of plot.py
# Simulated data, set in the context of Stojkov & Warin (2019) [stojkov2019trilemma]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(1999)

# Simulated cross-section: 120 countries; the trilemma indices share a budget of about 2
n <- 120
g <- matrix(rgamma(3 * n, shape = 2), n)
share <- g / rowSums(g)                         # Dirichlet(2, 2, 2): how each country "gives up" the budget
idx <- pmin(pmax(1 - share + matrix(rnorm(3 * n, 0, 0.08), n), 0), 1)
ers <- idx[, 1]; mi <- idx[, 2]; ka <- idx[, 3]
d <- data.frame("Exch.-rate stability" = ers, "Monetary independence" = mi,
                "Financial openness" = ka,
                "Reserves (% GDP)" = 8 + 25 * ers + rnorm(n, 0, 7),
                "Trade openness (% GDP)" = 50 + 60 * ka + rnorm(n, 0, 25),
                "Inflation (%)" = 6 - 4 * mi + rgamma(n, shape = 2, scale = 1.5),
                "GDP growth (%)" = 3 + rnorm(n, 0, 2), check.names = FALSE)
vars <- names(d)
k <- length(vars)

# Pearson r and Holm-adjusted p-values for every pair (lower triangle: row after column)
pairs <- as.data.frame(t(combn(k, 2)))
names(pairs) <- c("col", "row")
tests <- mapply(function(i, j) cor.test(d[[i]], d[[j]]), pairs$col, pairs$row, SIMPLIFY = FALSE)
pairs$r <- sapply(tests, function(t) unname(t$estimate))
pairs$p_adj <- p.adjust(sapply(tests, `[[`, "p.value"), method = "holm")
pairs$sig <- pairs$p_adj < 0.05
pairs$x <- factor(vars[pairs$col], levels = vars[-k])
pairs$y <- factor(vars[pairs$row], levels = rev(vars[-1]))
sig <- pairs[pairs$sig, ]
ns <- pairs[!pairs$sig, ]

p <- ggplot(pairs, aes(x, y)) +
  geom_tile(aes(fill = ifelse(sig, r, NA)), width = 0.94, height = 0.94) +
  scale_fill_gradient2("Pearson r", low = "#B2182B", mid = "#F7F7F7", high = "#2166AC", midpoint = 0,
                       limits = c(-1, 1), breaks = seq(-1, 1, 0.5), na.value = "#F2F2F7") +
  geom_text(data = sig, aes(label = sprintf("%.2f", r), colour = abs(r) > 0.6), size = 3.7,
            show.legend = FALSE) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "#1D1D1F")) +
  # Not significant after Holm: a grey cross, with r in small print below
  geom_point(data = ns, shape = 4, size = 5, colour = "#AEAEB2", stroke = 0.6,
             position = position_nudge(y = 0.08)) +
  geom_text(data = ns, aes(label = sprintf("%.2f", r)), size = 2.8, colour = "#8E8E93",
            nudge_y = -0.3) +
  coord_equal() +
  labs(title = "The three trilemma indices pull against each other",
       subtitle = sprintf("Pearson correlations · n = %d countries (simulated) · × = not significant (Holm-adjusted p ≥ .05)",
                          n),
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text = element_text(colour = "#1D1D1F", size = 9.5),
        axis.text.x = element_text(angle = 35, hjust = 1),
        panel.grid = element_blank(),
        legend.title = element_text(size = 9.5, colour = "#3A3A3C"),
        legend.key.height = unit(2.2, "lines"))
print(p)
