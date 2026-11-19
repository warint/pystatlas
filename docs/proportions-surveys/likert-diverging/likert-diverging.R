# Diverging Likert chart: the R equivalent of plot.py
# Simulated data, set in the context of Melchior, Warin & Oliveira (2026) [melchior2026survey]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(42)

# Simulated survey: 600 Facebook users, 6 items, answers 1-5 driven by a shared latent attitude
items <- c("I check the source before sharing" = 0.8, "I trust COVID-19 news on Facebook" = -0.9,
           "I share news without reading it" = -1.1, "Friends' posts shape my views" = 0.1,
           "I can spot fake COVID-19 news" = 0.4, "I report false posts I see" = -0.4)
mood <- rnorm(600, 0, 0.8)
answers <- sapply(items, function(shift)
  findInterval(shift + mood + rnorm(600), c(-1.6, -0.5, 0.5, 1.6)) + 1)

levels <- c("Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree")
colors <- setNames(c("#C0392B", "#F1948A", "#D1D1D6", "#7FB3F5", "#0A64D8"), levels)
pct <- t(apply(answers, 2, function(a) 100 * prop.table(table(factor(a, levels = 1:5)))))
pct <- pct[order(pct[, 4] + pct[, 5]), ]        # sorted by total agreement, highest on top

# Friedman test: do the six items get the same answers from the same people?
fr <- friedman.test(answers)

# One rectangle per item and answer level, shifted so that neutral is centred on 0
start <- -(pct[, 1] + pct[, 2] + pct[, 3] / 2)
right <- start + t(apply(pct, 1, cumsum))
bars <- data.frame(item = rep(seq_len(nrow(pct)), 5), level = factor(rep(levels, each = nrow(pct)), levels),
                   share = as.vector(pct), xmax = as.vector(right))
bars$xmin <- bars$xmax - bars$share
bars$text <- ifelse(bars$level %in% levels[c(1, 5)], "white", "#1D1D1F")
totals <- data.frame(item = seq_len(nrow(pct)), disagree = pct[, 1] + pct[, 2],
                     agree = pct[, 4] + pct[, 5])

p <- ggplot(bars) +
  geom_vline(xintercept = 0, colour = "#3A3A3C", linewidth = 0.4) +
  geom_rect(aes(xmin = xmin, xmax = xmax, ymin = item - 0.33, ymax = item + 0.33, fill = level),
            colour = "white", linewidth = 0.4) +
  geom_text(data = bars[bars$share >= 6, ],
            aes(x = (xmin + xmax) / 2, y = item, label = sprintf("%.0f", share), colour = text),
            size = 3) +
  scale_colour_identity() +
  # Totals on both sides
  geom_text(data = totals, aes(x = -93, y = item, label = sprintf("%.0f%%", disagree)),
            colour = "#C0392B", fontface = "bold", size = 3.5) +
  geom_text(data = totals, aes(x = 93, y = item, label = sprintf("%.0f%%", agree)),
            colour = "#0A64D8", fontface = "bold", size = 3.5) +
  annotate("text", x = c(-93, 93), y = nrow(pct) + 0.62, label = c("Disagree", "Agree"),
           size = 3.1, colour = "#6E6E73") +
  scale_fill_manual(NULL, values = colors) +
  scale_x_continuous(breaks = seq(-60, 60, 20), labels = function(v) paste0(abs(v), "%"),
                     limits = c(-102, 102), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq_len(nrow(pct)), labels = rownames(pct),
                     limits = c(0.4, nrow(pct) + 0.9), expand = c(0, 0)) +
  labs(title = "Six statements on COVID-19 news, sorted by total agreement",
       subtitle = sprintf("n = %d respondents (simulated) · Friedman χ²(%d) = %.0f, %s · bar labels in %%",
                          nrow(answers), ncol(answers) - 1, fr$statistic,
                          if (fr$p.value < .001) "p < .001" else sprintf("p = %.3f", fr$p.value)),
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.y = element_text(colour = "#1D1D1F", size = 10),
        panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        legend.position = "bottom", legend.key.size = unit(0.8, "lines"),
        legend.text = element_text(size = 9))
print(p)
