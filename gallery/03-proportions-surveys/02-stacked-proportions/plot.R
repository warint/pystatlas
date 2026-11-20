# Stacked proportions by group: the R equivalent of plot.py
# Simulated data, set in the context of Sanger & Warin (2018) [sanger2018quebec]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2014)

# Simulated tweets: party mentioned and sentiment, with party-specific sentiment shares
sentiments <- c("Negative", "Neutral", "Positive")
probs <- list(PLQ = c(0.38, 0.37, 0.25), PQ = c(0.47, 0.33, 0.20),
              CAQ = c(0.33, 0.42, 0.25), QS = c(0.24, 0.40, 0.36))
sizes <- c(PLQ = 820, PQ = 900, CAQ = 420, QS = 260)
d <- do.call(rbind, lapply(names(sizes), function(party)
  data.frame(party = party, sentiment = sample(sentiments, sizes[[party]], replace = TRUE,
                                               prob = probs[[party]]))))
d$party <- factor(d$party, levels = names(sizes))
d$sentiment <- factor(d$sentiment, levels = sentiments)

# Pearson chi-squared test of independence + Cramér's V
tab <- table(d$party, d$sentiment)
chi <- chisq.test(tab, correct = FALSE)
v <- sqrt(unname(chi$statistic) / (sum(tab) * (min(dim(tab)) - 1)))

# Row percentages: the conditional distribution of sentiment within each party
pct <- as.data.frame(prop.table(tab, 1) * 100, responseName = "share")
names(pct)[1:2] <- c("party", "sentiment")
pct$sentiment <- factor(pct$sentiment, levels = rev(sentiments))    # negative at the bottom of the bar

p <- ggplot(pct, aes(party, share, fill = sentiment)) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.0f%%", share), colour = sentiment == "Neutral", group = sentiment),
            position = position_stack(vjust = 0.5), fontface = "bold", size = 3.7,
            show.legend = FALSE) +
  scale_colour_manual(values = c(`TRUE` = "#1D1D1F", `FALSE` = "white")) +
  scale_fill_manual(NULL, values = c(Negative = "#FF3B30", Neutral = "#C7C7CC", Positive = "#007AFF"),
                    breaks = sentiments) +
  scale_x_discrete(labels = sprintf("%s\n(n = %s tweets)", names(sizes),
                                    format(rowSums(tab), big.mark = ","))) +
  scale_y_continuous(breaks = seq(0, 100, 25), labels = function(b) paste0(b, "%"),
                     expand = c(0, 0)) +
  labs(title = "Sentiment shares by party, with a χ² test of independence",
       subtitle = sprintf("Pearson χ²(%d) = %.1f, %s, Cramér's V = %.2f · n = %s tweets (simulated)",
                          chi$parameter, chi$statistic,
                          if (chi$p.value < .001) "p < .001" else sprintf("p = %.3f", chi$p.value),
                          v, format(sum(tab), big.mark = ",")),
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.x = element_text(colour = "#1D1D1F", size = 10),
        panel.grid = element_blank(),
        legend.position = "top", legend.justification = "left",
        legend.key.size = unit(0.8, "lines"), legend.text = element_text(size = 9.5),
        legend.margin = margin(0, 0, 0, 0))
print(p)
