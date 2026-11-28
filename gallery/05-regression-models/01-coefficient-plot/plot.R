# Coefficient plot: the R equivalent of plot.py
# Simulated data, set in the context of Stojkov, Veljanovski & Warin (2024) [stojkov2024growth]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)
library(broom)

set.seed(2024)

# Data: 11 emerging-European economies × 20 years, six institutional and policy factors
n <- 220
labels <- c(rule_law = "Rule of law", reg_quality = "Regulatory quality",
            investment = "Investment (% of GDP)", openness = "Trade openness",
            inflation = "Inflation", gov_size = "Government consumption")
predictors <- names(labels)
df <- as.data.frame(matrix(rnorm(n * 6), n, dimnames = list(NULL, predictors)))
df$growth <- with(df, 3 + 0.9 * rule_law + 0.45 * reg_quality + 0.8 * investment
                  + 0.15 * openness - 0.7 * inflation - 0.05 * gov_size + rnorm(n, 0, 2.5))
df[predictors] <- scale(df[predictors])

model <- lm(reformulate(predictors, "growth"), data = df)

# Tidy the results: estimate, 95% CI and p-value per term (intercept dropped)
# (ggstats is not installed: broom::tidy() + ggplot2 do what ggcoef_model() does)
coefs <- subset(tidy(model, conf.int = TRUE), term != "(Intercept)")
coefs$label <- factor(labels[coefs$term], levels = labels[coefs$term[order(coefs$estimate)]])
coefs$sig <- coefs$p.value < 0.05
coefs$text <- sprintf("%+.2f%s", coefs$estimate,
                      ifelse(coefs$p.value < .001, "***", ifelse(coefs$p.value < .01, "**",
                             ifelse(coefs$p.value < .05, "*", ""))))

p <- ggplot(coefs, aes(estimate, label, colour = sig)) +
  geom_vline(xintercept = 0, colour = "#8E8E93", linewidth = 0.5, linetype = "dashed") +
  geom_linerange(aes(xmin = conf.low, xmax = conf.high), linewidth = 1.2) +
  geom_point(aes(fill = sig), shape = 21, size = 3.8, colour = "white", stroke = 1) +
  geom_label(aes(x = conf.high, label = text), hjust = 0, nudge_x = 0.04, size = 3.5,
             colour = "#3A3A3C", fill = "white", label.size = 0,
             label.padding = unit(0.1, "lines")) +              # white box: legible over the zero line
  scale_colour_manual(values = c(`TRUE` = "#007AFF", `FALSE` = "#AEAEB2"), guide = "none") +
  scale_fill_manual(values = c(`TRUE` = "#007AFF", `FALSE` = "#AEAEB2"), guide = "none") +
  scale_x_continuous(breaks = seq(-2, 2, 0.5), expand = expansion(mult = 0.15)) +
  labs(title = "Rule of law and investment go with faster growth",
       subtitle = sprintf("OLS · n = %d country-years (simulated) · R² = %.2f · * p<.05  ** p<.01  *** p<.001",
                          nobs(model), summary(model)$r.squared),
       x = "Effect on GDP-per-capita growth (pp per 1 SD), 95% CI", y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.y = element_text(colour = "#1D1D1F", size = 10),
        panel.grid.minor = element_blank(), panel.grid.major.y = element_blank())
print(p)
