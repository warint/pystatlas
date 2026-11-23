# Cross-table with standardized residuals: the R equivalent of plot.py
# Simulated data, set in the context of Kouloukoui et al. (2023) [kouloukoui2023esg]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2023)

# Simulated tweets: ESG sub-topic (rows) × world region (columns), topic mix varies by region
topics <- c("E · Climate change", "E · Pollution & waste", "S · Labour & human rights",
            "S · Diversity & inclusion", "G · Board & executive pay", "G · Corruption & ethics")
mix <- list("North America" = c(.30, .10, .12, .22, .18, .08), "Europe" = c(.38, .14, .12, .14, .14, .08),
            "Asia-Pacific" = c(.28, .24, .14, .10, .14, .10), "Latin America" = c(.24, .16, .20, .10, .08, .22),
            "Africa" = c(.22, .14, .24, .10, .08, .22))
sizes <- c(1400, 1200, 800, 350, 250)
d <- do.call(rbind, lapply(seq_along(mix), function(i)
  data.frame(region = names(mix)[i], topic = sample(topics, sizes[i], replace = TRUE, prob = mix[[i]]))))
obs <- table(factor(d$topic, topics), factor(d$region, names(mix)))
n <- sum(obs)

# Chi-squared test and adjusted (standardized) residuals:
# (O − E) / √(E · (1 − row share) · (1 − column share)), returned by chisq.test() as $stdres
chi <- chisq.test(obs, correct = FALSE)
v <- sqrt(unname(chi$statistic) / (n * (min(dim(obs)) - 1)))

comma <- function(x) format(x, big.mark = ",", trim = TRUE)
cells <- data.frame(topic = rownames(obs)[row(obs)], region = colnames(obs)[col(obs)],
                    count = as.vector(obs), resid = as.vector(chi$stdres),
                    row_pct = as.vector(100 * prop.table(obs, 1)))
# Totals in the axis labels; two-word regions on two lines; first topic on top
cells$topic <- factor(cells$topic, levels = rev(topics),
                      labels = rev(sprintf("%s  (n = %s)", topics, comma(rowSums(obs)))))
cells$region <- factor(cells$region, levels = names(mix),
                       labels = sprintf("%s\n(n = %s)", sub(" ", "\n", names(mix)), comma(colSums(obs))))
cells$ink <- ifelse(abs(cells$resid) > 3, "white", "#1D1D1F")
cells$face <- ifelse(abs(cells$resid) >= 2, "bold", "plain")

p <- ggplot(cells, aes(region, topic)) +
  geom_tile(aes(fill = resid), colour = "white", linewidth = 1.5) +     # white gutters between cells
  geom_text(aes(label = comma(count), colour = ink, fontface = face), size = 3.9, nudge_y = 0.12) +
  geom_text(aes(label = sprintf("%.0f%%", row_pct), colour = ink), size = 3, nudge_y = -0.2) +
  scale_colour_identity() +
  # red = fewer than expected, blue = more
  scale_fill_gradient2("Standardized\nresidual", low = "#B2182B", mid = "#F7F7F7", high = "#2166AC",
                       midpoint = 0, limits = c(-6, 6), breaks = seq(-6, 6, 2), oob = scales::squish) +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(title = "Each region tweets about a different side of ESG",
       subtitle = sprintf(paste("Pearson χ²(%d) = %.0f, %s, V = %.2f · n = %s tweets (simulated) ·",
                                "count & row %% · bold: |resid.| ≥ 2"),
                          chi$parameter, chi$statistic,
                          if (chi$p.value < .001) "p < .001" else sprintf("p = %.3f", chi$p.value),
                          v, comma(n)),
       x = "World region of the tweet", y = "ESG sub-topic") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text = element_text(colour = "#1D1D1F", size = 9.5),
        axis.text.x = element_text(lineheight = 0.95),
        panel.grid = element_blank(),
        legend.title = element_text(size = 9.5, colour = "#3A3A3C"),
        legend.key.height = unit(2.2, "lines"))
print(p)
