# Mosaic plot with Pearson residuals: the R equivalent of plot.py
# Simulated data, set in the context of Stojkov & Warin (2023) [stojkov2023refugee]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2015)

# Simulated tweets: country of origin × dominant discourse frame, frame shares vary by country
frames <- c("Humanitarian", "Security", "Economic", "Political")
mix <- list(Germany = c(0.46, 0.18, 0.18, 0.18), UK = c(0.30, 0.26, 0.26, 0.18),
            France = c(0.36, 0.26, 0.14, 0.24), Italy = c(0.32, 0.22, 0.16, 0.30),
            Hungary = c(0.16, 0.44, 0.10, 0.30))
sizes <- c(Germany = 1000, UK = 700, France = 550, Italy = 450, Hungary = 300)
d <- do.call(rbind, lapply(names(mix), function(country)
  data.frame(country = country, frame = sample(frames, sizes[[country]], replace = TRUE,
                                               prob = mix[[country]]))))
obs <- table(factor(d$country, names(mix)), factor(d$frame, frames))

# Chi-squared test and Pearson residuals: (observed − expected) / √expected
chi <- chisq.test(obs, correct = FALSE)
resid <- chi$residuals

# One tile per cell: column widths are the country shares, heights the frame shares within a country
gap <- 0.012
width <- rowSums(obs) / sum(obs)
left <- cumsum(width) - width
height <- prop.table(obs, 1)
bottom <- t(apply(height, 1, cumsum)) - height
tiles <- data.frame(country = rownames(obs)[row(obs)], frame = colnames(obs)[col(obs)],
                    count = as.vector(obs), resid = as.vector(resid),
                    xmin = left[row(obs)], xmax = left[row(obs)] + width[row(obs)] - gap,
                    ymin = as.vector(bottom), ymax = as.vector(bottom + height) - gap)
# Thin tiles get a one-line label
tiles$label <- sprintf(ifelse(as.vector(height) < 0.12, "%d  %+.1f", "%d\n%+.1f"), tiles$count,
                       round(tiles$resid, 1) + 0)
columns <- data.frame(x = left + (width - gap) / 2,
                      label = sprintf("%s\n(n = %d)", rownames(obs), rowSums(obs)))
rows <- tiles[tiles$country == rownames(obs)[1], ]          # row labels at the heights of the first column

p <- ggplot(tiles) +
  geom_rect(aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = resid)) +
  geom_text(aes((xmin + xmax) / 2, (ymin + ymax) / 2, label = label,
                colour = abs(resid) > 5), size = 3.1, lineheight = 1, show.legend = FALSE) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "#1D1D1F")) +
  geom_text(data = columns, aes(x, -0.03, label = label), vjust = 1, size = 3.5, colour = "#3A3A3C",
            lineheight = 1) +
  geom_text(data = rows, aes(-0.015, (ymin + ymax) / 2, label = frame), hjust = 1, size = 3.5,
            colour = "#3A3A3C") +
  # red = fewer than expected, blue = more
  scale_fill_gradient2("Pearson\nresidual", low = "#B2182B", mid = "#F7F7F7", high = "#2166AC",
                       midpoint = 0, limits = c(-8, 8), breaks = seq(-8, 8, 4), oob = scales::squish) +
  scale_x_continuous(limits = c(-0.17, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-0.13, 1), expand = c(0, 0)) +
  labs(title = "The mix of refugee frames changes from one country to the next",
       subtitle = sprintf("Pearson χ²(%d) = %.0f, %s · n = %s tweets (simulated) · tile labels: count and residual",
                          chi$parameter, chi$statistic,
                          if (chi$p.value < .001) "p < .001" else sprintf("p = %.3f", chi$p.value),
                          format(sum(obs), big.mark = ",")),
       x = "Country of the tweet", y = "Discourse frame") +
  theme_void(base_size = 11) +
  theme(plot.title = element_text(face = "bold", margin = margin(b = 4)),
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73", margin = margin(b = 6)),
        axis.title.x = element_text(size = 11, hjust = 0.58),
        axis.title.y = element_text(size = 11, angle = 90, hjust = 0.56),
        legend.title = element_text(size = 9.5, colour = "#3A3A3C"),
        legend.key.height = unit(2.2, "lines"),
        plot.margin = margin(8, 8, 8, 8))
print(p)
