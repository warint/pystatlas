# Correlogram with clustering: the R equivalent of plot.py
# Simulated data, set in the context of Dedinec, Stojkov, Warin & Kocarev (2026) [dedinec2026]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2026)

# Simulated countries: 12 indicators driven by three correlated latent factors
n <- 150
income <- rnorm(n)
human <- 0.35 * income + 0.94 * rnorm(n)
green <- -0.5 * income + 0.85 * rnorm(n)
loads <- list("GDP per head (log)" = list(income, .9), "Energy use per head" = list(income, .8),
              "CO\u2082 per head" = list(income, .75), "Urban population" = list(income, .65),
              "Internet users" = list(income, .7), "Life expectancy" = list(human, .85),
              "Years of schooling" = list(human, .8), "Electricity access" = list(human, .7),
              "Child survival" = list(human, .8), "Renewables share" = list(green, .8),
              "Agriculture (% GDP)" = list(green, .7), "Forest cover" = list(green, .55))
df <- as.data.frame(lapply(loads, function(l) l[[2]] * l[[1]] + sqrt(1 - l[[2]]^2) * rnorm(n)),
                    check.names = FALSE)
corr <- cor(df)

# Hierarchical clustering on the distance 1 - r, then reorder the matrix
# (corrplot is not installed: hclust() + cutree() do what order = "hclust", addrect = 3 do)
hc <- hclust(as.dist(1 - corr), method = "complete")
corr <- corr[hc$order, hc$order]
clusters <- cutree(hc, k = 3)[hc$order]
k <- nrow(corr)

# One circle per off-diagonal cell, drawn as a polygon so its radius is in data units
theta <- seq(0, 2 * pi, length.out = 60)
cells <- expand.grid(i = seq_len(k), j = seq_len(k))
cells <- cells[cells$i != cells$j, ]
cells$r <- corr[cbind(cells$i, cells$j)]
circles <- do.call(rbind, lapply(seq_len(nrow(cells)), function(m) {
  rad <- 0.47 * sqrt(abs(cells$r[m]))
  data.frame(id = m, r = cells$r[m], x = cells$j[m] + rad * cos(theta), y = cells$i[m] + rad * sin(theta))
}))

# Boxes around the clusters (consecutive runs along the diagonal)
runs <- rle(clusters)
ends <- cumsum(runs$lengths)
boxes <- data.frame(lo = ends - runs$lengths + 0.5, hi = ends + 0.5)

r_ge <- corr["GDP per head (log)", "Energy use per head"]
p <- ggplot() +
  geom_hline(yintercept = seq(0.5, k + 0.5), colour = "#E5E5EA", linewidth = 0.4) +
  geom_vline(xintercept = seq(0.5, k + 0.5), colour = "#E5E5EA", linewidth = 0.4) +
  geom_polygon(data = circles, aes(x, y, group = id, fill = r)) +
  geom_rect(data = boxes, aes(xmin = lo, xmax = hi, ymin = lo, ymax = hi), fill = NA,
            colour = "#1D1D1F", linewidth = 0.8) +
  scale_fill_distiller("Pearson r", palette = "RdBu", direction = 1, limits = c(-1, 1),
                       breaks = seq(-1, 1, 0.5)) +
  scale_x_continuous(breaks = seq_len(k), labels = colnames(corr), expand = c(0, 0)) +
  scale_y_reverse(breaks = seq_len(k), labels = rownames(corr), expand = c(0, 0)) +
  coord_equal(xlim = c(0.5, k + 0.5), ylim = c(k + 0.5, 0.5)) +
  labs(title = sprintf("Income, energy and emissions form one block (r = %.2f, GDP–energy)", r_ge),
       subtitle = sprintf("Pearson r · n = %d countries (simulated) · complete linkage clustering on 1 − r · circle area grows with |r|",
                          n),
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text = element_text(colour = "#1D1D1F", size = 9.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank(),
        legend.title = element_text(size = 9.5, colour = "#3A3A3C"),
        legend.key.height = unit(2.2, "lines"))
print(p)
