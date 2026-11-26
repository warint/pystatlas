# Pairs plot with correlations: the R equivalent of plot.py
# Simulated data, set in the context of Hadengue & Warin (2014) [hadengue2014]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2014)

# Simulated regions: country means differ; within a country, poorer regions grow faster
groups <- c("Brazil", "China", "India")
setup <- list(Brazil = c(27, 9.2, 2.5), China = c(31, 9.0, 9.0), India = c(28, 7.4, 6.0))
df <- do.call(rbind, lapply(groups, function(g) {
  n <- setup[[g]][1]; inc0 <- setup[[g]][2]; gr0 <- setup[[g]][3]
  dev <- rnorm(n, 0, 0.45)                              # region's income gap to its country mean
  spec <- 0.30 - 0.12 * dev + rnorm(n, 0, 0.05)
  data.frame("Growth (% per yr)" = gr0 - 2.2 * dev + rnorm(n, 0, 0.7),
             "Initial income (log)" = inc0 + dev,
             "Specialisation index" = spec,
             "Export diversification" = 3.4 - 4 * spec + rnorm(n, 0, 0.25),
             group = g, check.names = FALSE)
}))
cols <- names(df)[1:4]
palette <- c(Brazil = "#007AFF", China = "#FF9500", India = "#34C759")
stars <- function(p) ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", "")))
lims <- lapply(df[cols], function(v) range(v) + c(-0.05, 0.05) * diff(range(v)))

# GGally is not installed: one facet_grid with free scales, a panel per (row = y, column = x) pair
panel <- function(yv, xv, d) cbind(d, yvar = factor(yv, cols), xvar = factor(xv, cols))
pts <- dens <- txt <- frame <- list()
for (i in seq_along(cols)) for (j in seq_along(cols)) {
  yv <- cols[i]; xv <- cols[j]
  # invisible corners fix every panel to the variables' padded ranges
  frame[[length(frame) + 1]] <- panel(yv, xv, data.frame(x = lims[[xv]], y = lims[[yv]]))
  if (i == j) {                                         # diagonal: density per group, scaled to the row's range
    grid <- seq(lims[[xv]][1], lims[[xv]][2], length.out = 200)
    dd <- do.call(rbind, lapply(groups, function(s) {
      k <- density(df[df$group == s, xv], from = lims[[xv]][1], to = lims[[xv]][2], n = 200)
      data.frame(x = k$x, dens = k$y, group = s)
    }))
    dd$ymin <- lims[[yv]][1]
    dd$y <- lims[[yv]][1] + 0.92 * dd$dens / max(dd$dens) * diff(lims[[yv]])
    dens[[length(dens) + 1]] <- panel(yv, xv, dd)
  } else if (i > j) {                                   # lower: scatter by group
    pts[[length(pts) + 1]] <- panel(yv, xv, data.frame(x = df[[xv]], y = df[[yv]], group = df$group))
  } else {                                              # upper: correlation coefficients
    ct <- cor.test(df[[xv]], df[[yv]])
    rows <- data.frame(label = sprintf("r = %.2f%s", ct$estimate, stars(ct$p.value)), group = "all",
                       pos = 0.78, size = 3.9, face = "bold")
    for (k in seq_along(groups)) {
      s <- groups[k]
      cs <- cor.test(df[df$group == s, xv], df[df$group == s, yv])
      rows <- rbind(rows, data.frame(label = sprintf("%s: %.2f%s", s, cs$estimate, stars(cs$p.value)),
                                     group = s, pos = 0.52 - (k - 1) * 0.17, size = 3.2, face = "plain"))
    }
    rows$x <- mean(lims[[xv]])
    rows$y <- lims[[yv]][1] + rows$pos * diff(lims[[yv]])
    txt[[length(txt) + 1]] <- panel(yv, xv, rows)
  }
}
pts <- do.call(rbind, pts); dens <- do.call(rbind, dens)
txt <- do.call(rbind, txt); frame <- do.call(rbind, frame)

p <- ggplot(mapping = aes(x, y)) +
  geom_blank(data = frame) +
  geom_ribbon(data = dens, aes(ymin = ymin, ymax = y, fill = group), alpha = 0.25) +
  geom_line(data = dens, aes(colour = group), linewidth = 0.6) +
  geom_point(data = pts, aes(colour = group), size = 1.3, alpha = 0.65, stroke = 0) +
  geom_text(data = txt, aes(label = label, colour = group, size = size, fontface = face)) +
  scale_size_identity() +
  scale_colour_manual(values = c(palette, all = "#1D1D1F"), breaks = groups, name = NULL) +
  scale_fill_manual(values = palette, guide = "none") +
  facet_grid(yvar ~ xvar, scales = "free", switch = "both") +
  scale_x_continuous(n.breaks = 4) +
  scale_y_continuous(n.breaks = 4) +
  labs(title = "Within each country, poorer regions grow faster",
       subtitle = sprintf("Pearson r, overall (bold) and within country · n = %d regions (simulated) · * p<.05  ** p<.01  *** p<.001",
                          nrow(df)),
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        panel.grid = element_blank(),
        panel.border = element_rect(fill = NA, colour = "#D1D1D6", linewidth = 0.4),
        panel.spacing = unit(0.5, "lines"),
        strip.placement = "outside",
        strip.text = element_text(size = 9.5, colour = "#1D1D1F"),
        axis.text = element_text(size = 7.5, colour = "#6E6E73"),
        legend.position = "top", legend.justification = "left",
        legend.margin = margin(0, 0, 0, 0)) +
  guides(colour = guide_legend(override.aes = list(size = 2.5, alpha = 1, linewidth = 0, label = "")))
print(p)
