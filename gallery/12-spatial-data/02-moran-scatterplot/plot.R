# Moran scatterplot (global autocorrelation): the R equivalent of plot.py
# Simulated data, set in the context of the CIRANO GVCdtLab [gvcdtlab]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(sf)
library(spdep)
library(ggplot2)

set.seed(12)

# Schematic corridor: hexagonal zones within reach of the waterway, Duluth and Chicago to the Gulf
main <- rbind(c(-92.1, 46.8), c(-89.2, 48.4), c(-84.4, 46.5), c(-82.4, 44.0), c(-83.0, 42.3),
              c(-81.2, 42.2), c(-79.0, 42.9), c(-79.4, 43.6), c(-76.5, 44.2), c(-73.6, 45.5),
              c(-71.2, 46.8), c(-68.5, 48.5), c(-64.5, 49.2))
michigan <- rbind(c(-84.0, 45.6), c(-86.6, 45.4), c(-87.0, 43.5), c(-87.6, 41.9))
k <- cos(pi / 4)                                # shrink longitudes so the hexagons are regular
shrink <- function(m) cbind(m[, 1] * k, m[, 2])
water <- st_sfc(st_multilinestring(list(shrink(main), shrink(michigan))))

r <- 0.3
centres <- expand.grid(row = 0:35, col = 0:89)
centres$cx <- -66.5 + 1.5 * r * centres$col
centres$cy <- 40.5 + sqrt(3) * r * (centres$row + (centres$col %% 2) / 2)
near <- as.numeric(st_distance(st_as_sf(centres, coords = c("cx", "cy")), water)) < 0.95
centres <- centres[near, ]
hexagon <- function(cx, cy) {
  a <- seq(0, 360, 60) * pi / 180
  st_polygon(list(round(cbind(cx + r * cos(a), cy + r * sin(a)), 6)))
}
g <- st_sf(geometry = st_sfc(Map(hexagon, centres$cx, centres$cy)))
n <- nrow(g)
lon <- centres$cx / k                           # back to longitude, latitude
lat <- centres$cy

# Queen contiguity, row-standardised: the spatial lag is the average of the neighbours
w <- nb2listw(poly2nb(g, queen = TRUE), style = "W")

# Exposure to low water: high at shallow passages (Seaway, connecting channels, upper lakes),
# low in the deep tidal estuary, plus spatially smooth noise and one draft-limited harbour
bump <- function(lon0, lat0, s) exp(-((lon - lon0)^2 + (lat - lat0)^2) / (2 * s^2))
noise <- rnorm(n)
for (i in 1:2) noise <- lag.listw(w, noise)     # smooth the noise over neighbours
exposure <- 35 + 30 * bump(-74.8, 45.0, 1.6) + 22 * bump(-82.7, 42.7, 1.2) +
  20 * bump(-90.3, 47.6, 1.8) - 25 * bump(-67.0, 48.8, 2.2) + 12 * noise + rnorm(n, 0, 4)
exposure[which.min(sqrt((lon + 66.0)^2 + (lat - 49.0)^2))] <- 75      # outlier harbour
y <- round(pmin(pmax(exposure, 0), 100), 1)

# Moran's I with 999 permutations
mc <- moran.mc(y, w, nsim = 999)
I <- unname(mc$statistic)
sims <- mc$res[1:999]                           # I on the 999 shuffled maps
z <- (y - mean(y)) / sqrt(mean((y - mean(y))^2))
d <- data.frame(z = z, lag = lag.listw(w, z))   # lag: average of the neighbours' z-scores
d$quadrant <- ifelse(d$z > 0 & d$lag > 0, "high", ifelse(d$z < 0 & d$lag < 0, "low", "other"))

# Inset: reference distribution of I under spatial randomness
inset <- ggplot(data.frame(I = sims), aes(I)) +
  geom_histogram(bins = 30, fill = "#C7C7CC", colour = "white") +
  geom_vline(xintercept = I, colour = "#FF3B30", linewidth = 0.9) +
  labs(title = "999 random maps", x = NULL, y = NULL) +
  theme_minimal(base_size = 8) +
  theme(panel.grid = element_blank(), axis.text.y = element_blank(),
        plot.title = element_text(size = 8.5),
        plot.background = element_rect(fill = "white", colour = NA))

ylim <- c(-2.7, 2.8)
at <- function(lim, share) lim[1] + share * diff(lim)      # position as a share of an axis
p <- ggplot(d, aes(z, lag)) +
  geom_hline(yintercept = 0, colour = "#C7C7CC") +
  geom_vline(xintercept = 0, colour = "#C7C7CC") +
  geom_point(aes(colour = quadrant), size = 2.2, alpha = 0.75, show.legend = FALSE) +
  scale_colour_manual(values = c(high = "#FF3B30", low = "#007AFF", other = "#8E8E93")) +
  geom_abline(intercept = 0, slope = I, colour = "#1D1D1F", linewidth = 0.9) +
  annotate("text", x = c(Inf, -Inf, -Inf, Inf), y = c(Inf, -Inf, Inf, -Inf),
           label = c("high, among high", "low, among low", "low, among high", "high, among low"),
           hjust = c(1.1, -0.1, -0.1, 1.1), vjust = c(1.6, -0.8, 1.6, -0.8),
           size = 3.2, colour = "#6E6E73") +
  annotate("text", x = Inf, y = at(ylim, 0.22), hjust = 1.05, size = 3.5,
           label = sprintf("slope = Moran's I = %.2f", I)) +
  annotation_custom(ggplotGrob(inset), xmin = at(range(z), 0.04), xmax = at(range(z), 0.38),
                    ymin = at(ylim, 0.58), ymax = at(ylim, 0.90)) +
  coord_cartesian(ylim = ylim) +
  labs(title = sprintf("Exposed zones sit next to exposed zones (Moran's I = %.2f)", I),
       subtitle = sprintf(paste("Permutation p = %.3f · E[I] = %.3f under randomness ·",
                                "queen weights, row-standardised · n = %d corridor zones (simulated)"),
                          mc$p.value, -1 / (n - 1), n),
       x = "Exposure to low water, standardised (z)",
       y = "Neighbours' average exposure (spatial lag of z)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        panel.grid.minor = element_blank())
print(p)
