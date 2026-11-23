# Proportions with confidence intervals: the R equivalent of plot.py
# Simulated data, set in the context of Kouloukoui, de Marcellis-Warin & Warin (2025) [kouloukoui2025climate]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(88)

# Simulated coding: 240 firms × 7 plan elements (1 = included), linked by a firm-level ambition
elements <- c("Emission-reduction targets" = 1.3, "Board oversight of climate" = 0.7,
              "Interim (2030) milestones" = 0.2, "Capex aligned with the plan" = -0.6,
              "Scope 3 emissions covered" = -0.9, "Executive pay linked to climate" = -1.3,
              "Just-transition measures" = -1.9)
n <- 240
ambition <- rnorm(n)
plans <- sapply(elements, function(b) as.integer(b + ambition + rnorm(n) > 0))

# Proportion + Wilson 95% CI per element (prop.test without continuity correction), sorted
g <- data.frame(element = names(elements), k = colSums(plans), n = n)
g$p <- g$k / g$n
ci <- t(mapply(function(k, n) prop.test(k, n, correct = FALSE)$conf.int, g$k, g$n))
g$low <- ci[, 1]
g$high <- ci[, 2]
g$element <- reorder(g$element, g$p)

# Cochran's Q: do the seven elements appear equally often in the same firms' plans?
m <- ncol(plans)
Q <- (m - 1) * (m * sum(colSums(plans)^2) - sum(plans)^2) / (m * sum(plans) - sum(rowSums(plans)^2))
p_q <- pchisq(Q, m - 1, lower.tail = FALSE)
overall <- mean(g$p)

p <- ggplot(g, aes(p, element)) +
  geom_vline(xintercept = overall, colour = "#8E8E93", linewidth = 0.45, linetype = "dashed") +
  annotate("text", x = overall, y = m + 0.62, label = sprintf(" average %.0f%%", 100 * overall),
           hjust = 0, size = 3.1, colour = "#6E6E73") +
  geom_linerange(aes(xmin = low, xmax = high), colour = "#007AFF", linewidth = 1.2, alpha = 0.45) +
  geom_point(shape = 21, size = 3.8, fill = "#007AFF", colour = "white", stroke = 1) +
  geom_text(aes(x = 1.04, label = sprintf("%.0f%%", 100 * p)), hjust = 0, size = 3.5,
            fontface = "bold", colour = "#007AFF") +
  geom_text(aes(x = 1.12, label = sprintf("%d / %d", k, n)), hjust = 0, size = 3.3,
            colour = "#6E6E73") +
  scale_x_continuous(breaks = seq(0, 1, 0.2), labels = scales::percent, limits = c(0, 1.26),
                     expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = c(0.6, 0.9))) +
  labs(title = "Targets are common in transition plans; just-transition measures are rare",
       subtitle = sprintf("Cochran's Q(%d) = %.0f, %s for equal shares · n = %d firms (simulated)",
                          m - 1, Q, if (p_q < .001) "p < .001" else sprintf("p = %.3f", p_q), n),
       x = "Firms whose transition plan includes the element (Wilson 95% CI)", y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73"),
        axis.text.y = element_text(colour = "#1D1D1F", size = 10),
        panel.grid.minor = element_blank(), panel.grid.major.y = element_blank())
print(p)
