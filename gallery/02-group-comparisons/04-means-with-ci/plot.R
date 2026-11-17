# Interaction plot: the R equivalent of plot.py
# Simulated data, set in the context of de Marcellis-Warin et al. (2022) [dmw2022manip]
# R and NumPy draw different random numbers, so the values differ slightly from the Python figure.
library(ggplot2)

set.seed(2022)

# Data: susceptibility score (1–7) by age group and digital literacy, 60 consumers per cell
classes <- c("18–29", "30–44", "45–59", "60+")
d <- expand.grid(rep = 1:60, literacy = c("High", "Low"), age = 0:3)
low <- d$literacy == "Low"
d$score <- pmin(pmax(3.4 + 0.15 * d$age + low * (0.3 + 0.35 * d$age) + rnorm(nrow(d), 0, 1.0), 1), 7)
d$age_group <- factor(classes[d$age + 1], levels = classes)

# Two-way ANOVA with interaction + partial eta squared
# (the design is balanced, so these sequential sums of squares equal the type II ones)
aov_table <- anova(lm(score ~ age_group * literacy, data = d))
ss_res <- aov_table["Residuals", "Sum Sq"]
df_res <- aov_table["Residuals", "Df"]
term <- function(row, label) {                  # one "F(df1, df2) = …, p …, η²p = …" string per term
  r <- aov_table[row, ]
  pval <- if (r[["Pr(>F)"]] < .001) "< .001" else sprintf("= %.2f", r[["Pr(>F)"]])
  sprintf("%s F(%d, %d) = %.1f, p %s, η²p = %.2f", label, r$Df, df_res, r[["F value"]], pval,
          r[["Sum Sq"]] / (r[["Sum Sq"]] + ss_res))
}

# Means and t-based 95% CIs per cell
cells <- aggregate(score ~ age_group + literacy, data = d, FUN = function(v)
  c(mean = mean(v), half = qt(0.975, length(v) - 1) * sd(v) / sqrt(length(v))))
cells <- cbind(cells[c("age_group", "literacy")], cells$score)
gap <- cells$mean[cells$literacy == "Low"] - cells$mean[cells$literacy == "High"]

dodge <- position_dodge(width = 0.12)
p <- ggplot(cells, aes(age_group, mean, colour = literacy, group = literacy)) +
  geom_line(linewidth = 0.9, position = dodge) +
  geom_linerange(aes(ymin = mean - half, ymax = mean + half), linewidth = 0.9, position = dodge) +
  geom_point(aes(fill = literacy), shape = 21, size = 3.6, colour = "white", stroke = 1,
             position = dodge) +
  geom_text(data = cells[cells$age_group == "60+", ], aes(label = paste(literacy, "digital\nliteracy")),
            hjust = 0, nudge_x = 0.16, fontface = "bold", size = 3.6, lineheight = 0.95) +
  scale_colour_manual(values = c(High = "#007AFF", Low = "#FF3B30"), aesthetics = c("colour", "fill"),
                      guide = "none") +
  scale_x_discrete(expand = expansion(add = c(0.3, 0.7))) +
  labs(title = sprintf("The literacy gap in susceptibility widens with age, from %.1f to %.1f points",
                       gap[1], gap[4]),
       subtitle = sprintf("Two-way ANOVA (type II), n = %d consumers (simulated) · %s\n%s · %s",
                          nrow(d), term("literacy", "literacy"), term("age_group", "age"),
                          term("age_group:literacy", "age × literacy")),
       x = "Age group (60 consumers per literacy level in each group)",
       y = "Susceptibility to manipulation\n(1–7 score; mean ± 95% CI)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.subtitle = element_text(size = 8.5, colour = "#6E6E73", lineheight = 1.15),
        panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
print(p)
