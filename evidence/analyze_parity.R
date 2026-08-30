# Oligo engine parity analysis: reproduces the parity-report tables from
# evidence/oligo_engine_parity.csv.
#
#   Rscript evidence/analyze_parity.R
#
# Produces:
#   1. NA counts per method/metric (hairpin Tm NA rate is the headline number)
#   2. Distribution of differences vs IDT (violin + boxplot, per metric)
#   3. Duplex Tm slope vs IDT (per-method linear fit; the shared slope is an
#      IDT-side convention, the per-method offsets/slope excess are ours)
#   4. RMSE / mean abs diff table vs IDT
#
# dG convention reminder: strider/primer3/vienna report raw dG25 including the
# bimolecular initiation term, IDT reports structure-only dG, so a ~2
# kcal/mol offset on the dG panels is expected.

library(dplyr)
library(tidyr)
library(ggplot2)

dt <- read.csv("evidence/oligo_engine_parity.csv")

methods <- c("strider_m", "strider_sl", "primer3", "vienna")
metrics <- c("duplex_tm", "hairpin_dg", "hairpin_tm", "homodimer_dg")

# 1. NA counts per method/metric ------------------------------------------

cat("\n== NA counts (of", nrow(dt), "oligos) ==\n")
na_counts <- sapply(
  paste0(rep(methods, each = length(metrics)), "_", metrics),
  function(col) sum(is.na(dt[[col]]))
)
print(na_counts)

# 2. Differences vs IDT (long format) --------------------------------------

dt_long <- dt %>%
  pivot_longer(
    cols = -seq,
    names_to = c("method", "metric"),
    names_pattern = "(.*)_(duplex_tm|hairpin_dg|hairpin_tm|homodimer_dg)"
  ) %>%
  filter(method != "idt") %>%
  left_join(
    dt %>%
      pivot_longer(
        cols = starts_with("idt_"),
        names_prefix = "idt_",
        names_to = "metric",
        values_to = "idt_value"
      ),
    by = c("seq", "metric")
  ) %>%
  mutate(difference = value - idt_value)

dt_clean <- dt_long %>% filter(is.finite(difference))

ggplot(dt_clean, aes(x = method, y = difference, fill = method)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.1, fill = "white") +
  facet_wrap(~metric, scales = "free_y") +
  theme_minimal() +
  labs(title = "Distribution of Differences (vs IDT)",
       y = "Difference", x = "Method")

# 3. Duplex Tm slope vs IDT -------------------------------------------------

cat("\n== Duplex Tm difference vs IDT: linear fit ==\n")
for (m in methods) {
  ok <- is.finite(dt[[paste0(m, "_duplex_tm")]]) & is.finite(dt$idt_duplex_tm)
  fit <- lm(dt[[paste0(m, "_duplex_tm")]][ok] - dt$idt_duplex_tm[ok]
            ~ dt$idt_duplex_tm[ok])
  slope <- coef(fit)[2]
  intercept <- coef(fit)[1]
  cat(sprintf("%-10s n=%5d  slope=%+.3f C per IDT C  intercept=%+.2f  mean_diff=%+.2f\n",
              m, sum(ok), slope, intercept,
              mean(dt[[paste0(m, "_duplex_tm")]][ok] - dt$idt_duplex_tm[ok])))
}

# 4. RMSE / mean abs diff vs IDT --------------------------------------------

cat("\n== Error vs IDT by method and metric ==\n")
rmse_results <- dt_long %>%
  filter(!is.na(difference)) %>%
  group_by(method, metric) %>%
  summarise(
    n = n(),
    mean_diff = mean(difference),
    mean_abs_diff = mean(abs(difference)),
    rmse = sqrt(mean(difference^2)),
    .groups = "drop"
  ) %>%
  arrange(metric, rmse)

print(as.data.frame(rmse_results), row.names = FALSE)
