# =============================================================================
# Interactive Distribution Fitting — Step-by-Step in RStudio
#
# Run each section one at a time (Cmd+Enter / Ctrl+Enter) to walk through
# the distribution fitting process for a single metric.
#
# Change METRIC_TABLE and METRIC_COL below to explore different stats.
# =============================================================================

library(DBI)
library(RSQLite)
library(fitdistrplus)
library(MASS)

set.seed(42)

conn <- dbConnect(RSQLite::SQLite(), "data/cbb_prediction.db")


# =============================================================================
# STEP 1: Choose your metric
# =============================================================================
# Change these two lines to explore different metrics.
#
# Team efficiency:  adj_oe, adj_de, adj_tempo, barthag, overall_rank, wins, losses
# Player stats:     ortg, usage, efg_pct, ts_pct, ftr, ft_pct, two_p_pct,
#                   three_p_pct, orb_pct, drb_pct, ast_pct, to_pct, blk_pct,
#                   stl_pct, bpm, obpm, dbpm, pts_pg, ast_pg, treb_pg, adj_oe, adj_de
# Team BPI:         bpi, bpi_offense, bpi_defense, bpi_7day_change,
#                   sor_rank, sos_past_rank, wins, losses, top50_wins, top50_losses
# Games:            home_score, away_score, attendance

METRIC_TABLE <- "team_efficiency"   # table to pull from
METRIC_COL   <- "adj_de"            # column to analyze
WEIGHT_BY_MINUTES <- FALSE          # set TRUE only for player_stats


# =============================================================================
# STEP 2: Load the data
# =============================================================================

if (WEIGHT_BY_MINUTES && METRIC_TABLE == "player_stats") {
  sql <- sprintf(
    "SELECT %s, minutes FROM player_stats
     WHERE %s IS NOT NULL AND minutes IS NOT NULL AND minutes > 0",
    METRIC_COL, METRIC_COL
  )
  df <- dbGetQuery(conn, sql)
  values  <- as.numeric(df[[1]])
  minutes <- as.numeric(df$minutes)

  # Resample proportional to minutes played
  probs   <- minutes / sum(minutes)
  indices <- sample.int(length(values), size = length(values),
                        replace = TRUE, prob = probs)
  data <- values[indices]
  cat("Loaded", length(data), "observations (minutes-weighted resampling)\n")
} else {
  sql <- sprintf("SELECT %s FROM %s WHERE %s IS NOT NULL",
                 METRIC_COL, METRIC_TABLE, METRIC_COL)
  data <- as.numeric(dbGetQuery(conn, sql)[[1]])
  cat("Loaded", length(data), "observations\n")
}

# Drop non-finite values
data <- data[is.finite(data)]


# =============================================================================
# STEP 3: Explore the data
# =============================================================================

summary(data)
cat("Std Dev:", sd(data), "\n")

# Skewness (positive = right tail, negative = left tail)
n <- length(data)
m <- mean(data)
s <- sd(data)
skewness <- (n / ((n - 1) * (n - 2))) * sum(((data - m) / s)^3)
cat("Skewness:", round(skewness, 3), "\n")

# Excess kurtosis (0 = normal, positive = heavy tails)
kurtosis <- mean((data - m)^4) / s^4 - 3
cat("Excess kurtosis:", round(kurtosis, 3), "\n")

hist(data, breaks = 50, main = paste(METRIC_TABLE, "/", METRIC_COL),
      freq = FALSE)


# =============================================================================
# STEP 4: Fit — Normal
# =============================================================================

fit_norm <- fitdist(data, "norm")
summary(fit_norm)
plot(fit_norm)

ks_norm <- ks.test(data, "pnorm",
                   mean = fit_norm$estimate["mean"],
                   sd   = fit_norm$estimate["sd"])
cat("KS stat:", round(ks_norm$statistic, 4),
    " p-value:", round(ks_norm$p.value, 6), "\n")


# =============================================================================
# STEP 5: Fit — Logistic
# =============================================================================

fit_logis <- fitdist(data, "logis")
summary(fit_logis)
plot(fit_logis)

ks_logis <- ks.test(data, "plogis",
                    location = fit_logis$estimate["location"],
                    scale    = fit_logis$estimate["scale"])
cat("KS stat:", round(ks_logis$statistic, 4),
    " p-value:", round(ks_logis$p.value, 6), "\n")


# =============================================================================
# STEP 6: Fit — t (location-scale)
#
# fitdistrplus needs custom d/p/q functions for a 3-parameter t.
# These are defined here so you can see exactly what's happening.
# =============================================================================

# density: shift by loc, scale by scale, evaluate standard t
dt_ls <- function(x, df, loc = 0, scale = 1, log = FALSE) {
  z <- (x - loc) / scale
  if (log) dt(z, df, log = TRUE) - log(scale)
  else     dt(z, df) / scale
}

# CDF
pt_ls <- function(q, df, loc = 0, scale = 1,
                  lower.tail = TRUE, log.p = FALSE) {
  pt((q - loc) / scale, df, lower.tail = lower.tail, log.p = log.p)
}

# quantile function
qt_ls <- function(p, df, loc = 0, scale = 1,
                  lower.tail = TRUE, log.p = FALSE) {
  qt(p, df, lower.tail = lower.tail, log.p = log.p) * scale + loc
}

# Starting values: df=5, loc=median, scale≈MAD
start_t <- list(df = 5, loc = median(data), scale = mad(data, constant = 1.4826))
if (start_t$scale == 0) start_t$scale <- sd(data)

fit_t <- fitdist(data, "t_ls",
                 start = start_t,
                 lower = c(1.01, -Inf, 0.001))
summary(fit_t)
plot(fit_t)

ks_t <- ks.test(data, "pt_ls",
                df    = fit_t$estimate["df"],
                loc   = fit_t$estimate["loc"],
                scale = fit_t$estimate["scale"])
cat("KS stat:", round(ks_t$statistic, 4),
    " p-value:", round(ks_t$p.value, 6), "\n")


# =============================================================================
# STEP 7: Fit — Lognormal  (only works if all data > 0)
# =============================================================================

if (min(data) > 0) {
  fit_lnorm <- fitdist(data, "lnorm")
  summary(fit_lnorm)
  plot(fit_lnorm)

  ks_lnorm <- ks.test(data, "plnorm",
                       meanlog = fit_lnorm$estimate["meanlog"],
                       sdlog   = fit_lnorm$estimate["sdlog"])
  cat("KS stat:", round(ks_lnorm$statistic, 4),
      " p-value:", round(ks_lnorm$p.value, 6), "\n")
} else {
  cat("Skipping lognormal — data contains non-positive values\n")
}


# =============================================================================
# STEP 8: Fit — Gamma  (only works if all data > 0)
# =============================================================================

if (min(data) > 0) {
  fit_gamma <- fitdist(data, "gamma")
  print(summary(fit_gamma))
  plot(fit_gamma)

  ks_gamma <- ks.test(data, "pgamma",
                       shape = fit_gamma$estimate["shape"],
                       rate  = fit_gamma$estimate["rate"])
  cat("KS stat:", round(ks_gamma$statistic, 4),
      " p-value:", round(ks_gamma$p.value, 6), "\n")
} else {
  cat("Skipping gamma — data contains non-positive values\n")
}


# =============================================================================
# STEP 9: Fit — Weibull  (only works if all data > 0)
# =============================================================================

if (min(data) > 0) {
  fit_weibull <- fitdist(data, "weibull")
  summary(fit_weibull)
  plot(fit_weibull)

  ks_weibull <- ks.test(data, "pweibull",
                         shape = fit_weibull$estimate["shape"],
                         scale = fit_weibull$estimate["scale"])
  cat("KS stat:", round(ks_weibull$statistic, 4),
      " p-value:", round(ks_weibull$p.value, 6), "\n")
} else {
  cat("Skipping weibull — data contains non-positive values\n")
}


# =============================================================================
# STEP 10: Fit — Exponential  (only works if all data > 0)
# =============================================================================

if (min(data) > 0) {
  fit_exp <- fitdist(data, "exp")
  summary(fit_exp)
  plot(fit_exp)

  ks_exp <- ks.test(data, "pexp", rate = fit_exp$estimate["rate"])
  cat("KS stat:", round(ks_exp$statistic, 4),
      " p-value:", round(ks_exp$p.value, 6), "\n")
} else {
  cat("Skipping exponential — data contains non-positive values\n")
}


# =============================================================================
# STEP 11: Fit — Beta  (only works if all data in (0, 1))
# =============================================================================

if (min(data) > 0 && max(data) < 1) {
  fit_beta <- fitdist(data, "beta")
  summary(fit_beta)
  plot(fit_beta)

  ks_beta <- ks.test(data, "pbeta",
                      shape1 = fit_beta$estimate["shape1"],
                      shape2 = fit_beta$estimate["shape2"])
  cat("KS stat:", round(ks_beta$statistic, 4),
      " p-value:", round(ks_beta$p.value, 6), "\n")
} else {
  cat("Skipping beta — data not in (0, 1)\n")
}


# =============================================================================
# STEP 12: Compare all fits
#
# Lower AIC = better fit (penalizes extra parameters).
# KS p-value > 0.05 means we can't reject the distribution at 95% confidence.
# Note: KS p-values are often ~0 for large samples — that's normal.
#       Use AIC for ranking and KS stat magnitude for practical fit quality.
# =============================================================================

# Collect everything that was successfully fit
comparison <- data.frame(
  distribution = character(),
  aic          = numeric(),
  ks_stat      = numeric(),
  p_value      = numeric(),
  stringsAsFactors = FALSE
)

add_row <- function(df, name, fit_obj, ks_obj) {
  rbind(df, data.frame(
    distribution = name,
    aic          = fit_obj$aic,
    ks_stat      = unname(ks_obj$statistic),
    p_value      = ks_obj$p.value,
    stringsAsFactors = FALSE
  ))
}

# Always available
comparison <- add_row(comparison, "normal",   fit_norm,  ks_norm)
comparison <- add_row(comparison, "logistic", fit_logis, ks_logis)
comparison <- add_row(comparison, "t",        fit_t,     ks_t)

if (exists("fit_lnorm"))   comparison <- add_row(comparison, "lognormal",   fit_lnorm,   ks_lnorm)
if (exists("fit_gamma"))   comparison <- add_row(comparison, "gamma",       fit_gamma,    ks_gamma)
if (exists("fit_weibull")) comparison <- add_row(comparison, "weibull",     fit_weibull,  ks_weibull)
if (exists("fit_exp"))     comparison <- add_row(comparison, "exponential", fit_exp,      ks_exp)
if (exists("fit_beta"))    comparison <- add_row(comparison, "beta",        fit_beta,     ks_beta)

comparison <- comparison[order(comparison$aic), ]
comparison$aic_delta <- comparison$aic - comparison$aic[1]

cat("\n")
cat(strrep("=", 65), "\n")
cat(sprintf("  %s / %s  —  Comparison (sorted by AIC)\n", METRIC_TABLE, METRIC_COL))
cat(strrep("=", 65), "\n")
print(comparison, row.names = FALSE, digits = 4)
cat("\nBest fit:", comparison$distribution[1], "\n")


# =============================================================================
# STEP 13: Visual comparison — overlay best fit on histogram
# =============================================================================

best_name <- comparison$distribution[1]

hist(data, breaks = 50, freq = FALSE,
     main = paste(METRIC_COL, "—", best_name, "fit"),
     col = "grey85", border = "white",
     xlab = METRIC_COL)

x_seq <- seq(min(data), max(data), length.out = 500)

if (best_name == "normal") {
  lines(x_seq, dnorm(x_seq, fit_norm$estimate["mean"], fit_norm$estimate["sd"]),
        col = "red", lwd = 2)
} else if (best_name == "logistic") {
  lines(x_seq, dlogis(x_seq, fit_logis$estimate["location"], fit_logis$estimate["scale"]),
        col = "red", lwd = 2)
} else if (best_name == "t") {
  lines(x_seq, dt_ls(x_seq, fit_t$estimate["df"], fit_t$estimate["loc"], fit_t$estimate["scale"]),
        col = "red", lwd = 2)
} else if (best_name == "lognormal") {
  lines(x_seq, dlnorm(x_seq, fit_lnorm$estimate["meanlog"], fit_lnorm$estimate["sdlog"]),
        col = "red", lwd = 2)
} else if (best_name == "gamma") {
  lines(x_seq, dgamma(x_seq, fit_gamma$estimate["shape"], fit_gamma$estimate["rate"]),
        col = "red", lwd = 2)
} else if (best_name == "weibull") {
  lines(x_seq, dweibull(x_seq, fit_weibull$estimate["shape"], fit_weibull$estimate["scale"]),
        col = "red", lwd = 2)
} else if (best_name == "exponential") {
  lines(x_seq, dexp(x_seq, fit_exp$estimate["rate"]),
        col = "red", lwd = 2)
} else if (best_name == "beta") {
  lines(x_seq, dbeta(x_seq, fit_beta$estimate["shape1"], fit_beta$estimate["shape2"]),
        col = "red", lwd = 2)
}

legend("topright", legend = paste("Best:", best_name), col = "red", lwd = 2)


# =============================================================================
# STEP 14: Q-Q plot for best fit
#
# Points falling on the diagonal line = good fit.
# Deviations in the tails show where the distribution breaks down.
# =============================================================================

n_qq    <- length(data)
p_seq   <- (1:n_qq - 0.5) / n_qq
sorted  <- sort(data)

if (best_name == "normal") {
  theoretical <- qnorm(p_seq, fit_norm$estimate["mean"], fit_norm$estimate["sd"])
} else if (best_name == "logistic") {
  theoretical <- qlogis(p_seq, fit_logis$estimate["location"], fit_logis$estimate["scale"])
} else if (best_name == "t") {
  theoretical <- qt_ls(p_seq, fit_t$estimate["df"], fit_t$estimate["loc"], fit_t$estimate["scale"])
} else if (best_name == "lognormal") {
  theoretical <- qlnorm(p_seq, fit_lnorm$estimate["meanlog"], fit_lnorm$estimate["sdlog"])
} else if (best_name == "gamma") {
  theoretical <- qgamma(p_seq, fit_gamma$estimate["shape"], fit_gamma$estimate["rate"])
} else if (best_name == "weibull") {
  theoretical <- qweibull(p_seq, fit_weibull$estimate["shape"], fit_weibull$estimate["scale"])
} else if (best_name == "exponential") {
  theoretical <- qexp(p_seq, fit_exp$estimate["rate"])
} else if (best_name == "beta") {
  theoretical <- qbeta(p_seq, fit_beta$estimate["shape1"], fit_beta$estimate["shape2"])
}

plot(theoretical, sorted,
     main = paste("Q-Q Plot:", METRIC_COL, "vs", best_name),
     xlab = "Theoretical quantiles", ylab = "Sample quantiles",
     pch = 16, cex = 0.4, col = "steelblue")
abline(0, 1, col = "red", lwd = 2)


# =============================================================================
# Cleanup — run when done exploring
# =============================================================================
# dbDisconnect(conn)
