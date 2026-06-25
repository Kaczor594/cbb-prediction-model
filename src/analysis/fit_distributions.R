#!/usr/bin/env Rscript
# =============================================================================
# Distribution Fitting Analysis for CBB Prediction Model
#
# Tests key metrics against common statistical distributions using MLE fitting,
# KS goodness-of-fit tests, and AIC for model comparison.
#
# Player-level stats are weighted by minutes played via resampling: each
# player's observation is replicated proportionally to their share of total
# minutes, producing a weighted sample that is then fit normally.
#
# Usage:
#   Rscript src/analysis/fit_distributions.R
#   Rscript src/analysis/fit_distributions.R --verbose
#   Rscript src/analysis/fit_distributions.R --table team_efficiency
# =============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(RSQLite)
  library(fitdistrplus)
  library(MASS)
})

set.seed(42)

# ── CLI args ─────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
VERBOSE    <- "--verbose" %in% args
TABLE_FILTER <- NULL
for (i in seq_along(args)) {
  if (args[i] == "--table" && i < length(args)) {
    TABLE_FILTER <- args[i + 1]
  }
}

DB_PATH <- "data/cbb_prediction.db"

# ── Distribution candidates ──────────────────────────────────────────────────

# Each entry: list(name, fitdistrplus_name, requires_positive, requires_unit)
DISTRIBUTIONS <- list(
  list(name = "normal",      fname = "norm",    pos = FALSE, unit = FALSE),
  list(name = "lognormal",   fname = "lnorm",   pos = TRUE,  unit = FALSE),
  list(name = "gamma",       fname = "gamma",   pos = TRUE,  unit = FALSE),
  list(name = "beta",        fname = "beta",    pos = FALSE, unit = TRUE),
  list(name = "exponential", fname = "exp",     pos = TRUE,  unit = FALSE),
  list(name = "weibull",     fname = "weibull", pos = TRUE,  unit = FALSE),
  list(name = "logistic",    fname = "logis",   pos = FALSE, unit = FALSE),
  list(name = "t",           fname = "t",       pos = FALSE, unit = FALSE)
)

# ── Custom t-distribution support for fitdistrplus ───────────────────────────
# fitdistrplus needs dt/pt/qt functions with a single "df" shape parameter
# plus loc/scale. We define a 3-parameter t (df, loc, scale).

dt_ls <- function(x, df, loc = 0, scale = 1, log = FALSE) {
  z <- (x - loc) / scale
  if (log) {
    dt(z, df, log = TRUE) - log(scale)
  } else {
    dt(z, df) / scale
  }
}

pt_ls <- function(q, df, loc = 0, scale = 1, lower.tail = TRUE, log.p = FALSE) {
  pt((q - loc) / scale, df, lower.tail = lower.tail, log.p = log.p)
}

qt_ls <- function(p, df, loc = 0, scale = 1, lower.tail = TRUE, log.p = FALSE) {
  qt(p, df, lower.tail = lower.tail, log.p = log.p) * scale + loc
}

# ── Core fitting logic ──────────────────────────────────────────────────────

fit_one_distribution <- function(data, dist_info) {
  # Returns NULL on failure, otherwise a list with fit results
  tryCatch({
    dname  <- dist_info$fname
    dlabel <- dist_info$name

    # Use custom t distribution
    if (dlabel == "t") {
      med  <- median(data)
      sc   <- mad(data, constant = 1.4826)
      if (sc == 0) sc <- sd(data)
      start <- list(df = 5, loc = med, scale = sc)
      lower <- c(1.01, -Inf, 0.001)

      fit <- fitdist(data, "t_ls",
                     start = start, lower = lower,
                     method = "mle", keepdata = FALSE)

      params <- as.list(fit$estimate)
      ks     <- suppressWarnings(
        ks.test(data, "pt_ls",
                df = params$df, loc = params$loc, scale = params$scale)
      )

      return(list(
        name       = dlabel,
        aic        = fit$aic,
        ks_stat    = unname(ks$statistic),
        p_value    = ks$p.value,
        params     = params,
        loglik     = fit$loglik,
        fit_obj    = fit
      ))
    }

    # Standard distributions via fitdistrplus
    fit <- fitdist(data, dname, method = "mle", keepdata = FALSE)

    # KS test against fitted CDF
    params <- as.list(fit$estimate)
    pfun   <- match.fun(paste0("p", dname))
    ks     <- suppressWarnings(
      do.call(ks.test, c(list(x = data), pfun, params))
    )

    list(
      name       = dlabel,
      aic        = fit$aic,
      ks_stat    = unname(ks$statistic),
      p_value    = ks$p.value,
      params     = params,
      loglik     = fit$loglik,
      fit_obj    = fit
    )
  }, error = function(e) NULL)
}


fit_all_distributions <- function(data, label = "") {
  # Remove non-finite values
  data <- data[is.finite(data)]
  if (length(data) < 30) {
    return(list(results = list(), data = data))
  }

  data_min <- min(data)
  data_max <- max(data)

  results <- list()

  for (dist_info in DISTRIBUTIONS) {
    # Skip incompatible distributions
    if (dist_info$pos && data_min <= 0) next
    if (dist_info$unit && (data_min <= 0 || data_max >= 1)) next

    res <- fit_one_distribution(data, dist_info)
    if (!is.null(res)) {
      results[[length(results) + 1]] <- res
    }
  }

  # Sort by AIC
  if (length(results) > 0) {
    aics <- sapply(results, function(r) r$aic)
    results <- results[order(aics)]
  }

  list(results = results, data = data)
}


confidence_label <- function(p_value) {
  if (is.na(p_value)) return("  N/A")
  if (p_value >= 0.10) return(" HIGH")
  if (p_value >= 0.01) return("  MED")
  return("  LOW")
}


# ── Data loading ─────────────────────────────────────────────────────────────

load_team_efficiency <- function(conn) {
  metrics <- c("adj_oe", "adj_de", "adj_tempo", "barthag",
               "overall_rank", "wins", "losses")
  result <- list()
  for (col in metrics) {
    sql <- sprintf("SELECT %s FROM team_efficiency WHERE %s IS NOT NULL", col, col)
    vals <- dbGetQuery(conn, sql)[[1]]
    if (length(vals) >= 30) result[[col]] <- as.numeric(vals)
  }
  result
}


load_player_stats_weighted <- function(conn) {
  metrics <- c("ortg", "usage", "efg_pct", "ts_pct", "ftr",
               "ft_pct", "two_p_pct", "three_p_pct",
               "orb_pct", "drb_pct", "ast_pct", "to_pct", "blk_pct", "stl_pct",
               "bpm", "obpm", "dbpm",
               "pts_pg", "ast_pg", "treb_pg",
               "adj_oe", "adj_de")

  result <- list()

  for (col in metrics) {
    sql <- sprintf(
      "SELECT %s, minutes FROM player_stats
       WHERE %s IS NOT NULL AND minutes IS NOT NULL AND minutes > 0",
      col, col
    )
    df <- dbGetQuery(conn, sql)
    if (nrow(df) < 30) next

    values  <- as.numeric(df[[1]])
    minutes <- as.numeric(df$minutes)

    # Weighted resampling: resample proportional to minutes share
    probs   <- minutes / sum(minutes)
    indices <- sample.int(length(values), size = length(values),
                          replace = TRUE, prob = probs)
    result[[col]] <- values[indices]
  }
  result
}


load_team_bpi <- function(conn) {
  metrics <- c("bpi", "bpi_offense", "bpi_defense", "bpi_7day_change",
               "sor_rank", "sos_past_rank",
               "wins", "losses", "top50_wins", "top50_losses")
  result <- list()
  for (col in metrics) {
    sql <- sprintf("SELECT %s FROM team_bpi WHERE %s IS NOT NULL", col, col)
    vals <- dbGetQuery(conn, sql)[[1]]
    if (length(vals) >= 30) result[[col]] <- as.numeric(vals)
  }
  result
}


load_game_metrics <- function(conn) {
  result <- list()

  for (col in c("home_score", "away_score", "attendance")) {
    sql <- sprintf("SELECT %s FROM games WHERE %s IS NOT NULL", col, col)
    vals <- dbGetQuery(conn, sql)[[1]]
    if (length(vals) >= 30) result[[col]] <- as.numeric(vals)
  }

  # Derived metrics
  df <- dbGetQuery(conn,
    "SELECT home_score, away_score FROM games
     WHERE home_score IS NOT NULL AND away_score IS NOT NULL")
  if (nrow(df) > 0) {
    result[["score_differential"]] <- as.numeric(df$home_score - df$away_score)
    result[["total_points"]]       <- as.numeric(df$home_score + df$away_score)
  }
  result
}


# ── Reporting ────────────────────────────────────────────────────────────────

print_header <- function(table_name) {
  cat("\n")
  cat(strrep("=", 90), "\n")
  cat("  ", table_name, "\n")
  cat(strrep("=", 90), "\n")
  cat(sprintf("%-22s %-14s %8s %10s %5s  %-14s %8s\n",
              "Metric", "Best Fit", "KS Stat", "p-value", "Conf",
              "Runner-up", "AIC Dlt"))
  cat(sprintf("%-22s %-14s %8s %10s %5s  %-14s %8s\n",
              strrep("-", 22), strrep("-", 14), strrep("-", 8),
              strrep("-", 10), strrep("-", 5), strrep("-", 14), strrep("-", 8)))
}


print_results <- function(table_name, metrics) {
  print_header(table_name)
  low_confidence <- list()

  for (col in sort(names(metrics))) {
    data <- metrics[[col]]
    out  <- fit_all_distributions(data, col)
    results <- out$results

    if (length(results) == 0) {
      cat(sprintf("%-22s (insufficient data or fit failure)\n", col))
      next
    }

    best      <- results[[1]]
    conf      <- confidence_label(best$p_value)
    runner_up <- if (length(results) > 1) results[[2]]$name else "---"
    aic_delta <- if (length(results) > 1) results[[2]]$aic - best$aic else 0

    cat(sprintf("%-22s %-14s %8.4f %10.4f %5s  %-14s %8.1f\n",
                col, best$name, best$ks_stat, best$p_value, conf,
                runner_up, aic_delta))

    if (VERBOSE) {
      cat(sprintf("  %22s n=%d, mean=%.2f, sd=%.2f, skew=%.2f, kurt=%.2f\n",
                  "",
                  length(data), mean(data), sd(data),
                  moments_skew(data), moments_kurt(data)))
      for (r in results[1:min(4, length(results))]) {
        marker <- if (identical(r, best)) " <-- best" else ""
        cat(sprintf("  %22s %-14s AIC=%12.1f  KS=%.4f  p=%.4f%s\n",
                    "", r$name, r$aic, r$ks_stat, r$p_value, marker))
      }
    }

    if (trimws(conf) == "LOW") {
      low_confidence[[length(low_confidence) + 1]] <- list(
        col = col, data = data, results = results
      )
    }
  }

  low_confidence
}


print_low_confidence <- function(low_items) {
  if (length(low_items) == 0) return()

  cat("\n")
  cat(strrep("=", 90), "\n")
  cat("  LOW-CONFIDENCE FIT DETAILS\n")
  cat(strrep("=", 90), "\n")

  for (item in low_items) {
    d <- item$data
    cat(sprintf("\n  %s (n=%d)\n", item$col, length(d)))
    cat(sprintf("    Summary: mean=%.3f, median=%.3f, sd=%.3f\n",
                mean(d), median(d), sd(d)))
    cat(sprintf("    Range: [%.3f, %.3f]\n", min(d), max(d)))
    cat(sprintf("    Skewness: %.3f, Kurtosis: %.3f\n",
                moments_skew(d), moments_kurt(d)))
    pcts <- quantile(d, probs = c(0.05, 0.25, 0.75, 0.95))
    cat(sprintf("    Percentiles: 5th=%.3f, 25th=%.3f, 75th=%.3f, 95th=%.3f\n",
                pcts[1], pcts[2], pcts[3], pcts[4]))
    cat("    Top 5 fits:\n")
    for (r in item$results[1:min(5, length(item$results))]) {
      conf <- confidence_label(r$p_value)
      cat(sprintf("      %-14s KS=%.4f  p=%.6f  AIC=%12.1f  [%s]\n",
                  r$name, r$ks_stat, r$p_value, r$aic, trimws(conf)))
      pstr <- paste(
        mapply(function(n, v) sprintf("%s=%.4f", n, v),
               names(r$params), unlist(r$params)),
        collapse = ", ")
      cat(sprintf("        params: %s\n", pstr))
    }
  }
}


# ── Helper: moments ──────────────────────────────────────────────────────────

moments_skew <- function(x) {
  n  <- length(x)
  m  <- mean(x)
  s  <- sd(x)
  if (s == 0) return(0)
  (n / ((n - 1) * (n - 2))) * sum(((x - m) / s)^3)
}

moments_kurt <- function(x) {
  # Excess kurtosis (normal = 0)
  n <- length(x)
  m <- mean(x)
  s <- sd(x)
  if (s == 0) return(0)
  m4 <- mean((x - m)^4)
  m4 / s^4 - 3
}


# ── Main ─────────────────────────────────────────────────────────────────────

main <- function() {
  conn <- dbConnect(RSQLite::SQLite(), DB_PATH)
  on.exit(dbDisconnect(conn))

  all_low <- list()

  tables <- list(
    "team_efficiency"             = load_team_efficiency,
    "player_stats (min-weighted)" = load_player_stats_weighted,
    "team_bpi"                    = load_team_bpi,
    "games"                       = load_game_metrics
  )

  for (tname in names(tables)) {
    if (!is.null(TABLE_FILTER) && !grepl(TABLE_FILTER, tname, fixed = TRUE)) {
      next
    }
    metrics <- tables[[tname]](conn)
    low     <- print_results(tname, metrics)
    all_low <- c(all_low, low)
  }

  print_low_confidence(all_low)

  cat("\n")
  cat(strrep("=", 90), "\n")
  cat(sprintf("  Done. %d metrics had LOW confidence fits.\n", length(all_low)))
  if (length(all_low) > 0) {
    cat("  Review the LOW-CONFIDENCE section above -- these may need manual inspection\n")
    cat("  or a non-standard distribution (mixture model, KDE, etc.).\n")
  }
  cat(strrep("=", 90), "\n")
}

main()
