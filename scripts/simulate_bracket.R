# ── March Madness 2026 — Sweet 16 Bracket Simulator ─────────────────────────
#
# Simulates the remainder of the tournament starting from the Sweet 16
# using matchup probabilities from the XGBoost model (retrained with all
# regular-season + tournament data through the Round of 32).
#
# Prerequisites:
#   python3 .venv/bin/python3 -c "..."  (generate s16_matchup_probs.csv)
#
# Usage in RStudio:
#   source("scripts/simulate_bracket.R")
#
# Outputs saved to data/simulated_brackets_s16.csv

library(tidyverse)

# ── Load data ────────────────────────────────────────────────────────────────
teams  <- read_csv("data/s16_teams.csv", show_col_types = FALSE)
probs  <- read_csv("data/s16_matchup_probs.csv", show_col_types = FALSE)
venues <- read_csv("data/tourney_venues.csv", show_col_types = FALSE)
semis  <- read_csv("data/tourney_semis.csv", show_col_types = FALSE)

# ── Sweet 16 matchups (determined by bracket position) ───────────────────────
# Upper half: seeds 1/16/8/9 winner vs 5/12/4/13 winner
# Lower half: seeds 6/11/3/14 winner vs 7/10/2/15 winner
s16_matchups <- tribble(
  ~region,    ~game, ~team_a_id, ~team_a_name,              ~seed_a, ~team_b_id, ~team_b_name,              ~seed_b,
  "East",     "A",   150,        "Duke Blue Devils",        1,       2599,       "St. John's Red Storm",    5,

  "East",     "B",   127,        "Michigan State Spartans",  3,       41,         "UConn Huskies",           2,
  "West",     "A",   12,         "Arizona Wildcats",         1,       8,          "Arkansas Razorbacks",     4,
  "West",     "B",   251,        "Texas Longhorns",          11,      2509,       "Purdue Boilermakers",     2,
  "South",    "A",   2294,       "Iowa Hawkeyes",            9,       158,        "Nebraska Cornhuskers",    4,
  "South",    "B",   356,        "Illinois Fighting Illini",  3,       248,        "Houston Cougars",         2,
  "Midwest",  "A",   130,        "Michigan Wolverines",      1,       333,        "Alabama Crimson Tide",    4,
  "Midwest",  "B",   2633,       "Tennessee Volunteers",     6,       66,         "Iowa State Cyclones",     2
)

# ── Venue lookup ─────────────────────────────────────────────────────────────
region_venues <- c(
  East    = "Washington DC",
  West    = "San Jose",
  South   = "Houston",
  Midwest = "Chicago"
)
ff_venue <- "Indianapolis"

get_venue <- function(region, round_name) {
  if (round_name %in% c("Final Four", "Championship")) return(ff_venue)
  return(region_venues[region])
}

# ── Pre-index probability table ──────────────────────────────────────────────
prob_key <- paste(probs$team_a_id, probs$team_b_id, probs$venue, sep = "_")
prob_map <- setNames(probs$prob_a_wins, prob_key)

prob_avg <- probs %>%
  group_by(team_a_id, team_b_id) %>%
  summarise(prob_a_wins = mean(prob_a_wins), .groups = "drop")
prob_avg_key <- paste(prob_avg$team_a_id, prob_avg$team_b_id, sep = "_")
prob_avg_map <- setNames(prob_avg$prob_a_wins, prob_avg_key)

get_win_prob_fast <- function(id_a, id_b, venue_city) {
  if (is.na(id_a) || is.na(id_b)) return(0.5)
  if (id_a == id_b) return(0.5)

  if (id_a < id_b) {
    key <- paste(id_a, id_b, venue_city, sep = "_")
    p <- prob_map[key]
    if (!is.na(p)) return(as.numeric(p))
    avg_key <- paste(id_a, id_b, sep = "_")
    p <- prob_avg_map[avg_key]
    if (!is.na(p)) return(as.numeric(p))
    return(0.5)
  } else {
    key <- paste(id_b, id_a, venue_city, sep = "_")
    p <- prob_map[key]
    if (!is.na(p)) return(1.0 - as.numeric(p))
    avg_key <- paste(id_b, id_a, sep = "_")
    p <- prob_avg_map[avg_key]
    if (!is.na(p)) return(1.0 - as.numeric(p))
    return(0.5)
  }
}

# ── Simulate one game ───────────────────────────────────────────────────────
play_game <- function(team_a, team_b, venue_city) {
  prob_a <- get_win_prob_fast(team_a$id, team_b$id, venue_city)
  if (runif(1) < prob_a) {
    return(list(winner = team_a, loser = team_b, prob = prob_a))
  } else {
    return(list(winner = team_b, loser = team_a, prob = 1 - prob_a))
  }
}

# ── Simulate one bracket from Sweet 16 ──────────────────────────────────────
simulate_one_bracket <- function(sim_id) {
  set.seed(sim_id)

  game_log <- tibble(
    sim_id      = integer(),
    round       = character(),
    region      = character(),
    venue       = character(),
    team_a      = character(),
    seed_a      = integer(),
    team_b      = character(),
    seed_b      = integer(),
    prob_a      = double(),
    winner      = character(),
    winner_seed = integer()
  )

  add_game <- function(round_name, region, venue, ta, tb, prob_a, w) {
    game_log <<- bind_rows(game_log, tibble(
      sim_id      = sim_id,
      round       = round_name,
      region      = region,
      venue       = venue %||% "",
      team_a      = ta$name,
      seed_a      = ta$seed,
      team_b      = tb$name,
      seed_b      = tb$seed,
      prob_a      = round(prob_a, 4),
      winner      = w$name,
      winner_seed = w$seed
    ))
  }

  final_four <- list()

  for (reg in c("East", "West", "South", "Midwest")) {
    reg_matchups <- s16_matchups %>% filter(region == reg)
    venue <- get_venue(reg, "Sweet 16")

    # Sweet 16: 2 games per region
    s16_winners <- list()
    for (i in seq_len(nrow(reg_matchups))) {
      g <- reg_matchups[i, ]
      ta <- tibble(id = g$team_a_id, name = g$team_a_name,
                   seed = g$seed_a, region = reg)
      tb <- tibble(id = g$team_b_id, name = g$team_b_name,
                   seed = g$seed_b, region = reg)

      result <- play_game(ta, tb, venue)
      add_game("S16", reg, venue, ta, tb,
               get_win_prob_fast(ta$id, tb$id, venue), result$winner)
      s16_winners[[i]] <- result$winner
    }

    # Elite 8: 1 game per region (Game A winner vs Game B winner)
    ta <- s16_winners[[1]]
    tb <- s16_winners[[2]]
    e8_venue <- get_venue(reg, "Elite 8")

    result <- play_game(ta, tb, e8_venue)
    add_game("E8", reg, e8_venue, ta, tb,
             get_win_prob_fast(ta$id, tb$id, e8_venue), result$winner)
    final_four[[reg]] <- result$winner
  }

  # Final Four
  ff_winners <- list()
  for (i in seq_len(nrow(semis))) {
    reg_a <- semis$region_a[i]
    reg_b <- semis$region_b[i]
    ta <- final_four[[reg_a]]
    tb <- final_four[[reg_b]]

    result <- play_game(ta, tb, ff_venue)
    add_game("F4", paste(reg_a, "vs", reg_b), ff_venue, ta, tb,
             get_win_prob_fast(ta$id, tb$id, ff_venue), result$winner)
    ff_winners[[length(ff_winners) + 1]] <- result$winner
  }

  # Championship
  ta <- ff_winners[[1]]
  tb <- ff_winners[[2]]
  result <- play_game(ta, tb, ff_venue)
  add_game("Championship", "Final", ff_venue, ta, tb,
           get_win_prob_fast(ta$id, tb$id, ff_venue), result$winner)

  return(game_log)
}

# ── Print one bracket ────────────────────────────────────────────────────────
print_bracket <- function(games, sim_id) {
  cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
  cat(sprintf("  BRACKET #%d\n", sim_id))
  cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))

  round_order <- c("S16", "E8", "F4", "Championship")
  round_labels <- c(
    S16 = "SWEET 16", E8 = "ELITE 8",
    F4 = "FINAL FOUR", Championship = "CHAMPIONSHIP"
  )

  for (rd in round_order) {
    rd_games <- games %>% filter(round == rd)
    if (nrow(rd_games) == 0) next

    cat(sprintf("\n  -- %s --\n", round_labels[rd]))

    for (i in seq_len(nrow(rd_games))) {
      g <- rd_games[i, ]
      prob_display <- ifelse(g$prob_a > 0.5,
                             sprintf("%.0f%%", g$prob_a * 100),
                             sprintf("%.0f%%", (1 - g$prob_a) * 100))

      if (rd %in% c("S16", "E8")) {
        cat(sprintf("    %s [%s] %s (%2d) %s vs (%2d) %-22s [%s]\n",
                    ifelse(g$winner == g$team_a, ">>", "  "),
                    ifelse(g$winner == g$team_a, prob_display, ""),
                    sprintf("%-22s", g$team_a), g$seed_a,
                    g$region,
                    g$seed_b, g$team_b,
                    ifelse(g$winner == g$team_b, prob_display, "")))
      } else {
        cat(sprintf("    %s (%2d) %-22s vs  (%2d) %-22s  [Winner: %s %s]\n",
                    g$region, g$seed_a, g$team_a,
                    g$seed_b, g$team_b,
                    g$winner, prob_display))
      }
    }
  }

  champ <- games %>% filter(round == "Championship")
  cat(sprintf("\n  CHAMPION: (%d) %s\n",
              champ$winner_seed[1], champ$winner[1]))
  cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))
}

# ── Run 25 simulations ──────────────────────────────────────────────────────
cat("Simulating 25 brackets from the Sweet 16...\n\n")

n_sims <- 25
results <- vector("list", n_sims)
all_games <- tibble()

for (s in seq_len(n_sims)) {
  bracket_games <- simulate_one_bracket(sim_id = s)
  results[[s]] <- bracket_games
  all_games <- bind_rows(all_games, bracket_games)
  print_bracket(bracket_games, s)
}

# ── Save results ─────────────────────────────────────────────────────────────
write_csv(all_games, "data/simulated_brackets_s16.csv")
cat(sprintf("\nSaved %d games across %d brackets to data/simulated_brackets_s16.csv\n",
            nrow(all_games), n_sims))

# ── Summary ──────────────────────────────────────────────────────────────────
cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
cat("  SUMMARY ACROSS 25 BRACKETS\n")
cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))

champs <- all_games %>%
  filter(round == "Championship") %>%
  count(winner, winner_seed, sort = TRUE) %>%
  mutate(pct = n / n_sims)

cat("\n  Championship wins:\n")
for (i in seq_len(nrow(champs))) {
  ch <- champs[i, ]
  bar <- paste(rep("#", round(ch$pct * 25)), collapse = "")
  cat(sprintf("    (%2d) %-22s %2d/%d  %5.1f%%  %s\n",
              ch$winner_seed, ch$winner, ch$n, n_sims, ch$pct * 100, bar))
}

f4 <- all_games %>%
  filter(round %in% c("F4", "Championship")) %>%
  distinct(sim_id, winner) %>%
  count(winner, sort = TRUE) %>%
  mutate(pct = n / n_sims)

cat("\n  Final Four appearances:\n")
for (i in seq_len(nrow(f4))) {
  fi <- f4[i, ]
  cat(sprintf("    %-22s %2d/%d  %5.1f%%\n", fi$winner, fi$n, n_sims, fi$pct * 100))
}

e8 <- all_games %>%
  filter(round %in% c("E8", "F4", "Championship")) %>%
  distinct(sim_id, winner) %>%
  count(winner, sort = TRUE) %>%
  mutate(pct = n / n_sims)

cat("\n  Elite 8 appearances:\n")
for (i in seq_len(nrow(e8))) {
  ei <- e8[i, ]
  cat(sprintf("    %-22s %2d/%d  %5.1f%%\n", ei$winner, ei$n, n_sims, ei$pct * 100))
}

# ── Upset tracker ────────────────────────────────────────────────────────────
upsets <- all_games %>%
  filter(winner_seed > pmin(seed_a, seed_b) + 2) %>%
  count(winner, winner_seed, round, sort = TRUE)

if (nrow(upsets) > 0) {
  cat("\n  Most common upsets (seed diff > 2):\n")
  for (i in seq_len(min(15, nrow(upsets)))) {
    u <- upsets[i, ]
    cat(sprintf("    (%2d) %-22s in %-15s  %2d times\n",
                u$winner_seed, u$winner, u$round, u$n))
  }
}
