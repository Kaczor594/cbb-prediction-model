# ── March Madness 2026 — Perfect Bracket Tracker ────────────────────────────────
#
# Track the 50 simulated brackets against actual tournament results.
# Games can be entered in ANY order. As each result comes in, brackets with
# wrong picks are eliminated. Reports remaining perfect brackets and the
# probability that at least one finishes perfect.
#
# Run in terminal:
#   cd ~/claude-projects/cbb-prediction-model
#   R
#   source("scripts/bracket_tracker.R")
#
# Commands:
#   result("Duke")                  Enter a result (winner only)
#   result("Duke", "Siena")         Enter a result (winner and loser)
#   results(c("Duke", "Ohio State", "Kansas"))   Enter several at once
#   status()                        Show tracker summary
#   available()                     Show all games ready to be entered
#   alive()                         Details on surviving brackets
#   history()                       All entered results
#   undo()                          Undo the last entered result
#   save_progress()                 Save to disk
#   load_progress()                 Restore from disk
#   reset()                         Start over

library(tidyverse)

# ── Load bracket data ───────────────────────────────────────────────────────────
brackets <- read_csv("data/simulated_brackets.csv", show_col_types = FALSE)
teams_df <- read_csv("data/tourney_teams.csv", show_col_types = FALSE)

n_sims <- max(brackets$sim_id)
n_games <- brackets %>% filter(sim_id == 1) %>% nrow()  # 63

# Canonical game structure from bracket 1 (slot positions same for all brackets)
canonical <- brackets %>% filter(sim_id == 1) %>% mutate(slot = row_number())

# Team seed lookup
team_seeds <- setNames(teams_df$seed, teams_df$name)

cat(sprintf("Loaded %d brackets, each with %d games.\n", n_sims, n_games))

# ── Build game tree ─────────────────────────────────────────────────────────────
# Slots are ordered by region, then by round within region:
#   Slots  1-15: East   (8 R64, 4 R32, 2 S16, 1 E8)
#   Slots 16-30: West
#   Slots 31-45: South
#   Slots 46-60: Midwest
#   Slots 61-62: Final Four
#   Slot  63:    Championship
#
# Feeder relationships (within each region, local offset):
#   R32 slot 9  <- winners of (1, 2)
#   R32 slot 10 <- winners of (3, 4)
#   R32 slot 11 <- winners of (5, 6)
#   R32 slot 12 <- winners of (7, 8)
#   S16 slot 13 <- winners of (9, 10)
#   S16 slot 14 <- winners of (11, 12)
#   E8  slot 15 <- winners of (13, 14)
#
# F4:  slot 61 <- (East E8 = 15, South E8 = 45)
#      slot 62 <- (West E8 = 30, Midwest E8 = 60)
# Championship: slot 63 <- (61, 62)

feeders <- vector("list", n_games)
region_offsets <- c(0, 15, 30, 45)

for (offset in region_offsets) {
  for (i in 0:3) {
    feeders[[offset + 9 + i]] <- c(offset + 2 * i + 1, offset + 2 * i + 2)
  }
  feeders[[offset + 13]] <- c(offset + 9, offset + 10)
  feeders[[offset + 14]] <- c(offset + 11, offset + 12)
  feeders[[offset + 15]] <- c(offset + 13, offset + 14)
}
feeders[[61]] <- c(15, 45)
feeders[[62]] <- c(30, 60)
feeders[[63]] <- c(61, 62)

# Round and region labels for each slot
slot_round  <- character(n_games)
slot_region <- character(n_games)
region_names <- c("East", "West", "South", "Midwest")

for (i in seq_along(region_offsets)) {
  o <- region_offsets[i]
  slot_round[(o + 1):(o + 8)]   <- "R64"
  slot_round[(o + 9):(o + 12)]  <- "R32"
  slot_round[(o + 13):(o + 14)] <- "S16"
  slot_round[o + 15]            <- "E8"
  slot_region[(o + 1):(o + 15)] <- region_names[i]
}
slot_round[61:62] <- "F4"
slot_round[63]    <- "Championship"
slot_region[61]   <- "East/South"
slot_region[62]   <- "West/Midwest"
slot_region[63]   <- "Final"

# ── Build pick and probability matrices ─────────────────────────────────────────
pick_matrix <- matrix(NA_character_, nrow = n_sims, ncol = n_games)
prob_matrix <- matrix(NA_real_,      nrow = n_sims, ncol = n_games)

for (s in 1:n_sims) {
  sim_games <- brackets %>% filter(sim_id == s)
  for (g in 1:n_games) {
    row <- sim_games[g, ]
    pick_matrix[s, g] <- row$winner
    if (row$winner == row$team_a) {
      prob_matrix[s, g] <- row$prob_a
    } else {
      prob_matrix[s, g] <- 1.0 - row$prob_a
    }
  }
}

# ── State ───────────────────────────────────────────────────────────────────────
env <- new.env(parent = emptyenv())
env$game_completed <- rep(FALSE, n_games)
env$game_winner    <- rep(NA_character_, n_games)
env$game_loser     <- rep(NA_character_, n_games)
env$is_alive       <- rep(TRUE, n_sims)
env$entry_order    <- integer(0)

# ── Get the actual teams for a game slot ────────────────────────────────────────
get_slot_teams <- function(slot) {
  if (is.null(feeders[[slot]])) {
    # R64: fixed matchup
    return(c(canonical$team_a[slot], canonical$team_b[slot]))
  }
  f <- feeders[[slot]]
  if (!env$game_completed[f[1]] || !env$game_completed[f[2]]) {
    return(c(NA_character_, NA_character_))
  }
  return(c(env$game_winner[f[1]], env$game_winner[f[2]]))
}

# ── Get seed for a team ─────────────────────────────────────────────────────────
get_seed <- function(team_name) {
  s <- team_seeds[team_name]
  if (is.na(s)) return(NA_integer_)
  as.integer(s)
}

# ── Find all available (ready but unplayed) games ───────────────────────────────
get_available_slots <- function() {
  avail <- integer(0)
  for (g in 1:n_games) {
    if (env$game_completed[g]) next
    if (is.null(feeders[[g]])) {
      avail <- c(avail, g)
    } else {
      f <- feeders[[g]]
      if (env$game_completed[f[1]] && env$game_completed[f[2]]) {
        avail <- c(avail, g)
      }
    }
  }
  avail
}

# ── Match a team name flexibly ──────────────────────────────────────────────────
match_team <- function(input, candidates) {
  candidates <- candidates[!is.na(candidates)]
  if (length(candidates) == 0) return(NA_character_)
  if (input %in% candidates) return(input)
  idx <- which(tolower(candidates) == tolower(input))
  if (length(idx) == 1) return(candidates[idx])
  idx <- which(grepl(tolower(input), tolower(candidates), fixed = TRUE))
  if (length(idx) == 1) return(candidates[idx])
  return(NA_character_)
}

# ── Find dependents of a slot (later-round games feeding from it) ───────────────
find_dependents <- function(slot) {
  deps <- integer(0)
  for (g in 1:n_games) {
    f <- feeders[[g]]
    if (!is.null(f) && slot %in% f && env$game_completed[g]) {
      deps <- c(deps, g)
      deps <- c(deps, find_dependents(g))
    }
  }
  unique(deps)
}

# ── Recompute alive status from scratch ─────────────────────────────────────────
recompute_alive <- function() {
  env$is_alive <- rep(TRUE, n_sims)
  for (g in which(env$game_completed)) {
    for (s in which(env$is_alive)) {
      if (pick_matrix[s, g] != env$game_winner[g]) {
        env$is_alive[s] <- FALSE
      }
    }
  }
}

# ── Enter a single result ───────────────────────────────────────────────────────
result <- function(winner, loser = NULL) {
  avail <- get_available_slots()
  if (length(avail) == 0) {
    n_done <- sum(env$game_completed)
    if (n_done == n_games) {
      cat("Tournament complete! All 63 games entered.\n")
    } else {
      cat("No games available. Enter earlier-round results to unlock later rounds.\n")
    }
    return(invisible(NULL))
  }

  # Build map of available games and their teams
  avail_teams <- list()
  all_team_names <- character(0)
  for (g in avail) {
    teams <- get_slot_teams(g)
    avail_teams[[as.character(g)]] <- teams
    all_team_names <- c(all_team_names, teams)
  }

  # Match winner
  matched_winner <- match_team(winner, unique(all_team_names))
  if (is.na(matched_winner)) {
    cat(sprintf("'%s' not found in any available game.\n", winner))
    cat("Use available() to see games ready to enter.\n")
    return(invisible(NULL))
  }

  # Find which slot has this team
  matching_slots <- integer(0)
  for (g in avail) {
    teams <- avail_teams[[as.character(g)]]
    if (matched_winner %in% teams) {
      matching_slots <- c(matching_slots, g)
    }
  }

  if (length(matching_slots) == 0) {
    cat(sprintf("'%s' not found in any available game.\n", matched_winner))
    return(invisible(NULL))
  }

  if (length(matching_slots) > 1) {
    cat(sprintf("'%s' appears in multiple available games. Specify the loser too.\n",
                matched_winner))
    for (g in matching_slots) {
      teams <- avail_teams[[as.character(g)]]
      opp <- setdiff(teams, matched_winner)
      cat(sprintf("  [%s %s] %s vs %s\n", slot_round[g], slot_region[g],
                  matched_winner, opp))
    }
    return(invisible(NULL))
  }

  g <- matching_slots[1]
  teams <- avail_teams[[as.character(g)]]
  actual_loser <- setdiff(teams, matched_winner)

  # Validate loser if provided
  if (!is.null(loser)) {
    matched_loser <- match_team(loser, teams)
    if (is.na(matched_loser) || matched_loser == matched_winner) {
      cat(sprintf("Loser '%s' doesn't match. This game is: %s vs %s\n",
                  loser, teams[1], teams[2]))
      return(invisible(NULL))
    }
    actual_loser <- matched_loser
  }

  # Record result
  env$game_completed[g] <- TRUE
  env$game_winner[g]    <- matched_winner
  env$game_loser[g]     <- actual_loser
  env$entry_order       <- c(env$entry_order, g)

  # Eliminate brackets that picked wrong
  n_before <- sum(env$is_alive)
  for (s in which(env$is_alive)) {
    if (pick_matrix[s, g] != matched_winner) {
      env$is_alive[s] <- FALSE
    }
  }
  n_after  <- sum(env$is_alive)
  eliminated <- n_before - n_after

  seed_w <- get_seed(matched_winner)
  seed_l <- get_seed(actual_loser)
  cat(sprintf("[%s %s] (%d) %s def. (%d) %s",
              slot_round[g], slot_region[g],
              seed_w, matched_winner, seed_l, actual_loser))
  if (eliminated > 0) {
    cat(sprintf("  | -%d eliminated", eliminated))
  }
  cat(sprintf("  | %d/%d alive\n", n_after, n_sims))

  if (n_after > 0 && n_after <= 20) {
    show_alive_brief()
  }

  invisible(NULL)
}

# ── Enter multiple results (just winners) ───────────────────────────────────────
results <- function(winners) {
  for (w in winners) {
    result(w)
    if (sum(env$is_alive) == 0) {
      cat("\nAll brackets eliminated!\n")
      break
    }
  }
  cat("\n")
  status()
}

# ── Undo the last result (cascades to dependents) ──────────────────────────────
undo <- function() {
  if (length(env$entry_order) == 0) {
    cat("No results to undo.\n")
    return(invisible(NULL))
  }

  last_slot <- tail(env$entry_order, 1)
  deps <- find_dependents(last_slot)
  to_undo <- unique(c(deps, last_slot))

  for (g in to_undo) {
    if (env$game_completed[g]) {
      cat(sprintf("  Undoing: [%s %s] %s def. %s\n",
                  slot_round[g], slot_region[g], env$game_winner[g], env$game_loser[g]))
      env$game_completed[g] <- FALSE
      env$game_winner[g]    <- NA_character_
      env$game_loser[g]     <- NA_character_
    }
  }

  env$entry_order <- env$entry_order[!env$entry_order %in% to_undo]
  recompute_alive()

  cat(sprintf("%d game(s) undone. %d/%d brackets alive.\n",
              length(to_undo), sum(env$is_alive), n_sims))
  invisible(NULL)
}

# ── Show available games ────────────────────────────────────────────────────────
available <- function() {
  avail <- get_available_slots()
  if (length(avail) == 0) {
    n_done <- sum(env$game_completed)
    if (n_done == n_games) {
      cat("Tournament complete! All 63 games entered.\n")
    } else {
      cat("No games available. Enter earlier-round results to unlock later rounds.\n")
    }
    return(invisible(NULL))
  }

  cat(sprintf("\n  %d game(s) ready to enter:\n", length(avail)))
  for (rd in c("R64", "R32", "S16", "E8", "F4", "Championship")) {
    rd_slots <- avail[slot_round[avail] == rd]
    if (length(rd_slots) == 0) next

    rd_label <- switch(rd,
      R64 = "ROUND OF 64", R32 = "ROUND OF 32", S16 = "SWEET 16",
      E8 = "ELITE 8", F4 = "FINAL FOUR", Championship = "CHAMPIONSHIP")
    cat(sprintf("\n  -- %s --\n", rd_label))

    for (g in rd_slots) {
      teams <- get_slot_teams(g)
      seed_a <- get_seed(teams[1])
      seed_b <- get_seed(teams[2])
      cat(sprintf("    (%2d) %-22s vs (%2d) %-22s  [%s]\n",
                  seed_a, teams[1], seed_b, teams[2], slot_region[g]))
    }
  }
  cat("\n")
  invisible(NULL)
}

# ── Status display ──────────────────────────────────────────────────────────────
status <- function() {
  n_alive <- sum(env$is_alive)
  n_done  <- sum(env$game_completed)
  n_avail <- length(get_available_slots())

  cat(sprintf("\n%s\n", strrep("=", 70)))
  cat("  PERFECT BRACKET TRACKER\n")
  cat(sprintf("%s\n", strrep("=", 70)))

  round_labels <- c(R64 = "Round of 64", R32 = "Round of 32",
                    S16 = "Sweet 16", E8 = "Elite 8",
                    F4 = "Final Four", Championship = "Championship")
  round_totals <- c(R64 = 32, R32 = 16, S16 = 8, E8 = 4, F4 = 2, Championship = 1)

  cat(sprintf("\n  Games entered: %d / %d\n", n_done, n_games))
  for (rd in names(round_totals)) {
    rd_slots <- which(slot_round == rd)
    rd_done  <- sum(env$game_completed[rd_slots])
    rd_total <- round_totals[rd]
    bar <- paste0("[", strrep("#", rd_done), strrep(".", rd_total - rd_done), "]")
    cat(sprintf("    %-15s %2d/%-2d  %s\n", round_labels[rd], rd_done, rd_total, bar))
  }

  cat(sprintf("\n  Brackets alive:  %d / %d\n", n_alive, n_sims))
  cat(sprintf("  Games available: %d\n", n_avail))

  if (n_alive == 0 && n_done > 0) {
    cat("\n  All brackets have been eliminated.\n")
    # Find longest survivor
    last_correct <- rep(0L, n_sims)
    temp_alive <- rep(TRUE, n_sims)
    for (idx in seq_along(env$entry_order)) {
      g <- env$entry_order[idx]
      for (s in which(temp_alive)) {
        if (pick_matrix[s, g] != env$game_winner[g]) {
          last_correct[s] <- idx - 1L
          temp_alive[s] <- FALSE
        }
      }
    }
    last_correct[temp_alive] <- length(env$entry_order)
    best <- which.max(last_correct)
    cat(sprintf("  Longest survivor: Bracket #%d (correct through %d games)\n",
                best, last_correct[best]))
  } else if (n_alive > 0) {
    p_perfect <- compute_perfect_prob()
    cat(sprintf("\n  P(at least one perfect): %.6f%%  (1 in %s)\n",
                p_perfect * 100, format_odds(p_perfect)))

    if (n_alive <= 20) {
      show_alive_brief()
    }
  }

  cat(sprintf("%s\n", strrep("=", 70)))
  invisible(NULL)
}

# ── Compute P(at least one bracket finishes perfect) ────────────────────────────
compute_perfect_prob <- function() {
  remaining <- which(!env$game_completed)

  if (length(remaining) == 0) {
    return(as.double(sum(env$is_alive) > 0))
  }

  alive_ids <- which(env$is_alive)
  if (length(alive_ids) == 0) return(0.0)

  p_each <- numeric(length(alive_ids))
  for (i in seq_along(alive_ids)) {
    s <- alive_ids[i]
    p_each[i] <- prod(prob_matrix[s, remaining])
  }

  # P(at least one) = 1 - prod(1 - p_i), brackets are independent simulations
  1.0 - prod(1.0 - p_each)
}

# ── Format large odds ──────────────────────────────────────────────────────────
format_odds <- function(p) {
  if (p <= 0) return("Inf")
  odds <- 1.0 / p
  if (odds >= 1e15)      sprintf("%.1f quadrillion", odds / 1e15)
  else if (odds >= 1e12) sprintf("%.1f trillion", odds / 1e12)
  else if (odds >= 1e9)  sprintf("%.1f billion", odds / 1e9)
  else if (odds >= 1e6)  sprintf("%.1f million", odds / 1e6)
  else if (odds >= 1e3)  sprintf("%.1f thousand", odds / 1e3)
  else                    sprintf("%.1f", odds)
}

# ── Show alive brackets briefly ─────────────────────────────────────────────────
show_alive_brief <- function() {
  alive_ids <- which(env$is_alive)
  if (length(alive_ids) == 0) return(invisible(NULL))

  remaining <- which(!env$game_completed)

  cat(sprintf("\n  Alive brackets:\n"))
  for (s in alive_ids) {
    champ <- pick_matrix[s, n_games]
    if (length(remaining) > 0) {
      p_rest <- prod(prob_matrix[s, remaining])
      cat(sprintf("    #%02d  |  P(perfect): %.2e  |  Champion: %s\n",
                  s, p_rest, champ))
    } else {
      cat(sprintf("    #%02d  |  PERFECT!  |  Champion: %s\n", s, champ))
    }
  }
}

# ── Show alive bracket details ──────────────────────────────────────────────────
alive <- function() {
  alive_ids <- which(env$is_alive)

  if (length(alive_ids) == 0) {
    cat("No brackets are still alive.\n")
    return(invisible(NULL))
  }

  cat(sprintf("\n%d bracket(s) still alive:\n", length(alive_ids)))

  for (s in alive_ids) {
    sim_games <- brackets %>% filter(sim_id == s)
    champ <- sim_games %>% filter(round == "Championship") %>% pull(winner)
    f4 <- sim_games %>%
      filter(round %in% c("F4", "Championship")) %>%
      distinct(winner) %>% pull(winner)

    remaining <- which(!env$game_completed)
    p_rest <- if (length(remaining) > 0) prod(prob_matrix[s, remaining]) else 1.0

    cat(sprintf("\n  Bracket #%d\n", s))
    cat(sprintf("    Champion:  %s\n", champ))
    cat(sprintf("    Final Four: %s\n", paste(f4, collapse = ", ")))
    cat(sprintf("    P(finish perfect): %.2e (1 in %s)\n",
                p_rest, format_odds(p_rest)))

    # Show next pending picks in available games
    avail <- get_available_slots()
    if (length(avail) > 0) {
      cat("    Next picks:\n")
      for (g in avail[1:min(5, length(avail))]) {
        pick <- pick_matrix[s, g]
        p_pick <- prob_matrix[s, g]
        cat(sprintf("      [%s %s] %s (%.0f%%)\n",
                    slot_round[g], slot_region[g], pick, p_pick * 100))
      }
    }
  }
  cat("\n")
  invisible(NULL)
}

# ── Show result history ─────────────────────────────────────────────────────────
history <- function() {
  if (length(env$entry_order) == 0) {
    cat("No results entered yet.\n")
    return(invisible(NULL))
  }

  cat(sprintf("\n  Results entered (%d games):\n", length(env$entry_order)))
  for (i in seq_along(env$entry_order)) {
    g <- env$entry_order[i]
    w <- env$game_winner[g]
    l <- env$game_loser[g]
    cat(sprintf("    %2d. [%-13s %-12s] (%2d) %-22s def. (%2d) %s\n",
                i, slot_round[g], slot_region[g],
                get_seed(w), w, get_seed(l), l))
  }
  cat("\n")
  invisible(NULL)
}

# ── Save/Load progress ─────────────────────────────────────────────────────────
save_progress <- function(path = "data/bracket_tracker_progress.rds") {
  state <- list(
    game_completed = env$game_completed,
    game_winner    = env$game_winner,
    game_loser     = env$game_loser,
    is_alive       = env$is_alive,
    entry_order    = env$entry_order
  )
  saveRDS(state, path)
  cat(sprintf("Progress saved to %s (%d games).\n", path, sum(env$game_completed)))
}

load_progress <- function(path = "data/bracket_tracker_progress.rds") {
  if (!file.exists(path)) {
    cat("No saved progress found.\n")
    return(invisible(NULL))
  }
  state <- readRDS(path)
  env$game_completed <- state$game_completed
  env$game_winner    <- state$game_winner
  env$game_loser     <- state$game_loser
  env$is_alive       <- state$is_alive
  env$entry_order    <- state$entry_order
  cat(sprintf("Loaded progress: %d games, %d/%d brackets alive.\n",
              sum(env$game_completed), sum(env$is_alive), n_sims))
  invisible(NULL)
}

# ── Reset everything ────────────────────────────────────────────────────────────
reset <- function() {
  env$game_completed <- rep(FALSE, n_games)
  env$game_winner    <- rep(NA_character_, n_games)
  env$game_loser     <- rep(NA_character_, n_games)
  env$is_alive       <- rep(TRUE, n_sims)
  env$entry_order    <- integer(0)
  cat("Tracker reset. All 50 brackets alive.\n")
}

# ── Startup ─────────────────────────────────────────────────────────────────────
if (file.exists("data/bracket_tracker_progress.rds")) {
  load_progress()
  cat("\n")
  status()
} else {
  cat("\nReady! Enter results in any order:\n")
  cat("  result(\"Duke\")             - enter a winner\n")
  cat("  result(\"Duke\", \"Siena\")    - winner and loser\n")
  cat("  available()                - see games ready to enter\n")
  cat("  status()                   - tracker summary\n\n")
}
