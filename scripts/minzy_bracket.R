# ── Minzy's 2026 March Madness Bracket ────────────────────────────────────────
#
# A custom bracket for Minzy the cat, built from model probabilities with
# the following preferences:
#   - Cat mascot teams get a boost in close games (Wildcats, Cougars, Tigers,
#     Panthers, Pride/Lions)
#   - Dog mascot teams get a slight boost (Bulldogs, Huskies)
#   - Duke cannot go far (early exit)
#   - North Carolina goes further than Duke
#   - Michigan should probably win the whole thing
#   - Purdue (Boilermakers) is a favorite
#   - Ohio State is out early
#   - Michigan State should NOT make the Final Four
#   - Slight Big Ten bias
#   - Slight bias against SEC teams
#
# Usage in RStudio:
#   source("scripts/minzy_bracket.R")

library(tidyverse)

# ── Define Minzy's picks ─────────────────────────────────────────────────────
# Each game: list(team_a, seed_a, team_b, seed_b, winner, note)

cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("             MINZY'S 2026 MARCH MADNESS BRACKET\n")
cat("                    (a bracket by a cat, for cats)\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

print_game <- function(round, region, team_a, seed_a, team_b, seed_b, winner, note = "") {
  marker_a <- ifelse(winner == team_a, ">>", "  ")
  marker_b <- ifelse(winner == team_b, ">>", "  ")
  note_str <- ifelse(nchar(note) > 0, paste0("  [", note, "]"), "")
  cat(sprintf("  %s (%2d) %-22s vs  (%2d) %-22s %s%s\n",
              marker_a, seed_a, team_a, seed_b, team_b, marker_b, note_str))
}

# ══════════════════════════════════════════════════════════════════════════════
# EAST REGION
# ══════════════════════════════════════════════════════════════════════════════
cat("\n  ── EAST REGION ──\n")

cat("\n  Round of 64:\n")
print_game("R64", "East", "Duke", 1, "Siena", 16,
           "Duke", "Blue Devils survive... for now")
print_game("R64", "East", "Ohio State", 8, "TCU", 9,
           "TCU", "Minzy says no to Buckeyes")
print_game("R64", "East", "St. John's", 5, "Northern Iowa", 12,
           "Northern Iowa", "PANTHERS! Cat upset!")
print_game("R64", "East", "Kansas", 4, "Cal Baptist", 13,
           "Kansas", "Jayhawks too strong")
print_game("R64", "East", "Louisville", 6, "South Florida", 11,
           "Louisville", "Cardinals advance")
print_game("R64", "East", "Michigan State", 3, "North Dakota St", 14,
           "Michigan State", "Spartans handle business")
print_game("R64", "East", "UCLA", 7, "UCF", 10,
           "UCLA", "Big Ten Bruins move on")
print_game("R64", "East", "UConn", 2, "Furman", 15,
           "UConn", "Good dogs, the Huskies")

cat("\n  Round of 32:\n")
print_game("R32", "East", "Duke", 1, "TCU", 9,
           "TCU", "DUKE GOES HOME! Horned Frogs pull the upset")
print_game("R32", "East", "Northern Iowa", 12, "Kansas", 4,
           "Kansas", "Panthers had their moment")
print_game("R32", "East", "Louisville", 6, "Michigan State", 3,
           "Michigan State", "Spartans roll")
print_game("R32", "East", "UCLA", 7, "UConn", 2,
           "UConn", "Huskies are too good")

cat("\n  Sweet 16:\n")
print_game("S16", "East", "TCU", 9, "Kansas", 4,
           "Kansas", "Cinderella run ends for TCU")
print_game("S16", "East", "Michigan State", 3, "UConn", 2,
           "UConn", "MSU stopped short of the Final Four")

cat("\n  Elite 8:\n")
print_game("E8", "East", "Kansas", 4, "UConn", 2,
           "UConn", "Huskies are going to the Final Four!")

cat("\n  East Champion: (2) UConn Huskies\n")

# ══════════════════════════════════════════════════════════════════════════════
# WEST REGION
# ══════════════════════════════════════════════════════════════════════════════
cat("\n  ── WEST REGION ──\n")

cat("\n  Round of 64:\n")
print_game("R64", "West", "Arizona", 1, "Long Island", 16,
           "Arizona", "Wildcats! Meow!")
print_game("R64", "West", "Villanova", 8, "Utah State", 9,
           "Villanova", "Wildcats! Double meow!")
print_game("R64", "West", "Wisconsin", 5, "High Point", 12,
           "High Point", "PANTHERS upset the Badgers! Cats > Badgers")
print_game("R64", "West", "Arkansas", 4, "Hawai'i", 13,
           "Arkansas", "Razorbacks too tough")
print_game("R64", "West", "BYU", 6, "Texas", 11,
           "BYU", "Cougars! Cat mascot wins the close one")
print_game("R64", "West", "Gonzaga", 3, "Kennesaw St", 14,
           "Gonzaga", "Bulldogs are good boys")
print_game("R64", "West", "Miami (FL)", 7, "Missouri", 10,
           "Missouri", "Tigers! Cat > Hurricanes")
print_game("R64", "West", "Purdue", 2, "Queens (NC)", 15,
           "Purdue", "Minzy's Boilermakers cruise")

cat("\n  Round of 32:\n")
print_game("R32", "West", "Arizona", 1, "Villanova", 8,
           "Arizona", "Wildcat vs Wildcat! Arizona is the bigger cat")
print_game("R32", "West", "High Point", 12, "Arkansas", 4,
           "High Point", "Panthers Cinderella! Cat > SEC Razorbacks")
print_game("R32", "West", "BYU", 6, "Gonzaga", 3,
           "BYU", "Cat vs Dog: Cougars defeat Bulldogs!")
print_game("R32", "West", "Missouri", 10, "Purdue", 2,
           "Purdue", "Sorry Tigers, Minzy loves the Boilermakers")

cat("\n  Sweet 16:\n")
print_game("S16", "West", "Arizona", 1, "High Point", 12,
           "Arizona", "Bigger Wildcats win the cat battle")
print_game("S16", "West", "BYU", 6, "Purdue", 2,
           "Purdue", "Boilermakers > Cougars (Minzy's fave)")

cat("\n  Elite 8:\n")
print_game("E8", "West", "Arizona", 1, "Purdue", 2,
           "Purdue", "Boilermakers knock off the Wildcats! Big Ten pride!")

cat("\n  West Champion: (2) Purdue Boilermakers\n")

# ══════════════════════════════════════════════════════════════════════════════
# SOUTH REGION
# ══════════════════════════════════════════════════════════════════════════════
cat("\n  ── SOUTH REGION ──\n")

cat("\n  Round of 64:\n")
print_game("R64", "South", "Florida", 1, "Prairie View A&M", 16,
           "Florida", "Panthers tried but Gators too strong")
print_game("R64", "South", "Clemson", 8, "Iowa", 9,
           "Clemson", "Tigers! Cat beats Big Ten Hawkeyes")
print_game("R64", "South", "Vanderbilt", 5, "McNeese", 12,
           "Vanderbilt", "Commodores advance")
print_game("R64", "South", "Nebraska", 4, "Troy", 13,
           "Nebraska", "Big Ten Cornhuskers cruise")
print_game("R64", "South", "North Carolina", 6, "VCU", 11,
           "North Carolina", "UNC must outlast Duke")
print_game("R64", "South", "Illinois", 3, "Penn", 14,
           "Illinois", "Big Ten Illini roll")
print_game("R64", "South", "Saint Mary's", 7, "Texas A&M", 10,
           "Saint Mary's", "Gaels over Aggies (anti-SEC)")
print_game("R64", "South", "Houston", 2, "Idaho", 15,
           "Houston", "Cougars! Big cat energy")

cat("\n  Round of 32:\n")
print_game("R32", "South", "Florida", 1, "Clemson", 8,
           "Clemson", "TIGERS upset Gators! Cat > Gator always")
print_game("R32", "South", "Vanderbilt", 5, "Nebraska", 4,
           "Nebraska", "Big Ten > SEC in Minzy's world")
print_game("R32", "South", "North Carolina", 6, "Illinois", 3,
           "North Carolina", "UNC advances to Sweet 16 (further than Duke!)")
print_game("R32", "South", "Saint Mary's", 7, "Houston", 2,
           "Houston", "Cougars power through")

cat("\n  Sweet 16:\n")
print_game("S16", "South", "Clemson", 8, "Nebraska", 4,
           "Clemson", "Tiger magic continues! Cat > Cornhuskers")
print_game("S16", "South", "North Carolina", 6, "Houston", 2,
           "Houston", "Cougars send UNC home (UNC still went further than Duke)")

cat("\n  Elite 8:\n")
print_game("E8", "South", "Clemson", 8, "Houston", 2,
           "Houston", "Cat vs Cat in the Elite 8! Cougars > Tigers")

cat("\n  South Champion: (2) Houston Cougars\n")

# ══════════════════════════════════════════════════════════════════════════════
# MIDWEST REGION
# ══════════════════════════════════════════════════════════════════════════════
cat("\n  ── MIDWEST REGION ──\n")

cat("\n  Round of 64:\n")
print_game("R64", "Midwest", "Michigan", 1, "Howard", 16,
           "Michigan", "Minzy's champion begins")
print_game("R64", "Midwest", "Georgia", 8, "Saint Louis", 9,
           "Georgia", "Bulldogs! Good dog")
print_game("R64", "Midwest", "Texas Tech", 5, "Akron", 12,
           "Texas Tech", "Red Raiders advance")
print_game("R64", "Midwest", "Alabama", 4, "Hofstra", 13,
           "Hofstra", "PRIDE upset! Lions are cats! Bye-bye Bama")
print_game("R64", "Midwest", "Tennessee", 6, "Miami (OH)", 11,
           "Miami (OH)", "RedHawks upset Vols! Anti-SEC strikes")
print_game("R64", "Midwest", "Virginia", 3, "Wright State", 14,
           "Virginia", "Cavaliers advance")
print_game("R64", "Midwest", "Kentucky", 7, "Santa Clara", 10,
           "Kentucky", "Wildcats! Cat mascot gets the boost")
print_game("R64", "Midwest", "Iowa State", 2, "Tennessee St", 15,
           "Iowa State", "Tennessee St Tigers are cats but Cyclones too strong")

cat("\n  Round of 32:\n")
print_game("R32", "Midwest", "Michigan", 1, "Georgia", 8,
           "Michigan", "Wolverines dispatch the Bulldogs")
print_game("R32", "Midwest", "Texas Tech", 5, "Hofstra", 13,
           "Hofstra", "Lion Pride Cinderella run continues!")
print_game("R32", "Midwest", "Miami (OH)", 11, "Virginia", 3,
           "Virginia", "Cavaliers too much for RedHawks")
print_game("R32", "Midwest", "Kentucky", 7, "Iowa State", 2,
           "Kentucky", "Wildcats! Cat upset over Cyclones!")

cat("\n  Sweet 16:\n")
print_game("S16", "Midwest", "Michigan", 1, "Hofstra", 13,
           "Michigan", "Sorry Pride, Minzy's champion must advance")
print_game("S16", "Midwest", "Virginia", 3, "Kentucky", 7,
           "Kentucky", "Wildcats in the Elite 8! Cat power!")

cat("\n  Elite 8:\n")
print_game("E8", "Midwest", "Michigan", 1, "Kentucky", 7,
           "Michigan", "Wolverines defeat the Wildcats. Minzy's pick must advance")

cat("\n  Midwest Champion: (1) Michigan Wolverines\n")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL FOUR
# ══════════════════════════════════════════════════════════════════════════════
cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("  FINAL FOUR  --  Indianapolis\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

cat("\n  Semifinal 1: East vs South\n")
print_game("F4", "", "UConn", 2, "Houston", 2,
           "Houston", "Cat vs Dog: COUGARS defeat Huskies! Cats always win")

cat("\n  Semifinal 2: West vs Midwest\n")
print_game("F4", "", "Purdue", 2, "Michigan", 1,
           "Michigan", "Big Ten vs Big Ten! Wolverines edge Boilermakers")

# ══════════════════════════════════════════════════════════════════════════════
# CHAMPIONSHIP
# ══════════════════════════════════════════════════════════════════════════════
cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("  NATIONAL CHAMPIONSHIP\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

cat("\n")
print_game("Championship", "", "Houston", 2, "Michigan", 1,
           "Michigan", "Minzy's pick! Wolverines are national champions!")

cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("  NATIONAL CHAMPION: (1) MICHIGAN WOLVERINES\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

# ══════════════════════════════════════════════════════════════════════════════
# BRACKET SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
cat("\n")
cat("  ── MINZY'S KEY PICKS ──\n\n")
cat("  Cat Mascot Wins:        11 (Arizona, Villanova, High Point, BYU,\n")
cat("                              Missouri, Clemson x3, Houston x3,\n")
cat("                              Northern Iowa, Hofstra x2, Kentucky x3)\n")
cat("  Dog Mascot Wins:         3 (UConn x4, Gonzaga, Georgia)\n")
cat("  Cat vs Dog record:     2-1 (BYU > Gonzaga, Houston > UConn,\n")
cat("                              Michigan > Georgia)\n")
cat("  Duke exit:             R32 (lost to TCU)\n")
cat("  UNC exit:              S16 (further than Duke!)\n")
cat("  Ohio State exit:       R64 (lost to TCU)\n")
cat("  Michigan State exit:   S16 (no Final Four)\n")
cat("  Biggest cat upset:     (13) Hofstra Pride over (4) Alabama\n")
cat("  Cinderella:            High Point Panthers to Sweet 16\n")
cat("  Final Four:            UConn, Houston, Purdue, Michigan\n")
cat("  Champion:              Michigan Wolverines\n")
cat("\n")
cat("  Minzy approves this bracket.\n\n")
