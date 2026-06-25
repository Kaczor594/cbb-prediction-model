# ── Sammie's 2026 March Madness Bracket ────────────────────────────────────────
#
# A custom bracket for Sammie, built from model probabilities with
# the following preferences:
#   - UNC alum (MSLS) — Tar Heels go deep
#   - Duke absolutely cannot win anything ever
#   - Michigan is not welcome here (sorry Isaac)
#   - Michigan State should win it all
#   - Cat & dog mascot teams get a slight boost in close games
#   - Bird mascot teams are PENALIZED (Jayhawks, Cardinals, Hawkeyes,
#     Owls, RedHawks — all going down)
#   - Slight ACC bias
#   - Likes Purdue (great mascot, hilarious name)
#   - Anti-SEC and anti-Ohio State
#   - Library science program quality as tiebreaker
#
# Usage in RStudio:
#   source("scripts/sammie_bracket.R")

library(tidyverse)

# ── Define Sammie's picks ──────────────────────────────────────────────────────

cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("             SAMMIE'S 2026 MARCH MADNESS BRACKET\n")
cat("        (UNC forever, birds never, libraries always)\n")
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
           "Duke", "Ugh. Duke wins but not for long")
print_game("R64", "East", "Ohio State", 8, "TCU", 9,
           "TCU", "Ohio State can leave immediately")
print_game("R64", "East", "St. John's", 5, "Northern Iowa", 12,
           "Northern Iowa", "Panthers! Cat upset special")
print_game("R64", "East", "Kansas", 4, "Cal Baptist", 13,
           "Cal Baptist", "JAYHAWKS ARE BIRDS. Birds lose. Lancers upset!")
print_game("R64", "East", "Louisville", 6, "South Florida", 11,
           "South Florida", "Cardinals are BIRDS. ACC bias can't save birds")
print_game("R64", "East", "Michigan State", 3, "North Dakota St", 14,
           "Michigan State", "Sammie's champion starts rolling")
print_game("R64", "East", "UCLA", 7, "UCF", 10,
           "UCLA", "Bruins advance")
print_game("R64", "East", "UConn", 2, "Furman", 15,
           "UConn", "Huskies! Good dogs")

cat("\n  Round of 32:\n")
print_game("R32", "East", "Duke", 1, "TCU", 9,
           "TCU", "DUKE IS GONE! Horned Frogs end the nightmare!")
print_game("R32", "East", "Northern Iowa", 12, "Cal Baptist", 13,
           "Northern Iowa", "Panthers vs Lancers! Cats win the Cinderella battle")
print_game("R32", "East", "South Florida", 11, "Michigan State", 3,
           "Michigan State", "Spartans are too strong for the Bulls")
print_game("R32", "East", "UCLA", 7, "UConn", 2,
           "UConn", "Huskies are good boys and also good at basketball")

cat("\n  Sweet 16:\n")
print_game("S16", "East", "TCU", 9, "Northern Iowa", 12,
           "TCU", "Panthers had an amazing run but Horned Frogs advance")
print_game("S16", "East", "Michigan State", 3, "UConn", 2,
           "Michigan State", "Spartans over Huskies. Sorry dogs, Sammie needs MSU")

cat("\n  Elite 8:\n")
print_game("E8", "East", "TCU", 9, "Michigan State", 3,
           "Michigan State", "Spartans to the Final Four! Sammie's pick rolls on")

cat("\n  East Champion: (3) Michigan State Spartans\n")

# ══════════════════════════════════════════════════════════════════════════════
# WEST REGION
# ══════════════════════════════════════════════════════════════════════════════
cat("\n  ── WEST REGION ──\n")

cat("\n  Round of 64:\n")
print_game("R64", "West", "Arizona", 1, "Long Island", 16,
           "Arizona", "Wildcats! Cats are good")
print_game("R64", "West", "Villanova", 8, "Utah State", 9,
           "Villanova", "More Wildcats! Sammie approves")
print_game("R64", "West", "Wisconsin", 5, "High Point", 12,
           "High Point", "PANTHERS! Cat upset! Also Wisconsin's library school closed, so...")
print_game("R64", "West", "Arkansas", 4, "Hawai'i", 13,
           "Arkansas", "Razorbacks too much for Rainbow Warriors")
print_game("R64", "West", "BYU", 6, "Texas", 11,
           "BYU", "Cougars! Big cats win")
print_game("R64", "West", "Gonzaga", 3, "Kennesaw St", 14,
           "Gonzaga", "Bulldogs over Owls! OWLS ARE BIRDS. Dog > Bird every time")
print_game("R64", "West", "Miami (FL)", 7, "Missouri", 10,
           "Missouri", "Tigers! Cat mascot takes it")
print_game("R64", "West", "Purdue", 2, "Queens (NC)", 15,
           "Purdue", "BOILER UP! Best mascot in sports, honestly")

cat("\n  Round of 32:\n")
print_game("R32", "West", "Arizona", 1, "Villanova", 8,
           "Arizona", "Wildcat vs Wildcat! Desert cat is the bigger cat")
print_game("R32", "West", "High Point", 12, "Arkansas", 4,
           "High Point", "Panthers Cinderella! Anti-SEC helps the cats")
print_game("R32", "West", "BYU", 6, "Gonzaga", 3,
           "BYU", "Cat vs Dog — Cougars take this one")
print_game("R32", "West", "Missouri", 10, "Purdue", 2,
           "Purdue", "Sorry Tigers, Boilermaker Pete cannot be stopped")

cat("\n  Sweet 16:\n")
print_game("S16", "West", "Arizona", 1, "High Point", 12,
           "Arizona", "Wildcats win the cat showdown. Panthers had a great run")
print_game("S16", "West", "BYU", 6, "Purdue", 2,
           "Purdue", "Boilermakers! The name! The train! The hammer!")

cat("\n  Elite 8:\n")
print_game("E8", "West", "Arizona", 1, "Purdue", 2,
           "Purdue", "Boilermakers knock off the Wildcats! Choo choo!")

cat("\n  West Champion: (2) Purdue Boilermakers\n")

# ══════════════════════════════════════════════════════════════════════════════
# SOUTH REGION
# ══════════════════════════════════════════════════════════════════════════════
cat("\n  ── SOUTH REGION ──\n")

cat("\n  Round of 64:\n")
print_game("R64", "South", "Florida", 1, "Prairie View A&M", 16,
           "Florida", "Panthers are cats but Gators are a 1 seed")
print_game("R64", "South", "Clemson", 8, "Iowa", 9,
           "Clemson", "Tigers over Hawkeyes! CAT beats BIRD! ACC beats bird!")
print_game("R64", "South", "Vanderbilt", 5, "McNeese", 12,
           "McNeese", "Cowboys upset SEC Commodores! Anti-SEC strikes")
print_game("R64", "South", "Nebraska", 4, "Troy", 13,
           "Nebraska", "Cornhuskers advance")
print_game("R64", "South", "North Carolina", 6, "VCU", 11,
           "North Carolina", "TAR HEELS! Sammie's alma mater! MSLS represent!")
print_game("R64", "South", "Illinois", 3, "Penn", 14,
           "Illinois", "#1 ranked library science program rolls on")
print_game("R64", "South", "Saint Mary's", 7, "Texas A&M", 10,
           "Saint Mary's", "Gaels over Aggies. Anti-SEC, easy call")
print_game("R64", "South", "Houston", 2, "Idaho", 15,
           "Houston", "Cougars! Big cat energy")

cat("\n  Round of 32:\n")
print_game("R32", "South", "Florida", 1, "Clemson", 8,
           "Clemson", "TIGERS upset Gators! ACC cat > SEC reptile")
print_game("R32", "South", "McNeese", 12, "Nebraska", 4,
           "Nebraska", "Cowboys' run ends. Cornhuskers too much")
print_game("R32", "South", "North Carolina", 6, "Illinois", 3,
           "North Carolina", "BATTLE OF THE iSCHOOLS! #1 vs #3 library programs. Sammie has the degree from one of them. Go Heels!")
print_game("R32", "South", "Saint Mary's", 7, "Houston", 2,
           "Houston", "Cougars power through")

cat("\n  Sweet 16:\n")
print_game("S16", "South", "Clemson", 8, "Nebraska", 4,
           "Clemson", "Tigers keep roaring! ACC cat magic continues")
print_game("S16", "South", "North Carolina", 6, "Houston", 2,
           "North Carolina", "TAR HEELS OVER COUGARS! Sorry cats, this is Sammie's team!")

cat("\n  Elite 8:\n")
print_game("E8", "South", "Clemson", 8, "North Carolina", 6,
           "North Carolina", "All-ACC Elite 8! Tar Heels win! Sammie's team to the Final Four!")

cat("\n  South Champion: (6) North Carolina Tar Heels\n")

# ══════════════════════════════════════════════════════════════════════════════
# MIDWEST REGION
# ══════════════════════════════════════════════════════════════════════════════
cat("\n  ── MIDWEST REGION ──\n")

cat("\n  Round of 64:\n")
print_game("R64", "Midwest", "Michigan", 1, "Howard", 16,
           "Michigan", "Fine. Michigan wins a 1-16 game. Enjoy it while it lasts")
print_game("R64", "Midwest", "Georgia", 8, "Saint Louis", 9,
           "Georgia", "Bulldogs! Dogs are fine in Sammie's world")
print_game("R64", "Midwest", "Texas Tech", 5, "Akron", 12,
           "Texas Tech", "Red Raiders advance")
print_game("R64", "Midwest", "Alabama", 4, "Hofstra", 13,
           "Hofstra", "PRIDE upset! Lions > SEC Crimson Tide! Cat energy + anti-SEC!")
print_game("R64", "Midwest", "Tennessee", 6, "Miami (OH)", 11,
           "Tennessee", "RedHawks are BIRDS. Sammie picks SEC over birds. That's how much she hates birds")
print_game("R64", "Midwest", "Virginia", 3, "Wright State", 14,
           "Virginia", "Cavaliers advance! ACC!")
print_game("R64", "Midwest", "Kentucky", 7, "Santa Clara", 10,
           "Kentucky", "Wildcats! Cat mascot boost")
print_game("R64", "Midwest", "Iowa State", 2, "Tennessee St", 15,
           "Iowa State", "Cyclones are too strong even for Tiger cats")

cat("\n  Round of 32:\n")
print_game("R32", "Midwest", "Michigan", 1, "Georgia", 8,
           "Georgia", "WOLVERINES GO HOME! Bulldogs upset Michigan! Sammie sends her regards")
print_game("R32", "Midwest", "Texas Tech", 5, "Hofstra", 13,
           "Hofstra", "Pride Cinderella continues! Lions roar!")
print_game("R32", "Midwest", "Tennessee", 6, "Virginia", 3,
           "Virginia", "ACC Cavaliers dispatch SEC Vols! ACC > SEC")
print_game("R32", "Midwest", "Kentucky", 7, "Iowa State", 2,
           "Kentucky", "WILDCATS upset Cyclones! Cat power!")

cat("\n  Sweet 16:\n")
print_game("S16", "Midwest", "Georgia", 8, "Hofstra", 13,
           "Georgia", "Dogs end the Pride's incredible run. Bulldogs advance")
print_game("S16", "Midwest", "Virginia", 3, "Kentucky", 7,
           "Virginia", "ACC Cavaliers over Wildcats. Sorry cats, ACC bias wins")

cat("\n  Elite 8:\n")
print_game("E8", "Midwest", "Georgia", 8, "Virginia", 3,
           "Virginia", "ACC to the Final Four! Cavaliers are dancing!")

cat("\n  Midwest Champion: (3) Virginia Cavaliers\n")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL FOUR
# ══════════════════════════════════════════════════════════════════════════════
cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("  FINAL FOUR  --  Indianapolis\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

cat("\n  Semifinal 1: East vs South\n")
print_game("F4", "", "Michigan State", 3, "North Carolina", 6,
           "Michigan State", "Sammie's heart breaks but MSU is her pick to win it all. UNC had a great run")

cat("\n  Semifinal 2: West vs Midwest\n")
print_game("F4", "", "Purdue", 2, "Virginia", 3,
           "Purdue", "Boilermakers over ACC! Boilermaker Pete dances in Indy!")

# ══════════════════════════════════════════════════════════════════════════════
# CHAMPIONSHIP
# ══════════════════════════════════════════════════════════════════════════════
cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("  NATIONAL CHAMPIONSHIP\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

cat("\n")
print_game("Championship", "", "Michigan State", 3, "Purdue", 2,
           "Michigan State", "SPARTANS WIN IT ALL! Two teams Sammie likes, MSU takes the crown!")

cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("  NATIONAL CHAMPION: (3) MICHIGAN STATE SPARTANS\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

# ══════════════════════════════════════════════════════════════════════════════
# BRACKET SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
cat("\n")
cat("  ── SAMMIE'S KEY PICKS ──\n\n")
cat("  UNC exit:              Final Four (lost to eventual champ MSU)\n")
cat("  Duke exit:             R32 (lost to TCU — beautiful)\n")
cat("  Michigan exit:         R32 (Georgia Bulldogs send them packing)\n")
cat("  Ohio State exit:       R64 (first to go, as it should be)\n")
cat("  Bird teams record:     0-5 (Kansas R64, Louisville R64, Iowa R64,\n")
cat("                               Kennesaw St R64, Miami OH R64)\n")
cat("  Cat mascot wins:       Northern Iowa, High Point x2, BYU x2,\n")
cat("                         Clemson x3, Missouri, Houston, Hofstra x2,\n")
cat("                         Kentucky x2, Arizona x2\n")
cat("  Dog mascot wins:       UConn x2, Gonzaga, Georgia x3\n")
cat("  ACC teams in S16:      Clemson, UNC, Virginia (3 of 4!)\n")
cat("  Library science game:  UNC over Illinois R32 (iSchool showdown!)\n")
cat("  Biggest upset:         (13) Cal Baptist over (4) Kansas (bye birds)\n")
cat("  Cinderella:            Hofstra Pride to Sweet 16\n")
cat("  Final Four:            Michigan State, UNC, Purdue, Virginia\n")
cat("  Champion:              Michigan State Spartans\n")
cat("\n")
cat("  Sammie approves this bracket. No birds were left standing.\n\n")
