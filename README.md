# College Basketball Prediction Model

An XGBoost model for predicting NCAA Division I college basketball game outcomes, simulating March Madness brackets, and identifying market inefficiencies on Kalshi.

## Setup

```bash
cd cbb-prediction-model
source .venv/bin/activate
```

Database: SQLite at `data/cbb_prediction.db` (gitignored).

## Predicting Games

### Single matchup prediction

Predict a head-to-head matchup between any two teams. Uses full team names, ESPN IDs, or abbreviations. Games are assumed neutral-site by default.

```bash
# By team name (neutral site)
PYTHONPATH=. python scripts/tournament_predict.py --matchup "Duke" "TCU"

# With a specific venue (affects travel distance features)
PYTHONPATH=. python scripts/tournament_predict.py --matchup "Illinois" "UConn" --venue "Indianapolis"

# Non-neutral game: specify the home team with --home
PYTHONPATH=. python scripts/tournament_predict.py --matchup "Duke" "TCU" --home "Duke"

# Home game with venue for travel calculation
PYTHONPATH=. python scripts/tournament_predict.py --matchup "Duke" "TCU" --home "Duke" --venue "Durham"

# List available venue cities
PYTHONPATH=. python scripts/tournament_predict.py --list-venues
```

The `--home` flag sets `is_home`/`is_away` indicators in the model, which capture home-court advantage. Without it, the game is treated as neutral-site. Combine with `--venue` to also account for travel distance.

### Predict all scheduled tournament games

Pulls upcoming games from ESPN and predicts each one.

```bash
PYTHONPATH=. python scripts/tournament_predict.py --predict-scheduled
```

### Update data + predict

Fetches latest tournament results, box scores, and retrains the model before predicting.

```bash
# Update data only
PYTHONPATH=. python scripts/tournament_predict.py --update

# Update data then predict all scheduled games
PYTHONPATH=. python scripts/tournament_predict.py --update --predict-scheduled
```

### Monte Carlo tournament simulation

Simulates the full tournament bracket 100,000 times to produce advancement probabilities for every team.

```bash
# Default (100k sims)
PYTHONPATH=. python scripts/simulate_tournament.py

# Fewer sims for faster results
PYTHONPATH=. python scripts/simulate_tournament.py --sims 10000

# Print bracket matchups
PYTHONPATH=. python scripts/simulate_tournament.py --print-bracket

# Export advancement probabilities to CSV
PYTHONPATH=. python scripts/simulate_tournament.py --export-probs

# Simulate only the Final Four onward
PYTHONPATH=. python scripts/simulate_tournament.py --final-four

# Ignore injury adjustments
PYTHONPATH=. python scripts/simulate_tournament.py --no-injury
```

## Injury Reports

The model uses injury data to adjust roster strength features (BPM, depth, star count). Injuries flow through a three-layer eligibility system:

1. **Manual overrides** (highest priority) — `data/player_overrides.json`
2. **Scraped injury reports** — `player_injury_reports` DB table (from RotoWire)
3. **Cloglog GLM fallback** — statistical model based on box score history

### Scrape injury data from RotoWire

```bash
# Scrape all D1 teams
PYTHONPATH=. python src/data/fetch_injuries.py

# Scrape specific teams only
PYTHONPATH=. python src/data/fetch_injuries.py --teams Illinois Michigan UConn Arizona

# Scrape Final Four teams only
PYTHONPATH=. python src/data/fetch_injuries.py --final-four

# Preview what would be scraped (no DB write)
PYTHONPATH=. python src/data/fetch_injuries.py --dry-run
PYTHONPATH=. python src/data/fetch_injuries.py --final-four --dry-run
```

### Verify injury data in the database

After scraping, check what the model currently knows about injuries. This shows every injury record in the DB for the specified teams so you can cross-reference against what you know.

```bash
# Verify Final Four teams
PYTHONPATH=. python src/data/fetch_injuries.py --verify --final-four

# Verify specific teams
PYTHONPATH=. python src/data/fetch_injuries.py --verify --teams Illinois Michigan

# Verify all teams
PYTHONPATH=. python src/data/fetch_injuries.py --verify
```

**How to tell if the scrape was successful:**
- The scraper prints how many entries it received from RotoWire and how many it inserted.
- Run `--verify` afterward to see the full picture for your teams.
- If a team shows "no injury records," it means either (a) no one is injured or (b) RotoWire hasn't reported the injury yet.
- If "Unmatched teams" appear after insertion, those teams' injuries were dropped because the team name couldn't be mapped to the DB.

### Add injuries manually

If RotoWire is missing a known injury, create or edit `data/player_overrides.json`:

```json
{
  "overrides": [
    {
      "player_name": "John Smith",
      "team": "356",
      "status": "OUT",
      "note": "ACL tear, announced 3/30"
    }
  ]
}
```

- `team` can be an ESPN ID (numeric) or team name
- `status` options: `OUT`, `DOUBTFUL`, `QUESTIONABLE`, `PROBABLE`, `AVAILABLE`
- Manual overrides take highest priority — they override both scraped data and the statistical model

### Typical pre-game workflow

```bash
# 1. Scrape latest injuries
PYTHONPATH=. python src/data/fetch_injuries.py --final-four

# 2. Verify — cross-reference against what you know
PYTHONPATH=. python src/data/fetch_injuries.py --verify --final-four

# 3. (If needed) Add missing injuries to data/player_overrides.json

# 4. Predict the game
PYTHONPATH=. python scripts/tournament_predict.py --matchup "Illinois" "UConn" --venue "Indianapolis"
```

## Project Structure

```
src/
├── data/           # Data collection and pipeline modules
├── models/         # Feature engineering, training, eligibility
├── api/            # Kalshi API integration
├── utils.py        # Shared utilities (normalize_name, team lookup)
├── config.py       # Model params, venue coords, matchup builder
scripts/
├── tournament_predict.py       # Game predictions
├── simulate_tournament.py      # Monte Carlo bracket simulation
├── monthly_accuracy_analysis.py
├── betting_backtest.py
data/               # Local data storage (gitignored)
├── player_overrides.json       # Manual injury overrides
├── paper/                      # Model analysis artifacts
```

## Tech Stack

- **ML Framework**: XGBoost
- **Database**: SQLite
- **Data Sources**: ESPN API, Barttorvik, RotoWire
- **Language**: Python 3.10+

## License

Private repository - All rights reserved.
