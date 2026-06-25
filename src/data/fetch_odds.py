"""
Fetch historical betting odds from SportsData.io API.

Pulls pregame moneylines, spreads, and over/unders for all games
in our database. Matches games using the team_crosswalk table that
maps SportsData.io TeamIDs to our ESPN team IDs.

Usage:
    python src/data/fetch_odds.py                    # Fetch all missing
    python src/data/fetch_odds.py --seasons 2026     # Single season
    python src/data/fetch_odds.py --rate-limit 5     # Slower (safer for API limits)
"""

import os
import argparse
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection

API_BASE = "https://api.sportsdata.io/v3/cbb/odds/json"
API_KEY = os.environ.get("ODDS_API_KEY", "")


def moneyline_to_prob(ml: int) -> float:
    """Convert American moneyline odds to implied probability."""
    if ml is None:
        return None
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


def get_game_dates(conn, seasons: list[int] = None) -> list[str]:
    """Get unique game dates that still need odds data."""
    query = """
        SELECT DISTINCT DATE(date) as game_date
        FROM games
        WHERE status = 'STATUS_FINAL'
        AND game_id NOT IN (SELECT game_id FROM game_odds)
    """
    params = []
    if seasons:
        placeholders = ','.join('?' * len(seasons))
        query += f" AND season_year IN ({placeholders})"
        params.extend(seasons)
    query += " ORDER BY game_date"

    rows = conn.execute(query, params).fetchall()
    return [row['game_date'] for row in rows]


def build_sd_to_espn_map(conn) -> dict:
    """Build SportsData TeamID -> ESPN team ID mapping from crosswalk."""
    rows = conn.execute(
        "SELECT espn_id, sportsdata_id FROM team_crosswalk"
    ).fetchall()
    return {row['sportsdata_id']: row['espn_id'] for row in rows}


def fetch_odds_for_date(date_str: str) -> list[dict] | None:
    """Fetch game odds from SportsData.io for a specific date.

    Returns None on 401 (unauthorized / outside trial coverage).
    """
    url = f"{API_BASE}/GameOddsByDate/{date_str}"
    resp = requests.get(url, params={"key": API_KEY}, timeout=30)
    if resp.status_code == 401:
        return None
    resp.raise_for_status()
    return resp.json()


def get_consensus_odds(pregame_odds: list[dict]) -> dict | None:
    """Average odds across all available sportsbooks for a game."""
    if not pregame_odds:
        return None

    # Filter to pregame full-game odds with both moneylines present
    valid = [o for o in pregame_odds
             if o.get('OddType') == 'pregame'
             and o.get('HomeMoneyLine') is not None
             and o.get('AwayMoneyLine') is not None]

    if not valid:
        return None

    n = len(valid)
    home_ml = sum(o['HomeMoneyLine'] for o in valid) / n
    away_ml = sum(o['AwayMoneyLine'] for o in valid) / n

    sp_home = [o['HomePointSpread'] for o in valid if o.get('HomePointSpread') is not None]
    sp_away = [o['AwayPointSpread'] for o in valid if o.get('AwayPointSpread') is not None]
    ou_vals = [o['OverUnder'] for o in valid if o.get('OverUnder') is not None]

    return {
        'home_moneyline': round(home_ml),
        'away_moneyline': round(away_ml),
        'home_spread': round(sum(sp_home) / len(sp_home), 1) if sp_home else None,
        'away_spread': round(sum(sp_away) / len(sp_away), 1) if sp_away else None,
        'over_under': round(sum(ou_vals) / len(ou_vals), 1) if ou_vals else None,
        'home_implied_prob': moneyline_to_prob(round(home_ml)),
        'away_implied_prob': moneyline_to_prob(round(away_ml)),
        'num_sportsbooks': n,
    }


def main():
    parser = argparse.ArgumentParser(description='Fetch historical betting odds')
    parser.add_argument('--seasons', nargs='+', type=int, help='Season years')
    parser.add_argument('--rate-limit', type=float, default=3.0,
                        help='Seconds between API calls (default: 3.0)')
    parser.add_argument('--limit', type=int, help='Max dates to fetch')
    args = parser.parse_args()

    conn = get_connection()

    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_odds (
            game_id INTEGER PRIMARY KEY REFERENCES games(game_id),
            sportsdata_game_id INTEGER,
            home_moneyline INTEGER,
            away_moneyline INTEGER,
            home_spread REAL,
            away_spread REAL,
            over_under REAL,
            home_implied_prob REAL,
            away_implied_prob REAL,
            num_sportsbooks INTEGER,
            odds_source TEXT DEFAULT 'sportsdata.io',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Build TeamID crosswalk
    sd_to_espn = build_sd_to_espn_map(conn)
    if not sd_to_espn:
        print("ERROR: team_crosswalk table is empty. Run scripts/build_team_crosswalk.py first.")
        conn.close()
        return

    print(f"Loaded {len(sd_to_espn)} team crosswalk entries")

    # Get dates that need odds
    dates = get_game_dates(conn, args.seasons)
    if args.limit:
        dates = dates[:args.limit]

    print(f"Found {len(dates)} dates needing odds data")

    if not dates:
        print("All games already have odds data")
        conn.close()
        return

    matched = 0
    unmatched = 0
    no_odds = 0
    total_games = 0

    pbar = tqdm(dates, desc="Fetching odds", unit="date")

    skipped_401 = 0

    for date_str in pbar:
        try:
            sd_games = fetch_odds_for_date(date_str)
        except Exception as e:
            pbar.write(f"Error fetching {date_str}: {e}")
            time.sleep(args.rate_limit)
            continue

        if sd_games is None:
            # 401 = outside trial coverage, skip
            skipped_401 += 1
            pbar.set_postfix(matched=matched, no_odds=no_odds,
                             noMatch=unmatched, skipped=skipped_401)
            time.sleep(0.5)  # brief pause, no need for full rate limit
            continue

        # Build lookup: (espn_home_id, espn_away_id) -> sd_game
        sd_lookup = {}
        for sd_game in sd_games:
            sd_home = sd_game.get('HomeTeamId')
            sd_away = sd_game.get('AwayTeamId')
            espn_home = sd_to_espn.get(sd_home)
            espn_away = sd_to_espn.get(sd_away)
            if espn_home and espn_away:
                sd_lookup[(espn_home, espn_away)] = sd_game
                # Also store reversed for neutral-site matching
                sd_lookup[(espn_away, espn_home)] = sd_game

        # Get our games for this date
        our_games = conn.execute("""
            SELECT game_id, home_team_id, away_team_id
            FROM games
            WHERE DATE(date) = ?
            AND status = 'STATUS_FINAL'
            AND game_id NOT IN (SELECT game_id FROM game_odds)
        """, (date_str,)).fetchall()

        total_games += len(our_games)

        for our_game in our_games:
            key = (our_game['home_team_id'], our_game['away_team_id'])
            sd_game = sd_lookup.get(key)

            if sd_game:
                odds = get_consensus_odds(sd_game.get('PregameOdds', []))
                if odds:
                    conn.execute("""
                        INSERT OR IGNORE INTO game_odds
                        (game_id, sportsdata_game_id,
                         home_moneyline, away_moneyline,
                         home_spread, away_spread, over_under,
                         home_implied_prob, away_implied_prob,
                         num_sportsbooks)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        our_game['game_id'],
                        sd_game.get('GameId'),
                        odds['home_moneyline'],
                        odds['away_moneyline'],
                        odds['home_spread'],
                        odds['away_spread'],
                        odds['over_under'],
                        odds['home_implied_prob'],
                        odds['away_implied_prob'],
                        odds['num_sportsbooks'],
                    ))
                    matched += 1
                else:
                    no_odds += 1
            else:
                unmatched += 1

        conn.commit()
        pbar.set_postfix(matched=matched, no_odds=no_odds, noMatch=unmatched)
        time.sleep(args.rate_limit)

    pbar.close()
    conn.close()

    print(f"\nDone:")
    print(f"  {total_games} games processed across {len(dates)} dates")
    print(f"  {matched} matched with odds")
    print(f"  {no_odds} matched but no pregame odds available")
    print(f"  {unmatched} could not match to SportsData.io")
    if skipped_401:
        print(f"  {skipped_401} dates skipped (401 unauthorized / outside trial)")
    if total_games > 0:
        print(f"  Match rate: {(matched + no_odds) / total_games:.1%}")


if __name__ == '__main__':
    main()
