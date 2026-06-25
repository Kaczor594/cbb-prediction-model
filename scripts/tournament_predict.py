"""
Tournament Update & Predict — Continuous tournament prediction workflow.

Fetches latest tournament results from ESPN, updates the database,
fetches box scores, rebuilds the XGBoost model with all available data
(regular season + tournament), and predicts upcoming matchups.

Usage:
    # Update data and predict all upcoming R32 games
    python scripts/tournament_predict.py --update

    # Predict a specific matchup (by team name)
    python scripts/tournament_predict.py --matchup "Duke" "TCU"

    # Predict a matchup at a specific venue
    python scripts/tournament_predict.py --matchup "Duke" "TCU" --venue "Greenville"

    # Predict a home game for Duke
    python scripts/tournament_predict.py --matchup "Duke" "TCU" --home "Duke"

    # Home game with venue for travel calculation
    python scripts/tournament_predict.py --matchup "Duke" "TCU" --home "Duke" --venue "Durham"

    # Update data + predict all scheduled games
    python scripts/tournament_predict.py --update --predict-scheduled

    # Just predict scheduled games (no data update)
    python scripts/tournament_predict.py --predict-scheduled

    # List available venue cities
    python scripts/tournament_predict.py --list-venues
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.data.espn_client import ESPNClient, extract_game_data
from src.data.fetch_injuries import scrape_injuries
from src.models.feature_engineering import (
    build_features, load_games, haversine_miles,
)
from src.models.train_model import select_features
from src.config import (FINAL_FOUR_NAMES, VENUE_COORDS,
                        build_team_snapshots, build_matchup_features, predict_symmetric,
                        load_team_locations, build_trained_model)

# ── Data update functions ─────────────────────────────────────────────

def fetch_tournament_results(conn, dates=None):
    """
    Fetch completed tournament games from ESPN scoreboard and insert
    into the database. Returns count of new games added.
    """
    from src.data.database import upsert_game

    client = ESPNClient()
    new_games = 0
    updated_games = 0

    if dates is None:
        # Check recent dates (last 7 days)
        from datetime import datetime, timedelta
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime('%Y%m%d') for i in range(7)]

    for date_str in dates:
        games = client.get_scoreboard(date=date_str)
        for game in games:
            season_type = game.get('season', {}).get('type')
            if season_type != 3:  # Only tournament games
                continue

            status = game.get('status', {}).get('type', {}).get('name', '')
            if status != 'STATUS_FINAL':
                continue

            data = extract_game_data(game)
            # Fix season_type for scoreboard format
            if data['season_type'] is None:
                data['season_type'] = season_type

            # Check if we already have this game
            existing = conn.execute(
                "SELECT game_id FROM games WHERE espn_id = ?",
                (data['game_id'],)
            ).fetchone()

            upsert_game(conn, data)

            if existing:
                updated_games += 1
            else:
                new_games += 1

    conn.commit()
    return new_games, updated_games


def fetch_tournament_box_scores(conn):
    """Fetch box scores for tournament games that don't have them yet."""
    from src.data.fetch_box_scores import extract_player_stats, insert_player_stats

    client = ESPNClient()

    # Find tournament games without box scores
    rows = conn.execute("""
        SELECT g.game_id, g.espn_id
        FROM games g
        WHERE g.season_type = 3
          AND g.status = 'STATUS_FINAL'
          AND g.espn_id NOT IN (
              SELECT DISTINCT CAST(g2.espn_id AS TEXT)
              FROM games g2
              JOIN player_game_stats pgs ON pgs.game_id = g2.game_id
          )
    """).fetchall()

    if not rows:
        print("  All tournament games already have box scores.")
        return 0

    print(f"  Fetching box scores for {len(rows)} tournament games...")
    fetched = 0
    errors = 0

    for row in rows:
        game_id = row['game_id']
        espn_id = row['espn_id']
        try:
            summary = client.get_game_summary(str(espn_id))
            stats = extract_player_stats(espn_id, game_id, summary)
            if stats:
                insert_player_stats(conn, stats)
                fetched += 1
            time.sleep(0.5)
        except Exception as e:
            errors += 1
            if 'FOREIGN KEY' in str(e):
                pass  # Expected for Queens (NC) etc.
            else:
                print(f"    Error for game {espn_id}: {e}")

    conn.commit()
    print(f"  Fetched {fetched} box scores ({errors} errors)")
    return fetched


# ── Model building ────────────────────────────────────────────────────

def build_model_and_snapshots(conn):
    """
    Build feature matrix, train XGBoost on all available data,
    and create team snapshots for prediction.

    Scrapes current injury reports so OUT players get P(plays)=0 in the
    eligibility model and are effectively removed from roster BPM.

    Returns: (model, features, snapshots, team_locs)
    """
    # Scrape injuries before building features so eligibility picks them up
    scrape_injuries(conn, filter_teams=FINAL_FOUR_NAMES)

    print("\nBuilding feature matrix...")
    df = build_features(conn)

    features = select_features(df)
    print(f"Using {len(features)} features")

    # Train on all completed games (with augmentation for symmetry)
    train_df = df.dropna(subset=['home_winner'])
    print(f"Training on {len(train_df)} games (seasons: "
          f"{sorted(train_df['season_year'].unique())})")
    model = build_trained_model(train_df, features)

    # Build team snapshots from 2026 data
    snapshots = build_team_snapshots(df, season=2026)

    team_locs = load_team_locations(conn)

    print(f"Built snapshots for {len(snapshots)} teams")
    return model, features, snapshots, team_locs


# ── Prediction ────────────────────────────────────────────────────────

def predict_matchup(model, features, snapshots, team_locs,
                    team_a_id, team_b_id, team_a_name, team_b_name,
                    venue_city=None, neutral=True):
    """Predict a single matchup. Returns P(team_a wins)."""
    if team_a_id is None or team_b_id is None:
        return 0.85 if team_b_id is None else 0.15

    feat_df = build_matchup_features(
        team_a_id, team_b_id, snapshots, features,
        neutral=neutral, venue_city=venue_city, team_locs=team_locs,
    )
    prob_a = float(predict_symmetric(model, feat_df, features)[0])
    return prob_a


def find_team(conn, name_query):
    """Find a team by partial name match. Returns (espn_id, name) or None."""
    query = name_query.strip().lower()

    # Try exact match first
    row = conn.execute(
        "SELECT espn_id, name FROM teams WHERE LOWER(name) = ?", (query,)
    ).fetchone()
    if row:
        return row['espn_id'], row['name']

    # Try ESPN ID (numeric)
    if query.isdigit():
        row = conn.execute(
            "SELECT espn_id, name FROM teams WHERE espn_id = ?",
            (int(query),)
        ).fetchone()
        if row:
            return row['espn_id'], row['name']

    # Try abbreviation
    row = conn.execute(
        "SELECT espn_id, name FROM teams WHERE LOWER(abbreviation) = ?",
        (query,)
    ).fetchone()
    if row:
        return row['espn_id'], row['name']

    return None, None


def get_scheduled_tournament_games(conn):
    """Get upcoming/in-progress tournament games from ESPN scoreboard."""
    from datetime import datetime, timedelta

    client = ESPNClient()
    today = datetime.now()
    scheduled = []

    for day_offset in range(7):  # Check next 7 days
        date_str = (today + timedelta(days=day_offset)).strftime('%Y%m%d')
        games = client.get_scoreboard(date=date_str)

        for game in games:
            season_type = game.get('season', {}).get('type')
            if season_type != 3:
                continue

            status = game.get('status', {}).get('type', {}).get('name', '')
            if status == 'STATUS_FINAL':
                continue  # Already completed

            comps = game.get('competitions', [{}])
            if not comps:
                continue

            teams = comps[0].get('competitors', [])
            if len(teams) != 2:
                continue

            home = [t for t in teams if t.get('homeAway') == 'home']
            away = [t for t in teams if t.get('homeAway') == 'away']

            if home and away:
                h_team = home[0]['team']
                a_team = away[0]['team']
                scheduled.append({
                    'home_name': h_team.get('displayName', '?'),
                    'away_name': a_team.get('displayName', '?'),
                    'home_id': int(h_team.get('id', 0)),
                    'away_id': int(a_team.get('id', 0)),
                    'status': status,
                    'date': date_str,
                    'venue': comps[0].get('venue', {}).get('address', {}).get('city', ''),
                })

    return scheduled


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Tournament Update & Predict',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --update                          Update data, show summary
  %(prog)s --update --predict-scheduled      Update + predict upcoming games
  %(prog)s --predict-scheduled               Predict upcoming games (no update)
  %(prog)s --matchup "Duke" "TCU"            Predict a specific matchup
  %(prog)s --matchup "Duke" "TCU" --venue "Greenville"
  %(prog)s --matchup "Duke" "TCU" --home "Duke"   Home game for Duke
  %(prog)s --list-venues                     Show available venue cities
        """,
    )
    parser.add_argument('--update', action='store_true',
                        help='Fetch latest results + box scores from ESPN')
    parser.add_argument('--matchup', nargs=2, metavar=('TEAM_A', 'TEAM_B'),
                        help='Predict a specific matchup (team names)')
    parser.add_argument('--venue', type=str, default=None,
                        help='Venue city for travel distance (e.g., "Greenville")')
    parser.add_argument('--home', type=str, default=None,
                        help='Team playing at home (non-neutral game)')
    parser.add_argument('--predict-scheduled', action='store_true',
                        help='Predict all upcoming scheduled tournament games')
    parser.add_argument('--list-venues', action='store_true',
                        help='List available venue cities')
    args = parser.parse_args()

    if args.list_venues:
        print("\nAvailable venue cities:")
        for city in sorted(VENUE_COORDS.keys()):
            lat, lon, utc = VENUE_COORDS[city]
            print(f"  {city:20s}  ({lat:.2f}, {lon:.2f})  UTC{utc:+d}")
        return

    conn = get_connection()

    # ── Step 1: Update data if requested ──
    if args.update:
        print("=" * 60)
        print("  UPDATING TOURNAMENT DATA")
        print("=" * 60)

        print("\nFetching tournament results from ESPN...")
        new, updated = fetch_tournament_results(conn)
        print(f"  {new} new games, {updated} existing games updated")

        print("\nFetching box scores for new games...")
        fetch_tournament_box_scores(conn)

        # Show current tournament game count
        total = conn.execute(
            "SELECT COUNT(*) FROM games WHERE season_type = 3 AND status = 'STATUS_FINAL'"
        ).fetchone()[0]
        print(f"\nTotal completed tournament games in DB: {total}")

    # ── Step 2: Build model ──
    if args.matchup or args.predict_scheduled:
        model, features, snapshots, team_locs = build_model_and_snapshots(conn)

    # ── Step 3: Predict specific matchup ──
    if args.matchup:
        name_a, name_b = args.matchup
        id_a, full_name_a = find_team(conn, name_a)
        id_b, full_name_b = find_team(conn, name_b)

        if id_a is None:
            print(f"ERROR: Could not find team matching '{name_a}'")
            conn.close()
            return
        if id_b is None:
            print(f"ERROR: Could not find team matching '{name_b}'")
            conn.close()
            return

        # Handle --home flag
        neutral = True
        home_team_name = None
        if args.home:
            home_id, home_full_name = find_team(conn, args.home)
            if home_id is None:
                print(f"ERROR: Could not find home team matching '{args.home}'")
                conn.close()
                return
            if home_id == id_b:
                # Swap so home team is team_a (is_home applies to team_a)
                id_a, id_b = id_b, id_a
                full_name_a, full_name_b = full_name_b, full_name_a
            elif home_id != id_a:
                print(f"ERROR: --home team '{args.home}' doesn't match either team")
                conn.close()
                return
            neutral = False
            home_team_name = full_name_a

        venue = args.venue
        prob_a = predict_matchup(
            model, features, snapshots, team_locs,
            id_a, id_b, full_name_a, full_name_b,
            venue_city=venue, neutral=neutral,
        )
        prob_b = 1.0 - prob_a

        print(f"\n{'=' * 60}")
        print(f"  MATCHUP PREDICTION")
        print(f"{'=' * 60}")
        if home_team_name:
            venue_str = f" @ {venue}" if venue else ""
            label_a = f"{full_name_a} (home)"
            label_b = f"{full_name_b} (away)"
        else:
            venue_str = f" @ {venue}" if venue else " (neutral site)"
            label_a = full_name_a
            label_b = full_name_b
        print(f"\n  {label_a} vs {label_b}{venue_str}")
        print(f"\n  {label_a:30s}  {prob_a:>6.1%}")
        print(f"  {label_b:30s}  {prob_b:>6.1%}")

        # Show key differentials
        snap_a = snapshots.get(id_a, {})
        snap_b = snapshots.get(id_b, {})

        print(f"\n  {'Metric':30s}  {'':>12s}  {'':>12s}")
        print(f"  {'-' * 56}")

        metrics = [
            ('Win %', 'run_win_pct', '.1%'),
            ('Avg Margin', 'run_avg_margin', '+.1f'),
            ('Recent Win %', 'run_recent_win_pct', '.1%'),
            ('SOS', 'sos', '.3f'),
            ('Roster BPM', 'roster_bpm', '+.2f'),
            ('Top 5 BPM', 'top5_bpm', '+.2f'),
            ('Prior Barthag', 'prior_barthag', '.3f'),
            ('Prior Rank', 'prior_rank', '.0f'),
        ]

        for label, key, fmt in metrics:
            val_a = snap_a.get(key, 0)
            val_b = snap_b.get(key, 0)
            if val_a is None:
                val_a = 0
            if val_b is None:
                val_b = 0
            print(f"  {label:30s}  {format(val_a, fmt):>12s}  {format(val_b, fmt):>12s}")

        if venue and team_locs:
            if venue in VENUE_COORDS:
                v_lat, v_lon, v_off = VENUE_COORDS[venue]
                loc_a = team_locs.get(id_a)
                loc_b = team_locs.get(id_b)
                if loc_a:
                    dist_a = haversine_miles(loc_a[0], loc_a[1], v_lat, v_lon)
                    tz_a = abs(loc_a[2] - v_off)
                    print(f"  {'Travel (mi)':30s}  {dist_a:>12.0f}  ", end="")
                else:
                    print(f"  {'Travel (mi)':30s}  {'N/A':>12s}  ", end="")
                if loc_b:
                    dist_b = haversine_miles(loc_b[0], loc_b[1], v_lat, v_lon)
                    tz_b = abs(loc_b[2] - v_off)
                    print(f"{dist_b:>12.0f}")
                else:
                    print(f"{'N/A':>12s}")

    # ── Step 4: Predict all scheduled games ──
    if args.predict_scheduled:
        print(f"\n{'=' * 60}")
        print(f"  UPCOMING TOURNAMENT GAMES")
        print(f"{'=' * 60}")

        scheduled = get_scheduled_tournament_games(conn)

        if not scheduled:
            print("\n  No upcoming tournament games found.")
        else:
            for game in scheduled:
                home_id = game['home_id']
                away_id = game['away_id']
                home_name = game['home_name']
                away_name = game['away_name']

                # Try to match venue
                espn_venue = game.get('venue', '')
                matched_venue = None
                for vc in VENUE_COORDS:
                    if vc.lower() in espn_venue.lower():
                        matched_venue = vc
                        break

                prob_h = predict_matchup(
                    model, features, snapshots, team_locs,
                    home_id, away_id, home_name, away_name,
                    venue_city=matched_venue,
                )
                prob_a = 1.0 - prob_h

                status_str = ""
                if game['status'] == 'STATUS_IN_PROGRESS':
                    status_str = " [IN PROGRESS]"

                venue_str = f" @ {matched_venue}" if matched_venue else ""
                print(f"\n  {home_name} vs {away_name}{venue_str}{status_str}")
                print(f"    {home_name:30s}  {prob_h:>6.1%}")
                print(f"    {away_name:30s}  {prob_a:>6.1%}")

    # Show update-only summary
    if args.update and not args.matchup and not args.predict_scheduled:
        print("\nData updated. Use --predict-scheduled or --matchup to make predictions.")

    conn.close()


if __name__ == '__main__':
    main()
