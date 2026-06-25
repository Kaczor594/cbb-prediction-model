"""
ESPN BPI (Basketball Power Index) Data Collection

Fetches team power index data from ESPN's API including:
- BPI scores (overall, offense, defense)
- Strength of Record (SOR)
- Strength of Schedule (SOS)
- Win projections
- Tournament projections

Uses the full FITT API which provides data for all 365 D1 teams.
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.database import get_connection


# ESPN BPI FITT API endpoint - returns ALL D1 teams
BPI_API_URL = "https://site.web.api.espn.com/apis/fitt/v3/sports/basketball/mens-college-basketball/powerindex"


def create_bpi_table(conn: sqlite3.Connection):
    """Create table for BPI data."""
    # Data notes (from 2026-03 audit):
    #   - 365 rows, current season only (2026) — ESPN BPI API does not archive historical data
    #   - Populated columns: bpi, bpi_rank, bpi_offense/defense + ranks, bpi_7day_change,
    #     sor_rank, sos_past_rank, wins, losses, conf_wins/losses, top50_wins/losses
    #   - ALWAYS NULL columns (15): sor, sos_remaining_rank, sos_nonconf_rank, win_pct,
    #     conf_win_pct, proj_wins, proj_losses, proj_win_pct, proj_conf_wins, proj_conf_losses,
    #     proj_conf_win_pct, chance_win_conf, proj_seed, proj_seed_actual, tournament_region,
    #     chance_round32, chance_sweet16, chance_elite8, chance_final4, chance_championship,
    #     chance_title — ESPN FITT API does not provide these fields.
    #     Kept in schema in case ESPN adds them or a future data source populates them.
    #   - 3 records have espn_team_id values not in the teams table (non-D1 or data artifacts)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_bpi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            espn_team_id INTEGER NOT NULL,
            season_year INTEGER NOT NULL,
            last_updated TEXT,
            -- Core BPI metrics (populated)
            bpi REAL,
            bpi_rank INTEGER,
            bpi_offense REAL,
            bpi_offense_rank INTEGER,
            bpi_defense REAL,
            bpi_defense_rank INTEGER,
            bpi_7day_change INTEGER,
            -- Strength metrics (sor_rank populated; sos_remaining/nonconf ALWAYS NULL)
            sor_rank INTEGER,
            sos_past_rank INTEGER,
            sos_remaining_rank INTEGER,
            sos_nonconf_rank INTEGER,
            -- Current record (populated)
            wins INTEGER,
            losses INTEGER,
            conf_wins INTEGER,
            conf_losses INTEGER,
            -- Quality wins (populated)
            top50_wins INTEGER,
            top50_losses INTEGER,
            -- Projections — ALWAYS NULL: ESPN FITT API does not provide these
            proj_wins REAL,
            proj_losses REAL,
            proj_conf_wins REAL,
            proj_conf_losses REAL,
            -- Conference championship — ALWAYS NULL
            chance_win_conf REAL,
            -- Tournament projections — ALWAYS NULL
            proj_seed INTEGER,
            proj_seed_actual INTEGER,
            tournament_region TEXT,
            chance_round32 REAL,
            chance_sweet16 REAL,
            chance_elite8 REAL,
            chance_final4 REAL,
            chance_championship REAL,
            chance_title REAL,
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(espn_team_id, season_year)
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_bpi_team ON team_bpi(espn_team_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bpi_season ON team_bpi(season_year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bpi_rank ON team_bpi(bpi_rank)")

    conn.commit()


def get_value_by_name(team_categories: list, schema_categories: dict, category_name: str, stat_name: str):
    """
    Extract a stat value from team categories using the schema for field mapping.

    Args:
        team_categories: The team's categories list (contains only values)
        schema_categories: Dict mapping category names to their field names
        category_name: Which category to look in (e.g., 'bpi', 'resume', 'tournament')
        stat_name: The stat field name to extract
    """
    # Get the field names for this category from the schema
    if category_name not in schema_categories:
        return None

    names = schema_categories[category_name]
    if stat_name not in names:
        return None

    idx = names.index(stat_name)

    # Find the team's values for this category
    for cat in team_categories:
        if cat.get('name') == category_name:
            values = cat.get('values', [])
            if idx < len(values):
                return values[idx]
    return None


def get_rank_by_name(team_categories: list, schema_categories: dict, category_name: str, stat_name: str):
    """
    Extract a rank string from team categories using the schema for field mapping.
    """
    if category_name not in schema_categories:
        return None

    names = schema_categories[category_name]
    if stat_name not in names:
        return None

    idx = names.index(stat_name)

    for cat in team_categories:
        if cat.get('name') == category_name:
            ranks = cat.get('ranks', [])
            if idx < len(ranks):
                rank_str = ranks[idx]
                if rank_str and rank_str != '-':
                    # Convert "1st", "2nd", etc. to integer
                    return int(''.join(filter(str.isdigit, rank_str))) if any(c.isdigit() for c in rank_str) else None
    return None


def parse_bpi_team(team_data: dict, schema_categories: dict, season: int, last_updated: str) -> dict:
    """
    Parse a single team's BPI data from the FITT API response.

    Args:
        team_data: The team data dict from the API
        schema_categories: Dict mapping category names to their field names
        season: Season year
        last_updated: Timestamp string
    """
    team_info = team_data.get('team', {})
    team_id = team_info.get('id')

    if not team_id:
        return None

    categories = team_data.get('categories', [])

    return {
        'espn_team_id': int(team_id),
        'season_year': season,
        'last_updated': last_updated,
        # Core BPI (from 'bpi' category)
        'bpi': get_value_by_name(categories, schema_categories, 'bpi', 'bpi'),
        'bpi_rank': get_value_by_name(categories, schema_categories, 'bpi', 'bpirank'),
        'bpi_offense': get_value_by_name(categories, schema_categories, 'bpi', 'bpioffense'),
        'bpi_offense_rank': get_rank_by_name(categories, schema_categories, 'bpi', 'bpioffense'),
        'bpi_defense': get_value_by_name(categories, schema_categories, 'bpi', 'bpidefense'),
        'bpi_defense_rank': get_rank_by_name(categories, schema_categories, 'bpi', 'bpidefense'),
        'bpi_7day_change': get_value_by_name(categories, schema_categories, 'bpi', 'bpisevendaychangerank'),
        # Strength metrics (from 'resume' category)
        'sor_rank': get_value_by_name(categories, schema_categories, 'resume', 'sorrank'),
        'sos_past_rank': get_value_by_name(categories, schema_categories, 'resume', 'sospastrank'),
        'sos_remaining_rank': get_value_by_name(categories, schema_categories, 'bpi', 'sosremrank'),
        'sos_nonconf_rank': get_value_by_name(categories, schema_categories, 'resume', 'sosoutofconfpastrank'),
        # Current record
        'wins': get_value_by_name(categories, schema_categories, 'bpi', 'wins'),
        'losses': get_value_by_name(categories, schema_categories, 'bpi', 'losses'),
        'conf_wins': get_value_by_name(categories, schema_categories, 'bpi', 'confwins'),
        'conf_losses': get_value_by_name(categories, schema_categories, 'bpi', 'conflosses'),
        # Quality wins
        'top50_wins': get_value_by_name(categories, schema_categories, 'resume', 'top50bpiwins'),
        'top50_losses': get_value_by_name(categories, schema_categories, 'resume', 'top50bpilosses'),
        # Projections
        'proj_wins': get_value_by_name(categories, schema_categories, 'bpi', 'projtotalwins'),
        'proj_losses': get_value_by_name(categories, schema_categories, 'bpi', 'projtotallosses'),
        'proj_conf_wins': get_value_by_name(categories, schema_categories, 'bpi', 'projconfwins'),
        'proj_conf_losses': get_value_by_name(categories, schema_categories, 'bpi', 'projconflosses'),
        # Conference championship
        'chance_win_conf': get_value_by_name(categories, schema_categories, 'bpi', 'chancewinconfortie'),
        # Tournament (from 'tournament' category)
        'proj_seed': get_value_by_name(categories, schema_categories, 'tournament', 'projectedtournamentseed'),
        'proj_seed_actual': get_value_by_name(categories, schema_categories, 'tournament', 'projectedtournamentseedactual'),
        'tournament_region': get_value_by_name(categories, schema_categories, 'tournament', 'tournamentregion'),
        'chance_round32': get_value_by_name(categories, schema_categories, 'tournament', 'chanceroundof32'),
        'chance_sweet16': get_value_by_name(categories, schema_categories, 'tournament', 'chancesweet16'),
        'chance_elite8': get_value_by_name(categories, schema_categories, 'tournament', 'chanceelite8'),
        'chance_final4': get_value_by_name(categories, schema_categories, 'tournament', 'chancefinal4'),
        'chance_championship': get_value_by_name(categories, schema_categories, 'tournament', 'chancechampgame'),
        'chance_title': get_value_by_name(categories, schema_categories, 'tournament', 'chancencaachampion'),
    }


def upsert_bpi(conn: sqlite3.Connection, data: dict):
    """Insert or update BPI data."""
    conn.execute("""
        INSERT INTO team_bpi (
            espn_team_id, season_year, last_updated,
            bpi, bpi_rank, bpi_offense, bpi_offense_rank, bpi_defense, bpi_defense_rank, bpi_7day_change,
            sor_rank, sos_past_rank, sos_remaining_rank, sos_nonconf_rank,
            wins, losses, conf_wins, conf_losses,
            top50_wins, top50_losses,
            proj_wins, proj_losses, proj_conf_wins, proj_conf_losses,
            chance_win_conf,
            proj_seed, proj_seed_actual, tournament_region,
            chance_round32, chance_sweet16, chance_elite8, chance_final4, chance_championship, chance_title,
            updated_at
        ) VALUES (
            :espn_team_id, :season_year, :last_updated,
            :bpi, :bpi_rank, :bpi_offense, :bpi_offense_rank, :bpi_defense, :bpi_defense_rank, :bpi_7day_change,
            :sor_rank, :sos_past_rank, :sos_remaining_rank, :sos_nonconf_rank,
            :wins, :losses, :conf_wins, :conf_losses,
            :top50_wins, :top50_losses,
            :proj_wins, :proj_losses, :proj_conf_wins, :proj_conf_losses,
            :chance_win_conf,
            :proj_seed, :proj_seed_actual, :tournament_region,
            :chance_round32, :chance_sweet16, :chance_elite8, :chance_final4, :chance_championship, :chance_title,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT(espn_team_id, season_year) DO UPDATE SET
            last_updated = excluded.last_updated,
            bpi = excluded.bpi,
            bpi_rank = excluded.bpi_rank,
            bpi_offense = excluded.bpi_offense,
            bpi_offense_rank = excluded.bpi_offense_rank,
            bpi_defense = excluded.bpi_defense,
            bpi_defense_rank = excluded.bpi_defense_rank,
            bpi_7day_change = excluded.bpi_7day_change,
            sor_rank = excluded.sor_rank,
            sos_past_rank = excluded.sos_past_rank,
            sos_remaining_rank = excluded.sos_remaining_rank,
            sos_nonconf_rank = excluded.sos_nonconf_rank,
            wins = excluded.wins,
            losses = excluded.losses,
            conf_wins = excluded.conf_wins,
            conf_losses = excluded.conf_losses,
            top50_wins = excluded.top50_wins,
            top50_losses = excluded.top50_losses,
            proj_wins = excluded.proj_wins,
            proj_losses = excluded.proj_losses,
            proj_conf_wins = excluded.proj_conf_wins,
            proj_conf_losses = excluded.proj_conf_losses,
            chance_win_conf = excluded.chance_win_conf,
            proj_seed = excluded.proj_seed,
            proj_seed_actual = excluded.proj_seed_actual,
            tournament_region = excluded.tournament_region,
            chance_round32 = excluded.chance_round32,
            chance_sweet16 = excluded.chance_sweet16,
            chance_elite8 = excluded.chance_elite8,
            chance_final4 = excluded.chance_final4,
            chance_championship = excluded.chance_championship,
            chance_title = excluded.chance_title,
            updated_at = CURRENT_TIMESTAMP
    """, data)


def fetch_bpi_data(season: int, rate_limit: float = 0.5) -> tuple[list[dict], str]:
    """
    Fetch BPI data from ESPN FITT API.

    Args:
        season: Season year (e.g., 2026 for 2025-26)
        rate_limit: Seconds between requests

    Returns:
        Tuple of (list of parsed BPI data, last_updated timestamp)
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; CBBPredictionModel/1.0)"
    })

    params = {
        'limit': 400,
        'season': season,
    }

    response = session.get(BPI_API_URL, params=params)

    if response.status_code == 404:
        return [], None

    response.raise_for_status()
    data = response.json()

    last_updated = data.get('lastUpdated')
    teams = data.get('teams', [])
    current_season = data.get('currentSeason', {}).get('year', season)

    # Extract the schema (field names) from top-level categories
    schema_categories = {}
    for cat in data.get('categories', []):
        cat_name = cat.get('name')
        cat_names = cat.get('names', [])
        if cat_name and cat_names:
            schema_categories[cat_name] = cat_names

    parsed_teams = []
    for team_data in teams:
        parsed = parse_bpi_team(team_data, schema_categories, current_season, last_updated)
        if parsed:
            parsed_teams.append(parsed)

    return parsed_teams, last_updated


def collect_bpi_data(
    seasons: list[int],
    db_path: str = None,
    rate_limit: float = 0.5,
) -> dict:
    """
    Collect BPI data for multiple seasons.

    Args:
        seasons: List of season years
        db_path: Database path
        rate_limit: Seconds between requests

    Returns:
        Collection statistics
    """
    conn = get_connection(db_path)
    create_bpi_table(conn)

    results = {}

    for season in seasons:
        print(f"Fetching BPI data for {season-1}-{season} season...")

        items, last_updated = fetch_bpi_data(season, rate_limit=rate_limit)

        if not items:
            print(f"  No data available for {season-1}-{season}")
            results[season] = 0
            continue

        for item in items:
            upsert_bpi(conn, item)

        conn.commit()
        print(f"  Collected {len(items)} teams (updated: {last_updated})")
        results[season] = len(items)

        time.sleep(rate_limit)

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Collect ESPN BPI data")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2024, 2025, 2026],
        help="Season years to collect",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds between API requests",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ESPN BPI Data Collection (Full D1 Coverage)")
    print("=" * 60)

    results = collect_bpi_data(
        seasons=args.seasons,
        db_path=args.db,
        rate_limit=args.rate_limit,
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total = 0
    for season, count in results.items():
        print(f"  {season-1}-{season}: {count} teams")
        total += count
    print(f"\n  Total: {total} team records")


if __name__ == "__main__":
    main()
