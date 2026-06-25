"""
Database module for the College Basketball Prediction Model.

Handles SQLite database creation, schema management, and common operations.
"""

import sqlite3
from pathlib import Path
from typing import Optional

# Schema version for migrations
SCHEMA_VERSION = 1


def get_db_path(config_path: str = "config/config.yaml") -> str:
    """Get database path from config."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config["database"]["path"]


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Get a database connection.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection
    """
    if db_path is None:
        db_path = get_db_path()

    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database(db_path: Optional[str] = None):
    """
    Initialize the database with all required tables.

    Args:
        db_path: Path to SQLite database file
    """
    conn = get_connection(db_path)

    # Create schema version table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create teams table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            espn_id INTEGER UNIQUE,
            name TEXT NOT NULL,
            short_name TEXT,
            abbreviation TEXT,
            nickname TEXT,
            location TEXT,
            conference_id INTEGER,
            color TEXT,
            alternate_color TEXT,
            logo_url TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create conferences table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conferences (
            conference_id INTEGER PRIMARY KEY,
            espn_id INTEGER UNIQUE,
            name TEXT NOT NULL,
            short_name TEXT,
            abbreviation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create seasons table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seasons (
            season_id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL UNIQUE,
            start_date DATE,
            end_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create games table
    # Data notes (from 2026-03 audit):
    #   - 15,461 games across 3 seasons (2024-2026), regular season only (season_type=2)
    #   - Postseason/tournament games are NOT collected by the ESPN schedule endpoint
    #   - Scores range 28-134; sub-40 outliers are real games (verified)
    #   - spread, over_under: ALWAYS NULL — ESPN schedule API does not provide odds
    #   - venue_id: ALWAYS NULL — ESPN schedule API does not populate this field
    #   - venue_name/venue_city/venue_state: populated for most games
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY,
            espn_id TEXT UNIQUE,
            date TIMESTAMP NOT NULL,
            season_year INTEGER NOT NULL,
            season_type INTEGER,                    -- always 2 (regular season); no postseason data
            week INTEGER,
            neutral_site BOOLEAN DEFAULT FALSE,
            conference_game BOOLEAN DEFAULT FALSE,
            attendance INTEGER,
            status TEXT,
            -- Home team
            home_team_id INTEGER REFERENCES teams(espn_id),
            home_score INTEGER,
            home_winner BOOLEAN,
            home_rank INTEGER,
            -- Away team
            away_team_id INTEGER REFERENCES teams(espn_id),
            away_score INTEGER,
            away_winner BOOLEAN,
            away_rank INTEGER,
            -- Venue (venue_id is always NULL; name/city/state are populated)
            venue_id INTEGER,
            venue_name TEXT,
            venue_city TEXT,
            venue_state TEXT,
            -- Betting lines: ALWAYS NULL — ESPN schedule API does not provide odds.
            -- Kept in schema in case a future data source populates them.
            spread REAL,
            over_under REAL,
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create team_seasons table (for tracking team stats per season)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_seasons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER REFERENCES teams(espn_id),
            season_year INTEGER NOT NULL,
            conference_id INTEGER REFERENCES conferences(espn_id),
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            conference_wins INTEGER DEFAULT 0,
            conference_losses INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_id, season_year)
        )
    """)

    # Create game_stats table (for detailed game statistics)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER REFERENCES games(game_id),
            team_id INTEGER REFERENCES teams(espn_id),
            -- Basic stats
            points INTEGER,
            field_goals_made INTEGER,
            field_goals_attempted INTEGER,
            field_goal_pct REAL,
            three_pointers_made INTEGER,
            three_pointers_attempted INTEGER,
            three_point_pct REAL,
            free_throws_made INTEGER,
            free_throws_attempted INTEGER,
            free_throw_pct REAL,
            -- Rebounds
            offensive_rebounds INTEGER,
            defensive_rebounds INTEGER,
            total_rebounds INTEGER,
            -- Other stats
            assists INTEGER,
            steals INTEGER,
            blocks INTEGER,
            turnovers INTEGER,
            fouls INTEGER,
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id, team_id)
        )
    """)

    # Create player_game_stats table (per-player per-game box score data)
    # Source: ESPN game summary endpoint (boxscore.players)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_game_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL REFERENCES games(game_id),
            team_id INTEGER NOT NULL REFERENCES teams(espn_id),
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            starter BOOLEAN DEFAULT FALSE,
            did_not_play BOOLEAN DEFAULT FALSE,
            ejected BOOLEAN DEFAULT FALSE,
            -- Stats (NULL for DNP players)
            minutes INTEGER,
            points INTEGER,
            field_goals_made INTEGER,
            field_goals_attempted INTEGER,
            three_pointers_made INTEGER,
            three_pointers_attempted INTEGER,
            free_throws_made INTEGER,
            free_throws_attempted INTEGER,
            rebounds INTEGER,
            offensive_rebounds INTEGER,
            defensive_rebounds INTEGER,
            assists INTEGER,
            turnovers INTEGER,
            steals INTEGER,
            blocks INTEGER,
            fouls INTEGER,
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id, player_id)
        )
    """)

    # Team and venue geocoded locations (for travel distance features)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_locations (
            espn_id INTEGER PRIMARY KEY REFERENCES teams(espn_id),
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timezone TEXT NOT NULL,
            utc_offset REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS venue_locations (
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timezone TEXT NOT NULL,
            utc_offset REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(city, state)
        )
    """)

    # Historical betting odds (source: SportsData.io)
    # One row per game with consensus/first-available pregame odds.
    # Note: free-trial data has scrambled sportsbook names and slightly
    # perturbed odds values, but directional signals are preserved.
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

    # Player injury reports (historical, from scraping + manual entry)
    # Each row = one player's status on one report date.
    # Multiple sources may report different statuses; we keep them all.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_injury_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            team_id INTEGER NOT NULL REFERENCES teams(espn_id),
            season_year INTEGER NOT NULL,
            report_date DATE NOT NULL,
            status TEXT NOT NULL,           -- OUT, DOUBTFUL, QUESTIONABLE, PROBABLE, AVAILABLE
            source TEXT NOT NULL,           -- 'rotowire', 'espn', 'manual'
            detail TEXT,                    -- injury description if available
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_name, team_id, report_date, source)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_injury_team_date
        ON player_injury_reports(team_id, report_date)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_injury_player
        ON player_injury_reports(player_name, team_id, season_year)
    """)

    # Injury scrape log — tracks which teams were checked (even if no injuries found)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS injury_scrape_log (
            team_id INTEGER NOT NULL REFERENCES teams(espn_id),
            season_year INTEGER NOT NULL,
            scrape_date DATE NOT NULL,
            injuries_found INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'rotowire',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_id, season_year, scrape_date, source)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_scrape_log_team_date
        ON injury_scrape_log(team_id, scrape_date)
    """)

    # Create indices for common queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_season ON games(season_year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_home_team ON games(home_team_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_away_team ON games(away_team_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_team_seasons_team ON team_seasons(team_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_team_seasons_season ON team_seasons(season_year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pgs_game ON player_game_stats(game_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pgs_player ON player_game_stats(player_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pgs_team ON player_game_stats(team_id)")

    # Record schema version
    conn.execute(
        "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
        (SCHEMA_VERSION,)
    )

    conn.commit()
    conn.close()

    print(f"Database initialized at {db_path or get_db_path()}")


def initialize_finance_tables(db_path: Optional[str] = None):
    """
    Initialize finance-related tables.

    Args:
        db_path: Path to SQLite database file
    """
    conn = get_connection(db_path)

    # Bankroll tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bankroll (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            transaction_type TEXT NOT NULL,
            amount REAL NOT NULL,
            balance_after REAL NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Bets tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kalshi_order_id TEXT,
            game_id INTEGER REFERENCES games(game_id),
            placed_at TIMESTAMP NOT NULL,
            market_ticker TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            contracts INTEGER NOT NULL,
            price REAL NOT NULL,
            total_cost REAL NOT NULL,
            model_probability REAL,
            market_probability REAL,
            edge REAL,
            strategy TEXT,
            status TEXT DEFAULT 'open',
            settled_at TIMESTAMP,
            payout REAL,
            profit_loss REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Daily summary
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE NOT NULL,
            starting_bankroll REAL,
            ending_bankroll REAL,
            bets_placed INTEGER DEFAULT 0,
            bets_settled INTEGER DEFAULT 0,
            bets_won INTEGER DEFAULT 0,
            bets_lost INTEGER DEFAULT 0,
            gross_profit REAL DEFAULT 0,
            gross_loss REAL DEFAULT 0,
            net_pnl REAL DEFAULT 0,
            roi_percent REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Strategy performance
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            period_start DATE NOT NULL,
            period_end DATE NOT NULL,
            total_bets INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_rate REAL,
            total_wagered REAL,
            total_pnl REAL,
            roi_percent REAL,
            avg_edge REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    print("Finance tables initialized")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def upsert_team(conn: sqlite3.Connection, team_data: dict):
    """Insert or update a team."""
    conn.execute("""
        INSERT INTO teams (espn_id, name, short_name, abbreviation, nickname,
                          location, color, alternate_color, logo_url, is_active, updated_at)
        VALUES (:team_id, :name, :short_name, :abbreviation, :nickname,
                :location, :color, :alternate_color, :logo_url, :is_active, CURRENT_TIMESTAMP)
        ON CONFLICT(espn_id) DO UPDATE SET
            name = excluded.name,
            short_name = excluded.short_name,
            abbreviation = excluded.abbreviation,
            nickname = excluded.nickname,
            location = excluded.location,
            color = excluded.color,
            alternate_color = excluded.alternate_color,
            logo_url = excluded.logo_url,
            is_active = excluded.is_active,
            updated_at = CURRENT_TIMESTAMP
    """, team_data)


def upsert_game(conn: sqlite3.Connection, game_data: dict):
    """Insert or update a game."""
    conn.execute("""
        INSERT INTO games (espn_id, date, season_year, season_type, week,
                          neutral_site, conference_game, attendance, status,
                          home_team_id, home_score, home_winner, home_rank,
                          away_team_id, away_score, away_winner, away_rank,
                          venue_id, venue_name, venue_city, venue_state,
                          spread, over_under, updated_at)
        VALUES (:game_id, :date, :season, :season_type, :week,
                :neutral_site, :conference_competition, :attendance, :status,
                :home_team_id, :home_score, :home_winner, :home_rank,
                :away_team_id, :away_score, :away_winner, :away_rank,
                :venue_id, :venue_name, :venue_city, :venue_state,
                :spread, :over_under, CURRENT_TIMESTAMP)
        ON CONFLICT(espn_id) DO UPDATE SET
            date = excluded.date,
            status = excluded.status,
            home_score = excluded.home_score,
            home_winner = excluded.home_winner,
            away_score = excluded.away_score,
            away_winner = excluded.away_winner,
            attendance = excluded.attendance,
            spread = excluded.spread,
            over_under = excluded.over_under,
            updated_at = CURRENT_TIMESTAMP
    """, game_data)


if __name__ == "__main__":
    # Initialize database when run directly
    initialize_database()
    initialize_finance_tables()
