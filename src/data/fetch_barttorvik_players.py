"""
Barttorvik Player Stats Fetcher

Downloads and imports player-level stats from Barttorvik's public CSV endpoint.
URL: https://barttorvik.com/getadvstats.php?year=XXXX&csv=1

Includes recruiting rank, advanced metrics, shot location data, and per-game stats
for ~5,000 D1 players per season.

Usage:
    python src/data/fetch_barttorvik_players.py --seasons 2024 2025 2026
"""

import argparse
import csv
import io
import sqlite3
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.database import get_connection
from src.data.import_barttorvik import match_team_to_espn


PLAYER_STATS_URL = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"

# Column mapping from Barttorvik's pstatheaders.xlsx
# Index -> (column_name, db_type)
COLUMN_MAP = [
    (0, 'player_name'),
    (1, 'team'),
    (2, 'conf'),
    (3, 'games_played'),
    (4, 'min_pct'),
    (5, 'ortg'),
    (6, 'usage'),
    (7, 'efg_pct'),
    (8, 'ts_pct'),
    (9, 'orb_pct'),
    (10, 'drb_pct'),
    (11, 'ast_pct'),
    (12, 'to_pct'),
    (13, 'ftm'),
    (14, 'fta'),
    (15, 'ft_pct'),
    (16, 'two_pm'),
    (17, 'two_pa'),
    (18, 'two_p_pct'),
    (19, 'three_pm'),
    (20, 'three_pa'),
    (21, 'three_p_pct'),
    (22, 'blk_pct'),
    (23, 'stl_pct'),
    (24, 'ftr'),
    (25, 'class_year'),
    (26, 'height'),
    (27, 'jersey_num'),
    (28, 'porpag'),
    (29, 'adj_oe'),
    (30, 'pfr'),
    (31, 'season_year'),
    (32, 'player_id'),
    (33, 'hometown'),
    (34, 'rec_rank'),
    (35, 'ast_to_ratio'),
    (36, 'rim_made'),
    (37, 'rim_attempts'),
    (38, 'mid_made'),
    (39, 'mid_attempts'),
    (40, 'rim_pct'),
    (41, 'mid_pct'),
    (42, 'dunks_made'),
    (43, 'dunk_attempts'),
    (44, 'dunk_pct'),
    (45, 'pick'),
    (46, 'drtg'),
    (47, 'adj_de'),
    (48, 'dporpag'),
    (49, 'stops'),
    (50, 'bpm'),
    (51, 'obpm'),
    (52, 'dbpm'),
    (53, 'gbpm'),
    (54, 'minutes'),
    (55, 'ogbpm'),
    (56, 'dgbpm'),
    (57, 'oreb_pg'),
    (58, 'dreb_pg'),
    (59, 'treb_pg'),
    (60, 'ast_pg'),
    (61, 'stl_pg'),
    (62, 'blk_pg'),
    (63, 'pts_pg'),
    (64, 'role'),
    (65, 'three_p_per100'),
    (66, 'dob'),
]


def create_player_stats_table(conn: sqlite3.Connection):
    """Create table for player stats."""
    # Data notes (from 2026-03 audit):
    #   - 15,019 rows across 3 seasons (~5,000 per season)
    #   - High null-rate columns: pick (99.2%), rec_rank (75.9%), dunk_pct (45.6%), drtg (36.9%)
    #   - IMPORTANT: Rate/percentage fields (ortg, efg_pct, ts_pct, orb_pct, drb_pct,
    #     blk_pct, stl_pct, rim_pct, mid_pct, etc.) contain extreme outliers from
    #     low-minute players (e.g. ortg up to 300, efg_pct up to 150%, drb_pct up to 203).
    #     These fields should ALWAYS be used alongside a volume/weighting metric
    #     (games_played, minutes, min_pct, shot attempts, etc.) to avoid skewed results.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            team TEXT NOT NULL,
            conf TEXT,
            season_year INTEGER NOT NULL,
            class_year TEXT,
            height TEXT,
            hometown TEXT,
            role TEXT,
            dob TEXT,
            -- Recruiting
            rec_rank REAL,
            -- Usage & efficiency
            games_played INTEGER,
            min_pct REAL,
            minutes REAL,
            ortg REAL,
            usage REAL,
            efg_pct REAL,
            ts_pct REAL,
            ftr REAL,
            -- Shooting
            ftm REAL,
            fta REAL,
            ft_pct REAL,
            two_pm REAL,
            two_pa REAL,
            two_p_pct REAL,
            three_pm REAL,
            three_pa REAL,
            three_p_pct REAL,
            three_p_per100 REAL,
            -- Shot location
            rim_made REAL,
            rim_attempts REAL,
            rim_pct REAL,
            mid_made REAL,
            mid_attempts REAL,
            mid_pct REAL,
            dunks_made REAL,
            dunk_attempts REAL,
            dunk_pct REAL,
            -- Percentages
            orb_pct REAL,
            drb_pct REAL,
            ast_pct REAL,
            to_pct REAL,
            blk_pct REAL,
            stl_pct REAL,
            ast_to_ratio REAL,
            -- Per-game stats
            oreb_pg REAL,
            dreb_pg REAL,
            treb_pg REAL,
            ast_pg REAL,
            stl_pg REAL,
            blk_pg REAL,
            pts_pg REAL,
            -- Advanced
            porpag REAL,
            adj_oe REAL,
            adj_de REAL,
            drtg REAL,
            pfr REAL,
            pick REAL,
            dporpag REAL,
            stops REAL,
            bpm REAL,
            obpm REAL,
            dbpm REAL,
            gbpm REAL,
            ogbpm REAL,
            dgbpm REAL,
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_id, season_year)
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_player_id ON player_stats(player_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_player_team ON player_stats(team, season_year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_player_rec ON player_stats(rec_rank)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_player_class ON player_stats(class_year, season_year)")

    conn.commit()


def safe_float(val):
    if val is None or val == '' or val == 'N/A':
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    if val is None or val == '' or val == 'N/A':
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def parse_player_row(row: list) -> dict:
    """Parse a CSV row into a player dict using the column mapping."""
    if len(row) < 65:
        return None

    player_id = safe_int(row[32])
    if not player_id:
        return None

    return {
        'player_id': player_id,
        'player_name': row[0].strip().strip('"'),
        'team': row[1].strip().strip('"'),
        'conf': row[2].strip() if row[2] else None,
        'season_year': safe_int(row[31]),
        'class_year': row[25].strip() if row[25] else None,
        'height': row[26].strip() if row[26] else None,
        'hometown': row[33].strip().strip('"') if row[33] else None,
        'role': row[64].strip() if len(row) > 64 and row[64] else None,
        'dob': row[66].strip() if len(row) > 66 and row[66] else None,
        'rec_rank': safe_float(row[34]),
        'games_played': safe_int(row[3]),
        'min_pct': safe_float(row[4]),
        'minutes': safe_float(row[54]),
        'ortg': safe_float(row[5]),
        'usage': safe_float(row[6]),
        'efg_pct': safe_float(row[7]),
        'ts_pct': safe_float(row[8]),
        'ftr': safe_float(row[24]),
        'ftm': safe_float(row[13]),
        'fta': safe_float(row[14]),
        'ft_pct': safe_float(row[15]),
        'two_pm': safe_float(row[16]),
        'two_pa': safe_float(row[17]),
        'two_p_pct': safe_float(row[18]),
        'three_pm': safe_float(row[19]),
        'three_pa': safe_float(row[20]),
        'three_p_pct': safe_float(row[21]),
        'three_p_per100': safe_float(row[65]) if len(row) > 65 else None,
        'rim_made': safe_float(row[36]),
        'rim_attempts': safe_float(row[37]),
        'rim_pct': safe_float(row[40]),
        'mid_made': safe_float(row[38]),
        'mid_attempts': safe_float(row[39]),
        'mid_pct': safe_float(row[41]),
        'dunks_made': safe_float(row[42]),
        'dunk_attempts': safe_float(row[43]),
        'dunk_pct': safe_float(row[44]),
        'orb_pct': safe_float(row[9]),
        'drb_pct': safe_float(row[10]),
        'ast_pct': safe_float(row[11]),
        'to_pct': safe_float(row[12]),
        'blk_pct': safe_float(row[22]),
        'stl_pct': safe_float(row[23]),
        'ast_to_ratio': safe_float(row[35]),
        'oreb_pg': safe_float(row[57]),
        'dreb_pg': safe_float(row[58]),
        'treb_pg': safe_float(row[59]),
        'ast_pg': safe_float(row[60]),
        'stl_pg': safe_float(row[61]),
        'blk_pg': safe_float(row[62]),
        'pts_pg': safe_float(row[63]),
        'porpag': safe_float(row[28]),
        'adj_oe': safe_float(row[29]),
        'adj_de': safe_float(row[47]),
        'drtg': safe_float(row[46]),
        'pfr': safe_float(row[30]),
        'pick': safe_float(row[45]),
        'dporpag': safe_float(row[48]),
        'stops': safe_float(row[49]),
        'bpm': safe_float(row[50]),
        'obpm': safe_float(row[51]),
        'dbpm': safe_float(row[52]),
        'gbpm': safe_float(row[53]),
        'ogbpm': safe_float(row[55]),
        'dgbpm': safe_float(row[56]),
    }


def upsert_player(conn: sqlite3.Connection, data: dict):
    """Insert or update player stats."""
    conn.execute("""
        INSERT INTO player_stats (
            player_id, player_name, team, espn_team_id, conf, season_year, class_year, height,
            hometown, role, dob, rec_rank,
            games_played, min_pct, minutes, ortg, usage, efg_pct, ts_pct, ftr,
            ftm, fta, ft_pct, two_pm, two_pa, two_p_pct,
            three_pm, three_pa, three_p_pct, three_p_per100,
            rim_made, rim_attempts, rim_pct, mid_made, mid_attempts, mid_pct,
            dunks_made, dunk_attempts, dunk_pct,
            orb_pct, drb_pct, ast_pct, to_pct, blk_pct, stl_pct, ast_to_ratio,
            oreb_pg, dreb_pg, treb_pg, ast_pg, stl_pg, blk_pg, pts_pg,
            porpag, adj_oe, adj_de, drtg, pfr, pick, dporpag, stops,
            bpm, obpm, dbpm, gbpm, ogbpm, dgbpm,
            updated_at
        ) VALUES (
            :player_id, :player_name, :team, :espn_team_id, :conf, :season_year, :class_year, :height,
            :hometown, :role, :dob, :rec_rank,
            :games_played, :min_pct, :minutes, :ortg, :usage, :efg_pct, :ts_pct, :ftr,
            :ftm, :fta, :ft_pct, :two_pm, :two_pa, :two_p_pct,
            :three_pm, :three_pa, :three_p_pct, :three_p_per100,
            :rim_made, :rim_attempts, :rim_pct, :mid_made, :mid_attempts, :mid_pct,
            :dunks_made, :dunk_attempts, :dunk_pct,
            :orb_pct, :drb_pct, :ast_pct, :to_pct, :blk_pct, :stl_pct, :ast_to_ratio,
            :oreb_pg, :dreb_pg, :treb_pg, :ast_pg, :stl_pg, :blk_pg, :pts_pg,
            :porpag, :adj_oe, :adj_de, :drtg, :pfr, :pick, :dporpag, :stops,
            :bpm, :obpm, :dbpm, :gbpm, :ogbpm, :dgbpm,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT(player_id, season_year) DO UPDATE SET
            player_name = excluded.player_name,
            team = excluded.team,
            espn_team_id = excluded.espn_team_id,
            conf = excluded.conf,
            class_year = excluded.class_year,
            height = excluded.height,
            hometown = excluded.hometown,
            role = excluded.role,
            dob = excluded.dob,
            rec_rank = excluded.rec_rank,
            games_played = excluded.games_played,
            min_pct = excluded.min_pct,
            minutes = excluded.minutes,
            ortg = excluded.ortg,
            usage = excluded.usage,
            efg_pct = excluded.efg_pct,
            ts_pct = excluded.ts_pct,
            ftr = excluded.ftr,
            ftm = excluded.ftm,
            fta = excluded.fta,
            ft_pct = excluded.ft_pct,
            two_pm = excluded.two_pm,
            two_pa = excluded.two_pa,
            two_p_pct = excluded.two_p_pct,
            three_pm = excluded.three_pm,
            three_pa = excluded.three_pa,
            three_p_pct = excluded.three_p_pct,
            three_p_per100 = excluded.three_p_per100,
            rim_made = excluded.rim_made,
            rim_attempts = excluded.rim_attempts,
            rim_pct = excluded.rim_pct,
            mid_made = excluded.mid_made,
            mid_attempts = excluded.mid_attempts,
            mid_pct = excluded.mid_pct,
            dunks_made = excluded.dunks_made,
            dunk_attempts = excluded.dunk_attempts,
            dunk_pct = excluded.dunk_pct,
            orb_pct = excluded.orb_pct,
            drb_pct = excluded.drb_pct,
            ast_pct = excluded.ast_pct,
            to_pct = excluded.to_pct,
            blk_pct = excluded.blk_pct,
            stl_pct = excluded.stl_pct,
            ast_to_ratio = excluded.ast_to_ratio,
            oreb_pg = excluded.oreb_pg,
            dreb_pg = excluded.dreb_pg,
            treb_pg = excluded.treb_pg,
            ast_pg = excluded.ast_pg,
            stl_pg = excluded.stl_pg,
            blk_pg = excluded.blk_pg,
            pts_pg = excluded.pts_pg,
            porpag = excluded.porpag,
            adj_oe = excluded.adj_oe,
            adj_de = excluded.adj_de,
            drtg = excluded.drtg,
            pfr = excluded.pfr,
            pick = excluded.pick,
            dporpag = excluded.dporpag,
            stops = excluded.stops,
            bpm = excluded.bpm,
            obpm = excluded.obpm,
            dbpm = excluded.dbpm,
            gbpm = excluded.gbpm,
            ogbpm = excluded.ogbpm,
            dgbpm = excluded.dgbpm,
            updated_at = CURRENT_TIMESTAMP
    """, data)


def fetch_and_import_players(season: int, conn: sqlite3.Connection) -> dict:
    """Fetch and import player stats for a single season."""
    url = PLAYER_STATS_URL.format(year=season)
    print(f"  Fetching {url}...")

    response = requests.get(url)
    response.raise_for_status()

    reader = csv.reader(io.StringIO(response.text))
    imported = 0
    freshmen = 0
    with_rec_rank = 0
    errors = []

    # Cache team name → ESPN ID lookups
    team_id_cache = {}

    for row in reader:
        try:
            data = parse_player_row(row)
            if not data:
                continue

            if data['class_year'] == 'Fr':
                freshmen += 1
            if data['rec_rank'] is not None:
                with_rec_rank += 1

            # Match team to ESPN ID
            team_name = data['team']
            if team_name not in team_id_cache:
                team_id_cache[team_name] = match_team_to_espn(conn, team_name)
            data['espn_team_id'] = team_id_cache[team_name]

            upsert_player(conn, data)
            imported += 1
        except Exception as e:
            name = row[0] if row else 'Unknown'
            errors.append(f"{name}: {e}")

    conn.commit()

    matched_teams = sum(1 for v in team_id_cache.values() if v is not None)

    return {
        'imported': imported,
        'freshmen': freshmen,
        'with_rec_rank': with_rec_rank,
        'teams_matched': matched_teams,
        'teams_total': len(team_id_cache),
        'errors': errors,
    }


def collect_player_stats(
    seasons: list[int],
    db_path: str = None,
) -> dict:
    """Collect player stats for multiple seasons."""
    conn = get_connection(db_path)
    create_player_stats_table(conn)

    results = {}
    for season in seasons:
        print(f"Fetching player stats for {season-1}-{season} season...")
        try:
            result = fetch_and_import_players(season, conn)
            results[season] = result
            print(f"  Imported: {result['imported']} players "
                  f"({result['freshmen']} freshmen, "
                  f"{result['with_rec_rank']} with recruiting rank)")
            print(f"  Teams matched: {result['teams_matched']}/{result['teams_total']}")
            if result['errors']:
                print(f"  Errors: {len(result['errors'])}")
                for err in result['errors'][:3]:
                    print(f"    - {err}")
        except Exception as e:
            print(f"  Error: {e}")
            results[season] = {'error': str(e)}

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch Barttorvik player stats")
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

    args = parser.parse_args()

    print("=" * 60)
    print("Barttorvik Player Stats Collection")
    print("=" * 60)

    results = collect_player_stats(args.seasons, args.db)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_players = 0
    total_freshmen = 0
    total_ranked = 0
    for season, result in results.items():
        if 'error' in result:
            print(f"  {season-1}-{season}: ERROR - {result['error']}")
        else:
            print(f"  {season-1}-{season}: {result['imported']} players "
                  f"({result['freshmen']} Fr, {result['with_rec_rank']} ranked)")
            total_players += result['imported']
            total_freshmen += result['freshmen']
            total_ranked += result['with_rec_rank']
    print(f"\n  Total: {total_players} player records")
    print(f"  Total freshmen: {total_freshmen}")
    print(f"  Total with recruiting rank: {total_ranked}")


if __name__ == "__main__":
    main()
