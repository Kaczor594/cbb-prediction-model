"""
Barttorvik Data Import Script

Imports team efficiency metrics from Barttorvik CSV exports.
"""

import argparse
import csv
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.database import get_connection


def create_efficiency_table(conn: sqlite3.Connection):
    """Create table for team efficiency metrics."""
    # Data notes (from 2026-03 audit):
    #   - 1,091 rows across 3 seasons (362-365 teams each)
    #   - Populated columns: overall_rank, adj_oe, adj_de, adj_tempo, barthag, wins, losses
    #   - ALWAYS NULL columns (14): efg_pct, to_pct, or_pct, ft_rate, efg_pct_d, to_pct_d,
    #     or_pct_d, ft_rate_d, two_pt_pct, three_pt_pct, three_pt_rate, ft_pct, block_pct,
    #     steal_pct — the Barttorvik CSV export used does not include these fields.
    #     Kept in schema in case a future data source populates them.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_efficiency (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            espn_team_id INTEGER REFERENCES teams(espn_id),
            season_year INTEGER NOT NULL,
            -- Rankings
            overall_rank INTEGER,
            -- Efficiency metrics (populated)
            adj_oe REAL,           -- Adjusted Offensive Efficiency
            adj_de REAL,           -- Adjusted Defensive Efficiency
            adj_tempo REAL,        -- Adjusted Tempo
            barthag REAL,          -- Win probability vs average team
            -- Four Factors (Offense) — ALWAYS NULL: not in current CSV export
            efg_pct REAL,          -- Effective FG%
            to_pct REAL,           -- Turnover %
            or_pct REAL,           -- Offensive Rebound %
            ft_rate REAL,          -- Free Throw Rate
            -- Four Factors (Defense) — ALWAYS NULL: not in current CSV export
            efg_pct_d REAL,        -- Opponent Effective FG%
            to_pct_d REAL,         -- Opponent Turnover %
            or_pct_d REAL,         -- Opponent OR%
            ft_rate_d REAL,        -- Opponent FT Rate
            -- Additional metrics — ALWAYS NULL: not in current CSV export
            two_pt_pct REAL,
            three_pt_pct REAL,
            three_pt_rate REAL,
            ft_pct REAL,
            block_pct REAL,
            steal_pct REAL,
            -- Record
            wins INTEGER,
            losses INTEGER,
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_name, season_year)
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_efficiency_team
        ON team_efficiency(espn_team_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_efficiency_season
        ON team_efficiency(season_year)
    """)

    conn.commit()


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching."""
    # Remove common suffixes and clean up
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)
    return name.lower()


# Explicit Barttorvik → ESPN location mapping for names that don't match
# via simple substring search. Maps barttorvik name (lowercase) → ESPN location.
BARTTORVIK_TO_ESPN = {
    "alabama st.": "Alabama State",
    "alcorn st.": "Alcorn State",
    "appalachian st.": "App State",
    "arizona st.": "Arizona State",
    "arkansas pine bluff": "Arkansas-Pine Bluff",
    "arkansas st.": "Arkansas State",
    "ball st.": "Ball State",
    "bethune cookman": "Bethune-Cookman",
    "boise st.": "Boise State",
    "cal baptist": "California Baptist",
    "cal st. bakersfield": "Cal State Bakersfield",
    "cal st. fullerton": "Cal State Fullerton",
    "cal st. northridge": "Cal State Northridge",
    "chicago st.": "Chicago State",
    "cleveland st.": "Cleveland State",
    "colorado st.": "Colorado State",
    "coppin st.": "Coppin State",
    "delaware st.": "Delaware State",
    "east tennessee st.": "East Tennessee State",
    "florida st.": "Florida State",
    "fresno st.": "Fresno State",
    "gardner webb": "Gardner-Webb",
    "georgia st.": "Georgia State",
    "grambling st.": "Grambling",
    "hawaii": "Hawai'i",
    "idaho st.": "Idaho State",
    "illinois chicago": "UIC",
    "illinois st.": "Illinois State",
    "indiana st.": "Indiana State",
    "iowa st.": "Iowa State",
    "jackson st.": "Jackson State",
    "jacksonville st.": "Jacksonville State",
    "kansas st.": "Kansas State",
    "kennesaw st.": "Kennesaw State",
    "kent st.": "Kent State",
    "liu": "Long Island University",
    "long beach st.": "Long Beach State",
    "louisiana monroe": "UL Monroe",
    "mcneese st.": "McNeese",
    "miami fl": "Miami",
    "michigan st.": "Michigan State",
    "mississippi st.": "Mississippi State",
    "mississippi valley st.": "Mississippi Valley State",
    "missouri st.": "Missouri State",
    "montana st.": "Montana State",
    "morehead st.": "Morehead State",
    "morgan st.": "Morgan State",
    "murray st.": "Murray State",
    "n.c. state": "NC State",
    "nebraska omaha": "Omaha",
    "new mexico st.": "New Mexico State",
    "nicholls st.": "Nicholls",
    "norfolk st.": "Norfolk State",
    "north dakota st.": "North Dakota State",
    "northwestern st.": "Northwestern State",
    "ohio st.": "Ohio State",
    "oklahoma st.": "Oklahoma State",
    "oregon st.": "Oregon State",
    "penn st.": "Penn State",
    "portland st.": "Portland State",
    "sacramento st.": "Sacramento State",
    "sam houston st.": "Sam Houston",
    "san diego st.": "San Diego State",
    "san jose st.": "San José State",
    "south carolina st.": "South Carolina State",
    "south dakota st.": "South Dakota State",
    "southeast missouri st.": "Southeast Missouri State",
    "southeastern louisiana": "SE Louisiana",
    "tarleton st.": "Tarleton State",
    "tennessee martin": "UT Martin",
    "tennessee st.": "Tennessee State",
    "texas a&m corpus chris": "Texas A&M-Corpus Christi",
    "texas st.": "Texas State",
    "umkc": "Kansas City",
    "usc upstate": "South Carolina Upstate",
    "utah st.": "Utah State",
    "washington st.": "Washington State",
    "weber st.": "Weber State",
    "wichita st.": "Wichita State",
    "wright st.": "Wright State",
    "youngstown st.": "Youngstown State",
    # Additional player_stats mismatches not in team_efficiency
    "albany": "UAlbany",
    "american": "American University",
    "connecticut": "UConn",
    "fiu": "Florida International",
    "iu indy": "IU Indianapolis",
    "loyola md": "Loyola Maryland",
    "miami oh": "Miami (OH)",
    "mississippi": "Ole Miss",
    "penn": "Pennsylvania",
    "seattle": "Seattle U",
    "st. thomas": "St. Thomas-Minnesota",
    # Generic/short names that collide with substring matches
    # (e.g. "alabama" matches "Alabama A&M" before "Alabama")
    "alabama": "Alabama",
    "arizona": "Arizona",
    "california": "California",
    "florida": "Florida",
    "houston": "Houston",
    "idaho": "Idaho",
    "illinois": "Illinois",
    "indiana": "Indiana",
    "iona": "Iona",
    "kansas": "Kansas",
    "kentucky": "Kentucky",
    "maryland": "Maryland",
    "michigan": "Michigan",
    "missouri": "Missouri",
    "north carolina": "North Carolina",
    "northwestern": "Northwestern",
    "san diego": "San Diego",
    "southern": "Southern",
    "tennessee": "Tennessee",
    "texas": "Texas",
    "texas a&m": "Texas A&M",
    "utah": "Utah",
    "washington": "Washington",
}


def match_team_to_espn(conn: sqlite3.Connection, barttorvik_name: str) -> int | None:
    """
    Try to match a Barttorvik team name to an ESPN team ID.

    Uses an explicit mapping table for known mismatches, then falls back
    to substring search on ESPN name/short_name/location fields.

    Returns:
        ESPN team ID or None if no match found
    """
    normalized = normalize_team_name(barttorvik_name)

    # Check explicit mapping first
    if normalized in BARTTORVIK_TO_ESPN:
        espn_location = BARTTORVIK_TO_ESPN[normalized]
        cursor = conn.execute(
            "SELECT espn_id FROM teams WHERE location = ?",
            (espn_location,),
        )
        row = cursor.fetchone()
        if row:
            return row[0]

    # Try exact match on ESPN location field
    row = conn.execute(
        "SELECT espn_id FROM teams WHERE LOWER(location) = ?",
        (normalized,),
    ).fetchone()
    if row:
        return row[0]

    # Fall back to substring search on name/short_name
    cursor = conn.execute("""
        SELECT espn_id, name FROM teams
        WHERE LOWER(name) LIKE ? OR LOWER(short_name) LIKE ?
    """, (f"%{normalized}%", f"%{normalized}%"))

    results = cursor.fetchall()

    if len(results) == 1:
        return results[0][0]
    elif len(results) > 1:
        # Prefer exact full name match
        for espn_id, name in results:
            if normalize_team_name(name) == normalized:
                return espn_id
        return results[0][0]

    return None


def parse_barttorvik_row(row: dict, season: int) -> dict:
    """Parse a row from Barttorvik CSV export (direct URL format)."""

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
            # Handle float strings like "37.0"
            return int(float(val))
        except (ValueError, TypeError):
            return None

    # Parse record (e.g., "25-8" or "37-3")
    wins, losses = None, None
    record = row.get('record', row.get('Record', row.get('Rec', '')))
    if record and '-' in str(record):
        parts = str(record).split('-')
        if len(parts) == 2:
            wins = safe_int(parts[0])
            losses = safe_int(parts[1])

    # Get team name - handle both column naming conventions
    team_name = row.get('team', row.get('Team', '')).strip()

    # Get rank - the CSV has 'rank' column (appears twice, first one is overall rank)
    overall_rank = safe_int(row.get('rank', row.get('Rk', row.get('Rank', ''))))

    return {
        'team_name': team_name,
        'season_year': season,
        'overall_rank': overall_rank,
        # Core efficiency metrics - handle multiple naming conventions
        'adj_oe': safe_float(row.get('adjoe', row.get('AdjOE', row.get('AdjO', '')))),
        'adj_de': safe_float(row.get('adjde', row.get('AdjDE', row.get('AdjD', '')))),
        'adj_tempo': safe_float(row.get('adjt', row.get('AdjTempo', row.get('AdjT', '')))),
        'barthag': safe_float(row.get('barthag', row.get('Barthag', ''))),
        # Four Factors (these may not be in direct CSV, set to None if missing)
        'efg_pct': safe_float(row.get('efg', row.get('EFG%', row.get('eFG%', '')))),
        'to_pct': safe_float(row.get('tov', row.get('TO%', row.get('TOV%', '')))),
        'or_pct': safe_float(row.get('orb', row.get('OR%', row.get('ORB%', '')))),
        'ft_rate': safe_float(row.get('ftr', row.get('FTR', row.get('FT/FGA', '')))),
        'efg_pct_d': safe_float(row.get('efgd', row.get('EFG%D', row.get('OEFG%', '')))),
        'to_pct_d': safe_float(row.get('tovd', row.get('TO%D', row.get('OTOV%', '')))),
        'or_pct_d': safe_float(row.get('orbd', row.get('OR%D', row.get('DRB%', '')))),
        'ft_rate_d': safe_float(row.get('ftrd', row.get('FTRD', row.get('OFT/FGA', '')))),
        # Additional metrics
        'two_pt_pct': safe_float(row.get('2pt', row.get('2P%', ''))),
        'three_pt_pct': safe_float(row.get('3pt', row.get('3P%', ''))),
        'three_pt_rate': safe_float(row.get('3PA%', row.get('3PR', ''))),
        'ft_pct': safe_float(row.get('ft', row.get('FT%', ''))),
        'block_pct': safe_float(row.get('blk', row.get('Blk%', ''))),
        'steal_pct': safe_float(row.get('stl', row.get('Stl%', ''))),
        'wins': wins,
        'losses': losses,
    }


def import_barttorvik_csv(
    csv_path: str,
    season: int,
    db_path: str = None,
) -> dict:
    """
    Import Barttorvik data from CSV file.

    Args:
        csv_path: Path to CSV file
        season: Season year (e.g., 2024 for 2023-24)
        db_path: Database path

    Returns:
        Import statistics
    """
    conn = get_connection(db_path)
    create_efficiency_table(conn)

    imported = 0
    matched = 0
    errors = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        # Try to detect delimiter
        sample = f.read(2048)
        f.seek(0)

        if '\t' in sample:
            reader = csv.DictReader(f, delimiter='\t')
        else:
            reader = csv.DictReader(f)

        for row in reader:
            try:
                data = parse_barttorvik_row(row, season)

                if not data['team_name']:
                    continue

                # Try to match to ESPN team
                espn_id = match_team_to_espn(conn, data['team_name'])
                data['espn_team_id'] = espn_id

                if espn_id:
                    matched += 1

                # Insert or update
                conn.execute("""
                    INSERT INTO team_efficiency (
                        team_name, espn_team_id, season_year, overall_rank,
                        adj_oe, adj_de, adj_tempo, barthag,
                        efg_pct, to_pct, or_pct, ft_rate,
                        efg_pct_d, to_pct_d, or_pct_d, ft_rate_d,
                        two_pt_pct, three_pt_pct, three_pt_rate, ft_pct,
                        block_pct, steal_pct, wins, losses, updated_at
                    ) VALUES (
                        :team_name, :espn_team_id, :season_year, :overall_rank,
                        :adj_oe, :adj_de, :adj_tempo, :barthag,
                        :efg_pct, :to_pct, :or_pct, :ft_rate,
                        :efg_pct_d, :to_pct_d, :or_pct_d, :ft_rate_d,
                        :two_pt_pct, :three_pt_pct, :three_pt_rate, :ft_pct,
                        :block_pct, :steal_pct, :wins, :losses, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT(team_name, season_year) DO UPDATE SET
                        espn_team_id = excluded.espn_team_id,
                        overall_rank = excluded.overall_rank,
                        adj_oe = excluded.adj_oe,
                        adj_de = excluded.adj_de,
                        adj_tempo = excluded.adj_tempo,
                        barthag = excluded.barthag,
                        efg_pct = excluded.efg_pct,
                        to_pct = excluded.to_pct,
                        or_pct = excluded.or_pct,
                        ft_rate = excluded.ft_rate,
                        efg_pct_d = excluded.efg_pct_d,
                        to_pct_d = excluded.to_pct_d,
                        or_pct_d = excluded.or_pct_d,
                        ft_rate_d = excluded.ft_rate_d,
                        two_pt_pct = excluded.two_pt_pct,
                        three_pt_pct = excluded.three_pt_pct,
                        three_pt_rate = excluded.three_pt_rate,
                        ft_pct = excluded.ft_pct,
                        block_pct = excluded.block_pct,
                        steal_pct = excluded.steal_pct,
                        wins = excluded.wins,
                        losses = excluded.losses,
                        updated_at = CURRENT_TIMESTAMP
                """, data)

                imported += 1

            except Exception as e:
                errors.append(f"{row.get('Team', 'Unknown')}: {e}")

    conn.commit()
    conn.close()

    return {
        'imported': imported,
        'matched_to_espn': matched,
        'errors': errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Import Barttorvik efficiency data")
    parser.add_argument(
        "csv_file",
        help="Path to Barttorvik CSV export",
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g., 2024 for 2023-24)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path",
    )

    args = parser.parse_args()

    print(f"Importing {args.csv_file} for {args.season-1}-{args.season} season...")

    result = import_barttorvik_csv(
        csv_path=args.csv_file,
        season=args.season,
        db_path=args.db,
    )

    print(f"  Imported: {result['imported']} teams")
    print(f"  Matched to ESPN: {result['matched_to_espn']} teams")

    if result['errors']:
        print(f"  Errors: {len(result['errors'])}")
        for err in result['errors'][:5]:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
