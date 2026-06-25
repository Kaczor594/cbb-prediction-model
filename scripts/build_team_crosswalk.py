"""Build crosswalk between SportsData.io TeamIDs and ESPN team IDs."""
import os
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection

API_KEY = os.environ.get("ODDS_API_KEY", "")


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    conn = get_connection()

    # Get ESPN teams
    espn_teams = conn.execute("SELECT espn_id, name FROM teams").fetchall()
    espn_map = {t['name'].lower(): t['espn_id'] for t in espn_teams}
    espn_list = [(t['espn_id'], t['name']) for t in espn_teams]

    # Get SportsData teams
    resp = requests.get(
        "https://api.sportsdata.io/v3/cbb/scores/json/teams",
        params={"key": API_KEY}, timeout=30,
    )
    sd_teams = resp.json()

    # Create crosswalk table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_crosswalk (
            espn_id INTEGER PRIMARY KEY REFERENCES teams(espn_id),
            sportsdata_id INTEGER UNIQUE,
            sportsdata_key TEXT,
            sportsdata_name TEXT,
            match_quality REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    matched = 0
    unmatched_espn = []

    for espn_id, espn_name in espn_list:
        best_score = 0
        best_sd = None

        for sd in sd_teams:
            sd_full = f"{sd['School']} {sd['Name']}"

            # Try exact match
            if espn_name.lower() == sd_full.lower():
                best_score = 1.0
                best_sd = sd
                break

            # Try similarity
            score = similarity(espn_name, sd_full)
            if score > best_score:
                best_score = score
                best_sd = sd

        if best_sd and best_score >= 0.6:
            conn.execute("""
                INSERT OR REPLACE INTO team_crosswalk
                (espn_id, sportsdata_id, sportsdata_key, sportsdata_name, match_quality)
                VALUES (?, ?, ?, ?, ?)
            """, (
                espn_id, best_sd['TeamID'], best_sd['Key'],
                f"{best_sd['School']} {best_sd['Name']}", best_score,
            ))
            matched += 1
        else:
            unmatched_espn.append((espn_id, espn_name, best_score,
                                   f"{best_sd['School']} {best_sd['Name']}" if best_sd else "N/A"))

    conn.commit()

    print(f"Matched {matched}/{len(espn_list)} ESPN teams to SportsData.io")

    if unmatched_espn:
        print(f"\nUnmatched ({len(unmatched_espn)}):")
        for eid, ename, score, sd_name in sorted(unmatched_espn, key=lambda x: -x[2]):
            print(f"  ESPN: {ename:40s} -> Best: {sd_name:40s} (score={score:.2f})")

    # Show low-quality matches for review
    low_quality = conn.execute("""
        SELECT espn_id, sportsdata_name, match_quality,
               (SELECT name FROM teams WHERE espn_id = tc.espn_id) as espn_name
        FROM team_crosswalk tc
        WHERE match_quality < 0.8
        ORDER BY match_quality
    """).fetchall()

    if low_quality:
        print(f"\nLow-quality matches ({len(low_quality)}):")
        for row in low_quality:
            print(f"  ESPN: {row['espn_name']:40s} -> SD: {row['sportsdata_name']:40s} (score={row['match_quality']:.2f})")

    conn.close()


if __name__ == "__main__":
    main()
