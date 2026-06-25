"""
Shared utility functions for the CBB prediction model.
"""

import sqlite3


def normalize_name(name: str) -> str:
    """Normalize player name for matching across data sources."""
    if not name:
        return ''
    name = name.strip().lower()
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' ii', ' iii', ' iv', ' v']:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name


# Bayesian prior strength for blending running roster stats with prior-season values.
# pw = 1 / (1 + games_played / BAYESIAN_PRIOR_GAMES)
# At 5 games: pw ≈ 0.50 (equal weight). At 15 games: pw ≈ 0.25.
BAYESIAN_PRIOR_GAMES = 5


def lookup_team_espn_id(conn: sqlite3.Connection, name: str) -> int | None:
    """
    Look up an ESPN team ID by name, searching multiple columns.

    Searches: name, short_name, location, nickname, abbreviation.
    Returns espn_id or None if no match.
    """
    row = conn.execute("""
        SELECT espn_id FROM teams
        WHERE name = ? OR short_name = ? OR location = ? OR nickname = ?
           OR abbreviation = ?
    """, (name, name, name, name, name)).fetchone()
    return row['espn_id'] if row else None
