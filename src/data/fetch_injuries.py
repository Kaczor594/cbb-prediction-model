"""
Fetch college basketball injury reports from RotoWire and insert into DB.

RotoWire's injury report endpoint returns JSON with player injury data.
ESPN's API returns empty for college basketball, and Covers/CBS render
client-side only (not scrapable without a headless browser).

Usage:
    python src/data/fetch_injuries.py                     # all teams
    python src/data/fetch_injuries.py --teams Illinois Michigan  # specific teams
    python src/data/fetch_injuries.py --final-four        # Final Four teams only
    python src/data/fetch_injuries.py --dry-run            # preview without DB insert
    python src/data/fetch_injuries.py --verify             # show what's in the DB now
"""

import argparse
import sqlite3
from datetime import date

import requests

from src.data.database import get_connection
from src.utils import lookup_team_espn_id


ROTOWIRE_URL = "https://www.rotowire.com/cbasketball/tables/injury-report.php"

# RotoWire status -> DB status mapping
STATUS_MAP = {
    'out':                'OUT',
    'out for season':     'OUT',
    'game time decision': 'QUESTIONABLE',
    'day-to-day':         'QUESTIONABLE',
    'doubtful':           'DOUBTFUL',
    'questionable':       'QUESTIONABLE',
    'probable':           'PROBABLE',
}

# Injury descriptions to skip (not real injuries affecting game availability)
SKIP_INJURIES = {'redshirt', 'transfer portal', 'pro draft prep'}


# RotoWire naming quirks -> ESPN location/name
RW_TO_ESPN = {
    "St. Mary's (CAL)": "Saint Mary's",
    "Middle Tennessee St.": "Middle Tennessee",
}


def _match_team_to_espn_id(conn: sqlite3.Connection, team_name: str,
                            cache: dict) -> int | None:
    """Match a RotoWire team name to an ESPN team ID via the teams table."""
    if team_name in cache:
        return cache[team_name]

    lookup = RW_TO_ESPN.get(team_name, team_name)
    espn_id = lookup_team_espn_id(conn, lookup)
    cache[team_name] = espn_id
    return espn_id


def fetch_rotowire_injuries(filter_teams: list[str] = None) -> tuple[list[dict], set[str]]:
    """
    Fetch injury data from RotoWire JSON endpoint.

    Args:
        filter_teams: Optional list of team names to filter (e.g., ['Illinois', 'Michigan']).
                      If None, returns all teams.

    Returns:
        Tuple of (injuries list, all_team_names seen in the raw data).
    """
    params = {'pos': 'ALL', 'conf': 'ALL', 'team': 'ALL'}
    print("  Fetching RotoWire injury report...")
    resp = requests.get(ROTOWIRE_URL, params=params, timeout=60)

    if resp.status_code != 200:
        print(f"    HTTP {resp.status_code}")
        return [], set()

    data = resp.json()
    print(f"    Received {len(data)} total entries")

    rows = []
    all_team_names: set[str] = set()
    filter_set = {t.lower() for t in filter_teams} if filter_teams else None

    for entry in data:
        team_name = entry.get('team', '')
        if team_name:
            all_team_names.add(team_name)

        injury = entry.get('injury', '')
        status_raw = entry.get('status', '').lower()

        # Skip non-injury entries
        if injury.lower() in SKIP_INJURIES:
            continue
        if injury.lower() in ('not injury related', 'suspension'):
            continue

        # Apply team filter
        if filter_set and team_name.lower() not in filter_set:
            continue

        status = STATUS_MAP.get(status_raw)
        if status is None:
            continue

        rows.append({
            'player_name': entry.get('player', ''),
            'team_name': team_name,
            'position': entry.get('position', ''),
            'injury': injury,
            'status': status,
        })

    return rows, all_team_names


def insert_injuries(conn: sqlite3.Connection, injuries: list[dict],
                     season_year: int = 2026) -> dict:
    """
    Insert injury records into the player_injury_reports table.

    Returns dict with counts: {'inserted': N, 'skipped': N, 'unmatched_teams': [...]}
    """
    today = date.today().isoformat()
    team_cache = {}
    inserted = 0
    skipped = 0
    unmatched = set()

    for inj in injuries:
        team_id = _match_team_to_espn_id(conn, inj['team_name'], team_cache)
        if team_id is None:
            unmatched.add(inj['team_name'])
            continue

        try:
            conn.execute("""
                INSERT OR REPLACE INTO player_injury_reports
                (player_name, team_id, season_year, report_date, status, source, detail)
                VALUES (?, ?, ?, ?, ?, 'rotowire', ?)
            """, (
                inj['player_name'],
                team_id,
                season_year,
                today,
                inj['status'],
                inj['injury'],
            ))
            inserted += 1
        except sqlite3.Error as e:
            print(f"    DB error for {inj['player_name']}: {e}")
            skipped += 1

    conn.commit()
    return {
        'inserted': inserted,
        'skipped': skipped,
        'unmatched_teams': sorted(unmatched),
    }


def verify_injuries(conn: sqlite3.Connection, team_names: list[str] | None = None,
                     season_year: int = 2026) -> None:
    """
    Print a summary of all injury records currently in the DB.

    For each team, shows every player with a report, their status, date,
    and source — so you can cross-reference against what you know and spot
    anything missing.
    """
    team_cache = {}

    if team_names:
        team_ids = []
        unmatched = []
        for name in team_names:
            tid = _match_team_to_espn_id(conn, name, team_cache)
            if tid is not None:
                team_ids.append((name, tid))
            else:
                unmatched.append(name)
        if unmatched:
            print(f"  WARNING: Could not match team(s) to DB: {', '.join(unmatched)}")
        if not team_ids:
            print("No matching teams found in DB.")
            return
        placeholders = ','.join('?' for _ in team_ids)
        id_list = [tid for _, tid in team_ids]
        rows = conn.execute(f"""
            SELECT ir.player_name, ir.team_id, t.short_name AS team_name,
                   ir.status, ir.report_date, ir.source, ir.detail
            FROM player_injury_reports ir
            JOIN teams t ON t.espn_id = ir.team_id
            WHERE ir.season_year = ? AND ir.team_id IN ({placeholders})
            ORDER BY t.short_name, ir.status, ir.player_name
        """, [season_year] + id_list).fetchall()

        # Report teams with no injuries — check scrape log to distinguish
        # "scraped and healthy" from "never scraped"
        teams_with_records = {row['team_name'] for row in rows}
        for name, tid in team_ids:
            trow = conn.execute("SELECT short_name FROM teams WHERE espn_id = ?",
                                (tid,)).fetchone()
            short = trow['short_name'] if trow else name
            if short not in teams_with_records:
                # Check scrape log
                scrape_row = conn.execute("""
                    SELECT scrape_date FROM injury_scrape_log
                    WHERE team_id = ? AND season_year = ?
                    ORDER BY scrape_date DESC LIMIT 1
                """, (tid, season_year)).fetchone()
                if scrape_row:
                    print(f"  {short}: scraped {scrape_row['scrape_date']}, "
                          f"no injuries found (clean bill of health)")
                else:
                    print(f"  {short}: not yet scraped (no scrape log entry)")
    else:
        rows = conn.execute("""
            SELECT ir.player_name, ir.team_id, t.short_name AS team_name,
                   ir.status, ir.report_date, ir.source, ir.detail
            FROM player_injury_reports ir
            JOIN teams t ON t.espn_id = ir.team_id
            WHERE ir.season_year = ?
            ORDER BY t.short_name, ir.status, ir.player_name
        """, (season_year,)).fetchall()

    if not rows:
        print(f"No injury records in DB for {season_year} season.")
        if team_names:
            print("This could mean RotoWire had no injuries for these teams,")
            print("or injuries haven't been scraped yet.")
        return

    print(f"\n{'='*80}")
    print(f"  INJURY RECORDS IN DATABASE — {season_year} season")
    print(f"  {len(rows)} record(s) found")
    print(f"{'='*80}\n")

    current_team = None
    for row in rows:
        team = row['team_name']
        if team != current_team:
            if current_team is not None:
                print()
            print(f"  {team}")
            print(f"  {'-' * len(team)}")
            current_team = team
        print(f"    {row['player_name']:25s} {row['status']:15s} "
              f"{row['report_date']}  [{row['source']}]  {row['detail'] or ''}")

    # Show scraped-but-healthy teams from the scrape log
    teams_with_injuries = {row['team_id'] for row in rows}
    try:
        healthy_rows = conn.execute("""
            SELECT sl.team_id, t.short_name AS team_name,
                   sl.scrape_date, sl.source
            FROM injury_scrape_log sl
            JOIN teams t ON t.espn_id = sl.team_id
            WHERE sl.season_year = ? AND sl.injuries_found = 0
            ORDER BY sl.scrape_date DESC, t.short_name
        """, (season_year,)).fetchall()
        # Filter to teams not already shown in injury list
        healthy_teams = [r for r in healthy_rows if r['team_id'] not in teams_with_injuries]
        if healthy_teams:
            print(f"\n  SCRAPED — NO INJURIES FOUND:")
            for r in healthy_teams:
                print(f"    {r['team_name']:25s} scraped {r['scrape_date']}  [{r['source']}]")
    except sqlite3.OperationalError:
        pass  # table may not exist yet

    print(f"\n{'='*80}")
    print("  If a player you know about is MISSING from this list,")
    print("  add them to data/player_overrides.json:")
    print('  {"overrides": [{"player_name": "...", "team": "<espn_id>",')
    print('    "status": "OUT", "note": "reason"}]}')
    print(f"{'='*80}\n")


def scrape_injuries(conn: sqlite3.Connection,
                     filter_teams: list[str] = None,
                     season_year: int = 2026,
                     verbose: bool = True) -> set[int]:
    """
    Scrape RotoWire injuries and insert into DB.  Returns the set of ESPN
    team IDs that were *checked* (including teams with zero injuries).

    This is the programmatic entry point for the training/prediction pipeline.
    Injury records inserted here feed into the eligibility model's Layer 2
    (injury report lookup), setting P(plays)=0 for OUT players.  The returned
    set is logged to injury_scrape_log for historical tracking and --verify.
    """
    # Skip if we already scraped recently (within the last hour)
    try:
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        recent = conn.execute(
            "SELECT COUNT(*) AS n FROM injury_scrape_log WHERE created_at > ?",
            (cutoff,)
        ).fetchone()
        if recent and recent['n'] > 0:
            if verbose:
                print("Injury reports already scraped within the last hour, skipping.")
            # Return the set of team IDs from the most recent scrape
            rows = conn.execute(
                "SELECT DISTINCT team_id FROM injury_scrape_log "
                "WHERE scrape_date = date('now') AND season_year = ?",
                (season_year,)
            ).fetchall()
            return {row['team_id'] for row in rows}
    except sqlite3.OperationalError:
        pass  # table may not exist yet

    if verbose:
        print("Fetching injury reports from RotoWire...")
    injuries, all_team_names = fetch_rotowire_injuries(filter_teams)

    # Resolve every filtered team to an ESPN ID, even if they have no injuries
    team_cache: dict = {}
    scraped_ids: set[int] = set()

    if filter_teams:
        for name in filter_teams:
            tid = _match_team_to_espn_id(conn, name, team_cache)
            if tid is not None:
                scraped_ids.add(tid)
    else:
        # All-teams scrape: resolve every team that appeared in the raw data
        for team_name in all_team_names:
            tid = _match_team_to_espn_id(conn, team_name, team_cache)
            if tid is not None:
                scraped_ids.add(tid)

    # Count injuries per team (for logging) and insert into DB
    injury_counts: dict[int, int] = {}
    if injuries:
        if verbose:
            print(f"  {len(injuries)} injuries found")
            for inj in injuries:
                print(f"    {inj['player_name']:25s} {inj['team_name']:20s} "
                      f"{inj['status']:15s} {inj['injury']}")
        result = insert_injuries(conn, injuries, season_year=season_year)
        if verbose:
            print(f"  DB: {result['inserted']} inserted, {result['skipped']} skipped")
        for inj in injuries:
            tid = _match_team_to_espn_id(conn, inj['team_name'], team_cache)
            if tid is not None:
                scraped_ids.add(tid)
                injury_counts[tid] = injury_counts.get(tid, 0) + 1
    elif verbose:
        print("  No injuries found for specified teams")

    # Log all scraped teams (including those with zero injuries) for historical tracking
    today = date.today().isoformat()

    for tid in scraped_ids:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO injury_scrape_log
                (team_id, season_year, scrape_date, injuries_found, source)
                VALUES (?, ?, ?, ?, 'rotowire')
            """, (tid, season_year, today, injury_counts.get(tid, 0)))
        except sqlite3.Error:
            pass  # table may not exist in older DBs
    conn.commit()

    if verbose:
        healthy_count = sum(1 for tid in scraped_ids if injury_counts.get(tid, 0) == 0)
        print(f"  Scraped {len(scraped_ids)} team(s): "
              f"{len(injury_counts)} with injuries, {healthy_count} healthy")

    return scraped_ids


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NCAAB injury reports from RotoWire"
    )
    parser.add_argument('--teams', nargs='+',
                        help='Team names to filter (e.g., Illinois Michigan)')
    parser.add_argument('--final-four', action='store_true',
                        help='Fetch Final Four teams only')
    parser.add_argument('--dry-run', action='store_true',
                        help='Fetch and display without inserting into DB')
    parser.add_argument('--verify', action='store_true',
                        help='Show all injury records currently in the DB')
    parser.add_argument('--season', type=int, default=2026)
    args = parser.parse_args()

    if args.final_four:
        filter_teams = ['Illinois', 'Michigan', 'UConn', 'Arizona']
    elif args.teams:
        filter_teams = args.teams
    else:
        filter_teams = None

    # --verify: just show what's in the DB and exit
    if args.verify:
        conn = get_connection()
        verify_injuries(conn, filter_teams, season_year=args.season)
        conn.close()
        return

    if args.dry_run:
        # Fetch and display without DB insert
        injuries, _ = fetch_rotowire_injuries(filter_teams)
        if not injuries:
            print("No injury data found for specified teams.")
            return
        print(f"\n{len(injuries)} relevant injuries found:")
        for inj in injuries:
            print(f"  {inj['player_name']:25s} {inj['team_name']:20s} "
                  f"{inj['status']:15s} {inj['injury']}")
        print("\n[Dry run — not inserting into database]")
        return

    # Normal path: scrape, insert, and log
    conn = get_connection()
    scrape_injuries(conn, filter_teams=filter_teams, season_year=args.season)
    conn.close()


if __name__ == '__main__':
    main()
