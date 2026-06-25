"""
ESPN Data Collection Script

Collects historical college basketball data from ESPN API and stores it in SQLite.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.espn_client import ESPNClient, extract_game_data, extract_team_data
from src.data.database import get_connection, initialize_database, upsert_team, upsert_game


def collect_teams(client: ESPNClient, conn) -> int:
    """
    Collect all D1 teams and store in database.

    Returns:
        Number of teams collected
    """
    print("Collecting teams...")
    teams = client.get_teams(limit=400)

    count = 0
    for team in teams:
        team_data = extract_team_data(team)
        upsert_team(conn, team_data)
        count += 1

    conn.commit()
    print(f"  Collected {count} teams")
    return count


def collect_conferences(client: ESPNClient, conn) -> int:
    """
    Collect all D1 conferences, store in database, and link teams.

    The ESPN groups endpoint returns actual conferences nested under
    top-level groups, with team memberships included. Each conference's
    numeric ESPN group ID is resolved by querying one member team.

    Returns:
        Number of conferences collected
    """
    print("Collecting conferences...")
    conferences = client.get_conferences()

    count = 0
    teams_linked = 0
    for conf in conferences:
        espn_id = conf.get("id")
        if espn_id is not None:
            espn_id = int(espn_id)

        # Use abbreviation as fallback unique key when espn_id is None
        if espn_id is not None:
            conn.execute("""
                INSERT INTO conferences (espn_id, name, short_name, abbreviation)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(espn_id) DO UPDATE SET
                    name = excluded.name,
                    short_name = excluded.short_name,
                    abbreviation = excluded.abbreviation
            """, (
                espn_id,
                conf.get("name"),
                conf.get("shortName"),
                conf.get("abbreviation"),
            ))
        else:
            # No ESPN ID available — insert by name, skip if already exists
            existing = conn.execute(
                "SELECT conference_id FROM conferences WHERE abbreviation = ?",
                (conf.get("abbreviation"),),
            ).fetchone()
            if not existing:
                conn.execute("""
                    INSERT INTO conferences (name, short_name, abbreviation)
                    VALUES (?, ?, ?)
                """, (
                    conf.get("name"),
                    conf.get("shortName"),
                    conf.get("abbreviation"),
                ))

        count += 1

        # Link member teams to this conference
        conf_row = conn.execute(
            "SELECT conference_id FROM conferences WHERE abbreviation = ?",
            (conf.get("abbreviation"),),
        ).fetchone()

        if conf_row:
            conf_db_id = conf_row[0]
            for team_id in conf.get("team_ids", []):
                conn.execute(
                    "UPDATE teams SET conference_id = ? WHERE espn_id = ?",
                    (conf_db_id, int(team_id)),
                )
                teams_linked += 1

    conn.commit()
    print(f"  Collected {count} conferences, linked {teams_linked} teams")

    # The ESPN groups endpoint is incomplete (~25 of ~31 conferences).
    # Fill gaps by querying each unlinked team individually.
    unlinked = conn.execute(
        "SELECT espn_id FROM teams WHERE conference_id IS NULL"
    ).fetchall()

    if unlinked:
        print(f"  Resolving {len(unlinked)} unlinked teams...")
        new_confs = {}  # espn_conf_id -> {name, team_ids}

        for (team_id,) in unlinked:
            try:
                team_data = client.get_team(team_id)
                groups = team_data.get("groups", {})
                conf_id = groups.get("id")
                summary = team_data.get("standingSummary", "")
                conf_name = summary.split(" in ")[-1] if " in " in summary else None

                if conf_id:
                    if conf_id not in new_confs:
                        new_confs[conf_id] = {"name": conf_name, "team_ids": []}
                    new_confs[conf_id]["team_ids"].append(team_id)
            except Exception:
                pass

        extra_linked = 0
        for conf_espn_id, info in new_confs.items():
            conf_name = info["name"] or f"Conference {conf_espn_id}"
            existing = conn.execute(
                "SELECT conference_id FROM conferences WHERE espn_id = ?",
                (int(conf_espn_id),),
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO conferences (espn_id, name) VALUES (?, ?)",
                    (int(conf_espn_id), conf_name),
                )
                count += 1

            conf_row = conn.execute(
                "SELECT conference_id FROM conferences WHERE espn_id = ?",
                (int(conf_espn_id),),
            ).fetchone()
            if conf_row:
                for tid in info["team_ids"]:
                    conn.execute(
                        "UPDATE teams SET conference_id = ? WHERE espn_id = ?",
                        (conf_row[0], tid),
                    )
                    extra_linked += 1

        conn.commit()
        print(f"  Found {len(new_confs)} additional conferences, linked {extra_linked} more teams")

    total_confs = conn.execute("SELECT COUNT(*) FROM conferences").fetchone()[0]
    total_linked = conn.execute(
        "SELECT COUNT(*) FROM teams WHERE conference_id IS NOT NULL"
    ).fetchone()[0]
    total_teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    print(f"  Final: {total_confs} conferences, {total_linked}/{total_teams} teams linked")

    return count


def collect_team_schedule(
    client: ESPNClient,
    conn,
    team_id: int,
    season: int,
) -> int:
    """
    Collect schedule and results for a team.

    Returns:
        Number of games collected
    """
    try:
        games = client.get_team_schedule(team_id, season)
    except Exception as e:
        print(f"    Error fetching schedule for team {team_id}: {e}")
        return 0

    count = 0
    for game in games:
        try:
            game_data = extract_game_data(game)
            # Only store completed games
            if game_data.get("status") == "STATUS_FINAL":
                upsert_game(conn, game_data)
                count += 1
        except Exception as e:
            print(f"    Error processing game: {e}")
            continue

    return count


def collect_season_data(
    client: ESPNClient,
    conn,
    season: int,
    team_ids: list[int] = None,
) -> dict:
    """
    Collect all game data for a season.

    Args:
        client: ESPN client
        conn: Database connection
        season: Season year
        team_ids: Optional list of team IDs to collect (None = all teams)

    Returns:
        Collection statistics
    """
    print(f"\nCollecting data for {season-1}-{season} season...")

    if team_ids is None:
        # Get all team IDs from database
        cursor = conn.execute("SELECT espn_id FROM teams WHERE is_active = 1")
        team_ids = [row[0] for row in cursor.fetchall()]

    total_games = 0
    teams_processed = 0

    for team_id in team_ids:
        games = collect_team_schedule(client, conn, team_id, season)
        total_games += games
        teams_processed += 1

        # Commit periodically
        if teams_processed % 10 == 0:
            conn.commit()
            print(f"  Processed {teams_processed}/{len(team_ids)} teams ({total_games} games)")

    conn.commit()
    print(f"  Season complete: {total_games} games from {teams_processed} teams")

    return {
        "season": season,
        "teams_processed": teams_processed,
        "games_collected": total_games,
    }


def collect_historical_data(
    seasons: list[int],
    db_path: str = None,
    rate_limit: float = 1.0,
):
    """
    Collect historical data for multiple seasons.

    Args:
        seasons: List of season years to collect
        db_path: Path to database
        rate_limit: Seconds between API requests
    """
    print("=" * 60)
    print("ESPN Data Collection")
    print("=" * 60)

    # Initialize
    initialize_database(db_path)
    conn = get_connection(db_path)
    client = ESPNClient(rate_limit_seconds=rate_limit)

    # Collect reference data
    collect_teams(client, conn)
    collect_conferences(client, conn)

    # Collect season data
    results = []
    for season in seasons:
        result = collect_season_data(client, conn, season)
        results.append(result)

    conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("Collection Summary")
    print("=" * 60)
    total_games = 0
    for r in results:
        print(f"  {r['season']-1}-{r['season']}: {r['games_collected']} games")
        total_games += r["games_collected"]
    print(f"\n  Total: {total_games} games collected")

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect ESPN college basketball data")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2024, 2025, 2026],
        help="Season years to collect (e.g., 2024 for 2023-24 season)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path (default: from config)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds between API requests",
    )

    args = parser.parse_args()

    collect_historical_data(
        seasons=args.seasons,
        db_path=args.db,
        rate_limit=args.rate_limit,
    )


if __name__ == "__main__":
    main()
