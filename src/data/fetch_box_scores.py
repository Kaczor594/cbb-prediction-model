"""
Fetch per-game player box scores from ESPN game summary endpoint.

Iterates through all games in the database and collects player-level stats
(minutes, points, rebounds, assists, etc.) for each game. Stores results
in the player_game_stats table.

Usage:
    python src/data/fetch_box_scores.py                    # Fetch all missing
    python src/data/fetch_box_scores.py --seasons 2026     # Single season
    python src/data/fetch_box_scores.py --limit 100        # First N games
    python src/data/fetch_box_scores.py --rate-limit 0.5   # Faster (risky)
"""

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection
from src.data.espn_client import ESPNClient


def parse_split_stat(stat_str: str) -> tuple[int | None, int | None]:
    """Parse 'made-attempted' stat strings like '5-10'."""
    if not stat_str or stat_str == '--':
        return None, None
    parts = stat_str.split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    return None, None


def parse_int(val: str) -> int | None:
    """Parse an integer stat, returning None for non-numeric."""
    if not val or val == '--':
        return None
    try:
        return int(val)
    except ValueError:
        return None


def extract_player_stats(game_espn_id: str, game_id: int, summary: dict) -> list[dict]:
    """Extract per-player stats from ESPN game summary boxscore."""
    rows = []
    boxscore = summary.get('boxscore', {})
    players_data = boxscore.get('players', [])

    for team_data in players_data:
        team_id = team_data.get('team', {}).get('id')
        if not team_id:
            continue
        team_id = int(team_id)

        for stat_group in team_data.get('statistics', []):
            stat_names = stat_group.get('names', [])
            # Build index map for stat names
            idx = {name: i for i, name in enumerate(stat_names)}

            for athlete_data in stat_group.get('athletes', []):
                athlete = athlete_data.get('athlete', {})
                player_id = athlete.get('id')
                if not player_id:
                    continue
                player_id = int(player_id)
                player_name = athlete.get('displayName', 'Unknown')
                starter = athlete_data.get('starter', False)
                dnp = athlete_data.get('didNotPlay', False)
                ejected = athlete_data.get('ejected', False)

                stats = athlete_data.get('stats', [])

                if dnp or not stats:
                    rows.append({
                        'game_id': game_id,
                        'team_id': team_id,
                        'player_id': player_id,
                        'player_name': player_name,
                        'starter': starter,
                        'did_not_play': True,
                        'ejected': ejected,
                    })
                    continue

                # Parse stats by position in the array
                fg_made, fg_att = parse_split_stat(stats[idx['FG']] if 'FG' in idx else '')
                three_made, three_att = parse_split_stat(stats[idx['3PT']] if '3PT' in idx else '')
                ft_made, ft_att = parse_split_stat(stats[idx['FT']] if 'FT' in idx else '')

                rows.append({
                    'game_id': game_id,
                    'team_id': team_id,
                    'player_id': player_id,
                    'player_name': player_name,
                    'starter': starter,
                    'did_not_play': False,
                    'ejected': ejected,
                    'minutes': parse_int(stats[idx['MIN']] if 'MIN' in idx else ''),
                    'points': parse_int(stats[idx['PTS']] if 'PTS' in idx else ''),
                    'field_goals_made': fg_made,
                    'field_goals_attempted': fg_att,
                    'three_pointers_made': three_made,
                    'three_pointers_attempted': three_att,
                    'free_throws_made': ft_made,
                    'free_throws_attempted': ft_att,
                    'rebounds': parse_int(stats[idx['REB']] if 'REB' in idx else ''),
                    'offensive_rebounds': parse_int(stats[idx['OREB']] if 'OREB' in idx else ''),
                    'defensive_rebounds': parse_int(stats[idx['DREB']] if 'DREB' in idx else ''),
                    'assists': parse_int(stats[idx['AST']] if 'AST' in idx else ''),
                    'turnovers': parse_int(stats[idx['TO']] if 'TO' in idx else ''),
                    'steals': parse_int(stats[idx['STL']] if 'STL' in idx else ''),
                    'blocks': parse_int(stats[idx['BLK']] if 'BLK' in idx else ''),
                    'fouls': parse_int(stats[idx['PF']] if 'PF' in idx else ''),
                })

    return rows


def insert_player_stats(conn, rows: list[dict]):
    """Insert player game stats rows, skipping duplicates."""
    for row in rows:
        conn.execute("""
            INSERT OR IGNORE INTO player_game_stats (
                game_id, team_id, player_id, player_name,
                starter, did_not_play, ejected,
                minutes, points,
                field_goals_made, field_goals_attempted,
                three_pointers_made, three_pointers_attempted,
                free_throws_made, free_throws_attempted,
                rebounds, offensive_rebounds, defensive_rebounds,
                assists, turnovers, steals, blocks, fouls
            ) VALUES (
                :game_id, :team_id, :player_id, :player_name,
                :starter, :did_not_play, :ejected,
                :minutes, :points,
                :field_goals_made, :field_goals_attempted,
                :three_pointers_made, :three_pointers_attempted,
                :free_throws_made, :free_throws_attempted,
                :rebounds, :offensive_rebounds, :defensive_rebounds,
                :assists, :turnovers, :steals, :blocks, :fouls
            )
        """, {
            'game_id': row['game_id'],
            'team_id': row['team_id'],
            'player_id': row['player_id'],
            'player_name': row['player_name'],
            'starter': row.get('starter', False),
            'did_not_play': row.get('did_not_play', False),
            'ejected': row.get('ejected', False),
            'minutes': row.get('minutes'),
            'points': row.get('points'),
            'field_goals_made': row.get('field_goals_made'),
            'field_goals_attempted': row.get('field_goals_attempted'),
            'three_pointers_made': row.get('three_pointers_made'),
            'three_pointers_attempted': row.get('three_pointers_attempted'),
            'free_throws_made': row.get('free_throws_made'),
            'free_throws_attempted': row.get('free_throws_attempted'),
            'rebounds': row.get('rebounds'),
            'offensive_rebounds': row.get('offensive_rebounds'),
            'defensive_rebounds': row.get('defensive_rebounds'),
            'assists': row.get('assists'),
            'turnovers': row.get('turnovers'),
            'steals': row.get('steals'),
            'blocks': row.get('blocks'),
            'fouls': row.get('fouls'),
        })


def main():
    parser = argparse.ArgumentParser(description='Fetch ESPN box scores')
    parser.add_argument('--seasons', nargs='+', type=int, help='Season years to fetch')
    parser.add_argument('--limit', type=int, help='Max games to fetch')
    parser.add_argument('--rate-limit', type=float, default=0.6,
                        help='Seconds between requests (default: 0.6)')
    args = parser.parse_args()

    conn = get_connection()
    client = ESPNClient(rate_limit_seconds=args.rate_limit)

    # Get games that don't have box score data yet
    query = """
        SELECT g.game_id, g.espn_id, g.season_year
        FROM games g
        WHERE g.espn_id NOT IN (
            SELECT DISTINCT CAST(g2.espn_id AS TEXT)
            FROM games g2
            JOIN player_game_stats pgs ON pgs.game_id = g2.game_id
        )
        AND g.status = 'STATUS_FINAL'
    """
    params = []
    if args.seasons:
        placeholders = ','.join('?' * len(args.seasons))
        query += f" AND g.season_year IN ({placeholders})"
        params.extend(args.seasons)
    query += " ORDER BY g.date"
    if args.limit:
        query += " LIMIT ?"
        params.append(args.limit)

    games = conn.execute(query, params).fetchall()
    total = len(games)
    print(f"Found {total} games without box score data")

    if total == 0:
        print("All games already have box score data")
        conn.close()
        return

    success = 0
    errors = 0
    batch_size = 50

    pbar = tqdm(games, desc="Fetching box scores", unit="game",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    pbar.set_postfix(ok=0, err=0)

    for i, game in enumerate(pbar):
        game_id = game['game_id']
        espn_id = game['espn_id']
        season = game['season_year']

        try:
            summary = client.get_game_summary(str(espn_id))
            rows = extract_player_stats(str(espn_id), game_id, summary)

            if rows:
                insert_player_stats(conn, rows)
                success += 1
            else:
                errors += 1

            pbar.set_postfix(ok=success, err=errors)

            # Commit in batches
            if (i + 1) % batch_size == 0:
                conn.commit()

        except Exception as e:
            errors += 1
            pbar.set_postfix(ok=success, err=errors)
            if 'Too Many Requests' in str(e) or '429' in str(e):
                pbar.write(f"Rate limited — sleeping 10s")
                time.sleep(10)
                # Retry once
                try:
                    summary = client.get_game_summary(str(espn_id))
                    rows = extract_player_stats(str(espn_id), game_id, summary)
                    if rows:
                        insert_player_stats(conn, rows)
                        success += 1
                        errors -= 1
                        pbar.set_postfix(ok=success, err=errors)
                except Exception:
                    pass
            elif '404' in str(e):
                pass  # Game summary not available
            else:
                pbar.write(f"Error for game {espn_id}: {e}")

    pbar.close()
    conn.commit()
    conn.close()

    print(f"\nDone: {success} games fetched, {errors} errors out of {total} total")


if __name__ == '__main__':
    main()
