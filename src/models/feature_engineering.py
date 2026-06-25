"""
Feature engineering for CBB game prediction.

Builds a training dataset from the SQLite database with per-game features
computed using only information available BEFORE each game (no leakage).

Features:
  - Prior season team efficiency (Barttorvik adj_oe, adj_de, barthag, etc.)
  - Running current-season performance (win%, avg margin, recent form)
  - Game context (neutral site, conference game, AP rank)
  - Player impact (roster BPM weighted by P(plays) * cumulative minutes)
  - Travel/venue features (travel_advantage, timezone shift, neutral site)
  - All features computed as home vs away differentials + raw values

Usage:
    python src/models/feature_engineering.py
    python src/models/feature_engineering.py --out data/training_data.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection
from src.models.eligibility import build_player_eligibility_features
from src.utils import normalize_name, BAYESIAN_PRIOR_GAMES


def load_games(conn) -> pd.DataFrame:
    """Load all completed games with scores."""
    df = pd.read_sql_query("""
        SELECT
            game_id, date, season_year,
            neutral_site, conference_game,
            home_team_id, home_score, home_winner, home_rank,
            away_team_id, away_score, away_winner, away_rank
        FROM games
        WHERE status = 'STATUS_FINAL'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY date
    """, conn)
    df['date'] = pd.to_datetime(df['date'])
    df['home_winner'] = df['home_winner'].astype(int)
    df['score_margin'] = df['home_score'] - df['away_score']
    df['total_points'] = df['home_score'] + df['away_score']
    return df


def load_team_efficiency(conn) -> pd.DataFrame:
    """Load Barttorvik team efficiency metrics."""
    return pd.read_sql_query("""
        SELECT
            espn_team_id as team_id, season_year,
            adj_oe, adj_de, adj_tempo, barthag, overall_rank,
            wins, losses
        FROM team_efficiency
        WHERE espn_team_id IS NOT NULL
    """, conn)


def load_player_season_stats(conn) -> pd.DataFrame:
    """Load season-level player stats for roster-based team ratings."""
    return pd.read_sql_query("""
        SELECT
            espn_team_id as team_id, season_year, player_name,
            minutes, bpm, obpm, dbpm,
            rec_rank, class_year
        FROM player_stats
        WHERE espn_team_id IS NOT NULL
          AND minutes IS NOT NULL
          AND games_played >= 5
    """, conn)


def load_player_game_stats(conn) -> pd.DataFrame:
    """Load per-game player box scores (if available)."""
    count = conn.execute("SELECT COUNT(*) FROM player_game_stats").fetchone()[0]
    if count == 0:
        return pd.DataFrame()
    return pd.read_sql_query("""
        SELECT
            pgs.game_id, pgs.team_id, pgs.player_name,
            pgs.minutes
        FROM player_game_stats pgs
        WHERE pgs.did_not_play = 0
    """, conn)


# --------------------------------------------------------------------------
# Shared BPM lookup builders (used by build_running_roster_features and
# simulate_tournament.adjust_roster_for_injuries)
# --------------------------------------------------------------------------

DEFAULT_BPM = {'bpm': -2.0, 'obpm': -1.2, 'dbpm': -0.8}


def build_bpm_lookups(player_season_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Build prior-season BPM lookup dicts from season-level player stats.

    Returns:
        (prior_bpm_team, prior_bpm_any, freshman_bpm) where:
        - prior_bpm_team: (team_id, name, season) -> {bpm, obpm, dbpm}
        - prior_bpm_any: (name, season) -> {bpm, obpm, dbpm} (cross-team fallback)
        - freshman_bpm: (team_id, name, season) -> estimated {bpm, obpm, dbpm}
    """
    prior_bpm_team = {}
    prior_bpm_any = {}

    for _, p in player_season_df.iterrows():
        tid = int(p['team_id'])
        name = normalize_name(p['player_name'])
        season = int(p['season_year'])
        bpm_val = p['bpm'] if pd.notna(p['bpm']) else 0.0
        obpm_val = p['obpm'] if pd.notna(p['obpm']) else 0.0
        dbpm_val = p['dbpm'] if pd.notna(p['dbpm']) else 0.0

        stats = {'bpm': bpm_val, 'obpm': obpm_val, 'dbpm': dbpm_val}

        prior_bpm_team[(tid, name, season + 1)] = stats
        existing = prior_bpm_any.get((name, season + 1))
        if existing is None or bpm_val > existing['bpm']:
            prior_bpm_any[(name, season + 1)] = stats

    freshman_bpm = {}
    if 'rec_rank' in player_season_df.columns and 'class_year' in player_season_df.columns:
        freshmen = player_season_df[
            player_season_df['class_year'].str.strip().str.lower() == 'fr'
        ]
        for _, p in freshmen.iterrows():
            tid = int(p['team_id'])
            name = normalize_name(p['player_name'])
            season = int(p['season_year'])
            rec_rank = p['rec_rank']
            if pd.notna(rec_rank) and rec_rank > 0:
                est_bpm = 0.07 * rec_rank - 4.5
            else:
                est_bpm = -4.0
            freshman_bpm[(tid, name, season)] = {
                'bpm': est_bpm, 'obpm': est_bpm * 0.6, 'dbpm': est_bpm * 0.4,
            }

    return prior_bpm_team, prior_bpm_any, freshman_bpm


def get_prior_bpm(team_id: int, player_name: str, season: int,
                  prior_bpm_team: dict, prior_bpm_any: dict,
                  freshman_bpm: dict) -> dict:
    """Look up prior-season BPM for a player, with fallback chain."""
    name = normalize_name(player_name)
    stats = prior_bpm_team.get((team_id, name, season))
    if stats is not None:
        return stats
    stats = prior_bpm_any.get((name, season))
    if stats is not None:
        return stats
    stats = freshman_bpm.get((team_id, name, season))
    if stats is not None:
        return stats
    return DEFAULT_BPM


# --------------------------------------------------------------------------
# Prior-season team metrics
# --------------------------------------------------------------------------

def build_prior_season_features(team_eff: pd.DataFrame) -> dict:
    """
    Build a lookup: (team_id, season_year) -> prior season metrics.

    For season N, the prior is season N-1's team_efficiency.
    If no prior exists, use league averages.
    """
    league_avgs = {
        'prior_adj_oe': team_eff['adj_oe'].mean(),
        'prior_adj_de': team_eff['adj_de'].mean(),
        'prior_adj_tempo': team_eff['adj_tempo'].mean(),
        'prior_barthag': team_eff['barthag'].mean(),
        'prior_rank': team_eff['overall_rank'].median(),
        'prior_win_pct': 0.5,
    }

    lookup = {}
    for _, row in team_eff.iterrows():
        tid = int(row['team_id'])
        season = int(row['season_year'])
        wins = row['wins'] or 0
        losses = row['losses'] or 0
        total = wins + losses
        lookup[(tid, season + 1)] = {
            'prior_adj_oe': row['adj_oe'],
            'prior_adj_de': row['adj_de'],
            'prior_adj_tempo': row['adj_tempo'],
            'prior_barthag': row['barthag'],
            'prior_rank': row['overall_rank'],
            'prior_win_pct': wins / total if total > 0 else 0.5,
        }

    return lookup, league_avgs


# --------------------------------------------------------------------------
# Current-season running metrics (computed game-by-game, no leakage)
# --------------------------------------------------------------------------

def build_running_season_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute running stats for both teams using only
    games played BEFORE this game in the current season.

    Returns a DataFrame with game_id and running features for home/away.
    """
    # Sort by date for chronological processing
    games = games.sort_values('date').reset_index(drop=True)

    # Track per-team running stats: {(team_id, season): stats_dict}
    team_stats = {}

    def get_empty_stats():
        return {
            'wins': 0, 'losses': 0, 'games': 0,
            'points_for_sum': 0, 'points_against_sum': 0,
            'margin_sum': 0,
            # Last 10 games sliding window
            'recent_results': [],  # list of (margin, won)
            # Strength of schedule: track opponent win rates
            'opponent_win_sum': 0.0,  # sum of opponent win pcts at game time
            'opponent_count': 0,
            # Conference tracking
            'conf_wins': 0, 'conf_games': 0,
        }

    rows = []
    for _, game in games.iterrows():
        season = game['season_year']
        h_id = game['home_team_id']
        a_id = game['away_team_id']

        h_key = (h_id, season)
        a_key = (a_id, season)

        h_stats = team_stats.get(h_key, get_empty_stats())
        a_stats = team_stats.get(a_key, get_empty_stats())

        h_games = h_stats['games']
        a_games = a_stats['games']

        # --- Compute features BEFORE this game ---
        row = {'game_id': game['game_id']}

        # team_a (home) running stats
        row['team_a_run_games'] = h_games
        row['team_a_run_win_pct'] = h_stats['wins'] / h_games if h_games > 0 else 0.5
        row['team_a_run_avg_margin'] = h_stats['margin_sum'] / h_games if h_games > 0 else 0
        row['team_a_run_avg_pts_for'] = h_stats['points_for_sum'] / h_games if h_games > 0 else 70
        row['team_a_run_avg_pts_against'] = h_stats['points_against_sum'] / h_games if h_games > 0 else 70

        # Recent form (last 10)
        recent_h = h_stats['recent_results'][-10:]
        row['team_a_run_recent_win_pct'] = (
            sum(1 for _, w in recent_h if w) / len(recent_h) if recent_h else 0.5
        )
        row['team_a_run_recent_avg_margin'] = (
            sum(m for m, _ in recent_h) / len(recent_h) if recent_h else 0
        )

        # Strength of schedule
        row['team_a_sos'] = (
            h_stats['opponent_win_sum'] / h_stats['opponent_count']
            if h_stats['opponent_count'] > 0 else 0.5
        )
        # Conference record
        row['team_a_conf_win_pct'] = (
            h_stats['conf_wins'] / h_stats['conf_games']
            if h_stats['conf_games'] > 0 else 0.5
        )

        # team_b (away) running stats
        row['team_b_run_games'] = a_games
        row['team_b_run_win_pct'] = a_stats['wins'] / a_games if a_games > 0 else 0.5
        row['team_b_run_avg_margin'] = a_stats['margin_sum'] / a_games if a_games > 0 else 0
        row['team_b_run_avg_pts_for'] = a_stats['points_for_sum'] / a_games if a_games > 0 else 70
        row['team_b_run_avg_pts_against'] = a_stats['points_against_sum'] / a_games if a_games > 0 else 70

        recent_a = a_stats['recent_results'][-10:]
        row['team_b_run_recent_win_pct'] = (
            sum(1 for _, w in recent_a if w) / len(recent_a) if recent_a else 0.5
        )
        row['team_b_run_recent_avg_margin'] = (
            sum(m for m, _ in recent_a) / len(recent_a) if recent_a else 0
        )

        row['team_b_sos'] = (
            a_stats['opponent_win_sum'] / a_stats['opponent_count']
            if a_stats['opponent_count'] > 0 else 0.5
        )
        row['team_b_conf_win_pct'] = (
            a_stats['conf_wins'] / a_stats['conf_games']
            if a_stats['conf_games'] > 0 else 0.5
        )

        rows.append(row)

        # --- Update running stats AFTER computing features ---
        h_margin = game['home_score'] - game['away_score']
        h_won = game['home_winner'] == 1
        is_conf = bool(game.get('conference_game', 0))

        # Get opponent's win pct for SOS (use the pre-game value)
        a_win_pct = a_stats['wins'] / a_stats['games'] if a_stats['games'] > 0 else 0.5
        h_win_pct = h_stats['wins'] / h_stats['games'] if h_stats['games'] > 0 else 0.5

        # Update home team
        if h_key not in team_stats:
            team_stats[h_key] = get_empty_stats()
        s = team_stats[h_key]
        s['games'] += 1
        s['wins'] += int(h_won)
        s['losses'] += int(not h_won)
        s['points_for_sum'] += game['home_score']
        s['points_against_sum'] += game['away_score']
        s['margin_sum'] += h_margin
        s['recent_results'].append((h_margin, h_won))
        s['opponent_win_sum'] += a_win_pct
        s['opponent_count'] += 1
        if is_conf:
            s['conf_games'] += 1
            s['conf_wins'] += int(h_won)

        # Update away team
        if a_key not in team_stats:
            team_stats[a_key] = get_empty_stats()
        s = team_stats[a_key]
        s['games'] += 1
        s['wins'] += int(not h_won)
        s['losses'] += int(h_won)
        s['points_for_sum'] += game['away_score']
        s['points_against_sum'] += game['home_score']
        s['margin_sum'] += -h_margin
        s['recent_results'].append((-h_margin, not h_won))
        s['opponent_win_sum'] += h_win_pct
        s['opponent_count'] += 1
        if is_conf:
            s['conf_games'] += 1
            s['conf_wins'] += int(not h_won)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Roster-based team strength (from Barttorvik season stats)
# --------------------------------------------------------------------------

def _build_static_roster_strength(player_stats: pd.DataFrame) -> dict:
    """
    Compute aggregate roster strength per (team_id, season).
    Uses minutes-weighted BPM as the primary team strength metric.

    Returns lookup: (team_id, season) -> {roster_bpm, roster_obpm, roster_dbpm,
                                           top5_bpm, depth_count, star_count}
    """
    if player_stats.empty:
        return {}

    lookup = {}
    for (tid, season), group in player_stats.groupby(['team_id', 'season_year']):
        # Filter to players with meaningful minutes
        g = group[group['minutes'] >= 5].copy()  # at least 5 min/game average
        if g.empty:
            continue

        # Minutes-weighted BPM
        total_min = g['minutes'].sum()
        if total_min > 0 and g['bpm'].notna().any():
            g_bpm = g[g['bpm'].notna()]
            min_total = g_bpm['minutes'].sum()
            if min_total > 0:
                roster_bpm = (g_bpm['bpm'] * g_bpm['minutes']).sum() / min_total
                roster_obpm = (g_bpm['obpm'] * g_bpm['minutes']).sum() / min_total if g_bpm['obpm'].notna().any() else 0
                roster_dbpm = (g_bpm['dbpm'] * g_bpm['minutes']).sum() / min_total if g_bpm['dbpm'].notna().any() else 0
            else:
                roster_bpm = roster_obpm = roster_dbpm = 0
        else:
            roster_bpm = roster_obpm = roster_dbpm = 0

        # Top 5 players by minutes
        top5 = g.nlargest(5, 'minutes')
        top5_bpm = top5['bpm'].mean() if top5['bpm'].notna().any() else 0

        # Depth: players averaging 10+ minutes
        depth_count = len(g[g['minutes'] >= 10])

        # Star power: players with BPM > 5
        star_count = len(g[(g['bpm'].notna()) & (g['bpm'] > 5)])

        lookup[(int(tid), int(season))] = {
            'roster_bpm': roster_bpm,
            'roster_obpm': roster_obpm,
            'roster_dbpm': roster_dbpm,
            'top5_bpm': top5_bpm,
            'depth_count': depth_count,
            'star_count': star_count,
        }

    return lookup


def build_running_roster_features(games: pd.DataFrame,
                                  player_season_df: pd.DataFrame,
                                  player_game_df: pd.DataFrame,
                                  game_eligibility: dict = None) -> pd.DataFrame:
    """
    Compute running game-by-game roster strength using prior-season BPM
    weighted by eligibility-adjusted cumulative minutes.

    Unlike _build_static_roster_strength() which uses end-of-season data,
    this only uses information available before each game:
      - Prior-season BPM as the player quality signal
      - Weighted by P(plays) * cumulative minutes (eligibility-adjusted)
      - Falls back to raw cumulative minutes if no eligibility data

    Bayesian blends with static prior-season roster values early in the season.

    Args:
        games: DataFrame of games with game_id, date, season_year, etc.
        player_season_df: Season-level player stats (Barttorvik)
        player_game_df: Per-game box scores
        game_eligibility: Optional dict from build_player_eligibility_features().
            {game_id: {(team_id, player_name): {'p_plays', 'avg_min', 'source'}}}

    Returns DataFrame with game_id + running roster features per side.
    """
    if player_season_df.empty or player_game_df.empty:
        return pd.DataFrame()

    # ---- 1. Build prior-season BPM lookup ----
    prior_bpm_team, prior_bpm_any, freshman_bpm = build_bpm_lookups(player_season_df)

    # ---- 2. Build static prior-season roster values for Bayesian init ----
    static_roster = _build_static_roster_strength(player_season_df)

    # Build prior-season static lookup: for season S, use season S-1 static values
    prior_static = {}
    for (tid, season), vals in static_roster.items():
        prior_static[(tid, season + 1)] = vals

    default_static = {
        'roster_bpm': 0.0, 'roster_obpm': 0.0, 'roster_dbpm': 0.0,
        'top5_bpm': 0.0, 'depth_count': 7, 'star_count': 0,
    }

    # ---- 3. Process games chronologically ----
    games_sorted = games.sort_values('date').reset_index(drop=True)

    # Map game_id -> season_year
    game_season = dict(zip(games_sorted['game_id'], games_sorted['season_year']))

    # Build game_id -> date for ordering
    game_date = dict(zip(games_sorted['game_id'], games_sorted['date']))

    # Cumulative minutes tracker: {(team_id, season, normalized_name): total_minutes}
    cum_minutes = {}
    # Track games played per team: {(team_id, season): count}
    team_game_count = {}

    # Pre-index player_game_df by game_id for fast lookup
    if 'game_id' in player_game_df.columns:
        pgdf_by_game = dict(list(player_game_df.groupby('game_id')))
    else:
        pgdf_by_game = {}

    rows = []
    for _, game in games_sorted.iterrows():
        gid = game['game_id']
        season = game['season_year']
        h_id = game['home_team_id']
        a_id = game['away_team_id']

        row = {'game_id': gid}

        # Get eligibility data for this game (if available)
        elig_data = game_eligibility.get(gid, {}) if game_eligibility else {}

        for side, tid in [('team_a', h_id), ('team_b', a_id)]:
            team_key = (tid, season)
            games_played = team_game_count.get(team_key, 0)

            # Collect all players who have appeared for this team so far
            # Each entry: (name, weight, prior_bpm_dict, avg_min_per_game)
            # weight = P(plays) * cum_min (eligibility-adjusted), or raw cum_min (fallback)
            # This scales cumulative minutes by the player's probability of being
            # available, so an injured starter's historical minutes are discounted.
            player_data = []

            for (t, s, name), total_min in cum_minutes.items():
                if t == tid and s == season and total_min > 0:
                    prior = get_prior_bpm(tid, name, season,
                                          prior_bpm_team, prior_bpm_any, freshman_bpm)

                    # Use eligibility probability to weight BPM contributions
                    elig_info = elig_data.get((tid, name))
                    if elig_info is not None:
                        p_plays = elig_info['p_plays']
                        avg_min = elig_info['avg_min']
                        weight = p_plays * total_min
                    else:
                        # Fallback: raw cumulative minutes (original behavior)
                        weight = total_min
                        avg_min = total_min / games_played if games_played > 0 else 0

                    player_data.append((name, weight, prior, avg_min))

            if player_data and games_played > 0:
                # Minutes-weighted roster BPM (eligibility-adjusted cumulative minutes)
                total_weight = sum(w for _, w, _, _ in player_data)
                if total_weight > 0:
                    r_bpm = sum(p['bpm'] * w for _, w, p, _ in player_data) / total_weight
                    r_obpm = sum(p['obpm'] * w for _, w, p, _ in player_data) / total_weight
                    r_dbpm = sum(p['dbpm'] * w for _, w, p, _ in player_data) / total_weight
                else:
                    r_bpm = r_obpm = r_dbpm = 0.0

                # Top 5 by weight (eligibility-adjusted cumulative minutes)
                sorted_players = sorted(player_data, key=lambda x: x[1], reverse=True)
                top5 = sorted_players[:5]
                top5_bpm = np.mean([p['bpm'] for _, _, p, _ in top5])

                # Depth: players with avg >= 10 min/game who are likely available
                # Use weight > 0 to exclude players with P(plays) ≈ 0
                depth_count = sum(
                    1 for _, w, _, am in player_data
                    if am >= 10 and w > 0
                )

                # Star count: prior BPM > 5 and has played this season
                star_count = sum(
                    1 for _, _, p, _ in player_data if p['bpm'] > 5
                )

                # Bayesian blend with prior-season static values
                pw = 1 / (1 + games_played / BAYESIAN_PRIOR_GAMES)
                static = prior_static.get(team_key, default_static)

                row[f'{side}_roster_bpm'] = pw * static['roster_bpm'] + (1 - pw) * r_bpm
                row[f'{side}_roster_obpm'] = pw * static['roster_obpm'] + (1 - pw) * r_obpm
                row[f'{side}_roster_dbpm'] = pw * static['roster_dbpm'] + (1 - pw) * r_dbpm
                row[f'{side}_top5_bpm'] = pw * static['top5_bpm'] + (1 - pw) * top5_bpm
                row[f'{side}_depth_count'] = pw * static['depth_count'] + (1 - pw) * depth_count
                row[f'{side}_star_count'] = pw * static['star_count'] + (1 - pw) * star_count
            else:
                # No game data yet — use prior-season static values
                static = prior_static.get(team_key, default_static)
                row[f'{side}_roster_bpm'] = static['roster_bpm']
                row[f'{side}_roster_obpm'] = static['roster_obpm']
                row[f'{side}_roster_dbpm'] = static['roster_dbpm']
                row[f'{side}_top5_bpm'] = static['top5_bpm']
                row[f'{side}_depth_count'] = static['depth_count']
                row[f'{side}_star_count'] = static['star_count']

        rows.append(row)

        # ---- 4. Update cumulative minutes after processing this game ----
        game_box = pgdf_by_game.get(gid)
        if game_box is not None:
            for _, prow in game_box.iterrows():
                tid_p = prow['team_id']
                pname = normalize_name(prow['player_name'])
                pmin = prow['minutes'] if pd.notna(prow['minutes']) else 0
                key = (tid_p, season, pname)
                cum_minutes[key] = cum_minutes.get(key, 0) + pmin

        # Update game counts for both teams
        team_game_count[(h_id, season)] = team_game_count.get((h_id, season), 0) + 1
        team_game_count[(a_id, season)] = team_game_count.get((a_id, season), 0) + 1

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# Travel distance and timezone features
# --------------------------------------------------------------------------

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in miles."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in miles. All inputs are numpy arrays in degrees."""
    R = 3958.8
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_travel_data(conn) -> tuple[dict, dict]:
    """
    Load team home locations and venue locations from the database.

    Returns:
        team_locs: {espn_id: (lat, lng, utc_offset)}
        venue_locs: {(city, state): (lat, lng, utc_offset)}
    """
    team_locs = {}
    for row in conn.execute(
        "SELECT espn_id, latitude, longitude, utc_offset FROM team_locations"
    ).fetchall():
        team_locs[row['espn_id']] = (row['latitude'], row['longitude'], row['utc_offset'])

    venue_locs = {}
    for row in conn.execute(
        "SELECT city, state, latitude, longitude, utc_offset FROM venue_locations"
    ).fetchall():
        venue_locs[(row['city'], row['state'])] = (row['latitude'], row['longitude'], row['utc_offset'])

    return team_locs, venue_locs


def build_travel_features(games: pd.DataFrame, conn) -> pd.DataFrame:
    """
    Compute travel distance and timezone shift features for ALL games.

    Returns raw travel_advantage and tz_advantage columns (home/away orientation).
    These are negated in build_features() to convert to team_a/team_b symmetric form.

    Raw features computed:
      - travel_advantage: away_dist - home_dist (positive = home team closer)
      - tz_advantage: abs(away_tz - venue_tz) - abs(home_tz - venue_tz)
        Positive = away team has worse timezone shift (home team advantage).
        Computed for ALL games including neutral-site.
    """
    team_locs, venue_locs = load_travel_data(conn)

    if not team_locs or not venue_locs:
        print("  WARNING: No location data found. Run geocode_locations.py first.")
        return pd.DataFrame()

    # Load venue info for each game
    game_venues = pd.read_sql_query(
        "SELECT game_id, venue_city, venue_state FROM games WHERE status = 'STATUS_FINAL'",
        conn
    )
    venue_map = dict(zip(game_venues['game_id'],
                         zip(game_venues['venue_city'], game_venues['venue_state'])))

    # Map each game to venue location
    result = games[['game_id', 'home_team_id', 'away_team_id']].copy()
    result['venue_key'] = result['game_id'].map(venue_map)

    # Look up coordinates for venue, home team, away team
    result['v_lat'] = result['venue_key'].map(lambda k: venue_locs[k][0] if k and k in venue_locs else np.nan)
    result['v_lon'] = result['venue_key'].map(lambda k: venue_locs[k][1] if k and k in venue_locs else np.nan)
    result['v_tz'] = result['venue_key'].map(lambda k: venue_locs[k][2] if k and k in venue_locs else np.nan)

    result['h_lat'] = result['home_team_id'].map(lambda t: team_locs[t][0] if t in team_locs else np.nan)
    result['h_lon'] = result['home_team_id'].map(lambda t: team_locs[t][1] if t in team_locs else np.nan)
    result['h_tz'] = result['home_team_id'].map(lambda t: team_locs[t][2] if t in team_locs else np.nan)

    result['a_lat'] = result['away_team_id'].map(lambda t: team_locs[t][0] if t in team_locs else np.nan)
    result['a_lon'] = result['away_team_id'].map(lambda t: team_locs[t][1] if t in team_locs else np.nan)
    result['a_tz'] = result['away_team_id'].map(lambda t: team_locs[t][2] if t in team_locs else np.nan)

    # Mask: games with valid venue location
    valid = result['v_lat'].notna()

    # Vectorized haversine for home and away teams to venue
    h_has_loc = valid & result['h_lat'].notna()
    a_has_loc = valid & result['a_lat'].notna()

    h_dist = pd.Series(0.0, index=result.index)
    a_dist = pd.Series(0.0, index=result.index)

    if h_has_loc.any():
        h_dist[h_has_loc] = _haversine_vec(
            result.loc[h_has_loc, 'h_lat'].values, result.loc[h_has_loc, 'h_lon'].values,
            result.loc[h_has_loc, 'v_lat'].values, result.loc[h_has_loc, 'v_lon'].values,
        )
    if a_has_loc.any():
        a_dist[a_has_loc] = _haversine_vec(
            result.loc[a_has_loc, 'a_lat'].values, result.loc[a_has_loc, 'a_lon'].values,
            result.loc[a_has_loc, 'v_lat'].values, result.loc[a_has_loc, 'v_lon'].values,
        )

    # Compute features (NaN for games without venue location)
    result['travel_advantage'] = np.where(valid, a_dist - h_dist, np.nan)

    h_tz_shift = np.where(h_has_loc, np.abs(result['h_tz'] - result['v_tz']), 0)
    a_tz_shift = np.where(a_has_loc, np.abs(result['a_tz'] - result['v_tz']), 0)
    result['tz_advantage'] = np.where(valid, a_tz_shift - h_tz_shift, np.nan)

    return result[['game_id', 'travel_advantage', 'tz_advantage']].copy()


# --------------------------------------------------------------------------
# Assemble full feature matrix
# --------------------------------------------------------------------------

def build_features(conn) -> pd.DataFrame:
    """Build the complete feature matrix for all games."""
    print("Loading data...")
    games = load_games(conn)
    team_eff = load_team_efficiency(conn)
    player_stats = load_player_season_stats(conn)
    player_game = load_player_game_stats(conn)

    print(f"  {len(games)} games, {len(team_eff)} team-season records, "
          f"{len(player_stats)} player-season records, "
          f"{len(player_game)} player-game records")

    # 1. Prior season metrics
    print("Building prior-season features...")
    prior_lookup, league_avgs = build_prior_season_features(team_eff)

    # 2. Running season features
    print("Building running season features...")
    running = build_running_season_features(games)

    # 3. Player eligibility (P(plays) from cloglog GLM + overrides + injury reports)
    print("Building player eligibility features...")
    game_eligibility = {}
    if not player_game.empty:
        game_eligibility = build_player_eligibility_features(conn, games)
        if game_eligibility:
            print(f"  Eligibility data for {len(game_eligibility)} games")

    # 4. Running roster strength (game-by-game, no leakage)
    print("Building running roster features...")
    running_roster = pd.DataFrame()
    if not player_game.empty:
        running_roster = build_running_roster_features(
            games, player_stats, player_game,
            game_eligibility=game_eligibility,
        )
        if not running_roster.empty:
            print(f"  Running roster features for {len(running_roster)} games")

    # Static roster as fallback for games without box score data
    # Only build if running roster doesn't cover all games
    roster_lookup = {}
    if running_roster.empty or len(running_roster) < len(games):
        roster_lookup = _build_static_roster_strength(player_stats)

    # 5. Travel distance and timezone features
    print("Building travel distance features...")
    travel = build_travel_features(games, conn)
    if not travel.empty:
        print(f"  Travel features for {len(travel)} games")

    # 7. Assemble everything
    print("Assembling feature matrix...")
    feature_rows = []

    for _, game in games.iterrows():
        gid = game['game_id']
        season = game['season_year']
        h_id = game['home_team_id']
        a_id = game['away_team_id']

        row = {
            'game_id': gid,
            'date': game['date'],
            'season_year': season,
            'home_team_id': h_id,
            'away_team_id': a_id,
            'home_winner': game['home_winner'],
            'score_margin': game['score_margin'],
        }

        # --- Prior season metrics ---
        # team_a = original home, team_b = original away
        h_prior = prior_lookup.get((h_id, season), league_avgs)
        a_prior = prior_lookup.get((a_id, season), league_avgs)
        for key in ['prior_adj_oe', 'prior_adj_de', 'prior_adj_tempo',
                     'prior_barthag', 'prior_rank', 'prior_win_pct']:
            row[f'team_a_{key}'] = h_prior[key]
            row[f'team_b_{key}'] = a_prior[key]

        # --- Game context ---
        row['neutral_site'] = int(game['neutral_site'] or 0)
        row['conference_game'] = int(game['conference_game'] or 0)
        row['team_a_rank'] = game['home_rank'] if pd.notna(game['home_rank']) and game['home_rank'] <= 25 else np.nan
        row['team_b_rank'] = game['away_rank'] if pd.notna(game['away_rank']) and game['away_rank'] <= 25 else np.nan
        row['team_a_ranked'] = int(pd.notna(row['team_a_rank']))
        row['team_b_ranked'] = int(pd.notna(row['team_b_rank']))

        feature_rows.append(row)

    df = pd.DataFrame(feature_rows)

    # Merge running season features
    df = df.merge(running, on='game_id', how='left')

    # Merge running roster features (or fall back to static)
    # Running roster already produces team_a_*/team_b_* columns
    roster_cols = ['roster_bpm', 'roster_obpm', 'roster_dbpm',
                   'top5_bpm', 'depth_count', 'star_count']
    def _fill_roster_from_static(df, mask, roster_lookup, roster_cols):
        """Fill roster columns from static lookup for rows matching mask."""
        for _, row_data in df[mask].iterrows():
            idx = row_data.name
            h_roster = roster_lookup.get(
                (row_data['home_team_id'], row_data['season_year']), {}
            )
            a_roster = roster_lookup.get(
                (row_data['away_team_id'], row_data['season_year']), {}
            )
            for key in roster_cols:
                df.at[idx, f'team_a_{key}'] = h_roster.get(key, 0)
                df.at[idx, f'team_b_{key}'] = a_roster.get(key, 0)

    if not running_roster.empty:
        df = df.merge(running_roster, on='game_id', how='left')
        _fill_roster_from_static(df, df['team_a_roster_bpm'].isna(),
                                 roster_lookup, roster_cols)
    else:
        for side in ['team_a', 'team_b']:
            for key in roster_cols:
                df[f'{side}_{key}'] = 0.0
        _fill_roster_from_static(df, pd.Series(True, index=df.index),
                                 roster_lookup, roster_cols)

    # Merge travel features
    if not travel.empty:
        df = df.merge(travel, on='game_id', how='left')
        # Convert from home/away orientation to team_a/team_b symmetric form.
        # team_a = original home, team_b = original away, so negate both.
        # travel_advantage: raw = away_dist - home_dist → symmetric = team_a_dist - team_b_dist = -raw
        df['travel_advantage'] = -df['travel_advantage'].fillna(0)
        # tz_advantage: raw = away_shift - home_shift → symmetric = team_a_shift - team_b_shift = -raw
        df['tz_advantage'] = -df['tz_advantage'].fillna(0)
    else:
        df['travel_advantage'] = 0
        df['tz_advantage'] = 0

    # --- Venue indicators (Variant C) ---
    df['is_home'] = (df['neutral_site'] == 0).astype(int)
    df['is_away'] = 0  # Before augmentation, team_a is always original home

    # --- Differential features ---
    # Uniform convention: team_a − team_b. Positive = team_a has more.
    diff_metrics = [
        'prior_rank', 'prior_win_pct',
        'run_win_pct', 'run_avg_margin',
        'run_recent_win_pct', 'run_recent_avg_margin',
        'roster_bpm', 'top5_bpm',
        'sos', 'conf_win_pct',
    ]

    for metric in diff_metrics:
        a_col = f'team_a_{metric}'
        b_col = f'team_b_{metric}'
        if a_col in df.columns and b_col in df.columns:
            df[f'diff_{metric}'] = df[a_col] - df[b_col]

    # --- Season progress (for Bayesian weight as a feature) ---
    df['season_progress'] = df[['team_a_run_games', 'team_b_run_games']].mean(axis=1) / 30
    df['season_progress'] = df['season_progress'].clip(0, 1)

    print(f"Feature matrix: {df.shape[0]} games × {df.shape[1]} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (exclude target, IDs, metadata)."""
    exclude = {
        'game_id', 'date', 'season_year', 'home_team_id', 'away_team_id',
        'home_winner', 'score_margin', 'neutral_site',
    }
    return [c for c in df.columns if c not in exclude]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/training_data.csv')
    args = parser.parse_args()

    conn = get_connection()
    df = build_features(conn)
    conn.close()

    # Save
    df.to_csv(args.out, index=False)
    print(f"Saved training data to {args.out}")

    # Summary
    features = get_feature_columns(df)
    print(f"\nFeature count: {len(features)}")
    print(f"Target distribution: {df['home_winner'].mean():.3f} home win rate")
    print(f"Seasons: {sorted(df['season_year'].unique())}")
    print(f"Games per season: {df.groupby('season_year').size().to_dict()}")

    # Missing values
    missing = df[features].isnull().sum()
    if missing.any():
        print(f"\nColumns with missing values:")
        for col in missing[missing > 0].index:
            print(f"  {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")


if __name__ == '__main__':
    main()
