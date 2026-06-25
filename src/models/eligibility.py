"""
Player eligibility prediction module.

Computes P(plays next game) for each player using a three-layer system:
  1. Manual overrides (highest priority) — from data/player_overrides.json
  2. Injury report data — from player_injury_reports table (scraped)
  3. Cloglog GLM fallback — from box score participation history

The cloglog model was fit on 473k player-game observations across 3 seasons
(2024-2026) with player-level 5-fold CV. See data/paper/eligibility_model_analysis.md.

Output: P(plays) in [0, 1] for each player, which gets multiplied by
avg_min_when_playing to produce expected-minutes weighting for roster features.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import normalize_name

# ---------------------------------------------------------------------------
# Cloglog GLM coefficients (fit on 473k obs, Brier=0.0765, AUC=0.954)
# ---------------------------------------------------------------------------

CLOGLOG_COEFFICIENTS = {
    'intercept':            -1.1331,
    'log_consec_missed':    -0.9221,
    'avg_min_when_playing':  0.0390,
    'std_min_when_playing':  0.0028,
    'play_rate':             0.5980,
    'starter_rate':         -0.2864,
    'recent_play_rate':      1.0680,
}

# Injury status -> P(plays) mapping (naive priors, to be calibrated with data)
INJURY_STATUS_PROB = {
    'OUT':          0.00,
    'DOUBTFUL':     0.25,
    'QUESTIONABLE': 0.50,
    'PROBABLE':     0.75,
    'AVAILABLE':    1.00,
}


def cloglog_predict(eta: float) -> float:
    """Complementary log-log inverse link: P = 1 - exp(-exp(eta))."""
    # Clamp to avoid overflow
    eta = max(min(eta, 10.0), -10.0)
    return 1.0 - math.exp(-math.exp(eta))


def predict_eligibility_glm(consec_missed: int,
                             avg_min_when_playing: float,
                             std_min_when_playing: float,
                             play_rate: float,
                             starter_rate: float,
                             recent_play_rate: float) -> float:
    """
    Predict P(plays next game) using the fitted cloglog GLM.

    All inputs should be computed from data available BEFORE the game.
    """
    c = CLOGLOG_COEFFICIENTS
    eta = (c['intercept']
           + c['log_consec_missed'] * math.log1p(consec_missed)
           + c['avg_min_when_playing'] * avg_min_when_playing
           + c['std_min_when_playing'] * std_min_when_playing
           + c['play_rate'] * play_rate
           + c['starter_rate'] * starter_rate
           + c['recent_play_rate'] * recent_play_rate)
    return cloglog_predict(eta)


# ---------------------------------------------------------------------------
# Manual overrides
# ---------------------------------------------------------------------------

def load_manual_overrides(path: Optional[str] = None) -> dict:
    """
    Load manual player overrides from JSON file.

    Expected format:
    {
        "overrides": [
            {
                "player_name": "Player Name",
                "team": "Team Name or ESPN ID",
                "status": "OUT" | "DOUBTFUL" | "QUESTIONABLE" | "PROBABLE" | "AVAILABLE",
                "note": "optional reason"
            }
        ]
    }

    Returns dict keyed by (normalized_name, team_identifier).
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / 'data' / 'player_overrides.json'
    else:
        path = Path(path)

    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    overrides = {}
    for entry in data.get('overrides', []):
        name = normalize_name(entry['player_name'])

        team = entry.get('team', '')
        status = entry.get('status', 'OUT').upper()
        prob = INJURY_STATUS_PROB.get(status, 0.0)

        # If team is numeric, treat as espn_id; otherwise store as string
        try:
            team_key = int(team)
        except (ValueError, TypeError):
            team_key = team.strip().lower() if isinstance(team, str) else ''

        overrides[(name, team_key)] = {
            'status': status,
            'prob': prob,
            'source': 'manual',
            'note': entry.get('note', ''),
        }

    return overrides


# ---------------------------------------------------------------------------
# Injury report lookups
# ---------------------------------------------------------------------------

def get_injury_report_prob(conn, player_name: str, team_id: int,
                           game_date: str, season_year: int) -> Optional[float]:
    """
    Look up the most recent injury report for a player before a game date.

    Returns P(plays) based on injury status, or None if no report found.
    Looks back at most 7 days from game_date for a relevant report.
    """
    name = normalize_name(player_name)

    row = conn.execute("""
        SELECT status, source, detail
        FROM player_injury_reports
        WHERE player_name = ?
          AND team_id = ?
          AND season_year = ?
          AND report_date <= ?
          AND report_date >= date(?, '-7 days')
        ORDER BY report_date DESC
        LIMIT 1
    """, (name, team_id, season_year, game_date, game_date)).fetchone()

    if row is None:
        return None

    status = row['status'].upper() if row['status'] else 'OUT'
    return INJURY_STATUS_PROB.get(status, 0.0)


# ---------------------------------------------------------------------------
# Batch eligibility computation for feature engineering
# ---------------------------------------------------------------------------

def build_player_eligibility_features(conn, games_df: pd.DataFrame) -> dict:
    """
    Build per-player eligibility features from box score history.

    For each (team_id, season, player_name), tracks running statistics
    needed by the cloglog GLM:
      - consec_missed: consecutive games missed heading into this game
      - avg_min_when_playing: expanding mean of minutes in games played
      - std_min_when_playing: expanding std of minutes in games played
      - play_rate: cumulative fraction of roster games played
      - starter_rate: cumulative starts / games played
      - recent_play_rate: rolling 5-game play rate

    Returns:
        dict: {game_id: {(team_id, player_name): {
            'p_plays': float,
            'avg_min': float,
            'source': str,          # 'glm', 'injury_report', 'manual'
        }}}
    """
    # Load all player-game records (including DNPs)
    all_pgs = pd.read_sql_query("""
        SELECT
            pgs.game_id, pgs.team_id, pgs.player_name,
            pgs.did_not_play, pgs.minutes, pgs.starter
        FROM player_game_stats pgs
        ORDER BY pgs.game_id
    """, conn)

    if all_pgs.empty:
        return {}

    all_pgs['norm_name'] = all_pgs['player_name'].apply(normalize_name)

    # Map game_id -> (date, season_year)
    game_info = dict(zip(games_df['game_id'],
                         zip(games_df['date'], games_df['season_year'])))

    # Pre-build game_id -> (home_team_id, away_team_id) lookup
    game_teams = dict(zip(games_df['game_id'],
                          zip(games_df['home_team_id'], games_df['away_team_id'])))

    # Add date/season to player game stats
    all_pgs['date'] = all_pgs['game_id'].map(lambda gid: game_info.get(gid, (None, None))[0])
    all_pgs['season_year'] = all_pgs['game_id'].map(lambda gid: game_info.get(gid, (None, None))[1])
    all_pgs = all_pgs.dropna(subset=['date', 'season_year'])
    all_pgs['season_year'] = all_pgs['season_year'].astype(int)
    all_pgs = all_pgs.sort_values('date')

    # did_play indicator
    all_pgs['did_play'] = (~all_pgs['did_not_play'].astype(bool)).astype(int)
    all_pgs['min_played'] = all_pgs['minutes'].fillna(0).astype(float)
    all_pgs['is_starter'] = all_pgs['starter'].fillna(0).astype(int)

    # Load manual overrides
    overrides = load_manual_overrides()

    # Batch-load all injury reports into a lookup dict
    # Key: (norm_name, team_id, report_date) -> status
    # We'll find the most recent report within 7 days of each game date
    injury_reports = {}
    injury_rows = pd.read_sql_query(
        "SELECT player_name, team_id, season_year, report_date, status "
        "FROM player_injury_reports ORDER BY report_date",
        conn,
    )
    if not injury_rows.empty:
        for _, ir in injury_rows.iterrows():
            name = ir['player_name'].strip().lower() if isinstance(ir['player_name'], str) else ''
            key = (name, int(ir['team_id']), int(ir['season_year']))
            if key not in injury_reports:
                injury_reports[key] = []
            injury_reports[key].append((str(ir['report_date']), ir['status']))

    from datetime import datetime, timedelta

    def lookup_injury_prob(player_name: str, team_id: int, season: int,
                           game_date_str: str) -> Optional[float]:
        """Look up injury report from pre-loaded data."""
        reports = injury_reports.get((player_name, team_id, season))
        if not reports:
            return None
        # Find most recent report within 7 days before game_date
        try:
            gd = datetime.strptime(game_date_str, '%Y-%m-%d')
        except ValueError:
            return None
        cutoff = (gd - timedelta(days=7)).strftime('%Y-%m-%d')
        best = None
        for rdate, status in reports:
            if cutoff <= rdate <= game_date_str:
                best = status  # reports are sorted by date, last match wins
        if best is None:
            return None
        return INJURY_STATUS_PROB.get(best.upper(), 0.0)

    # Process each player-team-season group
    # Build running stats incrementally
    # Key: (team_id, season, norm_name) -> running state dict
    player_state = {}

    # game_id -> {(team_id, norm_name): eligibility_info}
    game_eligibility = {}

    # Group by game for chronological processing
    games_sorted = games_df.sort_values('date')
    game_order = list(games_sorted['game_id'])

    # Pre-index player records by game_id
    pgs_by_game = dict(list(all_pgs.groupby('game_id')))

    for gid in game_order:
        info = game_info.get(gid)
        if info is None:
            continue
        game_date, season = info
        game_date_str = str(game_date)[:10] if hasattr(game_date, 'strftime') else str(game_date)[:10]

        teams = game_teams.get(gid)
        if teams is None:
            continue
        h_id, a_id = teams

        game_players = pgs_by_game.get(gid, pd.DataFrame())
        if game_players.empty:
            continue

        elig = {}

        for team_id in [h_id, a_id]:
            team_players = game_players[game_players['team_id'] == team_id]

            for _, prow in team_players.iterrows():
                name = prow['norm_name']
                key = (team_id, season, name)

                state = player_state.get(key)
                if state is None:
                    state = {
                        'roster_games': 0,
                        'games_played': 0,
                        'starts': 0,
                        'consec_missed': 0,
                        'min_list': [],       # minutes in games played (for mean/std)
                        'recent_played': [],  # last 5 did_play indicators
                    }
                    player_state[key] = state

                # --- Compute features BEFORE this game ---
                roster_games = state['roster_games']
                games_played = state['games_played']

                # Need at least 3 prior roster games for reliable features
                if roster_games >= 3:
                    consec_missed = state['consec_missed']
                    min_list = state['min_list']

                    avg_min = np.mean(min_list) if min_list else 0.0
                    std_min = np.std(min_list, ddof=1) if len(min_list) > 1 else 0.0
                    play_rate = games_played / roster_games if roster_games > 0 else 0.0
                    starter_rate = state['starts'] / games_played if games_played > 0 else 0.0
                    recent = state['recent_played'][-5:]
                    recent_play_rate = np.mean(recent) if recent else play_rate

                    # Layer 1: Manual override
                    override = None
                    # Try with espn_id
                    override = overrides.get((name, team_id))
                    if override is None:
                        # Try with team name (would need team lookup - skip for batch)
                        pass

                    if override is not None:
                        p_plays = override['prob']
                        source = 'manual'
                    else:
                        # Layer 2: Injury report (from pre-loaded data)
                        injury_prob = lookup_injury_prob(
                            name, team_id, season, game_date_str
                        )
                        if injury_prob is not None:
                            p_plays = injury_prob
                            source = 'injury_report'
                        else:
                            # Layer 3: Cloglog GLM (no injury data for team)
                            p_plays = predict_eligibility_glm(
                                consec_missed=consec_missed,
                                avg_min_when_playing=avg_min,
                                std_min_when_playing=std_min,
                                play_rate=play_rate,
                                starter_rate=starter_rate,
                                recent_play_rate=recent_play_rate,
                            )
                            source = 'glm'

                    elig[(team_id, name)] = {
                        'p_plays': p_plays,
                        'avg_min': avg_min,
                        'source': source,
                    }

                # --- Update state AFTER computing features ---
                did_play = bool(prow['did_play'])
                state['roster_games'] += 1
                state['recent_played'].append(int(did_play))

                if did_play:
                    state['games_played'] += 1
                    state['starts'] += int(prow['is_starter'])
                    mins = prow['min_played']
                    if mins > 0:
                        state['min_list'].append(mins)
                    state['consec_missed'] = 0
                else:
                    state['consec_missed'] += 1

        if elig:
            game_eligibility[gid] = elig

    return game_eligibility


def get_team_expected_minutes(game_eligibility: dict, game_id: int,
                               team_id: int) -> dict:
    """
    Get expected minutes for all players on a team for a specific game.

    Returns dict: {player_name: {'p_plays': float, 'avg_min': float,
                                   'source': str}}
    """
    elig = game_eligibility.get(game_id, {})
    result = {}
    for (tid, name), info in elig.items():
        if tid == team_id:
            result[name] = info
    return result
