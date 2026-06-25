"""
Shared configuration constants for the CBB prediction model.

Centralizes XGBoost hyperparameters, tournament venue coordinates,
and shared utility functions used across multiple scripts.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.feature_engineering import haversine_miles

# ── XGBoost Hyperparameters ──────────────────────────────────────────────

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'max_depth': 4,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

XGB_NUM_BOOST_ROUND = 2500


def build_trained_model(train_df, features, augment=True,
                        test_df=None, augment_test=False,
                        early_stopping_rounds=None):
    """Train an XGBoost model with canonical hyperparameters.

    Args:
        train_df: Training DataFrame (must have 'home_winner' column).
        features: List of feature column names.
        augment: If True, apply augment_data() for symmetry (default True).
        test_df: Optional test DataFrame for early stopping evaluation.
        augment_test: If True, augment test_df as well.
        early_stopping_rounds: If set, enable early stopping with this patience.

    Returns:
        Trained xgb.Booster model.
    """
    from src.models.train_model import augment_data

    if augment:
        train_data = augment_data(train_df)
    else:
        train_data = train_df

    dtrain = xgb.DMatrix(train_data[features], label=train_data['home_winner'].values,
                          feature_names=features)
    evals = [(dtrain, 'train')]

    if test_df is not None:
        test_data = augment_data(test_df) if augment_test else test_df
        dtest = xgb.DMatrix(test_data[features], label=test_data['home_winner'].values,
                             feature_names=features)
        evals.append((dtest, 'test'))

    kwargs = {}
    if early_stopping_rounds is not None:
        kwargs['early_stopping_rounds'] = early_stopping_rounds

    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=XGB_NUM_BOOST_ROUND,
                       evals=evals, verbose_eval=0, **kwargs)
    return model


# ── 2026 Final Four Teams ────────────────────────────────────────────────
FINAL_FOUR_TEAMS = {
    'East':    (2, 'UConn', 41),
    'South':   (3, 'Illinois', 356),
    'West':    (1, 'Arizona', 12),
    'Midwest': (1, 'Michigan', 130),
}
FINAL_FOUR_NAMES = [name for _, name, _ in FINAL_FOUR_TEAMS.values()]
FINAL_FOUR_SITE = 'Indianapolis'


# ── Tournament Venue Coordinates ─────────────────────────────────────────
# (latitude, longitude, utc_offset)

VENUE_COORDS = {
    'Dayton':        (39.7589, -84.1916, -5),
    'Buffalo':       (42.8750, -78.8764, -5),
    'Greenville':    (34.8526, -82.3940, -5),
    'Oklahoma City': (35.4634, -97.5151, -6),
    'Portland':      (45.5316, -122.6668, -8),
    'Tampa':         (27.9428, -82.4517, -5),
    'Philadelphia':  (39.9012, -75.1720, -5),
    'San Diego':     (32.7535, -117.1653, -8),
    'St. Louis':     (38.6270, -90.1994, -6),
    'Houston':       (29.7508, -95.3621, -6),
    'San Jose':      (37.3326, -121.9010, -8),
    'Chicago':       (41.8807, -87.6742, -6),
    'Washington DC': (38.8981, -77.0209, -5),
    'Indianapolis':  (39.7638, -86.1555, -5),
}


# ── Shared Functions ─────────────────────────────────────────────────────

def load_team_locations(conn) -> dict:
    """Load team locations from DB. Returns {espn_id: (lat, lon, utc_offset)}."""
    team_locs = {}
    try:
        for row in conn.execute(
            "SELECT espn_id, latitude, longitude, utc_offset FROM team_locations"
        ).fetchall():
            team_locs[row['espn_id']] = (row['latitude'], row['longitude'], row['utc_offset'])
    except Exception:
        pass
    return team_locs


def build_team_snapshots(df: pd.DataFrame, season: int = 2026) -> dict:
    """
    Build a snapshot of current features for every team from the feature matrix.

    For each team, takes the most recent game's features (from either the
    team_a or team_b side) and strips the side prefix, producing a
    side-neutral dict.

    Args:
        df: Full feature DataFrame from build_features().
        season: Season year to snapshot.

    Returns:
        dict: espn_team_id -> {feature_name: value, ...}
    """
    season_df = df[df['season_year'] == season]
    snapshots = {}

    # team_a = original home, team_b = original away
    for side, id_col in [('team_a', 'home_team_id'), ('team_b', 'away_team_id')]:
        side_cols = [c for c in df.columns if c.startswith(f'{side}_')]
        for tid, group in season_df.groupby(id_col):
            latest = group.sort_values('date').iloc[-1]
            tid = int(tid)
            if tid not in snapshots:
                snapshots[tid] = {}
            for col in side_cols:
                base_name = col[len(f'{side}_'):]
                val = latest[col]
                if base_name not in snapshots[tid] or pd.notna(val):
                    snapshots[tid][base_name] = val
            games_col = f'{side}_run_games'
            if games_col in season_df.columns:
                snapshots[tid]['run_games'] = latest[games_col]

    if 'season_progress' in season_df.columns:
        latest_progress = season_df['season_progress'].iloc[-1]
        for tid in snapshots:
            snapshots[tid]['season_progress'] = latest_progress

    return snapshots


def build_matchup_features(team_a_id: int, team_b_id: int,
                           snapshots: dict, features: list[str],
                           neutral: bool = True,
                           venue_city: str = None,
                           team_locs: dict = None) -> pd.DataFrame:
    """
    Construct feature vector for a hypothetical matchup (Variant C).

    team_a = higher seed by convention. Uses symmetric team_a_*/team_b_*
    naming with is_home/is_away venue indicators.

    Args:
        team_a_id: ESPN ID for team A.
        team_b_id: ESPN ID for team B.
        snapshots: Dict from build_team_snapshots().
        features: List of feature column names.
        neutral: Whether this is a neutral-site game (default True for tournament).
        venue_city: Optional venue city name (must be in VENUE_COORDS).
        team_locs: Optional dict of espn_id -> (lat, lon, utc_offset).

    Returns:
        Single-row DataFrame with the expected feature columns.
    """
    a = snapshots.get(team_a_id, {})
    b = snapshots.get(team_b_id, {})

    row = {}

    # Prior season features
    for feat in ['prior_adj_oe', 'prior_adj_de', 'prior_adj_tempo',
                 'prior_barthag', 'prior_rank', 'prior_win_pct']:
        row[f'team_a_{feat}'] = a.get(feat, 100 if 'rank' in feat else 0.5)
        row[f'team_b_{feat}'] = b.get(feat, 100 if 'rank' in feat else 0.5)

    # Running season features
    for feat in ['run_games', 'run_win_pct', 'run_avg_margin',
                 'run_avg_pts_for', 'run_avg_pts_against',
                 'run_recent_win_pct', 'run_recent_avg_margin']:
        row[f'team_a_{feat}'] = a.get(feat, 0)
        row[f'team_b_{feat}'] = b.get(feat, 0)

    row['team_a_sos'] = a.get('sos', 0.5)
    row['team_b_sos'] = b.get('sos', 0.5)
    row['team_a_conf_win_pct'] = a.get('conf_win_pct', 0.5)
    row['team_b_conf_win_pct'] = b.get('conf_win_pct', 0.5)

    # Roster features
    for feat in ['roster_bpm', 'roster_obpm', 'roster_dbpm',
                 'top5_bpm', 'depth_count', 'star_count']:
        row[f'team_a_{feat}'] = a.get(feat, 0)
        row[f'team_b_{feat}'] = b.get(feat, 0)

    # Game context
    row['conference_game'] = 0
    row['team_a_rank'] = np.nan
    row['team_b_rank'] = np.nan
    row['team_a_ranked'] = 0  # tournament teams treated as unranked by default
    row['team_b_ranked'] = 0

    # Venue indicators (Variant C)
    row['is_home'] = 0 if neutral else 1
    row['is_away'] = 0

    # Travel features (symmetric)
    row['travel_advantage'] = 0  # team_a_dist − team_b_dist
    row['tz_advantage'] = 0     # abs(team_a_tz − venue_tz) − abs(team_b_tz − venue_tz)

    if venue_city and team_locs and venue_city in VENUE_COORDS:
        v_lat, v_lon, v_offset = VENUE_COORDS[venue_city]
        a_loc = team_locs.get(team_a_id)
        b_loc = team_locs.get(team_b_id)
        a_dist = haversine_miles(a_loc[0], a_loc[1], v_lat, v_lon) if a_loc else 0
        b_dist = haversine_miles(b_loc[0], b_loc[1], v_lat, v_lon) if b_loc else 0
        row['travel_advantage'] = a_dist - b_dist  # positive = team_a farther

        # tz_advantage: team_a's tz mismatch minus team_b's tz mismatch
        a_tz_shift = abs(a_loc[2] - v_offset) if a_loc else 0
        b_tz_shift = abs(b_loc[2] - v_offset) if b_loc else 0
        row['tz_advantage'] = a_tz_shift - b_tz_shift

    # Differential features (uniform: team_a − team_b)
    diff_metrics = [
        'prior_rank', 'prior_win_pct',
        'run_win_pct', 'run_avg_margin',
        'run_recent_win_pct', 'run_recent_avg_margin',
        'roster_bpm', 'top5_bpm',
        'sos', 'conf_win_pct',
    ]
    for metric in diff_metrics:
        a_val = row.get(f'team_a_{metric}', 0) or 0
        b_val = row.get(f'team_b_{metric}', 0) or 0
        row[f'diff_{metric}'] = a_val - b_val

    # Season progress
    row['season_progress'] = a.get('season_progress', 1.0)

    # Build DataFrame with only the expected features
    result = pd.DataFrame([row])
    for f in features:
        if f not in result.columns:
            result[f] = 0

    return result[features]


def predict_symmetric(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """
    Predict from both orientations and average for symmetry.

    For each row, predicts P(team_a wins) on the original orientation and
    on the mirrored orientation (teams swapped), then returns the average:
    (p_orig + (1 - p_mirror)) / 2.

    Constructs the mirror directly by swapping team_a/team_b columns,
    negating diff_*/travel features, and flipping is_home/is_away (Variant C).

    Args:
        model: Trained XGBoost Booster.
        df: Feature DataFrame (single or multiple rows).
        features: List of feature column names.

    Returns:
        np.ndarray of averaged P(team_a wins) probabilities.
    """
    feat_df = df[features]
    dmat_orig = xgb.DMatrix(feat_df, feature_names=features)
    p_orig = model.predict(dmat_orig)

    # Build mirror directly (avoids augment_data's concat overhead)
    mirror = feat_df.copy()

    # Swap team_a_* ↔ team_b_*
    a_cols = [c for c in features if c.startswith('team_a_')]
    for ac in a_cols:
        bc = f'team_b_{ac[7:]}'
        if bc in features:
            mirror[ac], mirror[bc] = feat_df[bc].values.copy(), feat_df[ac].values.copy()

    # Negate diff_* features
    for c in features:
        if c.startswith('diff_'):
            mirror[c] = -feat_df[c]

    # Negate travel features
    if 'travel_advantage' in features:
        mirror['travel_advantage'] = -feat_df['travel_advantage']
    if 'tz_advantage' in features:
        mirror['tz_advantage'] = -feat_df['tz_advantage']

    # Swap is_home/is_away (Variant C) — swap, don't use 1-x.
    # 1-x produces is_home=1,is_away=1 for neutral games (never in training data).
    # Swapping correctly maps: home→away, away→home, neutral→neutral.
    if 'is_home' in features and 'is_away' in features:
        mirror['is_home'], mirror['is_away'] = feat_df['is_away'].values.copy(), feat_df['is_home'].values.copy()
    elif 'is_home' in features:
        mirror['is_home'] = 1 - feat_df['is_home']
    elif 'is_away' in features:
        mirror['is_away'] = 1 - feat_df['is_away']

    dmat_mirror = xgb.DMatrix(mirror, feature_names=features)
    p_mirror = model.predict(dmat_mirror)

    return (p_orig + (1.0 - p_mirror)) / 2.0
