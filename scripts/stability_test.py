"""
XGBoost Stability Test — Feature Reduction + Hyperparameter Sweep.

Tests 8 feature sets × 5 hyperparameter configs × 20 seeds = 800 model trains.
Walk-forward eval: train pre-2026, test 2026.
Records accuracy, AUC, Brier, log_loss, early-stopping round, and prediction
stability for two reference matchups (Michigan vs Arizona, UConn vs Illinois).

Usage:
    python scripts/stability_test.py
    python scripts/stability_test.py --seeds 5        # quick test with fewer seeds
    python scripts/stability_test.py --feature-sets F1 F5  # subset of feature sets
    python scripts/stability_test.py --hyper-configs H1 H3  # subset of configs
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.models.feature_engineering import build_features, get_feature_columns
from src.models.train_model import select_features, augment_data
from src.config import build_team_snapshots, build_matchup_features, predict_symmetric

# ── Feature set definitions (features to DROP from baseline) ────────────────

DROP_SETS = {
    'F1': set(),  # Full baseline

    'F2': {
        'team_a_top5_bpm', 'team_b_top5_bpm', 'diff_top5_bpm',
        'team_a_star_count', 'team_b_star_count',
        'team_a_ranked', 'team_b_ranked',
    },

    'F3': {
        'team_a_roster_bpm', 'team_b_roster_bpm', 'diff_roster_bpm',
        'team_a_top5_bpm', 'team_b_top5_bpm', 'diff_top5_bpm',
        'team_a_star_count', 'team_b_star_count',
    },

    'F4': {
        'team_a_roster_obpm', 'team_b_roster_obpm',
        'team_a_roster_dbpm', 'team_b_roster_dbpm',
        'team_a_top5_bpm', 'team_b_top5_bpm', 'diff_top5_bpm',
        'team_a_star_count', 'team_b_star_count',
        'team_a_ranked', 'team_b_ranked',
        'team_a_run_games', 'team_b_run_games',
    },

    'F5': None,  # built from F4 below
    'F6': None,  # built from F5 below
    'F7': None,  # built from F3 below
    'F8': None,  # built from F6 below
}

# Build composite sets
DROP_SETS['F5'] = DROP_SETS['F4'] | {
    'diff_run_recent_avg_margin', 'diff_run_recent_win_pct',
    'diff_conf_win_pct',
    'conference_game',
}

DROP_SETS['F6'] = DROP_SETS['F5'] | {
    'diff_prior_rank',
    'team_a_depth_count', 'team_b_depth_count',
}

DROP_SETS['F7'] = DROP_SETS['F3'] | {
    'diff_run_recent_avg_margin', 'diff_run_recent_win_pct',
    'team_a_run_games', 'team_b_run_games',
    'team_a_ranked', 'team_b_ranked',
    'conference_game', 'diff_conf_win_pct',
}

DROP_SETS['F8'] = DROP_SETS['F6'] | {
    'team_a_prior_adj_tempo', 'team_b_prior_adj_tempo',
    'tz_advantage',
    'team_a_rank', 'team_b_rank',
}

# ── Hyperparameter configurations ──────────────────────────────────────────

HYPER_CONFIGS = {
    'H1': {'max_depth': 6, 'learning_rate': 0.05, 'n_rounds': 500},
    'H2': {'max_depth': 4, 'learning_rate': 0.02, 'n_rounds': 1500},
    'H3': {'max_depth': 3, 'learning_rate': 0.02, 'n_rounds': 2000},
    'H4': {'max_depth': 4, 'learning_rate': 0.01, 'n_rounds': 2500},
    'H5': {'max_depth': 3, 'learning_rate': 0.01, 'n_rounds': 3000},
}

# Shared XGBoost params (overridden per config)
BASE_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

# ── Reference matchup ESPN IDs ─────────────────────────────────────────────

MATCHUPS = {
    'Michigan vs Arizona': {
        'team_a_name': 'Michigan Wolverines',
        'team_b_name': 'Arizona Wildcats',
    },
    'UConn vs Illinois': {
        'team_a_name': 'UConn Huskies',
        'team_b_name': 'Illinois Fighting Illini',
    },
}


def lookup_espn_id(conn, name: str) -> int:
    """Look up ESPN ID by full team name."""
    row = conn.execute(
        "SELECT espn_id FROM teams WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Team not found: {name}")
    return row['espn_id']


def get_features_for_set(baseline_features: list[str], drop_set: set) -> list[str]:
    """Remove drop_set features from the baseline, preserving order."""
    return [f for f in baseline_features if f not in drop_set]


def run_single_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    params: dict,
    seed: int,
    matchup_dfs: dict[str, pd.DataFrame],
) -> dict:
    """
    Train one model and collect eval metrics + matchup predictions.

    Returns dict with metrics and predictions.
    """
    n_rounds = params.get('n_rounds', 500)
    # Exclude n_rounds from params dict (not a valid xgb.train parameter)
    p = {k: v for k, v in params.items() if k != 'n_rounds'}
    p['seed'] = seed

    # Augment training data
    train_aug = augment_data(train_df)

    X_train = train_aug[features].copy()
    y_train = train_aug['home_winner'].values
    y_test = test_df['home_winner'].values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features,
                         enable_categorical=False)
    dtest = xgb.DMatrix(test_df[features].copy(), label=y_test,
                        feature_names=features, enable_categorical=False)

    model = xgb.train(
        p,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=0,
    )

    # Eval metrics (symmetric prediction)
    y_prob = predict_symmetric(model, test_df, features)
    y_pred = (y_prob >= 0.5).astype(int)

    result = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'brier': brier_score_loss(y_test, y_prob),
        'log_loss': log_loss(y_test, y_prob),
        'best_iteration': model.best_iteration,
        'seed': seed,
    }

    # Reference matchup predictions
    for name, mdf in matchup_dfs.items():
        mdf_feat = mdf.reindex(columns=features, fill_value=0)
        # For features that should be NaN (like rank), restore NaN
        for col in features:
            if col in mdf.columns and mdf[col].isna().any():
                mdf_feat[col] = mdf[col]
        prob = predict_symmetric(model, mdf_feat, features)
        result[f'{name}_prob'] = float(prob[0])

    return result


def main():
    parser = argparse.ArgumentParser(description='XGBoost stability test')
    parser.add_argument('--seeds', type=int, default=20,
                        help='Number of random seeds per combination')
    parser.add_argument('--feature-sets', nargs='+', default=None,
                        help='Subset of feature sets to test (e.g., F1 F5)')
    parser.add_argument('--hyper-configs', nargs='+', default=None,
                        help='Subset of hyper configs to test (e.g., H1 H3)')
    args = parser.parse_args()

    feature_set_names = args.feature_sets or sorted(DROP_SETS.keys())
    hyper_config_names = args.hyper_configs or sorted(HYPER_CONFIGS.keys())
    n_seeds = args.seeds

    total_combos = len(feature_set_names) * len(hyper_config_names)
    total_runs = total_combos * n_seeds
    print(f"Stability test: {len(feature_set_names)} feature sets × "
          f"{len(hyper_config_names)} hyper configs × {n_seeds} seeds "
          f"= {total_runs} total runs\n")

    # ── Load data ───────────────────────────────────────────────────────
    print("Building feature matrix...")
    conn = get_connection()
    df = build_features(conn)

    # Get baseline feature set (same as train_model.py)
    baseline_features = select_features(df)
    print(f"Baseline feature count: {len(baseline_features)}")

    # Print feature set sizes
    for fs_name in sorted(DROP_SETS.keys()):
        feats = get_features_for_set(baseline_features, DROP_SETS[fs_name])
        # Check for features in drop set that aren't in baseline
        missing = DROP_SETS[fs_name] - set(baseline_features)
        if missing:
            print(f"  WARNING: {fs_name} tries to drop features not in baseline: {missing}")
        print(f"  {fs_name}: {len(feats)} features (dropped {len(baseline_features) - len(feats)})")

    # Train/test split: train pre-2026, test 2026
    train_df = df[df['season_year'] < 2026].copy()
    test_df = df[df['season_year'] == 2026].copy()
    print(f"\nTrain: {len(train_df)} games (seasons < 2026)")
    print(f"Test:  {len(test_df)} games (season 2026)")

    # ── Build reference matchup feature vectors ─────────────────────────
    print("\nBuilding reference matchup snapshots...")
    snapshots = build_team_snapshots(df, season=2026)

    matchup_dfs = {}
    for matchup_name, info in MATCHUPS.items():
        team_a_id = lookup_espn_id(conn, info['team_a_name'])
        team_b_id = lookup_espn_id(conn, info['team_b_name'])
        # Build with full baseline features (we'll subset per experiment)
        mdf = build_matchup_features(
            team_a_id, team_b_id, snapshots, baseline_features, neutral=True,
        )
        matchup_dfs[matchup_name] = mdf
        print(f"  {matchup_name}: {info['team_a_name']} (ID {team_a_id}) vs "
              f"{info['team_b_name']} (ID {team_b_id})")

    conn.close()

    # ── Run experiments ─────────────────────────────────────────────────
    all_results = []
    combo_idx = 0
    t_start = time.time()

    for fs_name in feature_set_names:
        features = get_features_for_set(baseline_features, DROP_SETS[fs_name])

        for hc_name in hyper_config_names:
            combo_idx += 1
            params = {**BASE_PARAMS, **HYPER_CONFIGS[hc_name]}

            elapsed = time.time() - t_start
            print(f"\n[{combo_idx}/{total_combos}] {fs_name} × {hc_name} "
                  f"({len(features)} features, depth={params['max_depth']}, "
                  f"lr={params['learning_rate']}, rounds={params['n_rounds']}) "
                  f"[{elapsed:.0f}s elapsed]")

            seed_results = []
            for seed in range(n_seeds):
                result = run_single_experiment(
                    train_df, test_df, features, params, seed, matchup_dfs,
                )
                result['feature_set'] = fs_name
                result['hyper_config'] = hc_name
                result['n_features'] = len(features)
                seed_results.append(result)

                # Progress dot
                print('.', end='', flush=True)

            print()  # newline after dots

            # Summarize this combo
            sr = pd.DataFrame(seed_results)
            print(f"  Acc={sr['accuracy'].mean():.4f}  AUC={sr['auc'].mean():.4f}  "
                  f"Brier={sr['brier'].mean():.4f}  LogLoss={sr['log_loss'].mean():.4f}  "
                  f"StopRound={sr['best_iteration'].mean():.0f}")

            for matchup_name in MATCHUPS:
                col = f'{matchup_name}_prob'
                print(f"  {matchup_name}: mean={sr[col].mean():.4f}  "
                      f"std={sr[col].std():.4f}  "
                      f"spread={sr[col].max() - sr[col].min():.4f}")

            all_results.extend(seed_results)

    # ── Aggregate results ───────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    # Save full results
    out_dir = Path('data/paper')
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / 'stability_analysis.csv', index=False)
    print(f"\nFull results saved to {out_dir / 'stability_analysis.csv'}")

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (sorted by Michigan vs Arizona prediction std, ascending)")
    print("=" * 100)

    summary_rows = []
    for (fs, hc), group in results_df.groupby(['feature_set', 'hyper_config']):
        row = {
            'feature_set': fs,
            'hyper_config': hc,
            'n_features': group['n_features'].iloc[0],
            'accuracy': group['accuracy'].mean(),
            'auc': group['auc'].mean(),
            'brier': group['brier'].mean(),
            'log_loss': group['log_loss'].mean(),
            'mean_stop_round': group['best_iteration'].mean(),
        }
        for matchup_name in MATCHUPS:
            col = f'{matchup_name}_prob'
            row[f'{matchup_name}_mean'] = group[col].mean()
            row[f'{matchup_name}_std'] = group[col].std()
            row[f'{matchup_name}_spread'] = group[col].max() - group[col].min()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values('Michigan vs Arizona_std')

    # Print formatted table
    print(f"\n{'FS':<4} {'HC':<4} {'#F':>3} {'Acc':>6} {'AUC':>6} {'Brier':>6} "
          f"{'LgLoss':>6} {'Stop':>5} │ "
          f"{'MI-AZ μ':>7} {'MI-AZ σ':>7} {'MI-AZ Δ':>7} │ "
          f"{'UC-IL μ':>7} {'UC-IL σ':>7} {'UC-IL Δ':>7}")
    print("─" * 100)

    for _, r in summary.iterrows():
        print(f"{r['feature_set']:<4} {r['hyper_config']:<4} {r['n_features']:>3.0f} "
              f"{r['accuracy']:>6.4f} {r['auc']:>6.4f} {r['brier']:>6.4f} "
              f"{r['log_loss']:>6.4f} {r['mean_stop_round']:>5.0f} │ "
              f"{r['Michigan vs Arizona_mean']:>7.4f} "
              f"{r['Michigan vs Arizona_std']:>7.4f} "
              f"{r['Michigan vs Arizona_spread']:>7.4f} │ "
              f"{r['UConn vs Illinois_mean']:>7.4f} "
              f"{r['UConn vs Illinois_std']:>7.4f} "
              f"{r['UConn vs Illinois_spread']:>7.4f}")

    # Save summary too
    summary.to_csv(out_dir / 'stability_summary.csv', index=False)
    print(f"\nSummary saved to {out_dir / 'stability_summary.csv'}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time / 60:.1f}m)")


if __name__ == '__main__':
    main()
