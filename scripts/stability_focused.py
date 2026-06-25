"""
Focused XGBoost configuration comparison with confidence intervals.

Tests 5 candidate configurations × 100 seeds across walk-forward folds
(train→2025, train→2026). Reports bootstrap 95% CIs and paired significance
tests to determine whether metric differences are real or noise.

Usage:
    python scripts/stability_focused.py
    python scripts/stability_focused.py --seeds 50  # faster
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.models.feature_engineering import build_features
from src.models.train_model import select_features, augment_data
from src.config import predict_symmetric

# ── Configurations under test ───────────────────────────────────────────────

DROP_SETS = {
    'F1': set(),
    'F2': {
        'team_a_top5_bpm', 'team_b_top5_bpm', 'diff_top5_bpm',
        'team_a_star_count', 'team_b_star_count',
        'team_a_ranked', 'team_b_ranked',
    },
    'F4': {
        'team_a_roster_obpm', 'team_b_roster_obpm',
        'team_a_roster_dbpm', 'team_b_roster_dbpm',
        'team_a_top5_bpm', 'team_b_top5_bpm', 'diff_top5_bpm',
        'team_a_star_count', 'team_b_star_count',
        'team_a_ranked', 'team_b_ranked',
        'team_a_run_games', 'team_b_run_games',
    },
    'F8': set(),  # built below
}
DROP_SETS['F8'] = DROP_SETS['F4'] | {
    'diff_run_recent_avg_margin', 'diff_run_recent_win_pct',
    'diff_conf_win_pct', 'conference_game',
} | {
    'diff_prior_rank',
    'team_a_depth_count', 'team_b_depth_count',
} | {
    'team_a_prior_adj_tempo', 'team_b_prior_adj_tempo',
    'tz_advantage',
    'team_a_rank', 'team_b_rank',
}

CONFIGS = {
    'F1×H4': ('F1', {'max_depth': 4, 'learning_rate': 0.01, 'n_rounds': 2500}),
    'F2×H4': ('F2', {'max_depth': 4, 'learning_rate': 0.01, 'n_rounds': 2500}),
    'F4×H4': ('F4', {'max_depth': 4, 'learning_rate': 0.01, 'n_rounds': 2500}),
    'F1×H5': ('F1', {'max_depth': 3, 'learning_rate': 0.01, 'n_rounds': 3000}),
    'F8×H4': ('F8', {'max_depth': 4, 'learning_rate': 0.01, 'n_rounds': 2500}),
}

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

METRIC_NAMES = ['accuracy', 'auc', 'brier', 'log_loss']
# Higher is better for accuracy/auc, lower is better for brier/log_loss
METRIC_HIGHER_BETTER = {'accuracy': True, 'auc': True, 'brier': False, 'log_loss': False}


def train_and_eval(train_df, test_df, features, params, seed):
    """Train one model, return per-game probabilities and labels."""
    n_rounds = params.get('n_rounds', 500)
    p = {k: v for k, v in params.items() if k != 'n_rounds'}
    p['seed'] = seed

    train_aug = augment_data(train_df)
    dtrain = xgb.DMatrix(train_aug[features], label=train_aug['home_winner'],
                         feature_names=features, enable_categorical=False)
    dtest = xgb.DMatrix(test_df[features].copy(), label=test_df['home_winner'],
                        feature_names=features, enable_categorical=False)

    model = xgb.train(
        p, dtrain, num_boost_round=n_rounds,
        evals=[(dtest, 'test')], early_stopping_rounds=50, verbose_eval=0,
    )

    y_prob = predict_symmetric(model, test_df, features)
    y_true = test_df['home_winner'].values
    best_round = model.best_iteration
    return y_true, y_prob, best_round


def compute_metrics(y_true, y_prob):
    """Compute all metrics from labels and probabilities."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob),
    }


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return lo, hi


def paired_test(vals_a, vals_b):
    """Paired t-test on per-seed metric differences. Returns (mean_diff, p_value)."""
    diffs = np.array(vals_a) - np.array(vals_b)
    if np.std(diffs) < 1e-12:
        return np.mean(diffs), 1.0
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    return np.mean(diffs), p_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=100)
    args = parser.parse_args()
    n_seeds = args.seeds

    print(f"Focused comparison: {len(CONFIGS)} configs × {n_seeds} seeds × 2 folds")
    print(f"Configs: {', '.join(CONFIGS.keys())}\n")

    # ── Build features ──────────────────────────────────────────────────
    print("Building feature matrix...")
    conn = get_connection()
    df = build_features(conn)
    conn.close()

    baseline_features = select_features(df)
    print(f"Baseline features: {len(baseline_features)}")

    seasons = sorted(df['season_year'].unique())
    # Walk-forward folds: each season after the first is a test fold
    folds = []
    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        folds.append((train_seasons, test_season))
    print(f"Walk-forward folds: {folds}\n")

    # Precompute feature lists
    config_features = {}
    for cfg_name, (fs_name, _) in CONFIGS.items():
        feats = [f for f in baseline_features if f not in DROP_SETS[fs_name]]
        config_features[cfg_name] = feats
        print(f"  {cfg_name}: {len(feats)} features")

    # ── Run all experiments ─────────────────────────────────────────────
    # Store per-seed, per-fold metrics for each config
    # results[cfg_name][fold_idx] = list of metric dicts (one per seed)
    results = {cfg: {fi: [] for fi in range(len(folds))} for cfg in CONFIGS}
    # Also store per-seed pooled metrics (across folds)
    pooled_results = {cfg: [] for cfg in CONFIGS}

    t_start = time.time()

    for seed in range(n_seeds):
        if seed % 10 == 0:
            elapsed = time.time() - t_start
            print(f"Seed {seed}/{n_seeds} [{elapsed:.0f}s elapsed]")

        # For pooled metrics, accumulate y_true/y_prob across folds per config
        pooled_true = {cfg: [] for cfg in CONFIGS}
        pooled_prob = {cfg: [] for cfg in CONFIGS}

        for fi, (train_seasons, test_season) in enumerate(folds):
            train_df = df[df['season_year'].isin(train_seasons)]
            test_df = df[df['season_year'] == test_season]

            for cfg_name, (fs_name, hyper) in CONFIGS.items():
                features = config_features[cfg_name]
                params = {**BASE_PARAMS, **hyper}

                y_true, y_prob, best_round = train_and_eval(
                    train_df, test_df, features, params, seed
                )

                metrics = compute_metrics(y_true, y_prob)
                metrics['best_round'] = best_round
                metrics['fold'] = test_season
                results[cfg_name][fi].append(metrics)

                pooled_true[cfg_name].append(y_true)
                pooled_prob[cfg_name].append(y_prob)

        # Compute pooled metrics across both folds for this seed
        for cfg_name in CONFIGS:
            all_true = np.concatenate(pooled_true[cfg_name])
            all_prob = np.concatenate(pooled_prob[cfg_name])
            pooled_metrics = compute_metrics(all_true, all_prob)
            pooled_results[cfg_name].append(pooled_metrics)

    total_time = time.time() - t_start
    total_models = n_seeds * len(folds) * len(CONFIGS)
    print(f"\nDone: {total_models} models trained in {total_time:.0f}s "
          f"({total_time/60:.1f}m)\n")

    # ── Per-fold results ────────────────────────────────────────────────
    print("=" * 100)
    print("PER-FOLD RESULTS (mean ± 95% CI across seeds)")
    print("=" * 100)

    for fi, (train_seasons, test_season) in enumerate(folds):
        n_test = len(df[df['season_year'] == test_season])
        print(f"\nFold: train {train_seasons} → test {test_season} (n={n_test})")
        print(f"{'Config':<10} {'Accuracy':>20} {'AUC':>20} {'Brier':>20} "
              f"{'LogLoss':>20} {'StopRnd':>8}")
        print("─" * 100)

        for cfg_name in CONFIGS:
            fold_metrics = results[cfg_name][fi]
            parts = []
            for m in METRIC_NAMES:
                vals = [r[m] for r in fold_metrics]
                mean = np.mean(vals)
                lo, hi = bootstrap_ci(vals)
                parts.append(f"{mean:.4f} [{lo:.4f},{hi:.4f}]")
            stop_vals = [r['best_round'] for r in fold_metrics]
            parts.append(f"{np.mean(stop_vals):>6.0f}")
            print(f"{cfg_name:<10} {'  '.join(parts)}")

    # ── Pooled results (primary comparison) ─────────────────────────────
    print("\n" + "=" * 100)
    print("POOLED RESULTS — all folds combined (mean ± 95% CI across seeds)")
    print("=" * 100)

    print(f"\n{'Config':<10} {'Accuracy':>20} {'AUC':>20} {'Brier':>20} "
          f"{'LogLoss':>20}")
    print("─" * 92)

    pooled_summary = {}
    for cfg_name in CONFIGS:
        pooled_summary[cfg_name] = {}
        parts = []
        for m in METRIC_NAMES:
            vals = [r[m] for r in pooled_results[cfg_name]]
            mean = np.mean(vals)
            lo, hi = bootstrap_ci(vals)
            pooled_summary[cfg_name][m] = {'mean': mean, 'lo': lo, 'hi': hi,
                                           'vals': vals}
            parts.append(f"{mean:.4f} [{lo:.4f},{hi:.4f}]")
        print(f"{cfg_name:<10} {'  '.join(parts)}")

    # ── Identify best config per metric ─────────────────────────────────
    print("\n" + "=" * 100)
    print("BEST CONFIG PER METRIC")
    print("=" * 100)

    for m in METRIC_NAMES:
        higher_better = METRIC_HIGHER_BETTER[m]
        ranked = sorted(CONFIGS.keys(),
                        key=lambda c: pooled_summary[c][m]['mean'],
                        reverse=higher_better)
        best = ranked[0]
        best_mean = pooled_summary[best][m]['mean']
        best_lo = pooled_summary[best][m]['lo']
        best_hi = pooled_summary[best][m]['hi']
        print(f"\n{m} — best: {best} = {best_mean:.4f} [{best_lo:.4f}, {best_hi:.4f}]")
        # Show whether other configs' CIs overlap with best
        for cfg in ranked[1:]:
            cfg_mean = pooled_summary[cfg][m]['mean']
            cfg_lo = pooled_summary[cfg][m]['lo']
            cfg_hi = pooled_summary[cfg][m]['hi']
            # Check CI overlap
            if higher_better:
                overlaps = cfg_hi >= best_lo
            else:
                overlaps = cfg_lo <= best_hi
            diff = cfg_mean - best_mean
            status = "CIs overlap" if overlaps else "SIGNIFICANT"
            print(f"  {cfg:<10} = {cfg_mean:.4f} [{cfg_lo:.4f}, {cfg_hi:.4f}]  "
                  f"Δ={diff:+.4f}  {status}")

    # ── Pairwise paired t-tests ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("PAIRWISE COMPARISONS (paired t-test on per-seed pooled metrics)")
    print("=" * 100)

    config_names = list(CONFIGS.keys())
    for m in METRIC_NAMES:
        higher_better = METRIC_HIGHER_BETTER[m]
        print(f"\n{m} ({'↑' if higher_better else '↓'} better):")
        print(f"  {'Pair':<22} {'Δ mean':>8} {'p-value':>10} {'Signif':>8}")
        print(f"  {'─'*50}")
        for i in range(len(config_names)):
            for j in range(i + 1, len(config_names)):
                a, b = config_names[i], config_names[j]
                vals_a = pooled_summary[a][m]['vals']
                vals_b = pooled_summary[b][m]['vals']
                mean_diff, p_val = paired_test(vals_a, vals_b)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  {a} vs {b:<10} {mean_diff:>+8.5f} {p_val:>10.4f} {sig:>8}")

    # ── Save detailed results ───────────────────────────────────────────
    out_dir = Path('data/paper')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save pooled per-seed results
    rows = []
    for cfg_name in CONFIGS:
        for seed_idx, metrics in enumerate(pooled_results[cfg_name]):
            row = {'config': cfg_name, 'seed': seed_idx, **metrics}
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / 'stability_focused_pooled.csv', index=False)

    # Save per-fold per-seed results
    rows = []
    for cfg_name in CONFIGS:
        for fi in range(len(folds)):
            for seed_idx, metrics in enumerate(results[cfg_name][fi]):
                row = {'config': cfg_name, 'seed': seed_idx, **metrics}
                rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / 'stability_focused_folds.csv', index=False)

    print(f"\nResults saved to {out_dir / 'stability_focused_pooled.csv'}")
    print(f"Results saved to {out_dir / 'stability_focused_folds.csv'}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)")


if __name__ == '__main__':
    main()
