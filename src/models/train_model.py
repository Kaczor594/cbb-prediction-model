"""
Train XGBoost model for CBB game win probability prediction.

Uses walk-forward validation: train on earlier seasons, test on later ones.
Supports multiple evaluation strategies:
  - Season holdout: train 2024+2025 → test 2026
  - Walk-forward: train 2024 → test 2025, train 2024+2025 → test 2026
  - Time-series split within seasons

Usage:
    python src/models/train_model.py
    python src/models/train_model.py --eval walk-forward
    python src/models/train_model.py --tune   # hyperparameter search
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    confusion_matrix
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection
from src.models.feature_engineering import build_features, get_feature_columns
from src.config import XGB_PARAMS, XGB_NUM_BOOST_ROUND


def select_features(df: pd.DataFrame) -> list[str]:
    """
    Select feature columns for training (Variant C: is_home + is_away).
    Drops columns that are mostly null, redundant pairs, and neutral_site
    (redundant with is_home=0, is_away=0).
    """
    all_features = get_feature_columns(df)

    # Drop columns with >50% missing
    valid = []
    for col in all_features:
        if df[col].isnull().mean() < 0.5:
            valid.append(col)

    # Drop redundant team_a/team_b pairs where diff_* captures the same info.
    # Keeps: BPM pairs (roster_bpm, top5_bpm) because individual
    # team talent levels carry info beyond the differential.
    drop_redundant = {
        # prior pairs (keep diffs)
        'team_a_prior_rank', 'team_b_prior_rank',
        'team_a_prior_win_pct', 'team_b_prior_win_pct',
        # running season pairs (keep diffs)
        'team_a_run_win_pct', 'team_b_run_win_pct',
        'team_a_run_avg_margin', 'team_b_run_avg_margin',
        'team_a_run_recent_win_pct', 'team_b_run_recent_win_pct',
        'team_a_run_recent_avg_margin', 'team_b_run_recent_avg_margin',
        # schedule/conference pairs (keep diffs)
        'team_a_sos', 'team_b_sos',
        'team_a_conf_win_pct', 'team_b_conf_win_pct',
    }
    valid = [c for c in valid if c not in drop_redundant]

    return valid


def augment_data(df: pd.DataFrame, variant: str = 'C') -> pd.DataFrame:
    """
    Duplicate rows with teams swapped to guarantee symmetry.

    For each original row, creates a mirror row where:
    - team_a_* ↔ team_b_* features are swapped
    - diff_* features are negated
    - travel_advantage and tz_advantage are negated
    - target (home_winner) is flipped
    - is_home and is_away are flipped per variant rules

    Args:
        variant: 'A1'/'A2' (no home/away flip), 'B' (flip is_home only),
                 'C' (flip both is_home and is_away). Default 'C'.
    """
    mirror = df.copy()

    # Identify paired columns
    a_cols = [c for c in df.columns if c.startswith('team_a_')]
    a_to_b = {}
    for ac in a_cols:
        bc = f'team_b_{ac[7:]}'
        if bc in df.columns:
            a_to_b[ac] = bc

    # Swap team_a ↔ team_b
    for ac, bc in a_to_b.items():
        mirror[ac], mirror[bc] = df[bc].values.copy(), df[ac].values.copy()

    # Negate diff_* features
    for c in df.columns:
        if c.startswith('diff_'):
            mirror[c] = -df[c]

    # Negate travel features
    if 'travel_advantage' in df.columns:
        mirror['travel_advantage'] = -df['travel_advantage']
    if 'tz_advantage' in df.columns:
        mirror['tz_advantage'] = -df['tz_advantage']

    # Flip target
    mirror['home_winner'] = 1 - df['home_winner']

    # Flip is_home / is_away for non-neutral games (variant-dependent)
    if variant in ('B', 'C') and 'is_home' in mirror.columns:
        neutral = df.get('neutral_site', pd.Series(0, index=df.index)) == 1
        mirror.loc[~neutral, 'is_home'] = 1 - df.loc[~neutral, 'is_home']
    if variant == 'C' and 'is_away' in mirror.columns:
        neutral = df.get('neutral_site', pd.Series(0, index=df.index)) == 1
        mirror.loc[~neutral, 'is_away'] = 1 - df.loc[~neutral, 'is_away']

    # Swap metadata team IDs (for reference, not used as features)
    if 'home_team_id' in mirror.columns and 'away_team_id' in mirror.columns:
        mirror['home_team_id'], mirror['away_team_id'] = (
            df['away_team_id'].values.copy(), df['home_team_id'].values.copy()
        )

    return pd.concat([df, mirror], ignore_index=True)


def train_evaluate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    params: dict,
    verbose: bool = True,
    use_augmentation: bool = True,
) -> dict:
    """Train on train_df, evaluate on test_df. Returns metrics dict.

    When use_augmentation=True (default), training data is augmented with
    team-swapped mirror rows for symmetry (Variant C). Evaluation always
    uses original (non-augmented) test data with predict_symmetric to
    avoid artificially inflated AUC from augmented evaluation.
    """
    from src.config import predict_symmetric

    if use_augmentation:
        train_aug = augment_data(train_df)
    else:
        train_aug = train_df

    X_train = train_aug[features].copy()
    y_train = train_aug['home_winner'].values

    y_test = test_df['home_winner'].values

    # XGBoost handles NaN natively — no need to impute
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features,
                         enable_categorical=False)
    dtest = xgb.DMatrix(test_df[features].copy(), label=y_test,
                        feature_names=features, enable_categorical=False)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get('n_rounds', 500),
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=50 if verbose else 0,
    )

    # Predictions (symmetric averaging for unbiased evaluation)
    y_prob = predict_symmetric(model, test_df, features)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'brier_score': brier_score_loss(y_test, y_prob),
        'log_loss': log_loss(y_test, y_prob),
        'auc': roc_auc_score(y_test, y_prob),
        'home_win_rate_actual': y_test.mean(),
        'home_win_rate_predicted': y_prob.mean(),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'best_iteration': model.best_iteration,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Results ({len(train_df)} train → {len(test_df)} test)")
        print(f"{'='*60}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f} (lower is better)")
        print(f"  Log Loss:    {metrics['log_loss']:.4f}")
        print(f"  AUC:         {metrics['auc']:.4f}")
        print(f"  Actual home win rate:    {metrics['home_win_rate_actual']:.3f}")
        print(f"  Predicted home win rate: {metrics['home_win_rate_predicted']:.3f}")
        print(f"  Best iteration: {metrics['best_iteration']}")
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    return metrics, model


def walk_forward_evaluation(
    df: pd.DataFrame,
    features: list[str],
    params: dict,
    verbose: bool = True,
) -> list[dict]:
    """
    Walk-forward validation across seasons.
    Train on all seasons before test season, test on each subsequent season.
    """
    seasons = sorted(df['season_year'].unique())
    results = []

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_df = df[df['season_year'].isin(train_seasons)]
        test_df = df[df['season_year'] == test_season]

        if verbose:
            print(f"\n{'#'*60}")
            print(f"Walk-forward: train={train_seasons} → test={test_season}")
            print(f"{'#'*60}")

        metrics, model = train_evaluate_split(
            train_df, test_df, features, params, verbose
        )
        metrics['train_seasons'] = train_seasons
        metrics['test_season'] = test_season
        results.append((metrics, model))

    return results


def feature_importance_analysis(model: xgb.Booster, features: list[str],
                                 top_n: int = 20) -> pd.DataFrame:
    """Get and display feature importance."""
    importance = model.get_score(importance_type='gain')

    # Map feature indices to names if needed
    imp_df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in importance.items()
    ]).sort_values('importance', ascending=False)

    print(f"\nTop {top_n} Features (by gain):")
    print("-" * 50)
    for _, row in imp_df.head(top_n).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.1f}")

    return imp_df


def calibration_analysis(y_true: np.ndarray, y_prob: np.ndarray,
                          n_bins: int = 10):
    """Analyze probability calibration."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    print(f"\nCalibration Analysis ({n_bins} bins):")
    print(f"  {'Bin':>12s} {'Predicted':>10s} {'Actual':>10s} {'Count':>8s} {'Gap':>8s}")
    print(f"  {'-'*52}")

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        pred_mean = y_prob[mask].mean()
        actual_mean = y_true[mask].mean()
        count = mask.sum()
        gap = abs(pred_mean - actual_mean)
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}    {pred_mean:10.3f} {actual_mean:10.3f} {count:8d} {gap:8.3f}")


def hyperparameter_search(
    df: pd.DataFrame,
    features: list[str],
    verbose: bool = True,
) -> dict:
    """Simple grid search over key XGBoost hyperparameters."""
    seasons = sorted(df['season_year'].unique())
    # Use second-to-last season as validation
    train_df = df[df['season_year'].isin(seasons[:-1])]
    test_df = df[df['season_year'] == seasons[-1]]

    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
    }

    best_score = float('inf')
    best_params = None
    results = []

    # Random search (50 combinations)
    import random
    random.seed(42)

    combos = []
    for _ in range(50):
        combo = {k: random.choice(v) for k, v in param_grid.items()}
        combos.append(combo)

    for i, combo in enumerate(combos):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'n_rounds': 500,
            **combo,
        }

        metrics, _ = train_evaluate_split(
            train_df, test_df, features, params, verbose=False
        )

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/50] Best log_loss so far: {best_score:.4f}")

        if metrics['log_loss'] < best_score:
            best_score = metrics['log_loss']
            best_params = params.copy()

        results.append({**combo, **metrics})

    if verbose:
        print(f"\nBest hyperparameters (log_loss={best_score:.4f}):")
        for k, v in best_params.items():
            if k not in ('objective', 'eval_metric', 'tree_method', 'n_rounds'):
                print(f"  {k}: {v}")

    return best_params, pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', choices=['holdout', 'walk-forward'],
                        default='walk-forward')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter search')
    parser.add_argument('--save', default='data/model',
                        help='Directory to save model and results')
    args = parser.parse_args()

    # Build features
    conn = get_connection()
    df = build_features(conn)
    conn.close()

    features = select_features(df)
    print(f"\nUsing {len(features)} features")

    # Default parameters (from shared config, with n_rounds for train_evaluate_split)
    params = {**XGB_PARAMS, 'n_rounds': XGB_NUM_BOOST_ROUND}

    # Hyperparameter tuning
    if args.tune:
        print("\nRunning hyperparameter search...")
        best_params, search_results = hyperparameter_search(df, features)
        params.update(best_params)

    # Walk-forward evaluation
    if args.eval == 'walk-forward':
        results = walk_forward_evaluation(df, features, params)
        # Use the last fold's model as the final model
        final_metrics, final_model = results[-1]
    else:
        # Simple holdout: train on all but last season
        seasons = sorted(df['season_year'].unique())
        train_df = df[df['season_year'].isin(seasons[:-1])]
        test_df = df[df['season_year'] == seasons[-1]]
        final_metrics, final_model = train_evaluate_split(
            train_df, test_df, features, params
        )

    # Feature importance
    imp_df = feature_importance_analysis(final_model, features)

    # Calibration
    seasons = sorted(df['season_year'].unique())
    test_df = df[df['season_year'] == seasons[-1]]
    X_test = test_df[features]
    y_test = test_df['home_winner'].values
    dtest = xgb.DMatrix(X_test, feature_names=features)
    y_prob = final_model.predict(dtest)
    calibration_analysis(y_test, y_prob)

    # Export SHAP values for all features (for R visualization)
    contribs = final_model.predict(dtest, pred_contribs=True)  # (n, n_features+1)
    shap_cols = {}
    for i, feat in enumerate(features):
        shap_cols[f'{feat}_value'] = test_df[feat].values
        shap_cols[f'{feat}_shap'] = contribs[:, i]
    shap_cols['neutral_site'] = test_df['neutral_site'].values
    shap_cols['home_winner'] = test_df['home_winner'].values
    shap_cols['season_progress'] = test_df['season_progress'].values
    shap_df = pd.DataFrame(shap_cols)
    shap_path = Path('data/shap_all.csv')
    shap_df.to_csv(shap_path, index=False)
    print(f"\nSHAP values exported to {shap_path} ({len(shap_df)} games x {len(features)} features)")

    # Save model and results
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model
    model_path = save_dir / f'xgb_model_{timestamp}.json'
    final_model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Save feature importance
    imp_path = save_dir / f'feature_importance_{timestamp}.csv'
    imp_df.to_csv(str(imp_path), index=False)

    # Save metrics
    metrics_path = save_dir / f'metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2, default=str)

    # Save feature list
    features_path = save_dir / f'features_{timestamp}.json'
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=2)

    print(f"\nAll artifacts saved to {save_dir}/")

    # Summary
    print(f"\n{'='*60}")
    print("FINAL MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"  Features:     {len(features)}")
    print(f"  Accuracy:     {final_metrics['accuracy']:.4f}")
    print(f"  Brier Score:  {final_metrics['brier_score']:.4f}")
    print(f"  Log Loss:     {final_metrics['log_loss']:.4f}")
    print(f"  AUC:          {final_metrics['auc']:.4f}")


if __name__ == '__main__':
    main()
