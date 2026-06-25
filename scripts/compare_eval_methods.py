"""
Compare evaluation methods: train_model (augmented test) vs backtest (original test).

Runs walk-forward evaluation on the same game set using three approaches:
  1. train_model style: augmented test data, raw model.predict
  2. backtest style: original test data, raw model.predict
  3. backtest + symmetric: original test data, predict_symmetric

This isolates whether the AUC difference comes from augmented evaluation,
predict_symmetric, or something else.

Usage:
    python scripts/compare_eval_methods.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.models.feature_engineering import build_features
from src.models.train_model import select_features, augment_data
from src.config import predict_symmetric, build_trained_model


def main():
    conn = get_connection()

    print("Building features...")
    df = build_features(conn)
    features = select_features(df)
    conn.close()

    print(f"Using {len(features)} features")
    print(f"Total games: {len(df)}")

    seasons = sorted(df['season_year'].unique())
    print(f"Seasons: {seasons}")

    # Walk-forward: train on prior seasons, test on next
    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_df = df[df['season_year'].isin(train_seasons)].dropna(subset=['home_winner'])
        test_df = df[df['season_year'] == test_season].dropna(subset=['home_winner'])

        if len(train_df) < 100 or len(test_df) < 100:
            continue

        print(f"\n{'='*70}")
        print(f"  Train: {train_seasons} ({len(train_df)} games) → Test: {test_season} ({len(test_df)} games)")
        print(f"{'='*70}")

        # Train model (same for all three methods)
        test_aug = augment_data(test_df)
        model = build_trained_model(train_df, features, augment=True,
                                    test_df=test_df, augment_test=True,
                                    early_stopping_rounds=50)
        print(f"  Best iteration: {model.best_iteration}")

        # Build augmented test DMatrix for Method 1
        dtest_aug = xgb.DMatrix(
            test_aug[features], label=test_aug['home_winner'].values,
            feature_names=features
        )

        # ── Method 1: train_model style (augmented test, raw predict) ──
        y_aug = test_aug['home_winner'].values
        p_aug = model.predict(dtest_aug)

        # ── Method 2: backtest style (original test, raw predict) ──
        dtest_orig = xgb.DMatrix(
            test_df[features], feature_names=features
        )
        y_orig = test_df['home_winner'].values
        p_raw = model.predict(dtest_orig)

        # ── Method 3: backtest + predict_symmetric ──
        p_sym = predict_symmetric(model, test_df, features)

        # ── Print comparison ──
        methods = [
            ('1. Augmented test (train_model)', y_aug, p_aug, len(test_aug)),
            ('2. Original test, raw predict',    y_orig, p_raw, len(test_df)),
            ('3. Original test, predict_symmetric', y_orig, p_sym, len(test_df)),
        ]

        print(f"\n  {'Method':<42s} {'N':>6s} {'Acc':>7s} {'Brier':>7s} {'LogLoss':>8s} {'AUC':>7s}")
        print(f"  {'-'*80}")

        for name, y, p, n in methods:
            y_pred = (p >= 0.5).astype(int)
            acc = accuracy_score(y, y_pred)
            brier = brier_score_loss(y, p)
            ll = log_loss(y, p)
            auc = roc_auc_score(y, p)
            print(f"  {name:<42s} {n:>6d} {acc:>7.4f} {brier:>7.4f} {ll:>8.4f} {auc:>7.4f}")

        # ── Deeper analysis: check prediction symmetry ──
        # For each original game, compare raw p vs 1-p_mirror
        mirror_half = test_aug.iloc[len(test_df):]
        dtest_mirror = xgb.DMatrix(
            mirror_half[features], feature_names=features
        )
        p_mirror_raw = model.predict(dtest_mirror)

        asym = np.abs(p_raw - (1.0 - p_mirror_raw))
        print(f"\n  Model asymmetry: |p_raw - (1 - p_mirror)|")
        print(f"    Mean:   {asym.mean():.6f}")
        print(f"    Median: {np.median(asym):.6f}")
        print(f"    Max:    {asym.max():.6f}")
        print(f"    Std:    {asym.std():.6f}")

        # Show what predict_symmetric actually changes
        sym_diff = np.abs(p_raw - p_sym)
        print(f"\n  Effect of predict_symmetric: |p_raw - p_sym|")
        print(f"    Mean:   {sym_diff.mean():.6f}")
        print(f"    Median: {np.median(sym_diff):.6f}")
        print(f"    Max:    {sym_diff.max():.6f}")


if __name__ == '__main__':
    main()
