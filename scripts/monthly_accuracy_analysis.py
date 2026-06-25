"""
Monthly accuracy & feature degradation analysis for the CBB prediction model.

Splits 2026 season games by month and analyzes:
  1. Model accuracy per month
  2. Average values of key features (to detect convergence)
  3. Standard deviation of differential features (to detect loss of discriminating power)
  4. Feature importance contribution per month (SHAP-like analysis via prediction variance)

Goal: Understand why model goes from ~82% accuracy in November to ~65% in March.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.database import get_connection
from src.models.feature_engineering import build_features


def main():
    # ---- 1. Load data and model ----
    print("=" * 80)
    print("MONTHLY ACCURACY & FEATURE DEGRADATION ANALYSIS")
    print("=" * 80)

    conn = get_connection()
    df = build_features(conn)
    conn.close()

    # Load the latest model
    model_dir = PROJECT_ROOT / "data" / "model"
    model_files = sorted(model_dir.glob("xgb_model_*.json"))
    latest_model_path = model_files[-1]
    latest_ts = latest_model_path.stem.replace("xgb_model_", "")

    features_path = model_dir / f"features_{latest_ts}.json"
    with open(features_path) as f:
        features = json.load(f)

    model = xgb.Booster()
    model.load_model(str(latest_model_path))
    print(f"\nLoaded model: {latest_model_path.name}")
    print(f"Features: {len(features)}")

    # ---- 2. Filter to 2026 season ----
    test_df = df[df["season_year"] == 2026].copy()
    test_df["month"] = test_df["date"].dt.month
    test_df["month_name"] = test_df["date"].dt.strftime("%B")

    # CBB season months: Nov=11, Dec=12, Jan=1, Feb=2, Mar=3
    month_order = [11, 12, 1, 2, 3]
    month_labels = {11: "November", 12: "December", 1: "January", 2: "February", 3: "March"}

    # Get predictions
    X_test = test_df[features].copy()
    dtest = xgb.DMatrix(X_test, feature_names=features)
    y_prob = model.predict(dtest)
    test_df["pred_prob"] = y_prob
    test_df["pred_winner"] = (y_prob >= 0.5).astype(int)

    # ================================================================
    # SECTION A: Monthly accuracy breakdown
    # ================================================================
    print("\n" + "=" * 80)
    print("SECTION A: MODEL ACCURACY BY MONTH")
    print("=" * 80)

    header = f"{'Month':<12} {'Games':>6} {'Accuracy':>9} {'Brier':>8} {'LogLoss':>8} {'AUC':>7} {'HomeWin%':>9} {'PredHW%':>9} {'AvgConf':>9}"
    print(header)
    print("-" * len(header))

    for m in month_order:
        mask = test_df["month"] == m
        if mask.sum() == 0:
            continue
        chunk = test_df[mask]
        y_true = chunk["home_winner"].values
        y_pred = chunk["pred_winner"].values
        y_p = chunk["pred_prob"].values

        acc = accuracy_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_p)
        ll = log_loss(y_true, y_p)
        try:
            auc = roc_auc_score(y_true, y_p)
        except ValueError:
            auc = float("nan")
        hw_actual = y_true.mean()
        hw_pred = y_p.mean()
        avg_conf = np.mean(np.abs(y_p - 0.5))  # average distance from 50/50

        print(
            f"{month_labels[m]:<12} {len(chunk):>6} {acc:>9.1%} {brier:>8.4f} {ll:>8.4f} {auc:>7.3f} {hw_actual:>9.1%} {hw_pred:>9.1%} {avg_conf:>9.3f}"
        )

    # Overall
    y_true_all = test_df["home_winner"].values
    y_pred_all = test_df["pred_winner"].values
    y_p_all = test_df["pred_prob"].values
    print("-" * len(header))
    print(
        f"{'OVERALL':<12} {len(test_df):>6} {accuracy_score(y_true_all, y_pred_all):>9.1%} "
        f"{brier_score_loss(y_true_all, y_p_all):>8.4f} {log_loss(y_true_all, y_p_all):>8.4f} "
        f"{roc_auc_score(y_true_all, y_p_all):>7.3f} {y_true_all.mean():>9.1%} "
        f"{y_p_all.mean():>9.1%} {np.mean(np.abs(y_p_all - 0.5)):>9.3f}"
    )

    # ================================================================
    # SECTION B: Confidence distribution by month
    # ================================================================
    print("\n" + "=" * 80)
    print("SECTION B: PREDICTION CONFIDENCE DISTRIBUTION BY MONTH")
    print("=" * 80)
    print("(Fraction of predictions in each confidence bucket)")

    buckets = [(0.0, 0.55, "50-55%"), (0.55, 0.65, "55-65%"), (0.65, 0.75, "65-75%"),
               (0.75, 0.85, "75-85%"), (0.85, 1.01, "85%+")]
    header2 = f"{'Month':<12}" + "".join(f"{b[2]:>10}" for b in buckets)
    print(header2)
    print("-" * len(header2))

    for m in month_order:
        mask = test_df["month"] == m
        if mask.sum() == 0:
            continue
        probs = test_df.loc[mask, "pred_prob"].values
        conf = np.maximum(probs, 1 - probs)  # confidence = max(p, 1-p)
        n = len(probs)
        parts = []
        for lo, hi, _ in buckets:
            frac = np.sum((conf >= lo) & (conf < hi)) / n
            parts.append(f"{frac:>10.1%}")
        print(f"{month_labels[m]:<12}" + "".join(parts))

    # ================================================================
    # SECTION C: Feature convergence -- mean values by month
    # ================================================================
    print("\n" + "=" * 80)
    print("SECTION C: KEY FEATURE MEAN VALUES BY MONTH")
    print("=" * 80)
    print("(Shows how running features converge as teams play more games)")

    key_features_means = [
        "team_a_run_games", "team_b_run_games",
        "season_progress",
        "team_a_run_win_pct", "team_b_run_win_pct",
        "team_a_run_avg_margin", "team_b_run_avg_margin",
        "team_a_sos", "team_b_sos",
        "neutral_site", "conference_game",
        "travel_advantage",
    ]

    header3 = f"{'Feature':<30}" + "".join(f"{month_labels[m]:>12}" for m in month_order)
    print(header3)
    print("-" * len(header3))

    for feat in key_features_means:
        if feat not in test_df.columns:
            continue
        parts = []
        for m in month_order:
            mask = test_df["month"] == m
            if mask.sum() == 0:
                parts.append(f"{'N/A':>12}")
            else:
                val = test_df.loc[mask, feat].mean()
                if abs(val) >= 100:
                    parts.append(f"{val:>12.1f}")
                else:
                    parts.append(f"{val:>12.3f}")
        print(f"{feat:<30}" + "".join(parts))

    # ================================================================
    # SECTION D: Differential feature SPREAD (std dev) by month
    # ================================================================
    print("\n" + "=" * 80)
    print("SECTION D: DIFFERENTIAL FEATURE SPREAD (STD DEV) BY MONTH")
    print("=" * 80)
    print("(Lower std dev = less variation = features can't distinguish teams = less predictive)")

    diff_features = [c for c in features if c.startswith("diff_")]
    diff_features_extra = [
        "team_a_run_win_pct", "team_b_run_win_pct",
        "team_a_run_avg_margin", "team_b_run_avg_margin",
        "team_a_prior_barthag", "team_b_prior_barthag",
        "travel_advantage",
    ]

    header4 = f"{'Feature':<35}" + "".join(f"{month_labels[m]:>12}" for m in month_order) + f"{'Nov/Mar':>10}"
    print(header4)
    print("-" * len(header4))

    for feat in diff_features + diff_features_extra:
        if feat not in test_df.columns:
            continue
        parts = []
        stds = {}
        for m in month_order:
            mask = test_df["month"] == m
            if mask.sum() == 0:
                parts.append(f"{'N/A':>12}")
            else:
                val = test_df.loc[mask, feat].std()
                stds[m] = val
                parts.append(f"{val:>12.4f}")
        # Compute ratio of Nov to Mar spread
        if 11 in stds and 3 in stds and stds[3] > 0:
            ratio = stds[11] / stds[3]
            parts.append(f"{ratio:>10.2f}x")
        else:
            parts.append(f"{'N/A':>10}")
        print(f"{feat:<35}" + "".join(parts))

    # ================================================================
    # SECTION E: Game composition analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("SECTION E: GAME COMPOSITION BY MONTH")
    print("=" * 80)

    header5 = f"{'Metric':<35}" + "".join(f"{month_labels[m]:>12}" for m in month_order)
    print(header5)
    print("-" * len(header5))

    # Neutral site games
    for metric_name, col in [
        ("Neutral site %", "neutral_site"),
        ("Conference game %", "conference_game"),
        ("Home ranked %", "team_a_ranked"),
        ("Away ranked %", "team_b_ranked"),
        ("Actual home win %", "home_winner"),
    ]:
        parts = []
        for m in month_order:
            mask = test_df["month"] == m
            if mask.sum() == 0:
                parts.append(f"{'N/A':>12}")
            else:
                val = test_df.loc[mask, col].mean()
                parts.append(f"{val:>12.1%}")
        print(f"{metric_name:<35}" + "".join(parts))

    # Average score margin
    parts = []
    for m in month_order:
        mask = test_df["month"] == m
        if mask.sum() == 0:
            parts.append(f"{'N/A':>12}")
        else:
            val = test_df.loc[mask, "score_margin"].abs().mean()
            parts.append(f"{val:>12.1f}")
    print(f"{'Avg absolute score margin':<35}" + "".join(parts))

    # Score margin std
    parts = []
    for m in month_order:
        mask = test_df["month"] == m
        if mask.sum() == 0:
            parts.append(f"{'N/A':>12}")
        else:
            val = test_df.loc[mask, "score_margin"].std()
            parts.append(f"{val:>12.1f}")
    print(f"{'Score margin std dev':<35}" + "".join(parts))

    # Close games (decided by <= 5 points)
    parts = []
    for m in month_order:
        mask = test_df["month"] == m
        if mask.sum() == 0:
            parts.append(f"{'N/A':>12}")
        else:
            close = (test_df.loc[mask, "score_margin"].abs() <= 5).mean()
            parts.append(f"{close:>12.1%}")
    print(f"{'Close games (<=5 pts) %':<35}" + "".join(parts))

    # ================================================================
    # SECTION F: Bayesian blend weight analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("SECTION F: BAYESIAN BLEND WEIGHT (PRIOR vs CURRENT) BY MONTH")
    print("=" * 80)
    print("(pw = weight on prior-season data; 1-pw = weight on current-season performance)")

    header6 = f"{'Metric':<35}" + "".join(f"{month_labels[m]:>12}" for m in month_order)
    print(header6)
    print("-" * len(header6))

    for side, label in [("team_a", "home"), ("team_b", "away")]:
        parts = []
        for m in month_order:
            mask = test_df["month"] == m
            if mask.sum() == 0:
                parts.append(f"{'N/A':>12}")
            else:
                games_played = test_df.loc[mask, f"{side}_run_games"].mean()
                pw = 1 / (1 + games_played / 5)
                parts.append(f"{pw:>12.3f}")
        row_label = f"{label}_prior_weight (pw)"
        print(f"{row_label:<35}" + "".join(parts))

        parts = []
        for m in month_order:
            mask = test_df["month"] == m
            if mask.sum() == 0:
                parts.append(f"{'N/A':>12}")
            else:
                games_played = test_df.loc[mask, f"{side}_run_games"].mean()
                parts.append(f"{games_played:>12.1f}")
        row_label = f"{label}_avg_games_played"
        print(f"{row_label:<35}" + "".join(parts))

    # ================================================================
    # SECTION G: Accuracy by prediction confidence
    # ================================================================
    print("\n" + "=" * 80)
    print("SECTION G: ACCURACY BY PREDICTION CONFIDENCE, BY MONTH")
    print("=" * 80)
    print("(Shows whether high-confidence picks are still accurate later in season)")

    conf_buckets = [(0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"), (0.8, 1.01, "80%+")]

    for lo, hi, label in conf_buckets:
        print(f"\n  Confidence bucket: {label}")
        header7 = f"    {'Month':<12} {'Games':>6} {'Accuracy':>9}"
        print(header7)
        for m in month_order:
            sub = test_df[test_df["month"] == m].copy()
            if len(sub) == 0:
                continue
            sub_conf = np.maximum(sub["pred_prob"].values, 1 - sub["pred_prob"].values)
            sub2 = sub[(sub_conf >= lo) & (sub_conf < hi)]
            if len(sub2) == 0:
                continue
            acc = accuracy_score(sub2["home_winner"].values, sub2["pred_winner"].values)
            print(f"    {month_labels[m]:<12} {len(sub2):>6} {acc:>9.1%}")

    # ================================================================
    # SECTION H: Which specific features lose discriminating power?
    # ================================================================
    print("\n" + "=" * 80)
    print("SECTION H: FEATURE DISCRIMINATING POWER (AUC PER FEATURE PER MONTH)")
    print("=" * 80)
    print("(Single-feature AUC: how well each feature alone separates winners from losers)")
    print("(Features that drop significantly from Nov to Mar are losing predictive power)")

    # Select the most important differential features + key raw features
    analyze_features = [
        "diff_roster_bpm", "diff_top5_bpm",
        "diff_run_win_pct", "diff_run_avg_margin",
        "diff_run_recent_win_pct", "diff_run_recent_avg_margin",
        "diff_prior_rank", "diff_prior_win_pct",
        "travel_advantage",
        "diff_sos", "diff_conf_win_pct",
        "neutral_site", "conference_game",
        "season_progress",
    ]

    header8 = f"{'Feature':<35}" + "".join(f"{month_labels[m]:>10}" for m in month_order) + f"{'Drop':>10}"
    print(header8)
    print("-" * len(header8))

    for feat in analyze_features:
        if feat not in test_df.columns:
            continue
        parts = []
        aucs = {}
        for m in month_order:
            mask = test_df["month"] == m
            if mask.sum() < 10:
                parts.append(f"{'N/A':>10}")
                continue
            chunk = test_df[mask]
            y = chunk["home_winner"].values
            x = chunk[feat].values

            # Skip if all same value or all NaN
            valid = ~np.isnan(x)
            if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
                parts.append(f"{'N/A':>10}")
                continue
            try:
                a = roc_auc_score(y[valid], x[valid])
                # For features where lower is better for home team, AUC < 0.5
                # Flip to get magnitude of discriminating power
                a_adj = max(a, 1 - a)
                aucs[m] = a_adj
                parts.append(f"{a_adj:>10.3f}")
            except ValueError:
                parts.append(f"{'N/A':>10}")

        # Drop from Nov to Mar
        if 11 in aucs and 3 in aucs:
            drop = aucs[11] - aucs[3]
            color = "***" if drop > 0.05 else ""
            parts.append(f"{drop:>+9.3f}{color}")
        else:
            parts.append(f"{'N/A':>10}")

        print(f"{feat:<35}" + "".join(parts))

    print("\n*** = features with >5 percentage point AUC drop from November to March")

    # ================================================================
    # SECTION I: Summary of findings
    # ================================================================
    print("\n" + "=" * 80)
    print("SECTION I: SUMMARY -- KEY DRIVERS OF ACCURACY DECLINE")
    print("=" * 80)

    # Compute some summary stats
    nov_mask = test_df["month"] == 11
    mar_mask = test_df["month"] == 3

    if nov_mask.sum() > 0 and mar_mask.sum() > 0:
        nov = test_df[nov_mask]
        mar = test_df[mar_mask]

        print(f"\n1. GAME DIFFICULTY")
        print(f"   Nov avg |score margin|: {nov['score_margin'].abs().mean():.1f} pts")
        print(f"   Mar avg |score margin|: {mar['score_margin'].abs().mean():.1f} pts")
        print(f"   Nov close games (<=5): {(nov['score_margin'].abs() <= 5).mean():.1%}")
        print(f"   Mar close games (<=5): {(mar['score_margin'].abs() <= 5).mean():.1%}")

        print(f"\n2. PRIOR-SEASON SIGNAL DECAY")
        nov_pw_h = 1 / (1 + nov["team_a_run_games"].mean() / 15)
        mar_pw_h = 1 / (1 + mar["team_a_run_games"].mean() / 15)
        print(f"   Nov prior weight: {nov_pw_h:.3f} (prior still dominant)")
        print(f"   Mar prior weight: {mar_pw_h:.3f} (current-season dominant)")
        print(f"   In November, the model relies heavily on last year's data (BPM, barthag, rank).")
        print(f"   These are stable, well-separated features. By March, running stats converge")
        print(f"   toward the mean as sample sizes grow and teams face tougher schedules.")

        print(f"\n3. SCHEDULE CONTEXT SHIFT")
        print(f"   Nov neutral site: {nov['neutral_site'].mean():.1%}")
        print(f"   Mar neutral site: {mar['neutral_site'].mean():.1%}")
        print(f"   Nov conference: {nov['conference_game'].mean():.1%}")
        print(f"   Mar conference: {mar['conference_game'].mean():.1%}")
        print(f"   Early season has more neutral-site showcases and mismatches.")
        print(f"   March is mostly conference play where teams are more evenly matched.")

        print(f"\n4. FEATURE SPREAD COMPRESSION")
        for feat in ["diff_roster_bpm", "diff_run_avg_margin",
                      "travel_advantage"]:
            if feat in test_df.columns:
                nov_std = nov[feat].std()
                mar_std = mar[feat].std()
                print(f"   {feat}: Nov std={nov_std:.4f}, Mar std={mar_std:.4f} "
                      f"(ratio={nov_std/mar_std:.2f}x)" if mar_std > 0 else f"   {feat}: N/A")

        print(f"\n5. PREDICTION CONFIDENCE")
        nov_conf = np.mean(np.abs(nov["pred_prob"].values - 0.5))
        mar_conf = np.mean(np.abs(mar["pred_prob"].values - 0.5))
        print(f"   Nov avg confidence (|p-0.5|): {nov_conf:.3f}")
        print(f"   Mar avg confidence (|p-0.5|): {mar_conf:.3f}")

        nov_high_conf = (np.maximum(nov["pred_prob"].values, 1 - nov["pred_prob"].values) >= 0.7).mean()
        mar_high_conf = (np.maximum(mar["pred_prob"].values, 1 - mar["pred_prob"].values) >= 0.7).mean()
        print(f"   Nov high-confidence picks (>70%): {nov_high_conf:.1%}")
        print(f"   Mar high-confidence picks (>70%): {mar_high_conf:.1%}")


if __name__ == "__main__":
    main()
