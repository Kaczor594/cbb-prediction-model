"""
Final Four SHAP Matchup Analysis.

Builds the model the same way simulate_tournament.py does, then computes
SHAP contributions for all 6 possible Final Four matchups. Shows which
features most strongly drive each prediction, and compares how UConn and
Illinois differ when facing the same opponent.

Usage:
    cd ~/claude-projects/cbb-prediction-model
    source .venv/bin/activate
    python scripts/matchup_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.models.feature_engineering import build_features
from src.models.train_model import select_features, augment_data
from src.config import XGB_PARAMS, XGB_NUM_BOOST_ROUND, build_team_snapshots, build_matchup_features, load_team_locations
from scripts.simulate_tournament import adjust_roster_for_injuries, INJURED_PLAYERS

# ── Team definitions ────────────────────────────────────────────────────

TEAMS = {
    356: 'Illinois',
    41:  'UConn',
    12:  'Arizona',
    130: 'Michigan',
}

# All 6 matchups (team_a_id, team_b_id, description)
MATCHUPS = [
    (356, 41,  'Illinois vs UConn (semifinal)'),
    (41,  12,  'UConn vs Arizona'),
    (41,  130, 'UConn vs Michigan'),
    (356, 12,  'Illinois vs Arizona'),
    (356, 130, 'Illinois vs Michigan'),
    (12,  130, 'Arizona vs Michigan (semifinal)'),
]

VENUE_CITY = 'Indianapolis'


def main():
    conn = get_connection()

    # ── 1. Build model (same pipeline as simulate_tournament.py) ────────
    print("=" * 80)
    print("  BUILDING MODEL (same pipeline as simulate_tournament.py)")
    print("=" * 80)

    print("\nBuilding feature matrix...")
    df = build_features(conn)

    print("Building team snapshots...")
    snapshots = build_team_snapshots(df, season=2026)
    print(f"  Built snapshots for {len(snapshots)} teams")

    print("\nAdjusting roster for injuries...")
    snapshots = adjust_roster_for_injuries(snapshots, conn)

    team_locs = load_team_locations(conn)
    print(f"  Loaded locations for {len(team_locs)} teams")

    # Train model
    print("\nTraining model on all regular season data (with augmentation)...")
    features = select_features(df)
    print(f"  Using {len(features)} features")

    train_df = df.dropna(subset=['home_winner'])
    train_aug = augment_data(train_df)
    print(f"  {len(train_df)} original games -> {len(train_aug)} augmented rows")

    dtrain = xgb.DMatrix(train_aug[features], label=train_aug['home_winner'].values,
                          feature_names=features)
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=XGB_NUM_BOOST_ROUND,
                       evals=[(dtrain, 'train')], verbose_eval=0)

    # ── 2. Compute SHAP contributions for each matchup ──────────────────
    print("\n" + "=" * 80)
    print("  SHAP MATCHUP ANALYSIS -- Final Four at Indianapolis")
    print("=" * 80)

    # Store results for comparison analysis
    shap_results = {}  # (team_a_id, team_b_id) -> {feature: shap_value}
    feat_values = {}   # (team_a_id, team_b_id) -> {feature: feature_value}

    for team_a_id, team_b_id, desc in MATCHUPS:
        team_a_name = TEAMS[team_a_id]
        team_b_name = TEAMS[team_b_id]

        # Build matchup features
        feat_df = build_matchup_features(
            team_a_id, team_b_id, snapshots, features,
            neutral=True, venue_city=VENUE_CITY, team_locs=team_locs,
        )

        dmat = xgb.DMatrix(feat_df, feature_names=features)

        # Get prediction
        prob = float(model.predict(dmat)[0])

        # Get SHAP contributions (pred_contribs=True)
        contribs = model.predict(dmat, pred_contribs=True)
        # contribs shape: (1, n_features + 1), last element is bias
        contrib_values = contribs[0]
        bias = contrib_values[-1]
        feature_contribs = contrib_values[:-1]

        # Store for comparison
        shap_dict = {}
        fval_dict = {}
        for i, feat_name in enumerate(features):
            shap_dict[feat_name] = feature_contribs[i]
            fval_dict[feat_name] = float(feat_df[feat_name].iloc[0])
        shap_results[(team_a_id, team_b_id)] = shap_dict
        feat_values[(team_a_id, team_b_id)] = fval_dict

        # Print results
        print(f"\n{'─' * 80}")
        print(f"  {desc}")
        print(f"  {team_a_name} (team_a) vs {team_b_name} (team_b) @ {VENUE_CITY}")
        print(f"  Predicted P({team_a_name} wins) = {prob:.4f} ({prob:.1%})")
        print(f"  Bias (base rate): {bias:.4f}")
        print(f"{'─' * 80}")

        # Sort by absolute SHAP value
        sorted_feats = sorted(
            range(len(features)),
            key=lambda i: abs(feature_contribs[i]),
            reverse=True,
        )

        print(f"  {'Rank':>4s}  {'Feature':<40s} {'Value':>10s} {'SHAP':>10s}  Direction")
        print(f"  {'─'*4}  {'─'*40} {'─'*10} {'─'*10}  {'─'*20}")

        for rank, idx in enumerate(sorted_feats[:10], 1):
            feat_name = features[idx]
            feat_val = float(feat_df[feat_name].iloc[0])
            shap_val = feature_contribs[idx]
            direction = f"-> {team_a_name}" if shap_val > 0 else f"-> {team_b_name}"
            print(f"  {rank:>4d}  {feat_name:<40s} {feat_val:>10.4f} {shap_val:>+10.4f}  {direction}")

    # ── 3. Comparison Analysis ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  COMPARISON: UConn vs Illinois as team_a (against same opponents)")
    print("=" * 80)

    comparisons = [
        (41, 12, 356, 12, 'Arizona'),    # UConn vs Arizona compared to Illinois vs Arizona
        (41, 130, 356, 130, 'Michigan'),  # UConn vs Michigan compared to Illinois vs Michigan
    ]

    for uconn_a, opp_b, ill_a, opp_b2, opp_name in comparisons:
        uconn_shap = shap_results[(uconn_a, opp_b)]
        ill_shap = shap_results[(ill_a, opp_b2)]
        uconn_fv = feat_values[(uconn_a, opp_b)]
        ill_fv = feat_values[(ill_a, opp_b2)]

        # Compute UConn prediction vs Illinois prediction
        uconn_feat_df = build_matchup_features(
            uconn_a, opp_b, snapshots, features,
            neutral=True, venue_city=VENUE_CITY, team_locs=team_locs,
        )
        ill_feat_df = build_matchup_features(
            ill_a, opp_b2, snapshots, features,
            neutral=True, venue_city=VENUE_CITY, team_locs=team_locs,
        )
        uconn_prob = float(model.predict(xgb.DMatrix(uconn_feat_df, feature_names=features))[0])
        ill_prob = float(model.predict(xgb.DMatrix(ill_feat_df, feature_names=features))[0])

        print(f"\n{'─' * 80}")
        print(f"  UConn vs {opp_name}: P(UConn wins) = {uconn_prob:.4f} ({uconn_prob:.1%})")
        print(f"  Illinois vs {opp_name}: P(Illinois wins) = {ill_prob:.4f} ({ill_prob:.1%})")
        print(f"  Difference (Illinois - UConn): {ill_prob - uconn_prob:+.4f} ({(ill_prob - uconn_prob)*100:+.1f} pp)")
        print(f"{'─' * 80}")

        # Find features where SHAP contributions differ most
        diffs = {}
        for feat in features:
            uconn_s = uconn_shap.get(feat, 0)
            ill_s = ill_shap.get(feat, 0)
            diffs[feat] = {
                'uconn_shap': uconn_s,
                'ill_shap': ill_s,
                'shap_diff': ill_s - uconn_s,  # positive = feature helps Illinois more
                'uconn_val': uconn_fv.get(feat, 0),
                'ill_val': ill_fv.get(feat, 0),
            }

        # Sort by absolute difference in SHAP contribution
        sorted_diffs = sorted(diffs.items(), key=lambda x: abs(x[1]['shap_diff']), reverse=True)

        print(f"\n  Features where Illinois's SHAP contribution differs most from UConn's (vs {opp_name}):")
        print(f"  {'Rank':>4s}  {'Feature':<35s} {'UConn Val':>10s} {'Ill Val':>10s} "
              f"{'UConn SHAP':>11s} {'Ill SHAP':>11s} {'Diff':>10s}  Favors")
        print(f"  {'─'*4}  {'─'*35} {'─'*10} {'─'*10} {'─'*11} {'─'*11} {'─'*10}  {'─'*15}")

        for rank, (feat, d) in enumerate(sorted_diffs[:15], 1):
            favors = "Illinois" if d['shap_diff'] > 0 else "UConn"
            print(f"  {rank:>4d}  {feat:<35s} {d['uconn_val']:>10.4f} {d['ill_val']:>10.4f} "
                  f"{d['uconn_shap']:>+11.4f} {d['ill_shap']:>+11.4f} {d['shap_diff']:>+10.4f}  {favors}")

    # ── 4. Summary: team_a features across all matchups ─────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY: Key team_a features for each Final Four team")
    print("=" * 80)

    for tid, tname in TEAMS.items():
        # Find matchups where this team is team_a
        team_a_matchups = [(a, b, desc) for a, b, desc in MATCHUPS if a == tid]
        if not team_a_matchups:
            continue

        print(f"\n  {tname} as team_a:")
        for a_id, b_id, desc in team_a_matchups:
            opp_name = TEAMS[b_id]
            shap = shap_results[(a_id, b_id)]
            fv = feat_values[(a_id, b_id)]

            # Get prediction
            feat_df = build_matchup_features(
                a_id, b_id, snapshots, features,
                neutral=True, venue_city=VENUE_CITY, team_locs=team_locs,
            )
            prob = float(model.predict(xgb.DMatrix(feat_df, feature_names=features))[0])

            # Top 5 SHAP features
            top5 = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            top5_str = ", ".join(
                f"{name}({val:+.3f})" for name, val in top5
            )
            print(f"    vs {opp_name:10s}: P(win)={prob:.1%}  Top SHAP: {top5_str}")

    conn.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
