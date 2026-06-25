"""
Export model predictions and odds to CSV for R visualization.

Trains on 2024+2025, predicts all 2026 games, and writes:
  - data/predictions_2026.csv
  - data/game_odds_2026.csv

Usage:
    python scripts/export_predictions.py
"""

import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.models.feature_engineering import build_features
from src.models.train_model import select_features
from src.config import build_trained_model


def main():
    conn = get_connection()

    print("Building features...")
    df = build_features(conn)

    # Load team names
    teams = pd.read_sql_query(
        "SELECT espn_id, name FROM teams", conn
    )
    team_map = dict(zip(teams['espn_id'], teams['name']))

    # Load odds
    odds_df = pd.read_sql_query("""
        SELECT game_id, home_implied_prob, away_implied_prob
        FROM game_odds
    """, conn)
    conn.close()

    features = select_features(df)
    print(f"Using {len(features)} features")

    # Train on prior seasons, predict 2026
    seasons = sorted(df['season_year'].unique())
    train_df = df[df['season_year'].isin(seasons[:-1])]
    test_df = df[df['season_year'] == seasons[-1]].copy()

    print(f"Train: {len(train_df)} games ({seasons[:-1]})")
    print(f"Test:  {len(test_df)} games ({seasons[-1]})")

    model = build_trained_model(train_df, features, augment=False,
                               test_df=test_df, early_stopping_rounds=50)

    dtest = xgb.DMatrix(test_df[features], feature_names=features)
    test_df['predicted_prob'] = model.predict(dtest)

    # Build predictions CSV
    preds = test_df[['game_id', 'date', 'home_team_id', 'away_team_id',
                      'predicted_prob', 'home_winner']].copy()
    preds['home_team'] = preds['home_team_id'].map(team_map)
    preds['away_team'] = preds['away_team_id'].map(team_map)
    preds = preds[['game_id', 'date', 'home_team', 'away_team',
                    'predicted_prob', 'home_winner']]

    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)

    preds_path = out_dir / 'predictions_2026.csv'
    preds.to_csv(preds_path, index=False)
    print(f"\nWrote {len(preds)} predictions to {preds_path}")

    # Write odds CSV
    odds_path = out_dir / 'game_odds_2026.csv'
    odds_df.to_csv(odds_path, index=False)
    print(f"Wrote {len(odds_df)} odds rows to {odds_path}")

    # Quick accuracy check
    acc = ((test_df['predicted_prob'] >= 0.5).astype(int) == test_df['home_winner']).mean()
    print(f"\nModel accuracy on 2026: {acc:.4f} ({acc*100:.1f}%)")


if __name__ == '__main__':
    main()
