"""Export 2026 season predictions to CSV for R visualization."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.models.feature_engineering import build_features
from src.models.train_model import select_features
from src.config import build_trained_model


def main():
    conn = get_connection()
    df = build_features(conn)
    conn.close()

    features = select_features(df)
    seasons = sorted(df['season_year'].unique())

    train_df = df[df['season_year'].isin(seasons[:-1])]
    test_df = df[df['season_year'] == seasons[-1]].copy()

    model = build_trained_model(train_df, features, augment=False,
                               test_df=test_df, early_stopping_rounds=50)

    dtest = xgb.DMatrix(test_df[features], feature_names=features)
    y_prob = model.predict(dtest)

    # Get team names from database
    conn = get_connection()
    teams = pd.read_sql("SELECT espn_id, name FROM teams", conn)
    conn.close()
    team_map = dict(zip(teams['espn_id'], teams['name']))

    # Build output dataframe
    out = pd.DataFrame({
        'game_id': test_df['game_id'].values,
        'date': test_df['date'].values,
        'home_team': [team_map.get(tid, str(tid)) for tid in test_df['home_team_id'].values],
        'away_team': [team_map.get(tid, str(tid)) for tid in test_df['away_team_id'].values],
        'home_score': test_df['home_score'].values if 'home_score' in test_df.columns else np.nan,
        'away_score': test_df['away_score'].values if 'away_score' in test_df.columns else np.nan,
        'home_winner': y_test.astype(int),
        'predicted_prob': y_prob,
        'neutral_site': test_df['neutral_site'].values if 'neutral_site' in test_df.columns else 0,
        'conference_game': test_df['conference_game'].values if 'conference_game' in test_df.columns else 0,
    })

    out_path = Path(__file__).parent.parent / 'data' / 'predictions_2026.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Exported {len(out)} predictions to {out_path}")


if __name__ == '__main__':
    main()
