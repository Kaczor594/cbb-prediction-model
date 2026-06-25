"""
Backtest: compare model predictions vs market odds on 2026 season.

Evaluates:
  1. Model accuracy vs market accuracy on the same games
  2. Whether the model identifies profitable edges over the market
  3. Simulated betting performance (flat-stake, Kelly)
  4. Calibration comparison

Usage:
    python scripts/backtest_vs_market.py                # Single-train mode
    python scripts/backtest_vs_market.py --walk-forward  # Retrain daily
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.models.feature_engineering import build_features
from src.models.train_model import select_features
from src.config import predict_symmetric, build_trained_model


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    """Remove vig from implied probabilities to get fair odds."""
    total = home_prob + away_prob
    return home_prob / total, away_prob / total


def train_single(df, features):
    """Train once on 2024+2025, predict all of 2026."""
    seasons = sorted(df['season_year'].unique())
    train_df = df[df['season_year'].isin(seasons[:-1])]
    test_df = df[df['season_year'] == seasons[-1]].copy()

    print(f"Train: {len(train_df)} games ({seasons[:-1]})")
    print(f"Test:  {len(test_df)} games ({seasons[-1]})")

    model = build_trained_model(train_df, features, augment=False,
                               test_df=test_df, early_stopping_rounds=50)

    test_df['model_prob'] = predict_symmetric(model, test_df, features)
    test_df['model_pred'] = (test_df['model_prob'] >= 0.5).astype(int)
    return test_df


def train_walk_forward(df, features):
    """Retrain daily: for each game day in 2026, train on all prior data."""
    seasons = sorted(df['season_year'].unique())
    test_season = seasons[-1]
    prior_df = df[df['season_year'].isin(seasons[:-1])]
    season_df = df[df['season_year'] == test_season].copy()

    # Sort by date for chronological walk-forward
    season_df['date_parsed'] = pd.to_datetime(season_df['date'])
    season_df = season_df.sort_values('date_parsed')
    game_dates = season_df['date_parsed'].dt.date.unique()

    print(f"Prior seasons: {len(prior_df)} games ({seasons[:-1]})")
    print(f"Walk-forward:  {len(season_df)} games across {len(game_dates)} dates ({test_season})")

    all_preds = []

    for game_date in tqdm(game_dates, desc="Walk-forward", unit="day"):
        # Games to predict today
        today_mask = season_df['date_parsed'].dt.date == game_date
        today_df = season_df[today_mask]

        # Training set: all prior seasons + 2026 games before today
        past_season_mask = season_df['date_parsed'].dt.date < game_date
        past_season_df = season_df[past_season_mask]
        train_df = pd.concat([prior_df, past_season_df])

        model = build_trained_model(train_df, features, augment=False,
                                   early_stopping_rounds=50)

        preds = today_df.copy()
        preds['model_prob'] = predict_symmetric(model, today_df, features)
        preds['model_pred'] = (preds['model_prob'] >= 0.5).astype(int)
        all_preds.append(preds)

    test_df = pd.concat(all_preds)
    test_df.drop(columns=['date_parsed'], inplace=True)
    return test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--walk-forward', action='store_true',
                        help='Retrain model daily in walk-forward fashion')
    args = parser.parse_args()

    conn = get_connection()

    print("Building features...")
    df = build_features(conn)

    odds_df = pd.read_sql_query("""
        SELECT game_id, home_moneyline, away_moneyline,
               home_implied_prob, away_implied_prob,
               home_spread, over_under, num_sportsbooks
        FROM game_odds
    """, conn)
    conn.close()

    features = select_features(df)
    print(f"\nUsing {len(features)} features")

    if args.walk_forward:
        print("\n=== WALK-FORWARD MODE (retrain daily) ===\n")
        test_df = train_walk_forward(df, features)
    else:
        print("\n=== SINGLE-TRAIN MODE (train once) ===\n")
        test_df = train_single(df, features)

    # Merge with odds
    test_odds = test_df.merge(odds_df, on='game_id', how='inner')
    print(f"\nGames with both model predictions and market odds: {len(test_odds)}")

    # Remove vig from market probabilities
    fair_probs = test_odds.apply(
        lambda r: remove_vig(r['home_implied_prob'], r['away_implied_prob']),
        axis=1
    )
    test_odds['market_prob'] = [p[0] for p in fair_probs]
    test_odds['market_pred'] = (test_odds['market_prob'] >= 0.5).astype(int)

    # ============================================================
    # 1. HEAD-TO-HEAD ACCURACY
    # ============================================================
    print(f"\n{'='*65}")
    print("HEAD-TO-HEAD: MODEL vs MARKET")
    print(f"{'='*65}")

    model_acc = (test_odds['model_pred'] == test_odds['home_winner']).mean()
    market_acc = (test_odds['market_pred'] == test_odds['home_winner']).mean()

    print(f"  Model accuracy:  {model_acc:.4f} ({model_acc*100:.1f}%)")
    print(f"  Market accuracy: {market_acc:.4f} ({market_acc*100:.1f}%)")
    print(f"  Difference:      {(model_acc - market_acc)*100:+.1f} pp")

    # Full test set accuracy (including games without odds)
    full_acc = (test_df['model_pred'] == test_df['home_winner']).mean()
    print(f"\n  Model accuracy (full test set): {full_acc:.4f} ({full_acc*100:.1f}%)")

    # ============================================================
    # 2. AGREEMENT ANALYSIS
    # ============================================================
    print(f"\n{'='*65}")
    print("AGREEMENT ANALYSIS")
    print(f"{'='*65}")

    agree = test_odds['model_pred'] == test_odds['market_pred']
    disagree = ~agree

    print(f"  Agree on winner:    {agree.sum():5d} ({agree.mean()*100:.1f}%)")
    print(f"  Disagree on winner: {disagree.sum():5d} ({disagree.mean()*100:.1f}%)")

    if disagree.sum() > 0:
        disagree_games = test_odds[disagree]
        model_right = (disagree_games['model_pred'] == disagree_games['home_winner']).sum()
        market_right = (disagree_games['market_pred'] == disagree_games['home_winner']).sum()
        print(f"\n  When they disagree:")
        print(f"    Model correct:  {model_right}/{disagree.sum()} ({model_right/disagree.sum()*100:.1f}%)")
        print(f"    Market correct: {market_right}/{disagree.sum()} ({market_right/disagree.sum()*100:.1f}%)")

    # ============================================================
    # 3. EDGE ANALYSIS
    # ============================================================
    print(f"\n{'='*65}")
    print("EDGE ANALYSIS (Model prob vs Market prob)")
    print(f"{'='*65}")

    test_odds['edge'] = test_odds['model_prob'] - test_odds['market_prob']

    # Bin by edge magnitude
    bins = [(-1, -0.10), (-0.10, -0.05), (-0.05, -0.02), (-0.02, 0.02),
            (0.02, 0.05), (0.05, 0.10), (0.10, 1)]
    labels = ['< -10%', '-10 to -5%', '-5 to -2%', '-2 to +2%',
              '+2 to +5%', '+5 to +10%', '> +10%']

    print(f"\n  {'Edge Range':>14s} {'N':>6s} {'Model Acc':>10s} {'Market Acc':>11s} {'Home Win%':>10s}")
    print(f"  {'-'*55}")

    for (lo, hi), label in zip(bins, labels):
        mask = (test_odds['edge'] >= lo) & (test_odds['edge'] < hi)
        if mask.sum() == 0:
            continue
        subset = test_odds[mask]
        m_acc = (subset['model_pred'] == subset['home_winner']).mean()
        mk_acc = (subset['market_pred'] == subset['home_winner']).mean()
        hw = subset['home_winner'].mean()
        print(f"  {label:>14s} {mask.sum():>6d} {m_acc:>10.3f} {mk_acc:>11.3f} {hw:>10.3f}")

    # ============================================================
    # 4. SIMULATED BETTING (flat stake on model edge)
    # ============================================================
    print(f"\n{'='*65}")
    print("SIMULATED BETTING (flat $100 stakes)")
    print(f"{'='*65}")

    def ml_payout(ml: float, stake: float = 100.0) -> float:
        """Calculate profit on a winning bet at the given moneyline."""
        if ml > 0:
            return stake * ml / 100
        else:
            return stake * 100 / abs(ml)

    for min_edge in [0.02, 0.05, 0.10]:
        # Bet on home when model_prob - market_prob > min_edge
        home_bets = test_odds[test_odds['edge'] > min_edge].copy()
        # Bet on away when market_prob - model_prob > min_edge
        away_bets = test_odds[test_odds['edge'] < -min_edge].copy()

        total_bets = len(home_bets) + len(away_bets)
        if total_bets == 0:
            print(f"\n  Edge > {min_edge*100:.0f}%: No bets")
            continue

        # P&L using actual moneylines
        # Home bets: win if home_winner == 1, paid at home_moneyline
        home_pl = home_bets.apply(
            lambda r: ml_payout(r['home_moneyline']) if r['home_winner'] == 1 else -100.0,
            axis=1
        )
        # Away bets: win if home_winner == 0, paid at away_moneyline
        away_pl = away_bets.apply(
            lambda r: ml_payout(r['away_moneyline']) if r['home_winner'] == 0 else -100.0,
            axis=1
        )

        home_wins = (home_bets['home_winner'] == 1).sum()
        away_wins = (away_bets['home_winner'] == 0).sum()
        wins = home_wins + away_wins
        win_rate = wins / total_bets

        profit = home_pl.sum() + away_pl.sum()
        total_risked = total_bets * 100
        roi = profit / total_risked * 100

        avg_winner_ml = pd.concat([
            home_bets.loc[home_bets['home_winner'] == 1, 'home_moneyline'],
            away_bets.loc[away_bets['home_winner'] == 0, 'away_moneyline'],
        ])
        avg_loser_ml = pd.concat([
            home_bets.loc[home_bets['home_winner'] == 0, 'home_moneyline'],
            away_bets.loc[away_bets['home_winner'] == 1, 'away_moneyline'],
        ])

        print(f"\n  Edge > {min_edge*100:.0f}%:")
        print(f"    Bets:     {total_bets} ({len(home_bets)} home, {len(away_bets)} away)")
        print(f"    Win rate: {wins}/{total_bets} ({win_rate*100:.1f}%)")
        print(f"    P&L:      ${profit:+,.0f}")
        print(f"    ROI:      {roi:+.1f}%")
        if len(avg_winner_ml) > 0:
            print(f"    Avg ML (winners): {avg_winner_ml.mean():+.0f}")
        if len(avg_loser_ml) > 0:
            print(f"    Avg ML (losers):  {avg_loser_ml.mean():+.0f}")

    # ============================================================
    # 5. PROBABILITY CORRELATION
    # ============================================================
    print(f"\n{'='*65}")
    print("PROBABILITY ANALYSIS")
    print(f"{'='*65}")

    corr = test_odds['model_prob'].corr(test_odds['market_prob'])
    print(f"  Correlation (model prob vs market prob): {corr:.4f}")

    mae_model = np.abs(test_odds['model_prob'] - test_odds['home_winner']).mean()
    mae_market = np.abs(test_odds['market_prob'] - test_odds['home_winner']).mean()
    print(f"  MAE (model):  {mae_model:.4f}")
    print(f"  MAE (market): {mae_market:.4f}")

    from sklearn.metrics import brier_score_loss, log_loss
    brier_model = brier_score_loss(test_odds['home_winner'], test_odds['model_prob'])
    brier_market = brier_score_loss(test_odds['home_winner'], test_odds['market_prob'])
    print(f"  Brier (model):  {brier_model:.4f}")
    print(f"  Brier (market): {brier_market:.4f}")

    ll_model = log_loss(test_odds['home_winner'], test_odds['model_prob'])
    ll_market = log_loss(test_odds['home_winner'], test_odds['market_prob'])
    print(f"  Log loss (model):  {ll_model:.4f}")
    print(f"  Log loss (market): {ll_market:.4f}")

    # ============================================================
    # 6. MONTHLY BREAKDOWN
    # ============================================================
    print(f"\n{'='*65}")
    print("MONTHLY BREAKDOWN")
    print(f"{'='*65}")

    test_odds['month'] = pd.to_datetime(test_odds['date']).dt.strftime('%b')
    month_order = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']

    print(f"\n  {'Month':>5s} {'N':>6s} {'Model Acc':>10s} {'Market Acc':>11s} {'Agree%':>8s}")
    print(f"  {'-'*45}")

    for month in month_order:
        mask = test_odds['month'] == month
        if mask.sum() == 0:
            continue
        subset = test_odds[mask]
        m_acc = (subset['model_pred'] == subset['home_winner']).mean()
        mk_acc = (subset['market_pred'] == subset['home_winner']).mean()
        agree_pct = (subset['model_pred'] == subset['market_pred']).mean()
        print(f"  {month:>5s} {mask.sum():>6d} {m_acc:>10.3f} {mk_acc:>11.3f} {agree_pct:>8.1%}")


if __name__ == '__main__':
    main()
