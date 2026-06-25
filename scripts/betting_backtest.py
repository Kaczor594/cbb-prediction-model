"""
Walk-forward betting backtest: 10 strategies across 2025+2026 seasons.

Retrains XGBoost daily, simulates betting with independent purses,
and exports daily P/L in long format for R ggplot visualization.

Usage:
    python scripts/betting_backtest.py                # Full run + save cache
    python scripts/betting_backtest.py --save-cache   # Full run + save predictions
    python scripts/betting_backtest.py --load-cache   # Skip retraining, run strategies only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.models.feature_engineering import build_features
from src.models.train_model import select_features
from src.config import predict_symmetric, build_trained_model

CACHE_PATH = Path('data/backtest_predictions_cache.csv')
OUTPUT_CSV = Path('data/backtest_daily_pl_long.csv')
INITIAL_PURSE = 100.0
MIN_TRAINING_GAMES = 100

STRATEGY_NAMES = [
    'flat_model_winner',
    'flat_edge_no_vig',
    'flat_edge_with_vig',
    'confidence_sizing',
    'edge_sizing_no_vig',
    'edge_sizing_with_vig',
    'flat_edge_found_no_vig',
    'flat_edge_found_with_vig',
    'full_kelly',
    'half_kelly',
]


# ── Odds helpers ─────────────────────────────────────────────────────────────

def ml_profit(ml: float, stake: float) -> float:
    """Profit on a winning bet at American moneyline odds."""
    if ml > 0:
        return stake * ml / 100
    else:
        return stake * 100 / abs(ml)


def implied_prob_from_ml(ml: float) -> float:
    """Convert American moneyline to implied probability (vig included)."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def remove_vig(home_implied: float, away_implied: float) -> tuple[float, float]:
    """Remove vig from implied probabilities to get fair odds."""
    total = home_implied + away_implied
    return home_implied / total, away_implied / total


def kelly_fraction(prob: float, ml: float) -> float:
    """Compute Kelly fraction for a bet at American moneyline odds.

    b = decimal payout ratio: ML > 0 → b = ML/100; ML < 0 → b = 100/|ML|
    f* = (p * (b+1) - 1) / b
    """
    if ml > 0:
        b = ml / 100
    else:
        b = 100 / abs(ml)
    f = (prob * (b + 1) - 1) / b
    return max(f, 0.0)


# ── Walk-forward prediction engine ──────────────────────────────────────────

def walk_forward_predictions(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Walk-forward retraining across 2025 and 2026 seasons.

    2025: train on 2024, add 2025 games day by day.
    2026: train on 2024+2025, add 2026 games day by day.

    Returns DataFrame with game_id, model_prob, date, season_year for every
    game that was predicted (skipping dates with < MIN_TRAINING_GAMES).
    """
    df = df.copy()
    df['date_parsed'] = pd.to_datetime(df['date'])

    skipped_dates = []
    all_preds = []

    for test_season in [2025, 2026]:
        if test_season == 2025:
            prior_seasons = [2024]
        else:
            prior_seasons = [2024, 2025]

        prior_df = df[df['season_year'].isin(prior_seasons)]
        season_df = df[df['season_year'] == test_season].sort_values('date_parsed')
        game_dates = sorted(season_df['date_parsed'].dt.date.unique())

        print(f"\n{'='*60}")
        print(f"Season {test_season}: {len(season_df)} games across {len(game_dates)} dates")
        print(f"  Prior seasons: {prior_seasons} ({len(prior_df)} games)")
        print(f"{'='*60}")

        for game_date in tqdm(game_dates, desc=f"Walk-forward {test_season}", unit="day"):
            today_mask = season_df['date_parsed'].dt.date == game_date
            today_df = season_df[today_mask]

            past_mask = season_df['date_parsed'].dt.date < game_date
            past_df = season_df[past_mask]
            train_df = pd.concat([prior_df, past_df])

            if len(train_df) < MIN_TRAINING_GAMES:
                skipped_dates.append((test_season, str(game_date), len(train_df)))
                continue

            model = build_trained_model(train_df, features, augment=True,
                                       early_stopping_rounds=50)

            preds = today_df[['game_id', 'date', 'season_year', 'home_winner']].copy()
            preds['model_prob'] = predict_symmetric(model, today_df, features)
            preds['date_str'] = preds['date'].dt.strftime('%Y-%m-%d')
            all_preds.append(preds)

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df = pred_df[['game_id', 'model_prob', 'date_str', 'season_year', 'home_winner']]
    pred_df.rename(columns={'date_str': 'date'}, inplace=True)

    return pred_df, skipped_dates


# ── Strategy engine ──────────────────────────────────────────────────────────

def run_strategies(pred_df: pd.DataFrame, odds_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run all 10 strategies on the predicted games with odds.

    Returns:
        daily_records: long-format DataFrame for CSV export
        strategy_summaries: dict of strategy -> summary stats
    """
    # Merge predictions with odds
    merged = pred_df.merge(odds_df, on='game_id', how='inner')
    print(f"\nGames with both predictions and odds: {len(merged)}")

    # Compute derived probability columns
    merged['home_implied'] = merged['home_moneyline'].apply(implied_prob_from_ml)
    merged['away_implied'] = merged['away_moneyline'].apply(implied_prob_from_ml)
    fair = merged.apply(
        lambda r: remove_vig(r['home_implied'], r['away_implied']), axis=1
    )
    merged['home_fair'] = [p[0] for p in fair]
    merged['away_fair'] = [p[1] for p in fair]
    merged['away_model_prob'] = 1.0 - merged['model_prob']

    # Sort by date for chronological processing
    merged['date_parsed'] = pd.to_datetime(merged['date'])
    merged = merged.sort_values('date_parsed')
    unique_dates = merged['date_parsed'].dt.date.unique()

    # Initialize purses and tracking
    purses = {s: INITIAL_PURSE for s in STRATEGY_NAMES}
    cum_pl = {s: 0.0 for s in STRATEGY_NAMES}
    peak_purse = {s: INITIAL_PURSE for s in STRATEGY_NAMES}
    max_drawdown = {s: 0.0 for s in STRATEGY_NAMES}
    total_wagered = {s: 0.0 for s in STRATEGY_NAMES}
    total_bets = {s: 0 for s in STRATEGY_NAMES}
    total_wins = {s: 0 for s in STRATEGY_NAMES}
    topup_log = {s: [] for s in STRATEGY_NAMES}
    biggest_win = {s: ('', 0.0) for s in STRATEGY_NAMES}
    biggest_loss = {s: ('', 0.0) for s in STRATEGY_NAMES}

    daily_records = []

    for game_date in unique_dates:
        date_mask = merged['date_parsed'].dt.date == game_date
        day_games = merged[date_mask]
        season = day_games['season_year'].iloc[0]
        date_str = str(game_date)

        # Track daily stats per strategy
        day_bets = {s: 0 for s in STRATEGY_NAMES}
        day_wins = {s: 0 for s in STRATEGY_NAMES}
        day_pl = {s: 0.0 for s in STRATEGY_NAMES}

        for _, game in day_games.iterrows():
            model_prob = game['model_prob']
            away_model = game['away_model_prob']
            home_ml = game['home_moneyline']
            away_ml = game['away_moneyline']
            home_won = game['home_winner'] == 1
            home_implied = game['home_implied']
            away_implied = game['away_implied']
            home_fair = game['home_fair']
            away_fair = game['away_fair']

            # Determine bets for each strategy
            bets = _compute_bets(
                model_prob, away_model,
                home_ml, away_ml,
                home_implied, away_implied,
                home_fair, away_fair,
                purses,
            )

            for strat, (side, stake) in bets.items():
                if stake <= 0:
                    continue

                # Top up if needed
                while purses[strat] < stake:
                    purses[strat] += INITIAL_PURSE
                    cum_pl[strat] -= INITIAL_PURSE
                    topup_log[strat].append(date_str)

                # Place bet
                purses[strat] -= stake
                total_wagered[strat] += stake
                day_bets[strat] += 1
                total_bets[strat] += 1

                # Resolve
                bet_won = (side == 'home' and home_won) or (side == 'away' and not home_won)
                ml = home_ml if side == 'home' else away_ml

                if bet_won:
                    profit = ml_profit(ml, stake)
                    purses[strat] += stake + profit
                    day_pl[strat] += profit
                    cum_pl[strat] += profit
                    day_wins[strat] += 1
                    total_wins[strat] += 1
                    if profit > biggest_win[strat][1]:
                        biggest_win[strat] = (date_str, profit)
                else:
                    day_pl[strat] -= stake
                    cum_pl[strat] -= stake
                    if stake > biggest_loss[strat][1]:
                        biggest_loss[strat] = (date_str, stake)

        # Record daily stats
        for strat in STRATEGY_NAMES:
            # Update peak and drawdown
            if purses[strat] > peak_purse[strat]:
                peak_purse[strat] = purses[strat]
            dd = peak_purse[strat] - purses[strat]
            if dd > max_drawdown[strat]:
                max_drawdown[strat] = dd

            daily_records.append({
                'date': date_str,
                'season': season,
                'strategy': strat,
                'bets': day_bets[strat],
                'wins': day_wins[strat],
                'daily_pl': round(day_pl[strat], 4),
                'cumulative_pl': round(cum_pl[strat], 4),
                'purse': round(purses[strat], 4),
                'drawdown': round(max_drawdown[strat], 4),
            })

    # Build summary
    summaries = {}
    for strat in STRATEGY_NAMES:
        win_pct = total_wins[strat] / total_bets[strat] * 100 if total_bets[strat] > 0 else 0
        roi = cum_pl[strat] / total_wagered[strat] * 100 if total_wagered[strat] > 0 else 0
        summaries[strat] = {
            'bets': total_bets[strat],
            'wins': total_wins[strat],
            'win_pct': round(win_pct, 2),
            'wagered': round(total_wagered[strat], 2),
            'final_purse': round(purses[strat], 2),
            'pl': round(cum_pl[strat], 2),
            'roi': round(roi, 2),
            'max_drawdown': round(max_drawdown[strat], 2),
            'topups': len(topup_log[strat]),
            'biggest_win': biggest_win[strat],
            'biggest_loss': biggest_loss[strat],
            'topup_dates': topup_log[strat],
        }

    return pd.DataFrame(daily_records), summaries


def _compute_bets(
    model_prob: float, away_model: float,
    home_ml: float, away_ml: float,
    home_implied: float, away_implied: float,
    home_fair: float, away_fair: float,
    purses: dict,
) -> dict:
    """Compute bet side and stake for each strategy on a single game.

    Returns dict of strategy_name -> (side, stake) or (None, 0) for no bet.
    """
    bets = {}

    # ── 1. flat_model_winner: 1 unit on predicted winner ─────────────
    if model_prob >= 0.5:
        bets['flat_model_winner'] = ('home', 1.0)
    else:
        bets['flat_model_winner'] = ('away', 1.0)

    # ── 2. flat_edge_no_vig: 1 unit if model_prob > fair_prob for one side ──
    # After vig removal, one side always has edge (model vs fair can differ)
    if model_prob > home_fair:
        bets['flat_edge_no_vig'] = ('home', 1.0)
    else:
        bets['flat_edge_no_vig'] = ('away', 1.0)

    # ── 3. flat_edge_with_vig: 1 unit if model_prob > implied_prob for one side
    # With vig, both sides can be negative → no bet
    home_edge_vig = model_prob - home_implied
    away_edge_vig = away_model - away_implied
    if home_edge_vig > 0 and home_edge_vig >= away_edge_vig:
        bets['flat_edge_with_vig'] = ('home', 1.0)
    elif away_edge_vig > 0:
        bets['flat_edge_with_vig'] = ('away', 1.0)
    else:
        bets['flat_edge_with_vig'] = (None, 0)

    # ── 4. confidence_sizing: model_prob % of purse on predicted winner ──
    if model_prob >= 0.5:
        bets['confidence_sizing'] = ('home', model_prob * purses['confidence_sizing'])
    else:
        bets['confidence_sizing'] = ('away', away_model * purses['confidence_sizing'])

    # ── 5. edge_sizing_no_vig: edge % of purse if positive edge (vig removed) ──
    home_edge_nv = model_prob - home_fair
    away_edge_nv = away_model - away_fair
    if home_edge_nv > 0 and home_edge_nv >= away_edge_nv:
        bets['edge_sizing_no_vig'] = ('home', home_edge_nv * purses['edge_sizing_no_vig'])
    elif away_edge_nv > 0:
        bets['edge_sizing_no_vig'] = ('away', away_edge_nv * purses['edge_sizing_no_vig'])
    else:
        bets['edge_sizing_no_vig'] = (None, 0)

    # ── 6. edge_sizing_with_vig: edge % of purse if positive edge (vig included)
    if home_edge_vig > 0 and home_edge_vig >= away_edge_vig:
        bets['edge_sizing_with_vig'] = ('home', home_edge_vig * purses['edge_sizing_with_vig'])
    elif away_edge_vig > 0:
        bets['edge_sizing_with_vig'] = ('away', away_edge_vig * purses['edge_sizing_with_vig'])
    else:
        bets['edge_sizing_with_vig'] = (None, 0)

    # ── 7. flat_edge_found_no_vig: 1 unit on side with positive edge (vig removed)
    # Same trigger as #5
    if home_edge_nv > 0 and home_edge_nv >= away_edge_nv:
        bets['flat_edge_found_no_vig'] = ('home', 1.0)
    elif away_edge_nv > 0:
        bets['flat_edge_found_no_vig'] = ('away', 1.0)
    else:
        bets['flat_edge_found_no_vig'] = (None, 0)

    # ── 8. flat_edge_found_with_vig: 1 unit on side with positive edge (vig included)
    # Same trigger as #6
    if home_edge_vig > 0 and home_edge_vig >= away_edge_vig:
        bets['flat_edge_found_with_vig'] = ('home', 1.0)
    elif away_edge_vig > 0:
        bets['flat_edge_found_with_vig'] = ('away', 1.0)
    else:
        bets['flat_edge_found_with_vig'] = (None, 0)

    # ── 9. full_kelly: Kelly fraction of purse, bet when f* > 0 ──────
    kf_home = kelly_fraction(model_prob, home_ml)
    kf_away = kelly_fraction(away_model, away_ml)
    if kf_home > 0 and kf_home >= kf_away:
        bets['full_kelly'] = ('home', kf_home * purses['full_kelly'])
    elif kf_away > 0:
        bets['full_kelly'] = ('away', kf_away * purses['full_kelly'])
    else:
        bets['full_kelly'] = (None, 0)

    # ── 10. half_kelly: half Kelly fraction ───────────────────────────
    hkf_home = kf_home / 2
    hkf_away = kf_away / 2
    if hkf_home > 0 and hkf_home >= hkf_away:
        bets['half_kelly'] = ('home', hkf_home * purses['half_kelly'])
    elif hkf_away > 0:
        bets['half_kelly'] = ('away', hkf_away * purses['half_kelly'])
    else:
        bets['half_kelly'] = (None, 0)

    return bets


# ── Output formatting ────────────────────────────────────────────────────────

def _fmt(value: float, width: int = 12, signed: bool = False) -> str:
    """Format a number: use commas for small values, scientific notation for large ones."""
    abs_val = abs(value)
    if abs_val < 1e9:
        if signed:
            return f"{value:>+{width},.2f}"
        return f"{value:>{width},.2f}"
    else:
        if signed:
            return f"{value:>+{width}.4e}"
        return f"{value:>{width}.4e}"


def print_summary(summaries: dict):
    """Print the summary table for all 10 strategies."""
    print(f"\n{'='*110}")
    print("BETTING BACKTEST SUMMARY (2025 + 2026)")
    print(f"{'='*110}")
    print(f"{'Strategy':<28s} {'Bets':>6s} {'Win%':>7s} {'Wagered':>14s} "
          f"{'Purse':>14s} {'P/L':>14s} {'ROI':>8s} {'MaxDD':>14s} {'TopUps':>7s}")
    print(f"{'-'*110}")

    for strat in sorted(STRATEGY_NAMES, key=lambda s: summaries[s]['roi'], reverse=True):
        s = summaries[strat]
        print(f"{strat:<28s} {s['bets']:>6d} {s['win_pct']:>6.1f}% "
              f"{_fmt(s['wagered'], 14)} {_fmt(s['final_purse'], 14)} "
              f"{_fmt(s['pl'], 14, signed=True)} {s['roi']:>+7.1f}% "
              f"{_fmt(s['max_drawdown'], 14)} {s['topups']:>7d}")


def print_notable_events(summaries: dict):
    """Print notable events for each strategy."""
    print(f"\n{'='*80}")
    print("NOTABLE EVENTS")
    print(f"{'='*80}")

    for strat in STRATEGY_NAMES:
        s = summaries[strat]
        events = []
        if s['topup_dates']:
            unique_dates = sorted(set(s['topup_dates']))
            events.append(f"  Top-ups on: {', '.join(unique_dates[:10])}"
                          f"{'...' if len(unique_dates) > 10 else ''}")
        bw_date, bw_amt = s['biggest_win']
        bl_date, bl_amt = s['biggest_loss']
        if bw_amt > 0:
            events.append(f"  Biggest win:  {_fmt(bw_amt, 14, signed=True)} on {bw_date}")
        if bl_amt > 0:
            events.append(f"  Biggest loss: {_fmt(-bl_amt, 14, signed=True)} on {bl_date}")

        if events:
            print(f"\n  {strat}:")
            for e in events:
                print(e)


def print_bet_count_verification(summaries: dict):
    """Verify expected bet count relationships."""
    print(f"\n{'='*80}")
    print("BET COUNT VERIFICATION")
    print(f"{'='*80}")

    # Strategies 2, 5, 7 should have same bet count (no-vig edge → one side always positive)
    s2 = summaries['flat_edge_no_vig']['bets']
    s5 = summaries['edge_sizing_no_vig']['bets']
    s7 = summaries['flat_edge_found_no_vig']['bets']
    match_nv = "PASS" if s2 == s5 == s7 else "FAIL"
    print(f"  No-vig group (2,5,7): {s2}, {s5}, {s7} → {match_nv}")

    # Strategies 3, 6, 8 should have same bet count (with-vig edge)
    s3 = summaries['flat_edge_with_vig']['bets']
    s6 = summaries['edge_sizing_with_vig']['bets']
    s8 = summaries['flat_edge_found_with_vig']['bets']
    match_wv = "PASS" if s3 == s6 == s8 else "FAIL"
    print(f"  With-vig group (3,6,8): {s3}, {s6}, {s8} → {match_wv}")


def print_bet_trace(pred_df: pd.DataFrame, odds_df: pd.DataFrame, n: int = 10):
    """Print the first N flat_model_winner bets with full detail for verification."""
    merged = pred_df.merge(odds_df, on='game_id', how='inner')
    merged['date_parsed'] = pd.to_datetime(merged['date'])
    merged = merged.sort_values('date_parsed')

    print(f"\n{'='*115}")
    print(f"FLAT_MODEL_WINNER — FIRST {n} BETS (verification trace)")
    print(f"{'='*115}")
    print(f"  {'#':>3s}  {'Date':>10s}  {'GameID':>7s}  {'ModelP':>7s}  {'Side':>5s}  "
          f"{'ML':>6s}  {'Stake':>6s}  {'Won?':>5s}  {'Profit':>8s}  {'CumPL':>10s}")
    print(f"  {'-'*110}")

    cum_pl = 0.0
    count = 0
    for _, game in merged.iterrows():
        if count >= n:
            break

        model_prob = game['model_prob']
        if model_prob >= 0.5:
            side = 'home'
            ml = game['home_moneyline']
        else:
            side = 'away'
            ml = game['away_moneyline']

        home_won = game['home_winner'] == 1
        bet_won = (side == 'home' and home_won) or (side == 'away' and not home_won)
        stake = 1.0

        if bet_won:
            profit = ml_profit(ml, stake)
        else:
            profit = -stake

        cum_pl += profit
        count += 1

        won_str = 'WIN' if bet_won else 'LOSS'
        print(f"  {count:>3d}  {str(game['date'])[:10]:>10s}  {game['game_id']:>7d}  "
              f"{model_prob:>7.4f}  {side:>5s}  {ml:>+6.0f}  {stake:>6.2f}  "
              f"{won_str:>5s}  {profit:>+8.4f}  {cum_pl:>+10.4f}")


# ── Model quality metrics ─────────────────────────────────────────────────────

def print_model_metrics(pred_df: pd.DataFrame):
    """Compute and print walk-forward model quality metrics."""
    y_true = pred_df['home_winner'].values.astype(int)
    y_prob = pred_df['model_prob'].values
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*60}")
    print("WALK-FORWARD MODEL QUALITY METRICS")
    print(f"{'='*60}")
    print(f"  Predictions:  {len(pred_df)}")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Brier Score:  {brier:.4f} (lower is better)")
    print(f"  Log Loss:     {ll:.4f}")
    print(f"  AUC:          {auc:.4f}")
    print(f"  Home win rate (actual):    {y_true.mean():.4f}")
    print(f"  Home win rate (predicted): {y_prob.mean():.4f}")

    # Per-season breakdown
    for season in sorted(pred_df['season_year'].unique()):
        mask = pred_df['season_year'] == season
        s_true = y_true[mask]
        s_prob = y_prob[mask]
        s_pred = (s_prob >= 0.5).astype(int)
        print(f"\n  Season {season} ({mask.sum()} games):")
        print(f"    Accuracy:    {accuracy_score(s_true, s_pred):.4f}")
        print(f"    Brier:       {brier_score_loss(s_true, s_prob):.4f}")
        print(f"    Log Loss:    {log_loss(s_true, s_prob):.4f}")
        print(f"    AUC:         {roc_auc_score(s_true, s_prob):.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Walk-forward betting backtest')
    parser.add_argument('--save-cache', action='store_true',
                        help='Save walk-forward predictions to parquet after run')
    parser.add_argument('--load-cache', action='store_true',
                        help='Load cached predictions, skip retraining')
    args = parser.parse_args()

    conn = get_connection()

    # Load odds
    odds_df = pd.read_sql_query("""
        SELECT game_id, home_moneyline, away_moneyline,
               home_implied_prob, away_implied_prob
        FROM game_odds
        WHERE home_moneyline IS NOT NULL AND away_moneyline IS NOT NULL
    """, conn)
    print(f"Loaded {len(odds_df)} games with odds")

    cache_path = CACHE_PATH

    if args.load_cache:
        if not cache_path.exists():
            print(f"ERROR: Cache file not found at {cache_path}")
            print("Run with --save-cache first to generate predictions.")
            sys.exit(1)
        print(f"\nLoading cached predictions from {cache_path}...")
        pred_df = pd.read_csv(cache_path)
        print(f"  {len(pred_df)} cached predictions loaded")
        skipped_dates = []  # not available from cache
    else:
        print("\nBuilding features...")
        df = build_features(conn)
        features = select_features(df)
        print(f"Using {len(features)} features")

        print("\nStarting walk-forward backtesting...")
        pred_df, skipped_dates = walk_forward_predictions(df, features)
        print(f"\nTotal predictions: {len(pred_df)}")

        if args.save_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(cache_path, index=False)
            print(f"Predictions cached to {cache_path}")

    conn.close()

    # Report skipped dates
    if skipped_dates:
        print(f"\n{'='*60}")
        print(f"SKIPPED DATES ({len(skipped_dates)} dates with < {MIN_TRAINING_GAMES} training games)")
        print(f"{'='*60}")
        for season, date, n_train in skipped_dates:
            print(f"  {season} {date}: {n_train} training games")

    # Model quality metrics (all predictions, not just those with odds)
    print_model_metrics(pred_df)

    # Run strategies
    daily_df, summaries = run_strategies(pred_df, odds_df)

    # Output
    print_summary(summaries)
    print_bet_count_verification(summaries)
    print_notable_events(summaries)

    # Diagnostic: first 10 flat_model_winner bets
    print_bet_trace(pred_df, odds_df, n=10)

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDaily P/L saved to {OUTPUT_CSV} ({len(daily_df)} rows)")


if __name__ == '__main__':
    main()
