"""
Test GLM models for predicting player eligibility (P(plays next game)).

Builds features from player_game_stats box score data and tests:
  - Multiple GLM families (logistic, probit, cloglog)
  - Multiple variable combinations
  - Player-level and team-level cross-validation

Usage:
    python scripts/test_eligibility_model.py
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score, accuracy_score,
)
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial, Poisson
from statsmodels.genmod.families.links import Logit, Probit, CLogLog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.database import get_connection


def build_eligibility_dataset(conn) -> pd.DataFrame:
    """Build dataset with player-game observations and eligibility features."""
    print("Loading player-game records...")
    all_records = pd.read_sql_query("""
        SELECT
            pgs.game_id, pgs.team_id, pgs.player_id, pgs.player_name,
            pgs.did_not_play, pgs.starter, pgs.minutes,
            g.date, g.season_year
        FROM player_game_stats pgs
        JOIN games g ON pgs.game_id = g.game_id
        WHERE g.status = 'STATUS_FINAL'
    """, conn)

    print(f"Total player-game records: {len(all_records):,}")
    all_records['date'] = pd.to_datetime(all_records['date'])
    all_records['did_play'] = (all_records['did_not_play'] == 0).astype(np.int8)
    all_records['minutes'] = pd.to_numeric(all_records['minutes'], errors='coerce').fillna(0).astype(np.float32)
    all_records['starter'] = all_records['starter'].fillna(0).astype(np.int8)

    # Sort chronologically within each player-team-season
    all_records = all_records.sort_values(['team_id', 'season_year', 'player_id', 'date', 'game_id'])
    all_records = all_records.reset_index(drop=True)

    # Group key
    all_records['gk'] = (all_records['team_id'].astype(str) + '_' +
                          all_records['season_year'].astype(str) + '_' +
                          all_records['player_id'].astype(str))

    print("Computing features...")

    # Game number within group
    all_records['game_num'] = all_records.groupby('gk').cumcount()

    # Cumulative played (shifted = before current game)
    all_records['cum_played'] = all_records.groupby('gk')['did_play'].cumsum()
    all_records['cum_played'] = all_records.groupby('gk')['cum_played'].shift(1).fillna(0)

    # Minutes when playing (NaN for DNP)
    all_records['mwp'] = all_records['minutes'].where(all_records['did_play'] == 1)

    # Expanding mean/std of minutes when playing (shifted)
    all_records['avg_min_when_playing'] = (
        all_records.groupby('gk')['mwp']
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(0).astype(np.float32)
    )
    all_records['std_min_when_playing'] = (
        all_records.groupby('gk')['mwp']
        .transform(lambda x: x.expanding().std().shift(1))
        .fillna(0).astype(np.float32)
    )

    # Play rate
    all_records['play_rate'] = (all_records['cum_played'] / all_records['game_num']).fillna(0.5).astype(np.float32)

    # Starter rate
    all_records['cum_starts'] = all_records.groupby('gk')['starter'].cumsum()
    all_records['cum_starts'] = all_records.groupby('gk')['cum_starts'].shift(1).fillna(0)
    all_records['starter_rate'] = np.where(
        all_records['cum_played'] > 0,
        all_records['cum_starts'] / all_records['cum_played'],
        0
    ).astype(np.float32)

    # Consecutive games missed
    print("  Computing consecutive misses...")
    def consec_missed_fn(did_play_arr):
        consec = np.zeros(len(did_play_arr), dtype=np.int16)
        for i in range(1, len(did_play_arr)):
            consec[i] = consec[i - 1] + 1 if did_play_arr[i - 1] == 0 else 0
        return consec

    consec_vals = np.zeros(len(all_records), dtype=np.int16)
    for gk, idx in all_records.groupby('gk').groups.items():
        idx_sorted = idx.sort_values() if hasattr(idx, 'sort_values') else np.sort(idx)
        dp = all_records.loc[idx_sorted, 'did_play'].values
        consec_vals[idx_sorted] = consec_missed_fn(dp)
    all_records['consec_missed'] = consec_vals

    all_records['log_consec_missed'] = np.log1p(all_records['consec_missed']).astype(np.float32)

    # Recent play rate (last 5)
    all_records['recent_play_rate'] = (
        all_records.groupby('gk')['did_play']
        .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        .fillna(0.5).astype(np.float32)
    )

    # Derived features
    all_records['cv_minutes'] = np.where(
        all_records['avg_min_when_playing'] > 1,
        all_records['std_min_when_playing'] / all_records['avg_min_when_playing'],
        0
    ).astype(np.float32)

    all_records['consec_x_avgmin'] = (all_records['consec_missed'] * all_records['avg_min_when_playing']).astype(np.float32)

    # Filter: need at least 3 prior games
    df = all_records[all_records['game_num'] >= 3].copy()

    # Keep only needed columns to save memory
    keep_cols = ['game_id', 'team_id', 'player_id', 'player_name', 'season_year',
                 'date', 'did_play', 'consec_missed', 'log_consec_missed',
                 'avg_min_when_playing', 'std_min_when_playing', 'cv_minutes',
                 'play_rate', 'starter_rate', 'recent_play_rate', 'consec_x_avgmin']
    df = df[keep_cols].reset_index(drop=True)
    del all_records
    gc.collect()

    print(f"Eligibility dataset: {len(df):,} observations")
    print(f"  Players: {df['player_id'].nunique():,}")
    print(f"  Play rate: {df['did_play'].mean():.3f}")
    print(f"  Consec missed > 0: {(df['consec_missed'] > 0).sum():,} "
          f"({(df['consec_missed'] > 0).mean()*100:.1f}%)")
    return df


def run_cv(df, feature_cols, family, split_col='player_id', n_folds=5, seed=42):
    """Cross-validate with entity-level splits."""
    rng = np.random.RandomState(seed)
    entities = df[split_col].unique()
    rng.shuffle(entities)

    fold_size = len(entities) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(entities)
        folds.append(set(entities[start:end]))

    # Precompute feature matrix once
    X_all = sm.add_constant(df[feature_cols].values.astype(np.float64))
    y_all = df['did_play'].values.astype(np.float64)
    split_vals = df[split_col].values

    metrics_list = []
    for i, test_entities in enumerate(folds):
        test_mask = np.isin(split_vals, list(test_entities))
        train_mask = ~test_mask

        try:
            model = sm.GLM(y_all[train_mask], X_all[train_mask], family=family)
            result = model.fit(disp=0, maxiter=100)
            y_pred = np.clip(result.predict(X_all[test_mask]), 1e-6, 1 - 1e-6)
            y_test = y_all[test_mask]
        except Exception:
            continue

        metrics_list.append({
            'fold': i,
            'accuracy': accuracy_score(y_test, (y_pred >= 0.5).astype(int)),
            'brier': brier_score_loss(y_test, y_pred),
            'log_loss': log_loss(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred),
        })

    return pd.DataFrame(metrics_list)


def main():
    conn = get_connection()
    df = build_eligibility_dataset(conn)
    conn.close()

    # ================================================================
    # FEATURE SETS
    # ================================================================
    feature_sets = {
        'consec_only': ['consec_missed'],
        'log_consec_only': ['log_consec_missed'],
        'consec+avgmin': ['consec_missed', 'avg_min_when_playing'],
        'consec+avgmin+interact': ['consec_missed', 'avg_min_when_playing', 'consec_x_avgmin'],
        'consec+playrate': ['consec_missed', 'play_rate'],
        'consec+avgmin+std': ['consec_missed', 'avg_min_when_playing', 'std_min_when_playing'],
        'consec+avgmin+cv': ['consec_missed', 'avg_min_when_playing', 'cv_minutes'],
        'consec+avgmin+starter': ['consec_missed', 'avg_min_when_playing', 'starter_rate'],
        'consec+recentrate': ['consec_missed', 'recent_play_rate'],
        'log+avgmin': ['log_consec_missed', 'avg_min_when_playing'],
        'log+avgmin+starter': ['log_consec_missed', 'avg_min_when_playing', 'starter_rate'],
        'full_linear': ['consec_missed', 'avg_min_when_playing', 'std_min_when_playing',
                        'play_rate', 'starter_rate', 'recent_play_rate'],
        'full_log': ['log_consec_missed', 'avg_min_when_playing', 'std_min_when_playing',
                     'play_rate', 'starter_rate', 'recent_play_rate'],
    }

    # Only test logistic — probit/cloglog in initial sweep, narrow down after
    families_full = {
        'logistic': Binomial(link=Logit()),
        'probit': Binomial(link=Probit()),
        'cloglog': Binomial(link=CLogLog()),
    }

    # ================================================================
    # PLAYER-LEVEL CV (all combos)
    # ================================================================
    print(f"\n{'='*80}")
    print("PLAYER-LEVEL CROSS-VALIDATION (5-fold)")
    print(f"{'='*80}")

    results = []
    total = len(feature_sets) * len(families_full)
    done = 0
    for fs_name, features in feature_sets.items():
        for fam_name, family in families_full.items():
            done += 1
            sys.stdout.write(f"\r  [{done}/{total}] {fs_name:<30s} / {fam_name:<10s}")
            sys.stdout.flush()
            cv_results = run_cv(df, features, family, split_col='player_id')
            if len(cv_results) == 0:
                continue
            means = cv_results.mean(numeric_only=True)
            results.append({
                'features': fs_name,
                'family': fam_name,
                'accuracy': means['accuracy'],
                'brier': means['brier'],
                'log_loss': means['log_loss'],
                'auc': means['auc'],
            })
            gc.collect()

    print()
    results_df = pd.DataFrame(results).sort_values('brier')

    print(f"\n{'Features':<30s} {'Family':<10s} {'Acc':>7s} {'Brier':>7s} "
          f"{'LogLoss':>8s} {'AUC':>7s}")
    print(f"{'-'*70}")
    for _, r in results_df.iterrows():
        print(f"{r['features']:<30s} {r['family']:<10s} {r['accuracy']:>7.4f} "
              f"{r['brier']:>7.4f} {r['log_loss']:>8.4f} {r['auc']:>7.4f}")

    # ================================================================
    # TOP 5 — TEAM-LEVEL CV
    # ================================================================
    print(f"\n{'='*80}")
    print("TEAM-LEVEL CROSS-VALIDATION (top 5 by Brier)")
    print(f"{'='*80}")

    top5 = results_df.head(5)
    team_results = []
    for _, r in top5.iterrows():
        features = feature_sets[r['features']]
        family = families_full[r['family']]
        cv_results = run_cv(df, features, family, split_col='team_id')
        if len(cv_results) == 0:
            continue
        means = cv_results.mean(numeric_only=True)
        team_results.append({
            'features': r['features'],
            'family': r['family'],
            'accuracy': means['accuracy'],
            'brier': means['brier'],
            'log_loss': means['log_loss'],
            'auc': means['auc'],
        })
        gc.collect()

    team_df = pd.DataFrame(team_results).sort_values('brier')
    print(f"\n{'Features':<30s} {'Family':<10s} {'Acc':>7s} {'Brier':>7s} "
          f"{'LogLoss':>8s} {'AUC':>7s}")
    print(f"{'-'*70}")
    for _, r in team_df.iterrows():
        print(f"{r['features']:<30s} {r['family']:<10s} {r['accuracy']:>7.4f} "
              f"{r['brier']:>7.4f} {r['log_loss']:>8.4f} {r['auc']:>7.4f}")

    # ================================================================
    # BEST MODEL: FULL COEFFICIENTS
    # ================================================================
    print(f"\n{'='*80}")
    print("BEST MODEL — COEFFICIENTS AND SUMMARY")
    print(f"{'='*80}")

    best = results_df.iloc[0]
    best_features = feature_sets[best['features']]
    best_family = families_full[best['family']]

    X = sm.add_constant(df[best_features].values.astype(np.float64))
    y = df['did_play'].values.astype(np.float64)

    model = sm.GLM(y, X, family=best_family)
    result = model.fit(disp=0)

    print(f"\nModel: {best['family']} GLM")
    print(f"Features: {best['features']}")
    print(f"AIC: {result.aic:,.0f}")

    coef_names = ['const'] + best_features
    print(f"\n  {'Variable':<30s} {'Coef':>10s} {'Std Err':>10s} {'p-value':>10s}")
    print(f"  {'-'*65}")
    for name, coef, se, pval in zip(coef_names, result.params, result.bse, result.pvalues):
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"  {name:<30s} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {sig}")

    # ================================================================
    # CALIBRATION
    # ================================================================
    print(f"\n{'='*80}")
    print("CALIBRATION: Predicted vs Actual (by decile)")
    print(f"{'='*80}")

    y_pred = np.clip(result.predict(X), 1e-6, 1 - 1e-6)
    df['pred_prob'] = y_pred

    df['prob_bin'] = pd.qcut(df['pred_prob'], 10, duplicates='drop')
    cal = df.groupby('prob_bin', observed=True).agg(
        n=('did_play', 'count'),
        actual=('did_play', 'mean'),
        predicted=('pred_prob', 'mean'),
    ).reset_index()

    print(f"\n  {'Bin':<25s} {'N':>7s} {'Predicted':>10s} {'Actual':>8s} {'Diff':>7s}")
    print(f"  {'-'*60}")
    for _, row in cal.iterrows():
        diff = row['actual'] - row['predicted']
        print(f"  {str(row['prob_bin']):<25s} {row['n']:>7.0f} "
              f"{row['predicted']:>10.4f} {row['actual']:>8.4f} {diff:>+7.4f}")

    # ================================================================
    # P(plays) GRID
    # ================================================================
    print(f"\n{'='*80}")
    print("PREDICTED P(plays) BY CONSECUTIVE MISSES x MINUTES TIER")
    print(f"{'='*80}")

    min_tiers = [('Bench(5m)', 5), ('Rotation(15m)', 15),
                 ('Starter(25m)', 25), ('Star(32m)', 32)]

    print(f"\n  {'Miss':>5s}", end='')
    for label, _ in min_tiers:
        print(f"  {label:>15s}", end='')
    print()
    print(f"  {'-'*68}")

    for n_missed in range(0, 16):
        print(f"  {n_missed:>5d}", end='')
        for label, avg_min in min_tiers:
            feat_dict = {
                'consec_missed': n_missed,
                'log_consec_missed': np.log1p(n_missed),
                'avg_min_when_playing': avg_min,
                'std_min_when_playing': 5.0,
                'cv_minutes': 5.0 / avg_min if avg_min > 0 else 0,
                'play_rate': 0.85,
                'starter_rate': 0.8 if avg_min >= 20 else 0.1,
                'recent_play_rate': max(0.0, 1.0 - n_missed / 5.0),
                'consec_x_avgmin': n_missed * avg_min,
            }
            feat_vals = [feat_dict[f] for f in best_features]
            x = np.array([1.0] + feat_vals).reshape(1, -1)
            pred = result.predict(x)[0]
            print(f"  {pred:>15.3f}", end='')
        print()

    # ================================================================
    # LOGISTIC vs POISSON
    # ================================================================
    print(f"\n{'='*80}")
    print("LOGISTIC vs POISSON COMPARISON (consec+avgmin features)")
    print(f"{'='*80}")

    simple_feats = ['consec_missed', 'avg_min_when_playing']
    X_s = sm.add_constant(df[simple_feats].values.astype(np.float64))

    for fam_name, family in [('logistic', Binomial(link=Logit())), ('poisson', Poisson())]:
        try:
            m = sm.GLM(y, X_s, family=family).fit(disp=0)
            yp = m.predict(X_s)
            yp = np.clip(yp, 1e-6, 1.0) if fam_name == 'poisson' else np.clip(yp, 1e-6, 1 - 1e-6)
            print(f"\n  {fam_name.upper()}: AIC={m.aic:,.0f}  Brier={brier_score_loss(y, yp):.4f}  AUC={roc_auc_score(y, yp):.4f}")
            for n in [0, 1, 3, 5, 10]:
                p = min(m.predict(np.array([[1.0, n, 25.0]]))[0], 1.0)
                print(f"    P(plays | missed={n}, 25min): {p:.3f}")
        except Exception as e:
            print(f"\n  {fam_name.upper()}: FAILED ({e})")

    # ================================================================
    # SUBGROUP ANALYSIS
    # ================================================================
    print(f"\n{'='*80}")
    print("SUBGROUP: by minutes tier")
    print(f"{'='*80}")

    df['min_tier'] = pd.cut(df['avg_min_when_playing'],
                            bins=[0, 5, 10, 20, 50],
                            labels=['Walk-on(<5)', 'Bench(5-10)',
                                    'Rotation(10-20)', 'Starter(20+)'])

    print(f"\n  {'Tier':<18s} {'N':>8s} {'BaseRate':>9s} {'Brier':>7s} {'AUC':>7s}")
    print(f"  {'-'*52}")
    for tier in df['min_tier'].cat.categories:
        mask = df['min_tier'] == tier
        if mask.sum() < 100:
            continue
        sub = df[mask]
        b = brier_score_loss(sub['did_play'], sub['pred_prob'])
        try:
            a = roc_auc_score(sub['did_play'], sub['pred_prob'])
        except ValueError:
            a = float('nan')
        print(f"  {tier:<18s} {mask.sum():>8d} {sub['did_play'].mean():>9.3f} {b:>7.4f} {a:>7.4f}")

    # Calibration by consec missed
    print(f"\n  {'Missed':>7s} {'N':>8s} {'Actual':>8s} {'Predicted':>10s} {'Diff':>7s}")
    print(f"  {'-'*45}")
    for n in list(range(0, 16)) + ['15+']:
        if n == '15+':
            mask = df['consec_missed'] >= 15
        else:
            mask = df['consec_missed'] == n
        if mask.sum() < 10:
            continue
        sub = df[mask]
        actual = sub['did_play'].mean()
        predicted = sub['pred_prob'].mean()
        lbl = str(n)
        print(f"  {lbl:>7s} {mask.sum():>8d} {actual:>8.3f} {predicted:>10.3f} {actual - predicted:>+7.3f}")


if __name__ == '__main__':
    main()
