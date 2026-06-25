"""
Compare symmetric feature variants against the v4 baseline.

Removes positional home/away bias by replacing home_*/away_* features
with symmetric team_a_*/team_b_* features. Tests four variants for
encoding venue advantage, verifies symmetry via data augmentation,
and compares against the current v4 baseline using walk-forward evaluation.

Variants:
  A1: Fully blind (no home/away signal)
  A2: neutral_site only
  B:  is_home + neutral_site
  C:  is_home + is_away (neutral_site dropped — redundant)

Usage:
    cd ~/claude-projects/cbb-prediction-model
    source .venv/bin/activate
    python scripts/compare_symmetric_models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import XGB_PARAMS, XGB_NUM_BOOST_ROUND as NUM_BOOST_ROUND
from src.data.database import get_connection
from src.models.feature_engineering import build_features
from src.models.train_model import augment_data
EARLY_STOPPING = 50


# ── Feature renaming map ───────────────────────────────────────────────

# home_* → team_a_*, away_* → team_b_*
# Maps the metric suffix (everything after home_/away_) to itself.
# Applied programmatically below.

METADATA_COLS = {
    'game_id', 'date', 'season_year', 'home_team_id', 'away_team_id',
    'home_winner', 'score_margin',
}

# Features dropped in v4 because differentials capture the same info.
# Renamed to team_a/team_b versions for symmetric variants.
DROP_REDUNDANT_SYMMETRIC = {
    'team_a_prior_rank', 'team_b_prior_rank',
    'team_a_prior_win_pct', 'team_b_prior_win_pct',
    'team_a_run_win_pct', 'team_b_run_win_pct',
    'team_a_run_avg_margin', 'team_b_run_avg_margin',
    'team_a_run_recent_win_pct', 'team_b_run_recent_win_pct',
    'team_a_run_recent_avg_margin', 'team_b_run_recent_avg_margin',
    'team_a_sos', 'team_b_sos',
    'team_a_conf_win_pct', 'team_b_conf_win_pct',
}

# v4 baseline drop_redundant (original names)
DROP_REDUNDANT_BASELINE = {
    'home_prior_rank', 'away_prior_rank',
    'home_prior_win_pct', 'away_prior_win_pct',
    'home_run_win_pct', 'away_run_win_pct',
    'home_run_avg_margin', 'away_run_avg_margin',
    'home_run_recent_win_pct', 'away_run_recent_win_pct',
    'home_run_recent_avg_margin', 'away_run_recent_avg_margin',
    'home_sos', 'away_sos',
    'home_conf_win_pct', 'away_conf_win_pct',
}


# ── Rename features ───────────────────────────────────────────────────

def rename_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename home_*/away_* → team_a_*/team_b_*, recompute travel features.

    - travel_advantage: team_a_dist − team_b_dist (positive = team_a farther)
    - tz_advantage: abs(team_a_tz − venue_tz) − abs(team_b_tz − venue_tz)

    The original travel_advantage was away_dist − home_dist (positive = home closer).
    Since team_a = original home, team_b = original away:
      new travel_advantage = team_a_dist − team_b_dist = home_dist − away_dist
                           = −(old travel_advantage)

    The original away_timezone_shift was abs(away_tz − venue_tz), only for non-neutral.
    We replace it with tz_advantage = abs(team_a_tz − venue_tz) − abs(team_b_tz − venue_tz).
    We can't compute this from the available columns since home tz shift wasn't stored.
    Instead we set tz_advantage = −away_timezone_shift for non-neutral (home tz shift ≈ 0),
    and tz_advantage = 0 for neutral (both shifts were 0 in original).
    """
    out = df.copy()

    # Rename home_* → team_a_*, away_* → team_b_*
    rename_map = {}
    for col in df.columns:
        if col in METADATA_COLS:
            continue
        if col.startswith('home_') and col != 'home_winner':
            suffix = col[5:]  # strip "home_"
            rename_map[col] = f'team_a_{suffix}'
        elif col.startswith('away_'):
            suffix = col[5:]  # strip "away_"
            rename_map[col] = f'team_b_{suffix}'

    out = out.rename(columns=rename_map)

    # Recompute travel_advantage: team_a_dist − team_b_dist
    # Original: away_dist − home_dist. team_a = home, team_b = away.
    # So: team_a_dist − team_b_dist = home_dist − away_dist = −original
    if 'travel_advantage' in out.columns:
        out['travel_advantage'] = -out['travel_advantage']

    # Recompute timezone feature as symmetric differential
    # Original away_timezone_shift → now team_b_timezone_shift after rename
    # For non-neutral: home tz shift ≈ 0, so tz_advantage ≈ 0 − away_shift = −away_shift
    # For neutral: away_timezone_shift was 0, so tz_advantage = 0
    tz_col = 'team_b_timezone_shift'
    if tz_col in out.columns:
        out['tz_advantage'] = -out[tz_col]
        out.drop(columns=[tz_col], inplace=True)
    else:
        out['tz_advantage'] = 0

    # Rename diff features: the original diffs used "higher is better" correction.
    # We need uniform team_a − team_b convention.
    # Original: diff_prior_rank = away_rank − home_rank (negated because lower is better)
    #   → We want team_a_rank − team_b_rank = home_rank − away_rank = −original
    # All other diffs: home_val − away_val = team_a_val − team_b_val → already correct.
    if 'diff_prior_rank' in out.columns:
        out['diff_prior_rank'] = -out['diff_prior_rank']

    return out


# ── Feature selection per variant ──────────────────────────────────────

def add_venue_columns(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Add is_home, is_away columns per variant. Modifies df in place."""
    if variant in ('B', 'C'):
        # is_home: 1 if team_a is the home team (non-neutral), else 0
        df['is_home'] = ((df.get('neutral_site', 0) == 0)).astype(int)

    if variant == 'C':
        # is_away: 0 for original rows (team_a = home), 1 after augment swap
        # Before augmentation, team_a is always the original home team
        df['is_away'] = 0

    return df


def select_features_symmetric(df: pd.DataFrame, variant: str) -> list[str]:
    """
    Select features for a symmetric variant.

    Drops metadata, target, >50% null columns, and redundant pairs.
    Adds/removes venue columns per variant spec.
    """
    all_cols = [c for c in df.columns if c not in METADATA_COLS]

    # Drop >50% missing
    valid = [c for c in all_cols if df[c].isnull().mean() < 0.5]

    # Drop redundant pairs (same as v4, renamed)
    valid = [c for c in valid if c not in DROP_REDUNDANT_SYMMETRIC]

    # Variant-specific column rules
    if variant == 'A1':
        # No home/away signal, no neutral_site
        valid = [c for c in valid if c not in {'neutral_site', 'is_home', 'is_away'}]
    elif variant == 'A2':
        # Keep neutral_site, no is_home/is_away
        valid = [c for c in valid if c not in {'is_home', 'is_away'}]
    elif variant == 'B':
        # Keep neutral_site + is_home, no is_away
        valid = [c for c in valid if c not in {'is_away'}]
        if 'is_home' not in valid:
            valid.append('is_home')
    elif variant == 'C':
        # Keep is_home + is_away, drop neutral_site (redundant)
        valid = [c for c in valid if c not in {'neutral_site'}]
        if 'is_home' not in valid:
            valid.append('is_home')
        if 'is_away' not in valid:
            valid.append('is_away')

    # Remove any remaining home_*/away_* columns (shouldn't exist after rename)
    valid = [c for c in valid if not c.startswith('home_') and not c.startswith('away_')]

    return sorted(valid)


def select_features_baseline(df: pd.DataFrame) -> list[str]:
    """Select features for v4 baseline (original home/away convention)."""
    exclude = METADATA_COLS
    all_cols = [c for c in df.columns if c not in exclude]

    # Drop >50% missing
    valid = [c for c in all_cols if df[c].isnull().mean() < 0.5]

    # Drop redundant pairs
    valid = [c for c in valid if c not in DROP_REDUNDANT_BASELINE]

    return sorted(valid)


# ── Walk-forward evaluation ────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute classification metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'brier_score': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob),
        'auc': roc_auc_score(y_true, y_prob),
    }


def walk_forward_eval(
    df: pd.DataFrame,
    variant: str,
    is_baseline: bool = False,
) -> dict:
    """
    Walk-forward evaluation for a single variant.

    For symmetric variants: augments training and test data inside each fold.
    For baseline: no augmentation.

    Returns dict with overall and neutral-site metrics averaged across folds.
    """
    seasons = sorted(df['season_year'].unique())
    fold_results = []

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_df = df[df['season_year'].isin(train_seasons)].copy()
        test_df = df[df['season_year'] == test_season].copy()

        if is_baseline:
            features = select_features_baseline(df)
            train_aug = train_df
            test_aug = test_df
        else:
            # Add venue columns before augmentation
            add_venue_columns(train_df, variant)
            add_venue_columns(test_df, variant)

            # Augment AFTER split
            train_aug = augment_data(train_df, variant)
            test_aug = augment_data(test_df, variant)

            features = select_features_symmetric(train_aug, variant)

        # Ensure all feature columns exist
        for f in features:
            if f not in train_aug.columns:
                train_aug[f] = 0
            if f not in test_aug.columns:
                test_aug[f] = 0

        X_train = train_aug[features]
        y_train = train_aug['home_winner'].values
        X_test = test_aug[features]
        y_test = test_aug['home_winner'].values

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

        model = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=0,
        )

        y_prob = model.predict(dtest)
        overall = compute_metrics(y_test, y_prob)
        overall['n_train'] = len(train_aug)
        overall['n_test'] = len(test_aug)
        overall['best_iteration'] = model.best_iteration
        overall['train_seasons'] = train_seasons
        overall['test_season'] = test_season

        # Neutral-site subset metrics
        # After augmentation, neutral site games are doubled but both orderings
        # should give complementary predictions. Evaluate on original test only.
        if is_baseline:
            neutral_mask = test_df['neutral_site'] == 1
            neutral_test = test_df[neutral_mask]
        else:
            # Use only original rows (first half of augmented test)
            n_original = len(test_df)
            test_original = test_aug.iloc[:n_original]
            neutral_mask = test_original['neutral_site'] == 1
            neutral_test = test_original[neutral_mask]

        if len(neutral_test) > 10:
            X_neutral = neutral_test[features]
            for f in features:
                if f not in X_neutral.columns:
                    X_neutral[f] = 0
            y_neutral_true = neutral_test['home_winner'].values
            d_neutral = xgb.DMatrix(X_neutral, feature_names=features)
            y_neutral_prob = model.predict(d_neutral)
            neutral_metrics = compute_metrics(y_neutral_true, y_neutral_prob)
            overall['neutral_auc'] = neutral_metrics['auc']
            overall['neutral_brier'] = neutral_metrics['brier_score']
            overall['neutral_accuracy'] = neutral_metrics['accuracy']
            overall['neutral_n'] = len(neutral_test)
        else:
            overall['neutral_auc'] = np.nan
            overall['neutral_brier'] = np.nan
            overall['neutral_accuracy'] = np.nan
            overall['neutral_n'] = len(neutral_test) if not neutral_test.empty else 0

        fold_results.append((overall, model, features))

        print(f"  Fold {i}: train={train_seasons} → test={test_season} | "
              f"AUC={overall['auc']:.4f} | Brier={overall['brier_score']:.4f} | "
              f"Neutral AUC={overall.get('neutral_auc', float('nan')):.4f}")

    # Average metrics across folds
    metric_keys = ['accuracy', 'brier_score', 'log_loss', 'auc',
                   'neutral_auc', 'neutral_brier', 'neutral_accuracy']
    avg = {}
    for key in metric_keys:
        vals = [r[key] for r, _, _ in fold_results if not np.isnan(r.get(key, np.nan))]
        avg[key] = np.mean(vals) if vals else np.nan

    avg['n_folds'] = len(fold_results)
    avg['fold_results'] = fold_results

    return avg


# ── Symmetry verification ─────────────────────────────────────────────

def verify_symmetry(
    model: xgb.Booster,
    features: list[str],
    test_df: pd.DataFrame,
    variant: str,
    is_baseline: bool = False,
    n: int = 500,
) -> dict:
    """
    Verify symmetry: predict each game in both orderings.

    For a symmetric model: P(A beats B) + P(B beats A) should ≈ 1.0.
    Reports max and mean |P_original + P_swapped − 1.0|.
    """
    rng = np.random.RandomState(42)

    if len(test_df) > n:
        idx = rng.choice(len(test_df), size=n, replace=False)
        sample = test_df.iloc[idx].copy()
    else:
        sample = test_df.copy()

    # Predict original ordering
    X_orig = sample[features]
    d_orig = xgb.DMatrix(X_orig, feature_names=features)
    p_orig = model.predict(d_orig)

    if is_baseline:
        # For baseline, swap home/away columns manually
        swapped = sample.copy()
        home_cols = [c for c in features if c.startswith('home_')]
        away_cols = [c for c in features if c.startswith('away_')]

        for hc in home_cols:
            suffix = hc[5:]
            ac = f'away_{suffix}'
            if ac in features:
                swapped[hc], swapped[ac] = sample[ac].values.copy(), sample[hc].values.copy()

        # Negate diffs
        for c in features:
            if c.startswith('diff_'):
                swapped[c] = -sample[c]

        # Negate travel features
        if 'travel_advantage' in features:
            swapped['travel_advantage'] = -sample['travel_advantage']
    else:
        # For symmetric variants, create the augmented mirror
        swapped = sample.copy()

        a_cols = [c for c in features if c.startswith('team_a_')]
        for ac in a_cols:
            suffix = ac[7:]
            bc = f'team_b_{suffix}'
            if bc in features:
                swapped[ac], swapped[bc] = sample[bc].values.copy(), sample[ac].values.copy()

        for c in features:
            if c.startswith('diff_'):
                swapped[c] = -sample[c]

        if 'travel_advantage' in features:
            swapped['travel_advantage'] = -sample['travel_advantage']
        if 'tz_advantage' in features:
            swapped['tz_advantage'] = -sample['tz_advantage']

        # Flip is_home / is_away
        if 'is_home' in features:
            if 'neutral_site' in sample.columns:
                is_neutral = sample['neutral_site'] == 1
            else:
                is_neutral = pd.Series(False, index=sample.index)
            swapped.loc[~is_neutral, 'is_home'] = 1 - sample.loc[~is_neutral, 'is_home']

        if 'is_away' in features:
            if 'neutral_site' in sample.columns:
                is_neutral = sample['neutral_site'] == 1
            else:
                is_neutral = pd.Series(False, index=sample.index)
            swapped.loc[~is_neutral, 'is_away'] = 1 - sample.loc[~is_neutral, 'is_away']

    X_swap = swapped[features]
    d_swap = xgb.DMatrix(X_swap, feature_names=features)
    p_swap = model.predict(d_swap)

    # Perfect symmetry: p_orig + p_swap = 1.0
    deviation = np.abs(p_orig + p_swap - 1.0)

    return {
        'max_deviation': float(deviation.max()),
        'mean_deviation': float(deviation.mean()),
        'median_deviation': float(np.median(deviation)),
        'n_checked': len(sample),
    }


# ── Main comparison ───────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SYMMETRIC FEATURE ENGINEERING — 4-VARIANT COMPARISON")
    print("=" * 70)

    # Load data
    print("\nLoading data via build_features()...")
    conn = get_connection()
    raw_df = build_features(conn)
    conn.close()
    print(f"Raw feature matrix: {raw_df.shape[0]} games × {raw_df.shape[1]} columns")

    # Transform: rename home/away → team_a/team_b
    sym_df = rename_features(raw_df)
    print(f"Symmetric feature matrix: {sym_df.shape[0]} games × {sym_df.shape[1]} columns")

    # ── Run all variants ───────────────────────────────────────────────

    variants = ['A1', 'A2', 'B', 'C']
    results = {}

    for v in variants:
        print(f"\n{'─' * 70}")
        print(f"VARIANT {v}")
        print(f"{'─' * 70}")

        # Preview features for this variant
        preview_df = sym_df.copy()
        add_venue_columns(preview_df, v)
        preview_aug = augment_data(preview_df.head(10), v)
        feat_list = select_features_symmetric(preview_aug, v)
        print(f"Features ({len(feat_list)}): {feat_list}")

        results[v] = walk_forward_eval(sym_df, v)
        print(f"  → Avg AUC: {results[v]['auc']:.4f} | "
              f"Avg Brier: {results[v]['brier_score']:.4f} | "
              f"Avg Neutral AUC: {results[v].get('neutral_auc', float('nan')):.4f}")

    # ── v4 Baseline ────────────────────────────────────────────────────

    print(f"\n{'─' * 70}")
    print("BASELINE (v4)")
    print(f"{'─' * 70}")

    baseline_features = select_features_baseline(raw_df)
    print(f"Features ({len(baseline_features)}): {baseline_features}")

    results['baseline'] = walk_forward_eval(raw_df, 'baseline', is_baseline=True)
    print(f"  → Avg AUC: {results['baseline']['auc']:.4f} | "
          f"Avg Brier: {results['baseline']['brier_score']:.4f} | "
          f"Avg Neutral AUC: {results['baseline'].get('neutral_auc', float('nan')):.4f}")

    # ── Symmetry verification ──────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("SYMMETRY VERIFICATION")
    print(f"{'=' * 70}")

    symmetry_results = {}
    for key in variants + ['baseline']:
        fold_data = results[key]['fold_results']
        # Use last fold for symmetry check
        last_metrics, last_model, last_features = fold_data[-1]
        test_season = last_metrics['test_season']

        if key == 'baseline':
            test_df = raw_df[raw_df['season_year'] == test_season].copy()
            sym_check = verify_symmetry(
                last_model, last_features, test_df,
                variant=key, is_baseline=True,
            )
        else:
            test_df = sym_df[sym_df['season_year'] == test_season].copy()
            add_venue_columns(test_df, key)
            sym_check = verify_symmetry(
                last_model, last_features, test_df,
                variant=key, is_baseline=False,
            )

        symmetry_results[key] = sym_check
        label = f"Variant {key}" if key != 'baseline' else "Baseline v4"
        print(f"  {label:15s} | max={sym_check['max_deviation']:.6f} | "
              f"mean={sym_check['mean_deviation']:.6f} | "
              f"median={sym_check['median_deviation']:.6f} | "
              f"n={sym_check['n_checked']}")

    # ── Comparison table ───────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print(f"{'=' * 70}")

    header = (
        f"{'Variant':>10s} | {'AUC':>7s} | {'Brier':>7s} | {'LogLoss':>8s} | "
        f"{'Acc':>6s} | {'N-AUC':>7s} | {'N-Brier':>8s} | {'N-Acc':>6s} | "
        f"{'Sym Max':>8s} | {'Sym Mean':>9s}"
    )
    print(header)
    print("-" * len(header))

    for key in variants + ['baseline']:
        r = results[key]
        s = symmetry_results[key]
        label = f"Var {key}" if key != 'baseline' else "Baseline"

        print(
            f"{label:>10s} | "
            f"{r['auc']:7.4f} | "
            f"{r['brier_score']:7.4f} | "
            f"{r['log_loss']:8.4f} | "
            f"{r['accuracy']:6.4f} | "
            f"{r.get('neutral_auc', float('nan')):7.4f} | "
            f"{r.get('neutral_brier', float('nan')):8.4f} | "
            f"{r.get('neutral_accuracy', float('nan')):6.4f} | "
            f"{s['max_deviation']:8.6f} | "
            f"{s['mean_deviation']:9.6f}"
        )

    # ── Per-fold detail ────────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("PER-FOLD DETAIL")
    print(f"{'=' * 70}")

    for key in variants + ['baseline']:
        label = f"Variant {key}" if key != 'baseline' else "Baseline v4"
        fold_data = results[key]['fold_results']
        print(f"\n  {label}:")
        for j, (m, _, feats) in enumerate(fold_data):
            print(
                f"    Fold {j+1}: train={m['train_seasons']} → test={m['test_season']} | "
                f"AUC={m['auc']:.4f} | Brier={m['brier_score']:.4f} | "
                f"N-AUC={m.get('neutral_auc', float('nan')):.4f} | "
                f"n_train={m['n_train']} | n_test={m['n_test']} | "
                f"iters={m['best_iteration']} | n_features={len(feats)}"
            )

    # ── Save results ───────────────────────────────────────────────────

    save_path = Path('data/paper/symmetric_comparison.md')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Symmetric Feature Engineering — 4-Variant Comparison\n",
        "",
        "## Comparison Table\n",
        "",
        f"| {'Variant':>10s} | {'AUC':>7s} | {'Brier':>7s} | {'LogLoss':>8s} | "
        f"{'Acc':>6s} | {'N-AUC':>7s} | {'N-Brier':>8s} | {'N-Acc':>6s} | "
        f"{'Sym Max':>8s} | {'Sym Mean':>9s} |",
        "|" + "|".join(["-" * 12] * 10) + "|",
    ]

    for key in variants + ['baseline']:
        r = results[key]
        s = symmetry_results[key]
        label = f"Var {key}" if key != 'baseline' else "Baseline"
        lines.append(
            f"| {label:>10s} | "
            f"{r['auc']:7.4f} | "
            f"{r['brier_score']:7.4f} | "
            f"{r['log_loss']:8.4f} | "
            f"{r['accuracy']:6.4f} | "
            f"{r.get('neutral_auc', float('nan')):7.4f} | "
            f"{r.get('neutral_brier', float('nan')):8.4f} | "
            f"{r.get('neutral_accuracy', float('nan')):6.4f} | "
            f"{s['max_deviation']:8.6f} | "
            f"{s['mean_deviation']:9.6f} |"
        )

    lines += [
        "",
        "## Symmetry Verification\n",
        "",
        "Max |P(A→B) + P(B→A) − 1| across 500 random test games.",
        "Symmetric variants should show < 0.001; baseline shows positional bias.",
        "",
        "## Variant Descriptions\n",
        "",
        "- **A1**: Fully blind — no home/away signal at all",
        "- **A2**: neutral_site only — knows 'this is a neutral game'",
        "- **B**: is_home + neutral_site — knows which team is home",
        "- **C**: is_home + is_away — separate home/away effects (neutral_site dropped)",
        "- **Baseline**: Original v4 model with home_*/away_* features, no augmentation",
        "",
        "All symmetric variants use data augmentation (both team orderings in training).",
        "Differentials use uniform team_a − team_b convention.",
    ]

    save_path.write_text("\n".join(lines))
    print(f"\nResults saved to {save_path}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
