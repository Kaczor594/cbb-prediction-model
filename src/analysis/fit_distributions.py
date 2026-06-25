"""
Distribution Fitting Analysis for CBB Prediction Model

Tests key metrics against common statistical distributions using MLE fitting,
KS goodness-of-fit tests, and AIC for model comparison.

Player-level stats are weighted by minutes played via resampling: each player's
observation is replicated proportionally to their minutes share, producing a
weighted sample that is then fit normally.

Usage:
    python src/analysis/fit_distributions.py
    python src/analysis/fit_distributions.py --table team_efficiency
    python src/analysis/fit_distributions.py --verbose
"""

import argparse
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Distributions to test ────────────────────────────────────────────────────

DISTRIBUTIONS = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "beta": stats.beta,
    "skewnorm": stats.skewnorm,
    "t": stats.t,
    "weibull_min": stats.weibull_min,
    "exponential": stats.expon,
}

# Distributions that require positive data
POSITIVE_ONLY = {"lognormal", "gamma", "weibull_min", "exponential"}
# Distributions that require data in (0, 1)
UNIT_INTERVAL = {"beta"}


# ── Core fitting logic ───────────────────────────────────────────────────────

def fit_distribution(data: np.ndarray, dist_name: str, dist) -> dict | None:
    """
    Fit a single distribution to data via MLE and evaluate goodness-of-fit.

    Returns dict with params, ks_stat, p_value, aic, or None on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = dist.fit(data)

            # KS test against fitted distribution
            ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))

            # AIC = 2k - 2ln(L)
            log_likelihood = np.sum(dist.logpdf(data, *params))
            if not np.isfinite(log_likelihood):
                return None

            k = len(params)
            aic = 2 * k - 2 * log_likelihood

            return {
                "dist_name": dist_name,
                "params": params,
                "ks_stat": ks_stat,
                "p_value": p_value,
                "aic": aic,
                "log_likelihood": log_likelihood,
            }
    except Exception:
        return None


def fit_all_distributions(data: np.ndarray) -> list[dict]:
    """
    Fit all candidate distributions to data, returning results sorted by AIC.

    Automatically filters distribution candidates based on data range
    (e.g., skips lognormal for data with negative values).
    """
    data = data[np.isfinite(data)]
    if len(data) < 30:
        return []

    results = []
    data_min, data_max = data.min(), data.max()

    for dist_name, dist in DISTRIBUTIONS.items():
        # Skip distributions incompatible with data range
        if dist_name in POSITIVE_ONLY and data_min <= 0:
            continue
        if dist_name in UNIT_INTERVAL:
            if data_min <= 0 or data_max >= 1:
                continue

        result = fit_distribution(data, dist_name, dist)
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r["aic"])
    return results


def confidence_label(p_value: float) -> str:
    """Classify fit confidence based on KS test p-value."""
    if p_value >= 0.10:
        return "HIGH"
    elif p_value >= 0.01:
        return "MEDIUM"
    else:
        return "LOW"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_team_efficiency(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    """Load team_efficiency metrics, keyed by column name."""
    metrics = [
        "adj_oe", "adj_de", "adj_tempo", "barthag", "overall_rank",
        "wins", "losses",
    ]
    result = {}
    for col in metrics:
        rows = conn.execute(
            f"SELECT {col} FROM team_efficiency WHERE {col} IS NOT NULL"
        ).fetchall()
        if rows:
            result[col] = np.array([r[0] for r in rows], dtype=float)
    return result


def load_player_stats_weighted(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    """
    Load player_stats metrics weighted by minutes played.

    Weighting is done via resampling: each player's observation is replicated
    proportionally to their minutes share (normalized so total sample size is
    preserved). Players with zero or null minutes are excluded.
    """
    metrics = [
        "ortg", "usage", "efg_pct", "ts_pct", "ftr",
        "ft_pct", "two_p_pct", "three_p_pct",
        "orb_pct", "drb_pct", "ast_pct", "to_pct", "blk_pct", "stl_pct",
        "bpm", "obpm", "dbpm",
        "pts_pg", "ast_pg", "treb_pg",
        "adj_oe", "adj_de",
    ]

    result = {}
    rng = np.random.default_rng(42)  # reproducible

    for col in metrics:
        rows = conn.execute(f"""
            SELECT {col}, minutes FROM player_stats
            WHERE {col} IS NOT NULL AND minutes IS NOT NULL AND minutes > 0
        """).fetchall()

        if not rows or len(rows) < 30:
            continue

        values = np.array([r[0] for r in rows], dtype=float)
        minutes = np.array([r[1] for r in rows], dtype=float)

        # Normalize minutes to weights summing to sample size
        weights = minutes / minutes.sum() * len(values)

        # Resample with replacement proportional to weights
        indices = rng.choice(len(values), size=len(values), replace=True, p=minutes / minutes.sum())
        result[col] = values[indices]

    return result


def load_team_bpi(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    """Load team_bpi metrics."""
    metrics = [
        "bpi", "bpi_offense", "bpi_defense", "bpi_7day_change",
        "sor_rank", "sos_past_rank",
        "wins", "losses", "top50_wins", "top50_losses",
    ]
    result = {}
    for col in metrics:
        rows = conn.execute(
            f"SELECT {col} FROM team_bpi WHERE {col} IS NOT NULL"
        ).fetchall()
        if rows and len(rows) >= 30:
            result[col] = np.array([r[0] for r in rows], dtype=float)
    return result


def load_game_metrics(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    """Load game-level metrics and derived fields."""
    result = {}

    # Score distributions
    for col in ["home_score", "away_score", "attendance"]:
        rows = conn.execute(
            f"SELECT {col} FROM games WHERE {col} IS NOT NULL"
        ).fetchall()
        if rows and len(rows) >= 30:
            result[col] = np.array([r[0] for r in rows], dtype=float)

    # Derived: score differential, total points
    rows = conn.execute("""
        SELECT home_score, away_score FROM games
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
    """).fetchall()
    if rows:
        home = np.array([r[0] for r in rows], dtype=float)
        away = np.array([r[1] for r in rows], dtype=float)
        result["score_differential"] = home - away
        result["total_points"] = home + away

    return result


# ── Reporting ────────────────────────────────────────────────────────────────

def print_results(table_name: str, metrics: dict[str, np.ndarray], verbose: bool = False):
    """Fit distributions for all metrics in a table and print results."""
    print(f"\n{'=' * 80}")
    print(f"  {table_name}")
    print(f"{'=' * 80}")
    print(f"{'Metric':<22} {'Best Fit':<14} {'KS Stat':>8} {'p-value':>10} {'Conf':>6}  {'Runner-up':<14} {'AIC Δ':>8}")
    print(f"{'-' * 22} {'-' * 14} {'-' * 8} {'-' * 10} {'-' * 6}  {'-' * 14} {'-' * 8}")

    low_confidence = []

    for col, data in sorted(metrics.items()):
        results = fit_all_distributions(data)
        if not results:
            print(f"{col:<22} {'(insufficient data)'}")
            continue

        best = results[0]
        conf = confidence_label(best["p_value"])
        runner_up = results[1]["dist_name"] if len(results) > 1 else "—"
        aic_delta = results[1]["aic"] - best["aic"] if len(results) > 1 else 0

        print(
            f"{col:<22} {best['dist_name']:<14} {best['ks_stat']:>8.4f} "
            f"{best['p_value']:>10.4f} {conf:>6}  {runner_up:<14} {aic_delta:>8.1f}"
        )

        if conf == "LOW":
            low_confidence.append((col, data, results))

        if verbose:
            print(f"  {'':>22} n={len(data)}, mean={data.mean():.2f}, std={data.std():.2f}, "
                  f"skew={stats.skew(data):.2f}, kurt={stats.kurtosis(data):.2f}")
            for r in results[:4]:
                marker = " <-- best" if r is best else ""
                print(f"  {'':>22} {r['dist_name']:<14} AIC={r['aic']:>12.1f}  "
                      f"KS={r['ks_stat']:.4f}  p={r['p_value']:.4f}{marker}")

    return low_confidence


def print_low_confidence_detail(low_confidence_items: list):
    """Print detailed analysis for metrics with low-confidence fits."""
    if not low_confidence_items:
        return

    print(f"\n{'=' * 80}")
    print("  LOW-CONFIDENCE FIT DETAILS")
    print(f"{'=' * 80}")

    for col, data, results in low_confidence_items:
        print(f"\n  {col} (n={len(data)})")
        print(f"    Summary: mean={data.mean():.3f}, median={np.median(data):.3f}, "
              f"std={data.std():.3f}")
        print(f"    Range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"    Skewness: {stats.skew(data):.3f}, Kurtosis: {stats.kurtosis(data):.3f}")
        print(f"    Percentiles: 5th={np.percentile(data, 5):.3f}, "
              f"25th={np.percentile(data, 25):.3f}, "
              f"75th={np.percentile(data, 75):.3f}, "
              f"95th={np.percentile(data, 95):.3f}")
        print(f"    Top 5 fits:")
        for r in results[:5]:
            conf = confidence_label(r["p_value"])
            print(f"      {r['dist_name']:<14} KS={r['ks_stat']:.4f}  "
                  f"p={r['p_value']:.6f}  AIC={r['aic']:>12.1f}  [{conf}]")
            # Print distribution parameters in human-readable form
            param_names = _get_param_names(r["dist_name"])
            param_str = ", ".join(f"{n}={v:.4f}" for n, v in zip(param_names, r["params"]))
            print(f"        params: {param_str}")


def _get_param_names(dist_name: str) -> list[str]:
    """Return human-readable parameter names for a distribution."""
    names = {
        "normal": ["loc (μ)", "scale (σ)"],
        "lognormal": ["shape (σ)", "loc", "scale (eᵘ)"],
        "gamma": ["shape (a)", "loc", "scale"],
        "beta": ["a", "b", "loc", "scale"],
        "skewnorm": ["skew (a)", "loc", "scale"],
        "t": ["df", "loc", "scale"],
        "weibull_min": ["shape (c)", "loc", "scale"],
        "exponential": ["loc", "scale"],
    }
    default = [f"p{i}" for i in range(10)]
    return names.get(dist_name, default)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fit distributions to CBB metrics")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--table", type=str, default=None,
                        help="Only analyze a specific table")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed fit results for all metrics")
    args = parser.parse_args()

    db_path = args.db or "data/cbb_prediction.db"
    conn = sqlite3.connect(db_path)

    all_low_confidence = []

    tables = {
        "team_efficiency": load_team_efficiency,
        "player_stats (minutes-weighted)": load_player_stats_weighted,
        "team_bpi": load_team_bpi,
        "games": load_game_metrics,
    }

    for table_name, loader in tables.items():
        if args.table and args.table not in table_name:
            continue
        metrics = loader(conn)
        low = print_results(table_name, metrics, verbose=args.verbose)
        all_low_confidence.extend(low)

    print_low_confidence_detail(all_low_confidence)

    conn.close()

    print(f"\n{'=' * 80}")
    print(f"  Done. {len(all_low_confidence)} metrics had LOW confidence fits.")
    if all_low_confidence:
        print("  Review the LOW-CONFIDENCE section above — these may need manual inspection")
        print("  or a non-standard distribution (mixture model, kernel density, etc.).")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
