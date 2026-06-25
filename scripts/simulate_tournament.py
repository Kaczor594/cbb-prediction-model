"""
March Madness 2026 Tournament Simulator.

Uses the trained XGBoost model to simulate the NCAA tournament bracket.
Accounts for:
  - Current team strength (running features as of end of regular season)
  - Injured/ineligible players (roster BPM adjustment)
  - Neutral-site game context
  - Travel distance to venue (when site mappings available)
  - Monte Carlo simulation for advancement probabilities

Usage:
    python scripts/simulate_tournament.py
    python scripts/simulate_tournament.py --sims 50000
    python scripts/simulate_tournament.py --print-bracket
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.models.feature_engineering import (
    build_features, _build_static_roster_strength,
    load_player_season_stats, load_player_game_stats, load_games,
    load_team_efficiency, haversine_miles,
    build_bpm_lookups, get_prior_bpm,
)
from src.utils import normalize_name
from src.data.fetch_injuries import scrape_injuries
from src.models.train_model import select_features
from src.config import (FINAL_FOUR_TEAMS, FINAL_FOUR_NAMES, FINAL_FOUR_SITE, VENUE_COORDS,
                        build_matchup_features, predict_symmetric, load_team_locations,
                        build_trained_model)

# ── 2026 Tournament Bracket ─────────────────────────────────────────────
# ESPN team IDs for each team. Teams listed as (seed, name, espn_id).
# Bracket order: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15

EAST = [
    (1, 'Duke', 150), (16, 'Siena', 2561),
    (8, 'Ohio State', 194), (9, 'TCU', 2628),
    (5, "St. John's", 2599), (12, 'Northern Iowa', 2460),
    (4, 'Kansas', 2305), (13, 'Cal Baptist', 2856),
    (6, 'Louisville', 97), (11, 'South Florida', 58),
    (3, 'Michigan State', 127), (14, 'North Dakota St', 2449),
    (7, 'UCLA', 26), (10, 'UCF', 2116),
    (2, 'UConn', 41), (15, 'Furman', 231),
]

WEST = [
    (1, 'Arizona', 12), (16, 'Long Island', 112358),
    (8, 'Villanova', 222), (9, 'Utah State', 328),
    (5, 'Wisconsin', 275), (12, 'High Point', 2272),
    (4, 'Arkansas', 8), (13, "Hawai'i", 62),
    (6, 'BYU', 252), (11, 'Texas', 251),
    (3, 'Gonzaga', 2250), (14, 'Kennesaw St', 338),
    (7, 'Miami (FL)', 2390), (10, 'Missouri', 142),
    (2, 'Purdue', 2509), (15, 'Queens (NC)', None),  # not in DB
]

SOUTH = [
    (1, 'Florida', 57), (16, 'Prairie View A&M', 2504),
    (8, 'Clemson', 228), (9, 'Iowa', 2294),
    (5, 'Vanderbilt', 238), (12, 'McNeese', 2377),
    (4, 'Nebraska', 158), (13, 'Troy', 2653),
    (6, 'North Carolina', 153), (11, 'VCU', 2670),
    (3, 'Illinois', 356), (14, 'Penn', 219),
    (7, "Saint Mary's", 2608), (10, 'Texas A&M', 245),
    (2, 'Houston', 248), (15, 'Idaho', 70),
]

MIDWEST = [
    (1, 'Michigan', 130), (16, 'Howard', 47),
    (8, 'Georgia', 61), (9, 'Saint Louis', 139),
    (5, 'Texas Tech', 2641), (12, 'Akron', 2006),
    (4, 'Alabama', 333), (13, 'Hofstra', 2275),
    (6, 'Tennessee', 2633), (11, 'Miami (OH)', 193),
    (3, 'Virginia', 258), (14, 'Wright State', 2750),
    (7, 'Kentucky', 96), (10, 'Santa Clara', 2541),
    (2, 'Iowa State', 66), (15, 'Tennessee St', 2634),
]

REGIONS = {
    'East': EAST,
    'West': WEST,
    'South': SOUTH,
    'Midwest': MIDWEST,
}

# Semifinal matchups: East vs South, West vs Midwest (standard NCAA pairing)
SEMIFINAL_PAIRS = [('East', 'South'), ('West', 'Midwest')]

# Regional site assignments
REGIONAL_SITES = {
    'South': 'Houston',
    'West': 'San Jose',
    'Midwest': 'Chicago',
    'East': 'Washington DC',
}


# Pod-to-site mapping for first/second rounds
# Each pod = (1/16/8/9), (5/12/4/13), (6/11/3/14), (7/10/2/15)
# Key: region -> {frozenset of seeds in pod: city}
POD_SITES = {
    'East': {
        frozenset({1, 16, 8, 9}): 'Greenville',
        frozenset({5, 12, 4, 13}): 'San Diego',
        frozenset({6, 11, 3, 14}): 'Buffalo',
        frozenset({7, 10, 2, 15}): 'Philadelphia',
    },
    'South': {
        frozenset({1, 16, 8, 9}): 'Tampa',
        frozenset({5, 12, 4, 13}): 'Oklahoma City',
        frozenset({6, 11, 3, 14}): 'Greenville',
        frozenset({7, 10, 2, 15}): 'Oklahoma City',
    },
    'West': {
        frozenset({1, 16, 8, 9}): 'San Diego',
        frozenset({5, 12, 4, 13}): 'Portland',
        frozenset({6, 11, 3, 14}): 'Portland',
        frozenset({7, 10, 2, 15}): 'St. Louis',
    },
    'Midwest': {
        frozenset({1, 16, 8, 9}): 'Buffalo',
        frozenset({5, 12, 4, 13}): 'Tampa',
        frozenset({6, 11, 3, 14}): 'Philadelphia',
        frozenset({7, 10, 2, 15}): 'St. Louis',
    },
}

# ── Injured / ineligible players ────────────────────────────────────────
INJURED_PLAYERS = [
    # (espn_team_id, player_name)
    (150, 'Caleb Foster'),        # Duke
    (130, 'L.J. Cason'),          # Michigan
    (153, 'Caleb Wilson'),        # North Carolina
    (252, 'Richie Saunders'),     # BYU
    (2641, 'JT Toppin'),          # Texas Tech
    (228, 'Carter Welling'),      # Clemson
    (222, 'Matt Hodge'),          # Villanova
    (97, 'Mikel Brown Jr.'),      # Louisville
    (96, 'Jayden Quaintance'),    # Kentucky
    (8, 'Karter Knox'),           # Arkansas
    (333, 'Aden Holloway'),       # Alabama
]


# ── Team feature snapshot ───────────────────────────────────────────────

def _build_snapshots_with_df(conn) -> tuple:
    """
    Build feature matrix and team snapshots. Returns (snapshots, df).
    Thin wrapper around shared build_team_snapshots that also returns the df
    needed for model training.
    """
    from src.config import build_team_snapshots as _shared_build_snapshots

    print("Building feature matrix for team snapshots...")
    df = build_features(conn)
    snapshots = _shared_build_snapshots(df, season=2026)
    return snapshots, df


def adjust_roster_for_injuries(snapshots: dict, conn) -> dict:
    """
    Adjust roster features for injured/ineligible players.

    For each injured player, recompute the team's roster metrics
    by removing that player's contribution from the minutes-weighted BPM.
    """
    if not INJURED_PLAYERS:
        return snapshots

    print(f"Adjusting roster for {len(INJURED_PLAYERS)} injured players...")

    player_stats = load_player_season_stats(conn)
    player_game = load_player_game_stats(conn)

    if player_game.empty:
        print("  WARNING: No player game stats available for injury adjustment")
        return snapshots

    season = 2026

    # Build cumulative minutes per player per team for 2026
    games = load_games(conn)
    game_season = dict(zip(games['game_id'], games['season_year']))

    # Get cumulative minutes for 2026 season
    pg_2026 = player_game[player_game['game_id'].isin(
        games[games['season_year'] == season]['game_id']
    )]

    cum_minutes = {}  # (team_id, normalized_name) -> total_minutes
    for (tid, pname), grp in pg_2026.groupby(['team_id', 'player_name']):
        name = normalize_name(pname)
        cum_minutes[(int(tid), name)] = grp['minutes'].sum()

    prior_bpm_team, prior_bpm_any, freshman_bpm = build_bpm_lookups(player_stats)

    # For each injured player's team, recompute roster metrics
    injured_by_team = {}
    for tid, pname in INJURED_PLAYERS:
        injured_by_team.setdefault(tid, []).append(normalize_name(pname))

    for tid, injured_names in injured_by_team.items():
        if tid not in snapshots:
            print(f"  WARNING: Team {tid} not in snapshots, skipping")
            continue

        # Get all players who have minutes for this team
        team_players = []
        for (t, name), mins in cum_minutes.items():
            if t == tid and mins > 0:
                bp = get_prior_bpm(tid, name, season,
                                   prior_bpm_team, prior_bpm_any, freshman_bpm)
                team_players.append((name, mins, bp))

        if not team_players:
            continue

        # Compute original (full roster) metrics
        total_min_orig = sum(m for _, m, _ in team_players)

        # Filter out injured players
        active_players = [(n, m, bp) for n, m, bp in team_players
                          if n not in injured_names]
        removed = [(n, m, bp) for n, m, bp in team_players
                   if n in injured_names]

        if not removed:
            print(f"  WARNING: No injured players matched for team {tid}: {injured_names}")
            continue

        for n, m, bp in removed:
            games_count = snapshots[tid].get('run_games', 30)
            avg_min = m / games_count if games_count > 0 else 0
            print(f"  Removing {n} from team {tid}: "
                  f"BPM={bp['bpm']:+.1f}, {m:.0f} total min ({avg_min:.1f}/game)")

        if not active_players:
            continue

        total_min_active = sum(m for _, m, _ in active_players)

        if total_min_active > 0:
            new_bpm = sum(bp['bpm'] * m for _, m, bp in active_players) / total_min_active
            new_obpm = sum(bp['obpm'] * m for _, m, bp in active_players) / total_min_active
            new_dbpm = sum(bp['dbpm'] * m for _, m, bp in active_players) / total_min_active
        else:
            new_bpm = new_obpm = new_dbpm = 0.0

        # Top 5 by minutes
        sorted_active = sorted(active_players, key=lambda x: x[1], reverse=True)
        top5 = sorted_active[:5]
        new_top5_bpm = np.mean([bp['bpm'] for _, _, bp in top5])

        # Depth and star count
        games_played = snapshots[tid].get('run_games', 30)
        new_depth = sum(1 for _, m, _ in active_players if games_played > 0 and m / games_played >= 10)
        new_stars = sum(1 for _, _, bp in active_players if bp['bpm'] > 5)

        old_bpm = snapshots[tid].get('roster_bpm', 0)
        snapshots[tid]['roster_bpm'] = new_bpm
        snapshots[tid]['roster_obpm'] = new_obpm
        snapshots[tid]['roster_dbpm'] = new_dbpm
        snapshots[tid]['top5_bpm'] = new_top5_bpm
        snapshots[tid]['depth_count'] = new_depth
        snapshots[tid]['star_count'] = new_stars

        team_name = next((name for s, name, eid in
                         EAST + WEST + SOUTH + MIDWEST if eid == tid), f'Team {tid}')
        print(f"  {team_name}: roster_bpm {old_bpm:+.2f} -> {new_bpm:+.2f} "
              f"(delta {new_bpm - old_bpm:+.2f})")

    return snapshots


# ── Tournament simulation (uses build_matchup_features from src.config) ───────────────────────────────────────────────

def get_venue_for_round(region: str, round_name: str,
                        seed_a: int, seed_b: int) -> str | None:
    """Determine venue city for a tournament game."""
    if round_name == 'Final Four' or round_name == 'Championship':
        return FINAL_FOUR_SITE
    if round_name in ('Sweet 16', 'Elite 8'):
        return REGIONAL_SITES.get(region)
    if round_name in ('Round of 64', 'Round of 32'):
        region_pods = POD_SITES.get(region, {})
        for pod_seeds, city in region_pods.items():
            if seed_a in pod_seeds or seed_b in pod_seeds:
                return city
    return None


def simulate_tournament(model, features: list[str], snapshots: dict,
                        team_locs: dict = None,
                        n_sims: int = 10000, seed: int = 42) -> dict:
    """
    Run Monte Carlo simulation of the full tournament bracket.

    Returns: dict of team_id -> {
        'name': str, 'seed': int, 'region': str,
        'R64': float, 'R32': float, 'S16': float,
        'E8': float, 'F4': float, 'Finals': float, 'Champ': float
    }
    """
    rng = np.random.RandomState(seed)

    # Build team info lookup
    all_teams = {}
    for region_name, region_teams in REGIONS.items():
        for seed, name, eid in region_teams:
            if eid is not None:
                all_teams[eid] = {
                    'name': name, 'seed': seed, 'region': region_name,
                    'R64': 0, 'R32': 0, 'S16': 0,
                    'E8': 0, 'F4': 0, 'Finals': 0, 'Champ': 0,
                }

    # Precompute all possible first-round matchup probabilities
    print("Precomputing matchup probabilities...")
    prob_cache = {}

    def get_win_prob(team_a_id, team_b_id, venue_city=None):
        """Get P(team_a wins) from cache or compute."""
        if team_a_id is None or team_b_id is None:
            # Team not in DB (e.g., Queens NC) — heavy underdog as 15/16 seed
            if team_a_id is None:
                return 0.15  # team_a is the unknown team
            return 0.85  # team_b is the unknown team

        cache_key = (team_a_id, team_b_id, venue_city)
        if cache_key in prob_cache:
            return prob_cache[cache_key]

        feat_df = build_matchup_features(
            team_a_id, team_b_id, snapshots, features,
            neutral=True, venue_city=venue_city, team_locs=team_locs,
        )
        prob = float(predict_symmetric(model, feat_df, features)[0])

        prob_cache[cache_key] = prob
        prob_cache[(team_b_id, team_a_id, venue_city)] = 1.0 - prob
        return prob

    # Run simulations
    print(f"Running {n_sims:,} tournament simulations...")

    for sim in range(n_sims):
        # Simulate each region
        final_four = {}

        for region_name, region_teams in REGIONS.items():
            # Round of 64: 8 games per region
            r64_winners = []
            for i in range(0, 16, 2):
                seed_a, name_a, id_a = region_teams[i]
                seed_b, name_b, id_b = region_teams[i + 1]

                venue = get_venue_for_round(region_name, 'Round of 64', seed_a, seed_b)

                # Higher seed is "home" by convention
                if seed_a <= seed_b:
                    prob = get_win_prob(id_a, id_b, venue)
                    winner = (seed_a, name_a, id_a) if rng.random() < prob else (seed_b, name_b, id_b)
                else:
                    prob = get_win_prob(id_b, id_a, venue)
                    winner = (seed_b, name_b, id_b) if rng.random() < prob else (seed_a, name_a, id_a)

                if winner[2] is not None:
                    all_teams[winner[2]]['R64'] += 1
                r64_winners.append(winner)

            # Round of 32: 4 games
            r32_winners = []
            for i in range(0, 8, 2):
                sa, na, ia = r64_winners[i]
                sb, nb, ib = r64_winners[i + 1]

                venue = get_venue_for_round(region_name, 'Round of 32', sa, sb)

                if sa <= sb:
                    prob = get_win_prob(ia, ib, venue)
                    winner = (sa, na, ia) if rng.random() < prob else (sb, nb, ib)
                else:
                    prob = get_win_prob(ib, ia, venue)
                    winner = (sb, nb, ib) if rng.random() < prob else (sa, na, ia)

                if winner[2] is not None:
                    all_teams[winner[2]]['R32'] += 1
                r32_winners.append(winner)

            # Sweet 16: 2 games
            s16_winners = []
            for i in range(0, 4, 2):
                sa, na, ia = r32_winners[i]
                sb, nb, ib = r32_winners[i + 1]

                venue = get_venue_for_round(region_name, 'Sweet 16', sa, sb)

                if sa <= sb:
                    prob = get_win_prob(ia, ib, venue)
                    winner = (sa, na, ia) if rng.random() < prob else (sb, nb, ib)
                else:
                    prob = get_win_prob(ib, ia, venue)
                    winner = (sb, nb, ib) if rng.random() < prob else (sa, na, ia)

                if winner[2] is not None:
                    all_teams[winner[2]]['S16'] += 1
                s16_winners.append(winner)

            # Elite 8: 1 game
            sa, na, ia = s16_winners[0]
            sb, nb, ib = s16_winners[1]

            venue = get_venue_for_round(region_name, 'Elite 8', sa, sb)

            if sa <= sb:
                prob = get_win_prob(ia, ib, venue)
                winner = (sa, na, ia) if rng.random() < prob else (sb, nb, ib)
            else:
                prob = get_win_prob(ib, ia, venue)
                winner = (sb, nb, ib) if rng.random() < prob else (sa, na, ia)

            if winner[2] is not None:
                all_teams[winner[2]]['E8'] += 1
            final_four[region_name] = winner

        # Final Four
        ff_winners = []
        for region_a, region_b in SEMIFINAL_PAIRS:
            sa, na, ia = final_four[region_a]
            sb, nb, ib = final_four[region_b]

            venue = FINAL_FOUR_SITE

            if sa <= sb:
                prob = get_win_prob(ia, ib, venue)
                winner = (sa, na, ia) if rng.random() < prob else (sb, nb, ib)
            else:
                prob = get_win_prob(ib, ia, venue)
                winner = (sb, nb, ib) if rng.random() < prob else (sa, na, ia)

            if winner[2] is not None:
                all_teams[winner[2]]['F4'] += 1
            ff_winners.append(winner)

        # Championship
        sa, na, ia = ff_winners[0]
        sb, nb, ib = ff_winners[1]

        if sa <= sb:
            prob = get_win_prob(ia, ib, FINAL_FOUR_SITE)
            winner = (sa, na, ia) if rng.random() < prob else (sb, nb, ib)
        else:
            prob = get_win_prob(ib, ia, FINAL_FOUR_SITE)
            winner = (sb, nb, ib) if rng.random() < prob else (sa, na, ia)

        if winner[2] is not None:
            all_teams[winner[2]]['Finals'] += 1

        # The championship winner
        if winner[2] is not None:
            all_teams[winner[2]]['Champ'] += 1

    # Convert counts to probabilities
    for tid in all_teams:
        for rd in ['R64', 'R32', 'S16', 'E8', 'F4', 'Finals', 'Champ']:
            all_teams[tid][rd] /= n_sims

    return all_teams


def print_results(results: dict, n_sims: int):
    """Print tournament simulation results."""

    # Sort by championship probability
    sorted_teams = sorted(results.values(), key=lambda x: x['Champ'], reverse=True)

    print(f"\n{'='*90}")
    print(f"  2026 NCAA TOURNAMENT SIMULATION RESULTS ({n_sims:,} simulations)")
    print(f"{'='*90}")

    # Title contenders
    print(f"\n{'':3s}{'Team':22s} {'Seed':>4s} {'Region':>8s} "
          f"{'R64':>6s} {'R32':>6s} {'S16':>6s} {'E8':>6s} "
          f"{'F4':>6s} {'Final':>6s} {'Champ':>6s}")
    print(f"  {'-'*86}")

    for i, t in enumerate(sorted_teams):
        if t['R64'] < 0.01 and i > 30:
            continue
        print(f"  {t['name']:22s} {t['seed']:>4d} {t['region']:>8s} "
              f"{t['R64']:>5.1%} {t['R32']:>5.1%} {t['S16']:>5.1%} {t['E8']:>5.1%} "
              f"{t['F4']:>5.1%} {t['Finals']:>5.1%} {t['Champ']:>5.1%}")

    # Regional breakdown
    for region_name in ['East', 'West', 'South', 'Midwest']:
        region_teams = [t for t in sorted_teams if t['region'] == region_name]
        region_teams.sort(key=lambda x: x['seed'])

        print(f"\n  ── {region_name.upper()} REGION ──")
        print(f"  {'':3s}{'Team':22s} {'Seed':>4s} "
              f"{'R64':>6s} {'R32':>6s} {'S16':>6s} {'E8':>6s} "
              f"{'F4':>6s} {'Champ':>6s}")

        for t in region_teams:
            print(f"  {t['name']:22s} {t['seed']:>4d} "
                  f"{t['R64']:>5.1%} {t['R32']:>5.1%} {t['S16']:>5.1%} {t['E8']:>5.1%} "
                  f"{t['F4']:>5.1%} {t['Champ']:>5.1%}")

    # Most likely Final Four
    print(f"\n  ── MOST LIKELY FINAL FOUR ──")
    for region_name in ['East', 'South', 'West', 'Midwest']:
        region_teams = [t for t in sorted_teams if t['region'] == region_name]
        best = max(region_teams, key=lambda x: x['F4'])
        print(f"  {region_name:10s}: {best['name']:20s} ({best['seed']}) — {best['F4']:.1%}")

    # Most likely champion
    print(f"\n  ── CHAMPIONSHIP ODDS ──")
    top_10 = sorted_teams[:10]
    for t in top_10:
        bar = '█' * int(t['Champ'] * 200)
        print(f"  {t['name']:22s} ({t['seed']:>2d}) {t['Champ']:>5.1%}  {bar}")


def print_first_round_matchups(model, features, snapshots, team_locs=None):
    """Print all first-round matchups with win probabilities."""
    print(f"\n{'='*70}")
    print("  FIRST ROUND MATCHUP PROBABILITIES")
    print(f"{'='*70}")

    for region_name, region_teams in REGIONS.items():
        print(f"\n  ── {region_name.upper()} ──")
        for i in range(0, 16, 2):
            seed_a, name_a, id_a = region_teams[i]
            seed_b, name_b, id_b = region_teams[i + 1]

            venue = get_venue_for_round(region_name, 'Round of 64', seed_a, seed_b)

            if id_a is not None and id_b is not None:
                feat_df = build_matchup_features(
                    id_a, id_b, snapshots, features,
                    neutral=True, venue_city=venue, team_locs=team_locs,
                )
                prob_a = float(predict_symmetric(model, feat_df, features)[0])
            else:
                prob_a = 0.85  # default for team not in DB

            prob_b = 1.0 - prob_a
            venue_str = f" @ {venue}" if venue else ""
            print(f"    ({seed_a:>2d}) {name_a:22s} {prob_a:>5.1%}  vs  "
                  f"{prob_b:>5.1%} {name_b:22s} ({seed_b:>2d}){venue_str}")


def export_matchup_probs(model, features, snapshots, team_locs):
    """
    Export all pairwise matchup win probabilities to CSV for use in R.

    For every possible pair of tournament teams and every venue they could
    meet at, compute P(team_a wins). Writes:
      - data/tourney_teams.csv       (team metadata)
      - data/tourney_matchup_probs.csv (pairwise probabilities with venue)
      - data/tourney_bracket.csv     (bracket structure)
      - data/tourney_venues.csv      (venue assignments per round)
    """
    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)

    # Collect all tournament teams
    all_teams = []
    for region_name, region_teams in REGIONS.items():
        for seed, name, eid in region_teams:
            all_teams.append({
                'team_id': eid if eid is not None else -1,
                'name': name,
                'seed': seed,
                'region': region_name,
            })

    teams_df = pd.DataFrame(all_teams)
    teams_df.to_csv(out_dir / 'tourney_teams.csv', index=False)

    # Determine all venues teams might play at
    all_venues = set()
    for region_name in REGIONS:
        # First round pod sites
        for pod_seeds, city in POD_SITES.get(region_name, {}).items():
            all_venues.add(city)
        # Regional site
        if region_name in REGIONAL_SITES:
            all_venues.add(REGIONAL_SITES[region_name])
    all_venues.add(FINAL_FOUR_SITE)

    # Get unique team IDs (excluding None / -1)
    team_ids = [t['team_id'] for t in all_teams if t['team_id'] > 0]
    unique_ids = sorted(set(team_ids))

    print(f"\nExporting matchup probabilities for {len(unique_ids)} teams "
          f"at {len(all_venues)} venues...")

    rows = []
    total = 0
    for i, tid_a in enumerate(unique_ids):
        for tid_b in unique_ids[i + 1:]:
            for venue in all_venues:
                feat_df = build_matchup_features(
                    tid_a, tid_b, snapshots, features,
                    neutral=True, venue_city=venue, team_locs=team_locs,
                )
                prob = float(predict_symmetric(model, feat_df, features)[0])
                rows.append({
                    'team_a_id': tid_a,
                    'team_b_id': tid_b,
                    'venue': venue,
                    'prob_a_wins': round(prob, 6),
                })
                total += 1

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(unique_ids)} teams processed ({total} matchups)")

    probs_df = pd.DataFrame(rows)
    probs_df.to_csv(out_dir / 'tourney_matchup_probs.csv', index=False)
    print(f"  Wrote {len(probs_df)} matchup probabilities to data/tourney_matchup_probs.csv")

    # Export bracket structure (ordered matchup pairs per region)
    bracket_rows = []
    for region_name, region_teams in REGIONS.items():
        for i in range(0, 16, 2):
            seed_a, name_a, id_a = region_teams[i]
            seed_b, name_b, id_b = region_teams[i + 1]
            bracket_rows.append({
                'region': region_name,
                'game_idx': i // 2,
                'team_a_id': id_a if id_a is not None else -1,
                'team_a_name': name_a,
                'team_a_seed': seed_a,
                'team_b_id': id_b if id_b is not None else -1,
                'team_b_name': name_b,
                'team_b_seed': seed_b,
            })

    bracket_df = pd.DataFrame(bracket_rows)
    bracket_df.to_csv(out_dir / 'tourney_bracket.csv', index=False)

    # Export venue assignments
    venue_rows = []
    for region_name in REGIONS:
        for pod_seeds, city in POD_SITES.get(region_name, {}).items():
            for s in pod_seeds:
                venue_rows.append({
                    'region': region_name,
                    'round': 'R64_R32',
                    'seed': s,
                    'venue': city,
                })
        venue_rows.append({
            'region': region_name,
            'round': 'S16_E8',
            'seed': 0,
            'venue': REGIONAL_SITES.get(region_name, ''),
        })
    venue_rows.append({
        'region': 'All',
        'round': 'F4_Championship',
        'seed': 0,
        'venue': FINAL_FOUR_SITE,
    })
    venue_df = pd.DataFrame(venue_rows)
    venue_df.to_csv(out_dir / 'tourney_venues.csv', index=False)

    # Also export semifinal pairings
    with open(out_dir / 'tourney_semis.csv', 'w') as f:
        f.write('region_a,region_b\n')
        for a, b in SEMIFINAL_PAIRS:
            f.write(f'{a},{b}\n')

    print("  Exported: tourney_teams.csv, tourney_bracket.csv, "
          "tourney_venues.csv, tourney_semis.csv")


# ── Final Four simulation ──────────────────────────────────────────────

def simulate_final_four(model, features: list[str], snapshots: dict,
                        team_locs: dict = None,
                        n_sims: int = 100000, seed: int = 42) -> dict:
    """
    Run Monte Carlo simulation starting from the Final Four.

    Uses actual Final Four teams from FINAL_FOUR_TEAMS constant.
    Semifinal pairs: East vs South, West vs Midwest.

    Returns: dict of team_id -> {
        'name': str, 'seed': int, 'region': str,
        'Finals': float, 'Champ': float,
        'semi_prob': float  (model's head-to-head win prob in semifinal)
    }
    """
    rng = np.random.RandomState(seed)

    # Build results tracking
    results = {}
    for region_name, (s, name, eid) in FINAL_FOUR_TEAMS.items():
        results[eid] = {
            'name': name, 'seed': s, 'region': region_name,
            'Finals': 0, 'Champ': 0,
        }

    # Precompute semifinal and all possible championship probabilities
    prob_cache = {}

    def get_win_prob(team_a_id, team_b_id, venue_city=None):
        cache_key = (team_a_id, team_b_id, venue_city)
        if cache_key in prob_cache:
            return prob_cache[cache_key]
        feat_df = build_matchup_features(
            team_a_id, team_b_id, snapshots, features,
            neutral=True, venue_city=venue_city, team_locs=team_locs,
        )
        prob = float(predict_symmetric(model, feat_df, features)[0])
        prob_cache[cache_key] = prob
        prob_cache[(team_b_id, team_a_id, venue_city)] = 1.0 - prob
        return prob

    venue = FINAL_FOUR_SITE

    # Print head-to-head probabilities
    print(f"\n  Final Four Matchup Probabilities (at {venue}):")
    semi_probs = {}
    for region_a, region_b in SEMIFINAL_PAIRS:
        sa, na, ia = FINAL_FOUR_TEAMS[region_a]
        sb, nb, ib = FINAL_FOUR_TEAMS[region_b]
        if sa <= sb:
            prob = get_win_prob(ia, ib, venue)
            print(f"    ({sa}) {na:20s} {prob:>5.1%}  vs  {1-prob:>5.1%} {nb:20s} ({sb})")
            semi_probs[ia] = prob
            semi_probs[ib] = 1.0 - prob
        else:
            prob = get_win_prob(ib, ia, venue)
            print(f"    ({sb}) {nb:20s} {prob:>5.1%}  vs  {1-prob:>5.1%} {na:20s} ({sa})")
            semi_probs[ib] = prob
            semi_probs[ia] = 1.0 - prob

    # Print all possible championship matchup probabilities
    print(f"\n  Possible Championship Matchups:")
    for region_a, region_b in SEMIFINAL_PAIRS:
        for region_c, region_d in SEMIFINAL_PAIRS:
            if (region_a, region_b) == (region_c, region_d):
                continue
            for r1 in [region_a, region_b]:
                for r2 in [region_c, region_d]:
                    s1, n1, i1 = FINAL_FOUR_TEAMS[r1]
                    s2, n2, i2 = FINAL_FOUR_TEAMS[r2]
                    if s1 <= s2:
                        prob = get_win_prob(i1, i2, venue)
                        print(f"    ({s1}) {n1:20s} {prob:>5.1%}  vs  {1-prob:>5.1%} {n2:20s} ({s2})")
                    else:
                        prob = get_win_prob(i2, i1, venue)
                        print(f"    ({s2}) {n2:20s} {prob:>5.1%}  vs  {1-prob:>5.1%} {n1:20s} ({s1})")

    # Run simulations
    print(f"\n  Running {n_sims:,} Final Four simulations...")

    for sim in range(n_sims):
        ff_winners = []
        for region_a, region_b in SEMIFINAL_PAIRS:
            sa, na, ia = FINAL_FOUR_TEAMS[region_a]
            sb, nb, ib = FINAL_FOUR_TEAMS[region_b]

            if sa <= sb:
                prob = get_win_prob(ia, ib, venue)
                winner_id = ia if rng.random() < prob else ib
            else:
                prob = get_win_prob(ib, ia, venue)
                winner_id = ib if rng.random() < prob else ia

            results[winner_id]['Finals'] += 1
            ff_winners.append(winner_id)

        # Championship
        id_a, id_b = ff_winners
        s_a, s_b = results[id_a]['seed'], results[id_b]['seed']
        if s_a <= s_b:
            prob = get_win_prob(id_a, id_b, venue)
            champ_id = id_a if rng.random() < prob else id_b
        else:
            prob = get_win_prob(id_b, id_a, venue)
            champ_id = id_b if rng.random() < prob else id_a

        results[champ_id]['Champ'] += 1

    # Convert counts to probabilities
    for tid in results:
        results[tid]['Finals'] /= n_sims
        results[tid]['Champ'] /= n_sims
        results[tid]['semi_prob'] = semi_probs.get(tid, 0)

    return results


def print_final_four_results(results: dict, n_sims: int):
    """Print Final Four simulation results."""
    sorted_teams = sorted(results.values(), key=lambda x: x['Champ'], reverse=True)

    print(f"\n{'='*70}")
    print(f"  2026 FINAL FOUR SIMULATION ({n_sims:,} simulations)")
    print(f"  Venue: {FINAL_FOUR_SITE}")
    print(f"{'='*70}")

    print(f"\n  {'Team':22s} {'Seed':>4s} {'Region':>8s} "
          f"{'Semi':>7s} {'Final':>7s} {'Champ':>7s}")
    print(f"  {'-'*58}")

    for t in sorted_teams:
        print(f"  {t['name']:22s} {t['seed']:>4d} {t['region']:>8s} "
              f"{t['semi_prob']:>6.1%} {t['Finals']:>6.1%} {t['Champ']:>6.1%}")

    print(f"\n  ── CHAMPIONSHIP ODDS ──")
    for t in sorted_teams:
        bar = '█' * int(t['Champ'] * 100)
        print(f"  {t['name']:22s} ({t['seed']:>2d}) {t['Champ']:>6.1%}  {bar}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Simulate 2026 NCAA Tournament')
    parser.add_argument('--sims', type=int, default=100000,
                        help='Number of Monte Carlo simulations (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--print-bracket', action='store_true',
                        help='Print first-round matchup probabilities')
    parser.add_argument('--no-injury', action='store_true',
                        help='Skip injury adjustments')
    parser.add_argument('--export-probs', action='store_true',
                        help='Export pairwise matchup probabilities to CSV for R')
    parser.add_argument('--final-four', action='store_true',
                        help='Simulate from Final Four only (uses actual results)')
    args = parser.parse_args()

    conn = get_connection()

    # 0. Scrape injuries before building features so eligibility picks them up
    if not args.no_injury:
        scrape_injuries(conn, filter_teams=FINAL_FOUR_NAMES)

    # 1. Build team snapshots from feature matrix
    snapshots, full_df = _build_snapshots_with_df(conn)
    print(f"Built snapshots for {len(snapshots)} teams")

    # 2. Adjust for injuries (post-hoc removal of OUT players from snapshots)
    if not args.no_injury:
        snapshots = adjust_roster_for_injuries(snapshots, conn)

    # 3. Load team locations for travel features
    team_locs = load_team_locations(conn)

    # 4. Train model on all available data (with augmentation for symmetry)
    print("\nTraining model on all regular season data (with augmentation)...")
    features = select_features(full_df)
    print(f"Using {len(features)} features")

    train_df = full_df.dropna(subset=['home_winner'])
    print(f"  {len(train_df)} games")
    model = build_trained_model(train_df, features)

    if args.final_four:
        # Final Four only simulation
        results = simulate_final_four(
            model, features, snapshots, team_locs,
            n_sims=args.sims, seed=args.seed,
        )
        print_final_four_results(results, args.sims)
    else:
        # 5. Print first-round matchups if requested
        if args.print_bracket:
            print_first_round_matchups(model, features, snapshots, team_locs)

        # 5b. Export pairwise matchup probabilities for R
        if args.export_probs:
            export_matchup_probs(model, features, snapshots, team_locs)

        # 6. Run tournament simulation
        results = simulate_tournament(
            model, features, snapshots, team_locs,
            n_sims=args.sims, seed=args.seed,
        )

        # 7. Print results
        print_results(results, args.sims)

    conn.close()


if __name__ == '__main__':
    main()
