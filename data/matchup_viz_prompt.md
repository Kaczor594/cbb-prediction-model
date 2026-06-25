# XGBoost Matchup Breakdown: UConn Huskies vs Michigan Wolverines

**NCAA Championship Game — Indianapolis, IN — April 7, 2026**

Please create an interactive HTML artifact with four visualizations breaking down this XGBoost prediction. Use navy (#002868) for UConn and maize/gold (#FFCB05) for Michigan.

## Prediction

- **UConn Huskies: 38.8%**
- **Michigan Wolverines: 61.2%**
- Symmetric baseline: 50% (averaged from both team orientations)
- Net SHAP push: -0.4577 log-odds toward Michigan -> sigmoid = 38.8% UConn

## Visualization 1: Feature Category Summary

Horizontal bar chart showing net averaged SHAP contribution by category. Bars favoring UConn go right (navy), bars favoring Michigan go left (maize). Center the bars at 0 (50% baseline).

| Category | Net Avg SHAP | Direction |
|---|---|---|
| Travel & Venue | -0.1330 | Michigan |
| Player Quality (BPM/Roster) | +0.1010 | UConn |
| Current Season Performance | -0.2831 | Michigan |
| Prior Season Quality | +0.0465 | UConn |
| Schedule & Conference | -0.1796 | Michigan |
| Context | -0.0096 | Michigan |

**Key insight**: Michigan is a clear favorite at 61.2%. The biggest driver is **Current Season Performance** (-0.28): Michigan's dominant 18.1 scoring margin dwarfs UConn's 12.2, and Michigan outscores opponents 87.7 to UConn's 77.2 ppg. **Schedule & Conference** (-0.18) adds further separation — Michigan's Big Ten SOS of 0.717 vs UConn's 0.643. **Travel & Venue** (-0.13) also favors Michigan, being ~500 miles closer to Indianapolis. UConn's counterarguments come from **Player Quality** (+0.10) — led by their superior offensive BPM (3.22 vs 3.05) — and **Prior Season Quality** (+0.05) through their stronger prior offensive efficiency (120.8 vs 115.7). But these advantages are insufficient to overcome Michigan's dominant current-season form and schedule strength.

## Visualization 2: Raw Team Comparison

Side-by-side comparison table of key team metrics. Highlight which team leads each metric using team colors.

| Metric | UConn | Michigan |
|---|---|---|
| Win % (2026) | 86.5% | 92.1% |
| Avg Margin | +12.2 | +18.1 |
| Recent Win % (last 10) | 80.0% | 90.0% |
| Recent Avg Margin (last 10) | +9.2 | +11.6 |
| Avg Pts For | 77.2 | 87.7 |
| Avg Pts Against | 65.0 | 69.6 |
| Games Played | 37 | 38 |
| SOS | 0.643 | 0.717 |
| Conf Win % | 62.5% | 87.5% |
| Roster BPM | +4.55 | +6.36 |
| Roster OBPM | +3.22 | +3.05 |
| Roster DBPM | +1.32 | +3.31 |
| Top 5 BPM | +5.71 | +7.18 |
| Depth Count | 9.0 | 8.1 |
| Star Count | 3.1 | 3.9 |
| Prior Barthag | 0.884 | 0.917 |
| Prior Adj OE | 120.8 | 115.7 |
| Prior Adj DE | 101.2 | 93.9 |
| Prior Adj Tempo | 64.1 | 70.0 |
| Prior Rank | 36 | 22 |
| Prior Win % | 68.6% | 73.0% |
| AP Rank | — | — |
| Travel to Indianapolis | ~750 mi | ~250 mi |
| Timezone Advantage | Even (same timezone) |

## Visualization 3: SHAP Waterfall Chart

**Start from 50%** and show how each basketball factor pushes the probability toward UConn (right/navy) or Michigan (left/maize). Show the top 15 concepts individually by absolute contribution, group the rest as "Other". For paired features, show both teams' values in the bar label. The x-axis should show cumulative probability (convert from log-odds using sigmoid). Label the final bar with the prediction (38.8% UConn / 61.2% Michigan).

These SHAP values use **averaged symmetric SHAP** — computed from both team orientations and averaged so the baseline is 50% and there are no positional artifacts.

### Collapsed SHAP Contributions (rebased to 50%, sorted by |contribution|)

| Concept | UConn | Michigan | Avg SHAP | Direction |
|---|---|---|---|---|
| Scoring Margin | +12.2 | +18.1 | -0.4433 | Michigan |
| Strength of Schedule | 0.643 | 0.717 | -0.1813 | Michigan |
| Travel Distance | ~750 mi | ~250 mi | -0.1641 | Michigan |
| Offensive BPM | +3.22 | +3.05 | +0.1214 | UConn |
| Points Scored | 77.2 | 87.7 | +0.0858 | UConn |
| Prior Offensive Efficiency | 120.8 | 115.7 | +0.0820 | UConn |
| Prior Win % | 68.6% | 73.0% | +0.0403 | UConn |
| Overall Roster BPM | +4.55 | +6.36 | -0.0366 | Michigan |
| Recent Margin (last 10) | +9.2 | +11.6 | +0.0353 | UConn |
| Win Percentage | 86.5% | 92.1% | +0.0353 | UConn |
| Rotation Depth | 9.0 | 8.1 | +0.0336 | UConn |
| Prior Defensive Efficiency | 101.2 | 93.9 | -0.0310 | Michigan |
| Prior Tempo | 64.1 | 70.0 | -0.0262 | Michigan |
| Games Played | 37 | 38 | -0.0236 | Michigan |
| Prior Overall Efficiency | 0.884 | 0.917 | -0.0222 | Michigan |
| Venue (Home/Away) | 0 (neutral) | — | +0.0166 | UConn |
| Points Allowed | 65.0 | 69.6 | +0.0165 | UConn |
| Venue (Away) | 0 (neutral) | — | +0.0159 | UConn |
| Recent Win % (last 10) | 80% | 90% | +0.0109 | UConn |
| Top 5 Player Quality | +5.71 | +7.18 | -0.0099 | Michigan |
| Season Progress | 1.0 | — | -0.0087 | Michigan |
| Defensive BPM | +1.32 | +3.31 | -0.0058 | Michigan |
| Prior Season Rank | #36 | #22 | +0.0036 | UConn |
| Star Power (BPM > 5) | 3.1 | 3.9 | -0.0017 | Michigan |
| Timezone Advantage | 0 | — | -0.0015 | Michigan |
| Conference Win % | 62.5% | 87.5% | +0.0010 | UConn |
| Ranked Status | — | — | -0.0009 | Michigan |
| Conference Game | 0 | — | +0.0007 | UConn |

**Note on "Venue" features**: The venue features (Home/Away and Away) show a small residual asymmetry (+0.03 combined) from how the augmented model handles neutral sites. This is a minor modeling artifact, not a basketball signal. The category summary absorbs it into Travel & Venue.

## Visualization 4: Tree Path Diagrams

Show the 5 most impactful trees as decision tree path diagrams. Highlight the path taken for this matchup at each node. Show: feature name, split threshold, actual value, and direction taken (left/right). Use human-readable labels where possible.

### Top 5 Most Impactful Trees

**Tree #810 — leaf = +0.016588 (toward UConn), depth = 4**
1. `diff_top5_bpm` (Top-5 BPM gap) split=0.63, actual=-1.47 -> left (<) — Michigan has the top-5 talent edge
2. `team_a_roster_obpm` (UConn OBPM) split=3.00, actual=3.22 -> right (>=) — but UConn's offensive BPM is elite
3. `diff_prior_win_pct` (prior win% gap) split=0.005, actual=-0.044 -> left (<) — Michigan had better prior win%
4. `diff_run_recent_avg_margin` (recent margin gap) split=-1.6, actual=-2.4 -> left (<) — Michigan's recent form adds context

**Tree #2473 — leaf = +0.014285 (toward UConn), depth = 4**
1. `diff_top5_bpm` (Top-5 BPM gap) split=0.63, actual=-1.47 -> left (<) — Michigan's top-5 edge again
2. `team_a_roster_obpm` (UConn OBPM) split=3.00, actual=3.22 -> right (>=) — UConn's offensive BPM overrides
3. `team_a_top5_bpm` (UConn top-5 BPM) split=5.95, actual=5.71 -> left (<) — UConn below the threshold
4. `diff_run_recent_avg_margin` (recent margin gap) split=-0.3, actual=-2.4 -> left (<) — Michigan dominating recently

**Tree #1403 — leaf = +0.014262 (toward UConn), depth = 4**
1. `diff_top5_bpm` (Top-5 BPM gap) split=0.63, actual=-1.47 -> left (<) — Michigan's talent advantage
2. `team_a_roster_obpm` (UConn OBPM) split=3.00, actual=3.22 -> right (>=) — UConn's offensive engine
3. `team_b_run_avg_pts_against` (Michigan pts allowed) split=68.64, actual=69.58 -> right (>=) — Michigan allows too many
4. `team_b_roster_dbpm` (Michigan DBPM) split=1.82, actual=3.31 -> right (>=) — but Michigan's defensive BPM is elite

**Tree #1104 — leaf = +0.013164 (toward UConn), depth = 4**
1. `team_b_ranked` (Michigan ranked?) split=1, actual=0 -> left (<) — model treats as unranked for this prediction
2. `team_a_prior_adj_de` (UConn prior adj DE) split=99.50, actual=101.22 -> right (>=) — UConn's defense weaker
3. `team_a_star_count` (UConn stars) split=1.69, actual=3.12 -> right (>=) — UConn has enough stars
4. `team_b_run_avg_pts_for` (Michigan pts scored) split=86.60, actual=87.71 -> right (>=) — Michigan's prolific scoring

**Tree #1170 — leaf = -0.012323 (toward Michigan), depth = 4**
1. `team_b_top5_bpm` (Michigan top-5 BPM) split=7.38, actual=7.18 -> left (<) — Michigan just below threshold
2. `team_b_roster_bpm` (Michigan roster BPM) split=5.52, actual=6.36 -> right (>=) — but overall roster BPM is elite
3. `diff_run_avg_margin` (scoring margin gap) split=-14.38, actual=-5.92 -> right (>=) — Michigan's margin dominance
4. `team_b_prior_barthag` (Michigan prior Barthag) split=0.95, actual=0.917 -> left (<) — Michigan below elite threshold

## Narrative Summary

The model gives **Michigan a 61.2% win probability** over UConn in the national championship at Indianapolis — a meaningful but not dominant edge.

Starting from a 50% baseline (the augmented model is fully symmetric with a 50.0% base rate), the factors push as follows:

**Michigan's case (61.2%):**
- **Scoring margin dominance** (-0.44): The single largest factor by far. Michigan's +18.1 average margin crushes UConn's +12.2. Michigan scores 87.7 ppg and plays at a faster pace, while UConn averages 77.2. This alone is worth ~6 percentage points of win probability.
- **Schedule strength** (-0.18): Michigan's Big Ten schedule (SOS 0.717) was significantly tougher than UConn's Big East slate (SOS 0.643). UConn's 62.5% conference win rate vs Michigan's 87.5% compounds this.
- **Travel proximity** (-0.16): Michigan is ~500 miles closer to Indianapolis. Both teams are in the Eastern timezone so there's no timezone advantage, but the travel distance matters.
- **Roster BPM depth** (-0.04): Michigan's overall roster BPM (+6.36) and defensive BPM (+3.31) significantly exceed UConn's (+4.55 and +1.32). Michigan's top-5 BPM (7.18 vs 5.71) and star count (3.9 vs 3.1) add further talent separation.

**UConn's case (38.8%):**
- **Offensive BPM** (+0.12): UConn's offensive BPM of +3.22 edges Michigan's +3.05 — the trees repeatedly key on this, with several impactful trees routing through UConn's OBPM exceeding the 3.00 threshold.
- **Points scored paradox** (+0.09): Despite Michigan scoring 87.7 ppg to UConn's 77.2, the SHAP contribution actually favors UConn here, likely due to interaction effects with pace and defensive context.
- **Prior offensive efficiency** (+0.08): UConn's prior-season adjusted OE (120.8) significantly exceeds Michigan's (115.7), suggesting UConn's offensive system has a stronger pedigree.
- **Rotation depth** (+0.03): UConn fields 9 players averaging 10+ minutes vs Michigan's 8.1, offering more defensive flexibility.

**Bottom line**: Michigan's broad dominance in current-season performance (margin, SOS, win%) creates clear separation over UConn. While UConn has pockets of offensive talent superiority (OBPM, prior OE), these aren't enough to overcome Michigan's across-the-board statistical advantages. The 61/39 split suggests Michigan should win about 3 out of 5 times — a real but not prohibitive edge. UConn's upset path runs through their offensive BPM advantage and depth.

## Feature Glossary

- **Averaged SHAP**: SHAP values computed from both team orientations (UConn as team_a, then Michigan as team_a) and averaged. The biases cancel, giving a 50% baseline. Each bar directly shows "how much this factor pushes toward UConn or Michigan" without positional artifacts.
- **Collapsed concepts**: team_a_X and team_b_X features for the same metric are merged into one row (e.g., "Defensive BPM" combines UConn's +1.32 and Michigan's +3.31 into a single SHAP contribution).
- **travel_advantage**: team_a_dist - team_b_dist to venue. Positive = UConn is farther (~503 miles farther).
- **tz_advantage**: Timezone mismatch differential. Both teams are in Eastern time, so no advantage.
- **roster_bpm/obpm/dbpm**: Minutes-weighted BPM using prior-season Barttorvik stats, weighted by cumulative current-season minutes.
- **top5_bpm**: Average BPM of the 5 highest-minute players.
- **star_count**: Players with prior BPM > 5. **depth_count**: Players averaging 10+ min/game.
- **run_win_pct/avg_margin/recent_***: Running current-season stats (win%, scoring margin, last-10-games form).
- **prior_***: Prior season Barttorvik efficiency metrics (adj_oe, adj_de, barthag, rank).
- **sos**: Strength of schedule (opponents' average win%).
- **season_progress**: Fraction of season completed (0 to 1).
