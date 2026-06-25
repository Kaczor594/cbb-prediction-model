# Previous Game Effects in CBB: Clay & Bro (2015) Replication and Critique

## Paper Reviewed

**Clay, D.C. & Bro, A.S. (2015).** "Previous game success as a determinant of future game performance and outcomes in men's NCAA basketball." *International Journal of Sport Psychology*, 46: 441-455.

## Summary of Clay & Bro's Claims

**Data:** 5,048 game pairs from 111 teams in 9 major conferences, 3 seasons (2001-02, 2010-11, 2011-12). Conference regular-season games only.

**Method:** Within-team comparison across 4 categories of previous game result:
1. Major loss (lost by 10+ points)
2. Minor loss (lost by 1-9 points)
3. Minor win (won by 1-9 points)
4. Major win (won by 10+ points)

**Two hypotheses tested:**
- **"Vengeful loser":** Teams that lose, especially by a large margin, channel anger into a better performance next game.
- **"Overconfident winner":** Teams that win big become complacent and are more likely to lose next game.

**Key findings (Table I):**

| Previous Result | Win% Next Game | N |
|---|---|---|
| Lost by 10+ | **53.9%** | 294 |
| Lost by 1-9 | **52.2%** | 303 |
| Won by 1-9 | **48.1%** | 301 |
| Won by 10+ | **44.6%** | 291 |

ANOVA: F = 5.013, p < 0.01, R = **-0.11**

The paper also examines performance channels (FG%, offensive/defensive efficiency) and finds the effect is diffuse — transmitted through overall offensive and defensive efficiency rather than any single performance variable.

## Critical Assessment

### Strengths

1. **Within-team design** — Comparing each team to its own baseline performance across game categories is clever. It controls for team quality without needing an external metric (unlike the travel paper which relied on tournament seed).

2. **Conference-only restriction** — Limits scheduling mismatches from non-conference play where power-conference teams routinely beat weak opponents.

3. **Performance channels** — They don't just look at outcomes; they trace the effect through offensive/defensive efficiency, FG%, etc. This is methodologically richer than a pure win/loss analysis.

4. **Monotonic pattern** — Win% drops cleanly across all 4 categories, consistent with a real signal.

5. **Theoretically grounded** — The "vengeful loser" and "overconfident winner" hypotheses are well-articulated and connect to established sports psychology literature.

### Weaknesses

1. **Regression to the mean (CRITICAL)** — This is the fatal confound the paper fails to address. A team that just lost by 10+ almost certainly played a strong opponent. Their next conference game is drawn from the rest of the schedule, which is likely easier on average. Similarly, blowout wins signal a weak previous opponent. What looks like "motivation" is indistinguishable from mean reversion in opponent difficulty. Our data confirms this powerfully (see Test 3 below).

2. **No control for opponent quality of current game** — The within-team design controls for *own* team quality but ignores who you're *about to face*. Previous margin is correlated with both previous AND next opponent strength (see Test 3).

3. **No control for home/away** — The paper claims home/away is "randomized out" across the season, but it is absolutely not randomized *within* previous-result categories. A team that just lost big was disproportionately on the road (29% home); a team that just won big was disproportionately at home (71%). If the next game alternates (road → home), the "bounce back" is just the home court advantage kicking in (see Test 4).

4. **Arbitrary 10-point threshold** — Same researcher-degrees-of-freedom issue as the travel paper. They acknowledge testing other cutoffs.

5. **Small effect (R = -0.11)** — Explains ~1.2% of variance. With N = 1,189 aggregated observations, this is statistically significant but practically tiny.

6. **Causal framing without causal evidence** — The paper attributes the effect to "motivation" but never measures motivation. The conclusion section asserts that the observed effects are "most likely played out through how they differentially affect player motivation" — but this is speculative.

7. **Aggregated analysis masks confounds** — By aggregating to the year-team-previous-result level (1,189 observations from 111 teams × 3 seasons × ~4 categories), they smooth over game-level confounds that would be visible with proper controls.

## Our Replication Data

- **Dataset:** 16,689 D1 games (2024–2026 seasons), all game types
- **Paired observations:** 32,295 (each team-game linked to its previous game in the same season)
- **Quality metric:** Barttorvik barthag (continuous team strength, 0-1)

## Empirical Results

### Test 1: Raw Win% by Previous Game Result (All Games)

| Previous Result | Win% Next Game | N |
|---|---|---|
| Lost by 10+ | **39.4%** | 8,069 |
| Lost by 1-9 | **48.1%** | 7,649 |
| Won by 1-9 | **52.3%** | 8,097 |
| Won by 10+ | **59.8%** | 8,480 |

ANOVA: F = 244.17, p < 0.001, R = **+0.148**

**Our data shows the OPPOSITE pattern from Clay.** Teams that just won big win 59.8% of the time next game; teams that just lost big win only 39.4%. The correlation is positive (+0.148), not negative (-0.11).

This reversal occurs because we include all games (not just conference), and thus team quality is not controlled. Strong teams both win big AND win their next game; weak teams lose big AND lose their next game.

### Test 2: Non-Neutral Regular Season Games

| Previous Result | Win% Next Game | N |
|---|---|---|
| Lost by 10+ | **38.5%** | 6,923 |
| Lost by 1-9 | **47.6%** | 6,487 |
| Won by 1-9 | **53.6%** | 6,485 |
| Won by 10+ | **61.0%** | 6,719 |

ANOVA: F = 253.92, p < 0.001, R = +0.166

Same pattern. The raw effect is *stronger* for non-neutral games because home/away effects amplify it.

### Test 3: Regression to the Mean (The Key Confound)

**Previous opponent quality by previous result:**

| Previous Result | Prev Opp Barthag | Curr Opp Barthag | Own Barthag |
|---|---|---|---|
| Lost by 10+ | **0.628** | 0.494 | **0.393** |
| Lost by 1-9 | **0.523** | 0.490 | **0.464** |
| Won by 1-9 | **0.466** | 0.502 | **0.524** |
| Won by 10+ | **0.396** | 0.528 | **0.627** |

**This is the smoking gun.** Teams that lost by 10+ are weak teams (own barthag = 0.393) who just faced strong opponents (prev_opp = 0.628). Teams that won by 10+ are strong teams (0.627) who just beat weak opponents (0.396).

Correlations:
- prev_margin ↔ prev_opp quality: r = **-0.368** (p < 0.001) — big losses come from facing strong opponents
- prev_margin ↔ own team quality: r = **+0.373** (p < 0.001) — winners are better teams
- prev_margin ↔ next_opp quality: r = +0.053 (p < 0.001) — slight positive, meaning winners face slightly better opponents next

The "vengeful loser" effect is almost entirely explained by: weak teams lose big, and weak teams also lose their next game.

### Test 4: Home/Away Confound

**Previous game home% by previous result:**

| Previous Result | Prev Game Home% | Current Game Home% |
|---|---|---|
| Lost by 10+ | **29.0%** | 47.5% |
| Lost by 1-9 | **43.9%** | 50.3% |
| Won by 1-9 | **56.3%** | 49.8% |
| Won by 10+ | **71.3%** | 52.4% |

Teams that lost big were overwhelmingly on the road (71% away). Teams that won big were overwhelmingly at home (71.3%). This is not "randomized out" — it's a systematic confound. Conference schedules alternate home/away, so a team that just lost big on the road is more likely to play at home next, inflating their "bounce back" win rate. Clay's claim that home/away is random across categories is incorrect.

### Test 5: Logistic Regression with Quality Controls

| Model | β(prev_margin) | β(quality_diff) | β(is_home) | Accuracy |
|---|---|---|---|---|
| A: prev_margin only | **0.0216** | — | — | 56.3% |
| B: quality + home only | — | 6.162 | 0.896 | 74.7% |
| C: prev_margin + quality + home | **0.0008** | 6.147 | 0.898 | 74.7% |

**The prev_margin coefficient shrinks by 96.3% when quality and home/away are controlled.** Adding previous margin to a model with quality + home improves accuracy by only +0.04 percentage points — effectively zero.

### Test 6: Effect Stratified by Team Quality (Barthag Quartile)

| Quartile | After Lost 10+ | After Lost 1-9 | After Won 1-9 | After Won 10+ |
|---|---|---|---|---|
| Q1 (weak) | 26.0% | 32.4% | 35.6% | 36.0% |
| Q2 | 42.2% | 47.9% | 47.4% | 49.4% |
| Q3 | 50.4% | 55.8% | 58.6% | 60.1% |
| Q4 (strong) | 60.8% | 62.2% | 63.7% | 70.7% |

Within each quality quartile, the monotonic pattern **persists in the same direction** (won big → higher win% next game, not lower). This is the opposite of Clay's finding. Even within a quality band, previous winners are slightly more likely to win again — consistent with "success breeds success" (momentum), not "overconfident winner."

However, the within-quartile effects are much smaller than the across-quartile effects. The range within Q4 (60.8% to 70.7% = 9.9pp) is dwarfed by the range across quartiles for the same category (Lost 10+: 26.0% to 60.8% = 34.8pp). Team quality dominates.

## Why Clay's Results Differ from Ours

Clay's within-team design and conference-only restriction partially controls for team quality, which flips the sign of the correlation. In our unrestricted data, team quality dominates and winners keep winning. Clay's design nets out the team quality component, leaving a residual where losers appear to bounce back.

However, Clay's design does NOT control for:
1. **Opponent quality sequence** — within conference play, the schedule sequence still creates regression-to-mean effects
2. **Home/away alternation** — teams losing big on the road often return home, inflating "bounce back" rates
3. **Sample heterogeneity** — 9 major conferences may have systematic scheduling patterns

We believe the "vengeful loser" effect reported by Clay is largely an artifact of these uncontrolled confounds, particularly home/away alternation and opponent quality regression to the mean.

## Summary

| Claim | Clay (2015) | Our Data |
|---|---|---|
| Previous loss → higher next win% | ✓ (53.9% after big loss) | ✗ (39.4% after big loss, opposite direction) |
| Previous win → lower next win% | ✓ (44.6% after big win) | ✗ (59.8% after big win, opposite direction) |
| Effect is monotonic across categories | ✓ | ✓ (but in opposite direction) |
| Effect persists with quality controls | Not tested | ✗ (96.3% coefficient shrinkage, +0.04pp accuracy) |
| Home/away is random across categories | Claimed | ✗ (29% vs 71% home in prev game across categories) |
| Effect channeled through OFF/DEF efficiency | ✓ | Not tested (no game-level box score features) |

## Implications

The previous game margin contributes almost nothing once team quality and home/away status are properly controlled. Our XGBoost model already captures "momentum" through running features (`run_recent_win_pct`, `run_recent_avg_margin`) which are far more informative than a single previous game result, and these features rank in the top 10 by SHAP importance. There is no evidence that adding a dedicated "previous game margin" feature would improve predictions.
