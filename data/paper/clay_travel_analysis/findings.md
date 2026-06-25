# Travel & Timezone Effects in CBB: Clay et al. Replication and Critique

## Papers Reviewed

1. **Clay, D.C., Bro, A.S., & Clay, N.J. (2015).** "Geospatial Determinants of Game Outcomes in NCAA Men's Basketball." *International Journal of Sport and Society*, Vol. 4.
2. **Clay, D.C., Bro, A.S., & Clay, N.J. (2014).** "How Does Travel Affect NCAA Basketball Outcomes?" *Winthrop Intelligence*, Feb 17, 2014.

Both papers report the same underlying study. The Winthrop Intelligence piece is a popular-audience summary of the IJSS journal article.

## Summary of Clay et al.'s Claims

**Data:** 3,296 team performances from 1,648 NCAA Tournament games (1986–2011). 278 teams, 83 tournament sites, all geocoded.

**Method:** Logistic regression predicting win/loss from geospatial variables, controlling for tournament seed and round.

**Key findings:**
1. Traveling **east** across time zones significantly reduces odds of winning (OR = 0.861, p = 0.019). Traveling west is negative but not significant (OR = 0.937, p = 0.257).
2. Traveling **>150 miles** from home reduces odds by 33.6% (OR = 0.644, p = 0.004).
3. Elevation and temperature changes have **no significant effect**.
4. The NCAA's pod format (2002+) reduced travel for top seeds but had **no significant independent effect** on outcomes (OR = 1.002, p = 0.982).
5. The east/west asymmetry is **opposite** to NFL/NBA findings, hypothesized to be because tournament games are played during the day (favoring Eastern teams' circadian rhythms) vs. night games in pro sports.

## Our Replication Data

- **Dataset:** 16,689 D1 games (2024–2026 seasons), regular season + postseason
- **Features:** `travel_advantage` (haversine distance differential), `tz_advantage` (absolute TZ shift differential)
- **Model:** XGBoost with 46 features including team quality metrics (prior barthag, roster BPM, running margin, etc.)

## Empirical Results

### Test 1: Travel Distance vs. Winning

| Subset | Correlation (r) | p-value | N |
|---|---|---|---|
| Non-neutral games | -0.052 | < 0.001 | 14,721 |
| Neutral-site games | -0.024 | 0.297 | 1,968 |

Travel distance has a **statistically significant but small** effect for non-neutral games (0.27% of variance). For neutral-site games — Clay's entire dataset — the effect is **not statistically significant** in our data.

### Test 2: East vs. West Timezone Asymmetry

| TZ Crossed | Direction | Away Win % | N |
|---|---|---|---|
| 0 | same | 36.5% | 10,279 |
| 1 | east | 33.8% | 1,738 |
| 1 | west | 33.4% | 3,631 |
| 2 | east | 32.4% | 207 |
| 2 | west | 28.3% | 551 |
| 3 | east | 25.8% | 93 |
| 3 | west | 25.5% | 247 |

**Our data does not replicate Clay's east/west asymmetry.** At 2+ timezones, traveling *west* actually produces a worse away win rate (28.3%) than east (32.4%). Overall cross-TZ: east 33.3% vs west 31.5%.

### Test 3: Team Quality Confounding

| Travel Direction | Away Quality (barthag) | Home Quality | Gap (H-A) | N |
|---|---|---|---|---|
| Same TZ | 0.450 | 0.491 | +0.041 | 10,279 |
| Away went East | 0.520 | 0.566 | +0.046 | 2,038 |
| Away went West | 0.506 | 0.568 | +0.062 | 2,404 |

Teams traveling west face a **larger quality gap** than those traveling east. Logistic regression with quality control:

| Model | tz_east OR | tz_west OR | quality_diff OR |
|---|---|---|---|
| TZ only | 1.269 | 1.199 | — |
| TZ + quality | 1.207 | 1.167 | 0.044 |

Both TZ coefficients shrink ~25% when quality is added. The east/west gap narrows substantially.

### Test 4: XGBoost Feature Importance (SHAP)

| Rank | Feature | Mean |SHAP| | % of Total |
|---|---|---|---|
| 1 | diff_run_avg_margin | 0.568 | 19.1% |
| 2 | diff_roster_bpm | 0.261 | 8.8% |
| ... | ... | ... | ... |
| **8** | **travel_advantage** | **0.080** | **2.7%** |
| ... | ... | ... | ... |
| **36** | **tz_advantage** | **0.030** | **1.0%** |

Travel distance is the **8th most important feature** — meaningful but dwarfed by team quality. Timezone advantage ranks **36th of 46** features.

### Test 5: Team Distribution by Timezone

| Timezone | Teams | % |
|---|---|---|
| Eastern (UTC-5) | 187 | 52% |
| Central (UTC-6) | 112 | 31% |
| Mountain (UTC-7) | 24 | 7% |
| Pacific (UTC-8) | 38 | 10% |

The massive Eastern concentration means most cross-TZ games involve Eastern teams, creating systematic confounding.

### Test 6: tz_advantage for Neutral Sites

After refactoring `build_travel_features()` to compute timezone shifts for all games (including neutral-site), **896 of 1,968 neutral-site games** now have nonzero `tz_advantage`. Previously this feature was identically zero for neutral sites, giving the model no training signal for timezone effects at neutral sites — exactly the context where Clay claims they matter most. The fix computes each team's timezone offset relative to the actual game venue for all games.

## Critical Assessment

### Methodological Concerns

1. **Inadequate quality control:** Seed (1–16 ordinal) is too coarse. A continuous metric like barthag or KenPom efficiency captures team strength far better.
2. **Double-counting observations:** Using 3,296 "team performances" from 1,648 games creates dependent observations, inflating statistical power.
3. **Post-hoc threshold optimization:** The 150-mile cutoff was selected by testing multiple thresholds without multiple-comparison correction.
4. **Tournament selection bias:** Only top ~20% of D1 teams, with seeding committee explicitly managing travel distances.
5. **Untested mechanism:** The circadian rhythm explanation for east/west asymmetry is plausible but not tested (no time-of-day variable in the model).
6. **No distance × timezone interaction:** These are correlated but modeled independently.

### What Clay Gets Right

- Travel distance matters (our model confirms as feature #8)
- Elevation/temperature irrelevant for indoor basketball
- Fan proximity is a real advantage
- The circadian rhythm framing is theoretically sound

### What Clay Likely Overstates

- The east/west asymmetry effect size and significance (our data doesn't replicate it)
- The 33.6% odds reduction from >150 miles (inflated by inadequate quality controls)
- The practical significance of timezone crossing independent of distance/quality confounding
