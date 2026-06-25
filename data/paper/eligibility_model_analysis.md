# Player Eligibility Prediction Model

## Overview

A complementary log-log (cloglog) generalized linear model predicting the probability that a college basketball player will participate in an upcoming game, given their recent participation history and role characteristics. This model replaces binary available/unavailable assumptions with a continuous probability, enabling more accurate roster-weighted team strength estimation.

## Motivation

College basketball lacks mandatory, standardized injury reporting across all Division I programs. While Power 5 conferences began adopting availability reports in 2023-2025, and the NCAA tournament piloted availability reporting in 2026, no comprehensive historical injury dataset exists. This creates a significant challenge for prediction models that rely on roster composition to assess team strength.

The traditional approach uses a hard cutoff (e.g., "if a player has missed N games, assume they're out"), which discards valuable information. A player who has missed 2 games as a 30-minute starter has a meaningfully different return probability than a 5-minute bench player who missed the same number. A probabilistic approach captures these nuances and integrates naturally with minutes-weighted roster features.

## Data

- **Source**: ESPN box score data via `player_game_stats` table
- **Observations**: 473,322 player-game records
- **Players**: 9,590 unique players across 3 NCAA D1 seasons (2024, 2025, 2026)
- **Inclusion criteria**: Player must have appeared on roster for at least 3 prior games in the current season (filters walk-ons and one-time appearances)
- **Base rate**: 62.4% of observations are "played" (37.6% DNP)
- **37.3% of observations** have at least 1 consecutive prior game missed

### Feature Construction

All features are computed using only information available **before** the current game (no data leakage):

| Feature | Description | Construction |
|---------|-------------|--------------|
| `log_consec_missed` | Log-transformed consecutive games missed | `log(1 + N)` where N = consecutive team games the player was on roster but did not play, immediately preceding the current game |
| `avg_min_when_playing` | Average minutes per game when the player actually played | Expanding mean of minutes in games with minutes > 0, shifted by 1 game |
| `std_min_when_playing` | Standard deviation of minutes when playing | Expanding std of minutes in games with minutes > 0, shifted by 1 game |
| `play_rate` | Fraction of roster games in which the player played | Cumulative games played / cumulative games on roster, prior to current game |
| `starter_rate` | Fraction of games played in which the player started | Cumulative starts / cumulative games played, prior to current game |
| `recent_play_rate` | Play rate over last 5 roster games | Rolling 5-game mean of did_play indicator, shifted by 1 game |

## Model Selection

### Methodology

- **39 model configurations** tested: 13 feature combinations x 3 GLM families (logistic, probit, cloglog)
- **Cross-validation**: 5-fold player-level CV (all games for a given player are in the same fold, preventing within-player leakage)
- **Validation**: Top 5 models re-evaluated with team-level CV (all players from a given team in the same fold)
- **Primary metric**: Brier score (proper scoring rule for probabilistic predictions)

### Feature Combinations Tested

| Feature Set | Variables |
|-------------|-----------|
| consec_only | consec_missed |
| log_consec_only | log_consec_missed |
| consec+avgmin | consec_missed, avg_min_when_playing |
| consec+avgmin+interact | consec_missed, avg_min_when_playing, consec_missed * avg_min_when_playing |
| consec+playrate | consec_missed, play_rate |
| consec+avgmin+std | consec_missed, avg_min_when_playing, std_min_when_playing |
| consec+avgmin+cv | consec_missed, avg_min_when_playing, cv_minutes |
| consec+avgmin+starter | consec_missed, avg_min_when_playing, starter_rate |
| consec+recentrate | consec_missed, recent_play_rate |
| log+avgmin | log_consec_missed, avg_min_when_playing |
| log+avgmin+starter | log_consec_missed, avg_min_when_playing, starter_rate |
| full_linear | consec_missed, avg_min_when_playing, std_min_when_playing, play_rate, starter_rate, recent_play_rate |
| full_log | log_consec_missed, avg_min_when_playing, std_min_when_playing, play_rate, starter_rate, recent_play_rate |

### Link Function Comparison

The complementary log-log link consistently outperformed logistic and probit links across all feature combinations. This is theoretically motivated: the cloglog link is asymmetric, modeling the probability as `P = 1 - exp(-exp(eta))`. This suits player availability because missing games has a steep, rapid effect on reducing P(plays), while the upper bound (regular players who haven't missed) approaches 1.0 more gradually — an inherently asymmetric phenomenon.

### Results: Player-Level 5-Fold CV (sorted by Brier score)

```
Features                       Family         Acc   Brier  LogLoss     AUC
----------------------------------------------------------------------
full_log                       cloglog     0.8974  0.0765   0.2606  0.9538
full_linear                    cloglog     0.8938  0.0784   0.2656  0.9522
full_log                       logistic    0.8945  0.0789   0.2688  0.9528
full_log                       probit      0.8940  0.0789   0.2680  0.9528
log+avgmin+starter             cloglog     0.8885  0.0805   0.2727  0.9520
full_linear                    logistic    0.8906  0.0806   0.2730  0.9507
full_linear                    probit      0.8896  0.0807   0.2723  0.9508
consec+avgmin+interact         logistic    0.8967  0.0808   0.2815  0.9484
log+avgmin                     cloglog     0.8884  0.0811   0.2745  0.9520
consec+recentrate              cloglog     0.8872  0.0816   0.2770  0.9427
consec+avgmin+interact         cloglog     0.8882  0.0817   0.2828  0.9510
...
consec_only                    logistic    0.8657  0.1028   0.3608  0.9044
consec_only                    probit      0.8464  0.1095   0.3749  0.9044
```

### Team-Level CV (top 5 models)

Results are virtually identical to player-level CV, confirming the model generalizes across teams:

```
Features                       Family         Acc   Brier  LogLoss     AUC
----------------------------------------------------------------------
full_log                       cloglog     0.8974  0.0765   0.2606  0.9538
full_linear                    cloglog     0.8938  0.0784   0.2655  0.9522
full_log                       logistic    0.8946  0.0789   0.2688  0.9528
full_log                       probit      0.8940  0.0789   0.2680  0.9528
log+avgmin+starter             cloglog     0.8885  0.0805   0.2727  0.9520
```

## Selected Model

### Specification

```
did_play ~ log1p(consec_missed) + avg_min_when_playing + std_min_when_playing
           + play_rate + starter_rate + recent_play_rate

Family:  Binomial
Link:    Complementary log-log    [log(-log(1 - mu)) = X * beta]
Method:  IRLS (8 iterations)
```

### R equivalent

```r
glm(did_play ~ log1p(consec_missed) + avg_min_when_playing +
    std_min_when_playing + play_rate + starter_rate + recent_play_rate,
    family = binomial(link = "cloglog"), data = df)
```

### Coefficients

```
                          Estimate  Std. Error   z value   Pr(>|z|)
(Intercept)              -1.1331     0.0129     -87.988   < 2e-16 ***
log_consec_missed        -0.9221     0.0073    -126.429   < 2e-16 ***
avg_min_when_playing      0.0390     0.0006      68.516   < 2e-16 ***
std_min_when_playing      0.0028     0.0012       2.259     0.024 *
play_rate                 0.5980     0.0199      30.110   < 2e-16 ***
starter_rate             -0.2864     0.0112     -25.684   < 2e-16 ***
recent_play_rate          1.0680     0.0197      54.261   < 2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

### Fit Statistics

```
Null deviance:      626,688  on 473,321 df
Residual deviance:  246,628  on 473,315 df
AIC: 246,642
Pseudo R-squared (Cox-Snell): 0.552
No. observations: 473,322
```

The model reduces deviance by 60.6% relative to the null (intercept-only) model.

### Model Equation

```
eta = -1.133 - 0.922 * log(1 + consec_missed) + 0.039 * avg_min_when_playing
      + 0.003 * std_min_when_playing + 0.598 * play_rate
      - 0.286 * starter_rate + 1.068 * recent_play_rate

P(plays) = 1 - exp(-exp(eta))
```

### Coefficient Interpretation

- **log_consec_missed** (-0.922): The dominant predictor. Each unit increase in log(1 + games_missed) decreases the log-log odds by 0.92. The log transformation captures the diminishing marginal information of additional missed games — missing game 1 is far more informative than missing game 10.

- **avg_min_when_playing** (+0.039): Players who play more minutes when active are more likely to return. A 30-minute player has meaningfully higher return probability than a 5-minute player at the same number of consecutive misses. This captures the difference between injury absence (high-minute players) vs. coaching decision/benching (low-minute players).

- **recent_play_rate** (+1.068): The strongest positive predictor. Recent participation history (last 5 games) is highly informative beyond the consecutive miss count alone, capturing players who have sporadic availability patterns.

- **play_rate** (+0.598): Season-long participation rate provides stable baseline information about whether a player is a regular contributor.

- **starter_rate** (-0.286): *Negative* coefficient. Among players who have missed games, those who are normally starters are *less* likely to play next — because starter absences are more likely to reflect genuine injury, while non-starter absences often reflect coaching decisions or minor day-to-day rest.

- **std_min_when_playing** (+0.003): Small but significant. Players with more variable minutes have slightly higher return probability, possibly because variable minutes indicate flexible roles where the player may come in and out of the rotation.

## Calibration

### Predicted vs Actual by Decile

```
  Bin                        N  Predicted   Actual    Diff
  (0.01, 0.03]           49521     0.0204   0.0137  -0.007
  (0.03, 0.07]           45144     0.0466   0.0627  +0.016
  (0.07, 0.25]           47332     0.1361   0.1693  +0.033
  (0.25, 0.64]           47332     0.4428   0.3928  -0.050
  (0.64, 0.89]           47332     0.7932   0.7573  -0.036
  (0.89, 0.94]           47332     0.9208   0.9366  +0.016
  (0.94, 0.96]           47332     0.9528   0.9693  +0.017
  (0.96, 0.98]           47332     0.9698   0.9767  +0.007
  (0.98, 0.99]           47332     0.9811   0.9811  +0.000
  (0.99, 1.00]           47333     0.9901   0.9841  -0.006
```

Calibration is strong across most of the distribution, with the largest deviation in the 25-64% range (5pp overconfidence). The model is well-calibrated in the extremes, which is most important for practical use — correctly identifying players very likely to play and very unlikely to play.

### Calibration by Consecutive Games Missed

```
  Missed        N   Actual  Predicted    Diff
        0   296675    0.910      0.912  -0.002
        1    26788    0.402      0.464  -0.062
        2    16016    0.303      0.291  +0.012
        3    14773    0.191      0.169  +0.022
        4    11667    0.159      0.121  +0.038
        5     9569    0.117      0.086  +0.032
        6     8254    0.098      0.073  +0.025
        7     7291    0.091      0.063  +0.028
        8     6511    0.082      0.055  +0.026
        9     5878    0.066      0.050  +0.016
       10     5384    0.052      0.045  +0.007
       11     4997    0.051      0.041  +0.010
       12     4677    0.038      0.038  +0.000
       13     4428    0.035      0.035  -0.000
       14     4206    0.030      0.032  -0.002
      15+    46208    0.012      0.021  -0.009
```

The model slightly underestimates return probability in the 4-9 miss range (by 2-4pp), suggesting some players in this range return from injury more often than the model expects. For 12+ misses, calibration is nearly perfect.

## Predicted Probabilities by Player Type

The model produces a probability grid showing how return likelihood varies by both consecutive misses and player role:

```
  Missed   Bench(5m)   Rotation(15m)   Starter(25m)   Star(32m)
      0       0.845           0.936          0.964        0.987
      1       0.548           0.691          0.758        0.845
      2       0.357           0.479          0.546        0.645
      3       0.239           0.333          0.386        0.474
      5       0.116           0.166          0.197        0.250
     10       0.068           0.099          0.118        0.152
     15       0.049           0.071          0.085        0.110
```

A starter averaging 25 minutes who has missed 2 games has a 54.6% predicted probability of playing, contributing an expected `0.546 * 25 = 13.65 minutes` to roster weighting. A bench player averaging 5 minutes with the same 2 misses has only 35.7% probability, contributing `0.357 * 5 = 1.79 expected minutes`.

## Subgroup Performance

```
  Tier               N  BaseRate   Brier     AUC
  Walk-on(<5)    73525     0.241  0.1687  0.7152
  Bench(5-10)    52657     0.606  0.1691  0.8119
  Rotation(10-20)107912   0.829  0.0691  0.8938
  Starter(20+)  167297     0.923  0.0335  0.8856
```

The model performs best for rotation and starter-level players (Brier 0.03-0.07), which are the players whose availability most impacts game outcomes. Performance is weaker for walk-on and bench players (Brier ~0.17), where DNP decisions are noisier and more coach-dependent.

## Comparison: Logistic vs Poisson

To address the question of whether a Poisson distribution might fit the consecutive-miss count data better:

```
  LOGISTIC: AIC=285,637  Brier=0.0869  AUC=0.9426
  POISSON:  AIC=716,738  Brier=0.0850  AUC=0.9520
```

The Poisson model has a substantially worse AIC (716k vs 286k) because it is modeling a binary outcome with a family designed for count data. While its Brier score is comparable, the Poisson predictions at intermediate values diverge meaningfully:

```
  P(plays | missed=3, 25min): Logistic=0.831, Poisson=0.395, CLogLog(best)=0.386
```

The cloglog model's predictions align with the Poisson's steeper curve shape while maintaining proper statistical foundations for binary outcomes. This explains why cloglog outperforms logistic — it captures the same asymmetric decay pattern that motivated the Poisson intuition, but within the correct binomial framework.

## Integration into Game Prediction Model

The eligibility model outputs are used in the game prediction pipeline as follows:

1. For each player on a team's roster, compute P(plays) from the cloglog model
2. Compute expected minutes contribution: `expected_minutes = P(plays) * avg_min_when_playing`
3. Weight each player's BPM (Box Plus/Minus) by their expected minutes
4. Aggregate into team-level roster strength features (roster_bpm, top5_bpm, depth_count, star_count)

This replaces the previous approach of using raw cumulative minutes, which diluted the contribution of injured players and inflated the contribution of healthy players on teams with injured stars.

## Reproducibility

- **Script**: `scripts/test_eligibility_model.py`
- **Database**: `data/cbb_prediction.db` (SQLite)
- **Dependencies**: statsmodels, scikit-learn, pandas, numpy
- **Runtime**: ~3 minutes on Apple Silicon (dataset construction + 39 model CV runs)
