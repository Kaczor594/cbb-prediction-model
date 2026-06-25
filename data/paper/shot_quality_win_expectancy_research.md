# Post-Game Win Expectancy / Expected Shot Value Research

> Date: 2026-04-06

## Concept

A post-game win expectancy metric analogous to **xG (Expected Goals) in soccer**: given all shots taken, possessions, turnovers, offensive rebounds, etc. — but **ignoring which shots were actually made or missed** — what is the probability each team should have won? This strips out shot-making luck to evaluate the process of a game rather than the outcome.

---

## ShotQuality (shotquality.com) — The Gold Standard

ShotQuality is the most complete implementation of this concept for college basketball. They use **computer vision on broadcast TV footage** to extract player positions and compute shot-level expected values.

### Model Features (~100 variables across 5 categories)

| Category | Features | Publicly Available? |
|---|---|---|
| **Defensive Distance** | Nearest defender distance, positioning relative to basket, lane crowding, number of defenders contesting | No — proprietary CV data |
| **Shooter Ability** | Player shooting history by shot type, recency-weighted performance, league-average benchmarks | Partially (FG% by zone exists publicly) |
| **Play Type** | Pick-and-roll, transition, cuts, post-ups, isolation, etc. | No — only via Synergy Sports (paid) |
| **Shot Type** | Jump shot vs layup vs dunk, catch-and-shoot vs off-dribble, contested vs open | Partially (basic type from ESPN PBP text) |
| **Priors** | Player height/position, team shooting culture, conference strength | Reconstructable from public data |

### Data Coverage

- **137,000+ games since 2011** (NBA + NCAA + international combined)
- **19M+ shots** in dataset
- Player x/y coordinates for all 10 players on court
- Post-game **ShotQuality Scores** (expected scores for each team) for every game
- Shot-level expected probabilities, defender distance, shot type data

### Subscription Pricing

| Plan | Monthly | Annual | Key Features |
|---|---|---|---|
| **Standard** | $49.99/mo | $299.99/yr | Daily predictions, select live access, historical scores |
| **Premier** | $249.99/mo | $1,499.99/yr | All games live, **data export**, full API access |

- **Premier is required** for data export and API access needed to pull historical data in bulk
- Enterprise/API pricing may be negotiated separately — contact hello@shotquality.com
- No free trial (free pick of the day only)
- Promo codes occasionally available (e.g., "BETSMART" for 25% off)

### Key Claimed Performance

- Defender/location data improves NCAA-to-NBA 3PT prediction R-squared by **54%**
- Midrange prediction R-squared improves by **84%**
- Rim finishing prediction R-squared improves by **53%**
- Claimed 53.5% win rate as NCAA handicapper

---

## DIY Alternative: What Public Data Exists

### Available Shot-Level Features (Free)

| Feature | Source |
|---|---|
| Shot x,y coordinates | CBBD API, ncaahoopR, hoopR/ESPN |
| Shot distance (derived from x,y) | Calculated |
| Shot type (layup/jumper/dunk/hook) | ESPN `type_text` field via hoopR |
| Made/missed | All sources |
| 2PT vs 3PT | All sources |
| Assisted vs. unassisted | CBBD, ncaahoopR |
| Player identity + historical shooting % | Sports-Reference, Barttorvik |
| Game clock / period | All sources |
| Team offensive/defensive ratings | Barttorvik, KenPom |

### NOT Available Publicly for College Basketball

- Nearest defender distance
- Number of defenders in area / lane crowding
- Defender positioning relative to basket
- Court spacing
- Catch-and-shoot vs. off-dribble classification
- Detailed play type at scale (PnR, isolation, transition, cut)
- Contest level (open / contested / tightly contested)

### Data Sources for Building a DIY Model

1. **CBBD API** (collegebasketballdata.com) — Shot locations with x,y coordinates, shot type, assisted/unassisted. Free tier: 1,000 API calls/month. Python package: `cbbd`. This is what Jacob Pickle used.
2. **ncaahoopR** (github.com/lbenz730/ncaahoopR) — R package scraping ESPN PBP with shot locations. Court is 94x50 ft, origin at center court. Caveat: shot locations only available for high-major teams.
3. **hoopR** (hoopr.sportsdataverse.org) — R package for ESPN PBP data with shot coordinates and `type_text` field. ~2.9M rows of college basketball PBP data.
4. **ESPN Hidden API** — Public, no API key required. ncaahoopR and hoopR wrap these endpoints.

### Jacob Pickle's xP Model — Existing DIY Implementation

- Built by @pickleo7 on X/Twitter (July 2025)
- LightGBM model trained on ~1.7M shots from CBBD
- Features: shot distance and angle (location-based only, no defender tracking)
- Produces per-shot expected points, game-level expected scores
- Includes a "Deserve to Win O Meter" (Monte Carlo simulation of xP outcomes)
- College basketball specific, publicly shared on social media

### Impact of the Data Gap

Multiple studies quantify how much defender distance matters:
- Stanford CS229 study: shot distance + defender distance alone achieves ~64% accuracy for NBA shot prediction
- USPROC study: adding defender distance improved explained variance by 13.2%
- CMU Capstone: defender distance is #1 feature by Shapley value, exceeding shot distance
- Sport Journal: moving defender from 3.9 ft to 9+ ft raises 3PT probability by ~4pp

**Bottom line**: A DIY model is meaningful but has a lower ceiling. The gap is largest for mid-range and rim shots (where contest varies enormously) and smallest for open catch-and-shoot threes.

### Mitigation Strategies for DIY (No Defender Data)

- Use ESPN's `type_text` to infer contest (e.g., "Driving Layup" vs "Pullup Jump Shot")
- Use assisted vs. unassisted as proxy for openness
- Use team defensive ratings as prior for contest quality
- Identify transition plays from PBP timing (tend to produce more open looks)
- Use shot zone + player position combinations as features

---

## Other Related Metrics (Less Directly Relevant)

### KenPom "Luck" (kenpom.com)
- Deviation between actual W-L and expected W-L from game-by-game efficiencies
- **Not** shot-quality-based — uses aggregate scoring efficiency, not individual shot expected values
- Partially free (team-level luck ratings visible), full data $20/yr

### Barttorvik "Luck" (barttorvik.com)
- Same concept as KenPom luck — efficiency-based, not shot-quality-based
- Free on website; accessible via `toRvik` R package

### PBPStats Shot Quality Model (pbpstats.com)
- Expected eFG% per shot based on location and PBP context
- **NBA/WNBA/G-League only — no college basketball**
- Free, open-source Python package

---

## Sources

- [ShotQualityBets Build Page](https://shotqualitybets.com/build)
- [ShotQuality API](https://shotquality.com/api)
- [ShotQuality Stats Explained](https://shotquality.com/stats-explained)
- [ShotQuality Shot Probability Whitepaper](https://info.shotquality.com/hubfs/Shot%20Probability%20Model%20Whitepaper.pdf)
- [Q&A Session with ShotQuality — Huddle](https://huddle.tech/qa-session-with-shotquality/)
- [ShotQualityBets Review — SportBot AI](https://www.sportbotai.com/blog/tools/shotqualitybets-review)
- [ShotQuality Bets Review — BetSmart](https://www.betsmart.co/tool-reviews/shotquality-bets)
- [ShotQuality Advanced NBA Analytics — RotoGrinders](https://rotogrinders.com/shotquality)
- [NCAA ScoreCenter — ShotQualityBets](https://shotqualitybets.com/ncaa/shotquality-scores)
- [CBBD API](https://collegebasketballdata.com/)
- [ncaahoopR — GitHub](https://github.com/lbenz730/ncaahoopR)
- [hoopR — Sportsdataverse](https://hoopr.sportsdataverse.org/)
- [Jacob Pickle xP Model — @pickleo7 on X](https://x.com/pickleo7)
- [FiveThirtyEight Historical NCAA Forecasts — GitHub](https://github.com/fivethirtyeight/data/tree/master/historical-ncaa-forecasts)
- [Kaggle March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
