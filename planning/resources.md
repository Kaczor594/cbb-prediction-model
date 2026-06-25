# Resources & Data Sources

This document tracks all resources, APIs, and data sources used in the college basketball prediction model.

## Table of Contents
- [Kalshi API](#kalshi-api)
- [Player Data Sources](#player-data-sources)
- [Game & Statistical Data](#game--statistical-data)
- [Supplementary Data](#supplementary-data)
- [Reference Materials](#reference-materials)

---

## Kalshi API

### Overview
Kalshi is a regulated exchange for event contracts. Their API allows programmatic access to markets, including college basketball game outcomes.

### Setup Requirements
- [x] Create Kalshi account ✓
- [x] Enable API access in account settings ✓
- [x] Generate API keys (RSA key pair) ✓
- [x] Store credentials securely (private key at `~/.kalshi/private_key.pem`) ✓

### API Documentation
- **Official Docs**: https://docs.kalshi.com/
- **API Base URL (Production)**: `https://api.elections.kalshi.com/trade-api/v2`
- **API Base URL (Demo)**: `https://demo-api.kalshi.co/trade-api/v2`
- **Authentication**: RSA-PSS signature with SHA256

### Implementation Status
- [x] API client implemented (`src/api/kalshi_client.py`)
- [x] RSA-PSS authentication working
- [x] Connection tested successfully (2024-02-16)
- [ ] Order placement (awaiting account funding)
- [ ] Historical odds collection

### Key Endpoints
| Endpoint | Purpose | Status |
|----------|---------|--------|
| `GET /exchange/status` | Check exchange status | ✓ Implemented |
| `GET /portfolio/balance` | Get account balance | ✓ Implemented |
| `GET /portfolio/positions` | View current positions | ✓ Implemented |
| `GET /markets` | List available markets | ✓ Implemented |
| `GET /markets/{ticker}` | Get specific market details | ✓ Implemented |
| `GET /events` | List events | ✓ Implemented |
| `POST /orders` | Place orders | Pending |

### Historical Data Availability
- [ ] **TODO**: Investigate if Kalshi provides historical odds data via API
- [ ] **TODO**: Check if third-party sources archive Kalshi odds
- Alternative: Start collecting odds data now for future backtesting

### Rate Limits
- [ ] **TODO**: Document rate limits from API docs

---

## Player Data Sources

### Recruiting Data (Star Ratings)

#### 247Sports Composite
- **URL**: https://247sports.com/Season/2025-Basketball/CompositeRecruitRankings/
- **Data Available**:
  - Star ratings (5-star to 2-star)
  - National ranking
  - Position ranking
  - State ranking
  - Committed school
- **Access Method**: Web scraping required (no public API)
- **Historical Data**: Available back several years
- **Update Frequency**: Throughout recruiting cycle

#### Rivals.com
- **URL**: https://n.rivals.com/prospect_rankings/rivals100
- **Data Available**: Similar to 247Sports
- **Access Method**: Web scraping
- **Notes**: Good for cross-referencing 247Sports data

#### ESPN Recruiting
- **URL**: https://www.espn.com/college-sports/basketball/recruiting/playerrankings
- **Data Available**: ESPN 100 rankings
- **Access Method**: Web scraping or ESPN API (limited)

### Per-Game Performance Statistics

#### Sports Reference (College Basketball Reference)
- **URL**: https://www.sports-reference.com/cbb/
- **Data Available**:
  - Per-game stats (points, rebounds, assists, etc.)
  - Advanced stats (PER, WS, BPM, etc.)
  - Game logs
  - Season totals
- **Access Method**: Web scraping (respect robots.txt)
- **Historical Data**: Extensive historical archive
- **Status**: [ ] Investigate scraping feasibility

#### NCAA Official Statistics
- **URL**: https://stats.ncaa.org/
- **Data Available**: Official NCAA statistics
- **Access Method**: Web scraping or data downloads
- **Notes**: Most authoritative source

#### KenPom
- **URL**: https://kenpom.com/
- **Data Available**:
  - Advanced team ratings
  - Efficiency metrics
  - Tempo data
  - Player-level stats (with subscription)
- **Access Method**: Subscription required (~$25/year)
- **Status**: [ ] Consider subscription for advanced metrics

---

## Game & Statistical Data

### Game Results & Basic Stats

#### Sports Reference (College Basketball Reference)
- **URL**: https://www.sports-reference.com/cbb/
- **Data Available**:
  - Game results and scores
  - Box scores
  - Team statistics
  - Conference standings
- **Historical Data**: Back to 1890s for some data

#### ESPN API ✓ IMPLEMENTED
- **URL**: https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/
- **Data Available**:
  - Live scores
  - Team information
  - Schedule data
  - Game results with scores
  - Rankings
  - Betting lines (spread, over/under)
- **Access Method**: Undocumented API (JSON)
- **Status**: ✓ Implemented (`src/data/espn_client.py`)
- **Data Collected**: 15,461 games across 3 seasons

### Advanced Statistics

#### KenPom
- **Priority**: HIGH
- **Metrics**:
  - Adjusted Offensive/Defensive Efficiency
  - Tempo
  - Four Factors
  - Luck rating
  - Strength of Schedule
- **Status**: [ ] Subscription needed

#### Barttorvik
- **URL**: https://barttorvik.com/
- **Data Available**: Similar to KenPom, some free access
- **Notes**: Good alternative/supplement to KenPom

#### EvanMiya
- **URL**: https://evanmiya.com/
- **Data Available**:
  - Player ratings (BPR)
  - Team efficiency metrics
  - Game predictions
- **Notes**: Strong player-level analytics

### Geographical Data

#### Team Locations
- **Source**: Manual compilation or Sports Reference
- **Data Needed**:
  - Arena locations (lat/long)
  - Travel distances between schools
  - Time zones
- **Status**: [ ] Build team location database

#### Game Venue Data
- **Source**: Sports Reference game logs
- **Data Needed**:
  - Home/Away/Neutral designation
  - Actual venue for each game
  - Attendance (if available)

### Injury Data

#### CBS Sports Injuries
- **URL**: https://www.cbssports.com/college-basketball/injuries/
- **Update Frequency**: Daily during season
- **Access Method**: Web scraping

#### ESPN Injury Reports
- **URL**: ESPN team pages
- **Notes**: Less comprehensive than professional sports

#### DonBest / RotoWire
- **Notes**: May require subscription
- **Status**: [ ] Investigate options

#### Twitter/X Monitoring
- **Strategy**: Follow beat writers for major programs
- **Tools**: Twitter API or manual monitoring
- **Notes**: Often first source for injury news

---

## Supplementary Data

### Schedule & Conference Data
- **Source**: NCAA or Sports Reference
- **Data Needed**:
  - Conference affiliations
  - Non-conference schedule
  - Tournament seeding (historical)

### Coaching Data
- **Source**: Sports Reference
- **Data Needed**:
  - Head coach for each team/season
  - Coaching experience
  - Historical winning percentages

### Weather Data (for travel analysis)
- **Source**: Weather API (if needed)
- **Use Case**: Travel fatigue modeling

---

## Reference Materials

### Academic Papers
- [ ] Survey literature on sports prediction models
- [ ] Research on market efficiency in sports betting

### Books
- [ ] "Mathletics" by Wayne Winston
- [ ] "Scorecasting" by Moskowitz & Wertheim

### Online Resources
- Kaggle NCAA tournament datasets
- Reddit r/CollegeBasketball for qualitative insights

---

## Data Collection Priority

### Phase 1 (Essential)
1. ~~Kalshi API connection~~ ✓ COMPLETED
2. ~~Game results & basic stats~~ ✓ COMPLETED (ESPN API)
3. ~~Team efficiency metrics~~ ✓ COMPLETED (ESPN BPI - 365 teams, current season)

### Phase 2 (Important)
4. Player statistics
5. Recruiting data
6. Injury reports

### Phase 3 (Enhancement)
7. Geographical/travel data
8. Coaching data
9. Advanced player metrics

---

## Notes & TODOs

- [ ] Set up web scraping infrastructure with proper rate limiting
- [ ] Create data validation checks for each source
- [ ] Build data freshness monitoring
- [ ] Document data licensing/terms of service for each source
- [ ] Set up automated data collection pipelines
