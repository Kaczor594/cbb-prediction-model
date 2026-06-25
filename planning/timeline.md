# Project Timeline

This document outlines the development phases, milestones, and tasks for the college basketball prediction model.

---

## Current Progress

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Data Collection | **In Progress** | Kalshi ✓, ESPN Games ✓, ESPN BPI ✓, Barttorvik pending |
| Phase 2: Data Quality | Not Started | - |
| Phase 3: Model Development | Not Started | - |
| Phase 4: Backtesting | Not Started | - |
| Phase 5: Market Efficiency | Not Started | - |
| Phase 6: Automated Trading | Not Started | - |
| Phase 7: Dashboard | Not Started | - |

**Last Updated**: 2026-02-16

### Data Collected
- **Teams**: 365 D1 programs
- **Games**: 15,461 (3 seasons: 2023-24, 2024-25, 2025-26)
- **ESPN BPI**: 365 teams (current season only - API doesn't archive historical data)
- **Database**: SQLite at `data/cbb_prediction.db`

---

## Phase 1: Data Collection & Pipeline Setup

### Objective
Establish reliable data sources and build a clean, automated data pipeline to aggregate all required data into SQLite database.

### Tasks

#### 1.1 Kalshi API Integration ✓ COMPLETED (2024-02-16)
- [x] Enable API access on Kalshi account
- [x] Review API documentation thoroughly
- [x] Implement authentication module (RSA-PSS with SHA256)
- [x] Build market data fetching functions
- [ ] Test order placement (awaiting account funding)
- [ ] Document rate limits and error handling
- [ ] Set up historical odds collection (if available)

**Notes**:
- API client implemented at `src/api/kalshi_client.py`
- Production URL: `https://api.elections.kalshi.com/trade-api/v2`
- Private key stored securely at `~/.kalshi/private_key.pem`

#### 1.2 Game Results Data Pipeline ✓ COMPLETED (2024-02-16)
- [x] Set up ESPN API client (`src/data/espn_client.py`)
- [x] Build game results parser with extraction helpers
- [x] Create team/conference mapping tables (SQLite)
- [x] Implement historical data backfill (3 seasons collected)
- [ ] Schedule automated daily updates

**Notes**:
- Using ESPN API instead of Sports Reference (easier access, JSON format)
- Database: `data/cbb_prediction.db` (SQLite)
- Data collected:
  - 362 D1 teams
  - 15,461 games (2023-24, 2024-25, 2025-26 seasons)
- Collection script: `src/data/collect_espn_data.py`

#### 1.3 Advanced Statistics Pipeline ✓ PARTIALLY COMPLETE (2024-02-16)
- [x] Evaluate KenPom vs Barttorvik vs ESPN BPI vs TeamRankings
- [x] Build ESPN BPI pipeline (`src/data/espn_bpi.py`)
- [x] Build Barttorvik import script (`src/data/import_barttorvik.py`)
- [ ] Import Barttorvik data (manual CSV export needed)
- [x] Map team IDs across sources

**Notes**:
- ESPN BPI: Full API available (365 teams) via FITT endpoint - current season only
- Barttorvik: Covers all teams, requires manual CSV export (bot protection) - has historical data
- Recommendation: Use ESPN BPI for current season metrics, Barttorvik for historical analysis

#### 1.4 Player Statistics Pipeline
- [ ] Design player statistics schema
- [ ] Implement per-game stats collection
- [ ] Build recruiting data scraper (247Sports)
- [ ] Link players to teams by season

#### 1.5 Supplementary Data
- [ ] Build injury report collection
- [ ] Create geographical database
- [ ] Add coaching data

#### 1.6 Database Design
- [ ] Design SQLite schema for all tables
- [ ] Implement data models
- [ ] Create data validation layer
- [ ] Build data refresh orchestration

### Deliverables
- Working SQLite database with all core data
- Automated data refresh scripts
- Data documentation

---

## Phase 2: Data Quality & Validation

### Objective
Identify missing or unreliable data sources. Document which data points are unreliable and establish data quality standards.

### Tasks

#### 2.1 Data Completeness Audit
- [ ] Analyze missing data by source
- [ ] Document coverage gaps (e.g., missing games, players)
- [ ] Identify data that's consistently late/unreliable
- [ ] Create data quality dashboard

#### 2.2 Data Consistency Checks
- [ ] Cross-validate game results across sources
- [ ] Verify statistical calculations
- [ ] Check for duplicate records
- [ ] Validate team/player ID mappings

#### 2.3 Reliability Documentation
- [ ] Rate each data source by reliability
- [ ] Document refresh frequencies
- [ ] Note seasonal availability (recruiting vs in-season)
- [ ] Plan fallback sources for critical data

#### 2.4 Data Cleaning Pipeline
- [ ] Build automated cleaning scripts
- [ ] Handle missing value imputation strategy
- [ ] Create outlier detection
- [ ] Implement data versioning

### Deliverables
- Data quality report
- Documented reliability ratings
- Automated cleaning pipeline
- Missing data strategy document

---

## Phase 3: Model Development

### Objective
Build and train machine learning model optimizing on win probability prediction.

### Tasks

#### 3.1 Feature Engineering
- [ ] Define feature set based on available data
- [ ] Create team-level features (efficiency metrics, record, etc.)
- [ ] Build player-aggregated features
- [ ] Engineer matchup-specific features
- [ ] Create temporal features (rest days, travel, etc.)
- [ ] Handle categorical encoding (conferences, venues)

#### 3.2 Training Data Preparation
- [ ] Define training/validation/test splits
- [ ] Handle class imbalance (if any)
- [ ] Implement time-aware cross-validation
- [ ] Create feature scaling pipeline

#### 3.3 Model Selection & Training
- [ ] Baseline model (logistic regression)
- [ ] Random Forest implementation
- [ ] Gradient Boosting (XGBoost/LightGBM)
- [ ] Ensemble methods exploration
- [ ] Hyperparameter tuning (GridSearch/Optuna)

#### 3.4 Model Evaluation
- [ ] Implement evaluation metrics (accuracy, log-loss, Brier score)
- [ ] Calibration analysis
- [ ] Feature importance analysis
- [ ] Error analysis by game type

#### 3.5 Model Persistence
- [ ] Save trained models
- [ ] Version control for models
- [ ] Create prediction pipeline

### Deliverables
- Trained prediction model
- Feature importance documentation
- Model evaluation report
- Prediction API/module

---

## Phase 4: Backtesting & Monte Carlo Simulation

### Objective
Run Monte Carlo simulations to backtest model and optimize based on results from past 3 years of games.

### Tasks

#### 4.1 Backtesting Framework
- [ ] Build game-by-game backtesting engine
- [ ] Implement walk-forward validation
- [ ] Create seasonal simulation framework
- [ ] Track predictions vs actuals

#### 4.2 Monte Carlo Simulation
- [ ] Define simulation parameters
- [ ] Implement probability-weighted outcome generation
- [ ] Run 1000+ simulations per scenario
- [ ] Calculate confidence intervals

#### 4.3 Performance Analysis
- [ ] Analyze model performance by:
  - Conference
  - Game type (conference vs non-conference)
  - Favorite vs underdog
  - Point spread ranges
  - Time of season
- [ ] Identify systematic biases
- [ ] Document performance degradation patterns

#### 4.4 Model Optimization
- [ ] Iterate on features based on backtest results
- [ ] Retrain with optimized parameters
- [ ] Validate improvements don't overfit

### Deliverables
- Backtesting framework
- Monte Carlo simulation results
- Performance breakdown analysis
- Optimized model version

---

## Phase 5: Market Efficiency Analysis

### Objective
Test model against historical Kalshi market odds to identify market inefficiencies and develop betting strategies.

### Tasks

#### 5.1 Historical Odds Collection
- [ ] Gather historical Kalshi odds (if available)
- [ ] Alternatively, use opening/closing lines from other sources
- [ ] Map odds to game results database

#### 5.2 Edge Calculation
- [ ] Calculate model probability vs market implied probability
- [ ] Define "edge" thresholds
- [ ] Analyze edge distribution

#### 5.3 Strategy Development
- [ ] Define positive EV betting criteria
- [ ] Develop Kelly Criterion position sizing
- [ ] Create risk management rules
- [ ] Define stop-loss and take-profit strategies

#### 5.4 Strategy Backtesting
- [ ] Simulate betting with historical data
- [ ] Calculate returns by strategy
- [ ] Analyze drawdowns
- [ ] Stress test with different bankroll sizes

#### 5.5 Negative Strategy Analysis
- [ ] Identify consistently unprofitable patterns
- [ ] Document scenarios to avoid
- [ ] Create betting exclusion rules

### Deliverables
- Market efficiency analysis report
- Documented betting strategies
- Backtest results with P&L
- Risk management framework

---

## Phase 6: Automated Trading System

### Objective
Build logic for automated buying and selling of game odds on Kalshi.

### Tasks

#### 6.1 Trading Engine Core
- [ ] Build order management system
- [ ] Implement position tracking
- [ ] Create order execution module
- [ ] Handle API errors and retries

#### 6.2 Strategy Implementation
- [ ] Implement betting strategies from Phase 5
- [ ] Build position sizing calculator
- [ ] Create entry/exit signal generator
- [ ] Implement risk checks before order submission

#### 6.3 Safety Features
- [ ] Daily loss limits
- [ ] Maximum position sizes
- [ ] Concentration limits
- [ ] Kill switch functionality
- [ ] Notification system for errors

#### 6.4 Testing
- [ ] Paper trading mode
- [ ] Integration testing with Kalshi sandbox (if available)
- [ ] Gradual rollout with small stakes

### Deliverables
- Automated trading system
- Paper trading results
- Safety documentation
- Operations runbook

---

## Phase 7: Monitoring Dashboard

### Objective
Build monitoring dashboard using Plotly Dash to track model performance and betting results independently.

### Tasks

#### 7.1 Dashboard Architecture
- [ ] Design dashboard layout
- [ ] Set up Plotly Dash application
- [ ] Create database connections
- [ ] Implement real-time data refresh

#### 7.2 Model Performance Views
- [ ] Prediction accuracy over time
- [ ] Calibration plots
- [ ] Feature importance trends
- [ ] Model version comparison

#### 7.3 Betting Performance Views
- [ ] P&L tracking (daily, weekly, monthly, cumulative)
- [ ] ROI by bet type/strategy
- [ ] Win rate vs expected win rate
- [ ] Drawdown visualization
- [ ] Bankroll chart

#### 7.4 Operational Views
- [ ] Active positions
- [ ] Pending bets
- [ ] Recent trade history
- [ ] System health/errors
- [ ] Data pipeline status

#### 7.5 Alerts & Notifications
- [ ] Threshold-based alerts
- [ ] Email/Slack notifications
- [ ] Daily summary reports

### Deliverables
- Fully functional Plotly Dash dashboard
- Documentation for dashboard usage
- Alert configuration

---

## Milestone Summary

| Phase | Milestone | Dependencies |
|-------|-----------|--------------|
| 1 | Data pipeline operational | None |
| 2 | Data quality validated | Phase 1 |
| 3 | Model trained and evaluated | Phase 2 |
| 4 | Backtesting complete | Phase 3 |
| 5 | Strategies defined and tested | Phase 4 |
| 6 | Automated trading live | Phase 5 |
| 7 | Dashboard deployed | Phase 6 |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Kalshi API changes | High | Monitor API changelog, build abstraction layer |
| Data source becomes unavailable | High | Multiple sources per data type |
| Model overfitting | Medium | Robust cross-validation, out-of-sample testing |
| Market efficiency eliminates edge | Medium | Continuous model improvement, alternative strategies |
| API rate limiting | Low | Implement backoff, cache data |
| Regulatory changes | Medium | Stay informed, build flexible system |

---

## Notes

- Prioritize getting a basic end-to-end pipeline working before optimization
- Start with simpler models and add complexity as needed
- Document learnings throughout each phase
- Regular code reviews and testing
