# Finance Tracking

This directory contains financial management and tracking for the college basketball prediction model betting activities.

## Overview

All financial data is stored in the main SQLite database (`data/cbb_prediction.db`) for easy querying and analysis. This README documents the schema and usage.

## Database Schema

### Tables

#### `bankroll`
Tracks overall bankroll and capital movements.

```sql
CREATE TABLE bankroll (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    transaction_type TEXT NOT NULL,  -- 'deposit', 'withdrawal', 'adjustment'
    amount DECIMAL(10, 2) NOT NULL,
    balance_after DECIMAL(10, 2) NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `bets`
Records all bets placed.

```sql
CREATE TABLE bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kalshi_order_id TEXT,
    game_id INTEGER REFERENCES games(id),
    placed_at TIMESTAMP NOT NULL,
    market_ticker TEXT NOT NULL,
    bet_type TEXT NOT NULL,  -- 'yes', 'no'
    contracts INTEGER NOT NULL,
    price DECIMAL(5, 2) NOT NULL,  -- Price per contract (0.01 to 0.99)
    total_cost DECIMAL(10, 2) NOT NULL,
    model_probability DECIMAL(5, 4),
    market_probability DECIMAL(5, 4),
    edge DECIMAL(5, 4),  -- model_prob - market_prob
    strategy TEXT,  -- Which strategy triggered this bet
    status TEXT DEFAULT 'open',  -- 'open', 'won', 'lost', 'sold', 'cancelled'
    settled_at TIMESTAMP,
    payout DECIMAL(10, 2),
    profit_loss DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `daily_summary`
Daily aggregated performance metrics.

```sql
CREATE TABLE daily_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE UNIQUE NOT NULL,
    starting_bankroll DECIMAL(10, 2),
    ending_bankroll DECIMAL(10, 2),
    bets_placed INTEGER DEFAULT 0,
    bets_settled INTEGER DEFAULT 0,
    bets_won INTEGER DEFAULT 0,
    bets_lost INTEGER DEFAULT 0,
    gross_profit DECIMAL(10, 2) DEFAULT 0,
    gross_loss DECIMAL(10, 2) DEFAULT 0,
    net_pnl DECIMAL(10, 2) DEFAULT 0,
    roi_percent DECIMAL(8, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `strategy_performance`
Track performance by betting strategy.

```sql
CREATE TABLE strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_bets INTEGER,
    wins INTEGER,
    losses INTEGER,
    win_rate DECIMAL(5, 4),
    total_wagered DECIMAL(10, 2),
    total_pnl DECIMAL(10, 2),
    roi_percent DECIMAL(8, 4),
    avg_edge DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Key Metrics

### Performance Metrics
- **ROI (Return on Investment)**: Net P&L / Total Wagered
- **Win Rate**: Wins / Total Settled Bets
- **Edge**: Model Probability - Market Implied Probability
- **Expected Value (EV)**: (Win Probability × Potential Win) - (Loss Probability × Stake)

### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline in bankroll
- **Sharpe Ratio**: Risk-adjusted return measure
- **Kelly Fraction**: Optimal bet sizing based on edge and odds

## Bankroll Management Rules

### Position Sizing
- **Maximum single bet**: X% of bankroll (configurable)
- **Kelly Criterion**: Bet fraction = (bp - q) / b
  - b = odds received on the bet
  - p = probability of winning
  - q = probability of losing (1 - p)
- **Fractional Kelly**: Use 25-50% of full Kelly to reduce variance

### Risk Limits
- **Daily loss limit**: Maximum daily loss before stopping
- **Weekly loss limit**: Maximum weekly loss
- **Maximum exposure**: Total capital at risk at any time

## Reports

### Standard Reports
1. **Daily P&L Report**: End-of-day summary
2. **Weekly Performance**: Week-over-week comparison
3. **Strategy Analysis**: Performance breakdown by strategy
4. **Model Calibration**: Predicted vs actual win rates

### Export Formats
- CSV exports for external analysis
- JSON for API consumption
- Dashboard visualizations (Plotly Dash)

## Usage

### Initialize Finance Tables
```python
from src.data.database import initialize_finance_tables
initialize_finance_tables()
```

### Record a Deposit
```python
from src.finance.bankroll import record_deposit
record_deposit(amount=1000.00, notes="Initial funding")
```

### Log a Bet
```python
from src.finance.betting import log_bet
log_bet(
    game_id=12345,
    market_ticker="NCAABB-DUKE-UNC-2024",
    bet_type="yes",
    contracts=10,
    price=0.55,
    model_probability=0.62,
    strategy="high_edge"
)
```

### Generate Daily Summary
```python
from src.finance.reports import generate_daily_summary
generate_daily_summary(date="2024-03-15")
```

## Files in This Directory

```
finance/
├── README.md           # This file
├── exports/            # CSV/JSON exports (gitignored)
└── reports/            # Generated report files (gitignored)
```

## Notes

- All monetary values stored in USD
- Timestamps in UTC
- Kalshi fees should be factored into cost calculations
- Regular backups of financial data recommended
