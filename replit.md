# Quantitative Trading Simulator

## Overview
A web-based dashboard for a quantitative trading simulator that wraps a Python backtesting and paper trading framework. Users can upload market data (CSV), configure trading strategies (YAML), execute simulations (backtest or live paper trading), and view results (logs, metrics, HTML reports with equity curves).

## Project Architecture

### Frontend (React/Vite)
- `client/src/pages/Dashboard.tsx` - Main dashboard overview
- `client/src/pages/RunsList.tsx` - List of simulation runs
- `client/src/pages/RunDetail.tsx` - View run results and metrics
- `client/src/pages/NewRun.tsx` - Create new backtest or paper trading runs
- `client/src/pages/DataManagement.tsx` - Upload and manage market data CSV files
- `client/src/pages/Auth.tsx` - Sign in/sign up page
- `client/src/pages/Profile.tsx` - User profile and trading preferences
- `client/src/components/Header.tsx` - Header with user menu dropdown
- `client/src/hooks/useAuth.ts` - Authentication hook with React Query

### Backend (Express/Node.js)
- `server/routes.ts` - API routes for runs and dataset management
- `server/auth.ts` - Authentication routes and middleware (signup, signin, signout, profile)
- `server/storage.ts` - Database storage interface (PostgreSQL)
- `shared/schema.ts` - Drizzle ORM schema for users, sessions, runs, and datasets

### Python Quant Framework
- `quant-sim/src/quant_sim/` - Core simulation framework
  - `cli.py` - Main entry point (backtest, paper commands)
  - `strategies/` - Trading strategies (trend, meanrev, pairs, ensemble, ML)
  - `execution/` - Broker implementations
    - `coinbase_broker.py` - Coinbase public API paper broker
    - `ccxt_paper_broker.py` - CCXT unified exchange paper broker
    - `guards.py` - Rate limiting and disconnect guards
  - `portfolio/` - Portfolio and risk management
  - `analytics/` - Metrics and HTML report generation

## Key Commands

### Run Paper Trading (CCXT - Kraken)
```bash
cd quant-sim && PYTHONPATH=src python -m quant_sim.cli paper --config configs/paper_ccxt.yaml
```

### Run Paper Trading (CCXT - Coinbase)
```bash
cd quant-sim && PYTHONPATH=src python -m quant_sim.cli paper --config configs/paper_ccxt_coinbase.yaml
```

### Run Backtest
```bash
cd quant-sim && PYTHONPATH=src python -m quant_sim.cli backtest --config configs/backtest.yaml
```

### Run Dashboard (view run artifacts)
```bash
cd quant-sim && PYTHONPATH=src python -m quant_sim.cli dashboard --runs-dir runs --host 0.0.0.0 --port 8000
```

## Configuration Files

### paper_ccxt.yaml
Uses CCXT for unified exchange quotes:
- `exchange`: kraken, coinbase, etc. (Binance is blocked from US)
- `symbols`: CCXT format like "BTC/USD", "ETH/USD"

### Execution Guards
- `max_orders_per_min`: Rate limiter for order placement
- `max_stale_seconds`: Disconnect guard for stale quotes
- `max_failures`: Trip disconnect after N consecutive quote failures

## Recent Changes
- 2026-01-30: Added user authentication (signup, signin, signout) with bcrypt password hashing
- 2026-01-30: Added user profile page with trading preferences (exchange, currency, risk mode, leverage)
- 2026-01-30: Added user menu dropdown in header with profile/signout links
- 2026-01-29: Added CCXT paper broker for multi-exchange support
- 2026-01-29: Added execution guards (OrderRateLimiter, DisconnectGuard)
- 2026-01-29: Fixed Pandas 4.x deprecated Timestamp.utcnow()
- 2026-01-29: Changed default exchange to Kraken (Binance blocked from US)

## Database
PostgreSQL database stores run metadata and dataset references.
Schema in `shared/schema.ts`.

## Development
- Frontend: Port 5000 (Vite dev server proxied through Express)
- Backend: Same Express server
- Python: Spawned as subprocess for simulations
