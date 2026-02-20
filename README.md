# TradingQuant-Sim

A full-stack quantitative trading simulator with research, backtesting, and paper trading capabilities.

## Components

### Quant-Sim (Python Trading Engine)
The core quantitative trading simulator located in `/quant-sim`. Features include:
- Data ingestion, validation, and resampling
- Multiple strategies: trend following, mean reversion, pairs trading
- Execution engines: simulated broker and Coinbase paper trading
- Portfolio management with risk constraints
- Optional PyTorch ML signal layer with walk-forward validation
- Comprehensive reporting: CSV exports and HTML reports

See [quant-sim/README.md](quant-sim/README.md) for detailed setup and usage instructions.

### Web Application (TypeScript/React)
Full-stack web application for managing and monitoring trading operations:
- **Client**: React-based frontend with Tailwind CSS
- **Server**: Express.js backend with TypeScript
- **Database**: PostgreSQL with Drizzle ORM

## Quick Start

### Python Trading Engine
```bash
cd quant-sim
python -m pip install -U pip
pip install -e .
quant-sim backtest --config configs/backtest.yaml
quant-sim paper --config configs/paper_coinbase.yaml
```

### Web Application
```bash
npm install
npm run dev        # Development mode
npm run build      # Build for production
npm start          # Run production build
```

## Structure

```
.
├── quant-sim/          # Python trading simulator
│   ├── configs/        # Configuration files
│   ├── src/            # Source code
│   ├── tests/          # Test suite
│   └── runs/           # Output artifacts
├── client/             # React frontend
├── server/             # Express backend
├── shared/             # Shared TypeScript types
└── script/             # Build scripts
```

## License

MIT
