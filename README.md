# Quant Simulator (Research → Backtest → Paper (Coinbase) → Reports)

A modular quant trading research and paper-trading framework with:

Data ingestion + validation + resampling
Strategies: trend + mean reversion + pairs stub
Execution: sim broker + coinbase quote feed (paper)
Portfolio + risk constraints
Optional PyTorch ML signal layer (walk-forward)
Outputs: CSV + HTML report artifacts per run
Quickstart (Azure Bash / Linux)

cd quant-sim
python -m pip install -U pip
pip install -e .
quant-sim backtest --config configs/backtest.yaml
quant-sim paper --config configs/paper_coinbase.yaml
Artifacts are written to runs/<timestamp>_runid-.../ (csv + report.html).

## Web Services

This repository provides two web service options:

1. **Node.js service** (`index.js`) - Lightweight HTTP server for container deployment with health checks. Best for cloud platforms (Azure Container Apps, AWS ECS, etc.).
2. **Python FastAPI service** (`main.py`) - API service for Python-based integrations.

### Node.js Service (Docker + Local)

Local run:

```bash
node index.js
# or
PORT=8080 node index.js
```

Docker build + run:

```bash
docker build -t trading-quant-sim .
docker run --rm -p 8080:8080 trading-quant-sim
```

Health check:

```bash
curl -i http://localhost:8080/healthz
```
