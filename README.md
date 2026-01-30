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
