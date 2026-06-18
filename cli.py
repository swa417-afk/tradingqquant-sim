from __future__ import annotations
import argparse
import os
import time
from typing import Dict, Optional, Tuple
import pandas as pd

from quant_sim.config import load_yaml, Config
from quant_sim.run_artifacts import make_run_dir, snapshot_config, save_json
from quant_sim.utils.logging import setup_logger

from quant_sim.data.provider import load_csv_ohlcv
from quant_sim.data.resample import resample_ohlcv

from quant_sim.strategies.trend import TrendStrategy
from quant_sim.strategies.meanrev import MeanReversionStrategy
from quant_sim.strategies.pairs import PairsStrategy
from quant_sim.strategies.ensemble import EnsembleStrategy
from quant_sim.strategies.ml_pytorch import PyTorchMLSignal

from quant_sim.execution.orders import Order
from quant_sim.execution.sim_broker import SimBroker
from quant_sim.execution.coinbase_broker import CoinbasePaperBroker
from quant_sim.execution.ccxt_paper_broker import CCXTPaperBroker
from quant_sim.execution.guards import OrderRateLimiter, DisconnectGuard

from quant_sim.portfolio.portfolio import Portfolio
from quant_sim.portfolio.risk import RiskLimits, clamp_target, check_daily_loss
from quant_sim.portfolio.sizing import target_pct_to_qty

from quant_sim.analytics.metrics import compute_metrics
from quant_sim.analytics.report_csv import write_csv
from quant_sim.analytics.report_html import write_html_report

STRAT_REGISTRY = {
    "trend": TrendStrategy,
    "meanrev": MeanReversionStrategy,
    "pairs": PairsStrategy,
    "ensemble": EnsembleStrategy,
}

def _build_strategy(cfg: Config):
    name = str(cfg.get("strategy", "name", default="trend")).lower()
    params = cfg.get("strategy", "params", default={}) or {}
    if name == "ensemble":
        return EnsembleStrategy(params, registry=STRAT_REGISTRY)
    if name not in STRAT_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Options: {list(STRAT_REGISTRY.keys())}")
    return STRAT_REGISTRY[name](params)

def _risk_limits(cfg: Config) -> RiskLimits:
    r = cfg.get("risk", default={}) or {}
    return RiskLimits(
        max_gross_leverage=float(r.get("max_gross_leverage", 2.0)),
        max_position_pct=float(r.get("max_position_pct", 0.25)),
        max_daily_loss_pct=float(r.get("max_daily_loss_pct", 0.05)),
        kill_switch=bool(r.get("kill_switch", True)),
    )

def _prices_from_bars(bars: Dict[str, pd.DataFrame], ts: pd.Timestamp) -> Dict[str, float]:
    out = {}
    for sym, df in bars.items():
        s = df.loc[:ts, "close"]
        if len(s) > 0:
            out[sym] = float(s.iloc[-1])
    return out

def _align_bars_to_timeframe(bars_1m: Dict[str, pd.DataFrame], tf: str) -> Dict[str, pd.DataFrame]:
    if tf.lower() == "1m":
        return bars_1m
    return {sym: resample_ohlcv(df, tf) for sym, df in bars_1m.items()}

def run_backtest(config_path: str, ml_path_override: Optional[str] = None) -> str:
    raw = load_yaml(config_path)
    cfg = Config(raw)

    run_name = str(cfg.get("run", "name", default="run"))
    out_dir = str(cfg.get("run", "out_dir", default="runs"))
    artifacts = make_run_dir(out_dir, run_name)
    snapshot_config(config_path, artifacts.run_dir)
    logger = setup_logger("quant_sim", artifacts.logs_path)
    logger.info(f"Starting backtest: {run_name} -> {artifacts.run_dir}")

    # data
    csv_files = cfg.get("data", "csv_files", default=[]) or []
    if not csv_files:
        raise ValueError("No CSV files configured. Set data.csv_files in config.")
    bars_1m = load_csv_ohlcv(
        csv_files=csv_files,
        timestamp_col=str(cfg.get("data", "timestamp_col", default="timestamp")),
        tz=str(cfg.get("data", "tz", default="UTC")),
    )
    universe = cfg.get("universe", "symbols", default=list(bars_1m.keys())) or list(bars_1m.keys())
    bars_1m = {sym: bars_1m[sym] for sym in universe if sym in bars_1m}

    decision_tf = str(cfg.get("strategy", "timeframe", default="5m"))
    bars = _align_bars_to_timeframe(bars_1m, decision_tf)

    # broker
    commission_bps = float(cfg.get("execution", "commission_bps", default=2.0))
    slippage_bps = float(cfg.get("execution", "slippage_bps", default=1.0))
    broker = SimBroker(bars_1m, commission_bps=commission_bps, slippage_bps=slippage_bps)

    # portfolio
    starting_cash = float(cfg.get("portfolio", "starting_cash", default=100000.0))
    pf = Portfolio(cash=starting_cash)
    limits = _risk_limits(cfg)

    # ML
    ml_enabled = bool(cfg.get("ml", "enabled", default=False))
    ml_cfg_path = ml_path_override or cfg.get("ml", "config", default=None)
    ml_models: Dict[str, PyTorchMLSignal] = {}
    ml_preds: Dict[str, pd.Series] = {}

    if ml_enabled:
        if not ml_cfg_path:
            raise ValueError("ml.enabled is true but no ml.config provided.")
        ml_raw = load_yaml(str(ml_cfg_path))
        logger.info(f"ML enabled. Using config: {ml_cfg_path}")
        for sym, df1m in bars_1m.items():
            sig = PyTorchMLSignal(ml_raw)
            preds = sig.walk_forward_predict(df1m)
            ml_models[sym] = sig
            ml_preds[sym] = preds
        logger.info("ML walk-forward predictions generated.")

    strat = _build_strategy(cfg)

    # iterate
    common_index = None
    for sym, df in bars.items():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    if common_index is None or len(common_index) < 5:
        raise ValueError("Not enough aligned bars to run backtest.")

    trades = []
    equity_start_day = starting_cash
    day_anchor = pd.to_datetime(common_index[0]).date()

    for ts in common_index:
        if pd.to_datetime(ts).date() != day_anchor:
            last_eq = pf.equity_curve[-1]["equity"] if pf.equity_curve else starting_cash
            equity_start_day = float(last_eq)
            day_anchor = pd.to_datetime(ts).date()

        prices = _prices_from_bars(bars_1m, ts)
        row = pf.mark_to_market(prices, ts)

        if limits.kill_switch and check_daily_loss(equity_start_day, row["equity"], limits):
            logger.warning("Kill-switch triggered: max daily loss reached. Flattening positions and stopping.")
            for sym, pos in list(pf.positions.items()):
                if pos.qty != 0:
                    fill = broker.place_order(Order(sym, qty=-pos.qty), ts=ts)
                    pf.update_from_fill(fill)
                    trades.append(fill)
            break

        current_bars = {sym: df.loc[:ts] for sym, df in bars.items()}
        sigs = strat.generate_signals(current_bars)

        equity = float(row["equity"])
        for sym, sig in sigs.items():
            px = prices.get(sym)
            if px is None:
                continue

            target_pct = clamp_target(sym, sig.target_pct, equity, px, limits)

            if ml_enabled and sym in ml_preds:
                p = ml_preds[sym].loc[:ts].tail(1)
                if len(p) == 1:
                    val = float(p.iloc[0])
                    if str(load_yaml(str(ml_cfg_path)).get("model", {}).get("out", "p_up")) == "p_up":
                        scale = max(min((val - 0.5) * 2.0, 1.0), -1.0)
                        target_pct = target_pct * scale
                    else:
                        scale = max(min(val * 10.0, 1.0), -1.0)
                        target_pct = target_pct * scale

            current_qty = pf.position_qty(sym)
            target_qty = target_pct_to_qty(target_pct, equity, px)
            delta = target_qty - current_qty

            if abs(delta) * px < 10.0:
                continue

            fill = broker.place_order(Order(sym, qty=delta), ts=ts)
            pf.update_from_fill(fill)
            trades.append(fill)

    equity_df = pd.DataFrame(pf.equity_curve)
    trades_df = pd.DataFrame(trades)

    metrics = compute_metrics(equity_df) if not equity_df.empty else {}
    save_json(os.path.join(artifacts.run_dir, "metrics.json"), metrics)

    if bool(cfg.get("reports", "csv", default=True)):
        if not trades_df.empty:
            write_csv(artifacts.run_dir, "trades", trades_df)
        if not equity_df.empty:
            write_csv(artifacts.run_dir, "equity_curve", equity_df)

    if bool(cfg.get("reports", "html", default=True)) and not equity_df.empty:
        write_html_report(artifacts.run_dir, artifacts.run_id, metrics, equity_df, trades_df)

    logger.info(f"Backtest complete. Metrics: {metrics}")
    logger.info(f"Artifacts: {artifacts.run_dir}")
    return artifacts.run_dir

def run_paper(config_path: str, ml_path_override: Optional[str] = None) -> str:
    raw = load_yaml(config_path)
    cfg = Config(raw)

    run_name = str(cfg.get("run", "name", default="paper"))
    out_dir = str(cfg.get("run", "out_dir", default="runs"))
    artifacts = make_run_dir(out_dir, run_name)
    snapshot_config(config_path, artifacts.run_dir)
    logger = setup_logger("quant_sim", artifacts.logs_path)
    logger.info(f"Starting paper mode: {run_name} -> {artifacts.run_dir}")

    commission_bps = float(cfg.get("execution", "commission_bps", default=2.0))
    slippage_bps = float(cfg.get("execution", "slippage_bps", default=1.0))
    
    paper_broker_kind = str(cfg.get("paper", "broker", default="coinbase_public")).lower()
    
    if paper_broker_kind == "ccxt":
        exchange_id = str(cfg.get("ccxt", "exchange", default="binance")).lower()
        products = cfg.get("ccxt", "symbols", default=["BTC/USDT"]) or ["BTC/USDT"]
        broker = CCXTPaperBroker(
            exchange_id,
            products,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
    else:
        products = cfg.get("coinbase", "products", default=["BTC-USD"]) or ["BTC-USD"]
        broker = CoinbasePaperBroker(products, commission_bps=commission_bps, slippage_bps=slippage_bps)

    starting_cash = float(cfg.get("portfolio", "starting_cash", default=100000.0))
    pf = Portfolio(cash=starting_cash)
    limits = _risk_limits(cfg)
    strat = _build_strategy(cfg)

    poll_seconds = float(cfg.get("paper", "poll_seconds", default=3))
    decision_tf = str(cfg.get("strategy", "timeframe", default="5m")).lower()
    decision_rule = {"1m":60, "5m":300, "15m":900, "1h":3600}.get(decision_tf, 300)
    
    max_orders_per_min = int(cfg.get("paper", "max_orders_per_min", default=20))
    max_stale_seconds = float(cfg.get("paper", "max_stale_seconds", default=20))
    max_failures = int(cfg.get("paper", "max_failures", default=5))
    rate_limiter = OrderRateLimiter(max_orders=max_orders_per_min, window_seconds=60)
    disconnect = DisconnectGuard(max_stale_seconds=max_stale_seconds, max_failures=max_failures)

    ml_enabled = bool(cfg.get("ml", "enabled", default=False))
    ml_cfg_path = ml_path_override or cfg.get("ml", "config", default=None)
    ml_raw = load_yaml(str(ml_cfg_path)) if (ml_enabled and ml_cfg_path) else None
    ml_signal = PyTorchMLSignal(ml_raw) if ml_enabled else None

    bars_1m: Dict[str, pd.DataFrame] = {p: pd.DataFrame(columns=["open","high","low","close","volume"]) for p in products}
    trades = []

    equity_start_day = starting_cash
    day_anchor = pd.Timestamp.now("UTC").date()
    last_decision = time.time()

    logger.info("Paper mode running. Press Ctrl+C to stop.")
    try:
        while True:
            now = pd.Timestamp.now("UTC")
            if now.date() != day_anchor:
                last_eq = pf.equity_curve[-1]["equity"] if pf.equity_curve else starting_cash
                equity_start_day = float(last_eq)
                day_anchor = now.date()

            if disconnect.tripped():
                logger.warning("Disconnect guard tripped: stale quotes or too many failures. Stopping.")
                break
                
            prices = {}
            for sym in products:
                try:
                    px = broker.get_price(sym)
                    prices[sym] = float(px)
                    disconnect.ok()
                except Exception as e:
                    logger.warning(f"Quote failed for {sym}: {e}")
                    disconnect.fail()
                    continue
                df = bars_1m[sym]
                minute = now.floor("min")
                if df.empty or (df.index[-1] != minute):
                    new = pd.DataFrame(
                        {"open":[px], "high":[px], "low":[px], "close":[px], "volume":[0.0]},
                        index=pd.DatetimeIndex([minute], tz="UTC")
                    )
                    bars_1m[sym] = pd.concat([df, new]).tail(5000)
                else:
                    bars_1m[sym].loc[minute, "high"] = max(float(bars_1m[sym].loc[minute, "high"]), px)
                    bars_1m[sym].loc[minute, "low"] = min(float(bars_1m[sym].loc[minute, "low"]), px)
                    bars_1m[sym].loc[minute, "close"] = px

            row = pf.mark_to_market(prices, now)

            if limits.kill_switch and check_daily_loss(equity_start_day, row["equity"], limits):
                logger.warning("Kill-switch triggered: max daily loss reached. Flattening positions and stopping.")
                for sym, pos in list(pf.positions.items()):
                    if pos.qty != 0:
                        fill = broker.place_order(Order(sym, qty=-pos.qty), ts=now)
                        pf.update_from_fill(fill)
                        trades.append(fill)
                break

            if time.time() - last_decision >= decision_rule:
                last_decision = time.time()

                decision_bars = {}
                for sym, df in bars_1m.items():
                    if len(df) < 120:
                        continue
                    if decision_tf == "1m":
                        decision_bars[sym] = df.copy()
                    else:
                        decision_bars[sym] = resample_ohlcv(df, decision_tf)

                if not decision_bars:
                    time.sleep(poll_seconds)
                    continue

                sigs = strat.generate_signals(decision_bars)
                equity = float(row["equity"])

                for sym, sig in sigs.items():
                    px = prices.get(sym)
                    if px is None:
                        continue
                    target_pct = clamp_target(sym, sig.target_pct, equity, px, limits)

                    current_qty = pf.position_qty(sym)
                    target_qty = target_pct_to_qty(target_pct, equity, px)
                    delta = target_qty - current_qty
                    if abs(delta) * px < 10.0:
                        continue
                    
                    if not rate_limiter.allow():
                        logger.warning(f"Order rate limit hit. Skipping order for {sym}.")
                        continue

                    fill = broker.place_order(Order(sym, qty=delta), ts=now)
                    pf.update_from_fill(fill)
                    trades.append(fill)

                logger.info(f"Decision executed at {now.isoformat()} | equity={equity:.2f}")

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        logger.info("Stopping (Ctrl+C). Writing artifacts...")

    equity_df = pd.DataFrame(pf.equity_curve)
    trades_df = pd.DataFrame(trades)

    metrics = compute_metrics(equity_df) if not equity_df.empty else {}
    save_json(os.path.join(artifacts.run_dir, "metrics.json"), metrics)

    if bool(cfg.get("reports", "csv", default=True)):
        if not trades_df.empty:
            write_csv(artifacts.run_dir, "trades", trades_df)
        if not equity_df.empty:
            write_csv(artifacts.run_dir, "equity_curve", equity_df)

    if bool(cfg.get("reports", "html", default=True)) and not equity_df.empty:
        write_html_report(artifacts.run_dir, artifacts.run_id, metrics, equity_df, trades_df)

    logger.info(f"Paper run complete. Metrics: {metrics}")
    logger.info(f"Artifacts: {artifacts.run_dir}")
    return artifacts.run_dir

def run_report(run_dir: str) -> str:
    eq_path = os.path.join(run_dir, "equity_curve.csv")
    tr_path = os.path.join(run_dir, "trades.csv")
    m_path = os.path.join(run_dir, "metrics.json")

    equity_df = pd.read_csv(eq_path) if os.path.exists(eq_path) else pd.DataFrame()
    trades_df = pd.read_csv(tr_path) if os.path.exists(tr_path) else pd.DataFrame()

    import json
    metrics = {}
    if os.path.exists(m_path):
        with open(m_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    run_id = os.path.basename(run_dir.rstrip("/"))
    if equity_df.empty:
        raise ValueError("No equity_curve.csv found to render report.")
    path = write_html_report(run_dir, run_id, metrics, equity_df, trades_df)
    return path

def run_dashboard(runs_dir: str, host: str = "0.0.0.0", port: int = 8000):
    """Serve a simple dashboard to browse run artifacts."""
    import http.server
    import socketserver
    import json
    from urllib.parse import urlparse, parse_qs
    
    runs_dir = os.path.abspath(runs_dir)
    if not os.path.isdir(runs_dir):
        os.makedirs(runs_dir, exist_ok=True)
    
    class DashboardHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=runs_dir, **kwargs)
        
        def do_GET(self):
            parsed = urlparse(self.path)
            
            if parsed.path == "/" or parsed.path == "":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                
                runs = []
                for name in sorted(os.listdir(runs_dir), reverse=True):
                    run_path = os.path.join(runs_dir, name)
                    if os.path.isdir(run_path):
                        metrics_path = os.path.join(run_path, "metrics.json")
                        metrics = {}
                        if os.path.exists(metrics_path):
                            with open(metrics_path) as f:
                                metrics = json.load(f)
                        runs.append({"name": name, "metrics": metrics})
                
                html = """<!DOCTYPE html>
<html><head><title>Quant Sim Dashboard</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
h1 { color: #00d9ff; }
table { width: 100%; border-collapse: collapse; margin-top: 20px; }
th, td { padding: 12px; text-align: left; border-bottom: 1px solid #333; }
th { background: #16213e; color: #00d9ff; }
tr:hover { background: #16213e; }
a { color: #00d9ff; text-decoration: none; }
a:hover { text-decoration: underline; }
.positive { color: #4ade80; }
.negative { color: #f87171; }
</style></head><body>
<h1>Quant Sim Dashboard</h1>
<table>
<tr><th>Run</th><th>Total Return</th><th>Sharpe</th><th>Max DD</th><th>Report</th></tr>
"""
                for r in runs:
                    m = r["metrics"]
                    ret = m.get("total_return_pct", "N/A")
                    ret_class = "positive" if isinstance(ret, (int,float)) and ret > 0 else "negative"
                    ret_str = f"{ret:.2f}%" if isinstance(ret, (int,float)) else ret
                    sharpe = m.get("sharpe_ratio", "N/A")
                    sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int,float)) else sharpe
                    dd = m.get("max_drawdown_pct", "N/A")
                    dd_str = f"{dd:.2f}%" if isinstance(dd, (int,float)) else dd
                    
                    html += f'<tr><td>{r["name"]}</td>'
                    html += f'<td class="{ret_class}">{ret_str}</td>'
                    html += f'<td>{sharpe_str}</td>'
                    html += f'<td class="negative">{dd_str}</td>'
                    html += f'<td><a href="/{r["name"]}/report.html">View Report</a></td></tr>\n'
                
                html += "</table></body></html>"
                self.wfile.write(html.encode())
            else:
                super().do_GET()
    
    with socketserver.TCPServer((host, port), DashboardHandler) as httpd:
        print(f"Dashboard running at http://{host}:{port}")
        print(f"Serving runs from: {runs_dir}")
        httpd.serve_forever()


def main():
    ap = argparse.ArgumentParser(prog="quant-sim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_bt = sub.add_parser("backtest", help="Run historical backtest from CSV OHLCV files")
    ap_bt.add_argument("--config", required=True, help="Path to YAML config")
    ap_bt.add_argument("--ml", required=False, help="Optional ML YAML config override")

    ap_pp = sub.add_parser("paper", help="Run paper trading loop using Coinbase public price feed")
    ap_pp.add_argument("--config", required=True, help="Path to YAML config")
    ap_pp.add_argument("--ml", required=False, help="Optional ML YAML config override")

    ap_rp = sub.add_parser("report", help="Re-render HTML report for an existing run directory")
    ap_rp.add_argument("--run", required=True, help="Path to run directory (runs/<run_id>/)")

    ap_db = sub.add_parser("dashboard", help="Serve a dashboard to browse run artifacts")
    ap_db.add_argument("--runs-dir", default="runs", help="Path to runs directory")
    ap_db.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    ap_db.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")

    args = ap.parse_args()

    if args.cmd == "backtest":
        run_backtest(args.config, ml_path_override=args.ml)
    elif args.cmd == "paper":
        run_paper(args.config, ml_path_override=args.ml)
    elif args.cmd == "report":
        path = run_report(args.run)
        print(path)
    elif args.cmd == "dashboard":
        run_dashboard(args.runs_dir, args.host, args.port)

if __name__ == "__main__":
    main()
