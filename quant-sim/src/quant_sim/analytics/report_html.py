from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template

HTML_TMPL = """
<!doctype html>

<html>
<head>
  <meta charset="utf-8"/>
  <title>Quant Sim Report</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #eee; text-align: left; padding: 6px; font-size: 13px; }
    h1 { margin: 0 0 8px 0; }
    .muted { color: #666; font-size: 13px; }
    img { max-width: 100%; border-radius: 10px; border: 1px solid #eee; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>Quant Simulator Report</h1>
  <div class="muted">Run: <code>{{ run_id }}</code></div>
  <div class="grid" style="margin-top:16px;">
    <div class="card">
      <h3>Metrics</h3>
      <table>
        {% for k, v in metrics.items() %}
        <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
        {% endfor %}
      </table>
    </div>
    <div class="card">
      <h3>Equity Curve</h3>
      <img src="{{ equity_png }}" alt="Equity Curve"/>
    </div>
  </div>


  <div class="card" style="margin-top:16px;">
    <h3>Recent Trades</h3>
    <table>
      <tr>
        {% for col in trades_cols %}<th>{{ col }}</th>{% endfor %}
      </tr>
      {% for row in trades_rows %}
      <tr>
        {% for col in trades_cols %}<td>{{ row[col] }}</td>{% endfor %}
      </tr>
      {% endfor %}
    </table>
  </div>
</body>
</html>
"""


def _plot_equity(equity_curve: pd.DataFrame, png_path: str) -> None:
    plt.figure()
    plt.plot(pd.to_datetime(equity_curve["timestamp"]), equity_curve["equity"].astype(float))
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()

def write_html_report(run_dir: str, run_id: str, metrics: dict, equity_curve: pd.DataFrame, trades: pd.DataFrame) -> str:
    png_path = os.path.join(run_dir, "equity_curve.png")
    _plot_equity(equity_curve, png_path)

    tmpl = Template(HTML_TMPL)
    trades_cols = list(trades.columns) if not trades.empty else ["(no trades)"]
    trades_rows = trades.tail(50).to_dict(orient="records") if not trades.empty else [{"(no trades)": ""}]

    html = tmpl.render(
        run_id=run_id,
        metrics=metrics,
        equity_png=os.path.basename(png_path),
        trades_cols=trades_cols,
        trades_rows=trades_rows,
    )
    out = os.path.join(run_dir, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out
