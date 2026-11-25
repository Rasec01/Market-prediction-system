"""
Market prediction and walk-forward backtest.

    python main.py                                  # SPY from 2005
    python main.py --ticker QQQ --cost-bps 10 --chart outputs/qqq.png
    python main.py --csv data/prices.csv            # your own Date,Close file
"""
import argparse
from pathlib import Path

from src.backtest import backtest, make_signal, plot, summary
from src.data_loader import load_prices
from src.features import make_features
from src.model import forecast_quality, walk_forward


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--csv", default=None, help="CSV with Date and Close columns instead of downloading")
    p.add_argument("--horizon", type=int, default=5, help="forecast horizon in trading days")
    p.add_argument("--min-train", type=int, default=756, help="days of history before the first forecast")
    p.add_argument("--step", type=int, default=63, help="retrain every N days")
    p.add_argument("--threshold", type=float, default=0.0, help="hold the asset when the forecast exceeds this")
    p.add_argument("--cost-bps", type=float, default=5.0, help="cost per trade, basis points")
    p.add_argument("--chart", default=None, help="save the equity curve to this file")
    return p.parse_args()


def main():
    a = parse_args()
    close = load_prices(a.ticker, a.start, a.csv)
    df = make_features(close, a.horizon)
    df = walk_forward(df, horizon=a.horizon, min_train=a.min_train, step=a.step)

    print("\nForecast quality (out of sample):")
    for k, v in forecast_quality(df).items():
        print(f"  {k:<26}{v:.4f}" if isinstance(v, float) else f"  {k:<26}{v}")

    bt = backtest(df, make_signal(df, a.threshold), a.cost_bps)
    print(f"\nBacktest {bt.index[0].date()} → {bt.index[-1].date()}, costs {a.cost_bps:g} bps per trade:")
    print(summary(bt).round(4).to_string())

    if a.chart:
        Path(a.chart).parent.mkdir(parents=True, exist_ok=True)
        plot(bt, a.chart, f"{a.ticker if not a.csv else Path(a.csv).stem}: walk-forward, {a.horizon}-day forecasts")
        print(f"\nChart saved to {a.chart}")


if __name__ == "__main__":
    main()
