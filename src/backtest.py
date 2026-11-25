"""
Long/flat backtest with transaction costs.

The signal is decided at the close of day t and earns day t+1's return.
Each change of position costs `cost_bps` basis points of the traded amount.
The risk-free rate is set to zero in the Sharpe ratio.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def make_signal(df: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    """1 = hold the asset, 0 = stay in cash."""
    return (df["prediction"] > threshold).astype(float).where(df["prediction"].notna())


def backtest(df: pd.DataFrame, signal: pd.Series, cost_bps: float = 5.0) -> pd.DataFrame:
    """Daily strategy and buy-and-hold returns over the out-of-sample period."""
    s = signal.dropna()
    nxt = df.loc[s.index, "next_ret"]
    trades = s.diff().abs().fillna(s.iloc[0])  # entering at the start counts as a trade
    strat = s * nxt - trades * cost_bps / 1e4
    out = pd.DataFrame({"strategy": strat, "buy_and_hold": nxt, "position": s, "trades": trades})
    return out.dropna()


def performance(daily: pd.Series) -> dict:
    wealth = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    ann_ret = wealth.iloc[-1] ** (1 / years) - 1
    ann_vol = daily.std() * np.sqrt(TRADING_DAYS)
    return {
        "total_return": float(wealth.iloc[-1] - 1),
        "annual_return": float(ann_ret),
        "annual_volatility": float(ann_vol),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
    }


def summary(bt: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({"Strategy": performance(bt["strategy"]),
                          "Buy & hold": performance(bt["buy_and_hold"])})
    table.loc["time_in_market"] = [bt["position"].mean(), 1.0]
    table.loc["trades_per_year"] = [bt["trades"].sum() / (len(bt) / TRADING_DAYS), 0.0]
    return table


def plot(bt: pd.DataFrame, path=None, title=""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + bt["strategy"]).cumprod().plot(ax=ax, label="Strategy (after costs)")
    (1 + bt["buy_and_hold"]).cumprod().plot(ax=ax, label="Buy & hold", alpha=0.7)
    ax.set_title(title or "Walk-forward backtest")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
