"""Price data: download with yfinance, or read a CSV with Date and Close columns."""
from pathlib import Path
from typing import Optional

import pandas as pd


def load_prices(ticker: str = "SPY", start: str = "2005-01-01", csv: Optional[str] = None) -> pd.Series:
    """Return a date-indexed series of adjusted closing prices."""
    if csv:
        df = pd.read_csv(Path(csv), parse_dates=["Date"], index_col="Date")
        close = df["Close"]
    else:
        import yfinance as yf  # imported here so tests run without it

        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for '{ticker}'.")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # newer yfinance returns one column per ticker
            close = close.iloc[:, 0]
    close = close.dropna().astype(float).sort_index()
    close.name = "close"
    print(f"[data] {len(close)} prices, {close.index[0].date()} → {close.index[-1].date()}")
    return close
