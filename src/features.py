"""
Features and label.

All features are returns or ratios, so they look alike in 2008 and in 2024.
Raw price levels (a 10-day moving average of 180 vs 450) would put the test
period outside anything the trees saw in training.
"""
import pandas as pd

FEATURES = ["ret_1", "ret_5", "ret_20", "ma_gap", "price_to_ma50", "vol_10", "vol_ratio"]


def make_features(close: pd.Series, horizon: int = 5) -> pd.DataFrame:
    """
    Features use prices up to and including day t. The label is the return
    from the close of day t to the close of day t + horizon.
    """
    df = pd.DataFrame({"close": close})
    r = close.pct_change()
    df["ret_1"] = r
    df["ret_5"] = close.pct_change(5)
    df["ret_20"] = close.pct_change(20)
    df["ma_gap"] = close.rolling(10).mean() / close.rolling(30).mean() - 1
    df["price_to_ma50"] = close / close.rolling(50).mean() - 1
    df["vol_10"] = r.rolling(10).std()
    df["vol_ratio"] = r.rolling(10).std() / r.rolling(60).std()

    # Forward return over the horizon (NaN for the last `horizon` days)
    df["target"] = close.shift(-horizon) / close - 1
    # Next day's return, used only by the backtest
    df["next_ret"] = r.shift(-1)
    return df.dropna(subset=FEATURES)
