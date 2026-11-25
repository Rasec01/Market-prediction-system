import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest, make_signal, summary
from src.features import make_features
from src.model import walk_forward


def prices(n=1200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx, name="close")


def test_target_is_the_forward_horizon_return():
    close = prices(200)
    df = make_features(close, horizon=5)
    t = df.index[10]
    i = close.index.get_loc(t)
    assert df.loc[t, "target"] == pytest.approx(close.iloc[i + 5] / close.iloc[i] - 1)


def test_features_are_scale_free():
    close = prices(300)
    a = make_features(close)
    b = make_features(close * 10)
    pd.testing.assert_frame_equal(a.drop(columns="close"), b.drop(columns="close"))


def test_changing_future_prices_does_not_change_past_forecasts():
    close = prices()
    k = 1000
    changed = close.copy()
    changed.iloc[k:] *= np.linspace(1, 3, len(changed) - k)  # rewrite the future
    kw = dict(horizon=5, min_train=400, step=50, n_estimators=30)
    p1 = walk_forward(make_features(close), **kw)["prediction"]
    p2 = walk_forward(make_features(changed), **kw)["prediction"]
    before = p1.index < close.index[k]
    pd.testing.assert_series_equal(p1[before], p2[before])


def test_backtest_trades_on_the_next_day_and_charges_costs():
    idx = pd.bdate_range("2020-01-01", periods=4)
    df = pd.DataFrame({"prediction": [1.0, -1.0, 1.0, 1.0], "next_ret": [0.01, 0.02, -0.01, 0.03]}, index=idx)
    bt = backtest(df, make_signal(df), cost_bps=10)
    # day 1: in, earns 1% minus 10 bps to enter; day 2: out, pays 10 bps; day 3: back in
    assert bt["strategy"].tolist() == pytest.approx([0.01 - 0.001, -0.001, -0.01 - 0.001, 0.03])
    assert summary(bt).loc["time_in_market", "Strategy"] == 0.75
