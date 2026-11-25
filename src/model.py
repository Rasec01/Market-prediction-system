"""
Walk-forward training.

The model is retrained every `step` days on all history whose label is
already known. A label for day i needs the price on day i + horizon, so on
day t the last usable training row is t - horizon (the embargo). Predictions
for each block come from a model that never saw that block.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from src.features import FEATURES


def walk_forward(df: pd.DataFrame, horizon: int = 5, min_train: int = 756, step: int = 63,
                 n_estimators: int = 300, min_samples_leaf: int = 50,
                 random_state: int = 42) -> pd.DataFrame:
    """Return df with an out-of-sample 'prediction' column (NaN before min_train)."""
    df = df.copy()
    df["prediction"] = np.nan
    X = df[FEATURES].to_numpy()
    y = df["target"].to_numpy()
    n = len(df)
    if n <= min_train:
        raise ValueError(f"Need more than {min_train} rows, got {n}.")

    for start in range(min_train, n, step):
        end = min(start + step, n)
        train_end = start - horizon  # last row whose label is known on day `start`
        rows = np.arange(train_end + 1)
        rows = rows[~np.isnan(y[rows])]
        model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                      n_jobs=-1, random_state=random_state)
        model.fit(X[rows], y[rows])
        df.iloc[start:end, df.columns.get_loc("prediction")] = model.predict(X[start:end])
    return df


def forecast_quality(df: pd.DataFrame) -> dict:
    """Information coefficient (rank correlation) and hit rate on out-of-sample rows."""
    oos = df.dropna(subset=["prediction", "target"])
    ic = spearmanr(oos["prediction"], oos["target"]).statistic
    hit = float((np.sign(oos["prediction"]) == np.sign(oos["target"])).mean())
    return {"oos_days": len(oos), "information_coefficient": float(ic), "hit_rate": hit}
