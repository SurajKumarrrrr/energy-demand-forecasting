"""Feature engineering for day-ahead (24 h horizon) load forecasting.

All lag features use lags >= 24 h, so every feature for the next day is known
at the forecast origin (midnight). This keeps the backtest free of leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 24
LAGS = (24, 48, 168, 336)


def build_features(df: pd.DataFrame, target: str = "load_mw") -> pd.DataFrame:
    """Return a feature matrix aligned to ``df.index``.

    ``temp_c`` is treated as a weather *forecast* for the target hour (in
    production it would come from a NWP provider; here the synthetic actual is
    used, i.e. a perfect-forecast assumption that is stated in the README).
    """
    y = df[target]
    idx = df.index
    feats = pd.DataFrame(index=idx)
    feats["hour"] = idx.hour
    feats["dow"] = idx.dayofweek
    feats["month"] = idx.month
    feats["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    feats["is_holiday"] = df["is_holiday"].astype(int)
    feats["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feats["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    feats["temp_c"] = df["temp_c"]
    feats["hdd"] = np.clip(15.0 - df["temp_c"], 0, None)
    for lag in LAGS:
        feats[f"lag_{lag}"] = y.shift(lag)
    # Daily mean of the most recent fully known day, broadcast to each hour.
    feats["roll_mean_24_lag24"] = y.shift(HORIZON).rolling(24).mean()
    feats["roll_mean_168_lag24"] = y.shift(HORIZON).rolling(168).mean()
    return feats
