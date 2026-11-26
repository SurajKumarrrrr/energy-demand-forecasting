"""Forecast error metrics."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error in percent. Undefined if y_true contains 0."""
    y_true = np.asarray(y_true, dtype=float)
    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined when y_true contains zeros")
    return float(100 * np.mean(np.abs((y_true - np.asarray(y_pred)) / y_true)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
    """MAE scaled by the MAE of a naive reference forecast on the same points."""
    denom = mae(y_true, y_naive)
    if denom == 0:
        raise ValueError("Naive forecast is perfect; MASE undefined")
    return mae(y_true, y_pred) / denom


def summarize(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> dict[str, float]:
    return {
        "MAE_MW": mae(y_true, y_pred),
        "RMSE_MW": rmse(y_true, y_pred),
        "MAPE_pct": mape(y_true, y_pred),
        "MASE_vs_weekly_naive": mase(y_true, y_pred, y_naive),
    }
