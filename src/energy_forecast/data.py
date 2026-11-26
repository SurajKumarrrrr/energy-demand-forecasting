"""Deterministic synthetic generator for German-style hourly electricity load.

The series is *synthetic*. It is shaped to resemble the public German grid load
(roughly 40-75 GW, double daily peak, weekend and holiday dips, higher demand in
winter) so that forecasting methods can be compared offline and reproducibly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.calendar import holiday_set

# Relative intraday profile for a working day (index = hour 0..23).
_WEEKDAY_PROFILE = np.array(
    [0.80, 0.77, 0.75, 0.75, 0.77, 0.83, 0.93, 1.03, 1.08, 1.10, 1.11, 1.12,
     1.12, 1.10, 1.08, 1.06, 1.06, 1.08, 1.11, 1.10, 1.05, 0.98, 0.91, 0.85]
)
# Weekends/holidays: later and flatter morning ramp.
_WEEKEND_PROFILE = np.array(
    [0.84, 0.80, 0.77, 0.76, 0.75, 0.76, 0.79, 0.84, 0.90, 0.96, 1.00, 1.03,
     1.04, 1.02, 0.99, 0.97, 0.97, 1.00, 1.05, 1.07, 1.04, 0.98, 0.92, 0.87]
)


def generate_load(
    start: str = "2022-01-01",
    end: str = "2024-12-31 23:00",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate hourly load (MW) and temperature (degC).

    Returns a DataFrame indexed by a naive hourly ``DatetimeIndex`` with columns
    ``load_mw``, ``temp_c`` and ``is_holiday``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    n = len(idx)
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()

    holidays = holiday_set(range(idx[0].year, idx[-1].year + 1))
    is_holiday = np.array([d in holidays for d in idx.date])
    # Christmas-to-New-Year industrial shutdown behaves like a soft holiday.
    xmas_lull = ((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 2))

    # Temperature: annual cycle + daily cycle + persistent weather anomalies (AR(1) on days).
    annual_temp = 10.0 - 9.5 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    daily_temp = 3.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    n_days = n // 24 + 2
    anom = np.zeros(n_days)
    shocks = rng.normal(0, 1.6, n_days)
    for i in range(1, n_days):
        anom[i] = 0.8 * anom[i - 1] + shocks[i]
    day_idx = (idx.normalize() - idx[0].normalize()).days.to_numpy()
    temp = annual_temp + daily_temp + anom[day_idx] + rng.normal(0, 0.7, n)

    # Load components.
    level = 57_000.0 * (1 - 0.015 * (idx.year.to_numpy() - idx[0].year))  # slow efficiency decline
    off_day = (dow >= 5) | is_holiday
    profile = np.where(off_day, _WEEKEND_PROFILE[hour], _WEEKDAY_PROFILE[hour])
    day_factor = np.select([is_holiday, dow == 6, dow == 5], [0.80, 0.82, 0.88], default=1.0)
    day_factor = np.where(xmas_lull & ~is_holiday, day_factor * 0.90, day_factor)
    seasonal = 1 + 0.06 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    # Heating (below 15 degC) dominates; a small cooling effect above 22 degC.
    heating = 260.0 * np.clip(15.0 - temp, 0, None)
    cooling = 180.0 * np.clip(temp - 22.0, 0, None)

    base = level * profile * day_factor * seasonal + heating + cooling

    # Autocorrelated noise (hourly AR(1)) to mimic unexplained demand swings.
    eps = rng.normal(0, 550.0, n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + eps[i]

    df = pd.DataFrame(
        {
            "load_mw": np.round(base + noise, 1),
            "temp_c": np.round(temp, 2),
            "is_holiday": is_holiday,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df
