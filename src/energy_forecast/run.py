"""End-to-end run: generate data, backtest all models, write report + plot."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from energy_forecast.backtest import daily_origins, rolling_backtest  # noqa: E402
from energy_forecast.data import generate_load  # noqa: E402
from energy_forecast.metrics import summarize  # noqa: E402
from energy_forecast.models import (  # noqa: E402
    GradientBoostingForecaster,
    SarimaForecaster,
    SeasonalNaive,
)

NAIVE_REF = "seasonal_naive_168h"


def score(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per model, scaled against the weekly naive forecast."""
    ref = results[results["model"] == NAIVE_REF].set_index("timestamp")["y_pred"]
    out = {}
    for name, g in results.groupby("model", sort=False):
        g = g.set_index("timestamp")
        out[name] = summarize(g["y_true"], g["y_pred"], ref.loc[g.index])
    return pd.DataFrame(out).T.sort_values("MAE_MW")


def score_by(results: pd.DataFrame, key: pd.Series, label: str) -> pd.DataFrame:
    """MAPE per model split by a grouping key (e.g. weekday vs weekend)."""
    r = results.assign(**{label: key.to_numpy()})
    r["ape"] = (r["y_true"] - r["y_pred"]).abs() / r["y_true"] * 100
    return r.pivot_table(index="model", columns=label, values="ape", aggfunc="mean")


def plot_week(results: pd.DataFrame, path: Path) -> None:
    last = results["timestamp"].max()
    window = results[results["timestamp"] > last - pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(11, 4))
    truth = window.drop_duplicates("timestamp").set_index("timestamp")["y_true"]
    ax.plot(truth.index, truth / 1000, color="black", lw=2, label="actual (synthetic)")
    for name, g in window.groupby("model", sort=False):
        ax.plot(g["timestamp"], g["y_pred"] / 1000, lw=1.2, label=name)
    ax.set_ylabel("load [GW]")
    ax.set_title("Day-ahead forecasts, last 7 days of the backtest")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=56, help="number of backtest origins")
    parser.add_argument("--no-sarima", action="store_true", help="skip the slow SARIMA model")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    df = generate_load()
    origins = daily_origins(df.index, args.days)
    factories = [
        lambda: SeasonalNaive(24),
        lambda: SeasonalNaive(168),
        lambda: GradientBoostingForecaster(),
    ]
    if not args.no_sarima:
        factories.append(lambda: SarimaForecaster())

    t0 = time.perf_counter()
    results = rolling_backtest(df, factories, origins, refit_every=7)
    elapsed = time.perf_counter() - t0

    metrics = score(results)
    ts = pd.to_datetime(results["timestamp"])
    holiday_dates = set(df.index[df["is_holiday"]].date)
    daytype = pd.Series(
        ["holiday" if d in holiday_dates else ("weekend" if w >= 5 else "weekday")
         for d, w in zip(ts.dt.date, ts.dt.dayofweek, strict=True)]
    )
    by_daytype = score_by(results, daytype, "day_type")

    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "forecasts.csv", index=False)
    metrics.to_csv(args.out / "metrics.csv", float_format="%.3f")
    plot_week(results, args.out / "last_week.png")

    print(f"Data: {len(df):,} hourly rows {df.index[0]} -> {df.index[-1]} (synthetic)")
    print(f"Backtest: {len(origins)} daily origins {origins[0].date()} -> {origins[-1].date()}, "
          f"{elapsed:.1f}s\n")
    print(metrics.round(3).to_markdown())
    print("\nMAPE % by day type:\n")
    print(by_daytype.round(2).to_markdown())
    return metrics


if __name__ == "__main__":
    main()
