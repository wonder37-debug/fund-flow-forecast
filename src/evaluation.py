"""Monthly chronological evaluation and amount-error metrics."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import numpy as np
import pandas as pd

from src.features.calendar import build_calendar
from src.models.baselines import DATE_COLUMN, TARGET_COLUMNS, BaselineSpec, SeasonalMeanBaseline

TARGET_WEIGHTS = {"purchase": 0.45, "redeem": 0.55}
BAD_DAY_THRESHOLD = 0.30
DAILY_COLUMNS = (DATE_COLUMN, *TARGET_COLUMNS)
SELECTION_FOLDS = 4


@dataclass(frozen=True)
class Fold:
    cutoff: pd.Timestamp
    future_dates: pd.DatetimeIndex
    role: str


class SingleTargetModel(Protocol):
    def fit(self, history: pd.DataFrame) -> object: ...

    def predict(self, future_calendar: pd.DataFrame) -> pd.Series: ...


def build_folds(horizon: str = "calendar-month") -> tuple[Fold, ...]:
    """Build four selection folds followed by the August evaluation fold."""
    if horizon not in {"calendar-month", "fixed30"}:
        raise ValueError(f"Unknown evaluation horizon: {horizon}")
    months = pd.date_range("2014-04-01", "2014-08-01", freq="MS")
    folds = []
    for index, start in enumerate(months):
        holdout = index == SELECTION_FOLDS
        days = 30 if horizon == "fixed30" else start.days_in_month
        future_dates = pd.date_range(start, periods=days)
        folds.append(Fold(start - pd.Timedelta(days=1), future_dates, "holdout" if holdout else "selection"))
    return tuple(folds)


def training_history(
    daily: pd.DataFrame, cutoff: pd.Timestamp, rolling_days: int | None = None
) -> pd.DataFrame:
    if rolling_days is not None and rolling_days < 1:
        raise ValueError("Rolling history length must be positive")
    mask = daily[DATE_COLUMN] <= cutoff
    if rolling_days is not None:
        mask &= daily[DATE_COLUMN] > cutoff - pd.Timedelta(days=rolling_days)
    history = daily.loc[mask, list(DAILY_COLUMNS)].sort_values(DATE_COLUMN).reset_index(drop=True)
    if history.empty:
        raise ValueError(f"No training history through {cutoff.date()}")
    start = history[DATE_COLUMN].iloc[0]
    if rolling_days is not None:
        start = cutoff - pd.Timedelta(days=rolling_days - 1)
    expected = pd.date_range(start, cutoff)
    if not pd.DatetimeIndex(history[DATE_COLUMN]).equals(expected):
        raise ValueError(f"Incomplete training date coverage for cutoff {cutoff.date()}")
    return history


def fold_actuals(daily: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    actual = daily.loc[daily[DATE_COLUMN].isin(fold.future_dates), list(DAILY_COLUMNS)]
    actual = actual.sort_values(DATE_COLUMN).reset_index(drop=True)
    if not pd.DatetimeIndex(actual[DATE_COLUMN]).equals(fold.future_dates):
        raise ValueError(f"Incomplete validation date coverage after {fold.cutoff.date()}")
    return actual


def split_fold(
    daily: pd.DataFrame, fold: Fold, rolling_days: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return training_history(daily, fold.cutoff, rolling_days), fold_actuals(daily, fold)


def predict_targets(
    history: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    model_factory: Callable[[str], SingleTargetModel],
    *,
    calendar_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build one model per target with observed amounts and declared calendar features."""
    history = history.loc[:, list(DAILY_COLUMNS)].reset_index(drop=True)
    history_dates = pd.DatetimeIndex(history[DATE_COLUMN])
    for dates in (history_dates, future_dates):
        if not len(dates) or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("History and forecast dates must be nonempty, unique and increasing")
    if future_dates[0] <= history_dates[-1]:
        raise ValueError("Forecast dates must be after the training cutoff")

    future_calendar = pd.DataFrame({DATE_COLUMN: future_dates})
    predictions = future_calendar.copy()
    if calendar_columns:
        trusted_calendar = build_calendar(history_dates[0], future_dates[-1]).set_index(DATE_COLUMN)
        declared_columns = list(calendar_columns)
        if len(set(declared_columns)) != len(declared_columns) or not set(declared_columns).issubset(
            trusted_calendar.columns.difference(DAILY_COLUMNS)
        ):
            raise ValueError("Requested calendar columns must be unique, known features from build_calendar")

        history = history.join(trusted_calendar.loc[history_dates, declared_columns].reset_index(drop=True))
        future_calendar = future_calendar.join(trusted_calendar.loc[future_dates, declared_columns].reset_index(drop=True))

    for target in TARGET_COLUMNS:
        model = model_factory(target)
        model.fit(history.loc[:, [DATE_COLUMN, target, *calendar_columns]])
        forecast = model.predict(future_calendar.copy())
        if not isinstance(forecast, pd.Series) or not forecast.index.equals(predictions.index):
            raise ValueError("Model predictions must be a Series aligned to the future calendar rows")
        if not (pd.api.types.is_integer_dtype(forecast) or pd.api.types.is_float_dtype(forecast)):
            raise ValueError("Model predictions must be numeric amounts")
        amounts = forecast.to_numpy(dtype=float)
        if not np.isfinite(amounts).all() or (amounts < 0).any():
            raise ValueError("Model predictions must be finite and nonnegative")
        predictions[target] = amounts
    return predictions


def predict_fold(
    daily: pd.DataFrame, fold: Fold, spec: BaselineSpec, rolling_days: int | None = None
) -> pd.DataFrame:
    history = training_history(daily, fold.cutoff, rolling_days)
    return predict_targets(history, fold.future_dates, partial(SeasonalMeanBaseline, spec))


def score_target(actual: pd.Series, predicted: pd.Series) -> tuple[dict[str, float | int], np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not actual.size:
        raise ValueError("Metrics require nonempty one-dimensional arrays of equal length")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs must be finite")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("Metric inputs must be nonnegative")

    absolute = np.abs(predicted - actual)
    valid = actual > 0
    relative = np.full(actual.size, np.nan)
    np.divide(absolute, actual, out=relative, where=valid)
    defined = relative[valid]
    total_actual = actual.sum()
    metrics = {
        "mape": float(defined.mean()) if defined.size else np.nan,
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(absolute).mean())),
        "wape": float(absolute.sum() / total_actual) if total_actual else np.nan,
        "bad_day_rate": float((defined > BAD_DAY_THRESHOLD).mean()) if defined.size else np.nan,
        "relative_error_days": int(defined.size),
    }
    return metrics, relative


def score_forecast(actual: pd.DataFrame, predicted: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    if (DATE_COLUMN in actual) != (DATE_COLUMN in predicted):
        raise ValueError("Provide dates on both the actual and predicted series")
    if DATE_COLUMN in actual:
        dates = pd.DatetimeIndex(actual[DATE_COLUMN])
        if (
            dates.hasnans
            or dates.has_duplicates
            or not dates.is_monotonic_increasing
            or not dates.equals(pd.DatetimeIndex(predicted[DATE_COLUMN]))
        ):
            raise ValueError("Actual and predicted dates must be unique, ordered and aligned")
    result = {}
    daily_errors = {}
    for target in TARGET_COLUMNS:
        target_metrics, relative = score_target(actual[target], predicted[target])
        result.update({f"{target}_{name}": value for name, value in target_metrics.items()})
        daily_errors[f"{target}_relative_error"] = relative

    result["weighted_mape"] = sum(TARGET_WEIGHTS[target] * result[f"{target}_mape"] for target in TARGET_COLUMNS)
    return result, pd.DataFrame(daily_errors)


def summarize_selection(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    selection = fold_metrics.loc[fold_metrics["role"] == "selection"]
    grouped = selection.groupby("model")
    if selection.empty or (grouped.size() != SELECTION_FOLDS).any() or (
        grouped["cutoff"].nunique() != SELECTION_FOLDS
    ).any():
        raise ValueError("Model selection requires four distinct selection folds per model")
    if not np.isfinite(selection["weighted_mape"]).all():
        raise ValueError("Model selection requires a finite weighted MAPE for each selection fold")

    metrics = ["weighted_mape"]
    metrics.extend(
        f"{target}_{metric}"
        for target in TARGET_COLUMNS
        for metric in ("mape", "mae", "rmse", "wape", "bad_day_rate")
        if f"{target}_{metric}" in selection
    )
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values(["weighted_mape_mean", "model"]).reset_index(drop=True)
