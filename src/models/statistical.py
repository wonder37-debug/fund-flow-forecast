"""Cutoff-fitted statistical models for known future dates."""

import warnings
from numbers import Integral
from typing import Self

import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.models.baselines import DATE_COLUMN, TARGET_COLUMNS


def _positive_days(name: str, days: int) -> int:
    if isinstance(days, bool) or not isinstance(days, Integral) or days < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(days)


class _DateModel:
    def __init__(self, target: str) -> None:
        if target not in TARGET_COLUMNS:
            raise ValueError(f"Unknown target: {target}")
        self.target = target
        self._cutoff: pd.Timestamp | None = None

    def _training_values(self, history: pd.DataFrame, days: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
        dates = pd.DatetimeIndex(history[DATE_COLUMN])
        if not len(dates) or dates.hasnans or not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError("History dates must be unique, increasing and nonempty")
        values = history[self.target].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("History targets must be finite and nonnegative")
        recent = dates > dates[-1] - pd.Timedelta(days=days)
        return dates[recent], values[recent]

    def _forecast_dates(self, future_calendar: pd.DataFrame) -> pd.DatetimeIndex:
        if self._cutoff is None:
            raise RuntimeError("Fit the model before forecasting")
        dates = pd.DatetimeIndex(future_calendar[DATE_COLUMN])
        if not len(dates) or dates.hasnans or not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError("Forecast dates must be unique, increasing and nonempty")
        if (dates <= self._cutoff).any():
            raise ValueError("Forecast dates must be after the training cutoff")
        return dates

    def _prediction(self, future_calendar: pd.DataFrame, values: np.ndarray) -> pd.Series:
        if values.shape != (len(future_calendar),) or not np.isfinite(values).all() or (values < 0).any():
            raise RuntimeError("Forecast must contain one finite nonnegative amount per date")
        return pd.Series(values, index=future_calendar.index, name=self.target)


class RecentWeekday(_DateModel):
    """Estimate each weekday from a trailing window measured in natural days."""

    def __init__(
        self, target: str, days: int, *, estimator: str = "mean", half_life: float | None = None
    ) -> None:
        super().__init__(target)
        self.days = _positive_days("days", days)
        if estimator not in {"mean", "median", "trimmed", "relative_median"}:
            raise ValueError(f"Unknown weekday estimator: {estimator}")
        if half_life is not None and (not np.isfinite(half_life) or half_life <= 0):
            raise ValueError("half_life must be finite and positive")
        self.estimator = estimator
        self.half_life = half_life

    def _estimate(self, values: np.ndarray, dates: pd.DatetimeIndex) -> float:
        order = np.argsort(values)
        values = values[order]
        dates = dates[order]
        if self.estimator == "trimmed":
            # Trim 20% of observations from each tail within the weekday.
            trim = len(values) // 5
            values, dates = values[trim : len(values) - trim], dates[trim : len(dates) - trim]
        if self.estimator == "relative_median":
            positive = values > 0
            if not positive.any():
                return 0.0
            values, dates = values[positive], dates[positive]

        log_weights = np.zeros(len(values))
        if self.half_life is not None:
            ages = ((dates.max() - dates) / pd.Timedelta(days=1)).to_numpy()
            with np.errstate(over="ignore"):
                log_weights = -ages / self.half_life
        if self.estimator == "relative_median":
            # Relative error excludes zero actuals. Floor the denominator so one
            # unusually low observation cannot carry unbounded relative weight.
            denominators = np.maximum(values, np.median(values / 2))
            log_weights -= np.log2(denominators)
        # Combine both weights before exponentiating so at least one stays positive.
        weights = np.exp2(log_weights - log_weights.max())
        if self.estimator in {"mean", "trimmed"}:
            return float(np.average(values, weights=weights))
        cumulative = np.cumsum(weights)
        midpoint = cumulative[-1] / 2
        lower = cumulative.searchsorted(midpoint, side="left")
        upper = cumulative.searchsorted(midpoint, side="right")
        return float(values[lower] / 2 + values[upper] / 2)

    def fit(self, history: pd.DataFrame) -> Self:
        dates, values = self._training_values(history, self.days)
        weekdays = dates.dayofweek.to_numpy()
        if self.estimator == "mean" and self.half_life is None:
            # Match the stage-one aggregation and its missing-weekday fallback.
            fallback = float(values.mean())
            estimates = pd.Series(values).groupby(weekdays).mean()
        else:
            fallback = self._estimate(values, dates)
            estimates_by_day = {}
            for weekday in np.unique(weekdays):
                same_day = weekdays == weekday
                estimates_by_day[weekday] = self._estimate(values[same_day], dates[same_day])
            estimates = pd.Series(estimates_by_day)
        self._weekdays = estimates.reindex(range(7), fill_value=fallback).to_numpy()
        self._cutoff = dates[-1]
        return self

    def predict(self, future_calendar: pd.DataFrame) -> pd.Series:
        dates = self._forecast_dates(future_calendar)
        return self._prediction(future_calendar, self._weekdays[dates.dayofweek])


class LevelWeekday(_DateModel):
    """Combine recent deseasonalized level with a longer multiplicative weekly shape."""

    _TREND_HALF_LIFE = 14
    _MAX_TREND_CHANGE = 0.25

    def __init__(
        self, target: str, *, level_days: int = 28, profile_days: int = 84, trend: bool = False
    ) -> None:
        super().__init__(target)
        self.level_days = _positive_days("level_days", level_days)
        self.profile_days = _positive_days("profile_days", profile_days)
        self.trend = trend

    def fit(self, history: pd.DataFrame) -> Self:
        dates, values = self._training_values(history, max(self.level_days, self.profile_days))
        ages = (dates[-1] - dates).days.to_numpy()
        weekdays = dates.dayofweek.to_numpy()
        profile_window = ages < self.profile_days
        profile_values = pd.Series(values[profile_window])
        weeks = profile_values.groupby(ages[profile_window] // 7)
        week_medians = weeks.transform("median").to_numpy()
        # Complete weeks give all weekdays equal opportunity to define the shape.
        usable = (weeks.transform("size").to_numpy() == 7) & (week_medians > 0)
        ratios = profile_values.to_numpy()[usable] / week_medians[usable]
        profile = (
            pd.Series(ratios)
            .groupby(weekdays[profile_window][usable])
            .median()
            .reindex(range(7), fill_value=1.0)
            .to_numpy()
        )
        shape = profile[weekdays]
        recent = (ages < self.level_days) & (shape > 0)
        deseasonalized = values[recent] / shape[recent]
        level = float(np.median(deseasonalized)) if len(deseasonalized) else 0.0
        slope = 0.0
        if self.trend and len(deseasonalized) > 1:
            slope = float(theilslopes(deseasonalized, -ages[recent]).slope)
        self._profile = profile
        self._level = level
        self._slope = slope
        self._cutoff = dates[-1]
        return self

    def predict(self, future_calendar: pd.DataFrame) -> pd.Series:
        dates = self._forecast_dates(future_calendar)
        horizon = ((dates - self._cutoff) / pd.Timedelta(days=1)).to_numpy()
        decay = np.log(2) / self._TREND_HALF_LIFE
        damped_days = -np.expm1(-decay * horizon) / decay
        max_change = self._level * self._MAX_TREND_CHANGE
        level_change = np.clip(self._slope * damped_days, -max_change, max_change)
        forecast = (self._level + level_change) * self._profile[dates.dayofweek]
        return self._prediction(future_calendar, forecast)


class WeeklyETS(_DateModel):
    """Holt–Winters smoothing in log1p space with additive weekly seasonality."""

    def __init__(self, target: str, *, days: int = 120) -> None:
        super().__init__(target)
        self.days = _positive_days("days", days)

    def fit(self, history: pd.DataFrame) -> Self:
        dates, values = self._training_values(history, self.days)
        if len(dates) < 14 or not dates.equals(pd.date_range(dates[0], dates[-1], freq="D")):
            raise ValueError("Weekly ETS needs at least 14 consecutive daily observations")
        try:
            model = ExponentialSmoothing(
                np.log1p(values), trend="add", damped_trend=True, seasonal="add", seasonal_periods=7
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                fitted = model.fit()
        except (ConvergenceWarning, ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"Weekly ETS fitting failed for {self.target}: {exc}") from exc
        if not fitted.mle_retvals.success or not np.isfinite(fitted.fittedvalues).all():
            raise RuntimeError(f"Weekly ETS fitting failed for {self.target}: invalid fitted state")
        self._fitted = fitted
        self._cutoff = dates[-1]
        return self

    def predict(self, future_calendar: pd.DataFrame) -> pd.Series:
        dates = self._forecast_dates(future_calendar)
        elapsed = dates - self._cutoff
        if (elapsed % pd.Timedelta(days=1) != pd.Timedelta(0)).any():
            raise ValueError("Weekly ETS forecast dates must be whole days after the training cutoff")
        offsets = elapsed.days.to_numpy()
        try:
            log_forecast = np.asarray(self._fitted.forecast(offsets[-1]), dtype=float)
            if log_forecast.shape != (offsets[-1],) or not np.isfinite(log_forecast).all():
                raise ValueError("invalid forecasting state")
            # Project log1p forecasts onto the nonnegative amount domain.
            with np.errstate(over="raise", invalid="raise"):
                values = np.expm1(np.maximum(log_forecast[offsets - 1], 0))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"Weekly ETS forecasting failed for {self.target}: {exc}") from exc
        return self._prediction(future_calendar, values)
