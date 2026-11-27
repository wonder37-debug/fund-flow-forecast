"""Single-target daily baselines fitted once at each forecast cutoff."""

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

DATE_COLUMN = "report_date"
TARGET_COLUMNS = ("purchase", "redeem")


@dataclass(frozen=True)
class BaselineSpec:
    code: str
    history_days: int | None
    group_by_weekday: bool

    @property
    def calendar_features(self) -> tuple[str, ...]:
        return ("weekday",) if self.group_by_weekday else ()


BASELINES = {
    spec.code: spec
    for spec in (
        BaselineSpec("B0", None, False),
        BaselineSpec("B1", 28, False),
        BaselineSpec("B2", None, True),
        BaselineSpec("B3", 28, True),
        BaselineSpec("B4", 56, True),
        BaselineSpec("B5", 84, True),
    )
}


class SeasonalMeanBaseline:
    def __init__(self, spec: BaselineSpec, target: str) -> None:
        if target not in TARGET_COLUMNS:
            raise ValueError(f"Unknown target: {target}")
        self.spec = spec
        self.target = target
        self._cutoff: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> Self:
        """Consume the date and this model's target from observed history."""
        rows = history.loc[:, [DATE_COLUMN, self.target]]
        dates = pd.DatetimeIndex(rows[DATE_COLUMN])
        if not len(dates) or dates.hasnans or not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError("History dates must be unique, increasing and nonempty")
        values = rows[self.target].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("History targets must be finite and nonnegative")

        cutoff = dates[-1]
        if self.spec.history_days is not None:
            recent = dates > cutoff - pd.Timedelta(days=self.spec.history_days)
            dates = dates[recent]
            values = values[recent]

        self._mean = float(values.mean())
        if self.spec.group_by_weekday:
            self._weekday_means = pd.Series(values).groupby(dates.dayofweek).mean()
        self._cutoff = cutoff
        return self

    def predict(self, future_calendar: pd.DataFrame) -> pd.Series:
        """Forecast every horizon using the state at the same cutoff."""
        if self._cutoff is None:
            raise RuntimeError("Fit the baseline before forecasting")
        dates = pd.DatetimeIndex(future_calendar[DATE_COLUMN])
        if dates.hasnans or not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError("Forecast dates must be unique and increasing")
        if (dates <= self._cutoff).any():
            raise ValueError("Forecast dates must be after the training cutoff")

        if self.spec.group_by_weekday:
            values = self._weekday_means.reindex(dates.dayofweek).fillna(self._mean).to_numpy()
        else:
            values = np.full(len(dates), self._mean)
        return pd.Series(values, index=future_calendar.index, name=self.target)
