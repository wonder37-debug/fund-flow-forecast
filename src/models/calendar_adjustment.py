"""Conservative calendar adjustments to a single-target seasonal baseline."""

import json
from collections.abc import Callable
from typing import Self

import numpy as np
import pandas as pd

from src.evaluation import SingleTargetModel
from src.features.calendar import CONFIG, build_calendar
from src.models.baselines import DATE_COLUMN

CALENDAR_COLUMNS = ("weekday", "is_holiday", "is_makeup_workday", "is_pre_holiday", "is_post_holiday")
_DAY = pd.Timedelta(days=1)
_EDGE_DAYS = 3
_MIN_HISTORY_DAYS = 14
_PRIOR_EVENTS = 3
_RATIO_LIMITS = (0.25, 4.0)


def _event_days(calendar: pd.DataFrame) -> dict[str, pd.Series]:
    ordinary = ~calendar["is_holiday"] & ~calendar["is_makeup_workday"]
    pre = calendar["is_pre_holiday"]
    post = calendar["is_post_holiday"]
    dates = calendar[DATE_COLUMN].dt
    return {
        "holiday": calendar["is_holiday"],
        "pre": ordinary & pre,
        "post": ordinary & post & ~pre,
        "month_end": ordinary & ~pre & ~post & (dates.days_in_month - dates.day < 3),
    }


class CalendarAdjusted:
    def __init__(
        self,
        target: str,
        base_factory: Callable[[str], SingleTargetModel],
        *,
        holidays: bool = True,
        makeup: str = "weekday",
        edges: bool = False,
        month_end: bool = False,
    ) -> None:
        if makeup not in {"weekday", "workday"}:
            raise ValueError(f"Unknown makeup treatment: {makeup}")
        self.target = target
        self.base_factory = base_factory
        self.holidays = holidays
        self.makeup = makeup
        self.edges = edges
        self.month_end = month_end
        self._cutoff: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> Self:
        """Learn one shrunk log-ratio per event category from prior event forecasts."""
        history = history.loc[:, [DATE_COLUMN, self.target, *CALENDAR_COLUMNS]].copy()
        history[DATE_COLUMN] = pd.to_datetime(history[DATE_COLUMN])
        base = self.base_factory(self.target)
        base.fit(history)
        history = history.set_index(DATE_COLUMN, drop=False)
        cutoff = history.index[-1]
        event_masks = _event_days(history)
        ratios: dict[str, list[float]] = {}

        def add_window(
            kind: str, start: pd.Timestamp, end: pd.Timestamp, reference_start: pd.Timestamp
        ) -> None:
            reference = history.loc[history.index < reference_start]
            window = history.loc[start:end]
            if len(reference) < _MIN_HISTORY_DAYS or len(window) != (end - start).days + 1:
                return
            event = window.loc[event_masks[kind].loc[start:end]]
            if event.empty:
                return
            base_at_event = self.base_factory(self.target)
            base_at_event.fit(reference.reset_index(drop=True))
            future = event.loc[:, [DATE_COLUMN, *CALENDAR_COLUMNS]].reset_index(drop=True)
            amounts = base_at_event.predict(future).to_numpy(dtype=float)
            valid = amounts > 0
            if valid.any():
                observed = event[self.target].to_numpy(dtype=float)[valid]
                ratio = np.clip(observed / amounts[valid], *_RATIO_LIMITS)
                ratios.setdefault(kind, []).append(float(np.median(np.log(ratio))))

        if self.holidays or self.edges:
            schedule = json.loads(CONFIG.read_text(encoding="utf-8"))["holidays"]
            for event in schedule:
                start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
                if end > cutoff:
                    continue
                if self.holidays:
                    add_window("holiday", start, end, start)
                if self.edges:
                    edge_start = start - _EDGE_DAYS * _DAY
                    add_window("pre", edge_start, start - _DAY, edge_start)
                    add_window("post", end + _DAY, end + _EDGE_DAYS * _DAY, edge_start)

        if self.month_end:
            for end in pd.date_range(history.index[0], cutoff, freq=pd.offsets.MonthEnd()):
                start = end - 2 * _DAY
                add_window("month_end", start, end, start)

        self._multipliers = {
            kind: float(np.exp(np.median(values) * len(values) / (len(values) + _PRIOR_EVENTS)))
            for kind, values in ratios.items()
        }
        self._base = base
        self._cutoff = cutoff
        return self

    def predict(self, future_calendar: pd.DataFrame) -> pd.Series:
        if self._cutoff is None:
            raise RuntimeError("Fit the calendar-adjusted baseline before forecasting")
        future = future_calendar.loc[:, [DATE_COLUMN, *CALENDAR_COLUMNS]].copy()
        future[DATE_COLUMN] = pd.to_datetime(future[DATE_COLUMN])
        forecast = self._base.predict(future).astype(float).copy()

        for kind, days in _event_days(future).items():
            forecast.loc[days] *= self._multipliers.get(kind, 1.0)

        makeup_days = future["is_makeup_workday"]
        if self.makeup == "workday" and makeup_days.any():
            reference = build_calendar(self._cutoff + _DAY, self._cutoff + 7 * _DAY)
            reference_forecast = self._base.predict(reference.loc[:, [DATE_COLUMN, *CALENDAR_COLUMNS]])
            profile = reference_forecast.set_axis(reference["weekday"])
            workday_level = float(profile.loc[profile.index < 5].mean())
            original_level = profile.reindex(future.loc[makeup_days, "weekday"]).to_numpy()
            positive = original_level > 0
            comparison = np.full(len(original_level), workday_level)
            comparison[positive] *= forecast.loc[makeup_days].to_numpy()[positive] / original_level[positive]
            forecast.loc[makeup_days] = comparison

        return forecast
