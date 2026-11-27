from functools import partial

import numpy as np
import pandas as pd
import pytest

from src.evaluation import predict_targets
from src.features.calendar import build_calendar
from src.models.baselines import BASELINES, SeasonalMeanBaseline
from src.models.calendar_adjustment import CALENDAR_COLUMNS, CalendarAdjusted


def _history(start, end):
    dates = pd.date_range(start, end)
    return pd.DataFrame({"report_date": dates, "purchase": 100.0, "redeem": 200.0})


def _forecast(history, future_dates, baseline="B0", **options):
    factory = partial(SeasonalMeanBaseline, BASELINES[baseline])
    return predict_targets(
        history,
        pd.DatetimeIndex(future_dates),
        lambda target: CalendarAdjusted(target, factory, **options),
        calendar_columns=CALENDAR_COLUMNS,
    )


@pytest.mark.parametrize("cutoff,makeup_day,regular_sunday", [
    ("2014-04-30", "2014-05-04", "2014-05-11"),
    ("2014-07-31", "2014-09-28", "2014-09-21"),
])
def test_makeup_uses_weekday_profile_only_on_the_official_makeup_date(cutoff, makeup_day, regular_sunday):
    history = _history("2014-04-01", cutoff)
    weekend = history["report_date"].dt.dayofweek.ge(5)
    history.loc[weekend, ["purchase", "redeem"]] = [30.0, 60.0]
    future_dates = sorted([makeup_day, regular_sunday])

    raw = _forecast(history, future_dates, baseline="B3", holidays=False, makeup="weekday")
    workday = _forecast(history, future_dates, baseline="B3", holidays=False, makeup="workday")

    assert raw["purchase"].tolist() == [30.0, 30.0]
    expected = [100.0 if day == makeup_day else 30.0 for day in future_dates]
    assert workday["purchase"].tolist() == pytest.approx(expected)
    assert workday["redeem"].tolist() == pytest.approx([value * 2 for value in expected])


def test_holiday_ratio_is_shrunk_and_calibrated_before_the_historical_event():
    history = _history("2013-12-01", "2014-01-20")
    history.loc[history["report_date"].eq("2014-01-01"), "purchase"] = 40.0
    changed = history.copy()
    changed.loc[changed["report_date"].gt("2014-01-01"), "purchase"] = 1000.0
    dates = ["2014-01-30", "2014-01-31"]

    neutral = _forecast(history, dates, holidays=False)
    result = _forecast(history, dates)
    changed_neutral = _forecast(changed, dates, holidays=False)
    changed_result = _forecast(changed, dates)
    ratio = result["purchase"] / neutral["purchase"]

    assert ratio.iat[0] == 1.0
    assert 0.40 < ratio.iat[1] < 1.0
    np.testing.assert_allclose(ratio, changed_result["purchase"] / changed_neutral["purchase"])
    assert result["redeem"].tolist() == [200.0, 200.0]


def test_partial_holiday_at_cutoff_cannot_calibrate_a_holiday_ratio():
    history = _history("2014-04-10", "2014-05-01")
    history.loc[history["report_date"].eq("2014-05-01"), "purchase"] = 1.0

    neutral = _forecast(history, ["2014-05-02"], holidays=False)
    result = _forecast(history, ["2014-05-02"])

    pd.testing.assert_frame_equal(result, neutral)


def test_holiday_ending_on_the_cutoff_is_available_for_calibration():
    history = _history("2014-04-10", "2014-05-03")
    history.loc[history["report_date"].ge("2014-05-01"), "purchase"] = 40.0

    neutral = _forecast(history, ["2014-05-31"], holidays=False)
    result = _forecast(history, ["2014-05-31"])

    assert 0.40 < result["purchase"].iat[0] / neutral["purchase"].iat[0] < 1.0


def test_edges_can_change_regular_pre_holiday_days_without_changing_makeup():
    history = _history("2013-12-01", "2014-07-31")
    calendar = build_calendar("2013-12-01", "2014-07-31")
    history.loc[calendar["is_pre_holiday"] & ~calendar["is_makeup_workday"], "purchase"] = 50.0
    future_dates = ["2014-09-28", "2014-09-29"]

    baseline = _forecast(history, future_dates, makeup="workday")
    adjusted = _forecast(history, future_dates, makeup="workday", edges=True)

    assert adjusted["purchase"].iat[0] == baseline["purchase"].iat[0]
    assert adjusted["purchase"].iat[1] < baseline["purchase"].iat[1]


def test_month_end_changes_only_ordinary_last_three_dates():
    history = _history("2014-06-01", "2014-07-31")
    dates = history["report_date"].dt
    history.loc[dates.days_in_month - dates.day < 3, "purchase"] = 150.0
    future_dates = ["2014-08-28", "2014-08-29", "2014-08-31"]

    neutral = _forecast(history, future_dates, holidays=False)
    result = _forecast(history, future_dates, holidays=False, month_end=True)

    assert result["purchase"].iat[0] == neutral["purchase"].iat[0]
    assert (result["purchase"].iloc[1:] > neutral["purchase"].iloc[1:]).all()


def test_zero_baseline_keeps_all_outputs_finite_and_nonnegative():
    history = _history("2013-12-01", "2014-04-30")
    history[["purchase", "redeem"]] = 0.0
    result = _forecast(
        history, pd.date_range("2014-05-01", "2014-05-05"), makeup="workday", edges=True, month_end=True
    )
    assert (result[["purchase", "redeem"]] == 0.0).all().all()
