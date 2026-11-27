import pandas as pd
import pytest

from src.features.calendar import build_calendar


def test_september_calendar_distinguishes_holiday_makeup_and_weekend():
    calendar = build_calendar("2014-09-01", "2014-09-30").set_index("report_date")
    assert len(calendar) == 30
    assert calendar.loc["2014-09-06", "is_holiday"]
    assert not calendar.loc["2014-09-06", "is_statutory_holiday"]
    assert calendar.loc["2014-09-08", "is_statutory_holiday"]
    assert not calendar.loc["2014-09-08", "is_workday"]
    assert calendar.loc["2014-09-14", "is_regular_weekend"]
    assert calendar.loc["2014-09-28", "is_makeup_workday"]
    assert calendar.loc["2014-09-28", "is_workday"]
    assert not calendar.loc["2014-09-28", "is_regular_weekend"]


def test_holiday_distance_is_independent_of_requested_window():
    day = build_calendar("2014-09-30", "2014-09-30").iloc[0]
    assert day["days_to_holiday"] == 1
    assert day["next_holiday"] == "national_day"
    assert day["is_pre_holiday"]


def test_fixed_shopping_markers_and_2013_makeup_days():
    calendar = build_calendar("2013-07-01", "2014-09-30").set_index("report_date")
    assert len(calendar) == 457
    assert calendar.loc["2013-09-22", "is_workday"]
    assert calendar.loc["2013-11-11", "shopping_event"] == "double_11"
    assert calendar.loc["2014-06-18", "shopping_event"] == "618"
    assert pd.isna(calendar.loc["2014-06-19", "shopping_event"])


@pytest.mark.parametrize("start,end", [("2014-10-08", "2014-10-08"), ("2014-09-02", "2014-09-01")])
def test_rejects_dates_without_calendar_coverage(start, end):
    with pytest.raises(ValueError):
        build_calendar(start, end)
