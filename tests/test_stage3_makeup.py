import pandas as pd
import pytest

from src.features.calendar import build_calendar


def _history() -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = build_calendar("2013-07-01", "2013-10-31")
    daily = calendar[["report_date"]].copy()
    amount = calendar["weekday"].lt(5).map({True: 200, False: 100})
    daily = daily.assign(transfer_amt=amount, tftocard_amt=amount, redeem=amount)
    return daily, calendar


def test_read_daily_filters_out_september_labels(tmp_path):
    from experiments.stage3_makeup import read_daily

    source = tmp_path / "daily.parquet"
    daily = pd.DataFrame(
        {
            "report_date": pd.to_datetime(["2014-08-31", "2014-09-01"]),
            "transfer_amt": [1, 999],
            "tftocard_amt": [1, 999],
            "redeem": [1, 999],
        }
    )
    daily.to_parquet(source, index=False)

    loaded = read_daily(source)

    assert loaded["report_date"].tolist() == [pd.Timestamp("2014-08-31")]
    assert loaded["redeem"].tolist() == [1]


def test_national_day_sunday_compares_prior_ordinary_sunday_monday_pairs():
    from experiments.stage3_makeup import estimate_windows

    daily, calendar = _history()
    daily.loc[daily["report_date"].eq("2013-09-22"), "transfer_amt"] = 100_000
    daily.loc[daily["report_date"].eq("2013-09-29"), "transfer_amt"] = 150
    daily.loc[daily["report_date"].eq("2013-09-30"), "transfer_amt"] = 150
    daily.loc[daily["report_date"].gt("2013-09-30"), "transfer_amt"] = 100_000

    results = estimate_windows(daily, calendar, lookback_days=56, followup_days=1)
    row = results.loc[results["event_date"].eq("2013-09-29") & results["metric"].eq("transfer_amt")].iloc[0]

    assert row["followup_dates"] == "2013-09-30"
    assert row["followup_pre_holiday"] and row["followup_month_end"]
    assert row["reference_anchors"] == (
        "2013-08-04|2013-08-11|2013-08-18|2013-08-25|2013-09-01|2013-09-08|2013-09-15"
    )
    assert row["reference_event_cents"] == 100
    assert row["reference_followup_cents"] == 200
    assert row["event_lift_pct"] == pytest.approx(50)
    assert row["followup_lift_pct"] == pytest.approx(-25)
    assert row["window_lift_pct"] == pytest.approx(0)
    assert row["offset_share_pct"] == pytest.approx(100)
    assert row["pattern"] == "shift_compatible"


def test_saturday_makeup_matches_two_day_gap_to_monday():
    from experiments.stage3_makeup import estimate_windows

    daily, calendar = _history()
    results = estimate_windows(daily, calendar, lookback_days=56, followup_days=1)
    row = results.loc[results["event_date"].eq("2013-10-12") & results["metric"].eq("redeem")].iloc[0]

    assert row["followup_dates"] == "2013-10-14"
    assert row["followup_offsets_days"] == "2"
    assert row["reference_anchors"] == "2013-08-17|2013-08-24|2013-08-31|2013-09-07|2013-09-14"
    assert row["reference_followup_cents"] == 200


def test_longer_window_that_crosses_national_day_is_reported_without_estimate():
    from experiments.stage3_makeup import estimate_windows

    daily, calendar = _history()
    results = estimate_windows(daily, calendar, lookback_days=56, followup_days=2)
    row = results.loc[results["event_date"].eq("2013-09-29") & results["metric"].eq("transfer_amt")].iloc[0]

    assert row["followup_dates"] == "2013-09-30|2013-10-08"
    assert row["followup_offsets_days"] == "1|9"
    assert row["status"] == "crosses_public_holiday"
    assert pd.isna(row["window_lift_pct"])


def test_window_overlapping_another_makeup_is_reported_without_estimate():
    from experiments.stage3_makeup import estimate_windows

    daily, calendar = _history()
    calendar.loc[calendar["report_date"].eq("2013-09-30"), "is_makeup_workday"] = True
    results = estimate_windows(daily, calendar, lookback_days=56, followup_days=1)
    row = results.loc[results["event_date"].eq("2013-09-29") & results["metric"].eq("transfer_amt")].iloc[0]

    assert row["followup_dates"] == "2013-10-08"
    assert row["status"] == "overlaps_makeup"
    assert pd.isna(row["window_lift_pct"])


@pytest.mark.parametrize(
    ("event", "followup", "expected"),
    [
        (200, 160, "partial_offset_with_surplus"),
        (160, 200, "surplus_without_offset"),
        (140, 120, "offset_with_shortfall"),
        (105, 175, "no_clear_makeup_uplift"),
    ],
)
def test_pattern_distinguishes_compensation_and_window_surplus(event, followup, expected):
    from experiments.stage3_makeup import classify_pattern

    assert classify_pattern(event, followup, 100, 200) == expected
