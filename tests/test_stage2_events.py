import pandas as pd

from experiments.stage2_events import evaluation_events


def test_holiday_is_predicted_as_one_block_even_across_month_boundary():
    events = evaluation_events()
    dragon_boat = next(event for event in events if event.name == "dragon_boat")
    assert dragon_boat.fold.cutoff == pd.Timestamp("2014-05-30")
    assert dragon_boat.fold.future_dates.equals(pd.date_range("2014-05-31", "2014-06-02"))
    assert dragon_boat.fold.role == "development"


def test_historical_makeup_days_are_distinguished_from_development_holdouts():
    makeup = [event for event in evaluation_events() if event.kind == "makeup"]
    assert len(makeup) == 6
    assert all(len(event.fold.future_dates) == 1 for event in makeup)
    assert all(event.fold.cutoff < event.fold.future_dates[0] for event in makeup)
    assert [str(event.fold.future_dates[0].date()) for event in makeup if event.fold.role == "development"] == [
        "2014-05-04"
    ]
