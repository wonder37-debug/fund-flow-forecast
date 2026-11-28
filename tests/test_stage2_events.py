import json

import pandas as pd

from experiments.stage2_events import evaluation_events, main, run_diagnostics


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


def _daily_through_august_start():
    dates = pd.date_range("2013-07-01", "2014-08-01")
    return pd.DataFrame({"report_date": dates, "purchase": 100 + dates.dayofweek, "redeem": 200 + dates.dayofweek})


def test_diagnostic_fingerprint_excludes_august_rows(tmp_path):
    daily = _daily_through_august_start()
    run_diagnostics(daily.iloc[:-1], tmp_path / "development")
    daily.loc[daily.index[-1], ["purchase", "redeem"]] = [9000, 8000]
    run_diagnostics(daily, tmp_path / "extended")

    before = json.loads((tmp_path / "development" / "manifest.json").read_text(encoding="utf-8"))
    after = json.loads((tmp_path / "extended" / "manifest.json").read_text(encoding="utf-8"))
    assert after["development_input_end"] == "2014-07-31"
    assert after["development_input_sha256"] == before["development_input_sha256"]
    for name in ("event_metrics.csv", "event_predictions.csv"):
        assert (tmp_path / "extended" / name).read_bytes() == (tmp_path / "development" / name).read_bytes()


def test_diagnostic_cli_leaves_unavailable_august_targets_unread(tmp_path, monkeypatch):
    daily = _daily_through_august_start()
    daily["purchase"] = daily["purchase"].astype("string")
    daily.loc[daily.index[-1], "purchase"] = "unavailable"
    path = tmp_path / "daily.csv"
    output = tmp_path / "output"
    daily.to_csv(path, index=False)
    monkeypatch.setattr("sys.argv", ["stage2_events", "--daily", str(path), "--output-dir", str(output)])

    main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["development_input_end"] == "2014-07-31"
