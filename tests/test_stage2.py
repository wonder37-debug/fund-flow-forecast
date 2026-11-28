import json

import pandas as pd
import pytest

from experiments.stage2 import DEVELOPMENT_END, _read_scoped_daily, run_development, run_holdout, select_candidate
from src.evaluation import build_folds, split_fold


def _write_daily(path, end="2014-08-31"):
    dates = pd.date_range("2013-07-01", end)
    pd.DataFrame(
        {
            "report_date": dates,
            "purchase": 100 + dates.dayofweek,
            "redeem": 200 + dates.dayofweek,
        }
    ).to_csv(path, index=False)


def _write_config(path):
    candidates = [
        {"id": "B3", "kind": "baseline", "specs": {"purchase": "B3", "redeem": "B3"}},
        {"id": "B3+B5", "kind": "baseline", "specs": {"purchase": "B3", "redeem": "B5"}},
        {"id": "invalid", "kind": "recent_weekday", "params": {"days": -1}},
    ]
    path.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")


def test_development_can_freeze_without_august_and_holdout_uses_same_candidates(tmp_path):
    daily = tmp_path / "daily.csv"
    config = tmp_path / "config.json"
    development = tmp_path / "development"
    holdout = tmp_path / "holdout"
    _write_daily(daily, "2014-07-31")
    _write_config(config)

    freeze_path = run_development(daily, config, development)

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert freeze["selected_model"] == "B3"
    assert [candidate["id"] for candidate in freeze["candidates"]] == ["B3", "B3+B5", "invalid"]
    predictions = pd.read_csv(development / "daily_predictions.csv")
    assert len(predictions) == 2 * 122
    assert predictions["report_date"].max() == "2014-07-31"
    assert predictions.loc[predictions.report_date == "2014-04-03", "is_ordinary"].all()
    assert not predictions.loc[predictions.report_date.isin(["2014-05-01", "2014-05-04"]), "is_ordinary"].any()
    failures = pd.read_csv(development / "failures.csv")
    assert failures["model"].tolist() == ["invalid"] * 4

    _write_daily(daily)
    run_holdout(daily, freeze_path, holdout)

    folds = pd.read_csv(holdout / "fold_metrics.csv")
    assert folds["model"].tolist() == ["B3", "B3+B5"]
    assert folds["forecast_days"].tolist() == [31, 31]
    assert folds["role"].tolist() == ["holdout", "holdout"]
    assert pd.read_csv(holdout / "failures.csv")["model"].tolist() == ["invalid"]
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_model"] == "B3"
    assert manifest["source_sha256"] == freeze["source_sha256"]
    assert manifest["calendar_sha256"] == freeze["calendar_sha256"]
    assert manifest["config_sha256"] == freeze["config_sha256"]
    assert not (holdout / "selection_summary.csv").exists()


@pytest.mark.parametrize("tamper", ["plan", "development_output", "development_input"])
def test_holdout_rejects_changes_to_frozen_evidence(tmp_path, tamper):
    daily = tmp_path / "daily.csv"
    config = tmp_path / "config.json"
    development = tmp_path / "development"
    holdout = tmp_path / "holdout"
    _write_daily(daily)
    _write_config(config)
    freeze_path = run_development(daily, config, development)
    if tamper == "plan":
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        freeze["selected_model"] = "B3+B5"
        freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    elif tamper == "development_output":
        (development / "selection_summary.csv").write_text("changed\n", encoding="utf-8")
    else:
        values = pd.read_csv(daily)
        values.loc[values.report_date == "2014-07-31", "purchase"] += 1
        values.to_csv(daily, index=False)

    with pytest.raises(ValueError, match="[Ff]rozen|[Ff]reeze"):
        run_holdout(daily, freeze_path, holdout)

    assert not holdout.exists()


def test_development_does_not_validate_august_amounts(tmp_path):
    daily = tmp_path / "daily.csv"
    config = tmp_path / "config.json"
    _write_daily(daily)
    _write_config(config)
    csv = daily.read_text(encoding="utf-8")
    daily.write_text(csv.replace("2014-08-01,104,204", "2014-08-01,unreadable,204"), encoding="utf-8")

    freeze_path = run_development(daily, config, tmp_path / "development")

    assert freeze_path.exists()
    with pytest.raises(ValueError, match="purchase"):
        run_holdout(daily, freeze_path, tmp_path / "holdout")


@pytest.mark.parametrize("unit", ["ms", "us"])
def test_parquet_datetime_unit_preserves_daily_split(tmp_path, unit):
    csv_path = tmp_path / "daily.csv"
    _write_daily(csv_path)
    parquet_path = tmp_path / "daily.parquet"
    daily = pd.read_csv(csv_path, parse_dates=["report_date"])
    daily["report_date"] = daily["report_date"].dt.as_unit(unit)
    daily.to_parquet(parquet_path, index=False)

    from_csv = _read_scoped_daily(csv_path, DEVELOPMENT_END)
    from_parquet = _read_scoped_daily(parquet_path, DEVELOPMENT_END)
    fold = build_folds()[0]
    for csv_frame, parquet_frame in zip(split_fold(from_csv, fold), split_fold(from_parquet, fold)):
        pd.testing.assert_frame_equal(parquet_frame, csv_frame)


def test_candidate_constructor_error_is_recorded_without_interrupting_other_models(tmp_path):
    daily = tmp_path / "daily.csv"
    config = tmp_path / "config.json"
    development = tmp_path / "development"
    _write_daily(daily)
    _write_config(config)
    settings = json.loads(config.read_text(encoding="utf-8"))
    settings["candidates"][-1]["params"] = {"days": 28, "unknown_parameter": True}
    config.write_text(json.dumps(settings), encoding="utf-8")

    freeze_path = run_development(daily, config, development)

    failures = pd.read_csv(development / "failures.csv")
    assert failures["model"].tolist() == ["invalid"] * 4
    assert failures["error_type"].tolist() == ["TypeError"] * 4
    assert json.loads(freeze_path.read_text(encoding="utf-8"))["selected_model"] == "B3"


def test_undefined_fold_mape_blocks_freeze_but_keeps_predictions(tmp_path):
    daily = tmp_path / "daily.csv"
    config = tmp_path / "config.json"
    development = tmp_path / "development"
    _write_daily(daily)
    _write_config(config)
    values = pd.read_csv(daily)
    values.loc[values.report_date.str.startswith("2014-04-"), "purchase"] = 0
    values.to_csv(daily, index=False)

    with pytest.raises(ValueError, match="No candidate completed all development folds"):
        run_development(daily, config, development)

    failures = pd.read_csv(development / "failures.csv")
    undefined = failures.loc[failures["error_type"].eq("UndefinedMetric")]
    assert undefined["model"].tolist() == ["B3", "B3+B5"]
    assert undefined["cutoff"].tolist() == ["2014-03-31"] * 2
    metrics = pd.read_csv(development / "fold_metrics.csv")
    assert metrics["cutoff"].min() == "2014-04-30"
    predictions = pd.read_csv(development / "daily_predictions.csv")
    assert predictions["report_date"].min() == "2014-04-01"
    assert not (development / "freeze.json").exists()


def test_holdout_reports_frozen_model_failure_when_august_metric_is_undefined(tmp_path):
    daily = tmp_path / "daily.csv"
    config = tmp_path / "config.json"
    holdout = tmp_path / "holdout"
    _write_daily(daily)
    _write_config(config)
    values = pd.read_csv(daily)
    values.loc[values.report_date.str.startswith("2014-08-"), "purchase"] = 0
    values.to_csv(daily, index=False)
    freeze_path = run_development(daily, config, tmp_path / "development")

    with pytest.raises(RuntimeError, match="frozen selected model failed on the holdout"):
        run_holdout(daily, freeze_path, holdout)

    assert pd.read_csv(holdout / "fold_metrics.csv").empty
    predictions = pd.read_csv(holdout / "daily_predictions.csv")
    assert len(predictions) == 62
    failures = pd.read_csv(holdout / "failures.csv")
    assert failures.loc[failures["error_type"].eq("UndefinedMetric"), "model"].tolist() == ["B3", "B3+B5"]


def test_selection_prefers_stability_only_within_half_a_percentage_point():
    summary = pd.DataFrame.from_records(
        [
            ("best-error", True, 0.20, 0.26, 0.04, 0.18),
            ("best-bad-days", True, 0.205, 0.21, 0.03, 0.19),
            ("best-bad-days-unstable", True, 0.202, 0.21, 0.05, 0.17),
            ("outside-band", True, 0.2051, 0.10, 0.02, 0.16),
            ("failed-fold", False, 0.18, 0.09, 0.02, 0.15),
        ],
        columns=[
            "model", "eligible", "weighted_mape_mean", "ordinary_weighted_bad_day_rate_mean",
            "ordinary_weighted_mape_std", "ordinary_weighted_mape_mean",
        ],
    )

    assert select_candidate(summary) == "best-bad-days"
