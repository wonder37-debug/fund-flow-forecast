import csv
import json
from hashlib import sha256

import pandas as pd
import pytest

from experiments.stage2 import run_development, run_holdout
from experiments import stage2_forecast
from experiments.stage2_forecast import run_forecast


def _expected_hashes(freeze, holdout):
    return {
        "expected_freeze_sha256": sha256(freeze.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "expected_holdout_sha256": sha256(holdout.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    }


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_artifact(holdout, filename, table):
    path = holdout.parent / filename
    table.to_csv(path, index=False, lineterminator="\n")
    manifest = json.loads(holdout.read_text(encoding="utf-8"))
    manifest["holdout_artifact_sha256"][filename] = sha256(path.read_bytes()).hexdigest()
    _write_manifest(holdout, manifest)


@pytest.fixture
def frozen_run(tmp_path):
    daily = tmp_path / "daily.csv"
    dates = pd.date_range("2013-07-01", "2014-09-01")
    pd.DataFrame({"report_date": dates, "purchase": 100 + dates.dayofweek, "redeem": 200 + dates.dayofweek}).to_csv(
        daily, index=False,
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"candidates": [
        {"id": "blend_equal", "kind": "baseline", "specs": {"purchase": "B3", "redeem": "B3"}},
        {"id": "cal_workday", "kind": "baseline", "specs": {"purchase": "B5", "redeem": "B5"}},
        {"id": "other", "kind": "baseline", "specs": {"purchase": "B3", "redeem": "B3"}},
    ]}), encoding="utf-8")
    freeze = run_development(daily, config, tmp_path / "development")
    holdout = run_holdout(daily, freeze, tmp_path / "holdout")
    sample = tmp_path / "sample.csv"
    sample.write_text("20140901,40000000,30000000\n20140902,40000000,30000000\n", encoding="utf-8")
    return daily, freeze, holdout, sample


def test_forecast_writes_confirmed_roles_and_ignores_september_actuals(tmp_path, frozen_run):
    daily, freeze, holdout, sample = frozen_run
    expected = _expected_hashes(freeze, holdout)
    output = tmp_path / "submission"
    replay = tmp_path / "replay"
    manifest = json.loads(run_forecast(daily, freeze, holdout, sample, output, **expected).read_text(encoding="utf-8"))
    assert manifest["training_input"]["scope_end"] == "2014-08-31"
    assert manifest["models"] == {"primary": "blend_equal", "backup": "cal_workday"}
    daily.write_text(daily.read_text(encoding="utf-8").replace("2014-09-01,100,200", "2014-09-01,unavailable,unavailable"), encoding="utf-8")
    run_forecast(daily, freeze, holdout, sample, replay, **expected)

    for model in ("primary", "backup"):
        name = f"stage2_{model}.csv"
        assert (output / name).read_bytes() == (replay / name).read_bytes()
        with (output / name).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 30
        assert rows[0] == ["20140901", "100", "200"]
        assert rows[-1] == ["20140930", "101", "201"]
    compare = pd.read_csv(output / "calendar_comparison.csv").set_index("report_date")
    assert compare.loc["2014-09-08", "is_holiday"]
    assert compare.loc["2014-09-28", "is_makeup_workday"]
    assert compare.loc["2014-09-28", "is_pre_holiday"]
    assert "2014-09-02" not in compare.index


@pytest.mark.parametrize(("tamper", "error"), [
    ("august_input", "August training inputs"),
    ("holdout_artifact", "fold_metrics.csv"),
    ("artifact_and_manifest", "trusted August manifest"),
    ("august_input_and_manifest", "trusted August manifest"),
    ("freeze_and_manifest", "trusted frozen plan"),
])
def test_forecast_rejects_changes_against_independently_fixed_hashes(tmp_path, frozen_run, tamper, error):
    daily, freeze, holdout, sample = frozen_run
    expected = _expected_hashes(freeze, holdout)
    if tamper in {"august_input", "august_input_and_manifest"}:
        frame = pd.read_csv(daily)
        frame.loc[frame["report_date"].eq("2014-08-31"), "purchase"] += 1
        frame.to_csv(daily, index=False, lineterminator="\n")
        if tamper == "august_input_and_manifest":
            august = frame.loc[frame["report_date"].between("2014-08-01", "2014-08-31")]
            digest = sha256(august.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
            manifest = json.loads(holdout.read_text(encoding="utf-8"))
            assert manifest["holdout_input"]["sha256"] != digest
            manifest["holdout_input"]["sha256"] = digest
            _write_manifest(holdout, manifest)
    elif tamper == "holdout_artifact":
        (holdout.parent / "fold_metrics.csv").write_text("changed\n", encoding="utf-8")
    elif tamper == "artifact_and_manifest":
        metrics = pd.read_csv(holdout.parent / "fold_metrics.csv")
        metrics.loc[0, "weighted_mape"] += 1
        _rewrite_artifact(holdout, "fold_metrics.csv", metrics)
    elif tamper == "freeze_and_manifest":
        freeze.write_bytes(freeze.read_bytes() + b"\n")
        manifest = json.loads(holdout.read_text(encoding="utf-8"))
        manifest["freeze_file_sha256"] = sha256(freeze.read_bytes()).hexdigest()
        _write_manifest(holdout, manifest)
    output = tmp_path / "exports" / "submission"
    with pytest.raises(ValueError, match=error):
        run_forecast(daily, freeze, holdout, sample, output, **expected)
    assert not output.parent.exists()


def test_forecast_accepts_git_newline_conversion_of_frozen_plan_and_holdout(tmp_path, frozen_run):
    daily, freeze, holdout, sample = frozen_run
    expected = _expected_hashes(freeze, holdout)
    for path in (freeze, holdout, *holdout.parent.glob("*.csv")):
        original = path.read_bytes()
        clean = original.replace(b"\r\n", b"\n")
        path.write_bytes(clean if original != clean else clean.replace(b"\n", b"\r\n"))
        assert path.read_bytes() != original
    manifest = run_forecast(daily, freeze, holdout, sample, tmp_path / "submission", **expected)
    assert json.loads(manifest.read_text(encoding="utf-8"))["holdout_manifest_lf_sha256"] == expected["expected_holdout_sha256"]


def test_forecast_defaults_to_the_versioned_release_record(tmp_path, frozen_run, monkeypatch):
    daily, freeze, holdout, sample = frozen_run
    release = tmp_path / "stage2_release.json"
    expected = _expected_hashes(freeze, holdout)
    _write_manifest(release, {
        "schema_version": 1,
        "source_git_commit": "1" * 40,
        "freeze_lf_sha256": expected["expected_freeze_sha256"],
        "holdout_manifest_lf_sha256": expected["expected_holdout_sha256"],
    })
    monkeypatch.setattr(stage2_forecast, "RELEASE_RECORD", release, raising=False)
    assert run_forecast(daily, freeze, holdout, sample, tmp_path / "submission").exists()
    holdout.write_bytes(holdout.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="trusted August manifest"):
        run_forecast(daily, freeze, holdout, sample, tmp_path / "other")
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("provided", ["expected_freeze_sha256", "expected_holdout_sha256"])
def test_forecast_requires_both_explicit_hashes(tmp_path, frozen_run, provided):
    daily, freeze, holdout, sample = frozen_run
    expected = _expected_hashes(freeze, holdout)
    with pytest.raises(ValueError, match="together"):
        run_forecast(daily, freeze, holdout, sample, tmp_path / "submission", **{provided: expected[provided]})


@pytest.mark.parametrize(("field", "replacement"), [
    ("candidates", []),
    ("selected_model", "other"),
    ("config_sha256", "0" * 64),
    ("source_sha256", {}),
    ("calendar_sha256", "0" * 64),
    ("environment", {}),
    ("development_input", {"scope_end": "2014-07-30"}),
    ("development_cutoffs", ["2014-04-01"]),
    ("holdout_dates", ["2014-08-02", "2014-08-31"]),
    ("plan_sha256", "0" * 64),
    ("freeze_file_sha256", "0" * 64),
])
def test_forecast_cross_checks_anchored_holdout_provenance(tmp_path, frozen_run, field, replacement):
    daily, freeze, holdout, sample = frozen_run
    manifest = json.loads(holdout.read_text(encoding="utf-8"))
    manifest[field] = replacement
    _write_manifest(holdout, manifest)
    output = tmp_path / "submission"
    # The fixture pins the changed manifest to exercise validation past the digest layer.
    with pytest.raises(ValueError, match=field):
        run_forecast(daily, freeze, holdout, sample, output, **_expected_hashes(freeze, holdout))
    assert not output.exists()


@pytest.mark.parametrize(("field", "replacement"), [
    ("sha256", "invalid"), ("scope_start", "2014-08-02"), ("scope_end", "2014-09-01"),
    ("rows", 30), ("rows", 31.0), ("serialization", "unrecognized"), ("unexpected", "value"),
])
def test_forecast_checks_august_fingerprint_schema(tmp_path, frozen_run, field, replacement):
    daily, freeze, holdout, sample = frozen_run
    manifest = json.loads(holdout.read_text(encoding="utf-8"))
    manifest["holdout_input"][field] = replacement
    _write_manifest(holdout, manifest)
    output = tmp_path / "submission"
    with pytest.raises(ValueError, match="holdout_input"):
        run_forecast(daily, freeze, holdout, sample, output, **_expected_hashes(freeze, holdout))
    assert not output.exists()


@pytest.mark.parametrize(("filename", "tamper"), [
    ("fold_metrics.csv", "missing"), ("fold_metrics.csv", "duplicate"),
    ("fold_metrics.csv", "wrong_model"), ("fold_metrics.csv", "wrong_role"),
    ("fold_metrics.csv", "wrong_cutoff"), ("fold_metrics.csv", "wrong_forecast_start"),
    ("fold_metrics.csv", "wrong_forecast_end"), ("fold_metrics.csv", "wrong_forecast_days"),
    ("fold_metrics.csv", "wrong_train_start"), ("fold_metrics.csv", "wrong_train_end"),
    ("fold_metrics.csv", "wrong_train_days"),
    ("daily_predictions.csv", "missing"), ("daily_predictions.csv", "duplicate"),
    ("daily_predictions.csv", "wrong_model"), ("daily_predictions.csv", "wrong_role"),
    ("daily_predictions.csv", "wrong_cutoff"), ("daily_predictions.csv", "wrong_report_date"),
    ("daily_predictions.csv", "wrong_horizon_day"),
    ("day_type_metrics.csv", "missing"), ("day_type_metrics.csv", "duplicate"),
    ("horizon_metrics.csv", "missing"), ("horizon_metrics.csv", "duplicate"),
])
def test_forecast_rejects_invalid_anchored_experiment_records(tmp_path, frozen_run, filename, tamper):
    daily, freeze, holdout, sample = frozen_run
    records = pd.read_csv(holdout.parent / filename, keep_default_na=False)
    if tamper == "missing":
        records = records.drop(records.index[-1])
    elif tamper == "duplicate":
        records = pd.concat([records, records.iloc[[-1]]], ignore_index=True)
    else:
        column = tamper.removeprefix("wrong_")
        records.loc[records.index[-1], column] = {
            "model": "unknown", "role": "selection", "cutoff": "2014-08-01", "report_date": "2014-09-01",
            "forecast_start": "2014-08-02", "forecast_end": "2014-08-30", "forecast_days": 30,
            "train_start": "2013-07-02", "train_end": "2014-07-30", "train_days": 395, "horizon_day": 0,
        }[column]
    _rewrite_artifact(holdout, filename, records)
    output = tmp_path / "submission"
    with pytest.raises(ValueError, match=filename):
        run_forecast(daily, freeze, holdout, sample, output, **_expected_hashes(freeze, holdout))
    assert not output.exists()


@pytest.mark.parametrize("error_type", ["RuntimeError", "UndefinedMetric"])
def test_forecast_accepts_explicit_failures_for_non_submission_candidates(tmp_path, frozen_run, error_type):
    daily, freeze, holdout, sample = frozen_run
    metrics = pd.read_csv(holdout.parent / "fold_metrics.csv")
    _rewrite_artifact(holdout, "fold_metrics.csv", metrics.loc[metrics["model"].ne("other")])
    if error_type == "RuntimeError":
        for filename in ("daily_predictions.csv", "day_type_metrics.csv", "horizon_metrics.csv"):
            frame = pd.read_csv(holdout.parent / filename)
            _rewrite_artifact(holdout, filename, frame.loc[frame["model"].ne("other")])
    _rewrite_artifact(holdout, "failures.csv", pd.DataFrame([{
        "model": "other", "role": "holdout", "cutoff": "2014-07-31", "error_type": error_type,
        "message": "weighted_mape" if error_type == "UndefinedMetric" else "fit failed",
    }]))
    output = tmp_path / "submission"
    assert run_forecast(daily, freeze, holdout, sample, output, **_expected_hashes(freeze, holdout)).exists()


@pytest.mark.parametrize("tamper", ["unknown", "duplicate", "overlap", "wrong_role", "wrong_cutoff", "blank_error",
                                    "blank_message", "missing_prediction", "unexpected_prediction"])
def test_forecast_rejects_invalid_anchored_failures(tmp_path, frozen_run, tamper):
    daily, freeze, holdout, sample = frozen_run
    if tamper != "overlap":
        metrics = pd.read_csv(holdout.parent / "fold_metrics.csv")
        _rewrite_artifact(holdout, "fold_metrics.csv", metrics.loc[metrics["model"].ne("other")])
    failure = {
        "model": "unknown" if tamper == "unknown" else "other",
        "role": "selection" if tamper == "wrong_role" else "holdout",
        "cutoff": "2014-08-01" if tamper == "wrong_cutoff" else "2014-07-31",
        "error_type": "" if tamper == "blank_error" else (
            "RuntimeError" if tamper == "unexpected_prediction" else "UndefinedMetric"
        ),
        "message": "" if tamper == "blank_message" else "weighted_mape",
    }
    _rewrite_artifact(holdout, "failures.csv", pd.DataFrame([failure] * (2 if tamper == "duplicate" else 1)))
    if tamper == "missing_prediction":
        daily_predictions = pd.read_csv(holdout.parent / "daily_predictions.csv")
        _rewrite_artifact(holdout, "daily_predictions.csv", daily_predictions.loc[daily_predictions["model"].ne("other")])
    output = tmp_path / "submission"
    expected_error = "daily_predictions.csv" if tamper.endswith("prediction") else "failures.csv"
    with pytest.raises(ValueError, match=expected_error):
        run_forecast(daily, freeze, holdout, sample, output, **_expected_hashes(freeze, holdout))
    assert not output.exists()
