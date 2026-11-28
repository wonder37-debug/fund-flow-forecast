"""Refit the frozen stage-two primary and backup through August and export September."""

import argparse
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments.baselines import DEFAULT_DAILY, DEFAULT_SAMPLE, _sha256
from experiments.stage2 import (
    DEVELOPMENT_ARTIFACTS,
    DEVELOPMENT_END,
    HOLDOUT_END,
    RELIABILITY_METRICS,
    ROOT,
    _data_fingerprint,
    _git_head,
    _read_scoped_daily,
    build_model_factory,
    read_freeze,
)
from src.evaluation import predict_targets, training_history
from src.features.calendar import build_calendar
from src.models.baselines import DATE_COLUMN, TARGET_COLUMNS
from src.submission import SUBMISSION_DATES, write_submission


DEFAULT_FREEZE = Path("results/metrics/stage2-review-development/freeze.json")
DEFAULT_HOLDOUT = Path("results/metrics/stage2-review-holdout/manifest.json")
DEFAULT_OUTPUT = Path("results/submissions/stage2")
RELEASE_RECORD = ROOT / "configs/stage2_release.json"
BACKUP_MODEL = "cal_workday"
CALENDAR_FLAGS = ("is_holiday", "is_makeup_workday", "is_pre_holiday", "is_post_holiday")


def _lf_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _release_digests(expected_freeze: str | None, expected_holdout: str | None) -> tuple[str, str]:
    if expected_freeze is None and expected_holdout is None:
        record = json.loads(RELEASE_RECORD.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ValueError("Unsupported stage-two release record schema")
        expected_freeze = record.get("freeze_lf_sha256")
        expected_holdout = record.get("holdout_manifest_lf_sha256")
    elif expected_freeze is None or expected_holdout is None:
        raise ValueError("Pass --expected-freeze-sha256 and --expected-holdout-sha256 together")
    if not _is_sha256(expected_freeze) or not _is_sha256(expected_holdout):
        raise ValueError("Expected release fingerprints must be lowercase SHA256 digests")
    return expected_freeze, expected_holdout


def _holdout_records(
    data: bytes, filename: str, allowed_models: set[str], keys: tuple[str, ...], columns: tuple[str, ...],
) -> pd.DataFrame:
    try:
        records = pd.read_csv(
            BytesIO(data), usecols=["model", "role", "cutoff", *keys, *columns],
            dtype={"model": "string"}, keep_default_na=False,
        )
    except ValueError as error:
        raise ValueError(f"Invalid August evidence columns in {filename}") from error
    if (
        not records["model"].isin(allowed_models).all()
        or not records["role"].eq("holdout").all()
        or not records["cutoff"].eq(DEVELOPMENT_END.date().isoformat()).all()
        or records.duplicated(["model", *keys]).any()
    ):
        raise ValueError(f"Invalid August model, role, cutoff or duplicate record in {filename}")
    return records


def _day_type_counts(dates: pd.DatetimeIndex) -> dict[str, int]:
    calendar = build_calendar(dates[0], dates[-1])
    holiday = calendar["is_holiday"]
    makeup = calendar["is_makeup_workday"]
    weekend = calendar["is_regular_weekend"]
    ordinary = (calendar["is_workday"] | weekend) & ~(holiday | makeup)
    pre = calendar["is_pre_holiday"]
    post = calendar["is_post_holiday"]
    days = calendar[DATE_COLUMN].dt
    segments = {
        "ordinary": ordinary,
        "regular_workday": ~(holiday | makeup | weekend),
        "regular_weekend": weekend & ~(holiday | makeup),
        "holiday": holiday,
        "makeup_workday": makeup,
        "pre_holiday": ordinary & pre,
        "post_holiday": ordinary & post & ~pre,
        "month_end": ordinary & ~pre & ~post & days.day.ge(days.days_in_month - 2),
    }
    return {name: int(mask.sum()) for name, mask in segments.items() if mask.any()}


def _validate_holdout_records(artifacts: dict[str, bytes], plan: dict, required_models: set[str]) -> None:
    known_models = {candidate["id"] for candidate in plan["candidates"]}
    dates = pd.date_range(*plan["holdout_dates"])
    development = plan["development_input"]
    fold_values = {
        "train_start": development["scope_start"], "train_end": development["scope_end"],
        "train_days": development["rows"], "forecast_start": plan["holdout_dates"][0],
        "forecast_end": plan["holdout_dates"][-1], "forecast_days": len(dates),
    }
    metrics = _holdout_records(
        artifacts["fold_metrics.csv"], "fold_metrics.csv", known_models, (), (*fold_values, *RELIABILITY_METRICS),
    )
    metric_values = metrics[list(RELIABILITY_METRICS)].apply(pd.to_numeric, errors="coerce")
    if any(not metrics[column].eq(value).all() for column, value in fold_values.items()) or not np.isfinite(
        metric_values.to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid August training, forecast or metric values in fold_metrics.csv")
    completed = set(metrics["model"])
    if not required_models.issubset(completed):
        raise ValueError("Both submission models must have completed August records in fold_metrics.csv")

    failures = _holdout_records(artifacts["failures.csv"], "failures.csv", known_models, (), ("error_type", "message"))
    failed = set(failures["model"])
    if completed & failed or any(not failures[column].astype("string").str.strip().ne("").all()
                                 for column in ("error_type", "message")):
        raise ValueError("Invalid August failure details or overlapping completion in failures.csv")
    if completed | failed != known_models:
        raise ValueError("August fold_metrics.csv and failures.csv must account for every frozen candidate")

    # Undefined scores are recorded separately after the model has produced a complete forecast.
    predicted = completed | set(failures.loc[failures["error_type"].eq("UndefinedMetric"), "model"])
    amount_columns = tuple(f"{target}_{kind}" for target in TARGET_COLUMNS for kind in ("actual", "predicted"))
    daily = _holdout_records(
        artifacts["daily_predictions.csv"], "daily_predictions.csv", predicted, (DATE_COLUMN,),
        ("horizon_day", *amount_columns),
    )
    horizon_days = dict(zip(dates.strftime("%Y-%m-%d"), range(1, len(dates) + 1)))
    amounts = daily[list(amount_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if (
        set(daily["model"]) != predicted
        or not daily.groupby("model").size().eq(len(dates)).all()
        or not daily["horizon_day"].eq(daily[DATE_COLUMN].map(horizon_days)).all()
        or not np.isfinite(amounts).all()
        or (amounts < 0).any()
    ):
        raise ValueError("Invalid or incomplete August predictions in daily_predictions.csv")

    for filename, segment, counts in (
        ("day_type_metrics.csv", "day_type", _day_type_counts(dates)),
        ("horizon_metrics.csv", "horizon_band", {"1-7": 7, "8-14": 7, "15-21": 7, "22-31": len(dates) - 21}),
    ):
        records = _holdout_records(artifacts[filename], filename, predicted, (segment,), ("days",))
        expected = {(model, label) for model in predicted for label in counts}
        if set(zip(records["model"], records[segment])) != expected or not records["days"].eq(records[segment].map(counts)).all():
            raise ValueError(f"Invalid or incomplete August segments in {filename}")


def _validated_holdout(path: Path, manifest_lf: bytes, frozen_lf: bytes, plan: dict, models: dict[str, str]) -> dict:
    manifest = json.loads(manifest_lf)
    if not isinstance(manifest, dict) or manifest.get("role") != "holdout":
        raise ValueError("August manifest role must be holdout")
    fields = ("plan_sha256", "selected_model", "candidates", "config_sha256", "source_sha256", "calendar_sha256",
              "environment", "development_input")
    # Existing manifests express the protocol dates through the input scopes and the fold records.
    fields += tuple(field for field in ("development_cutoffs", "holdout_dates") if field in manifest)
    for field in fields:
        if json.dumps(manifest.get(field), sort_keys=True) != json.dumps(plan[field], sort_keys=True):
            raise ValueError(f"August manifest {field} does not match the frozen plan")

    # Earlier runs recorded the hash before Git normalized the JSON line endings.
    freeze_hashes = {hashlib.sha256(data).hexdigest() for data in (frozen_lf, frozen_lf.replace(b"\n", b"\r\n"))}
    if manifest.get("freeze_file_sha256") not in freeze_hashes:
        raise ValueError("August manifest freeze_file_sha256 does not match the frozen plan")

    fingerprint = manifest.get("holdout_input")
    start, end = plan["holdout_dates"]
    if (
        not isinstance(fingerprint, dict)
        or set(fingerprint) != {"sha256", "scope_start", "scope_end", "rows", "serialization"}
        or not _is_sha256(fingerprint["sha256"])
        or (fingerprint["scope_start"], fingerprint["scope_end"]) != (start, end)
        or type(fingerprint["rows"]) is not int
        or fingerprint["rows"] != len(pd.date_range(start, end))
        or fingerprint["serialization"] != plan["development_input"]["serialization"]
    ):
        raise ValueError("August manifest holdout_input fingerprint is invalid")

    hashes = manifest.get("holdout_artifact_sha256", {})
    if not isinstance(hashes, dict) or set(hashes) != DEVELOPMENT_ARTIFACTS - {"selection_summary.csv"}:
        raise ValueError("August evidence is incomplete")
    artifacts = {}
    for filename, expected in hashes.items():
        artifact = path.parent / filename
        if not artifact.is_file():
            raise ValueError(f"August evidence is missing: {filename}")
        data = _lf_bytes(artifact)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"August evidence does not match: {filename}")
        artifacts[filename] = data
    _validate_holdout_records(artifacts, plan, set(models.values()))
    return manifest


def _calendar_comparison(output: Path, submissions: dict[str, Path]) -> None:
    calendar = build_calendar(SUBMISSION_DATES[0], SUBMISSION_DATES[-1])
    columns = [DATE_COLUMN, "holiday", "previous_holiday", "next_holiday", *CALENDAR_FLAGS]
    comparison = calendar.loc[calendar[list(CALENDAR_FLAGS)].any(axis=1), columns].copy()
    for role, path in submissions.items():
        amounts = pd.read_csv(path, header=None, names=[DATE_COLUMN, *TARGET_COLUMNS], dtype={DATE_COLUMN: "string"})
        amounts[DATE_COLUMN] = pd.to_datetime(amounts[DATE_COLUMN], format="%Y%m%d")
        comparison = comparison.merge(amounts.rename(columns={target: f"{role}_{target}" for target in TARGET_COLUMNS}),
                                      on=DATE_COLUMN, how="left", validate="one_to_one")
    for target in TARGET_COLUMNS:
        comparison[f"primary_minus_backup_{target}"] = comparison[f"primary_{target}"] - comparison[f"backup_{target}"]
    comparison.to_csv(output, index=False, lineterminator="\n", date_format="%Y-%m-%d")


def run_forecast(
    daily_path: Path, freeze: Path, holdout: Path, sample: Path, output: Path, *,
    expected_freeze_sha256: str | None = None, expected_holdout_sha256: str | None = None,
) -> Path:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    expected_freeze, expected_holdout = _release_digests(expected_freeze_sha256, expected_holdout_sha256)
    frozen_lf = _lf_bytes(freeze)
    if hashlib.sha256(frozen_lf).hexdigest() != expected_freeze:
        raise ValueError("The trusted frozen plan SHA256 does not match")
    manifest_lf = _lf_bytes(holdout)
    if hashlib.sha256(manifest_lf).hexdigest() != expected_holdout:
        raise ValueError("The trusted August manifest SHA256 does not match")
    plan = read_freeze(freeze)
    if plan != json.loads(frozen_lf):
        raise ValueError("The trusted frozen plan changed during validation")
    models = {"primary": plan["selected_model"], "backup": BACKUP_MODEL}
    candidates = {item["id"]: item for item in plan["candidates"]}
    if len(set(models.values())) != 2 or not set(models.values()).issubset(candidates):
        raise ValueError("The frozen plan must contain distinct primary and backup candidates")
    evidence = _validated_holdout(holdout, manifest_lf, frozen_lf, plan, models)

    history = training_history(_read_scoped_daily(daily_path, HOLDOUT_END), HOLDOUT_END)
    development_rows = history[DATE_COLUMN].le(DEVELOPMENT_END)
    if _data_fingerprint(history.loc[development_rows]) != plan["development_input"]:
        raise ValueError("Training inputs through July do not match the frozen plan")
    if _data_fingerprint(history.loc[~development_rows]) != evidence["holdout_input"]:
        raise ValueError("August training inputs do not match the validated August evidence")

    forecasts = {}
    for role, model_id in models.items():
        factory, columns = build_model_factory(candidates[model_id], plan["candidates"])
        forecasts[role] = predict_targets(history, SUBMISSION_DATES, factory, calendar_columns=columns)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        prepared = Path(temporary) / "artifacts"
        prepared.mkdir()
        submissions = {
            role: write_submission(forecast, prepared / f"stage2_{role}.csv", sample)
            for role, forecast in forecasts.items()
        }
        _calendar_comparison(prepared / "calendar_comparison.csv", submissions)
        manifest = {
            "models": models,
            "training_input": _data_fingerprint(history),
            "forecast_start": SUBMISSION_DATES[0].date().isoformat(),
            "forecast_end": SUBMISSION_DATES[-1].date().isoformat(),
            "forecast_days": len(SUBMISSION_DATES),
            "format": "headerless YYYYMMDD,purchase_cents,redeem_cents, rounded half up",
            "frozen_plan_sha256": plan["plan_sha256"],
            "calendar_sha256": plan["calendar_sha256"],
            "holdout_manifest_lf_sha256": expected_holdout,
            "sample_sha256": _sha256(sample),
            "git_commit": _git_head(),
            "export_source_sha256": {name: _sha256(ROOT / name) for name in (
                "experiments/stage2_forecast.py", "src/submission.py",
            )},
            "artifacts_sha256": {path.name: _sha256(path) for path in sorted(prepared.iterdir())},
        }
        (prepared / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n",
        )
        prepared.rename(output)
    return output / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--expected-freeze-sha256", help="Independently approved LF SHA256 of a rerun's frozen plan")
    parser.add_argument("--expected-holdout-sha256", help="Independently approved LF SHA256 of a rerun's August manifest")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run_forecast(
        args.daily, args.freeze, args.holdout, args.sample, args.output_dir,
        expected_freeze_sha256=args.expected_freeze_sha256, expected_holdout_sha256=args.expected_holdout_sha256,
    ))


if __name__ == "__main__":
    main()
