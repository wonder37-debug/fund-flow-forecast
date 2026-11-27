"""Refit the frozen stage-two primary and backup through August and export September."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from experiments.baselines import DEFAULT_DAILY, DEFAULT_SAMPLE, _sha256
from experiments.stage2 import (
    DEVELOPMENT_ARTIFACTS,
    DEVELOPMENT_END,
    HOLDOUT_END,
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


DEFAULT_FREEZE = Path("results/metrics/stage2-development/freeze.json")
DEFAULT_HOLDOUT = Path("results/metrics/stage2-holdout/manifest.json")
DEFAULT_OUTPUT = Path("results/submissions/stage2")
BACKUP_MODEL = "cal_workday"
CALENDAR_FLAGS = ("is_holiday", "is_makeup_workday", "is_pre_holiday", "is_post_holiday")


def _validated_holdout(path: Path, freeze: Path, plan: dict, models: dict[str, str]) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    frozen_lf = freeze.read_bytes().replace(b"\r\n", b"\n")
    # Earlier runs recorded the hash before Git normalized the JSON line endings.
    freeze_hashes = {hashlib.sha256(data).hexdigest() for data in (frozen_lf, frozen_lf.replace(b"\n", b"\r\n"))}
    if (
        manifest.get("role") != "holdout"
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or manifest.get("freeze_file_sha256") not in freeze_hashes
        or manifest.get("selected_model") != models["primary"]
    ):
        raise ValueError("August evidence does not match the selected frozen plan")
    hashes = manifest.get("holdout_artifact_sha256", {})
    if set(hashes) != DEVELOPMENT_ARTIFACTS - {"selection_summary.csv"}:
        raise ValueError("August evidence is incomplete")
    for filename, expected in hashes.items():
        artifact = path.parent / filename
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"August evidence does not match: {filename}")
    metrics = pd.read_csv(path.parent / "fold_metrics.csv", usecols=["model", "role", "forecast_days"])
    complete = metrics.loc[metrics["role"].eq("holdout") & metrics["forecast_days"].eq(31), "model"]
    if not set(models.values()).issubset(complete):
        raise ValueError("Both submission models must have completed the August evaluation")
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


def run_forecast(daily_path: Path, freeze: Path, holdout: Path, sample: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    plan = read_freeze(freeze)
    models = {"primary": plan["selected_model"], "backup": BACKUP_MODEL}
    candidates = {item["id"]: item for item in plan["candidates"]}
    if len(set(models.values())) != 2 or not set(models.values()).issubset(candidates):
        raise ValueError("The frozen plan must contain distinct primary and backup candidates")
    evidence = _validated_holdout(holdout, freeze, plan, models)

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
            "holdout_manifest_lf_sha256": hashlib.sha256(holdout.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
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
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run_forecast(args.daily, args.freeze, args.holdout, args.sample, args.output_dir))


if __name__ == "__main__":
    main()
