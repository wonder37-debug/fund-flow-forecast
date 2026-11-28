"""Freeze Stage 3 selection, evaluate August once, and export September forecasts."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.stage2 import _git_head, _persist_tables
from experiments.stage2_forecast import DEFAULT_SAMPLE, _calendar_comparison
from experiments.stage3 import (
    DEFAULT_CACHE, DEFAULT_DAILY, DEVELOPMENT_END, HOLDOUT_END, ROOT, _feature_provenance, _fingerprint,
    _fold_samples, evaluate, feature_table, forecast_candidate, read_daily,
)
from src.evaluation import Fold, build_folds
from src.features.multistep import build_examples
from src.models.baselines import DATE_COLUMN, TARGET_COLUMNS
from src.submission import SUBMISSION_DATES, write_submission

DEFAULT_DEV = Path("results/metrics/stage3-development/manifest.json")
DEFAULT_FREEZE = Path("results/metrics/stage3-freeze/freeze.json")
DEFAULT_HOLDOUT = Path("results/metrics/stage3-holdout/manifest.json")
RELEASE = ROOT / "configs/stage3_release.json"
BASE = "route_special"
BACKUP = "lightgbm_purchase_route_redeem"
CONTROLS = {"fold_metrics.csv", "daily_predictions.csv", "day_type_metrics.csv", "horizon_metrics.csv",
            "feature_importance.csv", "failures.csv"}
AUGUST = build_folds()[-1]
RULE = "Use calendar routing for primary to prioritize cross-window stability; use LightGBM residual for backup purchase only."


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_pinned(path: Path, expected: str) -> dict:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("An independently recorded LF SHA256 digest is required")
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Trusted digest does not match {path.name}")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return result


def _write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _check_artifacts(manifest: dict, parent: Path, required: set[str]) -> None:
    hashes = manifest.get("artifacts_sha256", {})
    if set(hashes) != required:
        raise ValueError("Evidence file list is incomplete or unexpected")
    for name, expected in hashes.items():
        if _digest(parent / name) != expected:
            raise ValueError(f"Evidence digest does not match {name}")


def _approved_candidates(candidates: list[dict]) -> tuple[list[dict], dict[str, str]]:
    by_id = {candidate["id"]: candidate for candidate in candidates}
    if len(by_id) != len(candidates) or not {BASE, "lightgbm_ratio"}.issubset(by_id) or BACKUP in by_id:
        raise ValueError("Frozen candidates require unique IDs and both approved base candidates")
    mix = by_id["lightgbm_ratio"] | {"id": BACKUP, "targets": {"redeem": {"stat": BASE}}}
    return [*candidates, mix], {"primary": BASE, "backup": BACKUP}


def _release_sources() -> dict:
    return {name: _digest(ROOT / name) for name in (
        "experiments/stage3_release.py", "experiments/stage2_forecast.py", "src/submission.py",
    )}


def freeze_plan(development: Path, expected_dev: str, output: Path) -> None:
    dev = _read_pinned(development, expected_dev)
    _check_artifacts(dev, development.parent, CONTROLS | {"selection_summary.csv"})
    provenance = _feature_provenance()
    if (
        dev.get("stage") != 3 or dev.get("role") != "development"
        or dev.get("provisional_candidate") != BASE or dev.get("failure_count") != 0
        or dev["data"]["end"] != str(DEVELOPMENT_END.date())
        or any(dev.get(key) != value for key, value in provenance.items())
    ):
        raise ValueError("Development evidence differs from the reviewed Stage 3 inputs or selection")
    candidates, roles = _approved_candidates(dev["config"]["candidates"])
    if dev["success_count"] != len(candidates[:-1]) * len(dev["folds"]):
        raise ValueError("Development evidence has incomplete candidate results")
    plan = {
        "stage": 3, "role": "freeze", "selection_rule": RULE, "models": roles, "candidates": candidates,
        "development_manifest_lf_sha256": expected_dev, "development_input": dev["data"],
        "development_git_commit": dev["git_commit"], "development_config_sha256": dev["config_sha256"],
        "holdout_cutoff": str(AUGUST.cutoff.date()),
        "holdout_dates": [str(AUGUST.future_dates[0].date()), str(AUGUST.future_dates[-1].date())],
        "release_source_sha256": _release_sources(), **provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, plan)


def _read_plan(path: Path, expected: str) -> dict:
    plan = _read_pinned(path, expected)
    provenance = _feature_provenance()
    candidates, roles = _approved_candidates(plan["candidates"][:-1])
    required = {
        "role": "freeze", "stage": 3, "selection_rule": RULE, "models": roles, "candidates": candidates,
        "holdout_cutoff": str(AUGUST.cutoff.date()),
        "holdout_dates": [str(AUGUST.future_dates[0].date()), str(AUGUST.future_dates[-1].date())],
        "release_source_sha256": _release_sources(), **provenance,
    }
    for key, value in required.items():
        if plan.get(key) != value:
            raise ValueError(f"Frozen plan is inconsistent with current {key}")
    return plan


def _validate_august_records(
    metrics: pd.DataFrame, daily: pd.DataFrame, failures: pd.DataFrame, candidates: set[str], required: set[str],
) -> None:
    scored, failed = set(metrics["model"]), set(failures["model"])
    if (
        scored & failed or scored | failed != candidates or not required.issubset(scored)
        or len(metrics) != len(scored) or len(failures) != len(failed) or set(daily["model"]) != scored
    ):
        raise ValueError("August candidate results are incomplete, duplicated or missing a released candidate")
    for table in (metrics, daily, failures):
        if not table["role"].eq("holdout").all() or not table["cutoff"].eq(str(AUGUST.cutoff.date())).all():
            raise ValueError("August records have an incorrect role or cutoff")
    if not (
        metrics["forecast_start"].eq("2014-08-01").all() and metrics["forecast_end"].eq("2014-08-31").all()
        and metrics["forecast_days"].eq(31).all()
    ):
        raise ValueError("August dates in fold metrics do not match the frozen horizon")
    for _, rows in daily.groupby("model", sort=False):
        dates = pd.DatetimeIndex(pd.to_datetime(rows[DATE_COLUMN])).as_unit("ns")
        if not dates.equals(AUGUST.future_dates):
            raise ValueError("August dates do not cover the frozen horizon exactly once")
    amounts = daily[[f"{target}_predicted" for target in TARGET_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("August forecast amounts must be nonnegative and finite")


def _check_history(daily: pd.DataFrame, plan: dict) -> dict:
    earlier = daily[DATE_COLUMN].le(DEVELOPMENT_END)
    if _fingerprint(daily.loc[earlier]) != plan["development_input"]:
        raise ValueError("The observed history through July differs from the approved development data")
    return _fingerprint(daily.loc[~earlier])


def run_holdout(daily_path: Path, freeze: Path, expected_freeze: str, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    start = perf_counter()
    plan = _read_plan(freeze, expected_freeze)
    daily = read_daily(daily_path, HOLDOUT_END)
    august_input = _check_history(daily, plan)
    examples = feature_table(daily, AUGUST.cutoff, DEFAULT_CACHE)
    tables = evaluate(daily, examples, (AUGUST,), plan["candidates"])
    _validate_august_records(tables["fold_metrics.csv"], tables["daily_predictions.csv"], tables["failures.csv"],
                             {candidate["id"] for candidate in plan["candidates"]}, set(plan["models"].values()))
    hashes = _persist_tables(output, tables)
    manifest = {
        "stage": 3, "role": "holdout", "freeze_lf_sha256": expected_freeze, "models": plan["models"],
        "cutoff": plan["holdout_cutoff"], "dates": plan["holdout_dates"], "august_input": august_input,
        "success_count": len(tables["fold_metrics.csv"]), "failure_count": len(tables["failures.csv"]),
        "git_commit": _git_head(), "elapsed_seconds": perf_counter() - start, "artifacts_sha256": hashes,
    }
    _write_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def run_forecast(
    daily_path: Path, freeze: Path, holdout: Path, expected_freeze: str, expected_holdout: str,
    sample: Path, output: Path,
) -> Path:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    plan = _read_plan(freeze, expected_freeze)
    manifest = _read_pinned(holdout, expected_holdout)
    expected = {"stage": 3, "role": "holdout", "freeze_lf_sha256": expected_freeze, "models": plan["models"],
                "cutoff": plan["holdout_cutoff"], "dates": plan["holdout_dates"]}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("August evidence differs from the frozen plan")
    _check_artifacts(manifest, holdout.parent, CONTROLS)
    tables = [pd.read_csv(holdout.parent / name) for name in ("fold_metrics.csv", "daily_predictions.csv", "failures.csv")]
    candidates = {candidate["id"]: candidate for candidate in plan["candidates"]}
    _validate_august_records(*tables, set(candidates), set(plan["models"].values()))
    actual = read_daily(daily_path, HOLDOUT_END)
    if _check_history(actual, plan) != manifest["august_input"]:
        raise ValueError("Observed August history differs from the approved August evidence")

    past_examples = feature_table(actual, HOLDOUT_END - pd.Timedelta(days=1), DEFAULT_CACHE)
    september = build_examples(actual, [HOLDOUT_END], max_horizon=len(SUBMISSION_DATES))
    deploy = Fold(HOLDOUT_END, SUBMISSION_DATES, "forecast")
    examples = pd.concat([past_examples, september], ignore_index=True)
    train, future = _fold_samples(examples, actual, deploy)
    forecasts = {role: forecast_candidate(train, future, candidates[model])[0] for role, model in plan["models"].items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        prepared = Path(directory) / "artifacts"
        prepared.mkdir()
        files = {role: write_submission(prediction, prepared / f"stage3_{role}.csv", sample)
                 for role, prediction in forecasts.items()}
        _calendar_comparison(prepared / "calendar_comparison.csv", files)
        _write_json(prepared / "manifest.json", {
            "stage": 3, "models": plan["models"], "git_commit": _git_head(),
            "freeze_lf_sha256": expected_freeze, "holdout_manifest_lf_sha256": expected_holdout,
            "training_input": _fingerprint(actual), "forecast_start": "2014-09-01", "forecast_end": "2014-09-30",
            "forecast_days": len(SUBMISSION_DATES), "format": "headerless YYYYMMDD,purchase_cents,redeem_cents, half-up",
            "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
            "artifacts_sha256": {path.name: _digest(path) for path in sorted(prepared.iterdir())},
        })
        prepared.rename(output)
    return output / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "holdout", "forecast"))
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-development-sha256")
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--expected-holdout-sha256")
    args = parser.parse_args()
    if args.command == "freeze":
        if not args.expected_development_sha256:
            parser.error("freeze requires --expected-development-sha256")
        freeze_plan(args.development, args.expected_development_sha256, args.output)
        print(args.output)
        return

    release = json.loads(RELEASE.read_bytes()) if RELEASE.is_file() else {}
    if args.command == "forecast" and bool(args.expected_freeze_sha256) != bool(args.expected_holdout_sha256):
        parser.error("forecast expects both explicit digests for an independent rerun")
    expected_freeze = args.expected_freeze_sha256 or release.get("freeze_lf_sha256")
    if args.command == "holdout":
        print(run_holdout(args.daily, args.freeze, expected_freeze, args.output))
        return
    expected_holdout = args.expected_holdout_sha256 or release.get("holdout_manifest_lf_sha256")
    print(run_forecast(args.daily, args.freeze, args.holdout, expected_freeze, expected_holdout, args.sample, args.output))


if __name__ == "__main__":
    main()
