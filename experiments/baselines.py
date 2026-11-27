"""Run the phase-one baselines and render the selected September forecast."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from src.evaluation import (
    BAD_DAY_THRESHOLD,
    DAILY_COLUMNS,
    TARGET_WEIGHTS,
    Fold,
    build_folds,
    predict_targets,
    score_forecast,
    split_fold,
    summarize_selection,
    training_history,
)
from src.models.baselines import BASELINES, DATE_COLUMN, TARGET_COLUMNS, SeasonalMeanBaseline
from src.submission import SUBMISSION_DATES, read_sample, write_submission

DEFAULT_DAILY = Path("data/processed/daily.parquet")
DEFAULT_SAMPLE = Path("Purchase Redemption Data/comp_predict_table.csv")
DEFAULT_RESULTS = Path("results")


@dataclass(frozen=True)
class ExperimentResult:
    selected_model: str
    fold_metrics: pd.DataFrame
    daily_predictions: pd.DataFrame
    selection_summary: pd.DataFrame
    metrics_dir: Path
    submission_path: Path


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _code_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = ("experiments/baselines.py", "src/evaluation.py", "src/models/baselines.py", "src/submission.py")
    head = None
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *paths], cwd=root, stderr=subprocess.DEVNULL, text=True
        )
        dirty = bool(status.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        dirty = None
    return {
        "git_commit": head,
        "experiment_sources_dirty": dirty,
        "experiment_source_sha256": {path: _sha256(root / path) for path in paths},
    }


def _read_daily(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        daily = pd.read_parquet(path, columns=list(DAILY_COLUMNS))
        dates = pd.to_datetime(daily[DATE_COLUMN], errors="raise")
    elif path.suffix.lower() == ".csv":
        daily = pd.read_csv(path, usecols=list(DAILY_COLUMNS), dtype={DATE_COLUMN: "string"})
        date_text = daily[DATE_COLUMN]
        date_format = "%Y%m%d" if date_text.str.fullmatch(r"[0-9]{8}").all() else "%Y-%m-%d"
        dates = pd.to_datetime(date_text, format=date_format, errors="raise")
    else:
        raise ValueError("Daily input must be a .parquet or .csv file")

    if dates.isna().any() or (dates != dates.dt.normalize()).any():
        raise ValueError("Daily data must have valid dates at midnight")
    if dates.dt.tz is not None:
        raise ValueError("Daily data dates must not use a time zone")
    daily[DATE_COLUMN] = dates
    for target in TARGET_COLUMNS:
        values = daily[target]
        if not pd.api.types.is_integer_dtype(values) or values.isna().any() or (values < 0).any():
            raise ValueError(f"Daily {target} must be nonnegative integer cents")
    return daily


def _evaluate_fold(
    daily: pd.DataFrame, fold: Fold, rolling_days: int | None, run_id: str
) -> tuple[list[dict], list[pd.DataFrame]]:
    history, actual = split_fold(daily, fold, rolling_days)
    fold_metrics = []
    predictions_by_model = []
    for code, spec in BASELINES.items():
        start = perf_counter()
        predicted = predict_targets(history, fold.future_dates, partial(SeasonalMeanBaseline, spec))
        fit_predict_seconds = perf_counter() - start
        scores, relative = score_forecast(actual, predicted)
        identity = {"run_id": run_id, "model": code, "role": fold.role, "cutoff": fold.cutoff.date().isoformat()}
        effective_train_start = history[DATE_COLUMN].iloc[0]
        if spec.history_days is not None:
            effective_train_start = max(effective_train_start, fold.cutoff - pd.Timedelta(days=spec.history_days - 1))
        fold_metrics.append(
            {
                **identity,
                "train_start": history[DATE_COLUMN].iloc[0].date().isoformat(),
                "train_end": history[DATE_COLUMN].iloc[-1].date().isoformat(),
                "train_days": len(history),
                "effective_train_start": effective_train_start.date().isoformat(),
                "effective_train_days": (fold.cutoff - effective_train_start).days + 1,
                "forecast_start": fold.future_dates[0].date().isoformat(),
                "forecast_end": fold.future_dates[-1].date().isoformat(),
                "forecast_days": len(actual),
                "fit_predict_seconds": fit_predict_seconds,
                **scores,
            }
        )
        rows = actual.rename(columns={target: f"{target}_actual" for target in TARGET_COLUMNS})
        rows = rows.assign(
            **identity,
            horizon_day=np.arange(1, len(rows) + 1),
            **{f"{target}_predicted": predicted[target] for target in TARGET_COLUMNS},
        ).join(relative)
        predictions_by_model.append(rows)
    return fold_metrics, predictions_by_model


def run_experiment(
    daily_path: Path = DEFAULT_DAILY,
    sample_path: Path = DEFAULT_SAMPLE,
    results_dir: Path = DEFAULT_RESULTS,
    *,
    horizon: str = "calendar-month",
    rolling_days: int | None = None,
    seed: int = 42,
    run_id: str | None = None,
) -> ExperimentResult:
    start = perf_counter()
    started = datetime.now(UTC)
    if run_id is None:
        run_id = f"baselines-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id) is None:
        raise ValueError("Run ID must use letters, digits, hyphens or underscores")
    metrics_dir = results_dir / "metrics" / run_id
    submission_path = results_dir / "submissions" / f"{run_id}.csv"
    if metrics_dir.exists() or submission_path.exists():
        raise FileExistsError(f"Run ID already exists: {run_id}")

    sample_rows = read_sample(sample_path)
    data_hash = _sha256(daily_path)
    daily = _read_daily(daily_path)
    folds = build_folds(horizon)
    fold_rows = []
    daily_rows = []
    for fold in folds[:-1]:
        metrics, predictions = _evaluate_fold(daily, fold, rolling_days, run_id)
        fold_rows.extend(metrics)
        daily_rows.extend(predictions)

    summary = summarize_selection(pd.DataFrame(fold_rows))
    selected_model = str(summary.iloc[0]["model"])
    summary.insert(0, "run_id", run_id)
    holdout_metrics, holdout_predictions = _evaluate_fold(daily, folds[-1], rolling_days, run_id)
    fold_rows.extend(holdout_metrics)
    daily_rows.extend(holdout_predictions)

    submission_cutoff = SUBMISSION_DATES[0] - pd.Timedelta(days=1)
    submission_history = training_history(daily, submission_cutoff, rolling_days)
    submission_start = perf_counter()
    submission_factory = partial(SeasonalMeanBaseline, BASELINES[selected_model])
    submission = predict_targets(submission_history, SUBMISSION_DATES, submission_factory)
    submission_fit_predict_seconds = perf_counter() - submission_start

    fold_metrics = pd.DataFrame(fold_rows)
    daily_predictions = pd.concat(daily_rows, ignore_index=True)
    manifest = {
        "run_id": run_id,
        "started_at_utc": started.isoformat(),
        "seed": seed,
        "deterministic_models": True,
        "daily_input": str(daily_path),
        "daily_sha256": data_hash,
        "daily_rows": len(daily),
        "daily_date_start": daily[DATE_COLUMN].min().date().isoformat(),
        "daily_date_end": daily[DATE_COLUMN].max().date().isoformat(),
        "submission_sample": str(sample_path),
        "submission_sample_sha256": _sha256(sample_path),
        "submission_sample_rows": sample_rows,
        **_code_provenance(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "selection_horizon": horizon,
        "holdout_horizon": horizon,
        "training_window": "expanding" if rolling_days is None else "rolling",
        "rolling_days": rolling_days,
        "target_columns": list(TARGET_COLUMNS),
        "training_columns_per_target": {target: [DATE_COLUMN, target] for target in TARGET_COLUMNS},
        "future_calendar_columns": [DATE_COLUMN],
        "model_parameters": {
            code: {**asdict(spec), "calendar_features": spec.calendar_features}
            for code, spec in BASELINES.items()
        },
        "weighted_mape_weights": TARGET_WEIGHTS,
        "bad_day_strict_threshold": BAD_DAY_THRESHOLD,
        "selection_criterion": "lowest mean weighted MAPE over four selection folds, ties by model code",
        "selected_model": selected_model,
        "submission_cutoff": submission_cutoff.date().isoformat(),
        "submission_training_start": submission_history[DATE_COLUMN].iloc[0].date().isoformat(),
        "submission_forecast_start": SUBMISSION_DATES[0].date().isoformat(),
        "submission_forecast_end": SUBMISSION_DATES[-1].date().isoformat(),
        "submission_fit_predict_seconds": submission_fit_predict_seconds,
        "submission_file": str(submission_path.relative_to(results_dir).as_posix()),
    }

    metrics_dir.mkdir(parents=True, exist_ok=False)
    fold_metrics.to_csv(metrics_dir / "fold_metrics.csv", index=False)
    daily_predictions.to_csv(metrics_dir / "daily_predictions.csv", index=False, date_format="%Y-%m-%d")
    summary.to_csv(metrics_dir / "selection_summary.csv", index=False)
    write_submission(submission, submission_path, sample_path)
    manifest["elapsed_seconds"] = perf_counter() - start
    (metrics_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ExperimentResult(selected_model, fold_metrics, daily_predictions, summary, metrics_dir, submission_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行阶段一月度基线评估并生成 2014 年 9 月提交文件。")
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--horizon", choices=("calendar-month", "fixed30"), default="calendar-month")
    parser.add_argument("--training-window", choices=("expanding", "rolling"), default="expanding")
    parser.add_argument("--rolling-days", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.training_window == "rolling" and (args.rolling_days is None or args.rolling_days < 1):
        parser.error("--training-window rolling 需要提供正整数 --rolling-days")
    if args.training_window == "expanding" and args.rolling_days is not None:
        parser.error("--rolling-days 需要和 --training-window rolling 一同使用")

    result = run_experiment(
        args.daily, args.sample, args.results_dir,
        horizon=args.horizon, rolling_days=args.rolling_days, seed=args.seed, run_id=args.run_id,
    )
    print(f"入选基线 {result.selected_model}")
    print(f"评估产物 {result.metrics_dir}")
    print(f"提交文件 {result.submission_path}")


if __name__ == "__main__":
    main()
