"""Forward development experiments for direct and hybrid forecasts."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.stage2 import _git_head, _persist_tables, _provenance, _summarize, _weighted_scores, select_candidate
from src.evaluation import BAD_DAY_THRESHOLD, DAILY_COLUMNS, TARGET_WEIGHTS, Fold, build_folds, fold_actuals
from src.features.calendar import build_calendar
from src.models.baselines import DATE_COLUMN, TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAILY = Path("data/processed/daily.parquet")
DEFAULT_CONFIG = ROOT / "configs/stage3.json"
DEFAULT_CACHE = Path("data/cache/stage3-features")
DEVELOPMENT_END = pd.Timestamp("2014-07-31")
HOLDOUT_END = pd.Timestamp("2014-08-31")
MIN_HISTORY = 84
SOURCE_PATHS = (
    "experiments/stage3.py", "src/features/multistep.py", "src/models/machine_learning.py",
    "requirements-stage3.txt", "configs/stage2.json",
)
DAY_FLAGS = ("is_holiday", "is_makeup_workday", "is_regular_weekend", "is_pre_holiday", "is_post_holiday")
THRESHOLDS = (20, 30, 40)


def build_development_folds() -> tuple[Fold, ...]:
    earlier_starts = pd.to_datetime(["2014-02-01", "2014-03-01"])
    early = tuple(Fold(start - pd.Timedelta(days=1), pd.date_range(start, periods=start.days_in_month), "early")
                  for start in earlier_starts)
    rolling_cutoffs = pd.to_datetime(["2014-03-15", "2014-05-15", "2014-06-15", "2014-07-01"])
    fixed = tuple(Fold(cutoff, pd.date_range(cutoff + pd.Timedelta(days=1), periods=30), "fixed30")
                  for cutoff in rolling_cutoffs)
    return (*early, *build_folds()[:4], *fixed)


def read_daily(path: Path, end: pd.Timestamp) -> pd.DataFrame:
    daily = pd.read_parquet(path, filters=[(DATE_COLUMN, "<=", end)])
    if not set(DAILY_COLUMNS).issubset(daily):
        raise ValueError("Daily input must include report_date, purchase and redeem")
    daily[DATE_COLUMN] = pd.to_datetime(daily[DATE_COLUMN]).astype("datetime64[ns]")
    daily = daily.sort_values(DATE_COLUMN, ignore_index=True)
    if daily.empty or not pd.DatetimeIndex(daily[DATE_COLUMN]).equals(pd.date_range(daily[DATE_COLUMN].iat[0], end)):
        raise ValueError(f"Incomplete daily coverage through {end.date()}")
    targets = daily[list(TARGET_COLUMNS)].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Observed targets must be finite and nonnegative")
    return daily


def _fingerprint(daily: pd.DataFrame) -> dict:
    rows = pd.util.hash_pandas_object(daily, index=False).to_numpy(dtype="<u8")
    schema = json.dumps([(column, str(dtype)) for column, dtype in daily.dtypes.items()]).encode()
    return {
        "sha256": hashlib.sha256(schema + rows.tobytes()).hexdigest(),
        "hash_method": "ordered pandas row hashes plus ordered schema",
        "start": daily[DATE_COLUMN].iat[0].date().isoformat(),
        "end": daily[DATE_COLUMN].iat[-1].date().isoformat(),
        "rows": len(daily),
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_provenance() -> dict:
    stage2 = _provenance()
    return {
        "calendar_sha256": stage2["calendar_sha256"],
        "source_sha256": {**stage2["source_sha256"], **{name: _hash(ROOT / name) for name in SOURCE_PATHS}},
        "environment": {**stage2["environment"], "scikit-learn": version("scikit-learn"),
                        "lightgbm": version("lightgbm"), "platform": platform.system()},
    }


def feature_table(daily: pd.DataFrame, cutoff: pd.Timestamp, cache: Path | None = None) -> pd.DataFrame:
    from src.features.multistep import build_examples

    observed = daily.loc[daily[DATE_COLUMN].le(cutoff)].reset_index(drop=True)
    origins = observed[DATE_COLUMN].iloc[MIN_HISTORY - 1 :]
    if origins.empty:
        raise ValueError(f"At least {MIN_HISTORY} daily observations are required")
    key_data = {"daily": _fingerprint(observed), "max_horizon": 31, **_feature_provenance()}
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    path = cache / f"{key}.parquet" if cache is not None else None
    if path is not None and path.is_file():
        return pd.read_parquet(path)
    examples = build_examples(observed, origins, max_horizon=31)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A failed writer leaves no cache entry. Independent runs may generate the same entry.
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as temp:
            temporary = Path(temp.name)
        try:
            examples.to_parquet(temporary, index=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return examples


def _fold_samples(examples: pd.DataFrame, daily: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(examples).intersection(TARGET_COLUMNS):
        raise ValueError("Feature rows cannot include observed labels")
    future = examples.loc[examples["origin"].eq(fold.cutoff) & examples[DATE_COLUMN].isin(fold.future_dates)]
    future = future.sort_values(DATE_COLUMN, ignore_index=True)
    if not pd.DatetimeIndex(future[DATE_COLUMN]).equals(fold.future_dates):
        raise ValueError(f"Incomplete forecast feature coverage after {fold.cutoff.date()}")
    known_rows = examples.loc[examples[DATE_COLUMN].le(fold.cutoff)]
    if known_rows["origin"].ge(known_rows[DATE_COLUMN]).any() or known_rows.duplicated(["origin", DATE_COLUMN]).any():
        raise ValueError("Training feature rows must have ordered, unique origin/target pairs")
    known_labels = daily.loc[daily[DATE_COLUMN].le(fold.cutoff), list(DAILY_COLUMNS)]
    training = known_rows.merge(known_labels, on=DATE_COLUMN, how="left", validate="many_to_one")
    if training[list(TARGET_COLUMNS)].isna().any().any():
        raise ValueError("Missing observed labels from training range")
    return training, future


def forecast_candidate(training: pd.DataFrame, future: pd.DataFrame, candidate: dict) -> tuple[pd.DataFrame, list[dict]]:
    predicted = future[[DATE_COLUMN]].copy()
    importance = []
    for target in TARGET_COLUMNS:
        options = candidate | candidate.get("targets", {}).get(target, {})
        if "stat" in options:
            predicted[target] = future[f"stat_{options['stat']}_{target}"].to_numpy()
            continue
        from src.features.multistep import FEATURE_GROUPS
        from src.models.machine_learning import forecast_target

        features = tuple(dict.fromkeys(column for group in options["groups"] for column in FEATURE_GROUPS[group]))
        unknowns = {DATE_COLUMN, "origin", "is_ordinary", *TARGET_COLUMNS}
        if set(features).intersection(unknowns):
            raise ValueError("Model feature groups contain labels or time bookkeeping columns")
        values, weights = forecast_target(training, future, target, features, options["model"])
        predicted[target] = values
        importance.extend({"target": target, "feature": feature, "importance": value} for feature, value in weights.items())
    return predicted, importance


def _score(actual: pd.DataFrame, predicted: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    scores, errors = _weighted_scores(actual, predicted)
    names = [*(f"within_{threshold}" for threshold in THRESHOLDS), "excess_over_30"]
    for target in TARGET_COLUMNS:
        relative = errors[f"{target}_relative_error"].dropna()
        signs = (predicted[target].to_numpy() - actual[target].to_numpy())[relative.index]
        for threshold in THRESHOLDS:
            scores[f"{target}_within_{threshold}"] = float(relative.le(threshold / 100).mean())
        scores[f"{target}_excess_over_30"] = float(relative.sub(BAD_DAY_THRESHOLD).clip(lower=0).mean())
        severe = relative.gt(BAD_DAY_THRESHOLD)
        scores[f"{target}_severe_overprediction_rate"] = float((severe & (signs > 0)).mean())
        scores[f"{target}_severe_underprediction_rate"] = float((severe & (signs < 0)).mean())
    for name in names:
        scores[f"weighted_{name}"] = sum(weight * scores[f"{target}_{name}"] for target, weight in TARGET_WEIGHTS.items())
    return scores, errors


def _calendar_segments(fold: Fold) -> tuple[pd.DataFrame, dict, dict]:
    calendar = build_calendar(fold.future_dates[0], fold.future_dates[-1])
    calendar["is_ordinary"] = ~(calendar["is_holiday"] | calendar["is_makeup_workday"])
    calendar["day_type"] = np.select(
        [calendar["is_holiday"], calendar["is_makeup_workday"], calendar["is_regular_weekend"]],
        ["holiday", "makeup_workday", "regular_weekend"], default="regular_workday",
    )
    calendar["horizon_day"] = np.arange(1, len(calendar) + 1)
    segments = {name: calendar["day_type"].eq(name)
                for name in ("holiday", "makeup_workday", "regular_weekend", "regular_workday")}
    segments["ordinary"] = calendar["is_ordinary"]
    segments["pre_holiday"] = calendar["is_ordinary"] & calendar["is_pre_holiday"]
    segments["post_holiday"] = calendar["is_ordinary"] & calendar["is_post_holiday"]
    days = calendar["horizon_day"]
    horizons = {label: days.between(start, end) for label, start, end in (
        ("1-7", 1, 7), ("8-14", 8, 14), ("15-21", 15, 21), ("22-31", 22, 31),
    )}
    return calendar, segments, horizons


def evaluate(daily: pd.DataFrame, examples: pd.DataFrame, folds: tuple[Fold, ...], candidates: list[dict]) -> dict:
    metrics, detail, failures, day_types, horizons, importance = [], [], [], [], [], []
    for fold in folds:
        training, future = _fold_samples(examples, daily, fold)
        actual = fold_actuals(daily, fold)
        calendar, segments, bands = _calendar_segments(fold)
        for candidate in candidates:
            identity = {"model": candidate["id"], "role": fold.role, "cutoff": fold.cutoff.date().isoformat()}
            start = perf_counter()
            try:
                predicted, weights = forecast_candidate(training, future, candidate)
                elapsed = perf_counter() - start
                scores, relative = _score(actual, predicted)
                ordinary_scores, _ = _score(actual.loc[segments["ordinary"]], predicted.loc[segments["ordinary"]])
            except (ArithmeticError, KeyError, RuntimeError, TypeError, ValueError) as error:
                failures.append({**identity, "error_type": type(error).__name__, "message": str(error)})
                continue
            metrics.append({**identity, "fit_predict_seconds": elapsed, "train_rows": len(training),
                            "train_unique_label_days": training[DATE_COLUMN].nunique(),
                            "train_first_label": training[DATE_COLUMN].min().date().isoformat(),
                            "train_last_label": training[DATE_COLUMN].max().date().isoformat(),
                            "forecast_start": actual[DATE_COLUMN].iat[0].date().isoformat(),
                            "forecast_end": actual[DATE_COLUMN].iat[-1].date().isoformat(),
                            "forecast_days": len(actual), **scores,
                            **{f"ordinary_{name}": value for name, value in ordinary_scores.items()}})
            rows = actual.rename(columns={target: f"{target}_actual" for target in TARGET_COLUMNS})
            rows = rows.assign(**identity, **{f"{target}_predicted": predicted[target] for target in TARGET_COLUMNS})
            rows = rows.assign(**{
                f"{target}_signed_error": predicted[target] - actual[target] for target in TARGET_COLUMNS
            })
            detail.append(rows.join(relative).join(calendar[["horizon_day", "day_type", "is_ordinary", *DAY_FLAGS]]))
            importance.extend({**identity, **weight} for weight in weights)
            for key, masks, output in (("day_type", segments, day_types), ("horizon_band", bands, horizons)):
                for label, mask in masks.items():
                    if mask.any():
                        sub_scores, _ = _score(actual.loc[mask], predicted.loc[mask])
                        output.append({**identity, key: label, "days": int(mask.sum()), **sub_scores})
    joined = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    return {
        "fold_metrics.csv": pd.DataFrame(metrics, columns=None if metrics else ["model", "role", "cutoff"]),
        "daily_predictions.csv": joined,
        "day_type_metrics.csv": pd.DataFrame(day_types),
        "horizon_metrics.csv": pd.DataFrame(horizons),
        "feature_importance.csv": pd.DataFrame(importance, columns=["model", "role", "cutoff", "target", "feature", "importance"]),
        "failures.csv": pd.DataFrame(failures, columns=["model", "role", "cutoff", "error_type", "message"]),
    }


def run_development(daily_path: Path, config_path: Path, output: Path, cache: Path | None = DEFAULT_CACHE) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = config["candidates"]
    ids = [candidate["id"] for candidate in candidates]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be nonempty and unique")
    if output.exists():
        raise FileExistsError(f"Experiment output already exists: {output}")
    start = perf_counter()
    daily = read_daily(daily_path, DEVELOPMENT_END)
    folds = build_development_folds()
    examples = feature_table(daily, max(fold.cutoff for fold in folds), cache)
    tables = evaluate(daily, examples, folds, candidates)
    fold_metrics = tables["fold_metrics.csv"]
    selection = _summarize(fold_metrics.loc[fold_metrics["role"].eq("selection")], candidates, build_folds()[:4])
    tables["selection_summary.csv"] = selection
    hashes = _persist_tables(output, tables)
    result = {
        "role": "development", "stage": 3, "config": config, "config_sha256": _hash(config_path),
        "data": _fingerprint(daily), "git_commit": _git_head(), **_feature_provenance(),
        "folds": [{"role": fold.role, "cutoff": str(fold.cutoff.date()), "days": len(fold.future_dates)} for fold in folds],
        "success_count": len(fold_metrics), "failure_count": len(tables["failures.csv"]),
        "provisional_candidate": select_candidate(selection) if selection["eligible"].any() else None,
        "elapsed_seconds": perf_counter() - start, "artifacts_sha256": hashes,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("develop",))
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    manifest = run_development(args.daily, args.config, args.output_dir, args.cache_dir)
    print(f"Development complete: {manifest['success_count']} scored, {manifest['failure_count']} failed; "
          f"provisional candidate {manifest['provisional_candidate']}")


if __name__ == "__main__":
    main()
