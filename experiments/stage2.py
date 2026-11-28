"""Stage-two model development, selection freeze and fixed August evaluation."""

import argparse
from collections.abc import Callable
import hashlib
import json
from functools import partial
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.baselines import DEFAULT_DAILY, _sha256
from src.evaluation import (
    BAD_DAY_THRESHOLD,
    DAILY_COLUMNS,
    TARGET_WEIGHTS,
    Fold,
    SingleTargetModel,
    build_folds,
    predict_targets,
    score_forecast,
    split_fold,
)
from src.features.calendar import CONFIG as CALENDAR_CONFIG, build_calendar
from src.models.baselines import BASELINES, DATE_COLUMN, TARGET_COLUMNS, SeasonalMeanBaseline

DEFAULT_CONFIG = Path("configs/stage2.json")
DEVELOPMENT_END = pd.Timestamp("2014-07-31")
HOLDOUT_END = pd.Timestamp("2014-08-31")
ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "experiments/stage2.py",
    "experiments/baselines.py",
    "src/evaluation.py",
    "src/features/calendar.py",
    "src/models/baselines.py",
    "src/models/calendar_adjustment.py",
    "src/models/statistical.py",
)
DEVELOPMENT_ARTIFACTS = {
    "fold_metrics.csv", "daily_predictions.csv", "failures.csv", "day_type_metrics.csv", "horizon_metrics.csv",
    "selection_summary.csv",
}
SELECTION_KEYS = (
    "ordinary_weighted_bad_day_rate_mean",
    "ordinary_weighted_mape_std",
    "ordinary_weighted_mape_mean",
    "weighted_mape_mean",
    "model",
)
CLOSE_MARGIN = 0.005
SELECTION_POLICY = {
    "near_best_margin": CLOSE_MARGIN,
    "near_best_order": list(SELECTION_KEYS),
    "ordinary_days": "workday or regular weekend, excluding holidays and makeup workdays; includes holiday edges",
    "fold_weighting": "equal",
    "sample_std_ddof": 1,
    "target_weights": TARGET_WEIGHTS,
    "bad_day_threshold_strictly_greater_than": BAD_DAY_THRESHOLD,
}
FAILURE_COLUMNS = ("model", "role", "cutoff", "error_type", "message")
DAY_FLAGS = (
    "is_workday", "is_regular_weekend", "is_holiday", "is_makeup_workday", "is_pre_holiday", "is_post_holiday",
)
RELIABILITY_METRICS = ("weighted_mape", "ordinary_weighted_bad_day_rate", "ordinary_weighted_mape")
ModelFactory = Callable[[str], SingleTargetModel]


class MeanEnsemble:
    def __init__(self, target: str, factories: list[ModelFactory]) -> None:
        self.models = [factory(target) for factory in factories]

    def fit(self, history: pd.DataFrame) -> None:
        for model in self.models:
            model.fit(history.copy())

    def predict(self, future_calendar: pd.DataFrame) -> pd.Series:
        forecasts = [model.predict(future_calendar.copy()) for model in self.models]
        return sum(forecasts) / len(forecasts)


def build_model_factory(candidate: dict, candidates: list[dict]) -> tuple[ModelFactory, tuple[str, ...]]:
    """Construct the same factory for development, holdout and frozen-model consumers."""
    by_id = {item["id"]: item for item in candidates}

    def resolve(item: dict, chain: tuple[str, ...]) -> tuple[ModelFactory, tuple[str, ...]]:
        model_id, kind = item["id"], item["kind"]
        if model_id in chain:
            raise ValueError(f"Circular candidate dependency: {(*chain, model_id)}")
        chain = (*chain, model_id)
        params = item.get("params", {})
        if kind == "baseline":
            specs = item["specs"]
            if set(specs) != set(TARGET_COLUMNS) or not set(specs.values()).issubset(BASELINES):
                raise ValueError("Baseline candidate requires a known baseline for each target")
            return lambda target: SeasonalMeanBaseline(BASELINES[specs[target]], target), ()
        if kind in {"recent_weekday", "level_weekday", "weekly_ets"}:
            from src.models.statistical import LevelWeekday, RecentWeekday, WeeklyETS

            constructor = {"recent_weekday": RecentWeekday, "level_weekday": LevelWeekday, "weekly_ets": WeeklyETS}[kind]
            per_target = item.get("target_params", {})
            if not set(per_target).issubset(TARGET_COLUMNS):
                raise ValueError("Target parameters must use the purchase and redeem target names")
            return lambda target: constructor(target, **(params | per_target.get(target, {}))), ()
        if kind == "calendar_adjusted":
            from src.models.calendar_adjustment import CALENDAR_COLUMNS, CalendarAdjusted

            base_factory, base_columns = resolve(by_id[item["base"]], chain)
            columns = tuple(dict.fromkeys((*base_columns, *CALENDAR_COLUMNS)))
            return lambda target: CalendarAdjusted(target, base_factory, **params), columns
        if kind == "target_mix":
            members = item["members"]
            if set(members) != set(TARGET_COLUMNS):
                raise ValueError("Target mix requires a candidate for each target")
            resolved = {target: resolve(by_id[member], chain) for target, member in members.items()}
            columns = tuple(dict.fromkeys(column for _, features in resolved.values() for column in features))
            return lambda target: resolved[target][0](target), columns
        if kind == "mean_ensemble":
            members = item["members"]
            if len(members) < 2 or len(set(members)) != len(members):
                raise ValueError("Equal-weight ensemble requires at least two distinct members")
            resolved = [resolve(by_id[member], chain) for member in members]
            factories = [factory for factory, _ in resolved]
            columns = tuple(dict.fromkeys(column for _, features in resolved for column in features))
            return partial(MeanEnsemble, factories=factories), columns
        raise ValueError(f"Unknown candidate kind: {kind}")

    return resolve(candidate, ())


def _validate_candidates(candidates: list[dict]) -> None:
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("A nonempty ordered list of candidates is required")
    fields = {
        "baseline": {"specs"},
        "recent_weekday": {"params", "target_params"},
        "level_weekday": {"params", "target_params"},
        "weekly_ets": {"params", "target_params"},
        "calendar_adjusted": {"base", "params"},
        "target_mix": {"members"},
        "mean_ensemble": {"members"},
    }
    for item in candidates:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise ValueError("Every candidate needs a nonempty string ID")
        kind = item.get("kind")
        if kind not in fields or set(item).difference({"id", "kind"}, fields[kind]):
            raise ValueError(f"Unknown kind or fields for candidate {item['id']}")
    ids = {item["id"] for item in candidates}
    if len(ids) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    for item in candidates:
        referenced = [item.get("base")] if item["kind"] == "calendar_adjusted" else item.get("members", [])
        if item["kind"] == "target_mix":
            referenced = referenced.values()
        if not set(referenced).issubset(ids):
            raise ValueError(f"Unknown dependency in candidate {item['id']}")


def _read_scoped_daily(path: Path, last_date: pd.Timestamp) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        daily = pd.read_parquet(path, columns=list(DAILY_COLUMNS), filters=[(DATE_COLUMN, "<=", last_date)])
        dates = pd.to_datetime(daily[DATE_COLUMN], errors="raise")
    elif path.suffix.lower() == ".csv":
        # CSV has no predicate pushdown. Discard later rows before interpreting their targets.
        daily = pd.read_csv(path, usecols=list(DAILY_COLUMNS), dtype="string")
        date_text = daily[DATE_COLUMN]
        date_format = "%Y%m%d" if date_text.str.fullmatch(r"[0-9]{8}").all() else "%Y-%m-%d"
        dates = pd.to_datetime(date_text, format=date_format, errors="raise")
        daily = daily.loc[dates.le(last_date) | dates.isna()].copy()
        dates = dates.loc[daily.index]
        for target in TARGET_COLUMNS:
            if not daily[target].str.fullmatch(r"[0-9]+").fillna(False).all():
                raise ValueError(f"Daily {target} must be nonnegative integer cents")
            daily[target] = daily[target].astype("int64")
    else:
        raise ValueError("Daily input must be a .parquet or .csv file")

    if dates.dt.tz is not None or dates.isna().any() or (dates != dates.dt.normalize()).any():
        raise ValueError("Daily data must have valid dates at midnight without a time zone")
    daily[DATE_COLUMN] = dates.astype("datetime64[ns]")
    for target in TARGET_COLUMNS:
        values = daily[target]
        if not pd.api.types.is_integer_dtype(values) or values.isna().any() or (values < 0).any():
            raise ValueError(f"Daily {target} must be nonnegative integer cents")
    return daily.loc[:, list(DAILY_COLUMNS)].sort_values(DATE_COLUMN).reset_index(drop=True)


def _data_fingerprint(daily: pd.DataFrame) -> dict:
    if daily.empty:
        raise ValueError("Daily input is empty")
    content = daily.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n")
    return {
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "scope_start": daily[DATE_COLUMN].iloc[0].date().isoformat(),
        "scope_end": daily[DATE_COLUMN].iloc[-1].date().isoformat(),
        "rows": len(daily),
        "serialization": "sorted report_date,purchase,redeem UTF-8 CSV with LF and ISO dates",
    }


def _json_fingerprint(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _provenance() -> dict:
    return {
        "source_sha256": {name: _sha256(ROOT / name) for name in SOURCE_PATHS},
        "calendar_sha256": _sha256(CALENDAR_CONFIG),
        "environment": {"python": platform.python_version(), **{
            library: version(library) for library in ("numpy", "pandas", "scipy", "statsmodels", "pyarrow")
        }},
    }


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _weighted_scores(actual: pd.DataFrame, predicted: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    scores, relative = score_forecast(actual, predicted)
    scores["weighted_bad_day_rate"] = sum(
        TARGET_WEIGHTS[target] * scores[f"{target}_bad_day_rate"] for target in TARGET_COLUMNS
    )
    for target in TARGET_COLUMNS:
        residual = predicted[target].to_numpy(dtype=float) - actual[target].to_numpy(dtype=float)
        scores[f"{target}_mean_signed_error"] = float(residual.mean())
        scores[f"{target}_overprediction_rate"] = float((residual > 0).mean())
    return scores, relative


def _evaluate(daily: pd.DataFrame, folds: tuple[Fold, ...], candidates: list[dict]) -> tuple[dict, pd.DataFrame]:
    metrics, failures, day_types, horizons, predictions = [], [], [], [], []
    for fold in folds:
        history, actual = split_fold(daily, fold)
        calendar = build_calendar(fold.future_dates[0], fold.future_dates[-1])
        calendar["is_ordinary"] = (calendar["is_workday"] | calendar["is_regular_weekend"]) & ~(
            calendar["is_holiday"] | calendar["is_makeup_workday"]
        )
        calendar["day_type"] = np.select(
            [calendar["is_holiday"], calendar["is_makeup_workday"], calendar["is_regular_weekend"]],
            ["holiday", "makeup_workday", "regular_weekend"],
            default="regular_workday",
        )
        calendar["is_month_end"] = (
            calendar["is_ordinary"] & ~calendar["is_pre_holiday"] & ~calendar["is_post_holiday"]
            & calendar[DATE_COLUMN].dt.day.ge(calendar[DATE_COLUMN].dt.days_in_month - 2)
        )
        horizon_day = np.arange(1, len(calendar) + 1)
        segments = {
            "ordinary": calendar["is_ordinary"],
            "regular_workday": calendar["day_type"].eq("regular_workday"),
            "regular_weekend": calendar["day_type"].eq("regular_weekend"),
            "holiday": calendar["is_holiday"],
            "makeup_workday": calendar["is_makeup_workday"],
            "pre_holiday": calendar["is_ordinary"] & calendar["is_pre_holiday"],
            "post_holiday": calendar["is_ordinary"] & calendar["is_post_holiday"] & ~calendar["is_pre_holiday"],
            "month_end": calendar["is_month_end"],
        }
        horizon_segments = {
            "1-7": horizon_day <= 7,
            "8-14": (horizon_day >= 8) & (horizon_day <= 14),
            "15-21": (horizon_day >= 15) & (horizon_day <= 21),
            "22-31": horizon_day >= 22,
        }
        for candidate in candidates:
            identity = {"model": candidate["id"], "role": fold.role, "cutoff": fold.cutoff.date().isoformat()}
            start = perf_counter()
            try:
                factory, columns = build_model_factory(candidate, candidates)
                predicted = predict_targets(history, fold.future_dates, factory, calendar_columns=columns)
            except (ArithmeticError, KeyError, RuntimeError, TypeError, ValueError) as error:
                failures.append({**identity, "error_type": type(error).__name__, "message": str(error)})
                continue
            elapsed = perf_counter() - start
            scores, relative = _weighted_scores(actual, predicted)
            ordinary, _ = _weighted_scores(actual.loc[segments["ordinary"]], predicted.loc[segments["ordinary"]])
            fold_scores = {**scores, **{f"ordinary_{metric}": value for metric, value in ordinary.items()}}
            result = {
                **identity,
                "train_start": history[DATE_COLUMN].iloc[0].date().isoformat(),
                "train_end": history[DATE_COLUMN].iloc[-1].date().isoformat(),
                "train_days": len(history),
                "forecast_start": fold.future_dates[0].date().isoformat(),
                "forecast_end": fold.future_dates[-1].date().isoformat(),
                "forecast_days": len(actual),
                "fit_predict_seconds": elapsed,
                **fold_scores,
            }
            undefined = [name for name in RELIABILITY_METRICS if not np.isfinite(fold_scores[name])]
            if undefined:
                failures.append({**identity, "error_type": "UndefinedMetric", "message": ", ".join(undefined)})
            else:
                metrics.append(result)
            rows = actual.rename(columns={target: f"{target}_actual" for target in TARGET_COLUMNS}).assign(
                **identity,
                horizon_day=horizon_day,
                **{f"{target}_predicted": predicted[target] for target in TARGET_COLUMNS},
                **{f"{target}_signed_error": predicted[target] - actual[target] for target in TARGET_COLUMNS},
            ).join(relative).join(calendar.loc[:, ["weekday", "holiday", *DAY_FLAGS, "is_ordinary", "day_type", "is_month_end"]])
            predictions.append(rows)
            for label, masks, report in (("day_type", segments, day_types), ("horizon_band", horizon_segments, horizons)):
                for segment, mask in masks.items():
                    if mask.any():
                        segment_scores, _ = _weighted_scores(actual.loc[mask], predicted.loc[mask])
                        report.append({**identity, label: segment, "days": int(mask.sum()), **segment_scores})

    fold_metrics = pd.DataFrame(metrics) if metrics else pd.DataFrame(columns=["model", "role", "cutoff"])
    tables = {
        "fold_metrics.csv": fold_metrics,
        "daily_predictions.csv": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        "failures.csv": pd.DataFrame(failures, columns=FAILURE_COLUMNS),
        "day_type_metrics.csv": pd.DataFrame(day_types),
        "horizon_metrics.csv": pd.DataFrame(horizons),
    }
    return tables, fold_metrics


def _summarize(fold_metrics: pd.DataFrame, candidates: list[dict], folds: tuple[Fold, ...]) -> pd.DataFrame:
    required = {fold.cutoff.date().isoformat() for fold in folds}
    rows = []
    for candidate in candidates:
        model = candidate["id"]
        results = fold_metrics.loc[fold_metrics["model"].eq(model)] if not fold_metrics.empty else fold_metrics
        complete = len(results) == len(required) and set(results["cutoff"]) == required
        eligible = bool(complete and np.isfinite(results.loc[:, list(RELIABILITY_METRICS)]).all().all())
        values = results.select_dtypes("number").drop(columns=["train_days", "forecast_days"], errors="ignore")
        rows.append({
            "model": model,
            "completed_folds": len(results),
            "eligible": eligible,
            **{f"{key}_mean": value for key, value in values.mean().items()},
            **{f"{key}_std": value for key, value in values.std(ddof=1).items()},
        })
    summary = pd.DataFrame(rows)
    for key in SELECTION_KEYS[:-1]:
        if key not in summary:
            summary[key] = np.nan
    best_mape = summary.loc[summary["eligible"], "weighted_mape_mean"].min()
    gap = summary["weighted_mape_mean"] - best_mape
    summary["near_best"] = summary["eligible"] & gap.le(CLOSE_MARGIN + 1e-12)
    return summary.sort_values(
        ["eligible", "near_best", "weighted_mape_mean", "model"], ascending=[False, False, True, True]
    ).reset_index(drop=True)


def select_candidate(summary: pd.DataFrame) -> str:
    eligible = summary.loc[summary["eligible"]]
    if eligible.empty:
        raise ValueError("No candidate completed all development folds with finite selection metrics")
    best = eligible["weighted_mape_mean"].min()
    close = eligible.loc[eligible["weighted_mape_mean"].sub(best).le(CLOSE_MARGIN + 1e-12)]
    return str(close.sort_values(list(SELECTION_KEYS)).iloc[0]["model"])


def _persist_tables(output_dir: Path, tables: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, table in tables.items():
        path = output_dir / name
        table.to_csv(path, index=False, date_format="%Y-%m-%d", lineterminator="\n")
        hashes[name] = _sha256(path)
    return hashes


def run_development(daily_path: Path, config_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = config["candidates"]
    _validate_candidates(candidates)
    provenance = _provenance()
    daily = _read_scoped_daily(daily_path, DEVELOPMENT_END)
    data = _data_fingerprint(daily)
    folds = tuple(fold for fold in build_folds() if fold.role == "selection")
    tables, metrics = _evaluate(daily, folds, candidates)
    summary = _summarize(metrics, candidates, folds)
    tables["selection_summary.csv"] = summary.assign(selected=False)

    try:
        selected_model = select_candidate(summary)
    except ValueError:
        _persist_tables(output_dir, tables)
        raise

    tables["selection_summary.csv"]["selected"] = summary["model"].eq(selected_model)
    artifact_hashes = _persist_tables(output_dir, tables)
    plan = {
        "schema_version": 1,
        "selected_model": selected_model,
        "candidates": candidates,
        "config_sha256": _sha256(config_path),
        "development_input": data,
        "development_cutoffs": [fold.cutoff.date().isoformat() for fold in folds],
        "holdout_dates": ["2014-08-01", "2014-08-31"],
        "selection_policy": SELECTION_POLICY,
        **provenance,
        "development_artifact_sha256": artifact_hashes,
    }
    plan["plan_sha256"] = _json_fingerprint(plan)
    _write_json(output_dir / "manifest.json", {"role": "development", "git_commit": _git_head(), **plan})
    freeze_path = output_dir / "freeze.json"
    _write_json(freeze_path, plan)
    return freeze_path


def read_freeze(freeze_path: Path) -> dict:
    """Validate the frozen decision and its development evidence before reading holdout data."""
    plan = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_hash = plan.pop("plan_sha256", None)
    if expected_hash != _json_fingerprint(plan) or plan.get("schema_version") != 1:
        raise ValueError("Frozen plan digest or schema does not match")
    if plan.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("Frozen plan selection policy has changed")
    if any(plan.get(name) != value for name, value in _provenance().items()):
        raise ValueError("Frozen plan source, calendar or environment fingerprints do not match")
    *development, holdout = build_folds()
    if (
        plan.get("development_cutoffs") != [fold.cutoff.date().isoformat() for fold in development]
        or plan.get("holdout_dates") != [holdout.future_dates[0].date().isoformat(), holdout.future_dates[-1].date().isoformat()]
    ):
        raise ValueError("Frozen dates do not match the development and holdout protocol")
    _validate_candidates(plan["candidates"])
    artifact_hashes = plan.get("development_artifact_sha256", {})
    if set(artifact_hashes) != DEVELOPMENT_ARTIFACTS:
        raise ValueError("Frozen development evidence is incomplete")
    for name, expected in artifact_hashes.items():
        path = freeze_path.parent / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Frozen development evidence does not match: {name}")
    summary = pd.read_csv(
        freeze_path.parent / "selection_summary.csv",
        usecols=["eligible", *SELECTION_KEYS],
        dtype={"model": "string"},
        keep_default_na=False,
        na_values={column: [""] for column in SELECTION_KEYS[:-1]},
    )
    if select_candidate(summary) != plan["selected_model"]:
        raise ValueError("Frozen model does not match the development selection")
    return {**plan, "plan_sha256": expected_hash}


def run_holdout(daily_path: Path, freeze_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    plan = read_freeze(freeze_path)
    daily = _read_scoped_daily(daily_path, DEVELOPMENT_END)
    if _data_fingerprint(daily) != plan["development_input"]:
        raise ValueError("Frozen development input fingerprint does not match")
    daily = _read_scoped_daily(daily_path, HOLDOUT_END)
    if _data_fingerprint(daily.loc[daily[DATE_COLUMN].le(DEVELOPMENT_END)]) != plan["development_input"]:
        raise ValueError("Frozen development input changed while opening the holdout data")
    fold = next(fold for fold in build_folds() if fold.role == "holdout")
    tables, metrics = _evaluate(daily, (fold,), plan["candidates"])
    artifact_hashes = _persist_tables(output_dir, tables)
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, {
        "role": "holdout",
        "git_commit": _git_head(),
        "plan_sha256": plan["plan_sha256"],
        "freeze_file_sha256": _sha256(freeze_path),
        "selected_model": plan["selected_model"],
        "candidates": plan["candidates"],
        **{key: plan[key] for key in ("config_sha256", "source_sha256", "calendar_sha256", "environment")},
        "development_input": plan["development_input"],
        "holdout_input": _data_fingerprint(daily.loc[daily[DATE_COLUMN].gt(DEVELOPMENT_END)]),
        "holdout_artifact_sha256": artifact_hashes,
    })
    if metrics.empty or not metrics["model"].eq(plan["selected_model"]).any():
        raise RuntimeError("The frozen selected model failed on the holdout; see failures.csv")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    develop = commands.add_parser("develop", help="Evaluate April-July and write the selection freeze")
    develop.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    holdout = commands.add_parser("holdout", help="Verify the freeze and evaluate all frozen candidates in August")
    holdout.add_argument("--freeze", type=Path, required=True)
    for command in (develop, holdout):
        command.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "develop":
        path = run_development(args.daily, args.config, args.output_dir)
    else:
        path = run_holdout(args.daily, args.freeze, args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
