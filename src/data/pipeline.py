import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .aggregate import behavior_features_as_of, build_aggregates
from .audit import build_audit
from .inputs import load_inputs


__all__ = ["behavior_features_as_of", "load_inputs", "run_pipeline"]


def run_pipeline(raw_dir: Path, processed_dir: Path, output_dir: Path) -> None:
    inputs = load_inputs(Path(raw_dir))
    daily, groups = build_aggregates(inputs)
    audit, coverage = build_audit(inputs, daily, groups)

    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=processed_dir) as processed_temp, TemporaryDirectory(dir=output_dir) as audit_temp:
        parquet_files = {
            "transactions.parquet": inputs.transactions,
            "group_daily.parquet": groups,
            "daily.parquet": daily,
        }
        for name, table in parquet_files.items():
            table.to_parquet(Path(processed_temp) / name, index=False)
        (Path(audit_temp) / "data_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        coverage.to_csv(
            Path(audit_temp) / "coverage_by_day.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n"
        )

        for name in parquet_files:
            (Path(processed_temp) / name).replace(processed_dir / name)
        for name in ("data_audit.json", "coverage_by_day.csv"):
            (Path(audit_temp) / name).replace(output_dir / name)

    print(f"生成 {len(inputs.transactions):,} 条用户日记录、{len(groups):,} 条分群日记录、{len(daily):,} 条全量日记录")
    print(f"Parquet: {processed_dir} | 审计: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="预处理资金流入流出竞赛数据并导出聚合审计")
    parser.add_argument("--raw-dir", type=Path, default=Path("Purchase Redemption Data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/audit"))
    args = parser.parse_args()
    run_pipeline(args.raw_dir, args.processed_dir, args.output_dir)


if __name__ == "__main__":
    main()
