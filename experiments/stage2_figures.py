"""Render stage-two diagnostics from saved experimental tables."""

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


def plot_makeup(events: pd.DataFrame, output: Path) -> None:
    data = events.loc[events["kind"].eq("makeup")].pivot(
        index="start", columns="model", values="weighted_mape"
    ).sort_index()
    chinese = any(font.name == "Microsoft YaHei" for font in font_manager.fontManager.ttflist)
    palette = {"B3+B5": "#5d7890", "ordinary_workday": "#107468"}
    with plt.rc_context({"font.family": ["Microsoft YaHei", "DejaVu Sans"] if chinese else ["DejaVu Sans"]}):
        fig, ax = plt.subplots(figsize=(8.5, 3.8), layout="constrained", facecolor="#f8fafc")
        ax.set_facecolor("#f8fafc")
        for model, label, offset in (
            ("B3+B5", "自然星期" if chinese else "Calendar weekday", -0.16),
            ("ordinary_workday", "普通工作日参照" if chinese else "Regular workday reference", 0.16),
        ):
            rows = [index + offset for index in range(len(data))]
            values = data[model]
            ax.barh(rows, values, height=0.28, color=palette[model], label=label)
            for y, value in zip(rows, values):
                ax.text(value + 0.012, y, f"{value:.1%}", va="center", fontsize=9)
        ax.set_title("补班日逐日前向误差" if chinese else "Forward evaluation on makeup workdays", loc="left", weight="bold")
        ax.set_xlabel("加权 MAPE · 每次只使用该日之前的数据" if chinese else "Weighted MAPE · fitted before each event", loc="left")
        ax.set_yticks(range(len(data)), data.index)
        ax.invert_yaxis()
        ax.set_xlim(0, float(data.max().max()) + 0.14)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#d9e2e7")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        plt.close(fig)


def plot_selection(folds: pd.DataFrame, selected: str, output: Path) -> None:
    dev = folds.loc[folds["role"].eq("selection")]
    summary = dev.groupby("model").agg(
        folds=("weighted_mape", "count"),
        weighted_mape=("weighted_mape", "mean"),
        ordinary_bad=("ordinary_weighted_bad_day_rate", "mean"),
    ).query("folds == 4").sort_values("weighted_mape")
    required = summary.index.str.lower().isin({"b3", "b3_b5", "b3+b5", selected.lower()})
    chosen = summary.index.isin(summary.head(8).index) | required
    shown = summary.loc[chosen]
    chinese = any(font.name == "Microsoft YaHei" for font in font_manager.fontManager.ttflist)
    colors = ["#107468" if model == selected else "#7595a8" for model in shown.index]
    with plt.rc_context({"font.family": ["Microsoft YaHei", "DejaVu Sans"] if chinese else ["DejaVu Sans"]}):
        fig, panels = plt.subplots(1, 2, sharey=True, figsize=(9, 4.7), layout="constrained", facecolor="#f8fafc")
        for ax, field, title in (
            (panels[0], "weighted_mape", "整体加权 MAPE" if chinese else "Overall weighted MAPE"),
            (panels[1], "ordinary_bad", "普通日误差超 30% 比例" if chinese else "Ordinary-day weighted bad day rate"),
        ):
            ax.set_facecolor("#f8fafc")
            values = shown[field]
            ax.barh(shown.index, values, height=0.58, color=colors)
            for index, value in enumerate(values):
                ax.text(value + 0.006, index, f"{value:.1%}", va="center", fontsize=8)
            ax.set_xlim(0, float(values.max()) + 0.06)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title(title, loc="left")
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#d9e2e7")
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        panels[0].invert_yaxis()
        fig.suptitle("2014 年 4—7 月开发折 · 绿色为入选模型" if chinese else "April–July development folds · green marks selection", weight="bold")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("results/metrics/stage2-events/event_metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/figures/stage2_makeup_comparison.png"))
    parser.add_argument("--development", type=Path)
    parser.add_argument("--selection-output", type=Path, default=Path("results/figures/stage2_development_selection.png"))
    args = parser.parse_args()
    plot_makeup(pd.read_csv(args.events), args.output)
    if args.development is not None:
        freeze = json.loads((args.development / "freeze.json").read_text(encoding="utf-8"))
        plot_selection(pd.read_csv(args.development / "fold_metrics.csv"), freeze["selected_model"], args.selection_output)


if __name__ == "__main__":
    main()
