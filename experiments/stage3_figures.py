"""Plot completed Stage 3 development comparisons."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


MODELS = {
    "B3": "Stage 1 B3",
    "blend_equal": "Stage 2 equal blend",
    "cal_workday": "Stage 2 calendar",
    "route_special": "Calendar routing",
    "ridge_additive": "Ridge + additive residual",
    "lightgbm_ratio": "LightGBM + ratio residual",
    "ridge_log_amount": "Direct log-amount Ridge",
    "lightgbm_relative_mae": "Direct weighted-MAE LightGBM",
}


def plot_development(metrics_dir: Path, path: Path) -> None:
    folds = pd.read_csv(metrics_dir / "fold_metrics.csv")
    folds = folds.loc[folds["role"].eq("selection") & folds["model"].isin(MODELS)]
    scores = folds.pivot(index="model", columns="forecast_start", values="weighted_mape").loc[list(MODELS)] * 100
    if scores.shape[1] != 4 or scores.isna().any().any():
        raise ValueError("Development figure requires four completed monthly folds for every plotted candidate")
    scores.columns = pd.to_datetime(scores.columns).strftime("%b")
    averages = scores.mean(axis=1)
    order = averages.sort_values().index
    scores, averages = scores.loc[order], averages.loc[order]
    names = [MODELS[model] for model in order]
    grid = scores.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), gridspec_kw={"width_ratios": [1.3, 1]}, layout="constrained")
    fig.suptitle("Fund purchase and redemption | Stage 3 development", fontsize=15, fontweight="bold", x=.02, ha="left")
    matrix = axes[0].imshow(grid, cmap="YlGnBu", vmin=10, vmax=50, aspect="auto")
    axes[0].set_xticks(range(4), scores.columns)
    axes[0].set_yticks(range(len(scores)), names)
    axes[0].set_title("Calendar-month weighted MAPE (%)", loc="left", fontsize=11)
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            axes[0].text(column, row, f"{value:.1f}", ha="center", va="center",
                         color="white" if value >= 34 else "#12293d", fontsize=9)
    fig.colorbar(matrix, ax=axes[0], shrink=.82, pad=.025)
    colors = ["#0d7467" if model == "route_special" else "#5b7e99" for model in order]
    bars = axes[1].barh(range(len(scores)), averages, color=colors, height=.55)
    axes[1].set(ylim=(len(scores) - .5, -.5), xlim=(0, averages.max() * 1.20), yticks=[])
    axes[1].set_title("Four-month mean (%)", loc="left", fontsize=11)
    axes[1].bar_label(bars, labels=[f"{value:.2f}" for value in averages], padding=4, fontsize=9)
    for side in ("top", "right", "left"):
        axes[1].spines[side].set_visible(False)
    axes[1].grid(axis="x", alpha=.18)
    axes[1].set_axisbelow(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics/stage3-development"))
    parser.add_argument("--output", type=Path, default=Path("results/figures/stage3_development_comparison.png"))
    args = parser.parse_args()
    plot_development(args.metrics, args.output)


if __name__ == "__main__":
    main()
