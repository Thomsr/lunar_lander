import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(results_path, pop_size):
    with results_path.open() as input_file:
        payload = json.load(input_file)

    runs = [run for run in payload.get("runs", []) if run.get("pop_size") == pop_size]
    if not runs:
        raise ValueError(f"No cached runs found for pop_size={pop_size}")

    config_keys = [
        "max_gens",
        "max_tree_size",
        "n_trees",
        "train_episodes",
        "train_duration",
        "test_episodes",
        "test_duration",
    ]
    latest_config = max(
        ({key: run.get(key) for key in config_keys} for run in runs),
        key=lambda config: (
            config["max_gens"],
            config["train_episodes"],
            config["train_duration"],
            config["test_episodes"],
            config["test_duration"],
        ),
    )

    filtered_runs = [
        run
        for run in runs
        if all(run.get(key) == latest_config[key] for key in config_keys)
    ]
    return filtered_runs, latest_config


def summarize(runs):
    summary = {}
    for run in runs:
        summary.setdefault(run["init_method"], []).append(run)
    return summary


def plot_training_curve_comparison(runs, config, pop_size, output_path):
    grouped_runs = summarize(runs)
    methods = sorted(grouped_runs)

    fig, ax = plt.subplots(figsize=(9, 5))

    for method in methods:
        curves = [np.array(run["train_curve"], dtype=float) for run in grouped_runs[method]]
        max_curve_length = max(len(curve) for curve in curves)
        curve_matrix = np.full((len(curves), max_curve_length), np.nan)

        for row_idx, curve in enumerate(curves):
            curve_matrix[row_idx, : len(curve)] = curve

        mean_curve = np.nanmean(curve_matrix, axis=0)
        std_curve = np.nanstd(curve_matrix, axis=0)
        generations = np.arange(len(mean_curve))
        ax.plot(generations, mean_curve, marker="o", label=method)
        ax.fill_between(
            generations,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15,
        )

    ax.set_title(
        f"Initialization comparison, population size {pop_size}\n"
        f"Best training fitness by generation"
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best training fitness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cached initialization comparison results for one population size."
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("experiment_outputs/initialization_comparison_results.json"),
    )
    parser.add_argument("--pop-size", type=int, default=64)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("experiment_outputs/initialization_comparison_pop64.png"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runs, config = load_runs(args.results_path, args.pop_size)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_training_curve_comparison(runs, config, args.pop_size, args.output_path)

    print(f"Plotted {len(runs)} cached runs for pop_size={args.pop_size}")
    print(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
