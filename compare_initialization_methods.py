import argparse
import json
import random
from collections import deque, namedtuple
from functools import partial
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genepro.evo import Evolution
from genepro.node_impl import Constant, Div, Feature, Minus, Plus, Times


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
TRAIN_ENV = None


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        return self

    def __add__(self, other):
        self.memory = self.memory + other.memory
        return self


def get_train_env():
    global TRAIN_ENV
    if TRAIN_ENV is None:
        TRAIN_ENV = gym.make("LunarLander-v2")
    return TRAIN_ENV


def fitness_function_pt(multitree, num_episodes=5, episode_duration=300):
    env = get_train_env()
    memory = ReplayMemory(10000)
    rewards = []

    for _ in range(num_episodes):
        observation, _ = env.reset()

        for _ in range(episode_duration):
            input_sample = torch.from_numpy(observation.reshape((1, -1))).float()
            action = torch.argmax(multitree.get_output_pt(input_sample))
            observation, reward, terminated, truncated, _ = env.step(action.item())
            output_sample = torch.from_numpy(observation.reshape((1, -1))).float()

            rewards.append(reward)
            memory.push(
                input_sample,
                torch.tensor([[action.item()]]),
                output_sample,
                torch.tensor([reward]),
            )

            if terminated or truncated:
                break

    return float(np.sum(rewards)), memory


def get_test_score(tree, num_episodes=10, episode_duration=500, seed_offset=0):
    env = gym.make("LunarLander-v2")
    rewards = []

    for episode_idx in range(num_episodes):
        observation, _ = env.reset(seed=seed_offset + episode_idx)

        for _ in range(episode_duration):
            input_sample = torch.from_numpy(observation.reshape((1, -1))).float()
            action = torch.argmax(tree.get_output_pt(input_sample))
            observation, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)

            if terminated or truncated:
                break

    env.close()
    return float(np.sum(rewards))


def build_nodes():
    probe_env = gym.make("LunarLander-v2")
    num_features = probe_env.observation_space.shape[0]
    probe_env.close()

    leaf_nodes = [Feature(i) for i in range(num_features)] + [Constant()]
    internal_nodes = [Plus(), Minus(), Times(), Div()]
    return internal_nodes, leaf_nodes


def run_single_experiment(
    init_method,
    pop_size,
    seed,
    max_gens,
    max_tree_size,
    n_jobs,
    n_trees,
    train_episodes,
    train_duration,
    test_episodes,
    test_duration,
    verbose,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    internal_nodes, leaf_nodes = build_nodes()
    fitness_function = partial(
        fitness_function_pt,
        num_episodes=train_episodes,
        episode_duration=train_duration,
    )
    evo = Evolution(
        fitness_function,
        internal_nodes,
        leaf_nodes,
        n_trees,
        pop_size=pop_size,
        max_gens=max_gens,
        max_tree_size=max_tree_size,
        n_jobs=n_jobs,
        verbose=verbose,
        seed=seed,
        init_method=init_method,
    )

    evo.evolve()
    best = evo.best_of_gens[-1]
    train_curve = [float(individual.fitness) for individual in evo.best_of_gens]
    test_score = get_test_score(
        best,
        num_episodes=test_episodes,
        episode_duration=test_duration,
        seed_offset=10_000 + seed,
    )

    return {
        "init_method": init_method,
        "pop_size": pop_size,
        "seed": seed,
        "train_curve": train_curve,
        "final_train_fitness": train_curve[-1],
        "test_score": test_score,
        "best_size": len(best),
        "best_expression": best.get_readable_repr(),
    }


def experiment_cache_key(result_or_config):
    return (
        result_or_config["init_method"],
        int(result_or_config["pop_size"]),
        int(result_or_config["seed"]),
        int(result_or_config["max_gens"]),
        int(result_or_config["max_tree_size"]),
        int(result_or_config["n_trees"]),
        int(result_or_config["train_episodes"]),
        int(result_or_config["train_duration"]),
        int(result_or_config["test_episodes"]),
        int(result_or_config["test_duration"]),
    )


def load_cached_results(results_path):
    if not results_path.exists():
        return []

    with results_path.open() as input_file:
        payload = json.load(input_file)

    cached_results = payload.get("runs", [])
    for result in cached_results:
        config = payload.get("config", {})
        result.setdefault("max_gens", config.get("max_gens"))
        result.setdefault("max_tree_size", config.get("max_tree_size"))
        result.setdefault("n_trees", config.get("n_trees"))
        result.setdefault("train_episodes", config.get("train_episodes", 5))
        result.setdefault("train_duration", config.get("train_duration", 300))
        result.setdefault("test_episodes", config.get("test_episodes", 10))
        result.setdefault("test_duration", config.get("test_duration", 500))

    return cached_results


def summarize(results):
    summary = {}
    for result in results:
        key = (result["init_method"], result["pop_size"])
        summary.setdefault(key, []).append(result)

    records = []
    for (init_method, pop_size), grouped in sorted(summary.items()):
        test_scores = np.array([item["test_score"] for item in grouped], dtype=float)
        max_curve_length = max(len(item["train_curve"]) for item in grouped)
        curves = np.full((len(grouped), max_curve_length), np.nan)

        for row_idx, item in enumerate(grouped):
            curve = np.array(item["train_curve"], dtype=float)
            curves[row_idx, : len(curve)] = curve

        records.append(
            {
                "init_method": init_method,
                "pop_size": pop_size,
                "n_runs": len(grouped),
                "mean_test_score": float(np.mean(test_scores)),
                "std_test_score": float(np.std(test_scores)),
                "mean_train_curve": np.nanmean(curves, axis=0).tolist(),
                "std_train_curve": np.nanstd(curves, axis=0).tolist(),
            }
        )

    return records


def plot_final_scores(summary, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = sorted({record["init_method"] for record in summary})

    for method in methods:
        records = [record for record in summary if record["init_method"] == method]
        records.sort(key=lambda item: item["pop_size"])
        pop_sizes = [record["pop_size"] for record in records]
        means = [record["mean_test_score"] for record in records]
        stds = [record["std_test_score"] for record in records]
        ax.errorbar(pop_sizes, means, yerr=stds, marker="o", capsize=4, label=method)

    ax.set_title("Initialization method comparison")
    ax.set_xlabel("Population size")
    ax.set_ylabel("Mean test fitness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_learning_curves(summary, output_path):
    pop_sizes = sorted({record["pop_size"] for record in summary})
    fig, axes = plt.subplots(
        len(pop_sizes),
        1,
        figsize=(9, max(4, 3 * len(pop_sizes))),
        sharex=True,
        squeeze=False,
    )

    for ax, pop_size in zip(axes[:, 0], pop_sizes):
        records = [record for record in summary if record["pop_size"] == pop_size]
        for record in sorted(records, key=lambda item: item["init_method"]):
            mean_curve = np.array(record["mean_train_curve"], dtype=float)
            std_curve = np.array(record["std_train_curve"], dtype=float)
            generations = np.arange(len(mean_curve))

            ax.plot(generations, mean_curve, marker="o", label=record["init_method"])
            ax.fill_between(
                generations,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.15,
            )

        ax.set_title(f"Population size {pop_size}")
        ax.set_ylabel("Best train fitness")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1, 0].set_xlabel("Generation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ramped half-and-half initialization against random initialization."
    )
    parser.add_argument("--pop-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gens", type=int, default=50)
    parser.add_argument("--max-tree-size", type=int, default=31)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-trees", type=int, default=4)
    parser.add_argument("--train-episodes", type=int, default=5)
    parser.add_argument("--train-duration", type=int, default=300)
    parser.add_argument("--test-episodes", type=int, default=10)
    parser.add_argument("--test-duration", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_outputs"))
    parser.add_argument("--verbose-evo", action="store_true")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing JSON results and recompute all requested runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "initialization_comparison_results.json"
    final_plot_path = args.output_dir / "initialization_comparison_final_scores.png"
    curve_plot_path = args.output_dir / "initialization_comparison_learning_curves.png"

    init_methods = ["random", "ramped_half_and_half"]
    seeds = [args.seed + run_idx for run_idx in range(args.runs)]
    cached_results = [] if args.no_cache else load_cached_results(results_path)
    cache = {experiment_cache_key(result): result for result in cached_results}
    results = []

    total = len(init_methods) * len(args.pop_sizes) * len(seeds)
    completed = 0

    for pop_size in args.pop_sizes:
        for init_method in init_methods:
            for seed in seeds:
                completed += 1
                run_config = {
                    "init_method": init_method,
                    "pop_size": pop_size,
                    "seed": seed,
                    "max_gens": args.max_gens,
                    "max_tree_size": args.max_tree_size,
                    "n_trees": args.n_trees,
                    "train_episodes": args.train_episodes,
                    "train_duration": args.train_duration,
                    "test_episodes": args.test_episodes,
                    "test_duration": args.test_duration,
                }
                cache_key = experiment_cache_key(run_config)
                if cache_key in cache:
                    results.append(cache[cache_key])
                    print(
                        f"[{completed}/{total}] cache hit: init={init_method}, "
                        f"pop_size={pop_size}, seed={seed}",
                        flush=True,
                    )
                    continue

                print(
                    f"[{completed}/{total}] init={init_method}, "
                    f"pop_size={pop_size}, seed={seed}",
                    flush=True,
                )
                result = run_single_experiment(
                    init_method=init_method,
                    pop_size=pop_size,
                    seed=seed,
                    max_gens=args.max_gens,
                    max_tree_size=args.max_tree_size,
                    n_jobs=args.n_jobs,
                    n_trees=args.n_trees,
                    train_episodes=args.train_episodes,
                    train_duration=args.train_duration,
                    test_episodes=args.test_episodes,
                    test_duration=args.test_duration,
                    verbose=args.verbose_evo,
                )
                result.update(run_config)
                results.append(result)
                cache[cache_key] = result
                print(
                    f"    test_score={result['test_score']:.3f}, "
                    f"final_train={result['final_train_fitness']:.3f}",
                    flush=True,
                )

                summary = summarize(results)
                payload = {
                    "config": {
                        "pop_sizes": args.pop_sizes,
                        "runs": args.runs,
                        "seeds": seeds,
                        "max_gens": args.max_gens,
                        "max_tree_size": args.max_tree_size,
                        "n_jobs": args.n_jobs,
                        "n_trees": args.n_trees,
                        "train_episodes": args.train_episodes,
                        "train_duration": args.train_duration,
                        "test_episodes": args.test_episodes,
                        "test_duration": args.test_duration,
                    },
                    "summary": summary,
                    "runs": list(cache.values()),
                }
                with results_path.open("w") as output_file:
                    json.dump(payload, output_file, indent=2)

    summary = summarize(results)
    payload = {
        "config": {
            "pop_sizes": args.pop_sizes,
            "runs": args.runs,
            "seeds": seeds,
            "max_gens": args.max_gens,
            "max_tree_size": args.max_tree_size,
            "n_jobs": args.n_jobs,
            "n_trees": args.n_trees,
            "train_episodes": args.train_episodes,
            "train_duration": args.train_duration,
            "test_episodes": args.test_episodes,
            "test_duration": args.test_duration,
        },
        "summary": summary,
        "runs": list(cache.values()),
    }

    with results_path.open("w") as output_file:
        json.dump(payload, output_file, indent=2)

    plot_final_scores(summary, final_plot_path)
    plot_learning_curves(summary, curve_plot_path)

    print(f"Saved results to {results_path}")
    print(f"Saved final-score plot to {final_plot_path}")
    print(f"Saved learning-curve plot to {curve_plot_path}")


if __name__ == "__main__":
    main()
