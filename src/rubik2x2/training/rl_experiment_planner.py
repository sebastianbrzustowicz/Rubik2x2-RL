import os
import csv
import itertools
import time
from rubik2x2.training.train_rl import train_rl_agent
import torch


def generate_experiments():
    grid = {
        "reward_mode": [
            "bottom_layer_corners",
        ],
        "scramble_min": [1],
        "scramble_max": [12],
        "resets_per_jump": [100000],
        "total_steps": [3000000],
        "max_steps": [30],
        "batch_size": [256],
        "lr": [1e-4],
        "gamma": [0.99],
        "epsilon_start": [0.3],
        "epsilon_end": [0.01],
        "epsilon_decay": [0.99997],
        "update_epsilon": [False],
        "device": ["cuda" if torch.cuda.is_available() else "cpu"],
    }

    keys, values = zip(*grid.items())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def run_experiment(config):
    print(f"\n🚀 Starting experiment: {config}")

    start_time = time.time()
    model_path = train_rl_agent(
        total_steps=config["total_steps"],
        max_steps=config["max_steps"],
        reward_mode=config["reward_mode"],
        scramble_min=config["scramble_min"],
        scramble_max=config["scramble_max"],
        resets_per_jump=config["resets_per_jump"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        update_epsilon=config["update_epsilon"],
        device=config["device"],
        use_mlflow=True,
    )
    duration = time.time() - start_time

    return {
        **config,
        "model_path": model_path,
        "duration_sec": round(duration, 2),
    }


def main():
    os.makedirs("experiments", exist_ok=True)
    csv_path = os.path.join("experiments", "experiment_results.csv")

    experiments = list(generate_experiments())
    total_experiments = len(experiments)
    print(f"⚡ Total experiments to run: {total_experiments}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "reward_mode",
                "scramble_min",
                "scramble_max",
                "total_steps",
                "device",
                "duration_sec",
                "model_path",
            ],
        )
        writer.writeheader()

    for i, config in enumerate(experiments, start=1):
        print(f"\n🔹 Running experiment {i}/{total_experiments}")
        result = run_experiment(config)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            writer.writerow(result)

        print(f"✅ Done: {result['model_path']} (time: {result['duration_sec']}s)")

    print("\n🎯 All experiments completed.")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
