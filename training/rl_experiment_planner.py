import os
import csv
import itertools
import time
from training.train_rl import train_rl_agent
import torch

def generate_experiments():
    """Creates a list of all parameter combinations."""
    grid = {
        "reward_mode": ["bottom_layer_corners", "bottom_layer"],
        "scramble_min": [1],
        "scramble_max": [12],
        "resets_per_jump": [100000],
        "total_steps": [1000000],
        "max_steps": [10],
        "batch_size": [64], # 128 for more stable results
        "lr": [1e-4], # To test
        "gamma": [0.99], # To test
        "epsilon_start": [1.0],
        "epsilon_end": [0.05, 0.15, 0.05], # 0.25, 0.05 works good
        "epsilon_decay": [0.99995, 0.999995], # 0.99995, 0.999995 works good
        "device": ["cuda" if torch.cuda.is_available() else "cpu"],
    }

    keys, values = zip(*grid.items())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def run_experiment(config):
    """Wrapper to run a single experiment."""
    print(f"\n🚀 Starting experiment: {config}")

    start_time = time.time()
    model_path = train_rl_agent(
        total_steps=config["total_steps"],
        max_steps=config["max_steps"],
        reward_mode=config["reward_mode"],
        scramble_min=config["scramble_min"],
        resets_per_jump=config["resets_per_jump"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        device=config["device"],
        use_mlflow=True
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

    # Count total experiments
    experiments = list(generate_experiments())
    total_experiments = len(experiments)
    print(f"⚡ Total experiments to run: {total_experiments}")

    # CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "reward_mode", "scramble_min", "scramble_max",
            "total_steps", "device", "duration_sec", "model_path"
        ])
        writer.writeheader()

    for i, config in enumerate(experiments, start=1):
        print(f"\n🔹 Running experiment {i}/{total_experiments}")
        result = run_experiment(config)

        # Save results to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            writer.writerow(result)

        print(f"✅ Done: {result['model_path']} (time: {result['duration_sec']}s)")

    print("\n🎯 All experiments completed.")
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
