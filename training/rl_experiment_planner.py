import os
import csv
import itertools
import time
from training.train_rl import train_rl_agent
import torch

def generate_experiments():
    """Creates a list of all parameter combinations."""
    grid = {
        "reward_mode": ["basic", "sticker_plus_face", "lbl_progressive"],
        "scramble_min": [1, 3],
        "scramble_max": [5, 8],
        "total_steps": [5_000, 10_000],
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
        reward_mode=config["reward_mode"],
        scramble_min=config["scramble_min"],
        scramble_max=config["scramble_max"],
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

    # CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "reward_mode", "scramble_min", "scramble_max",
            "total_steps", "device", "duration_sec", "model_path"
        ])
        writer.writeheader()

    for config in generate_experiments():
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
