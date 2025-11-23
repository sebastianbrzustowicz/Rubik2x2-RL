import torch
import json
import numpy as np
from rubik2x2.envs.rubik2x2_env import Rubik2x2Env
from rubik2x2.training.train_il import ILClassifier
from rubik2x2.envs.render_utils import render_cube_ascii
from rubik2x2.envs.lbl_solver import solve_bottom_layer, apply_moves
import random
import os

MODEL_PATH = "models/il_classifier.pth"
ALGO_FILE = "datasets/upper_layer_algorithms_full.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRIALS = 1000


def evaluate_real_case(trials=TRIALS):
    with open(ALGO_FILE, "r") as f:
        algorithms = json.load(f)

    model = ILClassifier(input_dim=144, num_classes=len(algorithms))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    solved_count = 0
    success_rates = []

    for i in range(1, trials + 1):
        env = Rubik2x2Env()
        env.cube.reset()
        scramble_len = random.randint(5, 15)
        scramble_moves = random.choices(
            list(apply_moves.__globals__["MOVE_MAP"].keys()), k=scramble_len
        )
        apply_moves(env.cube, scramble_moves)

        lbl_moves = solve_bottom_layer(env.cube)

        obs = np.array(env.cube.state).flatten()
        x = np.eye(6, dtype=np.float32)[obs].flatten()
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1).item()

        predicted_name = list(algorithms.keys())[pred]
        predicted_moves = algorithms[predicted_name]

        apply_moves(env.cube, predicted_moves)

        solved = env.cube.is_entire_cube_solved()
        success_rates.append(solved)
        if solved:
            solved_count += 1
        else:
            print(f"\n--- Trial {i} FAILED ---")
            print(f"Scramble moves: {scramble_moves}")
            print("Cube state AFTER scramble:")
            print(render_cube_ascii(env.cube.state))
            print(f"LBL moves: {lbl_moves}")
            print("Cube state AFTER LBL (before IL):")
            print(render_cube_ascii(env.cube.state))
            print(f"Predicted algorithm: {predicted_name}")
            print(f"Predicted moves: {predicted_moves}")
            print("Cube state AFTER applying IL moves:")
            print(render_cube_ascii(env.cube.state))

        if i % 100 == 0:
            print(f"Trial {i}/{trials} completed")

    print(
        f"\nFull cube solved in {solved_count}/{trials} trials ({solved_count/trials*100:.2f}%)"
    )
    total_success_rate = np.mean(success_rates)
    print(f"Average success rate per trial: {total_success_rate*100:.2f}%")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    evaluate_real_case()
