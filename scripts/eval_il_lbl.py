import torch
import json
import numpy as np
from envs.rubik2x2_env import Rubik2x2Env
from training.train_il import ILClassifier
from envs.render_utils import render_cube_ascii
from envs.lbl_solver import solve_bottom_layer, apply_moves
import random
import os

MODEL_PATH = "models/il_classifier.pth"
ALGO_FILE = "datasets/upper_layer_algorithms.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_real_case():
    with open(ALGO_FILE, "r") as f:
        algorithms = json.load(f)

    env = Rubik2x2Env()
    model = ILClassifier(input_dim=144, num_classes=len(algorithms))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Scramble cube
    scramble_len = random.randint(5, 15)
    env.cube.reset()
    scramble_moves = random.choices(list(apply_moves.__globals__["MOVE_MAP"].keys()), k=scramble_len)
    apply_moves(env.cube, scramble_moves)
    print(f"\nScramble ({scramble_len} moves): {' '.join(scramble_moves)}\n")
    print("Scrambled cube:")
    print(render_cube_ascii(env.cube.state))

    # Solve bottom layer
    lbl_moves = solve_bottom_layer(env.cube)
    print(f"\nAfter solving bottom (yellow) layer with moves: {' '.join(lbl_moves)}\n")
    print(render_cube_ascii(env.cube.state))

    # Extract observation and classify top layer
"""     obs = np.array(env.cube.state).flatten()
    x = np.eye(6, dtype=np.float32)[obs].flatten()
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).item()

    predicted_name = list(algorithms.keys())[pred]
    print(f"\nModel prediction for top (white) layer pattern: {predicted_name}") """

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    evaluate_real_case()
