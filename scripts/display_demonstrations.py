import json
import os
from envs.render_utils import render_cube_ascii

DATASET_DIR = "datasets"
DEMO_FILE = os.path.join(DATASET_DIR, "demonstrations.json")
ALGO_FILE = os.path.join(DATASET_DIR, "upper_layer_algorithms_full.json")

with open(DEMO_FILE, "r") as f:
    demonstrations = json.load(f)

with open(ALGO_FILE, "r") as f:
    algorithms = json.load(f)

def invert_move(move: str) -> str:
    if move.endswith("2"):
        return move
    elif move.endswith("'"):
        return move[:-1]
    else:
        return move + "'"

for demo in demonstrations:
    pattern = demo["pattern"]
    state = demo["state"]
    algo = algorithms.get(pattern, [])
    reversed_algo = [invert_move(m) for m in reversed(algo)]

    print(f"\nPattern: {pattern}")
    print("Final state:")
    print(render_cube_ascii(state))
    print("Algorithm used (reversed):", " ".join(reversed_algo))
    input("Press Enter to continue...")
