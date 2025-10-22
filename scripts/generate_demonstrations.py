import json
import os
import copy
from envs.rubik2x2_env import Rubik2x2Env

DATASET_DIR = "datasets"
ALGO_FILE = os.path.join(DATASET_DIR, "upper_layer_algorithms.json")
OUTPUT_FILE = os.path.join(DATASET_DIR, "demonstrations.json")

os.makedirs(DATASET_DIR, exist_ok=True)

# Load algorithms
with open(ALGO_FILE, "r") as f:
    algorithms = json.load(f)

env = Rubik2x2Env()

# Map cube notation to (face_id, direction)
MOVE_MAP = {
    "U": (0, 0), "U'": (0, 1), "U2": (0, 2),
    "D": (1, 0), "D'": (1, 1), "D2": (1, 2),
    "F": (2, 0), "F'": (2, 1), "F2": (2, 2),
    "B": (3, 0), "B'": (3, 1), "B2": (3, 2),
    "L": (4, 0), "L'": (4, 1), "L2": (4, 2),
    "R": (5, 0), "R'": (5, 1), "R2": (5, 2),
}

# Function to invert moves
def invert_move(move: str) -> str:
    if move.endswith("2"):
        return move  # 180 stays the same
    elif move.endswith("'"):
        return move[:-1]  # remove prime
    else:
        return move + "'"  # add prime

demonstrations = []

for algo_id, (pattern_name, moves) in enumerate(algorithms.items()):
    env.cube.reset()

    # Reverse and invert algorithm
    reversed_moves = [invert_move(m) for m in reversed(moves)]

    for move in reversed_moves:
        if move not in MOVE_MAP:
            raise ValueError(f"Unknown move: {move}")
        face_id, direction = MOVE_MAP[move]

        if direction == 0:
            env.cube.rotate_cw(face_id)
        elif direction == 1:
            env.cube.rotate_ccw(face_id)
        else:
            env.cube.rotate_180(face_id)

    # Save cube state
    state = copy.deepcopy(env.cube.state).tolist()

    demonstrations.append({
        "pattern": pattern_name,
        "algo_id": algo_id,
        "state": state,
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(demonstrations, f, indent=2)

print(f"Generated {len(demonstrations)} demonstrations at {OUTPUT_FILE}.")
