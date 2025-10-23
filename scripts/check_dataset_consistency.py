import json
import numpy as np
from envs.rubik2x2_env import Rubik2x2Env

DATASET_PATH = "datasets/upper_layer_algorithms_full.json"

def main():
    with open(DATASET_PATH, "r") as f:
        algorithms = json.load(f)

    env = Rubik2x2Env()
    MOVE_MAP = {
        "U": (0, 0), "U'": (0, 1), "U2": (0, 2),
        "D": (1, 0), "D'": (1, 1), "D2": (1, 2),
        "F": (2, 0), "F'": (2, 1), "F2": (2, 2),
        "B": (3, 0), "B'": (3, 1), "B2": (3, 2),
        "L": (4, 0), "L'": (4, 1), "L2": (4, 2),
        "R": (5, 0), "R'": (5, 1), "R2": (5, 2),
    }

    state_map = {}  # key: flattened cube state tuple, value: set of labels

    for name, moves in algorithms.items():
        env.cube.reset()
        # Apply moves to cube
        for move in moves:
            face, direction = MOVE_MAP[move]
            if direction == 0:
                env.cube.rotate_cw(face)
            elif direction == 1:
                env.cube.rotate_ccw(face)
            else:
                env.cube.rotate_180(face)

        # Flatten cube state
        state_tuple = tuple(np.array(env.cube.state).flatten())

        if state_tuple not in state_map:
            state_map[state_tuple] = set()
        state_map[state_tuple].add(name)

    # Check for conflicts: same state, different labels
    conflicts = []
    for state, labels in state_map.items():
        if len(labels) > 1:
            conflicts.append((state, labels))

    if conflicts:
        print(f"Found {len(conflicts)} conflicting states!")
        for i, (state, labels) in enumerate(conflicts[:10]):
            print(f"\nConflict {i+1}:")
            print(f"Labels: {labels}")
            print(f"State: {state}")
        print("\nOther conflicts omitted...")
    else:
        print("No conflicts found. Dataset is consistent!")

if __name__ == "__main__":
    main()
