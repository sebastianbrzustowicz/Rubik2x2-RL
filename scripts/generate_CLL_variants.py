import json
from pathlib import Path
from envs.rubik2x2_env import Rubik2x2Env

# Input/output paths
input_path = Path("datasets/upper_layer_algorithms.json")
output_path = Path("datasets/upper_layer_algorithms_full.json")

# Load original algorithms
with open(input_path, "r", encoding="utf-8") as f:
    algorithms = json.load(f)

env = Rubik2x2Env()

def apply_moves(moves):
    """Return the cube state after applying a sequence of moves."""
    env.cube.reset()
    MOVE_MAP = {
        "U": (0, 0), "U'": (0, 1), "U2": (0, 2),
        "D": (1, 0), "D'": (1, 1), "D2": (1, 2),
        "F": (2, 0), "F'": (2, 1), "F2": (2, 2),
        "B": (3, 0), "B'": (3, 1), "B2": (3, 2),
        "L": (4, 0), "L'": (4, 1), "L2": (4, 2),
        "R": (5, 0), "R'": (5, 1), "R2": (5, 2),
    }
    for move in moves:
        face, direction = MOVE_MAP[move]
        if direction == 0:
            env.cube.rotate_cw(face)
        elif direction == 1:
            env.cube.rotate_ccw(face)
        else:
            env.cube.rotate_180(face)
    return tuple(env.cube.state.flatten())

def generate_mirrors(alg_name, moves):
    mirrors = {}
    mirror_mods = [("U", "U'"), ("U2", "U2"), ("U'", "U")]
    for i, (prefix, suffix) in enumerate(mirror_mods, 1):
        variant_name = f"{alg_name}_mirror{i}"
        mirrors[variant_name] = [prefix] + moves + [suffix]
    return mirrors

def generate_rotations(alg_name, moves):
    rotations = {}
    prefixes = ["U", "U'", "U2"]
    for i, prefix in enumerate(prefixes, 1):
        variant_name = f"{alg_name}_rot{i}"
        rotations[variant_name] = [prefix] + moves
    return rotations

# Step 1: track unique states
unique_states = {}
extended_algorithms = {}

# original algorithms
for name, moves in algorithms.items():
    state = apply_moves(moves)
    if state not in unique_states:
        unique_states[state] = name
        extended_algorithms[name] = moves

# mirrors
for name, moves in list(extended_algorithms.items()):
    for m_name, m_moves in generate_mirrors(name, moves).items():
        state = apply_moves(m_moves)
        if state not in unique_states:
            unique_states[state] = m_name
            extended_algorithms[m_name] = m_moves

# rotations
final_algorithms = dict(extended_algorithms)
for name, moves in list(extended_algorithms.items()):
    for r_name, r_moves in generate_rotations(name, moves).items():
        state = apply_moves(r_moves)
        if state not in unique_states:
            unique_states[state] = r_name
            final_algorithms[r_name] = r_moves

# Step 3: save file in one-line JSON format
lines = ["{"]
items = list(final_algorithms.items())
for i, (name, moves) in enumerate(items):
    moves_str = ", ".join(f'"{m}"' for m in moves)
    comma = "," if i < len(items) - 1 else ""
    lines.append(f'  "{name}": [{moves_str}]{comma}')
lines.append("}")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✅ Generated {len(final_algorithms)} unique algorithms (mirrors + rotations removed duplicates)")
print(f"📄 Saved to: {output_path}")
