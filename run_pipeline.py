import argparse
import json
import torch
import numpy as np
from envs.rubik2x2_env import Rubik2x2Env
from agents.dqn_agent import DQNAgent
from training.train_il import ILClassifier
from envs.render_utils import render_cube_ascii
from envs.lbl_solver import solve_bottom_layer, apply_moves as apply_moves_fn

def run_pipeline(
    rl_model_path="models/rl_agent.pth",
    il_model_path="models/il_classifier.pth",
    algo_file="datasets/upper_layer_algorithms_full.json",
    scramble_str=None,
    max_steps=100,
    device="cuda",
    debug=False
):
    if scramble_str is None:
        raise ValueError("You must provide a scramble via --scramble")

    env = Rubik2x2Env(
        max_steps=max_steps,
        reward_mode="bottom_layer_corners",
        scramble_min=1,
        scramble_max=1,
        scramble_mode="manual",
    )

    rl_agent = DQNAgent(env, device=device)
    rl_agent.load(rl_model_path)
    rl_agent.epsilon = 0.0

    with open(algo_file, "r") as f:
        algorithms = json.load(f)
    il_model = ILClassifier(input_dim=144, num_classes=len(algorithms))
    il_model.load_state_dict(torch.load(il_model_path, map_location=device))
    il_model.to(device)
    il_model.eval()

    env.cube.reset()
    scramble_moves = parse_scramble(scramble_str)
    for face_id, direction in scramble_moves:
        if direction == 0:
            env.cube.rotate_cw(face_id)
        elif direction == 1:
            env.cube.rotate_ccw(face_id)
        elif direction == 2:
            env.cube.rotate_180(face_id)

    cube_state_scramble = np.copy(env.cube.state)

    obs = env._get_obs()
    done = False
    step_count = 0
    rl_moves = []

    if debug:
        print("\n--- Running RL moves ---")
    while not done and step_count < max_steps:
        action = rl_agent.select_action(obs)
        obs, reward, terminated, truncated, info, action = env.step(action, return_applied_action=True)
        rl_moves.append(action)
        step_count += 1
        done = terminated or truncated

    cube_state_rl = np.copy(env.cube.state)

    cube_state = np.array(env.cube.state).flatten()
    x = np.eye(6, dtype=np.float32)[cube_state].flatten()
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = il_model(x)
        pred_idx = logits.argmax(1).item()

    alg_name = list(algorithms.keys())[pred_idx]
    alg_moves = algorithms[alg_name]
    apply_moves_fn(env.cube, alg_moves)

    if not env.cube.is_entire_cube_solved():
        raise RuntimeError(f"IL model failed to solve cube with predicted algorithm '{alg_name}'")

    cube_state_il = np.copy(env.cube.state)
    il_actions = parse_moves_to_actions(alg_moves)

    if debug:
        print("\n=== RUN PIPELINE ===\n")
        print(f"Scramble moves: {scramble_str}")
        print("\n--- Cube state after scramble ---")
        print(render_cube_ascii(cube_state_scramble))

        print("\n--- Cube state after RL model ---")
        print(render_cube_ascii(cube_state_rl))

        print("\n--- Cube state after IL model ---")
        print(render_cube_ascii(cube_state_il))
        print("\nRL moves:", [action_to_notation(a) for a in rl_moves])
        print("IL moves:", alg_moves)
    else:
        rl_notation = [action_to_notation(a) for a in rl_moves]
        print(f"\nFull solution ({len(rl_notation)+len(alg_moves)} moves): {' '.join(rl_notation + alg_moves)}")

    return rl_moves, alg_moves, cube_state_rl, cube_state_il


def parse_scramble(scramble_str):
    faces = {"U": 0, "D": 1, "F": 2, "B": 3, "L": 4, "R": 5}
    dirs = {"": 0, "'": 1, "2": 2}
    tokens = scramble_str.split()
    moves = []
    for tok in tokens:
        face = tok[0].upper()
        suffix = tok[1:] if len(tok) > 1 else ""
        if face not in faces or suffix not in dirs:
            raise ValueError(f"Invalid move in scramble: {tok}")
        moves.append((faces[face], dirs[suffix]))
    return moves


def parse_scramble_to_actions(scramble_moves):
    actions = []
    for face, direction in scramble_moves:
        actions.append(face + 6 * direction)
    return actions


def parse_moves_to_actions(move_list):
    faces = {"U":0,"D":1,"F":2,"B":3,"L":4,"R":5}
    dirs = {"":0,"'":1,"2":2}
    actions = []
    for m in move_list:
        face = faces[m[0]]
        direction = dirs[m[1:]] if len(m) > 1 else 0
        actions.append(face + 6*direction)
    return actions


def action_to_notation(action):
    faces = ["U", "D", "F", "B", "L", "R"]
    dirs = ["", "'", "2"]
    face = action % 6
    direction = action // 6
    return f"{faces[face]}{dirs[direction]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL + IL Rubik2x2 solver pipeline")
    parser.add_argument("--scramble", type=str, required=True)
    parser.add_argument("--rl_model_path", type=str, default="models/rl_agent.pth")
    parser.add_argument("--il_model_path", type=str, default="models/il_classifier.pth")
    parser.add_argument("--algo_file", type=str, default="datasets/upper_layer_algorithms_full.json")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_pipeline(
        rl_model_path=args.rl_model_path,
        il_model_path=args.il_model_path,
        algo_file=args.algo_file,
        scramble_str=args.scramble,
        max_steps=args.max_steps,
        device=args.device,
        debug=False
    )
