from envs.rubik2x2_env import Rubik2x2Env
from agents.dqn_agent import DQNAgent
from envs.render_utils import render_cube_ascii
import random


def evaluate_model(
    model_path="models/rl_agent.pth",
    scramble_min=1,
    scramble_max=5,
    max_steps=20,
    device="cuda",
):
    env = Rubik2x2Env(
        max_steps=max_steps,
        reward_mode="bottom_layer_corners",
        scramble_min=scramble_min,
        scramble_max=scramble_max,
        scramble_mode="gradual",
    )

    agent = DQNAgent(env, device=device)
    agent.load(model_path)
    agent.epsilon = 0.0

    print("\n=== EVALUATION START ===\n")

    total_scrambles = 0
    solved_scrambles = 0

    for scramble_len in range(scramble_min, scramble_max + 1):
        print(f"\n--- SCRAMBLE LENGTH: {scramble_len} ---")

        env.current_step = 0
        env.prev_face_id = None
        env.prev_correct_corners = set()
        env.cube.reset()

        scramble_moves = []
        while True:
            scramble_moves = env.cube.scramble(scramble_len, seed=random.randint(0, 10000))
            if not env.cube.is_solved():
                break

        print("\nScramble moves:")
        print(" → ".join([f"{face_dir(face, dirn)}" for face, dirn in scramble_moves]))

        print("\nInitial scrambled cube:")
        print(render_cube_ascii(env.cube.state))

        obs = env._get_obs()
        done = False
        step_count = 0
        move_seq = []

        while not done and step_count < max_steps:
            prev_eps = agent.epsilon
            agent.epsilon = 0.0
            action = agent.select_action(obs)
            agent.epsilon = prev_eps

            move_seq.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            print(f"\nStep {step_count}: {action_to_str(action)}")
            print(render_cube_ascii(env.cube.state))

            done = terminated or truncated

            if terminated:
                print("\n✅ SOLVED!")
                solved_scrambles += 1
                break

        if not terminated:
            print("\n❌ NOT SOLVED within max steps")

        total_scrambles += 1
        print(f"\nAgent moves ({len(move_seq)}): {' → '.join(action_to_str(a) for a in move_seq)}")
        print("=" * 60)

    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Total scrambles: {total_scrambles}")
    print(f"Solved scrambles: {solved_scrambles} ({solved_scrambles / total_scrambles * 100:.1f}%)")


def face_dir(face_id, direction):
    """Replace (face_id, direction) with text"""
    faces = ["U", "D", "F", "B", "L", "R"]
    dirs = ["CW", "CCW", "180"]
    return f"{faces[face_id]}-{dirs_map(direction)}"


def dirs_map(direction):
    if isinstance(direction, str):
        return direction
    if direction == 0:
        return "CW"
    elif direction == 1:
        return "CCW"
    elif direction == 2:
        return "180"
    return "?"


def action_to_str(action):
    """Replaces action number with a symbolic notation, e.g., ‘F_CW’."""
    face = action % 6
    direction = action // 6
    faces = ["U", "D", "F", "B", "L", "R"]
    dirs = ["CW", "CCW", "180"]
    return f"{faces[face]}_{dirs[direction]}"


if __name__ == "__main__":
    evaluate_model(
        model_path="models/rl_agent.pth",
        scramble_min=1,
        scramble_max=10,
        max_steps=35,
        device="cuda",
    )
