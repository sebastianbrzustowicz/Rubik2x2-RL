import numpy as np
from .reward_helpers import is_bottom_face_solved, is_bottom_layer_solved


def compute_reward(
    cube,
    solved,
    action,
    prev_cube=None,
    prev_face_id=None,
    mode="basic",
    current_scramble=1,
    scramble_max=5,
    prev_correct_corners=None,
):
    BONUS_FULL = 50.0
    face_id = action % 6
    direction = action // 6

    scramble_factor = np.exp(0.25 * current_scramble) / np.exp(0.25 * scramble_max)

    penalty_scale = 1.0 + 0.5 * (current_scramble / scramble_max) ** 1.5

    reward = 0.0

    if mode == "bottom_layer_corners":
        corners = [
            (0, (2, 2), (4, 3)),
            (1, (2, 3), (5, 2)),
            (2, (3, 3), (4, 2)),
            (3, (3, 2), (5, 3)),
        ]

        current_correct = set()
        for idx, (down_idx, (side1_id, side1_idx), (side2_id, side2_idx)) in enumerate(
            corners
        ):
            if (
                cube.state[1][down_idx] == 1
                and cube.state[side1_id][side1_idx] == side1_id
                and cube.state[side2_id][side2_idx] == side2_id
            ):
                current_correct.add(idx)

        n_correct = len(current_correct)
        n_prev = len(prev_correct_corners or set())
        delta = n_correct - n_prev

        # --- Progress reward ---
        if delta > 0:
            reward += 1.0 * delta * scramble_factor
        # --- Penalty for losing correct corners ---
        elif delta < 0:
            reward -= 0.4 * abs(delta) * penalty_scale

        # --- Base reward proportional to current progress ---
        reward += 0.1 * n_correct * scramble_factor

        # --- Bonus for complete solution ---
        if solved:
            reward += BONUS_FULL * scramble_factor

        # --- Additional reward for a simple D move if the bottom layer was complete ---
        if prev_cube is not None:
            prev_bottom_complete = all(prev_cube.state[1][i] == 1 for i in range(4))
            curr_bottom_correct = all(cube.state[1][i] == 1 for i in range(4))
            # Additionally, we check whether the bottom stickers on the front side have the correct colors
            front_bottom_correct = cube.state[2][2] == 2 and cube.state[2][3] == 2
            if (
                prev_bottom_complete
                and curr_bottom_correct
                and front_bottom_correct
                and face_id == 1
            ):  # D
                reward += 0.5 * scramble_factor  # bonus for a simple move

        # --- Penalty for pointless D/U movements if nothing changes ---
        if face_id in [0, 1] and delta == 0:
            reward -= 0.02 * penalty_scale

        # --- Penalty for destroying correct corners by D/U ---
        if (
            prev_correct_corners
            and n_correct < len(prev_correct_corners)
            and face_id in [0, 1]
        ):
            reward -= 0.5 * penalty_scale

        # --- Penalty for stagnation ---
        if delta == 0 and not solved:
            reward -= 0.01 * penalty_scale * (1 + 0.1 * current_scramble)

        return reward, current_correct

    elif mode == "basic":
        return 1.0 if solved else 0.0, None

    elif mode == "bottom_face":
        reward = 1.0 if is_bottom_face_solved(cube) else 0.0
        if reward == 1.0:
            reward += BONUS_FULL
        return reward, None

    elif mode == "bottom_layer":
        reward = 0.0
        if is_bottom_face_solved(cube):
            reward += 0.5
        if is_bottom_layer_solved(cube):
            reward += 0.5
        if reward >= 1.0:  # full completion of the bottom layer
            reward += BONUS_FULL
        return reward, None

    elif mode == "sticker_plus_face":
        # reward proportional to solved stickers
        reward = sum(np.sum(face == i) for i, face in enumerate(cube.state)) / 24.0
        # bonus for fully uniform face layers
        reward += sum(0.25 for face in cube.state if len(set(face)) == 1)
        # add bonus if the entire cube is solved
        if solved:
            reward += BONUS_FULL
        return reward, None

    else:
        return 0.0, None
