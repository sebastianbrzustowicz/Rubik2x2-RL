import numpy as np
from .reward_helpers import (
    is_bottom_face_solved,
    is_bottom_layer_solved
)

def compute_reward(cube, solved, action, prev_face_id=None, mode="basic", current_scramble=1, scramble_max=5, prev_correct_corners=None):
    BONUS_FULL = 25.0  # large bonus for fully achieving the goal
    face_id = action % 6
    direction = action // 6  # 0=CW, 1=CCW, 2=180

    # --- Exponential difficulty factor ---
    scramble_factor = np.exp(0.25 * current_scramble) / np.exp(0.25 * scramble_max)

    if mode == "bottom_layer_corners":
        reward = 0.0
        corners = [
            (0, (2, 2), (4, 3)),  # corner 0
            (1, (2, 3), (5, 2)),  # corner 1
            (2, (3, 3), (4, 2)),  # corner 2
            (3, (3, 2), (5, 3)),  # corner 3
        ]

        current_correct = set()
        for idx, (down_idx, (side1_id, side1_idx), (side2_id, side2_idx)) in enumerate(corners):
            if (
                cube.state[1][down_idx] == 1 and
                cube.state[side1_id][side1_idx] == side1_id and
                cube.state[side2_id][side2_idx] == side2_id
            ):
                reward += 0.1 * scramble_factor
                current_correct.add(idx)

        # --- Penalty for spoiling previously correct corners ---
        if prev_correct_corners is not None:
            lost_corners = prev_correct_corners - current_correct
            reward -= 0.3 * len(lost_corners)  # duża kara

        if solved:
            reward += BONUS_FULL * scramble_factor

        # --- Penalty depending on the length of the scramble ---
        reward -= 0.03 * (1 + 0.1 * current_scramble)

        # --- Penalty for U/D ---
        #if face_id in [0, 1]:
        #    reward -= 0.01

        # --- Penalty for repeating the same wall ---
        #if prev_face_id is not None and prev_face_id == face_id:
        #    reward -= 0.5

        # --- A small penalty for the length of the trajectory ---
        if reward > 0:
            reward *= 0.99
        else:
            reward *= 1.01

        return reward, current_correct

    elif mode == "basic":
        return 1.0 if solved else 0.0, None

    elif mode == "bottom_face":
        reward = 1.0 if is_bottom_face_solved(cube) else 0.0
        if reward == 1.0:
            reward += BONUS_FULL
        return reward

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
        return reward

    else:
        return 0.0
