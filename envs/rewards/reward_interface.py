import numpy as np
from .reward_helpers import (
    is_bottom_face_solved,
    is_bottom_layer_solved
)

def compute_reward(cube, solved, mode="basic"):
    """
    Compute reward for 2x2 Rubik's Cube.
    
    Modes:
    - "basic": 1 if fully solved, else 0
    - "bottom_face": reward for solving only the bottom face
    - "bottom_layer": reward for solving the entire bottom layer (face + adjacent stickers)
    - "sticker_plus_face": evaluates all stickers and fully uniform faces
    - "bottom_layer_corners": reward for each correctly positioned corner in the bottom layer
    """
    BONUS_FULL = 5.0  # large bonus for fully achieving the goal

    if mode == "basic":
        return 1.0 if solved else 0.0

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
        return reward

    elif mode == "sticker_plus_face":
        # reward proportional to solved stickers
        reward = sum(np.sum(face == i) for i, face in enumerate(cube.state)) / 24.0
        # bonus for fully uniform face layers
        reward += sum(0.25 for face in cube.state if len(set(face)) == 1)
        # add bonus if the entire cube is solved
        if solved:
            reward += BONUS_FULL
        return reward

    elif mode == "bottom_layer_corners":
        reward = 0.0
        corners = [
            (0, (2, 2), (5, 0)),
            (1, (5, 1), (3, 2)),
            (2, (3, 3), (4, 1)),
            (3, (4, 2), (2, 3)),
        ]

        for down_idx, (side1_id, side1_idx), (side2_id, side2_idx) in corners:
            if (
                cube.state[1][down_idx] == 1 and
                cube.state[side1_id][side1_idx] == side1_id and
                cube.state[side2_id][side2_idx] == side2_id
            ):
                reward += 0.25
        if reward >= 1.0:  # all 4 corners solved
            reward += BONUS_FULL

        # small penalty per move to encourage shorter solutions
        reward += -0.01

        return reward

    else:
        return 0.0
