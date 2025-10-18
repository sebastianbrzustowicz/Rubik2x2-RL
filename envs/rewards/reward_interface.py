import numpy as np

from .reward_modes import (
    compute_canonical_reward,
    compute_lbl_progressive_reward
)
from .reward_helpers import (
    is_bottom_face_solved,
    is_top_face_solved,
    is_bottom_layer_solved
)

def compute_reward(cube, solved, mode="basic"):
    if mode == "basic":
        return 1.0 if solved else 0.0

    elif mode == "shaped":
        reward = 0.0
        if is_bottom_face_solved(cube): reward += 0.25
        if is_bottom_layer_solved(cube): reward += 0.25
        if is_top_face_solved(cube): reward += 0.25
        if solved: reward += 0.25
        return reward

    elif mode == "sticker":
        return sum(np.sum(face == i) for i, face in enumerate(cube.state)) / 24.0

    elif mode == "sticker_plus_face":
        reward = sum(np.sum(face == i) for i, face in enumerate(cube.state)) / 24.0
        reward += sum(0.25 for face in cube.state if len(set(face)) == 1)
        return reward

    elif mode == "canonical_sticker_plus_face":
        return compute_canonical_reward(cube)

    elif mode == "lbl_progressive":
        return compute_lbl_progressive_reward(cube, solved)

    else:
        return 0.0
