import numpy as np
from .reward_helpers import count_correct_bottom_corners

def compute_canonical_reward(cube):
    target = cube.solved_state
    correct_stickers = np.sum(cube.state == target)
    reward = correct_stickers / 24.0

    for i in range(6):
        if np.all(cube.state[i] == target[i]):
            reward += 0.15

    if np.all(cube.state[0] == target[0]): reward += 0.1
    if np.all(cube.state[1] == target[1]): reward += 0.1
    if np.array_equal(cube.state, target): reward += 2

    return min(reward, 2.0)

def compute_lbl_progressive_reward(cube, solved):
    target = cube.solved_state
    reward = 0.0

    correct_bottom_corners = count_correct_bottom_corners(cube, target)
    reward += correct_bottom_corners * 0.15
    if correct_bottom_corners == 4: reward += 0.15

    top_color = target[0, 0]
    reward += (np.sum(cube.state[0] == top_color) / 4.0) * 0.15

    if np.array_equal(cube.state, target): reward += 0.25
    return min(reward, 1.0)
