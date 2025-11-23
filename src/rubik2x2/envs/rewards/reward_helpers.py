import numpy as np


def is_bottom_face_solved(cube):
    return len(set(cube.state[1])) == 1


def is_bottom_layer_solved(cube):
    return is_bottom_face_solved(cube) and cube.state[2][2] == cube.state[2][3]


def count_correct_bottom_corners(cube, target):
    correct = 0
    for face_id in range(6):
        if np.all(cube.state[face_id] == target[face_id]):
            correct += 1
    return min(correct, 4)
