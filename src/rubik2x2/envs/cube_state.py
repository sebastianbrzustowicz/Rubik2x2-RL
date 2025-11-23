import numpy as np
import random
from typing import List

MOVE_MAP = {
    "U": (0, 0),
    "U'": (0, 1),
    "U2": (0, 2),
    "D": (1, 0),
    "D'": (1, 1),
    "D2": (1, 2),
    "F": (2, 0),
    "F'": (2, 1),
    "F2": (2, 2),
    "B": (3, 0),
    "B'": (3, 1),
    "B2": (3, 2),
    "L": (4, 0),
    "L'": (4, 1),
    "L2": (4, 2),
    "R": (5, 0),
    "R'": (5, 1),
    "R2": (5, 2),
}


class Cube2x2:
    """Logical representation of the 2x2 Rubik’s Cube."""

    def __init__(self):
        self.state = np.zeros((6, 4), dtype=np.int8)
        self.reset()

        self.solved_state = np.array(
            [
                [0, 0, 0, 0],  # Up (white)
                [1, 1, 1, 1],  # Down (yellow)
                [2, 2, 2, 2],  # Front (green)
                [3, 3, 3, 3],  # Back (blue)
                [4, 4, 4, 4],  # Left (orange)
                [5, 5, 5, 5],  # Right (red)
            ],
            dtype=np.int8,
        )

        # Definition of neighbors – for each side
        self.neighbors = {
            0: ([2, 5, 3, 4], [[0, 1], [0, 1], [0, 1], [0, 1]]),  # Up: F, R, B, L
            1: ([2, 4, 3, 5], [[2, 3], [2, 3], [2, 3], [2, 3]]),  # Down: F, L, B, R
            2: ([0, 5, 1, 4], [[2, 3], [0, 2], [1, 0], [3, 1]]),  # Front: U, R, D, L
            3: ([0, 4, 1, 5], [[0, 1], [2, 0], [3, 2], [1, 3]]),  # Back: U, L, D, R
            4: ([0, 2, 1, 3], [[0, 2], [0, 2], [0, 2], [3, 1]]),  # Left: U, F, D, B
            5: ([0, 3, 1, 2], [[1, 3], [2, 0], [1, 3], [1, 3]]),  # Right: U, B, D, F
        }

    def reset(self):
        for i in range(6):
            self.state[i] = i

    def scramble(self, n=10, seed=None, debug=False, max_attempts=20):
        if seed is not None:
            random.seed(seed)

        for attempt in range(max_attempts):
            original_state = self.state.copy()
            performed_moves = []
            last_face = None

            for move_idx in range(n):
                possible_faces = [0, 1, 2, 3, 4, 5]
                if last_face is not None and last_face in possible_faces:
                    possible_faces.remove(last_face)

                face_id = random.choice(possible_faces)
                direction = random.choice(["CW", "CCW", "180"])
                if direction == "CW":
                    self.rotate_cw(face_id)
                elif direction == "CCW":
                    self.rotate_ccw(face_id)
                else:
                    self.rotate_180(face_id)

                performed_moves.append((face_id, direction))
                last_face = face_id

                if debug:
                    from envs.render_utils import render_cube_ascii

                    print(f"Move {move_idx+1}/{n}: face={face_id}, dir={direction}")
                    print(render_cube_ascii(self.state))
                    print("-" * 30)

            if not self.is_solved():
                return performed_moves
            else:
                self.state = original_state.copy()

        print("[WARN] Scramble failed to change state after multiple attempts!")
        return []

    def rotate_cw(self, face_id):
        """Clockwise rotation of the given face, including neighboring sides."""
        self.state[face_id] = self.state[face_id][[2, 0, 3, 1]]

        faces, indices = self.neighbors[face_id]
        temp = [self.state[f][idx].copy() for f, idx in zip(faces, indices)]

        direction = -1 if face_id in [0, 1] else 1

        for i in range(4):
            self.state[faces[(i + direction) % 4]][indices[(i + direction) % 4]] = temp[
                i
            ]

    def rotate_ccw(self, face_id):
        """Counter-clockwise rotation of the given face, including neighboring sides."""
        self.state[face_id] = self.state[face_id][[1, 3, 0, 2]]

        faces, indices = self.neighbors[face_id]
        temp = [self.state[f][idx].copy() for f, idx in zip(faces, indices)]

        direction = 1 if face_id in [0, 1] else -1

        for i in range(4):
            self.state[faces[(i + direction) % 4]][indices[(i + direction) % 4]] = temp[
                i
            ]

    def rotate_180(self, face_id):
        """180-degree rotation of the given face."""
        self.rotate_cw(face_id)
        self.rotate_cw(face_id)

    def apply_moves(self, moves: List[str]):
        for m in moves:
            face, dirn = MOVE_MAP[m]
            if dirn == 0:
                self.rotate_cw(face)
            elif dirn == 1:
                self.rotate_ccw(face)
            else:
                self.rotate_180(face)

    def is_solved(self, strict=True):
        """Check if the bottom layer is solved."""
        if not strict:
            return False

        bottom_face = self.state[1]
        front_face = self.state[2]

        bottom_correct = np.all(bottom_face == 1)
        front_bottom_correct = front_face[2] == 2 and front_face[3] == 2

        return bottom_correct and front_bottom_correct

    def is_entire_cube_solved(self) -> bool:
        """Check if the entire cube is solved (all faces uniform)."""
        return all(np.all(self.state[face] == self.state[face][0]) for face in range(6))

    def flatten(self, normalize=True):
        flat = self.state.flatten()
        return flat / 5.0 if normalize else flat

    def copy(self):
        new_cube = Cube2x2()
        new_cube.state = [face.copy() for face in self.state]
        return new_cube
