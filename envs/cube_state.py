import numpy as np
import random

class Cube2x2:
    """Logical representation of the 2x2 Rubik’s Cube."""

    def __init__(self):
        self.state = np.zeros((6, 4), dtype=np.int8)
        self.reset()

        self.solved_state = np.array([
            [0, 0, 0, 0],  # Up (white)
            [1, 1, 1, 1],  # Down (yellow)
            [2, 2, 2, 2],  # Front (green)
            [3, 3, 3, 3],  # Back (blue)
            [4, 4, 4, 4],  # Left (orange)
            [5, 5, 5, 5],  # Right (red)
        ], dtype=np.int8)

        # Definition of neighbors – for each side
        self.neighbors = {
            0: ([2, 5, 3, 4], [[0,1],[0,1],[0,1],[0,1]]),  # Up: F, R, B, L (upper layer)
            1: ([2, 4, 3, 5], [[2,3],[2,3],[2,3],[2,3]]),  # Down: F, L, B, R (bottom layer)
            2: ([0, 5, 1, 4], [[2,3],[0,2],[0,1],[1,3]]),  # Front: U, R, D, L
            3: ([0, 4, 1, 5], [[0,1],[0,2],[2,3],[1,3]]),  # Back: U, L, D, R
            4: ([0, 2, 1, 3], [[0,2],[0,2],[0,2],[3,1]]),  # Left: U, F, D, B
            5: ([0, 3, 1, 2], [[1,3],[0,2],[1,3],[3,1]]),  # Right: U, B, D, F
        }

    def reset(self):
        for i in range(6):
            self.state[i] = i

    def scramble(self, n=10, seed=None):
        """Randomly shuffles the cube in n moves and returns a list of the moves performed."""
        if seed is not None:
            random.seed(seed)
        performed_moves = []
        for _ in range(n):
            face_id = random.randint(0, 5)
            direction = random.choice(["CW", "CCW"])
            if direction == "CW":
                self.rotate_cw(face_id)
            else:
                self.rotate_ccw(face_id)
            performed_moves.append((face_id, direction))
        return performed_moves

    def rotate_cw(self, face_id):
        """Clockwise rotation of the given face, including neighboring sides."""
        # Rotation of the side itself (positions in 2x2)
        self.state[face_id] = self.state[face_id][[2, 0, 3, 1]]

        # Changing colors on neighboring sides
        faces, idxs = self.neighbors[face_id]
        temp = [self.state[f][idx].copy() for f, idx in zip(faces, idxs)]

        # For U and D, we rotate the neighbors in the opposite direction
        direction = -1 if face_id in [0, 1] else 1

        for i in range(4):
            self.state[faces[(i + direction) % 4]][idxs[(i + direction) % 4]] = temp[i]

    def rotate_ccw(self, face_id):
        """Counter-clockwise rotation of the given face, including neighboring sides."""
        # Rotation of the side itself (in the opposite direction)
        self.state[face_id] = self.state[face_id][[1, 3, 0, 2]]

        # Changing colors on neighboring sides
        faces, idxs = self.neighbors[face_id]
        temp = [self.state[f][idx].copy() for f, idx in zip(faces, idxs)]

        # For U and D, we rotate the neighbors in the opposite direction
        direction = 1 if face_id in [0, 1] else -1

        for i in range(4):
            self.state[faces[(i + direction) % 4]][idxs[(i + direction) % 4]] = temp[i]

    def is_solved(self, strict=True):
        """Check if cube is solved.
        - strict=True: must match canonical solved_state
        - strict=False: all faces uniform (any orientation)
        """
        if strict:
            return np.array_equal(self.state, self.solved_state)
        else:
            return all(len(set(face)) == 1 for face in self.state)

    def flatten(self, normalize=True):
        flat = self.state.flatten()
        return flat / 5.0 if normalize else flat
