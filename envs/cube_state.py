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
            0: ([2, 5, 3, 4], [[0,1],[0,1],[0,1],[0,1]]),  # Up: F, R, B, L
            1: ([2, 4, 3, 5], [[2,3],[2,3],[2,3],[2,3]]),  # Down: F, L, B, R
            2: ([0, 5, 1, 4], [[2,3],[0,2],[0,1],[1,3]]),  # Front: U, R, D, L
            3: ([0, 4, 1, 5], [[0,1],[0,2],[2,3],[1,3]]),  # Back: U, L, D, R
            4: ([0, 2, 1, 3], [[0,2],[0,2],[0,2],[3,1]]),  # Left: U, F, D, B
            5: ([0, 3, 1, 2], [[1,3],[0,2],[1,3],[3,1]]),  # Right: U, B, D, F
        }

    def reset(self):
        for i in range(6):
            self.state[i] = i

    def scramble(self, n=10, seed=None):
        """
        Randomly scrambles the cube in n moves.
        Ensures the first move does NOT rotate the upper (U) face,
        since the agent focuses only on the bottom layer.
        Now includes 180° turns.
        """
        if seed is not None:
            random.seed(seed)

        performed_moves = []
        last_face = None

        for move_idx in range(n):
            # If this is the first move — avoid the upper face (face 0)
            if move_idx == 0:
                possible_faces = [1, 2, 3, 4, 5]
            else:
                possible_faces = [0, 1, 2, 3, 4, 5]

            # Avoid repeating the same face twice in a row
            if last_face is not None and last_face in possible_faces:
                possible_faces.remove(last_face)

            face_id = random.choice(possible_faces)
            direction = random.choice(["CW", "CCW", "180"])  # now includes 180° turn

            if direction == "CW":
                self.rotate_cw(face_id)
            elif direction == "CCW":
                self.rotate_ccw(face_id)
            else:
                self.rotate_180(face_id)

            performed_moves.append((face_id, direction))
            last_face = face_id

        return performed_moves

    def rotate_cw(self, face_id):
        """Clockwise rotation of the given face, including neighboring sides."""
        # Rotate the face itself
        self.state[face_id] = self.state[face_id][[2, 0, 3, 1]]

        faces, indices = self.neighbors[face_id]
        temp = [self.state[f][idx].copy() for f, idx in zip(faces, indices)]

        # For U/D faces, rotate neighbors in the opposite direction
        direction = -1 if face_id in [0, 1] else 1

        for i in range(4):
            self.state[faces[(i + direction) % 4]][indices[(i + direction) % 4]] = temp[i]

    def rotate_ccw(self, face_id):
        """Counter-clockwise rotation of the given face, including neighboring sides."""
        # Rotate the face itself
        self.state[face_id] = self.state[face_id][[1, 3, 0, 2]]

        faces, indices = self.neighbors[face_id]
        temp = [self.state[f][idx].copy() for f, idx in zip(faces, indices)]

        # For U/D faces, rotate neighbors in the opposite direction
        direction = 1 if face_id in [0, 1] else -1

        for i in range(4):
            self.state[faces[(i + direction) % 4]][indices[(i + direction) % 4]] = temp[i]

    def rotate_180(self, face_id):
        """180-degree rotation of the given face."""
        # Two CW rotations = 180 degrees
        self.rotate_cw(face_id)
        self.rotate_cw(face_id)

    def is_solved(self, strict=True):
        """Check if the bottom layer is solved."""
        if not strict:
            return False

        bottom_face = self.state[1]
        front_face = self.state[2]

        bottom_correct = np.all(bottom_face == 1)
        front_bottom_correct = front_face[2] == 2 and front_face[3] == 2

        return bottom_correct and front_bottom_correct

    def flatten(self, normalize=True):
        flat = self.state.flatten()
        return flat / 5.0 if normalize else flat
