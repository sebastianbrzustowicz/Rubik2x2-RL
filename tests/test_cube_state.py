import numpy as np

from rubik2x2.envs.cube_state import Cube2x2


def test_reset_sets_correct_state():
    cube = Cube2x2()
    cube.reset()
    for face_id in range(6):
        assert np.all(cube.state[face_id] == face_id)


def test_is_solved_and_entire_cube_solved():
    cube = Cube2x2()
    cube.reset()

    assert cube.is_solved()
    assert cube.is_entire_cube_solved()

    cube.rotate_cw(0)
    assert cube.is_solved()
    assert not cube.is_entire_cube_solved()


def test_rotate_operations_reversibility():
    cube = Cube2x2()
    original = cube.copy()
    cube.rotate_cw(2)
    cube.rotate_ccw(2)
    assert np.array_equal(cube.state, original.state)

    cube.rotate_180(3)
    cube.rotate_180(3)
    assert np.array_equal(cube.state, original.state)


def test_scramble_changes_state():
    cube = Cube2x2()
    original = cube.copy()
    moves = cube.scramble(n=5, seed=42)
    assert not np.array_equal(cube.state, original.state)
    assert len(moves) > 0


def test_flatten_normalization():
    cube = Cube2x2()
    flat_norm = cube.flatten(normalize=True)
    assert flat_norm.shape[0] == 24
    assert np.all((flat_norm >= 0.0) & (flat_norm <= 1.0))

    flat_unnorm = cube.flatten(normalize=False)
    assert flat_unnorm.shape[0] == 24
    assert np.max(flat_unnorm) == 5


def test_apply_moves_consistency():
    cube1 = Cube2x2()
    cube2 = Cube2x2()
    moves = ["U", "R'", "F2"]
    cube1.apply_moves(moves)
    for m in moves:
        cube2.apply_moves([m])
    assert np.array_equal(cube1.state, cube2.state)
