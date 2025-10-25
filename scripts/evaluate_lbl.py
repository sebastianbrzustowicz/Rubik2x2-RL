import numpy as np
from envs.rubik2x2_env import Rubik2x2Env
from envs.lbl_solver import solve_bottom_layer, apply_moves

trials = 100_000
success_count = 0

for i in range(1, trials + 1):
    env = Rubik2x2Env()
    env.cube.reset()
    scramble_len = np.random.randint(5, 15)
    scramble_moves = np.random.choice(list(apply_moves.__globals__["MOVE_MAP"].keys()), size=scramble_len)
    apply_moves(env.cube, scramble_moves.tolist())

    solve_bottom_layer(env.cube)

    cube = env.cube
    bottom_ok = np.all(cube.state[1] == 1)
    side_ok = (
        cube.state[2][2] == 2 and cube.state[2][3] == 2 and
        cube.state[3][2] == 3 and cube.state[3][3] == 3 and
        cube.state[4][2] == 4 and cube.state[4][3] == 4 and
        cube.state[5][2] == 5 and cube.state[5][3] == 5
    )

    if bottom_ok and side_ok:
        success_count += 1

    if i % 1000 == 0:
        print(f"Iteration {i} completed.")

success_rate = success_count / trials
print(f"LBL success rate: {success_rate*100:.2f}% ({success_count}/{trials})")
