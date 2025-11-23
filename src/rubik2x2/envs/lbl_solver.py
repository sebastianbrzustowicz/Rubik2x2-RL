from typing import List, Optional, Set, Tuple
from rubik2x2.envs.render_utils import render_cube_ascii

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

CORNERS_POS = {
    "UFR": [(0, 3), (2, 1), (5, 0)],
    "UFL": [(0, 2), (2, 0), (4, 1)],
    "UBL": [(0, 0), (3, 1), (4, 0)],
    "UBR": [(0, 1), (3, 0), (5, 1)],
    "DFR": [(1, 1), (2, 3), (5, 2)],
    "DFL": [(1, 0), (2, 2), (4, 3)],
    "DBL": [(1, 2), (3, 3), (4, 2)],
    "DBR": [(1, 3), (3, 2), (5, 3)],
}

ORDERED_CORNERS = ["UFR", "UFL", "UBL", "UBR", "DFR", "DFL", "DBL", "DBR"]

SEXY_MOVE = ["R", "U", "R'", "U'"]
SEXY_MIRROR = ["F'", "U'", "F", "U"]
TRIPLE_SEXY = SEXY_MOVE * 3


def apply_moves(cube, moves: List[str]):
    for m in moves:
        face, dirn = MOVE_MAP[m]
        if dirn == 0:
            cube.rotate_cw(face)
        elif dirn == 1:
            cube.rotate_ccw(face)
        else:
            cube.rotate_180(face)


def corner_colors_by_name(cube, name: str) -> Tuple[int, int, int]:
    a, b, c = CORNERS_POS[name]
    return (
        int(cube.state[a[0]][a[1]]),
        int(cube.state[b[0]][b[1]]),
        int(cube.state[c[0]][c[1]]),
    )


def locate_corner(cube, target_colors: Tuple[int, int, int]) -> Optional[str]:
    color_set = set(target_colors)
    for name in ORDERED_CORNERS:
        cs = set(corner_colors_by_name(cube, name))
        if cs == color_set:
            return name
    return None


def is_corner_on_lower_layer(corner_name: str) -> bool:
    return corner_name in ["DFR", "DFL", "DBL", "DBR"]


def setup_corner_on_bottom(cube, corner_name: str) -> List[str]:
    moves = []
    if not is_corner_on_lower_layer(corner_name):
        return moves
    D_SETUP = {
        "DFR": [],
        "DBR": ["D'"],
        "DBL": ["D2"],
        "DFL": ["D"],
    }
    corner_moves = D_SETUP.get(corner_name, [])
    apply_moves(cube, corner_moves)
    moves.extend(corner_moves)
    return moves


def undo_D_moves(cube, moves: List[str]) -> List[str]:
    undo_map = {"D": "D'", "D'": "D", "D2": "D2"}
    undo = [undo_map[m] for m in reversed(moves)]
    apply_moves(cube, undo)
    return undo


def choose_insertion(cube) -> List[str]:
    corner_colors_vals = corner_colors_by_name(cube, "UFR")
    ufr_pos = CORNERS_POS["UFR"]
    yellow_idx = None
    for i, (face, sticker) in enumerate(ufr_pos):
        if int(cube.state[face][sticker]) == 1:
            yellow_idx = i
            break
    moves = []
    if yellow_idx is None:
        return moves
    if yellow_idx == 0:
        apply_moves(cube, TRIPLE_SEXY)
        moves.extend(TRIPLE_SEXY)
    elif yellow_idx == 2:
        apply_moves(cube, SEXY_MOVE)
        moves.extend(SEXY_MOVE)
    elif yellow_idx == 1:
        apply_moves(cube, SEXY_MIRROR)
        moves.extend(SEXY_MIRROR)
    return moves


def rotate_U_to_UFR(cube, corner_name: str) -> List[str]:
    moves = []
    if corner_name == "UFL":
        apply_moves(cube, ["U'"])
        moves.append("U'")
    elif corner_name == "UBL":
        apply_moves(cube, ["U2"])
        moves.append("U2")
    elif corner_name == "UBR":
        apply_moves(cube, ["U"])
        moves.append("U")
    return moves


def solve_bottom_layer(cube, debug: bool = False) -> List[str]:
    moves = []
    bottom_corners = [
        (1, 2, 4),
        (1, 2, 5),
        (1, 3, 4),
        (1, 3, 5),
    ]
    for idx, colors in enumerate(bottom_corners):
        if debug:
            print(f"\n--- Processing corner {idx+1}: colors {colors} ---")
        corner_name = locate_corner(cube, colors)
        if debug:
            print(f"Located at: {corner_name}")
            print(render_cube_ascii(cube.state))

        if corner_name is None:
            if debug:
                print(f"[WARN] Corner {colors} not found. Skipping.")
            continue

        corner_moves = []

        setup_moves = setup_corner_on_bottom(cube, corner_name)
        corner_moves.extend(setup_moves)
        if debug and setup_moves:
            print(f"Setup moves (extract to U): {setup_moves}")
            print(render_cube_ascii(cube.state))

        if setup_moves or corner_name == "DFR":
            apply_moves(cube, SEXY_MOVE)
            corner_moves.extend(SEXY_MOVE)
            if debug:
                print(f"SEXY_MOVE applied: {SEXY_MOVE}")
                print(render_cube_ascii(cube.state))

        undo_moves = undo_D_moves(cube, setup_moves)
        corner_moves.extend(undo_moves)
        if debug and undo_moves:
            print(f"Undo setup moves (return D): {undo_moves}")
            print(render_cube_ascii(cube.state))

        if corner_name in ["UFL", "UBL", "UBR"]:
            u_moves = rotate_U_to_UFR(cube, corner_name)
            corner_moves.extend(u_moves)
            if debug and u_moves:
                print(f"Rotated U to bring corner to UFR: {u_moves}")
                print(render_cube_ascii(cube.state))

        target_corner_name = ["DFL", "DFR", "DBL", "DBR"][idx]
        setup_moves = setup_corner_on_bottom(cube, target_corner_name)
        corner_moves.extend(setup_moves)
        if debug and setup_moves:
            print(
                f"Setup moves for insertion (target D position {target_corner_name}): {setup_moves}"
            )
            print(render_cube_ascii(cube.state))

        insertion_moves = choose_insertion(cube)
        corner_moves.extend(insertion_moves)
        if debug and insertion_moves:
            print(f"Insertion moves: {insertion_moves}")
            print(render_cube_ascii(cube.state))

        undo_moves = undo_D_moves(cube, setup_moves)
        corner_moves.extend(undo_moves)
        if debug and undo_moves:
            print(f"Undo setup moves after insertion: {undo_moves}")
            print(render_cube_ascii(cube.state))

        moves.extend(corner_moves)
        if debug:
            print(f"Total moves for corner {colors}: {corner_moves}")

    return moves
