COLOR_CODES = {
    0: "\033[97m",  # White (U)
    1: "\033[93m",  # Yellow (D)
    2: "\033[92m",  # Green (F)
    3: "\033[94m",  # Blue (B)
    4: "\033[38;5;208m",  # Orange (L)
    5: "\033[91m",  # Red (R)
}
RESET = "\033[0m"

COLOR_SYMBOLS = ["W", "Y", "G", "B", "O", "R"]


def render_cube_ascii(state, use_color=True):
    """
    Renders a 2x2 Rubik’s Cube as an ASCII net:
          U0 U1
          U2 U3
    L0 L1 F0 F1 R0 R1 B0 B1
    L2 L3 F2 F3 R2 R3 B2 B3
          D0 D1
          D2 D3
    """

    def c(face_id, idx):
        val = state[face_id][idx]
        if use_color:
            return f"{COLOR_CODES[val]}{COLOR_SYMBOLS[val]}{RESET}"
        return COLOR_SYMBOLS[val]

    lines = []
    lines.append("    " + c(0, 0) + " " + c(0, 1))
    lines.append("    " + c(0, 2) + " " + c(0, 3))
    lines.append(
        f"{c(4,0)} {c(4,1)} "
        f"{c(2,0)} {c(2,1)} "
        f"{c(5,0)} {c(5,1)} "
        f"{c(3,0)} {c(3,1)}"
    )
    lines.append(
        f"{c(4,2)} {c(4,3)} "
        f"{c(2,2)} {c(2,3)} "
        f"{c(5,2)} {c(5,3)} "
        f"{c(3,2)} {c(3,3)}"
    )
    lines.append("    " + c(1, 0) + " " + c(1, 1))
    lines.append("    " + c(1, 2) + " " + c(1, 3))

    return "\n".join(lines)


def render_cube_faces(state):
    """Alternative simple view – prints all cube faces."""
    colors = COLOR_SYMBOLS
    lines = []
    for i, face in enumerate(state):
        line = " ".join(colors[c] for c in face)
        lines.append(f"Face {i}: {line}")
    return "\n".join(lines)
