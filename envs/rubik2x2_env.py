import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .cube_state import Cube2x2
from .rewards.reward_interface import compute_reward
from .render_utils import render_cube_ascii

class Rubik2x2Env(gym.Env):
    """
    PyTorch-friendly 2x2 Rubik’s Cube environment.

    - Observation: 24-element flattened array (6 faces × 4 stickers)
    - Action space: 12 discrete actions (6 faces × 2 directions)
    - Reward: depends on reward_mode (via compute_reward)
    - Supports gradual or weighted scramble growth
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        max_steps: int = 100,
        reward_mode: str = "basic",
        scramble_min: int = 1,
        scramble_max: int = 20,
        resets_per_jump: int = 5000,
        scramble_jump: int = 1,
        scramble_mode: str = "gradual",  # "gradual" or "weighted"
        render_mode: str | None = None,
        device: str | None = None,
    ):
        super().__init__()

        # Core environment settings
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.scramble_min = scramble_min
        self.scramble_max = scramble_max
        self.resets_per_jump = resets_per_jump
        self.scramble_jump = scramble_jump
        self.scramble_mode = scramble_mode
        self.render_mode = render_mode
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

        # State management
        self.cube = Cube2x2()
        self.current_step = 0
        self.total_resets = 0
        self.current_scramble = scramble_min
        self.level_resets_count = 0

        # Spaces
        self.action_space = spaces.Discrete(18)  # 6 faces × 3 directions
        self.observation_space = spaces.Box(
            low=0.0, high=5.0, shape=(24 * 6,), dtype=np.float32
        )

    # --------------------------------------------------------------
    # Core Gym API
    # --------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cube.reset()
        self.current_step = 0
        self.total_resets += 1

        # Scramble difficulty growth
        self._update_scramble_difficulty()

        # Scramble until unsolved
        while True:
            self.cube.reset()
            self.cube.scramble(self.current_scramble)
            if not self.cube.is_solved():
                break

        obs = self._get_obs()
        info = {"current_scramble": self.current_scramble}
        return obs, info

    def step(self, action: int):
        face_id = action % 6
        direction = action // 6  # 0=CW, 1=CCW, 2=180

        if direction == 0:
            self.cube.rotate_cw(face_id)
        elif direction == 1:
            self.cube.rotate_ccw(face_id)
        else:
            self.cube.rotate_180(face_id)

        self.current_step += 1
        solved = self.cube.is_solved()

        # Reward calculation
        reward = float(compute_reward(self.cube, solved, self.reward_mode))

        # Termination conditions
        terminated = solved
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = {
            "step": self.current_step,
            "solved": solved,
            "terminated": terminated,
            "truncated": truncated,
            "current_scramble": self.current_scramble,
        }

        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------

    def _get_obs(self):
        flat = self.cube.state.flatten().astype(int)
        one_hot = np.eye(6, dtype=np.float32)[flat]  # (24,6)
        return one_hot.flatten()  # (144,)

    def _update_scramble_difficulty(self):
        """Handles dynamic scramble progression."""
        if self.scramble_mode == "gradual":
            if self.total_resets % self.resets_per_jump == 0:
                self.current_scramble = min(
                    self.current_scramble + self.scramble_jump, self.scramble_max
                )

        elif self.scramble_mode == "weighted":
            if self.level_resets_count >= self.resets_per_jump * self.current_scramble:
                self.current_scramble = min(
                    self.current_scramble + 1, self.scramble_max
                )
                self.level_resets_count = 0
            self.level_resets_count += 1

    def render(self):
        text = render_cube_ascii(self.cube.state)
        if self.render_mode == "human":
            print(text)
        return text

    @staticmethod
    def _has_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
