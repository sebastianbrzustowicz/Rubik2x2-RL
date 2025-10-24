import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .cube_state import Cube2x2
from .rewards.reward_interface import compute_reward
from .render_utils import render_cube_ascii
import random 
import copy

class Rubik2x2Env(gym.Env):

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        max_steps: int = 100,
        reward_mode: str = "basic",
        scramble_min: int = 1,
        scramble_max: int = 20,
        resets_per_jump: int = 5000,
        scramble_jump: int = 1,
        scramble_mode: str = "gradual",
        render_mode: str | None = None,
        device: str | None = None,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.scramble_min = scramble_min
        self.scramble_max = scramble_max
        self.resets_per_jump = resets_per_jump
        self.scramble_jump = scramble_jump
        self.scramble_mode = scramble_mode
        self.render_mode = render_mode
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

        self.cube = Cube2x2()
        self.current_step = 0
        self.total_resets = 0
        self.current_scramble = scramble_min
        self.level_resets_count = 0

        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(
            low=0.0, high=5.0, shape=(24 * 6,), dtype=np.float32
        )

        self.level_success_history = []
        self.adaptive_scramble_threshold = 0.7
        self.adaptive_scramble_window = 10000
        self.easy_scramble_prob = 0.1

        self.prev_face_id = None
        self.prev_correct_corners = set()


    def reset(self, *, seed=None, options=None, last_solved=None, last_scramble=None):
        super().reset(seed=seed)
        self.cube.reset()
        self.current_step = 0
        self.total_resets += 1
        self.prev_face_id = None

        self.prev_cube = copy.deepcopy(self.cube)

        if last_solved is not None and last_scramble == self.current_scramble:
            self.level_success_history.append(int(last_solved))
            if len(self.level_success_history) > self.adaptive_scramble_window:
                self.level_success_history.pop(0)
            self._update_scramble_difficulty()

        choices = [self.current_scramble - 1, self.current_scramble, self.current_scramble + 1]
        weights = [0.25, 0.5, 0.25]
        scramble_length = int(np.clip(random.choices(choices, weights=weights)[0], self.scramble_min, self.scramble_max))

        while True:
            self.cube.reset()
            self.cube.scramble(scramble_length)
            if not self.cube.is_solved():
                break

        obs = self._get_obs()
        info = {"current_scramble": scramble_length}
        return obs, info

    def step(self, action: int):
        face_id = action % 6
        direction = action // 6  # 0=CW, 1=CCW, 2=180

        if self.prev_face_id is not None and self.prev_face_id == face_id:
            possible_faces = [i for i in range(6) if i != self.prev_face_id]
            new_face_id = random.choice(possible_faces)
            face_id = new_face_id
            action = face_id + direction * 6

        self.prev_cube = self.cube.copy()

        if direction == 0:
            self.cube.rotate_cw(face_id)
        elif direction == 1:
            self.cube.rotate_ccw(face_id)
        else:
            self.cube.rotate_180(face_id)

        self.current_step += 1
        solved = self.cube.is_solved()

        reward, current_correct = compute_reward(
            cube=self.cube,
            solved=solved,
            action=action,
            prev_cube=self.prev_cube,
            prev_face_id=self.prev_face_id,
            mode=self.reward_mode,
            current_scramble=self.current_scramble,
            scramble_max=self.scramble_max,
            prev_correct_corners=self.prev_correct_corners
        )

        self.prev_face_id = face_id
        self.prev_correct_corners = current_correct

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

    def _get_obs(self):
        flat = np.array(self.cube.state).flatten().astype(int)
        one_hot = np.eye(6, dtype=np.float32)[flat]
        return one_hot.flatten()

    def _update_scramble_difficulty(self):
        if len(self.level_success_history) >= self.adaptive_scramble_window:
            avg_success = np.mean(self.level_success_history[-self.adaptive_scramble_window:])
            if avg_success >= self.adaptive_scramble_threshold:
                self.current_scramble = min(self.current_scramble + 1, self.scramble_max)
                self.level_success_history = []

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
