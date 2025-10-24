import mlflow
import numpy as np
import torch.nn as nn
import torch
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer
from collections import deque
import os

class DQNAgent:
    def __init__(
        self,
        env,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        batch_size=64,
        target_update_freq=1000,
        device="cuda",
        use_mlflow=False,
        model_path="models/rl_agent.pth",
        update_epsilon=False
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.model_path = model_path

        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_epsilon = update_epsilon

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_mlflow = use_mlflow

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        self.q_net = QNetwork(obs_dim, 256, act_dim).to(device)
        self.target_net = QNetwork(obs_dim, 256, act_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

        self.global_step = 0
        self.success_count = 0
        self.fail_count = 0
        self.episode_rewards = []
        self.last_window_results = deque(maxlen=1000)

        self.prev_face_id = None

    def select_action(self, state, explore=True):
        """
        Select action with hard rule: cannot choose an action that affects the same face
        as the previous action (self.prev_face_id). If explore=False, choose greedy (no
        epsilon random), otherwise use epsilon-greedy but sample only from allowed actions.
        """
        num_actions = self.env.action_space.n
        all_actions = np.arange(num_actions)

        if self.prev_face_id is None:
            allowed_mask = np.ones(num_actions, dtype=bool)
        else:
            forbidden_faces = self.prev_face_id
            allowed_mask = (all_actions % 6) != forbidden_faces

        do_random = explore and (np.random.rand() < self.epsilon)
        if do_random:
            allowed_actions = all_actions[allowed_mask]
            if allowed_actions.size == 0:
                allowed_actions = all_actions
            return int(np.random.choice(allowed_actions))

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t).cpu().numpy().ravel()

        if allowed_mask.sum() == 0:
            return int(int(np.argmax(q_values)))

        masked_q = q_values.copy()
        masked_q[~allowed_mask] = -1e9
        action = int(int(np.argmax(masked_q)))
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, total_steps=100_000, log_every=10000, update_freq=4):
        state, _ = self.env.reset()
        episode_reward = 0
        self.prev_scramble_length = self.env.current_scramble
        best_success_rate = 0.0 
        best_scramble_len = 0
        episodes_in_current_scramble_len = 0
        step = 0

        while True:
            self.global_step = step
            step += 1

            action = self.select_action(state, explore=True)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.push(state, action, reward, next_state, done)

            self.prev_face_id = (action % 6)

            state = next_state
            episode_reward += reward

            if done:
                last_solved = terminated
                last_scramble = self.env.current_scramble
                state, _ = self.env.reset(last_solved=last_solved, last_scramble=last_scramble)
                episodes_in_current_scramble_len += 1

                self.prev_face_id = None

                current_scramble = self.env.current_scramble
                if current_scramble > self.prev_scramble_length:
                    episodes_in_current_scramble_len = 0
                    if self.update_epsilon:
                        min_scramble = self.env.scramble_min
                        max_scramble = self.env.scramble_max
                        scramble_ratio = (current_scramble - min_scramble) / max(1, max_scramble - min_scramble)

                        base_decay = self.epsilon_decay
                        scale = 1.05

                        self.epsilon_decay = base_decay + (1.0 - base_decay) * (scramble_ratio ** scale)

                        self.epsilon_decay = min(self.epsilon_decay, 0.9999982)
                        self.epsilon = max(0.2, self.epsilon_start)
                        print(f"Current scramble: {current_scramble}, epsilon_decay: {self.epsilon_decay:.10f}")
                self.prev_scramble_length = current_scramble

                self.episode_rewards.append(episode_reward)
                episode_reward = 0

                if terminated:
                    self.success_count += 1
                    self.last_window_results.append(1)
                else:
                    self.fail_count += 1
                    self.last_window_results.append(0)

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if step % update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
                self.update()
            if step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if self.use_mlflow and step % log_every == 0 and step > 0:
                avg_reward = np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0.0
                total_episodes = self.success_count + self.fail_count
                success_rate = np.mean(self.last_window_results) if self.last_window_results else 0.0
                max_reward = np.max(self.episode_rewards[-100:]) if self.episode_rewards else 0.0

                if ((current_scramble > best_scramble_len and success_rate > (best_success_rate - 0.25)) \
                or (current_scramble == best_scramble_len and success_rate > best_success_rate)) and episodes_in_current_scramble_len > 5000:
                    best_scramble_len = current_scramble
                    best_success_rate = success_rate
                    os.makedirs("models", exist_ok=True)
                    best_path = self.model_path
                    self.save(best_path, debug=False)
                    print(f"🟢 New best model saved with success rate {best_success_rate:.3f} (at scramble len {best_scramble_len})")

                metrics = {
                    "success_rate": success_rate,
                    "epsilon": self.epsilon,
                    "total_episodes": total_episodes,
                    "current_scramble": info.get("current_scramble", 0),
                    "success_count": self.success_count,
                    "fail_count": self.fail_count,
                    "avg_reward": avg_reward,
                    "max_reward_in_window": max_reward,
                }

                mlflow.log_metrics(metrics, step=step)

                print(
                    f"[{step:07d}] "
                    f"ε={metrics['epsilon']:.3f} | "
                    f"avgR={metrics['avg_reward']:.3f} | "
                    f"maxR={metrics['max_reward_in_window']:.3f} | "
                    f"success_rate={metrics['success_rate']:.2%} | "
                    f"total_episodes={metrics['total_episodes']} | "
                    f"success_count={metrics['success_count']} | "
                    f"fail_count={metrics['fail_count']} | "
                    f"current_scramble={metrics['current_scramble']}"
                )

                if (best_scramble_len >= self.env.scramble_max and
                    best_success_rate >= 0.99 and
                    episodes_in_current_scramble_len >= 5000):
                    print(f"\n✅ Training complete: target performance reached! "
                        f"(scramble={best_scramble_len}, success_rate={best_success_rate:.3f})")
                    break

        print("✅ Training completed.")

    def save(self, path, debug=True):
        torch.save(self.q_net.state_dict(), path)
        if debug:
            print(f"Model saved to {path}")

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Model loaded from {path}")