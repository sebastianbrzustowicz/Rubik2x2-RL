import mlflow
import numpy as np
import torch.nn as nn
import torch
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer

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
        target_update_freq=500,
        device="cuda",
        use_mlflow=False
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_mlflow = use_mlflow

        # networks
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        self.q_net = QNetwork(obs_dim, 256, act_dim).to(device)
        self.target_net = QNetwork(obs_dim, 256, act_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

        # statistics
        self.global_step = 0
        self.success_count = 0
        self.fail_count = 0
        self.episode_rewards = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, total_steps=100_000, log_every=1000):
        state, _ = self.env.reset()
        episode_reward = 0

        for step in range(total_steps):
            self.global_step = step

            # Choosing actions
            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Replay buffer update
            self.replay_buffer.push(state, action, reward, next_state, done)
            loss = self.update()

            state = next_state
            episode_reward += reward

            if done:
                self.episode_rewards.append(episode_reward)
                if info.get("solved", False):
                    self.success_count += 1
                else:
                    self.fail_count += 1
                episode_reward = 0
                state, _ = self.env.reset()

            # Epsilon decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Target update
            if step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # MLflow logging every log_every steps
            if self.use_mlflow and step % log_every == 0 and step > 0:
                avg_reward = np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0.0
                total_episodes = self.success_count + self.fail_count
                success_rate = (
                    self.success_count / total_episodes if total_episodes > 0 else 0.0
                )

                mlflow.log_metrics({
                    "epsilon": self.epsilon,
                    "avg_reward": avg_reward,
                    "success_rate": success_rate,
                    "success_count": self.success_count,
                    "fail_count": self.fail_count
                }, step=step)

                print(
                    f"[{step:07d}] ε={self.epsilon:.3f} | "
                    f"avgR={avg_reward:.3f} | "
                    f"success_rate={success_rate:.2%}"
                )

        print("✅ Training completed.")

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Model loaded from {path}")