import mlflow
import numpy as np
import torch.nn as nn
import torch
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer
from collections import deque

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
        self.last_window_results = deque(maxlen=1000)

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

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device).unsqueeze(1)

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

        for step in range(total_steps):
            self.global_step = step

            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Push to replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                self.episode_rewards.append(episode_reward)
                
                if info.get("solved", False):
                    self.success_count += 1
                    self.last_window_results.append(1)
                else:
                    self.fail_count += 1
                    self.last_window_results.append(0)

                episode_reward = 0
                state, _ = self.env.reset()

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Update every `update_freq` steps
            if step % update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
                self.update()

            if step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Log every log_every steps
            if self.use_mlflow and step % log_every == 0 and step > 0:
                avg_reward = np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0.0
                total_episodes = self.success_count + self.fail_count
                success_rate = np.mean(self.last_window_results) if self.last_window_results else 0.0
                max_reward = np.max(self.episode_rewards[-100:]) if self.episode_rewards else 0.0

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

                # Log to MLflow
                mlflow.log_metrics(metrics, step=step)

                # Print the same metrics
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

        print("✅ Training completed.")

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Model loaded from {path}")