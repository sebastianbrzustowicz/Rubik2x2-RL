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

        # epsilon parameters
        self.epsilon_start = epsilon_start
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

        # previous face id to forbid repeating same-face moves
        self.prev_face_id = None

    def select_action(self, state, explore=True):
        """
        Select action with hard rule: cannot choose an action that affects the same face
        as the previous action (self.prev_face_id). If explore=False, choose greedy (no
        epsilon random), otherwise use epsilon-greedy but sample only from allowed actions.
        """
        num_actions = self.env.action_space.n
        all_actions = np.arange(num_actions)

        # compute mask of allowed actions (True = allowed)
        if self.prev_face_id is None:
            allowed_mask = np.ones(num_actions, dtype=bool)
        else:
            # forbid any action whose face_id equals prev_face_id
            forbidden_faces = self.prev_face_id
            allowed_mask = (all_actions % 6) != forbidden_faces

        # if exploring and random chance triggers
        do_random = explore and (np.random.rand() < self.epsilon)
        if do_random:
            allowed_actions = all_actions[allowed_mask]
            if allowed_actions.size == 0:
                # fallback: if somehow none allowed, allow all (shouldn't happen)
                allowed_actions = all_actions
            return int(np.random.choice(allowed_actions))

        # greedy selection with masking
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t).cpu().numpy().ravel()  # shape (num_actions,)

        if allowed_mask.sum() == 0:
            # fallback: pick argmax overall
            return int(int(np.argmax(q_values)))

        # mask forbidden actions by setting them to a very low value
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
        self.prev_scramble_length = self.env.current_scramble  # for epsilon reset when scramble increases

        for step in range(total_steps):
            self.global_step = step

            # select action (epsilon-greedy) — select_action enforces same-face ban
            action = self.select_action(state, explore=True)

            # step env
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # push transition
            self.replay_buffer.push(state, action, reward, next_state, done)

            # remember prev face id to forbid repeating
            self.prev_face_id = (action % 6)

            state = next_state
            episode_reward += reward

            if done:
                # Reset environment and pass success + scramble length (so env can adapt)
                # Note: env.reset signature in your code supports last_solved/last_scramble
                last_solved = terminated
                last_scramble = self.env.current_scramble
                state, _ = self.env.reset(last_solved=last_solved, last_scramble=last_scramble)

                # Reset prev_face_id when new episode begins
                self.prev_face_id = None

                # Reset epsilon only if scramble level increased
                current_scramble = self.env.current_scramble
                if current_scramble > self.prev_scramble_length:
                    min_scramble = self.env.scramble_min
                    max_scramble = self.env.scramble_max
                    scramble_ratio = (current_scramble - min_scramble) / max(1, max_scramble - min_scramble)

                    base_decay = self.epsilon_decay  # we store the decay value in the startup agent
                    scale = 1.08  # 1.08 was suitable

                    # exponential increase: decay increases towards 1 for larger scrambles
                    self.epsilon_decay = base_decay + (1.0 - base_decay) * (scramble_ratio ** scale)

                    # safety
                    self.epsilon_decay = min(self.epsilon_decay, 0.99999999999)
                    self.epsilon = max(0.2, self.epsilon_start)
                    print(f"Current scramble: {current_scramble}, epsilon_decay: {self.epsilon_decay:.10f}")
                self.prev_scramble_length = current_scramble

                # bookkeeping
                self.episode_rewards.append(episode_reward)
                episode_reward = 0

                if terminated:
                    self.success_count += 1
                    self.last_window_results.append(1)
                else:
                    self.fail_count += 1
                    self.last_window_results.append(0)

            # epsilon decay per step
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # update networks
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