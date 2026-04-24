"""TD3 Agent for Covert Semantic Communication
===============================================

该模块实现了 Twin Delayed DDPG (TD3) 算法，用于解决连续动作空间下的隐蔽语义通信控制问题。TD3 在 DDPG 基础上引入了双重 Critic、目标动作噪声和延迟策略更新，以缓解过估计偏差并提高稳定性。

本实现简化了网络结构和超参数配置，仍然能够作为参考架构用于本项目。请根据实际需要调参以获得更好性能。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TD3Config:
    state_dim: int
    action_dim: int
    uav_max_speed: float
    num_antennas: int
    max_episodes: int = 1000
    max_steps: int = 200
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005  # 软更新系数
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    replay_size: int = 100000
    start_steps: int = 1000
    noise_std: float = 0.1  # 初始探索噪声
    policy_noise: float = 0.2  # 目标动作噪声幅度
    noise_clip: float = 0.5  # 目标动作噪声截断范围
    policy_delay: int = 2  # 策略延迟更新步数


class ReplayBuffer:
    """经验回放缓冲区，存储状态转移样本。"""
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100000):
        self.capacity = capacity
        self.state_memory = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state_memory = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done_memory = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.state_memory[self.ptr] = state
        self.action_memory[self.ptr] = action
        self.reward_memory[self.ptr] = reward
        self.next_state_memory[self.ptr] = next_state
        self.done_memory[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.state_memory[idx]).float(),
            torch.tensor(self.action_memory[idx]).float(),
            torch.tensor(self.reward_memory[idx]).float(),
            torch.tensor(self.next_state_memory[idx]).float(),
            torch.tensor(self.done_memory[idx]).float(),
        )

    def __len__(self):
        return self.size


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class TD3Agent:
    def __init__(self, cfg: TD3Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Actor and target
        self.actor = Actor(cfg.state_dim, cfg.action_dim).to(self.device)
        self.actor_target = Actor(cfg.state_dim, cfg.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # Two critics and targets
        self.critic1 = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic2 = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic1_target = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic2_target = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr)
        # Replay buffer
        self.replay_buffer = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.replay_size)
        self.total_it = 0

    def select_action(self, state: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """根据当前策略选择动作，可添加探索噪声。返回 [-1,1] 区间内的动作。"""
        self.actor.eval()
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()
        self.actor.train()
        if noise_scale > 0.0:
            action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -1.0, 1.0)

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        self.total_it += 1
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        # --- Critic update ---
        with torch.no_grad():
            # 目标策略平滑
            noise = (
                torch.randn_like(actions) * self.cfg.policy_noise
            ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            # 计算目标 Q 值
            q1_target_next = self.critic1_target(next_states, next_actions)
            q2_target_next = self.critic2_target(next_states, next_actions)
            q_target = rewards + self.cfg.gamma * (1 - dones) * torch.min(q1_target_next, q2_target_next)
        # 当前 Q 值
        q1_current = self.critic1(states, actions)
        q2_current = self.critic2(states, actions)
        critic1_loss = nn.MSELoss()(q1_current, q_target)
        critic2_loss = nn.MSELoss()(q2_current, q_target)
        # 更新两个 Critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        # --- Actor & target update (delayed) ---
        if self.total_it % self.cfg.policy_delay == 0:
            actions_pred = self.actor(states)
            actor_loss = -self.critic1(states, actions_pred).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # 软更新目标网络
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data)
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """将归一化动作映射到实际动作空间 (delta_pos, phases)。"""
        delta = action[:2] * self.cfg.uav_max_speed
        phases = action[2:] * np.pi
        return np.concatenate([delta, phases])

    def train(self, env, num_episodes: int, max_steps: int, noise_start: float = 0.1, noise_end: float = 0.01) -> None:
        """训练 TD3 agent。"""
        noise_scale = noise_start
        # 线性递减探索噪声，直到 noise_end
        total_steps = 0
        noise_decay = (noise_start - noise_end) / max(max(num_episodes * max_steps - self.cfg.start_steps, 1), 1)
        for episode in range(num_episodes):
            obs = env.reset()
            state = self._obs_to_state(obs)
            episode_reward = 0.0
            for t in range(max_steps):
                if total_steps < self.cfg.start_steps:
                    # 随机动作探索
                    raw_action = np.random.uniform(-1.0, 1.0, size=self.cfg.action_dim)
                else:
                    raw_action = self.select_action(state, noise_scale)
                scaled_action = self.scale_action(raw_action)
                env_action = {
                    "delta_pos": scaled_action[:2],
                    "phases": scaled_action[2:],
                }
                next_obs, reward, done, info = env.step(env_action)
                next_state = self._obs_to_state(next_obs)
                # 存储转移样本
                self.replay_buffer.add(state, raw_action, reward, next_state, float(done))
                state = next_state
                episode_reward += reward
                total_steps += 1
                # 更新探索噪声并训练
                if total_steps > self.cfg.start_steps:
                    noise_scale = max(noise_end, noise_scale - noise_decay)
                    self.update(self.cfg.batch_size)
                if done:
                    break
            print(f"Episode {episode+1}/{num_episodes}, Reward {episode_reward:.3f}, Buffer size {len(self.replay_buffer)}")

    def _obs_to_state(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """将环境观测字典转换为一维状态向量。"""
        uav_pos = obs.get("uav_pos", np.zeros(3))
        willie_pos = obs.get("willie_pos", np.zeros(2))
        state = np.concatenate([uav_pos, willie_pos])
        return state.astype(np.float32)