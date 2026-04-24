"""DDPG Agent for Covert Semantic Communication
==============================================

该模块实现了用于连续控制的 DDPG 算法，以示例方式演示如何在隐蔽语义通信环境中训练轨迹控制和相位控制策略。实现遵循以下简化假设：

- 状态向量由 UAV 位置 (x,y,z) 和 Willie 位置 (x,y) 组成，维度为 5；
- 动作向量维度为 2 (delta_pos) + `num_antennas` (相位)，默认天线数为 4，总维度 = 6；
- 奖励由环境计算并传递，算法不假设奖励结构；
- 演示版不支持重启学习率调度、优先经验回放等高级特性。

使用方法：在训练脚本中初始化 `DDPGAgent`，调用 `train()` 方法运行若干 episode 的训练。

注意：本实现仅用于示范代码结构，未经过严格调参，实际效果需根据具体环境调整超参数。
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class DDPGConfig:
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
    start_steps: int = 1000  # 随机探索步数
    noise_std: float = 0.1  # 探索噪声标准差


class ReplayBuffer:
    """简单经验回放缓存。"""
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


class DDPGAgent:
    def __init__(self, cfg: DDPGConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(cfg.state_dim, cfg.action_dim).to(self.device)
        self.actor_target = Actor(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic_target = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.replay_buffer = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.replay_size)

    def select_action(self, state: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        self.actor.eval()
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()
        self.actor.train()
        if noise_scale > 0.0:
            action += noise_scale * np.random.randn(*action.shape)
        # Clip actions to [-1, 1]
        return np.clip(action, -1.0, 1.0)

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # target Q
            q_target_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.cfg.gamma * (1 - dones) * q_target_next
        q_current = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_current, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor loss
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """将 [-1,1] 范围的动作向量映射到环境动作空间。

        delta_pos 范围：[-uav_max_speed, uav_max_speed]
        phases 范围：[-pi, pi]
        """
        delta = action[:2] * self.cfg.uav_max_speed
        phases = action[2:] * np.pi
        return np.concatenate([delta, phases])

    def train(self, env, num_episodes: int, max_steps: int, noise_start: float = 0.1, noise_end: float = 0.01) -> None:
        """训练 DDPG agent。

        参数：
            env: 环境实例
            num_episodes: 训练回合数
            max_steps: 每个回合的最大步数
            noise_start: 初始探索噪声标准差
            noise_end: 最终探索噪声标准差
        """
        noise_scale = noise_start
        noise_decay = (noise_start - noise_end) / max(max(num_episodes * max_steps - self.cfg.start_steps, 1), 1)
        total_steps = 0
        for episode in range(num_episodes):
            obs = env.reset()
            state = self._obs_to_state(obs)
            episode_reward = 0.0
            for t in range(max_steps):
                if total_steps < self.cfg.start_steps:
                    # 随机探索
                    raw_action = np.random.uniform(-1.0, 1.0, size=self.cfg.action_dim)
                else:
                    raw_action = self.select_action(state, noise_scale)
                # 将归一化动作映射至实际动作空间
                scaled_action = self.scale_action(raw_action)
                # 构建环境动作字典
                env_action = {
                    "delta_pos": scaled_action[:2],
                    "phases": scaled_action[2:],
                }
                next_obs, reward, done, info = env.step(env_action)
                next_state = self._obs_to_state(next_obs)
                self.replay_buffer.add(state, raw_action, reward, next_state, float(done))
                state = next_state
                episode_reward += reward
                total_steps += 1
                # 更新探索噪声
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
        # 拼接为 (x, y, z, willie_x, willie_y)
        state = np.concatenate([uav_pos, willie_pos])
        return state.astype(np.float32)
