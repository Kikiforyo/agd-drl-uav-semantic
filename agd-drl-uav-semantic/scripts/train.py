"""泛化训练脚本
================

该脚本根据配置文件启动强化学习训练任务。支持多种算法：AGD‑DRL、TD3、DDPG、DRL‑JPPO 基线等。脚本将环境、算法和语义模块解耦，通过配置指定各部分参数。

当前实现仅提供训练框架和参数解析，不包含具体算法逻辑。请在 `agents/` 子目录下实现每个算法的核心类，并在此脚本中引用。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml
import numpy as np

# ----------------------------------------------------------------------
# Adjust the Python module search path
#
# When this script is invoked via ``python agd-drl-uav-semantic/scripts/train.py``
# the current working directory is the project root, but the package
# ``envs`` lives in the parent directory of this script. Without
# modifying ``sys.path``, Python will not be able to resolve imports
# like ``envs.covert_semantic_env`` because it only searches the
# directory of the script and standard locations. To ensure that the
# package root is available, we insert the parent directory of the
# script (i.e., the project root) at the beginning of ``sys.path``.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 环境及语义模块导入
from envs.covert_semantic_env import CovertSemanticEnv
from envs.semantic_interface import SemanticModule

# 导入算法代理
from agents.ddpg.agent import DDPGAgent, DDPGConfig

# TODO: 导入算法实现，例如 from agents.agddrl.agent import AGDDRLAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agents for covert semantic communication")
    parser.add_argument("--config", type=str, default="configs/train_agddrl.yaml", help="Path to training config file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="outputs/training", help="Directory to save training logs and checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 设置随机种子
    np.random.seed(args.seed)
    # 加载环境配置
    env_config_path = cfg.get("env_config", "configs/env_base.yaml")
    # Resolve the environment config path relative to the project root. This
    # allows specifying config files without worrying about the current
    # working directory.
    if not os.path.isabs(env_config_path):
        env_config_path = os.path.join(project_root, env_config_path)
    with open(env_config_path, "r", encoding="utf-8") as f_env:
        env_cfg = yaml.safe_load(f_env)
    # 初始化语义模块（如果不冻结）
    if not cfg.get("freeze_semantic", True):
        semantic_module = SemanticModule({})  # TODO: 替换为具体模型
    else:
        semantic_module = None
    # 创建环境实例
    env = CovertSemanticEnv(env_cfg, semantic_module=semantic_module)
    # 选择算法
    algo_name = cfg.get("algorithm", "agddrl").lower()
    if algo_name == "ddpg":
        # 提取动作维度信息
        num_antennas = env_cfg.get("num_antennas", 4)
        state_dim = 5  # (uav_x, uav_y, uav_z, willie_x, willie_y)
        action_dim = 2 + num_antennas
        # Convert hyperparameters to appropriate types. YAML will parse
        # scientific notation (e.g. ``1e-4``) as strings, so cast them
        # manually to float or int where necessary.
        def to_float(value, default):
            try:
                return float(value)
            except Exception:
                return float(default)

        def to_int(value, default):
            try:
                return int(value)
            except Exception:
                return int(default)

        ddpg_cfg = DDPGConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            uav_max_speed=float(env_cfg.get("uav_max_speed", 10.0)),
            num_antennas=num_antennas,
            max_episodes=to_int(cfg.get("episodes", 1000), 1000),
            max_steps=to_int(env_cfg.get("episode_length", 200), 200),
            batch_size=to_int(cfg.get("batch_size", 64), 64),
            gamma=to_float(cfg.get("gamma", 0.99), 0.99),
            tau=to_float(cfg.get("tau", 0.005), 0.005),
            actor_lr=to_float(cfg.get("actor_lr", 1e-4), 1e-4),
            critic_lr=to_float(cfg.get("critic_lr", 1e-3), 1e-3),
            replay_size=to_int(cfg.get("replay_size", 100000), 100000),
            start_steps=to_int(cfg.get("start_steps", 1000), 1000),
            noise_std=to_float(cfg.get("noise_std", 0.1), 0.1),
        )
        agent = DDPGAgent(ddpg_cfg)
        print("[INFO] Starting DDPG training...")
        agent.train(env, num_episodes=ddpg_cfg.max_episodes, max_steps=ddpg_cfg.max_steps,
                    noise_start=cfg.get("noise_start", 0.1), noise_end=cfg.get("noise_end", 0.01))
    else:
        print(f"[WARNING] Algorithm {algo_name} not implemented in this script.")


if __name__ == "__main__":
    main()
