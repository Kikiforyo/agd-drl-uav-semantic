"""beam_heuristic
================

实现简单的波束成形启发式策略。该策略在基础验证阶段用于评估波束增益对隐蔽通信的影响，作为 AGD‑DRL 等复杂算法的参考基线。

当前实现仅提供示例函数，使用固定或随机相位向量将能量聚焦到 Willie 的方向。
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any


def compute_beamforming_phases(uav_pos: np.ndarray, willie_pos: np.ndarray, num_antennas: int) -> np.ndarray:
    """根据 UAV 和 Willie 的相对位置计算简单的波束相位向量。

    该启发式将波束主瓣指向 Willie，忽略自干扰和其他因素。
    """
    # 计算方向向量
    diff = willie_pos - uav_pos[:2]
    angle = np.arctan2(diff[1], diff[0])
    # 生成均匀相移，使得波束主瓣朝向 angle
    # 假设各天线间距为半波长，此时相位差为 -2π * d * sin(theta)，这里简单使用角度
    phases = np.linspace(0, (num_antennas - 1) * np.pi, num_antennas)
    phases = phases * np.cos(angle)
    return phases


def heuristic_action(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """生成启发式策略的动作：UAV 位移和波束相位。

    - 位移方向简单朝向 Alice 或围绕固定轨迹；
    - 相位根据 beamforming heuristic 计算。
    """
    uav_pos = observation["uav_pos"]
    willie_pos = observation["willie_pos"]
    # 位移：朝向 Alice 与 Willie 中点以维持隐藏
    alice_pos = config.get("alice_pos", np.array([0.0, 0.0]))
    target = (alice_pos + willie_pos) / 2
    direction = target - uav_pos[:2]
    if np.linalg.norm(direction) > 1e-6:
        direction = direction / np.linalg.norm(direction)
    # 控制移动速度
    delta_pos = direction * config.get("uav_max_speed", 5.0) * 0.1  # 小步移动

    # 波束相位
    num_antennas = config.get("num_antennas", 4)
    phases = compute_beamforming_phases(uav_pos, willie_pos, num_antennas)
    return {
        "delta_pos": delta_pos,
        "phases": phases,
    }


if __name__ == "__main__":
    # 示例使用
    config = {
        "uav_max_speed": 5.0,
        "num_antennas": 4,
        "alice_pos": np.array([0.0, 0.0]),
    }
    obs = {"uav_pos": np.array([0.0, 0.0, 100.0]), "willie_pos": np.array([500.0, 0.0])}
    action = heuristic_action(obs, config)
    print("Heuristic action", action)
