"""broadcast_heuristic
======================

实现简单的全向广播策略。该策略忽略波束成形和轨迹控制，只保持 UAV 静止或沿固定轨迹飞行，发射均匀人工噪声。用于对比波束成形收益。
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any


def broadcast_action(observation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """生成广播 baseline 动作。
    UAV 保持静止或缓慢移动，发射均匀相位噪声。
    """
    # UAV 不移动
    delta_pos = np.zeros(2)
    # 生成随机相位或均匀分布
    num_antennas = config.get("num_antennas", 4)
    phases = np.zeros(num_antennas)
    return {
        "delta_pos": delta_pos,
        "phases": phases,
    }


if __name__ == "__main__":
    config = {"num_antennas": 4}
    obs = {"uav_pos": np.array([0.0, 0.0, 100.0]), "willie_pos": np.array([500.0, 0.0])}
    action = broadcast_action(obs, config)
    print("Broadcast action", action)
