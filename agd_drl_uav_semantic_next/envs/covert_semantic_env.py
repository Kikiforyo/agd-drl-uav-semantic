"""CovertSemanticEnv
=====================

此模块定义了用于模拟全双工多天线 UAV 隐蔽语义通信场景的环境类 `CovertSemanticEnv`。

环境遵循 OpenAI Gym 风格，包含 `reset()`、`step(action)` 等接口，支持与强化学习算法交互。

**注意**：本文件当前仅提供了环境的框架和部分基本逻辑，尚未实现具体的物理层建模、语义编码解码和隐蔽性约束计算。待进一步开发时，请根据论文和 `AGENTS.md` 中的描述完善以下方面：

- 三节点链路建模：地面发送端 Alice、全双工多天线 UAV、移动监测者 Willie；
- 信道模型：考虑视距/非视距传播、路径损耗、瑞利衰落等；
- 多天线波束成形：根据输入的相位向量计算发射与干扰波束；
- 闭式功率反推：利用 KL 散度隐蔽约束反推最小干扰功率；
- 语义通信模块：调用 `semantic_interface.py` 中的语义编码/解码接口；
- 奖励函数设计：综合语义吞吐量（SUDT）、隐蔽性约束、自干扰抑制等指标。

完成环境实现后，请确保能够运行单个 episode 的 smoke test，以便调试。
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Any


class CovertSemanticEnv:
    """隐蔽语义通信环境。

    状态 state: 包含 UAV 位置、Willie 位置、信道状态、语义缓冲等信息。
    动作 action: 由强化学习算法输出，包括 UAV 位移控制向量和天线相位向量。
    奖励 reward: 根据语义吞吐量和隐蔽性约束计算。

    参数在初始化时通过字典 config 提供。
    """

    def __init__(self, config: Dict[str, Any], semantic_module: Any | None = None):
        # 基础环境参数
        self.config = config
        self.time_step = 0
        # UAV 状态 (x, y, z)
        self.uav_pos = np.array(config.get("uav_init_pos", [0.0, 0.0, 100.0]), dtype=float)
        # Willie 状态 (x, y)；假定 Willie 在地面
        self.willie_pos = np.array(config.get("willie_init_pos", [500.0, 0.0]), dtype=float)
        # Alice 固定位置
        self.alice_pos = np.array(config.get("alice_pos", [0.0, 0.0]), dtype=float)
        # 语义模块，用于编码/解码
        self.semantic_module = semantic_module
        # 是否冻结语义模块
        self.freeze_semantic = config.get("freeze_semantic", True)
        # 其他状态变量
        self.current_observation: Dict[str, Any] = {}
        self.done = False

    def reset(self) -> Dict[str, Any]:
        """重置环境至初始状态并返回初始观测。"""
        self.time_step = 0
        self.uav_pos = np.array(self.config.get("uav_init_pos", [0.0, 0.0, 100.0]), dtype=float)
        self.willie_pos = np.array(self.config.get("willie_init_pos", [500.0, 0.0]), dtype=float)
        self.done = False
        # TODO: 重置语义模块缓存和信道状态
        self.current_observation = self._get_observation()
        return self.current_observation

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """执行一步动作。

        参数：
            action: 字典，包含两部分：
                - 'delta_pos': np.ndarray, UAV 在二维平面上的位移向量 (dx, dy)
                - 'phases': np.ndarray, 多天线的相位向量 (长度 = 天线数)

        返回值：(observation, reward, done, info)
        """
        if self.done:
            raise ValueError("环境已终止，请先调用 reset().")

        # 更新 UAV 位置
        delta = action.get("delta_pos", np.zeros(2))
        max_speed = self.config.get("uav_max_speed", 10.0)
        # 限制移动距离
        delta = np.clip(delta, -max_speed, max_speed)
        self.uav_pos[:2] += delta

        # 更新信道状态、Willie 位置等
        self._update_willie_position()
        self._update_channel_state(action)

        # 根据闭式功率反推计算干扰功率，生成噪声
        jam_power = self._compute_jamming_power()

        # 通过语义模块产生语义吞吐量
        sudt = self._compute_semantic_throughput(action)

        # 计算奖励
        reward = self._compute_reward(sudt, jam_power)

        # 观测下一状态
        self.current_observation = self._get_observation()

        # 更新终止条件
        self.time_step += 1
        max_steps = self.config.get("episode_length", 100)
        if self.time_step >= max_steps:
            self.done = True

        info = {
            "sudt": sudt,
            "jam_power": jam_power,
        }
        return self.current_observation, reward, self.done, info

    # ===== 内部方法 =====
    def _get_observation(self) -> Dict[str, Any]:
        """生成当前观测的向量。"""
        obs = {
            "uav_pos": self.uav_pos.copy(),
            "willie_pos": self.willie_pos.copy(),
            # TODO: 加入信道状态、语义缓冲等信息
        }
        return obs

    def _update_willie_position(self) -> None:
        """更新 Willie 的移动轨迹。此处使用简单随机游走，可根据需要替换为更复杂的模型。"""
        speed = self.config.get("willie_speed", 1.0)
        # 随机方向
        angle = np.random.uniform(0, 2 * np.pi)
        self.willie_pos += speed * np.array([np.cos(angle), np.sin(angle)])

    def _update_channel_state(self, action: Dict[str, np.ndarray]) -> None:
        """根据新位置更新信道状态。

        在简化模型中，我们假设所有链路均遵循自由空间路径损耗，
        信道增益与距离的平方成反比。该函数计算并存储 Alice→UAV
        和 UAV→Willie 的路径损耗，用于后续吞吐量和干扰功率估计。

        由于波束成形和自干扰抑制尚未实现，此处忽略相位向量对信道的影响。
        """
        # 计算 Alice→UAV 距离
        diff_au = self.uav_pos - np.append(self.alice_pos, 0.0)
        d_au = np.linalg.norm(diff_au) + 1e-6  # 避免除以零
        # 计算 UAV→Willie 距离（Willie 在地面，高度为 0）
        diff_uw = self.uav_pos - np.append(self.willie_pos, 0.0)
        d_uw = np.linalg.norm(diff_uw) + 1e-6
        # 自由空间路径损耗：增益与距离平方成反比
        self.channel_gain_au = 1.0 / (d_au ** 2)
        self.channel_gain_uw = 1.0 / (d_uw ** 2)
        # 记录相位向量方便调试
        self.last_phases = action.get("phases")

    def _compute_jamming_power(self) -> float:
        """计算干扰功率。

        在简化模型中，我们使用输入相位向量的幅度来估计发射干扰功率，
        并且添加一个固定的最小功率值以维持隐蔽性。实际情形下应
        根据 KL 散度约束反推功率，这里仅作为占位实现。
        """
        # 固定功率基线
        base_power = float(self.config.get("jam_power_fixed", 0.05))
        phases = getattr(self, "last_phases", None)
        if phases is None:
            return base_power
        # 使用相位向量范数估计额外功率开销
        extra_power = float(np.linalg.norm(phases) / max(len(phases), 1)) * 0.01
        return base_power + extra_power

    def _compute_semantic_throughput(self, action: Dict[str, np.ndarray]) -> float:
        """估算语义吞吐量 (SUDT)。

        如果 freeze_semantic 为 True 或未提供语义模块，则根据当前信道增益
        使用 Shannon 公式简化估计吞吐量：SUDT = log2(1 + SNR)。
        否则调用语义模块的编码/解码流程来计算互信息或准确率。
        """
        if self.freeze_semantic or self.semantic_module is None:
            # 采用 Shannon 容量公式近似 SUDT
            # 假设信号功率为 1，噪声功率为 1
            snr = self.channel_gain_au  # Alice→UAV 链路
            sudt = float(np.log2(1.0 + snr))
            return sudt
        # 使用语义模块进行编码、传输与解码
        # 这里给出示例流程：
        # 1. 随机生成任务数据（例如文本 ID）
        # 2. 通过语义模块编码
        # 3. 将编码结果通过信道增益衰减
        # 4. 解码并估计语义率
        # 注意：真实实现中需要传入任务数据及信道噪声
        dummy_data = "hello"
        encoded = self.semantic_module.encode(dummy_data)
        # 信道传输：乘以信道增益并加入噪声
        noise = np.random.normal(scale=0.1, size=encoded.shape)
        received = encoded * np.sqrt(self.channel_gain_au) + noise
        decoded = self.semantic_module.decode(received)
        sudt = float(self.semantic_module.estimate_semantic_rate(dummy_data, decoded))
        return sudt

    def _compute_reward(self, sudt: float, jam_power: float) -> float:
        """根据语义吞吐量与干扰功率计算奖励。

        在简化设计中，我们设定奖励为语义吞吐量减去干扰功率的惩罚项，
        并且乘以一个比例系数以便数值范围合适。若未来需要加入隐蔽
        性违规惩罚，可以在此基础上扩展。
        """
        penalty_weight = self.config.get("jam_power_penalty", 1.0)
        reward = sudt - penalty_weight * jam_power
        return float(reward)


if __name__ == "__main__":
    # 简单的 smoke test，初始化环境并执行随机动作
    env_config = {
        "episode_length": 10,
        "uav_init_pos": [0.0, 0.0, 100.0],
        "willie_init_pos": [500.0, 0.0],
        "alice_pos": [0.0, 0.0],
        "uav_max_speed": 5.0,
        "willie_speed": 1.0,
        "freeze_semantic": True,
    }
    env = CovertSemanticEnv(env_config)
    obs = env.reset()
    print("initial observation", obs)
    done = False
    while not done:
        action = {
            "delta_pos": np.random.uniform(-1, 1, size=2),
            "phases": np.random.uniform(-np.pi, np.pi, size=(4,)),  # 假设 4 根天线
        }
        obs, reward, done, info = env.step(action)
        print(f"t={env.time_step}, reward={reward}, info={info}")
