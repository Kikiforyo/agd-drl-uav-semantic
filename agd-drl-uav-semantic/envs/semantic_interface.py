"""Semantic Interface
====================

该模块定义了语义通信模型的接口，便于环境通过统一调用实现编码、信道传输和解码功能。具体的语义模型（如 DeepSC 和 MINE 模型）将在 `semantic/` 目录下实现。

接口类 `SemanticModule` 提供如下方法：

- `encode(data: Any) -> np.ndarray`：将原始任务数据编码为连续向量，输入可为文本、序列标识等；
- `decode(received: np.ndarray) -> Any`：将接收向量解码为任务数据；
- `estimate_semantic_rate(original: Any, decoded: Any) -> float`：估计语义层吞吐量或互信息，用作强化学习奖励；
- `train_step(batch: Any, channel: Any)`：若语义模块可训练，则提供单步训练接口，用于预训练或联合微调。

当前实现仅给出接口定义，具体逻辑在子类中实现。
"""
from __future__ import annotations

from typing import Any
import numpy as np


class SemanticModule:
    """语义通信模块基类。"""
    def __init__(self, config: dict[str, Any]):
        self.config = config
        # TODO: 初始化模型参数、词嵌入、编码器/解码器等

    def encode(self, data: Any) -> np.ndarray:
        """将输入数据编码为实数向量。

        子类需要重写该方法。
        """
        raise NotImplementedError

    def decode(self, received: np.ndarray) -> Any:
        """将接收到的连续向量解码为原始数据。

        子类需要重写该方法。
        """
        raise NotImplementedError

    def estimate_semantic_rate(self, original: Any, decoded: Any) -> float:
        """估计语义吞吐量或互信息。

        对于 DeepSC，可基于互信息下界 (MINE) 或自然语言任务准确率。
        """
        raise NotImplementedError

    def train_step(self, batch: Any, channel_state: Any) -> float:
        """执行单步语义模型训练。

        应返回训练损失或指标。用于预训练与联合微调。
        """
        raise NotImplementedError
