# reference_notes.md

## 1. 文档目的
本文件不是文献综述，而是给 Codex 的“实现参考摘要”。只保留对代码与实验设计直接有帮助的信息。

## 2. 参考文献 A：CUDT-SC / DRL-JPPO
论文：Covert UAV Data Transmission Via Semantic Communication: A DRL-Driven Joint Position and Power Optimization Method

### 2.1 可借鉴点
1. 使用全双工 UAV 发射人工噪声，提高隐蔽性。
2. 将语义通信引入 UAV 隐蔽传输场景。
3. 定义 SUDT 作为语义层吞吐量指标。
4. 将问题表述为 MDP，用 DRL 求解。

### 2.2 该文的控制变量
- UAV 位置
- UAV 人工噪声功率

### 2.3 该文的语义模块
- 使用 DeepSC 作为语义通信模型
- 使用 MINE 近似互信息并定义语义传输能力

### 2.4 该文实验呈现方式
主文中重点展示：
- 训练收敛情况
- 累积 reward
- UAV 与 Alice 的距离变化
- 累积 SUDT 对比

### 2.5 本文如何使用它
- 作为参考基线来源
- 不能直接照搬为本文的主方法，因为该文没有连续相位控制
- 实现时可提供 `drl_jppo_ref` 版本，用于和 AGD-DRL 做参考比较

### 2.6 对 Codex 的指令
- 不要把 DRL-JPPO 写成本文完全同构的主对比方法
- 若为了兼容本文环境做适配，请明确命名为 `drl_jppo_ref`
- 文档中注明：这是“参考文献基线”，不是“同维严格公平基线”

## 3. 参考文献 B：DeepSC
论文：Deep Learning Enabled Semantic Communication Systems

### 3.1 可借鉴点
1. DeepSC 是端到端语义通信框架。
2. 包含语义编码器 / 信道编码器 / 信道解码器 / 语义解码器。
3. 适合文本语义传输。
4. 支持预训练与再训练思路。

### 3.2 本文如何使用它
- 作为语义通信模块的结构参考
- 基础验证阶段可冻结
- 主性能对比阶段必须纳入完整系统，并允许联合微调

### 3.3 对 Codex 的指令
- 不要把 DeepSC 简化成一个完全黑盒常数函数
- 至少要保留一个清晰的接口层，例如：
  - `encode_text_to_symbols()`
  - `decode_symbols_to_text()`
  - `compute_semantic_rate()`
- 若首版实现过重，可先提供可替换的模块接口，并允许从预训练权重加载

## 4. 参考文献 C：多目标 / 连续控制 DRL 对比论文
论文：Aerial Reliable Collaborative Communications for Terrestrial Mobile Users via Evolutionary Multi-Objective Deep Reinforcement Learning

### 4.1 可借鉴点
1. 对比算法选择逻辑清楚：
   - 传统多目标算法
   - DDPG/TD3 类连续控制算法
   - 自身改进算法
2. 强调公平比较：
   - 相同参数
   - 相同环境
   - 相同评价指标
3. 输出标准化性能图表和指标

### 4.2 本文如何使用它
- 借鉴“如何组织对比实验”和“如何保证公平性”
- 不直接复用其具体多目标算法结构

### 4.3 对 Codex 的指令
- 主性能比较时，优先把 TD3 / DDPG 做成与 AGD-DRL 尽量同维的连续控制基线
- 统一训练预算与评估流程

## 5. 本文自己的关键实现约束
### 5.1 不可丢失的核心创新
1. 闭式功率反推
2. 连续相位控制
3. MHA 状态特征提取
4. 条件扩散动作生成
5. 双 Critic 稳定训练

### 5.2 实现优先级建议
#### 第一层
- 可运行环境
- 闭式功率反推
- baseline
- TD3/DDPG

#### 第二层
- AGD-DRL 主干
- DeepSC / MINE 预训练与加载
- 主性能联合微调

#### 第三层
- 消融实验接口
- 参数敏感性分析接口

## 6. 推荐的基线实现策略
### 6.1 同维公平基线
- TD3
- DDPG

动作：
- UAV 轨迹控制
- 连续相位控制
- 功率闭式反推

### 6.2 参考文献基线
- DRL-JPPO 参考实现

动作：
- 位置 / 轨迹
- 功率 或闭式功率适配版本
- 波束可用启发式规则

## 7. 不该做的事
- 不要只因为实现方便，就把所有算法都改成离散动作
- 不要把 AGD-DRL 的扩散策略替换成普通 MLP actor 后还继续叫 AGD-DRL
- 不要删除语义模块后仍宣称是在做完整主性能实验
- 不要在没有记录的情况下修改 SUDT 的定义口径
