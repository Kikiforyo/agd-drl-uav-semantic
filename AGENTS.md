# AGENTS.md

## 仓库目标
本仓库用于实现论文《全双工多天线 UAV 隐蔽语义通信的联合波束成形与轨迹优化：一种注意力引导的扩散强化学习方法》的可复现实验代码。重点任务是：
1. 先完成基础验证实验；
2. 再完成主性能对比实验；
3. 最终产出论文可直接使用的图表、日志与结果汇总。

本文系统是一个**完整的端到端隐蔽语义通信系统**，不是单纯的物理层控制问题。基础验证阶段允许冻结语义模块；主性能对比阶段必须使用完整系统并进行联合微调。

## 核心研究设定
### 场景
- 地面发送端 Alice：固定位置，发送私密文本数据。
- UAV：带内全双工（IBFD）多天线平台，同时执行语义接收与人工噪声发射。
- Willie：地面移动监测者，位置随时间变化。
- UAV 通过轨迹控制、阵列相位控制与闭式反推干扰功率来满足隐蔽性并提升语义吞吐量。

### 方法主线
- 语义通信模块：DeepSC 风格的语义编码器 / 信道编码器 / 信道解码器 / 语义解码器。
- 语义性能估计：MINE 或兼容的互信息下界估计器。
- 控制策略：AGD-DRL。
- 隐蔽保障：根据 KL 隐蔽性硬约束进行闭式功率反推。

### 动作定义
对 AGD-DRL / TD3 / DDPG：
- agent 仅输出：
  - UAV 速度或位移控制量；
  - 多天线连续相位向量。
- UAV 干扰功率 **不由策略直接输出**，而由隐蔽性硬约束闭式反推得到。

对参考方法 DRL-JPPO：
- 保留其“位置/轨迹 + 功率”风格作为参考基线；
- 若需要与本文环境兼容，可采用“轨迹学习 + 启发式波束 + 闭式功率反推”的参考实现；
- 不要假装它与 AGD-DRL 完全同构，需在注释和实验文档中明确说明。

## 实验阶段约束
### 阶段 A：基础验证
目标：验证物理链路、波束收益、自干扰抑制、闭式功率反推是否正确。
- `freeze_semantic = true`
- 允许固定或冻结 DeepSC / MINE / 语义模块参数
- 先跑启发式 baseline：
  - beam heuristic baseline
  - broadcast baseline
- 这一阶段不要把结果写成“最终主性能结论”

### 阶段 B：主性能对比
目标：验证完整 FDB-CUSC + AGD-DRL 的最终性能优势。
- `freeze_semantic = false`
- 使用完整系统：DeepSC + MINE + 控制策略
- 所有对比算法共享同一套预训练语义模块初始化
- 在主性能比较阶段允许联合微调
- 所有算法必须共享：
  - 相同环境参数
  - 相同训练预算
  - 相同随机种子集合
  - 相同评价脚本

## 对比算法
主性能对比优先实现以下算法：
1. AGD-DRL
2. TD3
3. DDPG
4. DRL-JPPO（参考基线）

## 主图与主表
必须支持自动生成以下主图：
1. 算法收敛图
2. 累积语义吞吐量对比图
3. 隐蔽性对比图
4. 干扰功率对比图

建议主表至少汇总：
- cumulative SUDT
- average semantic rate
- outage ratio
- KL violation ratio 或 covert infeasible ratio
- average gamma_u
- average / max P_jam
- 推理耗时（如已实现）

## 推荐目录结构
```text
configs/
  env_base.yaml
  pretrain_semantic.yaml
  train_agddrl.yaml
  train_td3.yaml
  train_ddpg.yaml
  train_drl_jppo_ref.yaml
  eval.yaml

envs/
  covert_semantic_env.py
  channels.py
  mobility.py
  beamforming.py
  covert_metrics.py
  semantic_interface.py

semantic/
  deepsc_model.py
  mine_estimator.py
  losses.py
  datasets/

agents/
  agddrl/
  td3/
  ddpg/
  drl_jppo_ref/

baselines/
  beam_heuristic.py
  broadcast_heuristic.py

scripts/
  pretrain_semantic.py
  train.py
  evaluate.py
  plot_main_results.py
  run_all.sh

outputs/
  pretrained/
  training/
  evaluation/
  figures/
  tables/

docs/
  experiment_brief.md
  reference_notes.md
  codex_task_main_performance.md
```

## 代码规范
- 使用 Python 3.10+。
- 优先使用 PyTorch。
- 不要把所有逻辑都写进一个脚本。
- 环境、算法、评估、绘图必须解耦。
- 配置优先，避免硬编码。
- 每个训练脚本都必须支持 `--config` 和 `--seed`。
- 每个评估脚本都必须支持从 checkpoint 加载。
- 统一日志格式，输出 CSV 和可读 summary.txt。

## 必须遵守的实现规则
1. **不要擅自改动论文核心设定。**
2. **不要把闭式功率反推删除或改成纯学习功率输出，除非用户明确要求。**
3. **不要在主性能对比阶段继续使用冻结语义模块模式。**
4. **不要为图好看而偷偷改不同算法的训练步数、种子或环境参数。**
5. **不要把参考方法包装成与本文方法完全同维同构。**
6. **不要在没有说明的情况下引入新的生产依赖。**
7. 若发现论文设定缺失，用 TODO 标注并在文档中列出待确认项，不要自行捏造关键公式常数。

## 运行与验证
在提交任何较大改动前，至少完成：
1. 单个 episode 的环境 smoke test
2. 单个 seed 的短训练 smoke test
3. checkpoint 保存与加载测试
4. 评估脚本可读取训练结果并生成图表

## 完成标准
一次任务完成，至少意味着：
- 代码可以运行，不只是生成文件；
- 能输出结构化日志；
- 能复现实验图所需的原始数据；
- 关键假设、限制与 TODO 已记录到 `docs/`；
- 若无法完成全部功能，需明确列出已完成部分与未完成部分。
