# codex_task_main_performance.md

下面这段文字可直接交给 Codex 作为主性能对比实验的任务提示词。

---

你正在为论文《全双工多天线 UAV 隐蔽语义通信的联合波束成形与轨迹优化：一种注意力引导的扩散强化学习方法》实现代码。

请先阅读：
- `AGENTS.md`
- `docs/experiment_brief.md`
- `docs/reference_notes.md`

## 任务目标
实现主性能对比实验代码，支持以下算法：
1. AGD-DRL
2. TD3
3. DDPG
4. DRL-JPPO 参考基线

## 重要实验模式
代码必须同时支持两种模式：
1. `freeze_semantic=true`：基础验证实验
2. `freeze_semantic=false`：主性能对比实验

其中：
- 主性能对比实验必须使用完整系统；
- 所有算法共享同一套预训练 DeepSC / MINE 初始权重；
- 进入主性能对比实验后允许联合微调；
- 不允许把主性能实验继续实现成冻结语义模块的纯控制实验。

## 强约束
1. agent 对 AGD-DRL / TD3 / DDPG 只输出轨迹控制量和连续相位向量；
2. 干扰功率由闭式功率反推得到，不直接由策略输出；
3. 所有算法共享相同环境、相同训练预算、相同种子集合；
4. 统一日志、统一评估、统一绘图；
5. 若论文参数缺失，请用 TODO 明确标记，不要擅自编造。

## 需要完成的模块
- 环境：`envs/covert_semantic_env.py`
- 语义模块接口：`envs/semantic_interface.py`
- 预训练脚本：`scripts/pretrain_semantic.py`
- 训练脚本：`scripts/train.py`
- 评估脚本：`scripts/evaluate.py`
- 绘图脚本：`scripts/plot_main_results.py`

## 主图要求
请确保评估阶段至少自动生成：
1. 收敛图
2. 累积语义吞吐量对比图
3. 隐蔽性对比图
4. 干扰功率对比图

## 结果落盘要求
```text
outputs/training/<algo>/<seed>/
outputs/evaluation/<algo>/<seed>/
outputs/figures/
outputs/tables/
```

## 开发顺序
请按以下顺序实施，不要一上来写全部：
1. 跑通环境与 baseline
2. 跑通 TD3 / DDPG
3. 实现 AGD-DRL 主干
4. 接入预训练 DeepSC / MINE
5. 实现主性能联合微调
6. 实现评估与绘图

## 每一步都要给出
- 已完成内容
- 未完成内容
- 运行命令
- 下一步建议

---
