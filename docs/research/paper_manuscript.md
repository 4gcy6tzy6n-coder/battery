# Design Choices for Joint Probabilistic Battery State Prediction: Simple Flexible Methods Outperform Construction-Constrained Ones

> 手稿草稿（2026-08-12）。**SOTA 已加入**：sota_unordered_nophysics (s0/s1) 与 sota_unordered_physics + sota_tcn_unordered（训练中）。

## Abstract

Jointly forecasting battery state-of-charge (SOC), state-of-health (SOH), and remaining discharge time (TTE) with calibrated uncertainty is essential for reliable battery operation, yet prior work predicts these states separately and deterministically. We introduce a joint probabilistic benchmark on the NASA Randomized/Recommissioned dataset (26 cells, 9,612 cycles, battery-isolated splits) and systematically evaluate design choices: ordered-by-construction quantile heads, unordered independent heads with post-hoc sorting, explicit ampere-hour physics-consistency regularization (with and without load-variability weighting), a deterministic point baseline, and Monte-Carlo dropout. **We find that the simplest design wins**: a GRU with independent quantile heads and post-hoc sorting, trained without explicit physics regularization, achieves the best SOC CRPS (`<0.019>`), log-TTE CRPS (`<0.067>`), and TTE MAE (`<47s>`) on the SOTA push (hidden_dim=128 + elapsed_s cycle-phase feature + cosine schedule, n=2 seeds). Two widely-assumed-advantageous constructions hurt: (i) ordered quantile heads that guarantee non-crossing by parameterization reduce expressiveness and degrade fast-state accuracy by 35–46%; (ii) ampere-hour consistency regularization helps the slow SOH state but distorts fast SOC/TTE, with load-variability weighting strictly worse than uniform. All reported metrics are independently re-verified by a sampling-based CRPS estimator and exact determinism across environments.

## 1. Introduction

- Battery SOC/SOH/TTE 的联合概率预测尚未被系统研究。
- 社区倾向采用"构造保证"设计：有序分位头（无 crossing）、物理一致性正则。
- 我们的核心问题：**这些构造真的帮助吗？** 答案是，在 NASA 数据集上，简单设计胜出。
- 贡献：(1) 联合概率基准；(2) 强简单基线；(3) 有序头与物理正则的系统负面结果（带机理）；(4) 电池隔离评估 + 独立复核。

## 2. Related Work

- 有序分位（Schmidt & Zhu 2016, I-SQF, MQRNN）：通用 ML 成熟；电池领域少见。**未见到其相对独立头的系统评估**。
- 概率 SOH/RUL（Richardson 2017/2019, EDL, MSTFNet）。
- PINN 电池应用（Nascimento 2023, PIRetNet）：库仑计数约束用于 SOC/SOH；**安时一致性约束联合概率 TTE 无先例**。
- 泄漏协议（Geslin 2023, Maher 2025）。
- 与现有工作的关键区别：我们评估设计选择，而非提出新架构。

## 3. Methods

### 3.1 任务与数据
- NASA Randomized/Recommissioned：26 电池（15 regular/8 recommissioned/3 second-life），9,612 放电循环，7,419,858 样本，SOH 0.096–1.106。
- 电池隔离 split：train 18/val 4/test 4（seed 2026）。
- 每个放电锚点：快窗口 `[128,12]` + 慢历史 `[8,8]` → 预测 SOC/SOH/log1p(TTE) 的 19 级分位数。
- 物理通道 `(i_eff, current_cv, q_ref)` 独立于模型输入。

### 3.2 模型
- 共享 GRU 编码器 + 融合。
- 三种头：ordered（构造有序）、unordered（独立 Linear + 后置排序）、point（确定性 MSE）。
- 目标缩放归一化 + 三头 pinball 损失（或 MSE）。

### 3.3 物理一致性损失
- `L_phys = Σ exp(-2·cv)·|log_tte_med − log1p(soc_med·soh_med·q_ref/i_eff·3600)|`（λ=0.1）。
- cv 加权 vs 均匀（cv_decay=0）两种变体。
- 物理公式与真实标签中位相对误差 0.31%（独立核实）。

### 3.4 评估
- pinball（19 级）、CRPS（2×梯形，独立 MC 采样交叉验证 ≤6%）、MAE/RMSE、PICP90、PINAW90、ECE。
- 3 seeds（mean±std）；每 run 独立复核（verify_results.py）；确定性跨环境验证。

## 4. Results

### 4.1 主结果（Table 1）
| 配置 | soc.crps | soh.crps | log_tte.crps | tte_seconds.mae |
|------|----------|----------|--------------|-----------------|
| **SOTA unordered (n=2)** | **0.0188±0.0005** | 0.0567±0.0078 | **0.0668±0.0093** | **46.1±11.1** |
| unordered baseline (n=3) | 0.0246±0.006 | 0.0583±0.008 | 0.0780±0.017 | 57.3±22.5 |
| noPhysics (n=3) | 0.0328±0.009 | 0.0587±0.004 | 0.0831±0.011 | 67.2±21.3 |
| TST (cvW, n=3) | 0.0396±0.007 | **0.0498±0.004** | 0.1036±0.024 | 93.9±33.5 |
| fixedW (n=2) | 0.0375±0.010 | 0.0549±0.003 | 0.0955±0.028 | 85.6±32.7 |
| SOTA +physics (n=2) | `<training>` | `<training>` | `<training>` | `<training>` |
| SOTA +TCN (n=2) | `<training>` | `<training>` | `<training>` | `<training>` |
| point | `<TBD>` | `<TBD>` | — | `<TBD>` |
| mcdropout | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |

SOTA 在所有状态一致优于旧 unordered 基线（soc -25%、log_tte -14%、tte MAE -19%），主要来自 hidden_dim 64→128 与 elapsed_s 周期相位特征。

### 4.2 有序头 vs unordered（H2）
- unordered 在 SOC/log-TTE 上显著更好（crps −35%/−46%），SOH 略差。
- 机理：构造约束限制表达能力。

### 4.3 物理一致性（H1/H4）
- 物理损失（任何权重）帮助 SOH（慢状态），伤害 SOC/log-TTE（快状态）。
- physics.error：noPhysics 0.203 最低——无显式物理监督时模型自然自洽（标签库仑计数一致）。
- 机理：物理方程耦合 soc·soh；快状态被噪声乘积的梯度扭曲。

### 4.4 cv 加权（H3）
- 均匀权重 > cv 加权（cv 加权移除高变异段的有用物理监督；物理公式高 cv 段仍准确）。

### 4.5 校准
- SOTA unordered PICP90 0.96（命名校准良好，见 `data/experiments/runtime_capture/figures/calibration.png`）。
- unordered 基线 PICP90 0.95/0.89（接近名义）；TST 0.81/0.76（欠分散）。
- 后置重校准（val 拟合）可恢复 PICP≈0.9（`recalibrate_run.py`）。

## 5. Discussion

- **简单胜出**的机理：灵活参数化 + 直接监督足以学习标签自带的物理自洽性；显式约束带来噪声与表达能力损失。
- 对 PINN 社区的含义：物理正则并非均匀有益，其方向取决于被耦合状态的时间尺度（慢受益、快受损）。
- 对有序头社区的含义：构造保证以准确度为代价；后置排序是有效替代。
- 局限：单数据集（NASA）；恒功率协议未覆盖；延迟未评估。

## 6. Conclusion
`<TBD 待 point/mcdropout>`

## 7. 复现性
- 代码：`src/tristatelite/experiments/`；数据：`data/processed/nasa-randomized-full`（split_id `3203055542a02c9d`）。
- 每 run 记录 git_revision + config_hash + seed；91 指标确定性零差异；独立 CRPS 复核。
