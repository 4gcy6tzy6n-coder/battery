# TriStateLite 论文研究路径（Research Plan）

> 状态：规划定稿。基于两轮文献调研（2026-08-10）、全量数据构建与数值核实。
> 配套：`docs/superpowers/specs/2026-08-06-phase2b-probabilistic-core-design.md`（概率核心设计）、
> `docs/superpowers/plans/2026-08-06-tristatelite-phase2b-probabilistic-core.md`（Phase 2B 实施计划）。

---

## 1. 论文主张（Thesis）

**单一轻量神经网络可同时输出 SOC、SOH、放电剩余时间（TTE）三个状态的校准概率分布（有序分位数），并借由一条跨状态物理一致性约束（安时积分关系）将三者耦合，使联合概率预测在负载随机变化的真实电池数据上优于无物理约束或无序分位数基线。**

- 工作名：**TriStateLite** —— Physics-consistent joint probabilistic prediction of battery SOC, SOH, and time-to-end-of-discharge.
- 三个状态用同一编码器、三个**有序分位数头**（`q05/q50/q95`，参数化保证 `q05≤q50≤q95`，SOC/SOH 有界 [0,1]、log-TTE 非负）。
- 物理一致性损失：`tte_phys = soc_med·soh_med·q_ref / i_eff × 3600`，对齐预测的 `log_tte_med`，仅当 `i_eff > 0.05 A` 激活，样本权重 `exp(-2·current_cv)`（负载越稳、物理约束越强）。

## 2. 新颖性定位（已调研交叉确认）

**核心卖点（开放，未检索到直接先例）**：安时积分物理一致性损失约束**联合概率** SOC/SOH/TTE 预测。三个交汇点：
1. 物理关系以**辅助一致性正则**（非机制模型、非主导损失）形式出现；
2. 联合概率输出的第三目标是 **log 尺度 TTE**（几乎无概率 TTE 先例，Dynaformer 等均为点预测）；
3. 物理损失权重由**负载电流变异度 `current_cv` 动态调制**——领域统计量权重未见报道（ReLoBRaLo 等通用自适应加权也未用该量）。

**邻近但不同（必须引用的对标）**：Nascimento 2023（联合 SOC/SOH 贝叶斯 PINN，无 TTE、无安时一致性损失）；Dynaformer 2023（老化感知 EoD 点预测）；Richardson 2017/2019（GPR/GP 转移，概率 SOH/RUL）；FeatureFormer-CQR 2025（共形分位回归）；MSTFNet 2025 / EDL 2026（校准评估规范）。

**明确不作卖点**：电池隔离评估协议（Geslin 2023 Joule / Maher 2025 已充分论证——作为方法学规范呈现）；有序分位数头本身（通用 ML 成熟——作为组件并引用 I-SQF、MQRNN、Brando）；NASA RUL 点预测精度。

**风险提示**：文献检索基于 WebSearch，投稿前需在 Google Scholar 用 "ampere-hour integral consistency loss"、"physics-informed log-TTE" 二次交叉验证。**Karvanen 2016 不存在，勿引用**（改用 Schmidt & Zhu 2016）。

## 3. 数据集与协议（已构建并核实）

- **数据集**：NASA Randomized/Recommissioned（26 电池，15 regular / 8 recommissioned / 3 second-life）。
- **产物**：`data/processed/nasa-randomized-full`（split_id `3203055542a02c9d`）——9,612 接受循环、7,419,858 样本、SOH 全范围 0.096–1.106（battery01 衰减 34.2%，信号充足）。
- **划分**：电池隔离，train 18 / val 4 / test 4（种子 2026 随机置换，**未按组分层**）。
- **任务**：对每个放电锚点（stride 10），给定因果窗口 `fast_x [128,12]`（11 个 `__scaled` 特征 + 布尔 availability）+ 慢速历史 `slow_x [8,8]`（仅更早完成循环摘要），预测 `targets [soc, soh, log1p(tte)]` 的 `q05/q50/q95`。
- **物理通道**：`physics = {i_eff_60s (A), current_cv_60s (无量纲), q_ref_ah (Ah)}` 独立于模型输入，只进一致性损失。
- **标签核实**：`soc = 1−Q_discharged/delivered_ah`，`soh = delivered_ah/q_ref`（q_ref=最早 3 循环中位容量），`log_tte = log1p(剩余秒)`；**物理公式与标签中位相对误差 0.31%、log1p 相关 0.9959**（2026-08-10 核实，见记忆 `full-dataset-facts`）。

## 4. 模型设计

```
fast_x [B,128,12] ──► GRU 编码器 (2层, hidden=64, 接 drop)
slow_x [B,8,8]   ──► mean 池化 ─► 拼接 ─► tanh 融合层 ─► [B,64]
                                                     │
                      ┌──────────────┬────────────────┴──────────────┐
              BoundedQuantileHead(64)  BoundedQuantileHead(64)  PositiveQuantileHead(64)
                      │                │                          │
                 soc q05/q50/q95   soh q05/q50/q95        log_tte q05/q50/q95
```

- 三个头共享融合向量，各自独立 `Linear(64→3)` + 参数化（sigmoid/softplus）保证有序。
- 损失：`L = L_pinball(soc) + L_pinball(soh) + L_pinball(log_tte) + λ · L_phys`，`λ` 为超参（默认 0.1）。
- 反解：预测 `log1p(tte)`，评估时 `tte = expm1`。

## 5. 实验矩阵

| 配置 | 说明 | 回答假设 |
|------|------|---------|
| **TST（主）** | GRU + 有序头 + 物理一致性（cv 加权，λ=0.1） | 主结果 |
| TST-noPhysics | 去掉物理损失（λ=0） | H1：物理约束价值 |
| TST-unordered | 独立 QR 输出 + 预测后排序 | H2：有序头价值 |
| TST-fixedW | 物理损失固定权重（不按 cv 调制） | H3：cv 加权价值 |
| Point-GRU | 同编码器，MSE/MAE 点回归三个目标 | 概率 vs 点预测价值 |
| MC-Dropout | 主模型 + 推理期 dropout 采样（替代基线） | 与主流不确定性法对比 |

**消融细分**（如需更细）：物理损失 λ ∈ {0, 0.01, 0.1, 1.0}；窗口长度 {64, 128, 256}；hidden ∈ {32, 64}。

## 6. 评估指标（全部按 test 电池隔离计算）

对 `q05/q50/q95`：
- **pinball loss**（三状态各自 + 平均）
- **CRPS**（由分位数近似积分）
- **中位绝对误差 MAE**（点预测质量，报告 rRMSE）
- **区间**：90% CI 的 PICP（覆盖率）、PINAW（锐度）
- **校准**：可靠性图 + ECE（q50 经验分位偏差）
- **物理一致性诊断**：测试集上 `|log_tte_med − log1p(soc_med·soh_med·q_ref/i_eff·3600)|` 相对基线——证明"预测自洽"
- **T 状态细分**：TTE 按电池、按负载段（低/高 current_cv）分层报告

**统计严谨性**：主结果与核心消融各 **5 个种子**（mean±std）；次级消融 3 种子；固定 seed=2026 split 不变；报告每配置训练墙钟时间与参数量。

## 7. 假设清单

- **H1（物理约束）**：TST 的 TTE/SOC/SOH 概率指标与校准显著优于 TST-noPhysics（同种子配对比较）。
- **H2（有序头）**：TST 校准优于 TST-unordered，且 TST 分位数永无 crossing（构造保证）。
- **H3（cv 加权）**：在高电流变异负载段，TST 优于 TST-fixedW。
- **H4（自洽性）**：TST 测试预测满足物理一致性等式（误差远小于 noPhysics）。

## 8. 里程碑与风险

| 里程碑 | 交付 | 风险/缓解 |
|--------|------|-----------|
| M1 实验基础设施（模型/训练/评估/CLI） | `src/tristatelite/experiments/` + 测试 | 单元测试锚定指标正确性 |
| M2 冒烟验证（小数据跑通全流程） | smoke 运行 + 指标合理性 | SOH 信号弱→用全量 |
| M3 主实验 5 种子 | 结果表 | 训练时间→控制 anchors/epoch |
| M4 消融 + MC-dropout 基线 | 消融表 + 校准图 | MPS 不可靠→CPU |
| M5 数据复核（E 阶段） | 复算记录 | 指标实现 bug→独立复算脚本 |
| M6 论文结果素材 | Results 表/图 + 主张核对表 | 引用完整→二次检索 |

**总训练预算（CPU，28–32 ms/step）**：主+消融 ≈ 8 配置 × 5 种子 = 40 runs；每 run 限 50k anchors/epoch、≤60 epoch、early-stop → ~25–40 min/run，总 ~20–28 小时，可分批后台跑。

## 9. 后续（F 阶段）论文素材

- 主结果表（三个状态 × 指标 × 配置）
- 消融表 + 校准可靠性图（3×1 grid：soc/soh/tte）
- 物理自洽性散点/箱线图
- 跨负载段（current_cv 分层）TTE 误差图
- 一张架构图 + 一张窗口/物理通道示意

## 10. 范围外（明确不做）

电压曲线逐点预测、共形预测框架、完整分布（flows）输出、多数据集跨库泛化、实时 BMS 部署、与其他模型族（TCN/Transformer）的工程竞速。
