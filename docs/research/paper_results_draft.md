# TriStateLite Results Draft (数字待最终实验填入)

> 状态：草稿。所有 `<TBD>` 将由 `scripts/aggregate_results.py` 输出填入。
> 实验配置见 `configs/experiments/`，运行产物在 `data/experiments/runs/{config}/seed{seed}/`。

## 主结果表结构（Table 1）

对 test 电池（battery10/13/22/54，隔离）上的全部锚点，报告每个配置 × 每个状态的：

| 指标 | 含义 |
|------|------|
| pinball | 平均分位数损失（19 级 0.05–0.95） |
| CRPS | 分位数 CRPS（2×梯形积分，~5% 偏差） |
| MAE | 中位数预测的点误差 |
| PICP90 | q05–q95 区间覆盖率（目标 0.90） |
| PINAW90 | 区间宽度 |
| ECE | 多分位期望校准误差 |

配置：**TST**（有序头+物理一致性，cv 加权）、TST-noPhysics、TST-unordered、TST-fixedW、Point-GRU、MC-Dropout。每个配置 3–5 seeds（mean±std）。

## 假设检验（H1–H4）预期对照

### H1 物理约束的价值
- TST vs TST-noPhysics：TST 的 TTE/SOC/SOH 概率指标应更优（尤其 TTE pinball/CRPS）。
- 判据：同种子配对差值的均值±std；`<TBD>`。

### H2 有序头价值
- TST vs TST-unordered：TST 校准（ECE）更优；TST 无 crossing（构造保证）；unordered 经后置排序。
- 判据：ECE 差值；`<TBD>`。

### H3 cv 加权价值
- TST vs TST-fixedW：在高电流变异负载段，TST 的 TTE 误差更小。
- 判据：按 `current_cv_60s` 分层（低/中/高）的 TTE 指标；`<TBD>`。

### H4 物理自洽性
- TST 测试预测应满足 `log_tte_med ≈ log1p(soc_med·soh_med·q_ref/i_eff·3600)`。
- 判据：`physics.error`（`scripts/verify_results.py` 与 `evaluate.physical_consistency_error`）；TST 应显著低于 noPhysics；`<TBD>`。

## 图表清单

1. **Fig 1**：架构图（GRU 编码器 + 融合 + 3 有序头 + 物理通道）。
2. **Fig 2**：三状态校准可靠性图（TST vs noPhysics vs unordered）。
3. **Fig 3**：物理自洽性散点（预测 log-TTE 中位 vs 物理 log-TTE）。
4. **Fig 4**：按负载段（current_cv 分层）的 TTE MAE 箱线图（TST vs fixedW）。
5. **Fig 5**：单个放电循环上 TTE 中位与 90% 区间的预测轨迹 vs 真值。

## 数据核实记录（E 阶段）

每个进入论文的数字必须满足：
- [ ] `scripts/verify_results.py --run <dir>` 输出 PASS（独立 MC-CRPS 重算偏差 <5%）。
- [ ] 与 `scripts/aggregate_results.py` 的 mean±std 一致。
- [ ] 每个配置的 `results.json` 含 `git_revision`（复现溯源）。
- [ ] 随机种子固定（`seed` 字段），split_id 一致（`3203055542a02c9d`）。
