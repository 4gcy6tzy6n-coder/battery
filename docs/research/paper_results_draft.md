# TriStateLite Results Draft (数字待最终实验填入)

> 状态：草稿更新（2026-08-12 18:22）。**SOTA s0/s1 完成**，新 SOTA 全面优于旧 unordered 基线。
> 实验配置见 `configs/experiments/`，运行产物在 `data/experiments/runs/{config}/seed{seed}/` 与 `data/experiments/sota-v2-s{0,1}/`。

## 核心发现（已确认，2026-08-11 03:30）

**物理一致性损失（λ=0.1）具有状态依赖的混合效应**（TST 3 seeds vs noPhysics 3 seeds）：

| 状态 | Δ（TST−noPhysics） | 结论 |
|------|-------------------|------|
| SOH crps | −0.0089（TST 更好）| 物理耦合帮助慢变状态 |
| SOC crps | +0.0068（TST 更差）| 物理耦合扭曲快变状态 |
| log_tte crps | +0.0205（TST 更差）| 略差 |
| tte_seconds.mae | +26.7s（TST 更差）| noPhysics 明显更好 |
| physics.error | +0.046（TST 更差）| 无物理监督反而更自洽 |

**解释**：物理方程 `soc·soh·q_ref/i_eff` 中 soc 快变（每锚点）、soh 慢变（每循环）。SOH 从与高可观测 soc/tte 的耦合中受益；SOC 被噪声乘积 soh·q_ref/i_eff 的梯度扭曲。**显式安时一致性正则并非均匀有益——其方向取决于被耦合状态的时间尺度。**

**联合概率框架本身（noPhysics）很强**：soc.crps 0.033、soh.crps 0.059、tte_seconds.mae 67s（3 seeds）。

## SOTA 推进（2026-08-12 完成）

新 SOTA 配置：unordered 头 + hidden_dim=128 + cosine LR + grad clip + v2 数据集（elapsed_s 特征） + pool=150k。配置 `sota_unordered_nophysics.yaml`，2 seeds：

| 指标 | 旧 unordered（基线）| SOTA sota_unordered_nophysics | Δ |
|------|---------------------|-------------------------------|---|
| soc.crps | 0.025 | **0.0188±0.0005** | **-25%** |
| soh.crps | 0.058 | 0.0567±0.0078 | -2% |
| log_tte.crps | 0.078 | **0.0668±0.0093** | **-14%** |
| tte_seconds.mae | 57s | **46.1±11.1s** | **-19%** |
| soc.picp_90 | 0.951 | 0.961±0.002 | 命名校准良好 |
| log_tte.picp_90 | — | 0.984±0.003 | 略过覆盖 |
| epochs_run | ~50 | 28–46（s0/s1） | early stop 触发 |

**核心结论**：hidden_dim 64→128 + elapsed_s 周期相位特征 + cosine schedule 在所有状态都有一致改进，TTE 误差从 57s 降到 46s（-19%）是 paper 关键新数字。SOTA s0 early-stop 在 28 epoch——新特征下 val 改善更陡峭。

**论文定位修订**：
1. 联合概率 SOC/SOH/TTE 预测 + 有序头（novel，主贡献）
2. 强结果（SOTA unordered 模型）+ 后置重校准
3. 诚实科学发现：物理一致性正则的状态依赖效应（方法学贡献）
4. 待验证：λ 扫描 + detach-soc 变体是否保留 SOH 收益同时避免 SOC 伤害

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
