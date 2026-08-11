# TriStateLite 论文研究进展日志

> 记录实验阶段的关键决策、发现与核实结果，供 F 阶段论文整理使用。

## 时间线（2026-08-10）

- **调研**：两轮 WebSearch 文献调研，确定新颖性定位（联合概率+物理一致性，无直接先例）。见 `paper_research_plan.md`。
- **数据**：全量数据集构建完成——26 电池（15 regular/8 recommissioned/3 second-life）、9,612 循环、7,419,858 样本、SOH 范围 0.096–1.106。split_id `3203055542a02c9d`。
- **物理公式核实**：`soc·soh·q_ref/i_eff·3600` 与真实 TTE 中位相对误差 0.31%（2026-08-10 独立验证）。
- **基础设施**：PreparedWindowDataset（numpy 预计算，batch 取数 1.2ms vs pandas 1332ms）、TriStateLiteNet（GRU+3 有序头，mask-exact packing）、训练/评估/复核/图表/校准全链路。127 测试全绿。
- **指标验证**：CRPS 用 `2×梯形积分`，与独立逐锚点 MC 采样交叉验证一致（soc 0.3%/soh 2.2%/log_tte 6.2%，log_tte 偏差为重型尾部分布的方法性差异）。pinball/PICP/PINAW/ECE 直接重算 0% 偏差。
- **确定性**：同配置+seed 跑两次，91 个标量指标零差异。CPU 训练完全可复现。

## 关键科学发现

### 校准模式（main seed0 实测）
- **soc/soh 欠分散**（区间过窄）：q50 经验覆盖 30%/9%（名义 50%）；PICP90 0.78/0.81
- **log_tte 过分散**（区间过宽）：q05 经验覆盖 25%（名义 5%）；PICP90 0.73
- **归因**：目标缩放权重 `w=1/mean|target|`（soc×2, soh×1.7, tte×0.37）导致模型优先拟合 soc/soh → 窄区间，log_tte 权重低 → 宽区间
- **对策**：per-state 经验分位重校准（`src/tristatelite/experiments/calibration.py`，共形风格，val 拟合 + test 应用）。合成验证 PICP 从 0.59→0.82 恢复。

### 物理一致性（H4 待对比确认）
- main seed0 physics error = 0.37（log1p 单位）。待与 noPhysics 对比判断物理损失是否改善自洽性。

### Per-battery 异质性
- battery54（second_life）log_tte CRPS 0.225 vs 其他 test 电池 ~0.06——跨电池组泛化差异是论文要点。

## 实验矩阵状态

- 配置：main（TST）、ablation_nophysics、ablation_unordered、ablation_fixedweight、baseline_point、baseline_mcdropout
- 预算：pool 100k、max_epochs 50、early_stop patience 6、batch 64、GRU(2层,64h)、19 分位数
- 并行：2 worker × 3 线程（5 worker 因内存交换失效，2 worker 健康 ~2.5min/epoch）
- 预计 ~14h 完成 18 runs；完成后：复核每个 run → 聚合 → 假设检验 → 决定重校准重跑

## 重大发现（2026-08-10 深夜，seed0 初步）

**物理一致性损失（λ=0.1）在大多数指标上劣于 noPhysics**：

| 指标 | TST | noPhysics |
|------|-----|-----------|
| soc.crps | 0.0477 | 0.0279 |
| soc.mae | 0.0781 | 0.0398 |
| tte_seconds.mae | 130.8s | 91.8s |
| physics.error | 0.3717 | 0.1890 |
| soh.crps | 0.0477 | 0.0618（TST 更好）|

**归因**：
- 物理损失把 soc/soh/log_tte 中位数拉向一致性方程，但 soc/soh 中位数误差经乘积放大 → 扭曲边缘预测
- 标签本身自洽（库仑计数），noPhysics 直接拟合标签即自然学会自洽性 → 物理损失既冗余又有害
- soh 是唯一 TST 更好的状态（物理约束帮助跨循环 SOH 平滑？待确认）

**影响**：论文核心假设（物理正则有益）方向存疑。待确认：
- [x] 全矩阵跨 seed 确认方向（见下）
- [ ] λ 扫描（0.001/0.01/0.05）找不伤害的权重
- [ ] 论文框架：正面（轻正则微增益）vs 诚实负面（物理损失不必要，模型自然自洽）+ 联合概率模型本身的价值

## 混合效应确认（2026-08-11 02:43，TST 3 seeds vs noPhysics 2 seeds）

物理一致性损失（λ=0.1）**混合效应**：

| 状态 | 效果 | 稳健性 |
|------|------|--------|
| SOC | 变差（crps 0.040 vs 0.035；mae +14%）| 一致 |
| SOH | 更好（crps 0.050 vs 0.061；mae -18%）| 稳健（无 std 重叠）|
| log_tte | 略差（crps 0.104 vs 0.087）| 中度 |
| tte_seconds.mae | noPhysics 更好（73s vs 94s）| TST std 高（±27）|
| physics.error | noPhysics 更低（0.195 vs 0.249）| noPhysics std 低 |

**机理**：物理方程耦合 soc·soh。SOH（慢变/每循环）从与高可观测 SOC 的耦合中受益；SOC（快变/每锚点）被噪声乘积 soh·q_ref/i_eff 扭曲。**这本身是论文级发现**：显式安时一致性正则并非均匀有益，其方向取决于被耦合状态的时间尺度。

## 待决

- [ ] H1-H4 假设检验最终结果
- [ ] 是否重跑关键配置（带 checkpoint）做重校准最终数字
- [ ] 论文结果表/图/主张核对

## H2 发现（2026-08-11 05:40，unordered seed0）

**有序头（构造保证无 crossing）劣于 unordered 独立头 + 后置排序**：

| 指标 | TST 有序(3) | unordered(1) |
|------|-----------|-------------|
| soc.crps | 0.0396 | 0.0196 |
| log_tte.crps | 0.1036 | 0.0686 |
| soc.ece | 0.1132 | 0.0517 |
| soc.picp_90 | 0.814 | 0.961 |
| soh.crps | 0.0498 | 0.0639（有序更好）|

**解释**：有序头参数化（q05=q50·σ(l), q95=q50+(1-q50)·σ(u)）限制表达能力；unordered 头每分位独立参数，更好拟合条件分位。参数数量相同，差异在参数化的灵活性。

**影响**：两个"聪明"设计（物理损失、有序头）均被朴素基线击败 → 论文转向诚实比较研究："简单设计（独立分位头 + 后置排序，无显式物理正则）在联合概率电池预测中胜过构造约束设计"。

## H2 确认（2026-08-11 06:41，unordered 2 seeds）

| 状态 | TST 有序(3) | unordered(2) | 结论 |
|------|-----------|-------------|------|
| soc.crps | 0.0396 | 0.0213 | unordered 好 46% |
| log_tte.crps | 0.1036 | 0.0680 | unordered 好 35% |
| soh.crps | 0.0498 | 0.0566 | 有序略好 |

**统一机制**：有序头（构造约束）与物理损失（耦合约束）都**帮助慢变状态 SOH、伤害快变状态 SOC/log_tte**。慢状态受益于耦合/正则（提供跨状态信息），快状态损失表达能力（约束+噪声梯度）。

## 全面聚合排名（2026-08-11 08:42，3 configs × 3 seeds）

| 指标 | unordered | noPhysics | TST(main) |
|------|-----------|-----------|-----------|
| soc.crps | **0.025** | 0.033 | 0.040 |
| log_tte.crps | **0.078** | 0.083 | 0.104 |
| tte_seconds.mae | **57s** | 67s | 94s |
| soh.crps | 0.058 | 0.059 | **0.050** |
| soc.picp_90 | **0.951** | 0.894 | 0.814 |

**结论**：unordered（灵活分位头 + 后置排序）是最佳整体模型；noPhysics 次之；TST（有序头）最差于快状态但最佳于 SOH。**"简单设计胜出"叙事牢固确立**。unorderd 校准最好（PICP≈0.95/0.89 接近名义），重校准对其可能不那么必要。

## H3 发现（2026-08-11 09:44，fixedW seed0）

**cv 加权劣于固定权重**：log_tte.crps cvW 0.104 vs fixed 0.067；tte_seconds.mae cvW 94s vs fixed **53s**；soh cvW 略好。

**重大含义**：固定权重物理损失（λ=0.1 均匀）TTE MAE 53s **优于 noPhysics 67s**——物理损失可能确实帮助 TTE，问题在 cv 加权（物理公式高 cv 段依然准确，降权移除有用监督）。**待 fixedW 3 seeds 确认**。若确认，论文叙事部分复活：物理一致性帮助 TTE（均匀加权），cv 加权与有序头是不必要的复杂化。
