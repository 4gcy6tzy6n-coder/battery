# TriStateLite — 论文手稿骨架

> 状态：Methods 已完成（不依赖结果）；Abstract/Results 待实验数字填入。数字以 `<TBD>` 标注。

---

## Title (建议)

**Physics-Consistent Joint Probabilistic Prediction of Battery SOC, SOH, and Time-to-End-of-Discharge**

（备选：*A lightweight physics-coupled quantile network for battery state and remaining-discharge-time forecasting*）

## Abstract（待填数字）

Jointly predicting state-of-charge (SOC), state-of-health (SOH), and remaining discharge time (TTE) with calibrated uncertainty is essential for robust battery operation, yet most methods predict a single state deterministically and ignore the physical relation linking all three. We introduce **TriStateLite**, a lightweight GRU network with ordered quantile heads that simultaneously outputs calibrated probability intervals for SOC, SOH, and log-TTE at every discharge anchor. A load-conditioned physics-consistency loss couples the three distributions through the ampere-hour relation `tte_phys = soc·soh·q_ref / i_eff · 3600`, weighting the constraint by load-current variability. On the NASA Randomized/Recommissioned battery dataset (26 cells, 9,612 discharge cycles, battery-isolated splits), TriStateLite achieves `<TBD>` TTE MAE, `<TBD>` SOC CRPS, and `<TBD>` SOH ECE, improving TTE probabilistic accuracy by `<TBD>`% over an identical network without physics coupling while remaining physically self-consistent (`<TBD>`). The ordered-head construction guarantees non-crossing quantiles, and the grid CRPS is independently verified against sampling-based estimates.

## 1. Introduction

- 电池 SOC/SOH/RUL 概率预测的重要性与现状（引 QR-SOC 拥挤、联合点预测多、概率 TTE 空白）。
- 现有不足：(i) 多数方法单状态；(ii) 概率 TTE 几乎空白；(iii) 物理关系（安时积分）在概率框架中未被用作跨状态一致性约束。
- 本文贡献（三条）：
  1. 单一网络联合输出 SOC/SOH/log-TTE 的有序分位数（构造保证 q05≤q50≤q95，无 crossing）；
  2. 负载条件物理一致性损失耦合三分布（`soc·soh·q_ref/i_eff→tte`），权重由电流变异度调制；
  3. 在电池隔离协议下系统评估（pinball/CRPS/PICP/ECE + multi-seed），并给出独立复算的指标验证。

## 2. Related Work（对应调研结论）

- 有序分位数：Schmidt & Zhu 2016；I(S)QF (Park 2022)；Deep Non-Crossing (Brando 2022)；MQRNN (2024)。电池领域 QR 多不保证单调。
- 概率 SOH/RUL：Richardson 2017/2019 (GP)；EDL (Energies 2026)；MSTFNet 2025。
- 物理信息：Nascimento 2023 (贝叶斯 PINN)；PIRetNet；CC-TCN-BiGRU（库仑计数约束但非 TTE）。**"安时积分一致性损失约束联合概率 SOC/SOH/TTE"无直接先例**。
- 泄漏/协议：Geslin 2023 (Joule)；Maher 2025。电池隔离为规范。
- TTE 标杆：Dynaformer (Applied Energy 2023) 点预测。

## 3. Methods

### 3.1 问题设定
放电循环中每个锚点 t，给定因果快窗口 `X_fast ∈ R^{128×12}`（11 个 `__scaled` 特征 + 温度可用性布尔）与慢速历史 `X_slow ∈ R^{8×8}`（仅更早完成循环的摘要），预测三状态 `(soc, soh, log1p(tte))` 的 19 级有序分位数 `q_{0.05},...,q_{0.95}`。物理通道 `(i_eff_60s, current_cv_60s, q_ref_ah)` 独立于模型输入，仅进一致性损失。

### 3.2 标签
- `soc = clip(1 − Q_discharged/delivered_ah, 0, 1)`（库仑计数）
- `soh = clip(delivered_ah/q_ref, 0, 1)`，`q_ref` = 该电池最早 3 个有效循环中位容量
- `log_tte = log1p(剩余放电秒数)`

### 3.3 模型（Fig 1）
GRU(2 层, hidden=64) 编码 fast 窗口（mask-exact packing，左补零不入递归）→ 末隐态 + slow 历史 mask-aware mean 池化 → tanh 融合 → 三个头：
- `BoundedQuantileHead`（soc/soh）：`q50=σ(m)`；向下 `q_{i-1}=q_i·σ(l_i)`；向上 `q_{i+1}=q_i+(1−q_i)σ(u_i)` → 有序且 ∈[0,1]
- `PositiveQuantileHead`（log_tte）：`q50=softplus(m)`；向下乘积；向上 `+softplus(u)` → 有序且 ≥0

### 3.4 损失
`L = Σ_state w_state·pinball(q_state, y_state, levels) + λ·L_phys`
- `w_state = 1/mean|y_state|`（训练池估计）
- `L_phys = Σ_active exp(−2·clamp(cv,0))·|log_tte_med − log1p(soc_med·soh_med·q_ref/i_eff·3600)| / n_active`，仅 `i_eff>0.05A` 激活，TTE clamp [0, 604800]
- 消融：noPhysics（λ=0）、unordered（独立 Linear 头+后置排序）、fixedW（cv_decay=0）、point（MSE）、MC-dropout（推理期 dropout 采样, mc=30, dropout=0.3）

### 3.5 评估
- 电池隔离 split：train 18 / val 4 / test 4（seed 2026），test 电池 battery10/13/22/54
- 指标：pinball(19级)、CRPS（2×梯形积分，~5% 偏差）、MAE/RMSE、PICP90、PINAW90、ECE；TTE 秒空间 MAE
- multi-seed（主 5、消融 3），mean±std
- 独立复算：`verify_results.py` 用逐锚点 MC 采样重算 CRPS（交叉验证 <5%）

### 3.6 实现/复现
PyTorch 2.13、CPU 训练（MPS 更慢且 autograd 不稳定）、`data/processed/nasa-randomized-full`（split_id `3203055542a02c9d`）、每 run 记录 `git_revision`、config_hash、seed。代码在 `src/tristatelite/experiments/`。

## 4. Results（待填 `<TBD>`）

### 4.1 主结果（Table 1）
### 4.2 消融：物理约束（H1）
### 4.3 消融：有序头（H2）
### 4.4 消融：cv 加权（H3）
### 4.5 物理自洽性（H4）

## 5. Discussion & Limitations

- 本数据集上物理公式在所有电流变异段均高度准确（中位误差 0.31%）——cv 加权在"可靠性"论据上的实证结果（见 H3）。
- SOH 信号：26 电池、SOH 0.096–1.106（battery01 衰减 34.2%）。
- 局限：单一数据集；恒功率/变负载协议未全覆盖；与 BMS 实时部署的延迟未评估。

## 6. Conclusion

`<TBD>`

## References

见 `paper_novelty_positioning` 记忆与 `paper_research_plan.md` 引用集。
