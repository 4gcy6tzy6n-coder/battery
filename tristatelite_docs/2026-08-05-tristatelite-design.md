# TriStateLite 研究设计规范

## 1. 研究目标

构建一个可在 Apple Silicon Mac 上训练的轻量时序模型，使用公开电池循环数据，在严格无泄漏的跨电池划分下，联合预测：

1. 当前荷电状态 SOC；
2. 当前健康状态 SOH；
3. 到达放电截止条件的剩余时间 TTE；
4. 三个任务的 5%、50%、95% 分位数区间。

新项目不把 RLS、EKF、Monte Carlo 和规则式监督层的简单组合视为创新。原稿中的 CA-EKF 只保留为后续传统基线或参考系统。

## 2. 核心研究问题

在测试电池、负载模式和老化阶段未进入训练集时，轻量模型能否利用“当前短时动态 + 历史循环退化摘要”形成统一电池状态表示，并输出物理一致、经校准的 SOC、SOH 和 TTE 概率预测？

## 3. 数据路线

### 3.1 主数据

NASA Randomized and Recommissioned Battery Dataset。

第一阶段必须先执行数据审计，不预先假设 ZIP 内部文件名、字段名或目录层级。只有在审计确认存在可恢复的时间、电压、电流及循环边界后，才进入标签构建。

### 3.2 外部验证

CALCE CS2/CX2。外部验证只在 NASA 主数据闭环完成后实施，且单独编写适配器，不能把 CALCE 预处理逻辑混入 NASA 适配器。

### 3.3 数据可用性门槛

NASA 数据满足以下条件才作为主数据：

- 可识别不少于 10 个独立 battery/pack ID；
- 每个可用电池至少有 3 个完整放电循环；
- 每条时序至少包含时间、电压、电流；
- 能通过电流积分或随附容量字段获得每循环放电容量；
- 能定义明确的放电结束点，从而生成 TTE 标签。

温度不是硬门槛。若缺失，模型增加 `temperature_available` 掩码，温度值使用训练集均值填充。

若上述任一关键门槛失败，则停止模型训练，将 CALCE CS2/CX2 升级为主数据源，而不是用猜测字段继续运行。

## 4. 任务定义

### 4.1 SOC

在完整放电循环 c 内：

\[
SOC_{c,t}=1-\frac{Q_{c,0:t}}{Q_{c,\mathrm{end}}}.
\]

`Q_{c,end}` 只用于生成监督标签，不得进入输入特征、标准化统计量或慢时间尺度摘要。

### 4.2 SOH

\[
SOH_c=\frac{Q_{c,\mathrm{end}}}{Q_{\mathrm{ref}}}.
\]

支持两种协议：

- `calibrated`：每个电池最早 3 个完整放电循环用于估计 `Q_ref`，这些循环不进入测试指标；
- `zero_shot`：使用训练电池早期循环的全局中位参考容量，测试电池不使用未来或完整容量信息。

MVP 先完成 `calibrated`，随后必须补充 `zero_shot`。

### 4.3 TTE

\[
TTE_{c,t}=t_{c,\mathrm{end}}-t.
\]

模型训练目标使用 `log1p(TTE_seconds)`，报告时转换回秒。MAPE 只在真实 TTE 不低于 60 秒的样本上报告，主指标使用 MAE、RMSE 和 sMAPE。

## 5. 严格无泄漏规则

- 按 battery/pack ID 划分训练、验证和测试；同一电池的任何循环不得跨集合。
- 标准化器只在训练集拟合。
- 慢时间尺度摘要只能使用当前时刻之前已经完成的循环。
- 当前循环最终容量不得成为输入。
- 不能使用双向 RNN、双向 Transformer 或未来窗口。
- 重采样必须因果化：按时间桶取最后观测值，再仅向前填充；不得用未来值线性插值当前时刻。
- 任何缓存文件必须包含 split ID、预处理配置哈希和原始数据校验信息。

## 6. 模型架构

### 6.1 快时间尺度编码器

输入最近 128 秒、1 Hz 的因果窗口。基础特征：

- voltage_v；
- current_a，统一规定放电为正；
- temperature_c；
- delta_voltage_v；
- delta_current_a；
- delta_temperature_c；
- current_mean_10s；
- current_std_10s；
- current_mean_60s；
- voltage_slope_30s；
- temperature_slope_60s；
- temperature_available。

编码器使用三层因果 TCN，通道数 32、48、64，kernel size 3，dilation 1、2、4，残差连接和 LayerNorm，输出 64 维表示。

### 6.2 慢时间尺度编码器

取最近 8 个已完成循环，每循环仅使用在该循环结束后才能合法获得的摘要：

- delivered_ah；
- discharge_duration_s；
- mean_voltage_v；
- voltage_slope_mean；
- mean_current_a；
- current_std_a；
- mean_temperature_c；
- temperature_rise_c；
- previous_predicted_soh（训练时使用模型分离生成或置空，禁止直接使用真实当前 SOH）。

单层 GRU，hidden size 32，带循环掩码。没有历史循环时使用全零向量与 `history_length=0` 掩码。

### 6.3 融合

将 64 维快表示映射到 64 维，将 32 维慢表示映射到 64 维，使用门控融合：

\[
g=\sigma(W_g[h_f;h_s]),\quad h=g\odot h_f+(1-g)\odot h_s.
\]

### 6.4 概率输出头

每个任务输出 0.05、0.50、0.95 分位数，并保证顺序。

SOC/SOH：

\[
q_{50}=\sigma(m),\quad q_{05}=q_{50}(1-\sigma(l)),\quad q_{95}=q_{50}+(1-q_{50})\sigma(u).
\]

TTE 在 log 空间：

\[
q_{50}=\mathrm{softplus}(m),\quad q_{05}=q_{50}\sigma(l),\quad q_{95}=q_{50}+\mathrm{softplus}(u).
\]

## 7. 物理一致性约束

只对放电且最近 60 秒平均电流高于 0.05 A 的样本启用：

\[
TTE_{phys}=\frac{SOC_{50}\cdot SOH_{50}\cdot Q_{ref}}{I_{eff}}\cdot 3600.
\]

负载稳定权重：

\[
w=\exp\left(-2\frac{\sigma_I}{|\mu_I|+10^{-3}}\right).
\]

一致性损失在 log 空间计算：

\[
L_{cons}=w\left|\log(1+TTE_{50})-\log(1+TTE_{phys})\right|.
\]

该约束必须可通过配置关闭，以支持消融。

## 8. 训练设置

- Python 3.11；
- PyTorch；
- float32；
- 设备顺序：MPS → CUDA → CPU；
- batch size 64；
- AdamW，lr 1e-3，weight decay 1e-4；
- 最多 50 epochs；
- early stopping patience 8；
- gradient clipping 1.0；
- 训练种子 0、1、2；
- 默认 `num_workers=0`，避免 macOS 多进程问题；
- 参数量必须小于 500,000；
- 不使用大型 Transformer、自监督预训练或外部预训练权重。

总损失：

\[
L=L_{SOC}+L_{SOH}+L_{TTE}+0.1L_{cons}.
\]

前三项为三个分位数的平均 pinball loss。

## 9. 基线和消融

### 基线

1. Last-value / simple coulomb feature baseline；
2. 单任务 GRU-SOC；
3. 单任务 GRU-SOH；
4. 单任务 GRU-TTE；
5. 共享 GRU 多任务模型；
6. 仅快 TCN 多任务模型。

### 消融

1. Full TriStateLite；
2. 去除慢时间尺度编码器；
3. 去除物理一致性损失；
4. 去除概率头，改为点预测；
5. 使用非门控拼接融合；
6. 只预测 SOC；
7. 只预测 SOH；
8. 只预测 TTE。

## 10. 核心指标

- SOC：RMSE、MAE、MaxAE；
- SOH：RMSE、MAE、R²；
- TTE：MAE、RMSE、sMAPE、受限 MAPE；
- 概率：90% coverage、mean interval width、normalized interval width、pinball loss；
- 系统：参数量、模型文件大小、单窗口推理延迟、Mac 每 epoch 时间；
- 分组：按电池、负载组、老化阶段、TTE 区间分别报告。

## 11. MVP 验收标准

- 全部单元测试通过；
- 数据审计报告生成且满足主数据门槛；
- split manifest 中无 battery ID 交叉；
- 训练集 scaler 不读取验证/测试数据；
- 所有输出满足 q05 ≤ q50 ≤ q95；
- 单电池小样本过拟合测试能显著下降损失；
- 完整模型参数量 < 500k；
- 一个种子的完整 NASA 训练、验证和测试闭环成功；
- 生成机器可读 `metrics.json` 和论文可用 `summary.csv`；
- 再运行种子 0、1、2 并输出均值±标准差。
